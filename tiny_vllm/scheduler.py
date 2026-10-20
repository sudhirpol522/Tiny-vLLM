"""Scheduler extracted verbatim from tinyserve; see UPSTREAM.md."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from tiny_vllm.kv_cache import KVCacheManager, Request
from tiny_vllm.scheduler_config import SchedulerConfig as Config

@dataclass
class ScheduledRequest:
    req: Request        # the live request
    num_computed: int   # KV present before this foward pass
    num_new: int        # how many tokens this forward will process

    @property
    def completes_prompt(self):
        """
        True: This step reaches the end of the current sequence (finish prefill
              or decode). The sampled next token is real and we append it
        
        False: chunked prefill stops mid prompt, you still run those tokens
               to write KV, but th elast positions sampled token is junk and we 
               throw it away, the next step continues the prompt
        
        num_tokens here is len(token_ids) which during prefill is just the prompt, 
        after a real sample, the scheduler's caller appends a token, so next step
        num_tokens grows by 1 and decode is num_new=1
        """
        return self.num_computed + self.num_new >= self.req.num_tokens

class Scheduler:
    """Called once per step: which requests run, and how many tokens each.

    1. Running requests go first (so decodes are never starved). If the KV
       cache is full, PREEMPT the newest running request: free its blocks and
       push it back to the waiting queue. It is recomputed later, and prefix
       caching usually makes that cheap.
    2. Then admit waiting requests while there is token budget, a free
       sequence slot, and KV space. Long prompts are CHUNKED to fit the budget.
    """

    def __init__(self, cfg: Config, kv: KVCacheManager):
        self.cfg = cfg
        self.kv = kv
        self.waiting: deque[Request] = deque() # FIFO of not yet admitted (or preempted) requests
        self.running: list[Request] = [] # currently holding KV pages (from oldest to newest)

    def add(self, req: Request):
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> list[ScheduledRequest]:

        # one token budget is shared by prefill chunks and decode tokens, so a long
        # prompt can never stall decodes for more than one steps worth of tokens
        budget = self.cfg.max_num_batched_tokens

        # plan for each request id how many tokens it runs this step
        plan: dict[str, int] = {}

        # if we evict anyone, dont admit new work
        preempted = False

        # 1. running requests
        i = 0
        # self.running is all of the currently running tasks, we loop through all of them
        # as long as we have some budget left
        while i < len(self.running) and budget > 0:

            # grab that request that is running
            req = self.running[i]

            # what this request still owes. 1 token in decode (the token just
            # sampled), or the rest of its prompt chunk. num_tokens counts 
            # prompt + everything generated so far
            n = min(req.num_tokens - req.num_computed_tokens, budget)

            # allocate() checks if a request can fit or not. 
            # running is in admit order (oldest at [0] newest at [-1])
            # we only ever evict from the newest end, so a request is never
            # preempted to make room for one younger than it. 
            while not self.kv.allocate(req, n):

                # pop will always evict the newest entry, this is so we can get the
                # older stuff done first before resuming the newer ones
                # for example, lets say we have requests [A B C], where C is the newest
                # and A is the oldest. i is currently on A, we try allocate(A, n), if
                # it fails, we will evict C and retry. If allocate(A, n) fails again, 
                # we will evict B and retry. If it still breaks (the victim is the req)
                # then we evict A and break, A is no longer running so the outer loop
                # stops too and A is not added to the plan
                victim = self.running.pop()

                # drops the victims pages. All shared prefix pages (ref_cnt > 1)
                # will survive, the rest return to the free queue (tail first). Hashes
                # are also kept so full blocks remain findable for future potential
                # cache hits
                self.kv.free_request(victim)

                # Its KV is gone, so it restarts from scratch next time. Its
                # token_ids (prompt + generated so far) are untouched, so the
                # recompute covers both. full blocks usually come back from the
                # prefix cache, the partial tail page is always recomputed.
                victim.num_computed_tokens = 0

                # we will put the victim at the front of the waiting list so it retries
                # before any brand new requests. We dont do this on this call, as preempted=True
                # will skip phase 2, we dont admit anyone new as our scheduler is currently
                # oversubscribed
                self.waiting.appendleft(victim)

                # victim always comes from index >= i and plan[] is only written after this 
                # loop, so this is always 0 in practice anyway, but we keep it just for 
                # completeness
                budget += plan.pop(victim.request_id, 0)

                # set flag that something was skipped
                preempted = True

                # req is the newest running request, there is nobody younger to evict 
                # anymore, so it preempts itself. If it is also the only running request
                # the cache simply cannot hold it and will retry forever. We break here
                # but a more complete version would verify that the cache fits the max_model_len
                # tokens, but we keep it simple here
                if victim is req:
                    break

            # the req preempted itself above, stop scheduling running requests, 
            # entirely this step.
            if req not in self.running:
                break

            # if req survived, we will reserve n tokens
            plan[req.request_id] = n

            # remove n tokens from budget
            budget -= n

            # move onto the next request
            i += 1

        # 2. waiting requests (skip if we just had to preempt as the cache is already tight)
        while (not preempted and self.waiting and budget > 0
               and len(self.running) < self.cfg.max_num_seqs):

            # whats currently on the waiting list
            req = self.waiting[0]

            # see if anythign is cached here to reuse (whole blocks only)
            cached = self.kv.find_cached_prefix(req)

            # Tokens not covered by cache hits, capped by the budget. Vecause
            # find_cached_prefix caps hits at (num_tokens - 1) // B, atleast one
            # token always runs, so the results can produce logits
            n = min(req.num_tokens - len(cached) * self.cfg.block_size, budget)

            # No preemption in this phase, an admitted request outranks a waiting one, 
            # then we break, not continue. The head of the queue blocks the rest (FIFO)
            # even if a shorter request behind it would ift
            if not self.kv.allocate(req, n, cached):
                break

            # if it fit, then we will remove the req from our list of waiting
            self.waiting.popleft()

            # KV that already exists for this request before the forward pass
            req.num_computed_tokens = len(cached) * self.cfg.block_size

            # this is for reporting only, keeps the best hit ever seen (a readmission
            # after preemption can be colder if pages were evicted meanwhile)
            req.num_cached_tokens = max(req.num_cached_tokens, req.num_computed_tokens)

            # newest = tail = first victim if phase 1 of a future step runs out of pages
            self.running.append(req)

            # reserve this requests slice of the current forward pass
            plan[req.request_id] = n
            budget -= n

        # 3. emit the batch in running order
        batch = []
        for req in self.running:
            # for every running request create a ScheduledRequest
            if req.request_id in plan:
                # Capture num_computed_tokens before advancing it: the runner
                # needs the old value for positions and slots, and
                # completes_prompt compares it against num_tokens.
                batch.append(ScheduledRequest(req, req.num_computed_tokens, plan[req.request_id]))

                # Advanced optimistically, before the model has run.
                req.num_computed_tokens += plan[req.request_id]

        return batch

    def finish(self, req: Request, reason: str) -> None:
        req.finished, req.finish_reason = True, reason
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        self.kv.free_request(req)
