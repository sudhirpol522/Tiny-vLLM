"""Offline generation engine extracted and adapted from tinyserve."""
from dataclasses import dataclass
import uuid

import torch
from transformers import AutoTokenizer, GenerationConfig

from tiny_vllm.config import EngineConfig
from tiny_vllm.kv_cache import KVCacheManager, Request, SamplingParams
from tiny_vllm.runner import ModelRunner
from tiny_vllm.scheduler import Scheduler


@dataclass
class Output:
    request_id: str
    text: str                   # full text so far
    delta: str                  # new text since the last output
    token_ids: list[int]
    finished: bool
    finish_reason: str | None
    num_prompt_tokens: int
    num_cached_tokens: int
    logprobs: list[torch.Tensor] | None = None   # [vocab] per generated token, on CPU

class Detokenizer:
    """
    Incrementally convert generated token IDs into text.

    Instead of decoding the entire output on every step (which would be
    O(n^2) over a long generation), we keep track of how much we've already
    decoded and only look at the new portion.

    We also avoid emitting a trailing U+FFFD ("�") because it can mean that
    a multi-byte UTF-8 character is incomplete and needs another token.
    """

    def __init__(self, tok):
        # The tokenizer used to convert token IDs -> text.
        self.tok = tok

        # Complete text that we've already returned to the caller.
        self.text = ""

        # Start of the token region we're currently decoding.
        self.prefix = 0

        # Number of token IDs we've already processed.
        self.read = 0

    def update(self, ids, final) -> str:
        # Decode the tokens we've already seen.
        #
        # Example:
        #   ids = [10, 20, 30, 40]
        #   prefix = 0
        #   read = 2
        #
        #   ids[prefix:read] = [10, 20]
        #
        # This gives us the text that was previously available.
        old = self.tok.decode(
            ids[self.prefix:self.read],
            skip_special_tokens=True,
        )

        # Decode everything from prefix up to the current end.
        # This lets us see what new text has appeared since the last update.
        new = self.tok.decode(
            ids[self.prefix:],
            skip_special_tokens=True,
        )

        # Don't emit anything if:
        # 1. new didn't get any longer, meaning no new text appeared.
        #
        # OR
        #
        # 2. new ends in "�" and this isn't the final update.
        #
        # "�" can indicate an incomplete multi-byte UTF-8 character.
        # We wait for another token before emitting it.
        if len(new) <= len(old) or (new.endswith("�") and not final):
            return ""

        # `old` is the text we already know about.
        # Everything after it is newly generated text.
        #
        # Example:
        #   old = "Hello"
        #   new = "Hello world"
        #   delta = " world"
        delta = new[len(old):]

        # Add the newly generated text to our complete output.
        self.text += delta

        # Move our bookkeeping forward:
        #
        # prefix becomes the old read position, and
        # read becomes the current number of tokens.
        #
        # This means the next call only needs to consider the new tokens.
        self.prefix, self.read = self.read, len(ids)

        # Return only the newly generated text.
        return delta

class Engine:
    """
    Put it all together!
    """
    def __init__(self, cfg: EngineConfig):

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.stop_ids = {self.tokenizer.eos_token_id}

        # some model could have multiple stop ids so just incase we make a list
        try:
            eos = GenerationConfig.from_pretrained(cfg.model).eos_token_id
            self.stop_ids |= set(eos if isinstance(eos, list) else [eos])
        except Exception:
            pass

        # If None is in here just remove it
        self.stop_ids.discard(None)

        # Set up the runner
        if cfg.backend == 'mlx':
            from tiny_vllm.mlx_runner import MLXRunner
            self.runner = MLXRunner(cfg)
        else:
            self.runner = ModelRunner(cfg)
        num_blocks = self.runner.allocate_kv_cache()
        # Set up the kv cache and scheduler
        self.kv = KVCacheManager(num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.scheduler = Scheduler(cfg, self.kv)

        # set up storage for the rest
        self.detok: dict[str, Detokenizer] = {}
        self.requests: dict[str, Request] = {}

    def add_request(self, 
                    prompt: str | list[int], 
                    params: SamplingParams,
                    request_id: str | None = None):

        # tokenize if we have to
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)

        # assert it fits
        if not 0 < len(ids) < self.cfg.max_model_len:
            raise ValueError('Prompt must fit with room for an output token')
        if any(not isinstance(token, int) or isinstance(token, bool)
               or not 0 <= token < self.runner.model.cfg.vocab_size for token in ids):
            raise ValueError('Prompt contains an invalid token ID')
        if not isinstance(params.max_tokens, int) or isinstance(params.max_tokens, bool) or params.max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        if params.temperature < 0 or not 0 < params.top_p <= 1:
            raise ValueError('Use nonnegative temperature and top_p in (0, 1]')
        if params.forced_token_ids is not None:
            required = min(params.max_tokens, self.cfg.max_model_len - len(ids))
            if len(params.forced_token_ids) < required:
                raise ValueError('Not enough forced tokens for the requested output')
            if any(not isinstance(token, int) or not 0 <= token < self.runner.model.cfg.vocab_size
                   for token in params.forced_token_ids):
                raise ValueError('Invalid forced token ID')

        # unique id for request
        rid = request_id or uuid.uuid4().hex

        if rid in self.requests:
            raise ValueError('Request ID is already active')

        # save the request
        self.requests[rid] = req = Request(rid, ids, params)

        # Add to scheduler
        self.scheduler.add(req)

        # add to detokenizer
        self.detok[rid] = Detokenizer(self.tokenizer)

        return rid

    def abort(self, request_id: str) -> None:
        """Client went away: stop the request and give its KV blocks back."""
        req = self.requests.pop(request_id, None)
        if req is not None and not req.finished:
            self.scheduler.finish(req, "abort")
            self.detok.pop(request_id, None)

    def has_work(self):
        return self.scheduler.has_work() 

    def step(self):

        # Who runs this forward, and how many tokens each (phase 1+2 of schedule).
        batch = self.scheduler.schedule()

        # if nothing to do return nothing
        if not batch:
            return []

        # packed forward. tokens[i] is a sample from the last query row of batch[i]
        tokens, logits = self.runner.run(batch)

        outputs = []
        for i, (s, tok) in enumerate(zip(batch, tokens)):
            req = s.req

            # chunked prefill that stops mid-prompt, we wrote k/v for those tokens
            # but the last position is not the next output token, so we throw
            # these samples away
            if not s.completes_prompt:
                continue

            # Teacher forcing (equivalence tests). ignore the sample and
            # append a predetermined id so two engines walk the same path.
            # this is just for debugging
            forced = req.params.forced_token_ids
            if forced is not None:
                tok = forced[len(req.output_ids)] # output ids is tokens after prompt

            # get the full next token distribution  (one tensor per generated step)
            # this is also just for debugging
            if req.params.return_logprobs:
                req.logprobs.append(logits[i].log_softmax(-1).cpu())

            req.token_ids.append(tok) # grows output_ids (token_ids[prompt_len:])
            n_out = len(req.output_ids)

            # Stop conditions. finish() drops KV pages onto the free list
            # (hashes kept) and removes the request from running/waiting.
            if tok in self.stop_ids and not req.params.ignore_eos:
                self.scheduler.finish(req, "stop")
            elif n_out >= req.params.max_tokens or req.num_tokens >= self.cfg.max_model_len:
                self.scheduler.finish(req, "length")

            # incremental detokenization
            d = self.detok[req.request_id]
            delta = d.update(req.output_ids, req.finished)
            outputs.append(Output(
                req.request_id, d.text, delta, req.output_ids, req.finished,
                req.finish_reason, req.prompt_len, req.num_cached_tokens,
                req.logprobs if req.params.return_logprobs else None,
            ))
            if req.finished:
                del self.detok[req.request_id]
                del self.requests[req.request_id]

        return outputs

    def generate(self, prompts: list, params: SamplingParams | list[SamplingParams]):
        """Offline batch API, add everything, step until done."""
        params = params if isinstance(params, list) else [params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError('Provide one sampling configuration per prompt')
        rids = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        done: dict[str, Output] = {}
        while self.has_work():
            for o in self.step():
                if o.finished:
                    done[o.request_id] = o
        return [done[r] for r in rids]

