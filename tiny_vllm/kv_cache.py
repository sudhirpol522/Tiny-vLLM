"""KV-cache bookkeeping extracted from priyammaz/tinyserve; see UPSTREAM.md."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0 # 0 means greedy decode
    top_p: float = 1.0
    top_k: int = -1          # -1 means off
    max_tokens: int = 64     # max tokens to gen
    ignore_eos: bool = False # normally we stop gen on eos token

    ### for testing ###
    forced_token_ids: list[int] | None = None # teacher forcing to compare log probs
    return_logprobs: bool = False

class Request:
    """
    This is the metadata that is needed by the scheduler about a single request

    The main field we care about is num_computed_tokens, it is how
    many token_ids already have K/V in the cache, each step the scheduler
    picks how many new tokens to process for this request:

    for example, if we have a prompt of 1000 tokens but a budget of 512, 
    then we will first do 512 tokens, then the remaining 488 (this is
    chunked prefill), afterwards during generation it is one token per step
    """

    def __init__(self, 
                 request_id: str, 
                 prompt_ids: list[int], 
                 params: SamplingParams):

        self.request_id = request_id
        self.prompt_len = len(prompt_ids)

        # prompt + generated so far
        self.token_ids = list(prompt_ids) 
        self.params = params

        # how many of those tokens already have K/V written
        self.num_computed_tokens = 0

        # this requests page table (what blocks contain the KV cache
        # for this specific request?)
        self.block_ids: list[int] = []


        # One has per full block, for prefix reuse
        self.block_hashes: list[int] = [] 
        self.num_cached_tokens = 0 # prefix-cache hits at admission
        self.logprobs: list[torch.Tensor] = []
        self.finished = False
        self.finish_reason: str | None = None

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def output_ids(self):
        # everything after the prompt is what was generated
        return self.token_ids[self.prompt_len:]

class KVCacheManager:
    """
    This manages which phyisical  KV-cache blocks each request owns

    The block manager o nly manages block ids, it doesnt touch GPU memory
    the model runner later converts those ids to actual kv addresses. 

    Each request sees its kv cache as a sequence of logical blocks, while
    the blocks themselves can live anywhere in GPU memory. For example
    with block_size = 4 and a request containing:
    
    [A B C D E F G H I]

    the request needs three logical blocks::

        logical block 0 -> tokens [A B C D]
        logical block 1 -> tokens [E F G H]
        logical block 2 -> tokens [I]

    The block manager might assign physical blocks:
    
        req.block_ids = [17, 42, 81]

    so tokens A-D live in physical block 17, E-H in block 42, and I in block 81.
    The physical blocks do not need to be contiguous.

    For token i:

        logical_block = i // block_size
        offset        = i % block_size
        block_id      = req.block_ids[logical_block]

    if the cache is viewed as one flat array the corresponding slot is:
    
        slot = block_id * block_size + offset

    For example, token F has i = 5:

        logical_block = 5 // 4 = 1
        offset = 5 % 4 = 1
        block_id = req.block_ids[1] = 42
        slot = 42 * 4 + 1 = 169

    Prefix Caching: full blocks can be reused by later requests (i.e. things
    like the system prompt can be reused for each request). Each full block is
    identified by a hash of its own parent blocks hash and its own tokens:

        block_hash = hash(parent_hash, tokens)

    The parent hash matters beacuse the same tokens can have different KV values
    when they occur after different prefixes. For example:

    Request A: 
        [system prompt] -> block 17 -> [A B C D] -> block 42 -> [E F G H] 
    
    Request B: 
        [system prompt] -> block 17 -> [A B C D] -> block 42 -> [E F G H] -> [X Y]


    Request B can reuse block 17 and 42 instead of recomputing their KV values
    it only needs a new block for [X Y]. 

    Only complete blocks are entered into the prefix cache, a partial final block
    cannot be reused until it is full. When an active request finishes, its
    cached blocks are not immediately erased, they become available for reuse
    and are placed on a least recently used (lru) free list while their hash-to-block
    mapping stays searchable. If a later request asks for the same prefix, the block can
    be taken back without recomputing its KV values. 

    Eventually, if GPU is memory is needed for a new block, the least recently used
    cached block is evicted and its physical storage is reused. At that point
    its old prefix KV is lost and must be recomputed if requested again.  

    """

    def __init__(self, 
                 num_blocks: int, 
                 block_size: int, 
                 enable_caching: bool):

        self.block_size = block_size
        self.enable_caching = enable_caching

        # therefore block 2 cannot be evicted while other requests are still using it
        self.ref_cnt = [0] * num_blocks

        # if a physical block contains a cached prefix, what prefxi does it represent
        self.block_hash : list[int | None] = [None] * num_blocks

        # ordered dict is double linked list, allows for O(1) pop-front
        # and O(1) removal from the middle. 
        # first we start at index 1, as block 0 is the null block used for padding tokens
        # their value doesnt matter as we mask it during attention anyway

        # This is effectively our LRU queue, it is a list of our unused blocks 
        self.free : OrderedDict[int, None] = OrderedDict((i, None) for i in range(1, num_blocks))

        # given this prefix hash, where is its kv data (hash -> physical block) ?
        self.hash_to_block: dict[int, int] = {}

    @property
    def num_free(self):
        return len(self.free)

    def _update_hashes(self, req: Request):
        """
        We will grow req.block_hashes here so there is one hash per completed
        block of token_ids. Effectively:

        hash_i = hash( (hash_{i-1} or None, tokens[i*B : (i+1)*B]) )

        Parent has is in the tuple so a block's identity commits the 
        entire prefix, now just its own tokens
        """

        # how many items per block
        bs = self.block_size

        # assume our requested tokens are 
        # [A B C D E F G H I]
        # and our block size is 4
        
        # The request needs hashes for complete blocks:
        # [A B C D]
        # [E F G H]
        # but not the final
        # [I]
        # as that block isnt full yet

        # so if we have 9 tokens and blocks size of 4, we have 2 
        # complete blocks and 1 incomplete. So we can log the hash
        # for the two compelte ones
        while len(req.block_hashes) < req.num_tokens // bs:

            # initially i = 0 (we start empty) and len(req.block_hashes) = 0
            i = len(req.block_hashes)

            # if it wasnt empty (after first iteration) then the parent is the
            # last block available for this specific request (-1)
            # i = 0: parent = None
            # i = 1: parent = hash(block 0)
            # i = 2: parent = hash(block 1)
            # which gives us the chain

            # we hash the parent along with the tokens
            # at i = 0 req.token_ids[0:4] is [A B C D], 
            # at i = 1 req.token_ids[4:8] gives [E F G H]
            # and our hashes become
            # H0 = hash(None, [A B C D])
            # H1 = hash(H0, [E F G H])

            parent = req.block_hashes[-1] if req.block_hashes else None
            req.block_hashes.append(hash((parent, tuple(req.token_ids[i * bs:(i + 1) * bs]))))
    
    def find_cached_prefix(self, req: Request):
        """
        This provides us the block ids of the longest cached prefix, 
        telling us how much of this requests prefix have we already
        computed?

        This is done by:

            1. refresh the hashes using the _update_hashes method
            2. walk the hashes left to rigth, stopping at the first miss
            3. cap at (num_tokens - 1) // B so atleast one token is 
               always computed (we need its logits)
        """
        if not self.enable_caching:
            return []

        # generate the requests hashes
        # suppose the request is [A B C D E F G H I J]
        # H0 = hash(None, [A B C D])
        # H1 = hash(H0,   [E F G H])
        # and remember from an earlier request this would be the same thing!
        # hash of the same input MUST give the same hash integer value 
        # regardless of the request it came from
        self._update_hashes(req)

        # we have to always leave atleast one token to compute. Suppose
        # num_tokens = 8 and block_size = 4, naively, 8 // 4 = 2 blocks. 
        # but instead we will do (8 - 1) // 4 = 1. To generate the next
        # token you need the logits at the last prompt position, which come
        # from running that token through the whole model. It shidden states 
        # passes through every layer and then the LM head, the KV cache only
        # keeps the K,V vectors of teh past tokens to later tokens can attend 
        # to them, it doesnt keep hidden states or logits. So if a requests
        # entire prompt came from the cache, the scheduler later would have
        # 0 tokens to run, nothing to feed the model, no logits, no first 
        # output tokens, num_tokens - 1 caps the cache hit so atleast the last
        # token is always run
        max_blocks = (req.num_tokens - 1) // self.block_size

        # Now we look up for matching hashes
        hits = []

        # suppose H0 -> block 17, H1 -> block 42, H2 -> missing
        # then our hits are block 17 and 42
        for h in req.block_hashes[:max_blocks]:
            b = self.hash_to_block.get(h)

            # cache hits have to be contiguous, we cant have 2 hits, a miss
            # and then another hit, as the K,V values of the third hit
            # are different when attended to the different miss
            if b is None:
                break

            hits.append(b)

        return hits

    def allocate(self,
                 req: Request, 
                 num_new_tokens: int, 
                 cached: list[int] = ()):
        """
        We need to ensure the page table covers the total

            total = already_computed + |cached| * B + num_new_tokens
            need  = ceil(total / B) - |req.block_ids| - |cached|

        basically: how many NEW physical pages do I need to pull
        off of the free list so this requests page table covers every 
        token that will exist after this step?
        """

        # first we calculate the total tokens
        # pretend we have  8 tokens already, 4 tokens in a cached prefix
        # and we want 6 new tokens, that is a total of 18 tokens. We need
        # ceil(18 / 4) = 5 blocks to fit that
        total = (
            req.num_computed_tokens 
            + len(cached) * self.block_size
            + num_new_tokens
        )

        # compute ceiling integer division
        total_needed_blocks = -(-total // self.block_size)

        # subtract out the number of blocks the request already has
        # as well as the blocks we are about to reuse. need can be 
        # 0 as in a previous step we may have allocated a whole block
        # and this decode step is filling unused slots in it
        need = total_needed_blocks - len(req.block_ids) - len(cached)

        # cached hits with ref_cnt == 0 are still sitting on the free list
        # attaching them removes them from free, so they look like
        # they consume a free slot even though we arent actually grabbing
        # a blank page
        #   need: brand new pages we will add
        #   reclaim: which hashed pages will you take off free
        reclaim = sum(
            1 for b in cached if self.ref_cnt[b] == 0
        )

        # if i need new blocks plus cached blocks that must be reclaimed from
        # the free pool, do I have enough free capacity?
        if need + reclaim > self.num_free:
            return False # if false then nothing to do until space opens up

        # claim cached blocks
        for b in cached:
            # suppose cached = [17, 42]
            if self.ref_cnt[b] == 0:

                # if block 17 was sitting on the free list
                # free = [17, 23, 31, 44]
                # we now just have free = [23, 31, 44]
                # which means that this block is no longer cached or fre, 
                # an active request owns it
                del self.free[b]

            # update the reference counter, if it was 0 we now have our new
            # reference. If it wasnt 0, lets say 1 as somthing else was using
            # it, now it would be 2, as 2 requests are using this block
            self.ref_cnt[b] += 1

            # the request now points at the cached physical block
            req.block_ids.append(b)

        # now we allocate new blocks we need
        for _ in range(max(need, 0)):
            b, _ = self.free.popitem(last=False) # take the first item

            # we are evicting b, but it still may be in our mapping
            # so we will delete it from there too (all the old metadata)
            # if we dont , then the recycled block would claim some information
            # is in there, when really that information has changed
            if self.block_hash[b] is not None:
                self.hash_to_block.pop(self.block_hash[b], None)
                self.block_hash[b] = None

            # give the block to the request
            # for example blocks 16 now has ref_count = 1 and 
            # is owned by the new request
            self.ref_cnt[b] = 1
            req.block_ids.append(b) 

        self._register_full_blocks(req, total)
        return True
    
    def _register_full_blocks(self, 
                              req: Request, 
                              num_tokens: int):
        """
        Did any of the blocks we just allocated become filled with known 
        tokens? if yes then we add them to the prefix cache
        """
        if not self.enable_caching:
            return 

        # make sure all relevant block hashes exist
        self._update_hashes(req)

        # Loop over only complete full blocks
        for i in range(min(num_tokens, req.num_tokens) // self.block_size):

            # get the block and hash
            b, h = req.block_ids[i], req.block_hashes[i]

            # register the hash to the block
            if self.block_hash[b] is None:
                self.block_hash[b] = h
                # we use set default as if the hash is already present
                # for some reason, there is no need to overwrite it
                self.hash_to_block.setdefault(h, b)

    def free_request(self, req: Request):
        # clear out everything at the end of a request
        # resetting the state back and returning the blocks
        # back for use by other requests
        for b in reversed(req.block_ids):
            self.ref_cnt[b] -= 1
            if self.ref_cnt[b] == 0:
                self.free[b] = None
        req.block_ids = []
