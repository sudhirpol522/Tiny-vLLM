"""Paged reference attention extracted from tinyserve; see UPSTREAM.md."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass
class AttnMetadata:
    """
    Batch descriptor for one engine step, shared by every attention layer.

    q is packed [T, H, D] and has no batch dimension. These fields say how
    that stream splits into requests, where each new token's K/V is written
    and how to read each requests paged prefix. 

    There are two lengths per request:

        q_len: tokens that are run this step (rows of q)
        kv_len: keys that exist after the write (already computed + q_len)
                as we dont compute attention until the kv is fully updated 
                with new incoming items

    For example if we have 4 tokens in cache, and we run two more at positions
    4 and 5, our q_len = 2 (running 2 tokens) and the kv_len = 6

    T = sum of q_len over the batch
    B = num_seqs
    """

    # Flat cache address for each of the T new tokens.
    #   slot = block_id * block_size + (t % block_size)
    # Layer writes K/V to k_cache.view(-1, Hkv, D)[slot_mapping[t]]
    # before attention. Attention itself does not read this field.
    slot_mapping: torch.Tensor          # [T]

    # Prefix sums of this-step query lengths. Request i owns q[qsl[i]:qsl[i+1]].
    # qsl[0] = 0, qsl[B] = T. Flash-attn's cu_seqlens_q.
    #   A prefill 3 + B decode 1 + C prefill 4 ->  [0, 3, 4, 8]
    query_start_loc: torch.Tensor       # [B+1]

    # Prefix sums of full context lengths (kv_len), not q_len.
    # cu_k[i+1] - cu_k[i] == seq_lens[i]. Flash-attn's cu_seqlens_k.
    #   A kv=10, B kv=50, C kv=7  ->  [0, 10, 60, 67]
    # Does not mean K is packed densely, K stays paged. This only tells
    # the kernel how long each request's prefix is.
    cu_seqlens_k: torch.Tensor          # [B+1]

    # Page table. Row i is req.block_ids padded to max_blocks.
    #   block_table[i, t // bs] = physical page holding token t of request i
    # Only the first ceil(kv_len / bs) entries are live; the rest is
    # null-page padding and must not be gathered. Flash-attn reads this
    # in-kernel, torch_ref slices the live prefix and gathers in Python.
    block_table: torch.Tensor           # [B, max_blocks]

    # B. Live rows of block_table / entries in the cu_seqlens arrays.
    num_seqs: int

    # max(q_len_i) this step
    max_query_len: int

    # max(kv_len_i) this step
    max_seq_len: int

    # Host copies so torch_ref can index in Python without a GPU sync per request. 
    query_start_loc_cpu: list[int]      # [B+1]

    # Per-request kv_len on host. Not prefix-summed.
    # seq_lens_cpu[i] == cu_seqlens_k[i+1] - cu_seqlens_k[i]
    # torch_ref uses this directly as kv_len, flash-attn uses cu_seqlens_k.
    seq_lens_cpu: list[int] # [B]

def torch_ref_attention(
        q,         # [T ]
        k_cache, 
        v_cache, 
        meta, 
        scale
):
    """
    Paged attention.

    q:        [T, H, D]              packed queries for this step
    k/v_cache:[n_pages, bs, Hkv, D]  paged pool
    meta:     query_start_loc, seq_lens, block_table
    scale:    1/sqrt(D)

    For each request: gather its pages into a dense prefix, build a
    bottom-right causal mask, call SDPA. Slow (Python loop, copy of
    every context, host sync) but for understanding
    """
    out = torch.empty_like(q)
    
    # H = query heads, Hkv = KV heads (GQA if H > Hkv), bs = page size
    H, Hkv, bs = q.shape[1], k_cache.shape[2], k_cache.shape[1]

    # qsl[i]:qsl[i+1] = this request's slice of packed q
    # seq_lens[i]     = full context length after this step (kv_len)
    qsl, seq_lens = meta.query_start_loc_cpu, meta.seq_lens_cpu

    # for each sequence that has been packed
    for i in range(meta.num_seqs):
        s, e, kv_len = qsl[i], qsl[i + 1], seq_lens[i]

        # tokens processed for i this step
        q_len = e - s

        # live pages only: ceil(kv_len / bs). drop the padded tail of the row (null page 0)
        blocks = meta.block_table[i, : -(-kv_len // bs)].long()

        # Gather to a dense prefix
        #   k_cache[blocks] : [n_pages, bs, Hkv, D]
        #   flatten(0, 1)   : [n_pages*bs, Hkv, D]   pages stitched on the token axis
        #   [:kv_len]       : drop unused slots in the last page
        k = k_cache[blocks].flatten(0, 1)[:kv_len]
        v = v_cache[blocks].flatten(0, 1)[:kv_len]

        # SDPA wants H KV heads. GQA: each KV head is shared by H/Hkv query heads
        k = k.transpose(0, 1).repeat_interleave(H // Hkv, 0)
        v = v.transpose(0, 1).repeat_interleave(H // Hkv, 0)    

        # bottom right causal, this steps queries are teh last q_len positions
        # of a length kv_len sequence not 0 ... q_len - 1
        #   full prefill:  q_len == kv_len -> square causal mask
        #   chunked_prefill: q_len < kv_len -> suffix strip
        #   decode: q_len == 1 -> 1 row, all keys
        # if we have 4 kv positions already cached, and q_len = 2, then
        # our kv_len = 6 (as we will have 6 kv positions after this op) 
        # and our q_len = 2 for positions 4 and 5. Effectively, q at position
        # 4 can attend to all the kv before it (positions 0-3) and the new 
        # kv at the new position 4. Similarly, the q at position 5 can attend to
        # all the positions before it (0-3 + 4 just made) as well as the kv at 
        # the new position 5

        # keys    0  1  2  3  4  5
        # q@4     T  T  T  T  T  x  
        # q@5     T  T  T  T  T  T
        q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)
        mask = torch.arange(kv_len, device=q.device)[None, :] <= q_pos[:, None]

        # use SDPA
        o = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1), k, v, attn_mask=mask, scale=scale
        )
        out[s:e] = o.transpose(0, 1)

    return out
