"""Sampling extracted verbatim from tinyserve; see UPSTREAM.md."""
import torch

from tiny_vllm.kv_cache import SamplingParams

def sample(logits, params: SamplingParams):
    """One batched call for mixed sampling settings.

    logits: [B, V]  next-token logits, one row per request that produced
                    a real sample this step (completes_prompt).
    params: length B, per-request temperature / top_k / top_p.
    """
    greedy = logits.argmax(-1)

    # if temp is 0 we assume greedy decode
    if all(p.temperature == 0 for p in params):
        return greedy.tolist()

    # what gpu
    dev = logits.device

    # Per-row temperature. temp==0 would divide by 0, so those rows use 1.0
    # here and get overwritten by greedy at the end via is_greedy.
    temp = torch.tensor([p.temperature or 1.0 for p in params], device=dev)
    logits = logits / temp[:, None] # [B, V]  higher temp -> flatter distribution

    # get vocab size
    V = logits.shape[-1]

    # top_k <= 0 means "off" -> keep the whole vocab for that row.
    top_k = torch.tensor([p.top_k if p.top_k > 0 else V for p in params], device=dev)
    top_p = torch.tensor([p.top_p for p in params], device=dev)

    # sort from largest to smallest logit (most to least likely), topk is a prefix
    # of the row, idx[b, r] is the original vocab_id of the rth loargest logit
    # so we can backtrack
    sorted_logits, idx = logits.sort(-1, descending=True)

    # ranks is 0 to V-1, where 0 cooresponds to the largest logit
    ranks = torch.arange(V, device=dev)[None, :]

    # get the probs themselves
    probs = sorted_logits.softmax(-1)

    # drop a sorted position if it fails topk or topp
    # ranks >= topk: keeps only the k largest (per row)
    # cumsum(p) - p > top_p, this token starts after the cumulative prob
    #                        has already reached top_p
    drop = (ranks >= top_k[:, None]) | (probs.cumsum(-1) - probs > top_p[:, None])
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))

    # Scatter filtered logits back to vocab order, then softmax again. But now that 
    # dropped positions are -inf, they no longer contribute to the probs in softmax
    probs = torch.empty_like(logits).scatter_(-1, idx, sorted_logits).softmax(-1)

    # Gumbel-max: argmax(log p + Gumbel(0,1)) ~ Categorical(p).
    # G = -log Exp(1), so argmax(p / Exp(1)) is the same thing.
    gumbel = -torch.empty_like(probs).exponential_().log()
    sampled = (probs.log() + gumbel).argmax(-1)

    # if we wanted some samples to be greedy decode we replace that here
    is_greedy = torch.tensor([p.temperature == 0 for p in params], device=dev)
    
    return torch.where(is_greedy, greedy, sampled).tolist()
