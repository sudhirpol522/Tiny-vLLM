"""Synthetic inputs and independent dense oracle for attention validation."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tiny_vllm.attention import AttnMetadata


@dataclass
class AttentionCase:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    meta: AttnMetadata
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    scale: float


def make_case(spec, device='cpu', seed=7, heads=8, kv_heads=2, dim=64, block_size=16):
    """spec contains (query length, full KV length) for each request."""
    assert heads % kv_heads == 0
    assert all(0 < q <= kv for q, kv in spec)
    generator = torch.Generator().manual_seed(seed)
    counts = [(kv + block_size - 1) // block_size for _, kv in spec]
    pages = torch.randperm(sum(counts), generator=generator) + 1
    shape = (sum(counts) + 1, block_size, kv_heads, dim)
    # Poison null page and unused tail slots to catch invalid page gathers.
    k_cache = torch.full(shape, float('nan'))
    v_cache = torch.full(shape, float('nan'))
    table = torch.zeros(len(spec), max(counts), dtype=torch.int32)
    queries, keys, values, new_slots = [], [], [], []
    qsl, cu_k = [0], [0]
    offset = 0
    for i, ((q_len, kv_len), count) in enumerate(zip(spec, counts)):
        selected = pages[offset:offset + count]
        offset += count
        table[i, :count] = selected.to(torch.int32)
        q = torch.randn(q_len, heads, dim, generator=generator)
        k = torch.randn(kv_len, kv_heads, dim, generator=generator)
        v = torch.randn(kv_len, kv_heads, dim, generator=generator)
        positions = torch.arange(kv_len)
        slots = selected[positions // block_size] * block_size + positions % block_size
        k_cache.view(-1, kv_heads, dim)[slots] = k
        v_cache.view(-1, kv_heads, dim)[slots] = v
        new_slots.extend(slots[-q_len:].tolist())
        queries.append(q)
        keys.append(k.to(device))
        values.append(v.to(device))
        qsl.append(qsl[-1] + q_len)
        cu_k.append(cu_k[-1] + kv_len)
    meta = AttnMetadata(
        slot_mapping=torch.tensor(new_slots, device=device),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
        block_table=table.to(device), num_seqs=len(spec),
        max_query_len=max(q for q, _ in spec), max_seq_len=max(kv for _, kv in spec),
        query_start_loc_cpu=qsl, seq_lens_cpu=[kv for _, kv in spec])
    return AttentionCase(torch.cat(queries).to(device), k_cache.to(device), v_cache.to(device),
                         meta, keys, values, dim ** -0.5)


def dense_oracle(case):
    """CPU float64 matmul/softmax; no SDPA and no reading the paged pool."""
    pieces = []
    for i, (key, value) in enumerate(zip(case.keys, case.values)):
        s, e = case.meta.query_start_loc_cpu[i:i + 2]
        q = case.q[s:e].cpu().double()
        key, value = key.cpu().double(), value.cpu().double()
        # Select the KV head associated with each query head explicitly.
        mapping = torch.arange(q.shape[1]) // (q.shape[1] // key.shape[1])
        key, value = key[:, mapping], value[:, mapping]
        scores = torch.einsum('qhd,khd->hqk', q, key) * case.scale
        for row in range(e - s):
            allowed = len(key) - (e - s) + row + 1
            scores[:, row, allowed:] = float('-inf')
        pieces.append(torch.einsum('hqk,khd->qhd', scores.softmax(-1), value))
    return torch.cat(pieces).float()


def dense_control(case):
    """Same SDPA/mask/head expansion as upstream, with K/V already contiguous.

    A diagnostic control for gathering overhead, not another inference engine.
    """
    out = torch.empty_like(case.q)
    for i, (key, value) in enumerate(zip(case.keys, case.values)):
        s, e = case.meta.query_start_loc_cpu[i:i + 2]
        heads = case.q.shape[1] // key.shape[1]
        k = key.transpose(0, 1).repeat_interleave(heads, 0)
        v = value.transpose(0, 1).repeat_interleave(heads, 0)
        q_pos = torch.arange(len(key) - (e - s), len(key), device=case.q.device)
        mask = torch.arange(len(key), device=case.q.device)[None, :] <= q_pos[:, None]
        out[s:e] = F.scaled_dot_product_attention(
            case.q[s:e].transpose(0, 1), k, v, attn_mask=mask,
            scale=case.scale).transpose(0, 1)
    return out
