"""Local fixtures for testing the upstream model with random, untrained weights."""
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tiny_vllm import model as model_module
from tiny_vllm.attention import AttnMetadata


def tiny_config(**overrides):
    values = dict(model_type='qwen3', vocab_size=1024, hidden_size=256,
                  intermediate_size=768, num_hidden_layers=4,
                  num_attention_heads=8, num_key_value_heads=2, head_dim=32,
                  rms_norm_eps=1e-6, rope_theta=1000000.0,
                  tie_word_embeddings=False, attention_bias=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@contextmanager
def attention_context(meta):
    """Restore the upstream global metadata even if a forward call fails."""
    previous = model_module.ATTN_META
    model_module.ATTN_META = meta
    try:
        yield
    finally:
        model_module.ATTN_META = previous


def make_model(cfg, device='cpu', max_position=1024):
    # Upstream builds float64 RoPE tables. Build on CPU before moving to MPS.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = model_module.Model(cfg, max_position).eval()
    return model.to(device)


def attach_cache(model, lengths, block_size=16):
    counts = [(length + block_size - 1) // block_size for length in lengths]
    num_blocks = 1 + sum(counts)
    generator = torch.Generator().manual_seed(19)
    pages = torch.randperm(num_blocks - 1, generator=generator) + 1
    tables, cursor = [], 0
    for count in counts:
        tables.append(pages[cursor:cursor + count].tolist())
        cursor += count
    for layer in model.layers:
        layer.self_attn.kv_cache = torch.full(
            (2, num_blocks, block_size, model.cfg.num_key_value_heads, layer.self_attn.D),
            float('nan'), device=model.embed_tokens.weight.device)
    return tables


def make_inputs(lengths, starts, tables, device='cpu', block_size=16, vocab=1024):
    tokens, positions, slots = [], [], []
    qsl, cuk = [0], [0]
    padded = []
    width = max(len(row) for row in tables)
    for i, (length, start, pages) in enumerate(zip(lengths, starts, tables)):
        assert 0 <= start < length
        pos = torch.arange(start, length)
        ids = (pos * 7 + i * 101 + 3) % vocab
        tokens.append(ids)
        positions.append(pos)
        slots.extend(pages[p // block_size] * block_size + p % block_size for p in pos.tolist())
        padded.append(pages + [0] * (width - len(pages)))
        qsl.append(qsl[-1] + length - start)
        cuk.append(cuk[-1] + length)
    meta = AttnMetadata(
        slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(cuk, dtype=torch.int32, device=device),
        block_table=torch.tensor(padded, dtype=torch.int32, device=device),
        num_seqs=len(lengths), max_query_len=max(length - start for length, start in zip(lengths, starts)),
        max_seq_len=max(lengths), query_start_loc_cpu=qsl, seq_lens_cpu=list(lengths))
    return torch.cat(tokens).to(device), torch.cat(positions).to(device), meta


def forward(model, inputs):
    ids, positions, meta = inputs
    with attention_context(meta):
        return model.logits(model(ids, positions))


def dense_model_oracle(model, ids):
    """Independent CPU float64 forward: dense attention, no cache or upstream ops.

    Reads the same untrained weights; deliberately supports only plain Qwen3 RoPE.
    """
    cfg = model.cfg
    weights = {name: value.detach().cpu().double() for name, value in model.named_parameters()}

    def linear(x, name):
        return F.linear(x, weights[name + '.weight'], weights.get(name + '.bias'))

    def norm(x, name):
        return x / torch.sqrt(x.square().mean(-1, keepdim=True) + cfg.rms_norm_eps) * weights[name + '.weight']

    dim = cfg.head_dim
    pos = torch.arange(len(ids), dtype=torch.float64)
    inv = cfg.rope_theta ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    angles = pos[:, None] * inv[None, :]

    def rotate(x):
        half = dim // 2
        first, second = x[..., :half], x[..., half:]
        return torch.cat((first * angles.cos()[:, None] - second * angles.sin()[:, None],
                          second * angles.cos()[:, None] + first * angles.sin()[:, None]), dim=-1)

    x = F.embedding(ids.cpu(), weights['embed_tokens.weight'])
    for i in range(cfg.num_hidden_layers):
        root = f'layers.{i}'
        attn = root + '.self_attn'
        normalized = norm(x, root + '.input_layernorm')
        q = linear(normalized, attn + '.q_proj').reshape(len(ids), cfg.num_attention_heads, dim)
        k = linear(normalized, attn + '.k_proj').reshape(len(ids), cfg.num_key_value_heads, dim)
        v = linear(normalized, attn + '.v_proj').reshape(len(ids), cfg.num_key_value_heads, dim)
        if cfg.model_type == 'qwen3':
            q, k = norm(q, attn + '.q_norm'), norm(k, attn + '.k_norm')
        q, k = rotate(q), rotate(k)
        index = torch.arange(cfg.num_attention_heads) // (cfg.num_attention_heads // cfg.num_key_value_heads)
        k, v = k[:, index], v[:, index]
        scores = torch.einsum('qhd,khd->hqk', q, k) / dim ** 0.5
        future = torch.ones(len(ids), len(ids), dtype=torch.bool).triu(1)
        scores.masked_fill_(future, float('-inf'))
        attended = torch.einsum('hqk,khd->qhd', scores.softmax(-1), v).reshape(len(ids), -1)
        x = x + linear(attended, attn + '.o_proj')
        normalized = norm(x, root + '.post_attention_layernorm')
        gate = linear(normalized, root + '.mlp.gate_proj')
        up = linear(normalized, root + '.mlp.up_proj')
        x = x + linear(gate.sigmoid() * gate * up, root + '.mlp.down_proj')
    x = norm(x, 'norm')
    head = weights['embed_tokens.weight'] if cfg.tie_word_embeddings else weights['lm_head.weight']
    return (x @ head.t()).float()
