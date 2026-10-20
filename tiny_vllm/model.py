"""Model definitions extracted from tinyserve; see UPSTREAM.md."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_vllm.attention import AttnMetadata, torch_ref_attention

# The Mac-compatible reference backend is the default for this component.
ATTN_META: AttnMetadata | None = None
ATTN_FN = torch_ref_attention

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) \
                                 + self.eps) * self.weight.float()).to(x.dtype)
    
def rotary(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)

class Attention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.H = cfg.num_attention_heads
        self.Hkv = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H

        ### qwen2 hardcodes true, but other models (llama, qwen3) expose attention_bias in config
        qkv_bias = getattr(cfg, "attention_bias", False) or cfg.model_type == "qwen2"
        self.q_proj = nn.Linear(cfg.hidden_size, self.H * self.D, bias=qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias)
        self.o_proj = nn.Linear(self.H * self.D, cfg.hidden_size, bias=False)

        self.qk_norm = cfg.model_type == "qwen3"  # Qwen3 normalizes q and k per head
        if self.qk_norm:
            self.q_norm = RMSNorm(self.D, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.D, cfg.rms_norm_eps)

        self.layer_idx = layer_idx
        self.kv_cache: torch.Tensor | None = None  # [2, num_blocks, block_size, Hkv, D]

    def forward(self, x, cos, sin):
        T = x.shape[0] # x is passed in as packed

        q = self.q_proj(x).view(T, self.H, self.D)
        k = self.k_proj(x).view(T, self.Hkv, self.D)
        v = self.v_proj(x).view(T, self.Hkv, self.D)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q, k = rotary(q, cos, sin), rotary(k, cos, sin)

        meta = ATTN_META # read from global

        if meta is None:  # profiling run: shapes only
            return self.o_proj(q.reshape(T, -1) * 0)

        # Write this step's new K/V into the page pool before attention.
        # k[:n] is packed [T, Hkv, D] in the same order as q.
        # slot_mapping[t] = page * block_size + offset for that token.
        # view(-1, Hkv, D) flattens (page, offset) onto dim 0 so
        #   cache.view(...)[slot] == cache[page, offset]
        # index_copy_(0, ...) does cache_flat[slot_mapping[t]] = k[t]
        # for every new token at once. Prefix pages not in slot_mapping
        # are left as-is.
        k_cache, v_cache = self.kv_cache[0], self.kv_cache[1]
        n = meta.slot_mapping.shape[0]
        k_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, k[:n])
        v_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, v[:n])

        # attend over each request's whole context through its block table
        out = ATTN_FN(q[:n], k_cache, v_cache, meta, self.D ** -0.5)

        return self.o_proj(out.reshape(n, -1))

### WRITE THE REST OF THE DECODER LAYER 
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.self_attn = Attention(cfg, i)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))

### WRITE THE MODEL
class Model(nn.Module):
    def __init__(self, cfg, max_position: int):
        super().__init__()

        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.tie = getattr(cfg, "tie_word_embeddings", False)

        if not self.tie:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # RoPE table indexed by absolute position
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        assert rope.get("rope_type", "default") == "default", "only plain RoPE in this tutorial"
        
        theta = getattr(cfg, "rope_theta", None) or rope.get("rope_theta", 10000.0)
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

        inv_freq = 1.0 / theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
        freqs = torch.outer(torch.arange(max_position, dtype=torch.float64), inv_freq)
        self.register_buffer("cos", freqs.cos().float(), persistent=False)
        self.register_buffer("sin", freqs.sin().float(), persistent=False)

    def forward(self, input_ids, positions):
        cos, sin = self.cos[positions], self.sin[positions]
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

    def logits(self, hidden):
        w = self.embed_tokens.weight if self.tie else self.lm_head.weight
        return (hidden @ w.t()).float()
