"""Functional MLX port of our Qwen-style decoder; no CUDA graph emulation."""
from collections import OrderedDict

import numpy as np

from tiny_vllm.mlx_runtime import mx


class MLXModel:
    """Read-only weights and explicit paged K/V inputs/outputs for compilation."""

    def __init__(self, torch_model, block_size, num_blocks, compile_decode=False):
        self.cfg = torch_model.cfg
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.compile_decode = compile_decode
        self.max_blocks_per_seq = -(-len(torch_model.cos) // block_size)
        self.weights = {name: mx.array(value.detach().float().cpu().numpy())
                        for name, value in torch_model.named_parameters()}
        self.cos = mx.array(torch_model.cos.cpu().numpy())
        self.sin = mx.array(torch_model.sin.cpu().numpy())
        self.heads = self.cfg.num_attention_heads
        self.kv_heads = self.cfg.num_key_value_heads
        self.dim = getattr(self.cfg, 'head_dim', None) or self.cfg.hidden_size // self.heads
        self._compiled = OrderedDict()
        self.max_compiled_variants = 16
        mx.eval(self.weights, self.cos, self.sin)

    @property
    def compiled_variants(self):
        return len(self._compiled)

    def empty_cache(self):
        shape = (self.num_blocks * self.block_size, self.kv_heads, self.dim)
        cache = [mx.zeros(shape, dtype=mx.float32) for _ in range(2 * self.cfg.num_hidden_layers)]
        mx.eval(cache)
        return cache

    def _linear(self, x, name):
        out = x @ self.weights[name + '.weight'].T
        bias = self.weights.get(name + '.bias')
        return out if bias is None else out + bias

    def _norm(self, x, name):
        return mx.fast.rms_norm(x, self.weights[name + '.weight'], self.cfg.rms_norm_eps)

    def _rotary(self, x, cos, sin):
        first, second = mx.split(x, 2, axis=-1)
        cos, sin = cos[:, None, :], sin[:, None, :]
        return mx.concatenate((first * cos - second * sin, second * cos + first * sin), axis=-1)

    def _function(self, query_lengths, buckets):
        boundaries = np.cumsum((0,) + query_lengths).tolist()

        def forward(ids, positions, slots, table, lengths, rows, cache):
            x = self.weights['embed_tokens.weight'][ids]
            cos, sin = self.cos[positions], self.sin[positions]
            updated = []
            for layer in range(self.cfg.num_hidden_layers):
                root = f'layers.{layer}'
                attention = root + '.self_attn'
                normalized = self._norm(x, root + '.input_layernorm')
                q = self._linear(normalized, attention + '.q_proj').reshape(-1, self.heads, self.dim)
                k = self._linear(normalized, attention + '.k_proj').reshape(-1, self.kv_heads, self.dim)
                v = self._linear(normalized, attention + '.v_proj').reshape(-1, self.kv_heads, self.dim)
                if self.cfg.model_type == 'qwen3':
                    q, k = self._norm(q, attention + '.q_norm'), self._norm(k, attention + '.k_norm')
                q, k = self._rotary(q, cos, sin), self._rotary(k, cos, sin)
                kc = mx.put_along_axis(cache[2 * layer], slots[:, None, None], k, axis=0)
                vc = mx.put_along_axis(cache[2 * layer + 1], slots[:, None, None], v, axis=0)
                updated.extend((kc, vc))
                pieces = []
                for i, bucket in enumerate(buckets):
                    start, end = boundaries[i:i + 2]
                    pages = table[i, :bucket // self.block_size]
                    shape = (self.num_blocks, self.block_size, self.kv_heads, self.dim)
                    key = kc.reshape(shape)[pages].reshape(bucket, self.kv_heads, self.dim)
                    value = vc.reshape(shape)[pages].reshape(bucket, self.kv_heads, self.dim)
                    keys = mx.arange(bucket)
                    mask = (keys[None, :] <= positions[start:end, None]) & (keys[None, :] < lengths[i])
                    out = mx.fast.scaled_dot_product_attention(
                        q[start:end].transpose(1, 0, 2)[None],
                        key.transpose(1, 0, 2)[None], value.transpose(1, 0, 2)[None],
                        scale=self.dim ** -0.5, mask=mask)
                    pieces.append(out[0].transpose(1, 0, 2).reshape(end - start, -1))
                x = x + self._linear(mx.concatenate(pieces, axis=0), attention + '.o_proj')
                normalized = self._norm(x, root + '.post_attention_layernorm')
                gate = self._linear(normalized, root + '.mlp.gate_proj')
                up = self._linear(normalized, root + '.mlp.up_proj')
                x = x + self._linear(gate * mx.sigmoid(gate) * up, root + '.mlp.down_proj')
            hidden = self._norm(x, 'norm')[rows]
            head = 'embed_tokens.weight' if self.cfg.tie_word_embeddings else 'lm_head.weight'
            return hidden @ self.weights[head].T, updated

        return forward

    def forward(self, ids, positions, slots, table, lengths, rows, cache, query_lengths, seq_lens):
        query_lengths = tuple(query_lengths)
        # Host lengths choose a bucket, but actual lengths/positions remain tensor inputs.
        # Masking handles padded slots, including reserved null-page entries.
        buckets = []
        for length in seq_lens:
            pages = -(-length // self.block_size)
            bucket_pages = min(1 << (pages - 1).bit_length(), self.max_blocks_per_seq)
            buckets.append(bucket_pages * self.block_size)
        signature = (query_lengths, tuple(buckets))
        if self.compile_decode and all(q == 1 for q in query_lengths):
            if signature not in self._compiled:
                if len(self._compiled) >= self.max_compiled_variants:
                    self._compiled.popitem(last=False)
                self._compiled[signature] = mx.compile(self._function(*signature))
            self._compiled.move_to_end(signature)
            function = self._compiled[signature]
        else:
            function = self._function(*signature)
        return function(ids, positions, slots, table, lengths, rows, cache)
