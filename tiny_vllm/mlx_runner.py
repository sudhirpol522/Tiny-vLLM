"""MLX execution with the existing scheduler and page-ownership manager."""
import numpy as np
import torch

from tiny_vllm.loading import load_model
from tiny_vllm.mlx_model import MLXModel
from tiny_vllm.mlx_runtime import mx


def sample_mlx(logits, params):
    greedy = mx.argmax(logits, axis=-1)
    if all(p.temperature == 0 for p in params):
        return greedy.tolist()
    temperature = mx.array([p.temperature or 1.0 for p in params])
    scaled = logits / temperature[:, None]
    order = mx.argsort(-scaled, axis=-1)
    sorted_logits = mx.take_along_axis(scaled, order, axis=-1)
    probs = mx.softmax(sorted_logits, axis=-1)
    top_k = mx.array([p.top_k if p.top_k > 0 else logits.shape[-1] for p in params])
    top_p = mx.array([p.top_p for p in params])
    drop = (mx.arange(logits.shape[-1])[None, :] >= top_k[:, None]) | (mx.cumsum(probs, axis=-1) - probs > top_p[:, None])
    filtered = mx.where(drop, -float('inf'), sorted_logits)
    ranks = mx.random.categorical(filtered, axis=-1)
    sampled = mx.take_along_axis(order, ranks[:, None], axis=-1).squeeze(-1)
    return mx.where(mx.array([p.temperature == 0 for p in params]), greedy, sampled).tolist()


class MLXRunner:
    def __init__(self, cfg):
        if not mx.metal.is_available():
            raise RuntimeError('MLX Metal is unavailable')
        if cfg.dtype != torch.float32:
            raise ValueError('The MLX backend currently supports float32 only')
        self.cfg = cfg
        mx.random.seed(cfg.seed)
        reference = load_model(cfg.model, cfg.dtype, cfg.max_model_len, 'cpu')
        self.model = MLXModel(reference, cfg.block_size, cfg.num_blocks, cfg.compile_decode)
        self.cache = None

    def allocate_kv_cache(self):
        self.cache = self.model.empty_cache()
        return self.cfg.num_blocks

    def prepare(self, batch):
        if not batch:
            raise ValueError('Cannot run an empty batch')
        ids, positions, slots, rows, lengths, query_lengths = [], [], [], [], [], []
        table = np.zeros((len(batch), self.model.max_blocks_per_seq), dtype=np.int32)
        for i, item in enumerate(batch):
            start, count = item.num_computed, item.num_new
            pos = list(range(start, start + count))
            ids.extend(item.req.token_ids[start:start + count])
            positions.extend(pos)
            slots.extend(item.req.block_ids[p // self.cfg.block_size] * self.cfg.block_size + p % self.cfg.block_size
                         for p in pos)
            rows.append(len(ids) - 1)
            lengths.append(start + count)
            query_lengths.append(count)
            table[i, :len(item.req.block_ids)] = item.req.block_ids
        arrays = [mx.array(value, dtype=mx.int32) for value in (ids, positions, slots, table, lengths, rows)]
        return arrays, query_lengths, lengths

    def run_logits(self, batch):
        arrays, query_lengths, lengths = self.prepare(batch)
        logits, self.cache = self.model.forward(*arrays, self.cache, query_lengths, lengths)
        # Force lazy execution and materialize cache updates before the next scheduler step.
        mx.eval(logits, self.cache)
        return logits

    def run(self, batch):
        logits = self.run_logits(batch)
        tokens = sample_mlx(logits, [item.req.params for item in batch])
        # Avoid copying a full vocabulary to PyTorch unless debugging requests it.
        reference_logits = (torch.from_numpy(np.array(logits))
                            if any(item.req.params.return_logprobs for item in batch) else None)
        return tokens, reference_logits
