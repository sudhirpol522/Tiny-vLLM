"""Apple GPU: PyTorch MPS, eager MLX, and compiled MLX with full float32."""
import argparse
import json
import os
import platform
import statistics
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3Config

from benchmarks.generation_inputs import generation_fixture
from benchmarks.model import fixture
from benchmarks.model_inputs import forward, tiny_config
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.kv_cache import KVCacheManager, SamplingParams
from tiny_vllm.mlx_model import MLXModel
from tiny_vllm.mlx_runtime import mx
from tiny_vllm.scheduler import Scheduler


MODES = ['pytorch_mps', 'mlx_eager', 'mlx_compiled']
FIXED = [('decode-128', [(1, 128)]), ('decode-512', [(1, 512)]),
         ('decode-8x128', [(1, 128)] * 8), ('decode-8x512', [(1, 512)] * 8),
         ('prefill-128', [(128, 128)])]


def synchronize(mode):
    if mode == 'pytorch_mps':
        torch.mps.synchronize()
    else:
        mx.synchronize()


def summarize(times):
    return dict(median_ms=statistics.median(times), min_ms=min(times), max_ms=max(times), samples_ms=times)


@torch.inference_mode()
def fixed_bench(repeats, calls):
    rows = []
    cfg = tiny_config()
    for name, spec in FIXED:
        reference, inputs, expected = fixture(cfg, spec, 'cpu')
        ids, positions, meta = inputs
        expected = forward(reference, inputs).numpy()
        counts = tuple(q for q, _ in spec)
        lengths = [length for _, length in spec]
        caches = [mx.array(pool.numpy().reshape(-1, cfg.num_key_value_heads, cfg.head_dim))
                  for layer in reference.layers for pool in layer.self_attn.kv_cache]
        num_blocks = reference.layers[0].self_attn.kv_cache.shape[1]
        arrays = [mx.array(value.numpy(), dtype=mx.int32) for value in
                  [ids, positions, meta.slot_mapping, meta.block_table,
                   torch.tensor(lengths), torch.arange(len(ids))]]
        models = {mode: MLXModel(reference, 16, num_blocks, mode == 'mlx_compiled') for mode in MODES[1:]}
        states = {mode: list(caches) for mode in MODES[1:]}
        mx.eval(caches, arrays)
        reference = reference.to('mps')
        for layer in reference.layers:
            layer.self_attn.kv_cache = layer.self_attn.kv_cache.to('mps')
        gpu_meta = replace(meta, **{key: getattr(meta, key).to('mps') for key in
                           ['slot_mapping', 'query_start_loc', 'cu_seqlens_k', 'block_table']})
        gpu_inputs = (ids.to('mps'), positions.to('mps'), gpu_meta)

        def run(mode):
            if mode == 'pytorch_mps':
                return forward(reference, gpu_inputs)
            result, states[mode] = models[mode].forward(*arrays, states[mode], counts, lengths)
            mx.eval(result, states[mode])
            return result

        first, errors = {}, {}
        for mode in MODES:
            synchronize(mode)
            start = time.perf_counter_ns()
            result = run(mode)
            synchronize(mode)
            first[mode] = (time.perf_counter_ns() - start) / 1e6
            actual = result.cpu().numpy() if mode == 'pytorch_mps' else np.array(result)
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
            errors[mode] = float(np.max(np.abs(actual - expected)))
            for _ in range(3):
                run(mode)
        samples = {mode: [] for mode in MODES}
        for trial in range(repeats):
            order = MODES[trial % len(MODES):] + MODES[:trial % len(MODES)]
            for mode in order:
                synchronize(mode)
                start = time.perf_counter_ns()
                for _ in range(calls):
                    run(mode)
                synchronize(mode)
                samples[mode].append((time.perf_counter_ns() - start) / 1e6 / calls)
        for mode in MODES:
            row = dict(workload=name, mode=mode, **summarize(samples[mode]),
                       first_call_ms=first[mode], max_abs_error=errors[mode],
                       compiled_used=mode == 'mlx_compiled' and all(q == 1 for q in counts),
                       compiled_variants=models[mode].compiled_variants if mode != 'pytorch_mps' else 0)
            rows.append(row)
            print(f"fixed {name:16} {mode:13}: {row['median_ms']:.3f} ms", flush=True)
    return rows


def generation_workloads():
    return [('single-32', [list(range(4, 36))]),
            ('mixed-8', [[4 + i * 11 + t for t in range(length)]
                         for i, length in enumerate([16, 32, 48, 64, 80, 96, 112, 128])]),
            ('shared-8', [list(range(4, 68)) + [100 + i] * 16 for i in range(8)])]


def reset_metadata(engine):
    assert not engine.has_work()
    engine.kv = KVCacheManager(engine.cfg.num_blocks, engine.cfg.block_size, True)
    engine.scheduler = Scheduler(engine.cfg, engine.kv)


@torch.inference_mode()
def generation_bench(repeats):
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        config = Qwen3Config(vocab_size=1024, hidden_size=256, intermediate_size=768,
                             num_hidden_layers=4, num_attention_heads=8,
                             num_key_value_heads=2, head_dim=32, tie_word_embeddings=False)
        generation_fixture(folder, config=config)
        base = dict(model=folder, max_model_len=256, block_size=16, num_blocks=129,
                    max_num_seqs=8, max_num_batched_tokens=128)
        params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
        for name, prompts in generation_workloads():
            cpu = Engine(EngineConfig(**base, device='cpu'))
            expected = [out.token_ids for out in cpu.generate(prompts, params)]
            del cpu
            engines = {mode: Engine(EngineConfig(**base, device='mps',
                       backend='torch' if mode == 'pytorch_mps' else 'mlx',
                       compile_decode=mode == 'mlx_compiled')) for mode in MODES}
            first = {}

            def run(mode):
                reset_metadata(engines[mode])
                return engines[mode].generate(prompts, params)

            for mode in MODES:
                synchronize(mode)
                start = time.perf_counter_ns()
                outputs = run(mode)
                synchronize(mode)
                first[mode] = (time.perf_counter_ns() - start) / 1e6
                assert [out.token_ids for out in outputs] == expected, (name, mode)
                run(mode)
            samples = {mode: [] for mode in MODES}
            for trial in range(repeats):
                order = MODES[trial % len(MODES):] + MODES[:trial % len(MODES)]
                for mode in order:
                    synchronize(mode)
                    start = time.perf_counter_ns()
                    outputs = run(mode)
                    synchronize(mode)
                    samples[mode].append((time.perf_counter_ns() - start) / 1e6)
                    assert [out.token_ids for out in outputs] == expected, (name, mode)
            for mode in MODES:
                stats = summarize(samples[mode])
                tokens = len(prompts) * 32
                row = dict(workload=name, mode=mode, **stats, first_run_ms=first[mode],
                           output_tokens=tokens, output_tokens_per_second=tokens * 1000 / stats['median_ms'],
                           token_identical=True, compiled_variants=engines[mode].runner.model.compiled_variants
                           if mode != 'pytorch_mps' else 0)
                rows.append(row)
                print(f"generation {name:10} {mode:13}: {row['median_ms']:.2f} ms, "
                      f"{row['output_tokens_per_second']:.0f} output tok/s", flush=True)
            del engines
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--calls', type=int, default=10)
    args = ap.parse_args()
    if args.repeats < 3 or args.calls < 1:
        ap.error('Use at least 3 repeats and 1 call')
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        ap.error('Use MLX_ENABLE_TF32=0 for full-float32 comparisons')
    torch.set_num_threads(1)
    fixed = fixed_bench(args.repeats, args.calls)
    generation = generation_bench(args.repeats)
    data = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), torch=torch.__version__, os=platform.platform(),
                mlx=version('mlx'),
                parameters=3541504, vocab_size=1024, weights='random, no pretrained checkpoint',
                dtype='float32', mlx_enable_tf32=os.environ['MLX_ENABLE_TF32'], cpu_threads=1,
                samples=args.repeats, fixed_calls_per_sample=args.calls,
                fixed_scope='fixed-shape forward plus cache writes and logits; input prep excluded; first call separate',
                generation_scope='scheduler, input packing, model, greedy sampling, detokenization; loading excluded',
                generation_outputs_per_request=32, cache='prefix metadata reset for each generation run',
                order='mode order rotated each trial; all GPU work completed before/after timing',
                compilation='decode only; explicit cache inputs/outputs; power-of-two page-count buckets; max 16 cached variants',
                fixed_rows=fixed, generation_rows=generation)
    output = Path('benchmarks/results')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'mlx.json').write_text(json.dumps(data, indent=2) + '\n')
    from benchmarks.charts import plot_mlx
    plot_mlx(data, output / 'mlx.png')


if __name__ == '__main__':
    main()
