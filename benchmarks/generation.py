"""Offline generation vs HF using the same small local random-weight fixture."""
import argparse
import json
import platform
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.charts import plot_generation
from benchmarks.generation_inputs import generation_fixture
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.kv_cache import KVCacheManager, SamplingParams
from tiny_vllm.scheduler import Scheduler


def workloads():
    return [
        ('single-32', [list(range(4, 36))]),
        ('mixed-8', [[4 + (i * 11 + t) % 80 for t in range(length)]
                     for i, length in enumerate([16, 24, 32, 40, 48, 56, 64, 72])]),
        ('shared-8', [list(range(4, 28)) + [40 + i] * 8 for i in range(8)]),
    ]


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


@torch.inference_mode()
def hf_generate(model, prompts, device):
    width = max(map(len, prompts))
    ids = torch.zeros(len(prompts), width, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(prompts):
        ids[row, -len(prompt):] = torch.tensor(prompt, device=device)
        mask[row, -len(prompt):] = 1
    output = model.generate(ids, attention_mask=mask, do_sample=False,
                            max_new_tokens=16, pad_token_id=0, eos_token_id=None)
    return output[:, width:].cpu().tolist()


def reset_prefix_cache(engine):
    # Tensor storage stays allocated; only metadata is reset between samples.
    assert not engine.has_work() and not engine.requests and not engine.detok
    cfg = engine.cfg
    engine.kv = KVCacheManager(cfg.num_blocks, cfg.block_size, cfg.enable_prefix_caching)
    engine.scheduler = Scheduler(cfg, engine.kv)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--devices', nargs='+', choices=['cpu', 'mps'], default=['cpu'])
    args = ap.parse_args()
    devices = list(dict.fromkeys(args.devices))
    if 'mps' in devices and not torch.backends.mps.is_available():
        ap.error('MPS unavailable')
    torch.set_num_threads(1)
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        hf_cpu, _ = generation_fixture(folder)
        parameters = sum(p.numel() for p in hf_cpu.parameters())
        for device in devices:
            hf = hf_cpu.to(device)
            engine = Engine(EngineConfig(model=folder, device=device, max_model_len=128,
                                         block_size=8, max_num_seqs=8,
                                         max_num_batched_tokens=64, num_blocks=129))
            params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
            for name, prompts in workloads():
                expected = hf_generate(hf, prompts, device)
                reset_prefix_cache(engine)
                warm = engine.generate(prompts, params)
                assert [out.token_ids for out in warm] == expected
                samples = {'tiny_vllm': [], 'hf_static': []}
                cached = []
                for trial in range(7):
                    for method in (['tiny_vllm', 'hf_static'] if trial % 2 == 0 else ['hf_static', 'tiny_vllm']):
                        if method == 'tiny_vllm':
                            reset_prefix_cache(engine)
                        sync(device)
                        begin = time.perf_counter_ns()
                        if method == 'tiny_vllm':
                            outputs = engine.generate(prompts, params)
                            result = [out.token_ids for out in outputs]
                        else:
                            result = hf_generate(hf, prompts, device)
                        sync(device)
                        samples[method].append((time.perf_counter_ns() - begin) / 1e6)
                        assert result == expected, (name, device, method)
                        if method == 'tiny_vllm':
                            cached.append(sum(out.num_cached_tokens for out in outputs))
                row = dict(workload=name, device=device, requests=len(prompts),
                           prompt_lengths=list(map(len, prompts)), output_tokens=16 * len(prompts),
                           tiny_vllm_median_ms=statistics.median(samples['tiny_vllm']),
                           hf_static_median_ms=statistics.median(samples['hf_static']),
                           samples_ms=samples, cached_prompt_tokens=cached,
                           token_identical=True)
                rows.append(row)
                print(f"{device} {name}: Tiny-vLLM {row['tiny_vllm_median_ms']:.2f} ms, "
                      f"HF {row['hf_static_median_ms']:.2f} ms; token-identical", flush=True)
            del engine
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), torch=torch.__version__, os=platform.platform(),
                  parameters=parameters, vocab_size=97, weights='local random Qwen3 fixture, seed 23',
                  devices=devices, cpu_threads=1, dtype='float32', samples=7, warmup_runs=1,
                  output_tokens_per_request=16, generation='greedy, EOS ignored for fixed output length',
                  engine_config=dict(max_model_len=128, block_size=8, max_num_seqs=8,
                                     max_num_batched_tokens=64, num_blocks=129),
                  hf_backend='eager attention; one padded static batch per workload',
                  timing_scope='offline generation loop, input packing, sampling and returned output IDs; engine also detokenizes',
                  excluded='checkpoint loading, cache allocation, warmup and correctness checks',
                  cache='prefix metadata reset before each measured engine run',
                  synchronization='MPS synchronized before and after each generation run',
                  rows=rows)
    output = Path('benchmarks/results')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'generation.json').write_text(json.dumps(result, indent=2) + '\n')
    plot_generation(rows, output / 'generation.png')


if __name__ == '__main__':
    main()
