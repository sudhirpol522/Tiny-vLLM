"""Synthetic, untrained decoder forward-pass benchmark; no generation claims."""
import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.model_inputs import attach_cache, forward, make_inputs, make_model, tiny_config


WORKLOADS = [
    ('prefill-32', [(32, 32)]),
    ('prefill-128', [(128, 128)]),
    ('decode-256', [(1, 256)]),
    ('decode-8x256', [(1, 256)] * 8),
    ('chunk-32-of-256', [(32, 256)]),
    ('mixed-3', [(1, 128), (32, 32), (16, 96)]),
]


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def fixture(cfg, spec, device):
    lengths = [length for _, length in spec]
    model = make_model(cfg, device, max_position=512)
    pages = attach_cache(model, lengths)
    full = make_inputs(lengths, [0] * len(lengths), pages, device, vocab=cfg.vocab_size)
    full_logits = forward(model, full)
    selected, cursor = [], 0
    for q_len, length in spec:
        selected.append(full_logits[cursor + length - q_len:cursor + length])
        cursor += length
    inputs = make_inputs(lengths, [length - q for q, length in spec], pages, device, vocab=cfg.vocab_size)
    return model, inputs, torch.cat(selected).cpu()


def render_chart(rows, output):
    from benchmarks.charts import plot_model
    plot_model(rows, output)


@torch.inference_mode()
def benchmark(devices):
    torch.set_num_threads(1)
    cfg = tiny_config()
    rows = []
    parameters = None
    for name, spec in WORKLOADS:
        reference, _, expected = fixture(cfg, spec, 'cpu')
        parameters = sum(p.numel() for p in reference.parameters())
        del reference
        for device in devices:
            model, inputs, full_result = fixture(cfg, spec, device)
            actual = forward(model, inputs).cpu()
            torch.testing.assert_close(actual, full_result, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
            errors = dict(vs_full_prefill=(actual - full_result).abs().max().item(),
                          vs_cpu_full_prefill=(actual - expected).abs().max().item())
            for _ in range(3):
                forward(model, inputs)
            samples = []
            for _ in range(7):
                sync(device)
                begin = time.perf_counter_ns()
                for _ in range(5):
                    forward(model, inputs)
                sync(device)
                samples.append((time.perf_counter_ns() - begin) / 1e6 / 5)
            row = dict(workload=name, device=device,
                       spec=[dict(query_tokens=q, kv_tokens=k) for q, k in spec],
                       median_ms=statistics.median(samples), samples_ms=samples,
                       max_abs_error=errors)
            rows.append(row)
            print(f"{device:3s} {name:20s} {row['median_ms']:.3f} ms "
                  f"max_error={max(errors.values()):.2e}", flush=True)
            del model, inputs
    return cfg, parameters, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--devices', nargs='+', choices=['cpu', 'mps'], default=['cpu'])
    args = ap.parse_args()
    devices = list(dict.fromkeys(args.devices))
    if 'mps' in devices and not torch.backends.mps.is_available():
        ap.error('MPS is unavailable in this process')
    cfg, parameters, rows = benchmark(devices)
    hardware = subprocess.run(['system_profiler', 'SPHardwareDataType'], capture_output=True, text=True, check=False)
    chip = next((line.strip() for line in hardware.stdout.splitlines() if 'Chip:' in line), platform.machine())
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  torch=torch.__version__, os=platform.platform(), chip=chip, devices=devices,
                  config=vars(cfg), parameters=parameters, weights='random CPU initialization, seed 17; no checkpoint',
                  page_seed=19, max_position=512, block_size=16, dtype='float32', cpu_threads=1,
                  warmup_calls=3, samples=7, calls_per_sample=5,
                  timing_scope='embeddings, all decoder layers, cache writes, reference attention, final norm, all query-position logits',
                  excluded='model/input construction, full-context priming, correctness checks, sampling, scheduler',
                  replay='same query tokens, positions and context each call; query slots overwritten; no growing generation',
                  synchronization='MPS synchronized before and after each timed 5-call sample',
                  device_order='CPU then MPS per workload; no randomized device ordering',
                  tolerance=dict(atol=3e-5, rtol=3e-5), rows=rows)
    output = Path('benchmarks/results')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model.json').write_text(json.dumps(result, indent=2) + '\n')
    render_chart(rows, output / 'model.png')


if __name__ == '__main__':
    main()
