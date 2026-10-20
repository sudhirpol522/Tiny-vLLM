"""Time the unchanged paged reference against a pre-gathered dense control."""
import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.attention_inputs import dense_control, dense_oracle, make_case
from tiny_vllm.attention import torch_ref_attention


WORKLOADS = [
    ('decode-256', [(1, 256)]),
    ('decode-4096', [(1, 4096)]),
    ('decode-8x1024', [(1, 1024)] * 8),
    ('prefill-256', [(256, 256)]),
    ('chunk-128-of-1024', [(128, 1024)]),
    ('mixed-3', [(1, 1024), (64, 64), (32, 512)]),
]


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()


def render_chart(rows, device, path):
    from benchmarks.charts import plot_attention
    plot_attention(rows, device, path)


@torch.inference_mode()
def benchmark(devices):
    torch.set_num_threads(1)
    rows = []
    for device in devices:
        for name, spec in WORKLOADS:
            case = make_case(spec, device=device)
            oracle = dense_oracle(case)
            methods = {
                'paged': lambda: torch_ref_attention(case.q, case.k_cache, case.v_cache, case.meta, case.scale),
                'dense': lambda: dense_control(case),
            }
            errors = {}
            for label, fn in methods.items():
                actual = fn().cpu()
                torch.testing.assert_close(actual, oracle, atol=3e-5, rtol=3e-5)
                errors[label] = (actual - oracle).abs().max().item()
                for _ in range(3):
                    fn()
            synchronize(device)
            samples = {'paged': [], 'dense': []}
            for trial in range(7):
                for label in (['paged', 'dense'] if trial % 2 == 0 else ['dense', 'paged']):
                    synchronize(device)
                    start = time.perf_counter_ns()
                    for _ in range(5):
                        methods[label]()
                    synchronize(device)
                    samples[label].append((time.perf_counter_ns() - start) / 1e6 / 5)
            row = dict(device=device, workload=name, spec=[dict(query_tokens=q, kv_tokens=k) for q, k in spec],
                       paged_median_ms=statistics.median(samples['paged']),
                       dense_median_ms=statistics.median(samples['dense']), samples_ms=samples,
                       max_abs_error_vs_float64_oracle=errors)
            rows.append(row)
            print(f"{device:3s} {name:20s} paged={row['paged_median_ms']:.3f} ms "
                  f"dense={row['dense_median_ms']:.3f} ms max_error={max(errors.values()):.2e}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--devices', nargs='+', choices=['cpu', 'mps'], default=['cpu'])
    args = ap.parse_args()
    if 'mps' in args.devices and not torch.backends.mps.is_available():
        ap.error('MPS is unavailable in this process; run CPU only or allow Apple GPU access')
    devices = list(dict.fromkeys(args.devices))
    rows = benchmark(devices)
    hardware = subprocess.run(['system_profiler', 'SPHardwareDataType'], capture_output=True, text=True, check=False)
    chip = next((line.strip() for line in hardware.stdout.splitlines() if 'Chip:' in line), platform.machine())
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  torch=torch.__version__, os=platform.platform(), chip=chip,
                  devices=devices, dtype='float32', cpu_threads=torch.get_num_threads(),
                  query_heads=8, kv_heads=2, head_dim=64, block_size=16, seed=7,
                  warmup_calls=3, samples=7, calls_per_sample=5,
                  tolerance=dict(atol=3e-5, rtol=3e-5),
                  timing_scope='attention only, includes gather/head expansion/mask; excludes input construction, KV writes, validation',
                  synchronization='MPS synchronized before and after each timed 5-call sample',
                  control='same SDPA/mask/head expansion with K/V already contiguous; not a serving baseline',
                  rows=rows)
    output = Path('benchmarks/results')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'attention.json').write_text(json.dumps(result, indent=2) + '\n')
    for device in devices:
        render_chart(rows, device, output / f'attention_{device}.png')


if __name__ == '__main__':
    main()
