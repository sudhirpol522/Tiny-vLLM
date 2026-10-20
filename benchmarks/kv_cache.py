"""CPU metadata benchmark. Run from the repository root; no third-party packages."""
import argparse
import datetime
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from tiny_vllm.kv_cache import KVCacheManager, Request, SamplingParams


def workload(caching, prefix, requests, prompt_len, block_size):
    # Capacity for two full prompts prevents eviction from confounding reuse.
    mgr = KVCacheManager(1 + 2 * ((prompt_len + block_size - 1) // block_size),
                         block_size, caching)
    reused = 0
    start = time.perf_counter_ns()
    for i in range(requests):
        # Synthetic IDs: only the chosen prefix is shared across requests.
        ids = list(range(prefix)) + [10000 + i] * (prompt_len - prefix)
        req = Request(str(i), ids, SamplingParams())
        hits = mgr.find_cached_prefix(req)
        cached = len(hits) * block_size
        if not mgr.allocate(req, prompt_len - cached, hits):
            raise RuntimeError("Unexpected allocation failure")
        req.num_computed_tokens = prompt_len
        reused += cached
        mgr.free_request(req)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    assert mgr.num_free == len(mgr.ref_cnt) - 1
    assert not any(mgr.ref_cnt)
    expected = ((requests - 1) * (min(prefix, prompt_len - 1) // block_size)
                * block_size if caching else 0)
    assert reused == expected, (reused, expected)
    return elapsed_ms, reused


def render_chart(rows, destination):
    from benchmarks.charts import plot_kv
    plot_kv(rows, destination)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeats', type=int, default=9)
    args = ap.parse_args()
    if args.repeats < 2:
        ap.error('--repeats must be at least 2')
    rows = []
    cases = [(prefix, caching) for prefix in (0, 256, 768) for caching in (False, True)]
    for prefix, caching in cases:
        workload(caching, prefix, 100, 1024, 16)
    samples = {case: [] for case in cases}
    # Reverse order each round to reduce consistent order bias.
    for trial in range(args.repeats):
        for prefix, caching in (cases if trial % 2 == 0 else cases[::-1]):
            elapsed, reused = workload(caching, prefix, 1000, 1024, 16)
            samples[prefix, caching].append(elapsed)
    for prefix, caching in cases:
        times = samples[prefix, caching]
        reused = (999 * prefix) if caching else 0
        rows.append(dict(shared_prefix_tokens=prefix, caching=caching,
                         median_ms=statistics.median(times), min_ms=min(times),
                         max_ms=max(times), samples_ms=times,
                         reusable_prompt_tokens=reused,
                         reusable_percent=100 * reused / (1000 * 1024)))
    cpu = subprocess.run(['system_profiler', 'SPHardwareDataType'],
                         capture_output=True, text=True, check=False)
    # Keep chip information only; never record the machine serial number/UUID.
    chip = next((line.strip() for line in cpu.stdout.splitlines() if 'Chip:' in line), platform.machine())
    result = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  python=platform.python_version(), os=platform.platform(), chip=chip,
                  device='CPU', prompt_tokens=1024, block_size=16, requests=1000,
                  repeats=args.repeats, warmup_requests_per_case=100,
                  timing_scope='request construction, lookup, allocation, release; no KV tensors or model',
                  rows=rows)
    out = Path('benchmarks/results')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'kv_cache.json').write_text(json.dumps(result, indent=2) + '\n')
    render_chart(rows, out / 'kv_cache.png')
    for row in rows:
        print(f"prefix={row['shared_prefix_tokens']:3d} cache={str(row['caching']):5s} "
              f"median={row['median_ms']:.2f} ms reusable={row['reusable_percent']:.2f}%")
    print(chip)


if __name__ == '__main__':
    main()
