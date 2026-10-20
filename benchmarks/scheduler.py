"""CPU scheduling simulation; synthetic token arrivals, no model execution."""
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from tiny_vllm.kv_cache import KVCacheManager, Request, SamplingParams
from tiny_vllm.scheduler import Scheduler
from tiny_vllm.scheduler_config import SchedulerConfig


def simulate(seqs, budget, count=128):
    cfg = SchedulerConfig(16, seqs, budget)
    kv = KVCacheManager(4097, 16, True)
    scheduler = Scheduler(cfg, kv)
    requests = []
    for i in range(count):
        length = (64, 256, 512, 1024)[i % 4]
        ids = list(range(32)) + [10000 + i] * (length - 32)
        req = Request(str(i), ids, SamplingParams(max_tokens=32))
        scheduler.add(req)
        requests.append(req)
    steps = processed = 0
    schedule_ns = 0
    begin = time.perf_counter_ns()
    while scheduler.has_work():
        if steps >= 100000:
            raise RuntimeError('Simulation made insufficient progress')
        tick = time.perf_counter_ns()
        batch = scheduler.schedule()
        schedule_ns += time.perf_counter_ns() - tick
        steps += 1
        assert batch, 'Unexpected empty batch with sufficient capacity'
        tokens = sum(item.num_new for item in batch)
        assert tokens <= budget and len(batch) <= seqs
        processed += tokens
        for item in batch:
            if item.completes_prompt:
                # Stand in for the runner producing one sampled token.
                item.req.token_ids.append(20000 + len(item.req.output_ids))
                if len(item.req.output_ids) == item.req.params.max_tokens:
                    scheduler.finish(item.req, 'length')
    wall_ms = (time.perf_counter_ns() - begin) / 1e6
    assert all(req.finished and len(req.output_ids) == 32 for req in requests)
    assert kv.num_free == 4096 and not any(kv.ref_cnt)
    return dict(wall_ms=wall_ms, schedule_ms=schedule_ns / 1e6, steps=steps,
                processed_token_positions=processed,
                budget_utilization_percent=100 * processed / (steps * budget),
                cached_prompt_tokens=sum(req.num_cached_tokens for req in requests))


def chart(rows, output):
    from benchmarks.charts import plot_scheduler
    plot_scheduler(rows, output)


def main():
    cases = [(seqs, budget) for seqs in (1, 8, 32) for budget in (128, 512, 2048)]
    for seqs, budget in cases:
        simulate(seqs, budget, count=8)
    samples = {case: [] for case in cases}
    for trial in range(9):
        for seqs, budget in (cases if trial % 2 == 0 else cases[::-1]):
            samples[seqs, budget].append(simulate(seqs, budget))
    rows = []
    for seqs, budget in cases:
        runs = samples[seqs, budget]
        for key in ('steps', 'processed_token_positions', 'cached_prompt_tokens'):
            assert len({run[key] for run in runs}) == 1
        row = dict(max_num_seqs=seqs, token_budget=budget,
                   median_wall_ms=statistics.median(run['wall_ms'] for run in runs),
                   median_schedule_ms=statistics.median(run['schedule_ms'] for run in runs),
                   steps=runs[0]['steps'],
                   processed_token_positions=runs[0]['processed_token_positions'],
                   cached_prompt_tokens=runs[0]['cached_prompt_tokens'],
                   budget_utilization_percent=runs[0]['budget_utilization_percent'],
                   samples=runs)
        rows.append(row)
        print(f"seqs={seqs:2} budget={budget:4} rounds={row['steps']:4} "
              f"wall={row['median_wall_ms']:.2f} ms schedule={row['median_schedule_ms']:.2f} ms")
    # Reuse the previously collected chip information without a hardware inventory.
    previous = json.loads(Path('benchmarks/results/kv_cache.json').read_text())
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), os=platform.platform(), chip=previous['chip'],
                  device='CPU', requests=128, output_tokens_per_request=32,
                  prompt_lengths=[64, 256, 512, 1024], shared_prefix_tokens=32,
                  block_size=16, num_blocks=4097, repeats=9,
                  warmup_requests_per_case=8, arrival='all queued before timing',
                  timing_scope='scheduling, synthetic token append, completion and page release; excludes request construction',
                  rows=rows)
    output = Path('benchmarks/results')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'scheduler.json').write_text(json.dumps(result, indent=2) + '\n')
    chart(rows, output / 'scheduler.png')


if __name__ == '__main__':
    main()
