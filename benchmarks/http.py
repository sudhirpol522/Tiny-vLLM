"""Loopback HTTP/SSE timings against a local untrained fixture."""
import argparse
import json
import platform
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import torch

from benchmarks.charts import plot_http
from benchmarks.generation_inputs import generation_fixture
from benchmarks.http_inputs import live_server
from tiny_vllm.async_engine import AsyncEngine
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.server import build_app


def request(client, stream, expected):
    body = dict(model='Tiny-vLLM-demo', prompt=list(range(4, 20)), temperature=0,
                max_tokens=16, ignore_eos=True, stream=stream)
    begin = time.perf_counter_ns()
    first_event_ms = None
    if stream:
        body['stream_options'] = {'include_usage': True}
        text, usage, done = '', None, False
        with client.stream('POST', '/v1/completions', json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith('data: '):
                    continue
                data = line[6:]
                if data == '[DONE]':
                    done = True
                    break
                chunk = json.loads(data)
                assert 'error' not in chunk, chunk
                if first_event_ms is None:
                    first_event_ms = (time.perf_counter_ns() - begin) / 1e6
                if chunk['choices']:
                    text += chunk['choices'][0]['text']
                if 'usage' in chunk:
                    usage = chunk['usage']
        assert done and usage['completion_tokens'] == 16
    else:
        response = client.post('/v1/completions', json=body)
        response.raise_for_status()
        result = response.json()
        text = result['choices'][0]['text']
        assert result['usage']['completion_tokens'] == 16
    elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
    assert text == expected, (text, expected)
    return dict(elapsed_ms=elapsed_ms, first_event_ms=first_event_ms)


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
        hf, tokenizer = generation_fixture(folder)
        with torch.inference_mode():
            expected_ids = hf.generate(torch.tensor([list(range(4, 20))]), max_new_tokens=16,
                                       eos_token_id=None, pad_token_id=0)[0, 16:].tolist()
        expected = tokenizer.decode(expected_ids, skip_special_tokens=True)
        for device in devices:
            cfg = EngineConfig(model=folder, device=device, max_model_len=128,
                               max_num_batched_tokens=64, num_blocks=129)
            worker = AsyncEngine(Engine(cfg))
            try:
                with live_server(build_app(worker, 'Tiny-vLLM-demo')) as url:
                    with httpx.Client(base_url=url, timeout=30) as client:
                        for concurrency in (1, 4):
                            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                                def run(stream):
                                    begin = time.perf_counter_ns()
                                    requests = list(pool.map(lambda _: request(client, stream, expected),
                                                             range(concurrency)))
                                    elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
                                    return dict(wall_ms=elapsed_ms, requests=requests)
                                for stream in (False, True):
                                    run(stream)
                                samples = {False: [], True: []}
                                for trial in range(7):
                                    for stream in ([False, True] if trial % 2 == 0 else [True, False]):
                                        samples[stream].append(run(stream))
                                for stream in (False, True):
                                    runs = samples[stream]
                                    first = [r['first_event_ms'] for sample in runs for r in sample['requests']
                                             if r['first_event_ms'] is not None]
                                    row = dict(device=device, concurrency=concurrency, stream=stream,
                                               median_ms=statistics.median(r['wall_ms'] for r in runs),
                                               median_first_event_ms=statistics.median(first) if first else None,
                                               samples=runs, text_matches_hugging_face=True)
                                    rows.append(row)
                                    print(f"{device} concurrent={concurrency} stream={stream}: "
                                          f"{row['median_ms']:.2f} ms", flush=True)
            finally:
                worker.close()
    data = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), torch=torch.__version__, os=platform.platform(),
                devices=devices, cpu_threads=1, dtype='float32', parameters=86528, vocab_size=97,
                checkpoint='local random-weight Qwen3 fixture, seed 23', prompt_tokens=16,
                output_tokens_per_request=16, samples=7, warmup_runs_per_mode=1,
                cache='warm prefix cache, repeated identical prompt; metadata not reset',
                timing_scope='loopback TCP, HTTP, queueing, generation, serialization, and client parsing',
                excluded='server startup, checkpoint loading, warmup and connection-pool creation',
                first_event='time from client call to first SSE data event; not measured for nonstream',
                synchronization='HTTP completion waits for all generated outputs; no extra MPS sync',
                rows=rows)
    output = Path('benchmarks/results')
    (output / 'http.json').write_text(json.dumps(data, indent=2) + '\n')
    plot_http(rows, output / 'http.png')


if __name__ == '__main__':
    main()
