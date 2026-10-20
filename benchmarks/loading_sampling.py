"""Benchmark a local fixture load and upstream sampling on synthetic logits."""
import argparse
import json
import platform
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.checkpoint_inputs import checkpoint_fixture
from benchmarks.charts import plot_loading, plot_sampling
from tiny_vllm.kv_cache import SamplingParams
from tiny_vllm.loading import load_model
from tiny_vllm.sampling import sample


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def measure(fn, device, warmups=3, repeats=7, calls=5):
    for _ in range(warmups):
        fn()
    times = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter_ns()
        for _ in range(calls):
            result = fn()
        sync(device)
        times.append((time.perf_counter_ns() - start) / 1e6 / calls)
    return times, result


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--devices', nargs='+', choices=['cpu', 'mps'], default=['cpu'])
    args = ap.parse_args()
    devices = list(dict.fromkeys(args.devices))
    if 'mps' in devices and not torch.backends.mps.is_available():
        ap.error('MPS unavailable')
    torch.set_num_threads(1)
    base = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), torch=torch.__version__,
                os=platform.platform(), devices=devices, cpu_threads=1,
                dtype='float32', samples=7, synchronization='MPS synchronized before and after each sample')
    output = Path('benchmarks/results')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        hf = checkpoint_fixture(folder)
        state = hf.state_dict()
        parameters = sum(p.numel() for p in hf.parameters())
        checkpoint_bytes = (Path(folder) / 'model.safetensors').stat().st_size
        for device in devices:
            def load():
                return load_model(folder, max_position=64, device=device)
            times, model = measure(load, device, warmups=1, calls=1)
            # Loading must reproduce every parameter exactly, outside timing.
            for name, param in model.named_parameters():
                source = state.get('model.' + name, state.get(name))
                torch.testing.assert_close(param.cpu(), source, atol=0, rtol=0)
            row = dict(device=device, median_ms=statistics.median(times), samples_ms=times)
            rows.append(row)
            print(f"load {device}: {row['median_ms']:.3f} ms", flush=True)
        loading = dict(base, parameters=parameters, checkpoint_bytes=checkpoint_bytes,
                       checkpoint='local random-weight Qwen3 fixture, seed 23; no network',
                       warmup_loads=1, calls_per_sample=1, max_position=64,
                       timing_scope='config parse, CPU model construction, safetensors copies, device transfer; imports excluded',
                       cache='warm filesystem cache; not download or cold-disk timing', rows=rows)
        (output / 'loading.json').write_text(json.dumps(loading, indent=2) + '\n')
        plot_loading(rows, output / 'loading.png')
    rows = []
    generator = torch.Generator().manual_seed(37)
    for batch in (1, 8, 32):
        cpu_logits = torch.randn(batch, 32768, generator=generator)
        for mode in ('greedy', 'filtered'):
            params = [SamplingParams(temperature=0 if mode == 'greedy' or i % 4 == 3 else 0.7,
                                     top_k=50, top_p=0.9) for i in range(batch)]
            for device in devices:
                logits = cpu_logits.to(device)
                fn = lambda: sample(logits, params)
                times, chosen = measure(fn, device)
                for i, (token, param) in enumerate(zip(chosen, params)):
                    if param.temperature == 0:
                        assert token == cpu_logits[i].argmax().item()
                    else:
                        allowed = cpu_logits[i].topk(50).indices.tolist()
                        assert token in allowed
                row = dict(workload=f'{mode}-{batch}', device=device, batch=batch,
                           mode=mode, median_ms=statistics.median(times), samples_ms=times)
                rows.append(row)
                print(f"sample {device} {mode}-{batch}: {row['median_ms']:.3f} ms", flush=True)
    sampling = dict(base, vocab_size=32768, logit_seed=37, warmup_calls=3, calls_per_sample=5,
                    filtered='temperature 0.7, top-k 50, top-p 0.9; every fourth row greedy',
                    timing_scope='upstream sample() including filtering, RNG and returned Python token IDs',
                    rows=rows)
    (output / 'sampling.json').write_text(json.dumps(sampling, indent=2) + '\n')
    plot_sampling(rows, output / 'sampling.png')


if __name__ == '__main__':
    main()
