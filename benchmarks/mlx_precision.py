"""Reproduce the M5 float32 precision finding in two fresh MLX processes."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def child():
    import numpy as np
    from benchmarks.generation_inputs import generation_fixture
    from tiny_vllm.config import EngineConfig
    from tiny_vllm.engine import Engine
    from tiny_vllm.kv_cache import SamplingParams
    with tempfile.TemporaryDirectory() as folder:
        generation_fixture(folder)
        logits = {}
        for backend in ['torch', 'mlx']:
            cfg = EngineConfig(model=folder, backend=backend, device='cpu' if backend == 'torch' else 'mps',
                               max_model_len=64, block_size=4, num_blocks=17, max_num_batched_tokens=64)
            engine = Engine(cfg)
            engine.add_request(list(range(4, 25)), SamplingParams(temperature=0, max_tokens=1))
            batch = engine.scheduler.schedule()
            if backend == 'torch':
                _, output = engine.runner.run(batch)
                logits[backend] = output.cpu().numpy()
            else:
                logits[backend] = np.array(engine.runner.run_logits(batch))
        error = np.abs(logits['mlx'] - logits['torch'])
        print(json.dumps(dict(mlx_enable_tf32=os.environ['MLX_ENABLE_TF32'],
                              max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                              passes_tolerance=bool(np.allclose(logits['mlx'], logits['torch'], atol=3e-5, rtol=3e-5)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--child', action='store_true')
    args = ap.parse_args()
    if args.child:
        child()
        return
    rows = []
    for value in ['1', '0']:
        env = dict(os.environ, MLX_ENABLE_TF32=value)
        run = subprocess.run([sys.executable, '-m', 'benchmarks.mlx_precision', '--child'],
                             env=env, capture_output=True, text=True, check=True)
        rows.append(json.loads(run.stdout))
    data = dict(scope='same 21-token prefill, 86528 random parameters; CPU PyTorch vs eager MLX',
                dtype='float32 inputs and outputs', atol=3e-5, rtol=3e-5, rows=rows,
                note='Each precision setting runs in a fresh process; no tolerance relaxation')
    output = Path('benchmarks/results/mlx_precision.json')
    output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
