"""Run the local HTTP server: python -m tiny_vllm.serve --model PATH."""
import argparse
import tempfile

import uvicorn

from tiny_vllm.async_engine import AsyncEngine
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.server import build_app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', help='Local checkpoint directory or Hugging Face model ID')
    source.add_argument('--demo', action='store_true', help='Use a local random-weight fixture')
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--backend', choices=['torch', 'mlx'], default='torch')
    ap.add_argument('--compile-decode', action='store_true')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--max-model-len', type=int, default=512)
    ap.add_argument('--num-blocks', type=int, default=129)
    ap.add_argument('--max-num-seqs', type=int, default=8)
    ap.add_argument('--max-num-batched-tokens', type=int, default=512)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        if args.demo:
            from benchmarks.generation_inputs import generation_fixture
            generation_fixture(folder)
        cfg = EngineConfig(model=folder if args.demo else args.model, device=args.device,
                           backend=args.backend, compile_decode=args.compile_decode,
                           max_model_len=args.max_model_len, num_blocks=args.num_blocks,
                           max_num_seqs=args.max_num_seqs,
                           max_num_batched_tokens=args.max_num_batched_tokens)
        async_engine = AsyncEngine(Engine(cfg))
        try:
            app = build_app(async_engine, 'Tiny-vLLM-demo' if args.demo else args.model)
            uvicorn.run(app, host=args.host, port=args.port, workers=1)
        finally:
            async_engine.close()


if __name__ == '__main__':
    main()
