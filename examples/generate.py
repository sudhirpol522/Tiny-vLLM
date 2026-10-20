"""Offline generation using a local fixture or a compatible checkpoint."""
import argparse
import tempfile

from benchmarks.generation_inputs import generation_fixture
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.kv_cache import SamplingParams


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', help='Local directory or Hugging Face model ID')
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--prompt')
    ap.add_argument('--max-tokens', type=int, default=16)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        if args.model is None:
            generation_fixture(folder)
            print('Using random weights and a synthetic token vocabulary.')
        engine = Engine(EngineConfig(model=args.model or folder, device=args.device))
        prompt = args.prompt or ('The capital of France is' if args.model else 't4 t5 t6')
        result = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=args.max_tokens))[0]
        print('Prompt:', prompt)
        print('Output:', result.text)
        print('Token IDs:', result.token_ids)
        print('Finish reason:', result.finish_reason)


if __name__ == '__main__':
    main()
