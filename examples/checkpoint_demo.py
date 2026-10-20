"""Load a checkpoint and sample one token; defaults to a local untrained fixture."""
import argparse
import tempfile

import torch

from benchmarks.checkpoint_inputs import checkpoint_fixture
from benchmarks.model_inputs import attach_cache, forward, make_inputs
from tiny_vllm.kv_cache import SamplingParams
from tiny_vllm.loading import load_model
from tiny_vllm.sampling import sample


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', help='Local checkpoint directory or Hugging Face model ID')
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        if args.model is None:
            checkpoint_fixture(folder)
        model = load_model(args.model or folder, max_position=64, device=args.device)
        pages = attach_cache(model, [19])
        inputs = make_inputs([19], [0], pages, args.device, vocab=model.cfg.vocab_size)
        logits = forward(model, inputs)[-1:]
        print('Checkpoint:', args.model or 'local random-weight fixture')
        print('Device:', args.device)
        print('Greedy next token ID:', sample(logits, [SamplingParams(temperature=0)])[0])
        print('Filtered next token ID:', sample(logits, [SamplingParams(temperature=0.7, top_k=20, top_p=0.9)])[0])


if __name__ == '__main__':
    main()
