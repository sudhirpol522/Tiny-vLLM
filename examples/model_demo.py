"""Compare chunked and full random-weight model forwards without downloads."""
import argparse

import torch

from benchmarks.model_inputs import attach_cache, forward, make_inputs, make_model, tiny_config


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = ap.parse_args()
    if args.device == 'mps' and not torch.backends.mps.is_available():
        ap.error('MPS is unavailable in this process')
    cfg = tiny_config()
    model = make_model(cfg, args.device, max_position=64)
    pages = attach_cache(model, [23])
    full = forward(model, make_inputs([23], [0], pages, args.device, vocab=cfg.vocab_size))
    for layer in model.layers:
        layer.self_attn.kv_cache.fill_(float('nan'))
    pieces = []
    for start, end in [(0, 7), (7, 18), (18, 22), (22, 23)]:
        logits = forward(model, make_inputs([end], [start], pages, args.device, vocab=cfg.vocab_size))
        pieces.append(logits)
        print(f'Positions {start}:{end}: logits {tuple(logits.shape)}')
    chunked = torch.cat(pieces)
    torch.testing.assert_close(chunked, full, atol=3e-5, rtol=3e-5)
    print('Device:', args.device)
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    print('Maximum full/chunked logit difference:', (full - chunked).abs().max().item())
    print('Random weights only; no trained model or text generation.')


if __name__ == '__main__':
    main()
