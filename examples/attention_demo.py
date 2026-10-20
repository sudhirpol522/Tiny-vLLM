"""Mixed decode/prefill attention demo, without model weights or FlashAttention."""
import argparse

import torch

from benchmarks.attention_inputs import dense_oracle, make_case
from tiny_vllm.attention import torch_ref_attention


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = ap.parse_args()
    if args.device == 'mps' and not torch.backends.mps.is_available():
        ap.error('MPS is unavailable in this process')
    case = make_case([(1, 67), (19, 19), (5, 37)], device=args.device)
    output = torch_ref_attention(case.q, case.k_cache, case.v_cache, case.meta, case.scale)
    expected = dense_oracle(case)
    torch.testing.assert_close(output.cpu(), expected, atol=3e-5, rtol=3e-5)
    print('Device:', args.device)
    print('Packed queries:', tuple(case.q.shape))
    print('Query boundaries:', case.meta.query_start_loc_cpu)
    print('Full context lengths:', case.meta.seq_lens_cpu)
    print('Page table:', case.meta.block_table.cpu().tolist())
    print('Output:', tuple(output.shape))
    print('Maximum absolute error vs float64 dense oracle:',
          (output.cpu() - expected).abs().max().item())


if __name__ == '__main__':
    main()
