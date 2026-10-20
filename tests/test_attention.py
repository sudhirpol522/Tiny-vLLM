import unittest

import torch

from benchmarks.attention_inputs import dense_oracle, make_case
from tiny_vllm.attention import torch_ref_attention


class AttentionTests(unittest.TestCase):
    def check_case(self, spec, **kwargs):
        case = make_case(spec, **kwargs)
        out = torch_ref_attention(case.q, case.k_cache, case.v_cache, case.meta, case.scale)
        torch.testing.assert_close(out.cpu(), dense_oracle(case), atol=3e-5, rtol=3e-5)
        self.assertTrue(torch.isfinite(out).all())
        return case, out

    def test_full_prefill(self):
        self.check_case([(33, 33)])

    def test_chunked_prefill_uses_bottom_right_mask(self):
        self.check_case([(7, 41)])

    def test_decode_reads_complete_prefix(self):
        self.check_case([(1, 67)])

    def test_mixed_lengths_ignore_null_page_and_tail(self):
        self.check_case([(1, 67), (19, 19), (5, 37)])

    def test_equal_query_and_kv_heads(self):
        self.check_case([(9, 25)], heads=2, kv_heads=2)

    def test_single_kv_head(self):
        self.check_case([(4, 17)], heads=8, kv_heads=1)

    def test_future_values_do_not_change_earlier_query(self):
        case = make_case([(8, 8)])
        before = torch_ref_attention(case.q, case.k_cache, case.v_cache, case.meta, case.scale)
        last_slot = case.meta.slot_mapping[-1]
        case.v_cache.view(-1, 2, 64)[last_slot] += 100
        after = torch_ref_attention(case.q, case.k_cache, case.v_cache, case.meta, case.scale)
        torch.testing.assert_close(before[:-1], after[:-1], atol=0, rtol=0)
        self.assertFalse(torch.allclose(before[-1], after[-1]))

    def test_slot_mapping_writes_new_keys_and_values(self):
        case = make_case([(3, 19), (1, 33)])
        case.k_cache.view(-1, 2, 64)[case.meta.slot_mapping] = 0
        case.v_cache.view(-1, 2, 64)[case.meta.slot_mapping] = 0
        keys = torch.cat([k[-q:] for k, q in zip(case.keys, [3, 1])])
        values = torch.cat([v[-q:] for v, q in zip(case.values, [3, 1])])
        case.k_cache.view(-1, 2, 64).index_copy_(0, case.meta.slot_mapping, keys)
        case.v_cache.view(-1, 2, 64).index_copy_(0, case.meta.slot_mapping, values)
        out = torch_ref_attention(case.q, case.k_cache, case.v_cache, case.meta, case.scale)
        torch.testing.assert_close(out, dense_oracle(case), atol=3e-5, rtol=3e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable in this process')
    def test_mps_mixed_batch(self):
        self.check_case([(1, 67), (19, 19), (5, 37)], device='mps')


if __name__ == '__main__':
    unittest.main()
