import unittest

import torch

from benchmarks.model_inputs import (attach_cache, attention_context, dense_model_oracle,
                                     forward, make_inputs, make_model, tiny_config)
from tiny_vllm import model as model_module
from tiny_vllm.model import RMSNorm, rotary


class ModelTests(unittest.TestCase):
    def cfg(self, **overrides):
        values = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      vocab_size=97, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
        values.update(overrides)
        return tiny_config(**values)

    def test_rmsnorm_matches_float64_equation(self):
        x = torch.tensor([[1., -2., 3., -4.], [0., 0., 0., 0.]])
        norm = RMSNorm(4, 1e-6)
        expected = x.double() / (x.double().square().mean(-1, keepdim=True) + 1e-6).sqrt()
        torch.testing.assert_close(norm(x), expected.float())

    def test_rotary_is_identity_at_zero_and_preserves_norm(self):
        x = torch.randn(3, 2, 16)
        angle = torch.arange(24).reshape(3, 8) / 7
        torch.testing.assert_close(rotary(x, torch.ones(3, 8), torch.zeros(3, 8)), x)
        rotated = rotary(x, angle.cos(), angle.sin())
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))

    @torch.inference_mode()
    def test_full_model_matches_independent_dense_oracle(self):
        model = make_model(self.cfg(), max_position=64)
        pages = attach_cache(model, [19])
        inputs = make_inputs([19], [0], pages, vocab=97)
        actual = forward(model, inputs)
        expected = dense_model_oracle(model, inputs[0])
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    @torch.inference_mode()
    def test_chunked_prefill_and_decode_match_full_forward(self):
        model = make_model(self.cfg(), max_position=64)
        pages = attach_cache(model, [23])
        expected = forward(model, make_inputs([23], [0], pages, vocab=97))
        for layer in model.layers:
            layer.self_attn.kv_cache.fill_(float('nan'))
        pieces = []
        for start, end in [(0, 7), (7, 18), (18, 22), (22, 23)]:
            pieces.append(forward(model, make_inputs([end], [start], pages, vocab=97)))
        torch.testing.assert_close(torch.cat(pieces), expected, atol=2e-5, rtol=2e-5)

    @torch.inference_mode()
    def test_packed_requests_are_isolated(self):
        model = make_model(self.cfg(), max_position=64)
        pages = attach_cache(model, [9, 21])
        inputs = make_inputs([9, 21], [0, 0], pages, vocab=97)
        actual = forward(model, inputs)
        expected = torch.cat([dense_model_oracle(model, inputs[0][:9]),
                              dense_model_oracle(model, inputs[0][9:])])
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    @torch.inference_mode()
    def test_tied_output_head_matches_dense_oracle(self):
        model = make_model(self.cfg(tie_word_embeddings=True), max_position=64)
        self.assertFalse(hasattr(model, 'lm_head'))
        pages = attach_cache(model, [11])
        inputs = make_inputs([11], [0], pages, vocab=97)
        torch.testing.assert_close(forward(model, inputs), dense_model_oracle(model, inputs[0]),
                                   atol=1e-4, rtol=2e-5)

    @torch.inference_mode()
    def test_prefix_cache_slots_are_not_overwritten_by_decode(self):
        model = make_model(self.cfg(), max_position=64)
        pages = attach_cache(model, [17])
        prefix = make_inputs([16], [0], pages, vocab=97)
        forward(model, prefix)
        slots = prefix[2].slot_mapping
        before = [layer.self_attn.kv_cache.flatten(1, 2)[:, slots].clone() for layer in model.layers]
        forward(model, make_inputs([17], [16], pages, vocab=97))
        for layer, old in zip(model.layers, before):
            torch.testing.assert_close(layer.self_attn.kv_cache.flatten(1, 2)[:, slots], old, atol=0, rtol=0)

    def test_qwen2_bias_and_qwen3_head_normalization(self):
        qwen2 = make_model(self.cfg(model_type='qwen2'), max_position=16)
        qwen3 = make_model(self.cfg(), max_position=16)
        self.assertIsNotNone(qwen2.layers[0].self_attn.q_proj.bias)
        self.assertFalse(qwen2.layers[0].self_attn.qk_norm)
        self.assertIsNone(qwen3.layers[0].self_attn.q_proj.bias)
        self.assertTrue(qwen3.layers[0].self_attn.qk_norm)

    def test_metadata_restored_after_exception(self):
        previous = model_module.ATTN_META
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with attention_context(object()):
                raise RuntimeError('test')
        self.assertIs(model_module.ATTN_META, previous)

    def test_non_default_rope_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'plain RoPE'):
            make_model(self.cfg(rope_scaling={'rope_type': 'linear'}), max_position=16)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable in this process')
    @torch.inference_mode()
    def test_mps_logits_match_cpu_dense_oracle(self):
        model = make_model(self.cfg(), device='mps', max_position=64)
        pages = attach_cache(model, [19])
        inputs = make_inputs([19], [0], pages, device='mps', vocab=97)
        actual = forward(model, inputs).cpu()
        expected = dense_model_oracle(model, inputs[0])
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
