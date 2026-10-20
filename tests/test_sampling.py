import unittest

import torch

from tiny_vllm.kv_cache import SamplingParams
from tiny_vllm.sampling import sample


class SamplingTests(unittest.TestCase):
    def test_greedy_returns_argmax(self):
        logits = torch.tensor([[0., 3., 1.], [5., 1., 2.]])
        self.assertEqual(sample(logits, [SamplingParams(temperature=0)] * 2), [1, 0])

    def test_top_k_one_returns_highest_logit(self):
        logits = torch.tensor([[0., 3., 1.]]).repeat(100, 1)
        self.assertEqual(sample(logits, [SamplingParams(top_k=1)] * 100), [1] * 100)

    def test_top_k_never_samples_excluded_tokens(self):
        logits = torch.tensor([[0., 3., 2., -100.]]).repeat(200, 1)
        self.assertLessEqual(set(sample(logits, [SamplingParams(top_k=2)] * 200)), {1, 2})

    def test_nucleus_excludes_tail(self):
        logits = torch.tensor([[4., 3., -20., -21.]]).repeat(200, 1)
        self.assertLessEqual(set(sample(logits, [SamplingParams(top_p=0.8)] * 200)), {0, 1})

    def test_mixed_settings_keep_greedy_rows(self):
        logits = torch.tensor([[1., 5., 2.], [5., 1., 2.], [1., 2., 5.]])
        params = [SamplingParams(temperature=0), SamplingParams(top_k=1), SamplingParams(top_p=0.01)]
        self.assertEqual(sample(logits, params), [1, 0, 2])

    def test_seed_reproducibility_on_cpu(self):
        logits = torch.zeros(100, 8)
        params = [SamplingParams()] * 100
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            first = sample(logits, params)
            torch.manual_seed(29)
            self.assertEqual(first, sample(logits, params))

    def test_uniform_distribution_sanity(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            chosen = sample(torch.zeros(4000, 4), [SamplingParams()] * 4000)
        frequencies = torch.bincount(torch.tensor(chosen), minlength=4).float() / 4000
        self.assertTrue(torch.all((frequencies - 0.25).abs() < 0.04))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_mixed_sampling(self):
        logits = torch.tensor([[1., 5., 2.], [5., 1., 2.]], device='mps')
        self.assertEqual(sample(logits, [SamplingParams(temperature=0), SamplingParams(top_k=1)]), [1, 0])


if __name__ == '__main__':
    unittest.main()
