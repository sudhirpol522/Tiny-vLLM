import tempfile
import unittest

import torch

from benchmarks.generation_inputs import generation_fixture
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.kv_cache import SamplingParams


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.hf, cls.tokenizer = generation_fixture(cls.folder.name)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def engine(self, **options):
        cfg = dict(model=self.folder.name, max_model_len=64, block_size=4,
                   max_num_seqs=3, max_num_batched_tokens=8, num_blocks=49)
        cfg.update(options)
        return Engine(EngineConfig(**cfg))

    @torch.inference_mode()
    def hf_tokens(self, prompt, count):
        ids = torch.tensor([prompt])
        result = self.hf.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                  max_new_tokens=count, pad_token_id=0, eos_token_id=None)
        return result[0, len(prompt):].tolist()

    def test_greedy_generation_matches_hugging_face(self):
        prompts = [[4, 5, 6], list(range(4, 25)), list(range(4, 39))]
        outputs = self.engine().generate(prompts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
        self.assertEqual([out.token_ids for out in outputs], [self.hf_tokens(p, 8) for p in prompts])
        self.assertTrue(all(out.finished and out.finish_reason == 'length' for out in outputs))

    def test_chunked_prefill_does_not_emit_early_tokens(self):
        engine = self.engine(max_num_batched_tokens=4)
        engine.add_request(list(range(4, 15)), SamplingParams(temperature=0, max_tokens=1))
        self.assertEqual(engine.step(), [])
        self.assertEqual(engine.step(), [])
        self.assertEqual(len(engine.step()[0].token_ids), 1)

    def test_prefix_reuse_preserves_generation(self):
        engine = self.engine()
        prompt = list(range(4, 25))
        params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
        first = engine.generate([prompt], params)[0]
        second = engine.generate([prompt], params)[0]
        self.assertGreaterEqual(second.num_cached_tokens, 20)
        self.assertEqual(first.token_ids, second.token_ids)
        self.assertEqual(engine.kv.num_free, 48)

    def test_prefix_on_and_off_match(self):
        prompts = [list(range(4, 25)), list(range(4, 25)) + [40]]
        params = SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)
        on = self.engine().generate(prompts, params)
        off = self.engine(enable_prefix_caching=False).generate(prompts, params)
        self.assertEqual([o.token_ids for o in on], [o.token_ids for o in off])

    def test_new_request_joins_ongoing_decode(self):
        engine = self.engine()
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        first = engine.add_request([4, 5], params, 'first')
        self.assertEqual(engine.step()[0].request_id, first)
        engine.add_request([6, 7], params, 'second')
        self.assertEqual({out.request_id for out in engine.step()}, {'first', 'second'})
        while engine.has_work():
            engine.step()
        self.assertEqual(engine.kv.num_free, 48)
        self.assertFalse(engine.requests)
        self.assertFalse(engine.detok)

    def test_preemption_recovers_correct_outputs(self):
        engine = self.engine(num_blocks=17)
        prompts = [list(range(4, 28)), list(range(30, 50)), list(range(4, 40))]
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        ids = [engine.add_request(p, params) for p in prompts]
        done, preempted = {}, False
        for _ in range(200):
            if not engine.has_work():
                break
            previous = {rid: req.num_computed_tokens for rid, req in engine.requests.items()}
            for out in engine.step():
                if out.finished:
                    done[out.request_id] = out.token_ids
            preempted |= any(req.num_computed_tokens < previous.get(rid, 0)
                             for rid, req in engine.requests.items())
        self.assertFalse(engine.has_work())
        self.assertTrue(preempted)
        self.assertEqual([done[rid] for rid in ids], [self.hf_tokens(p, 8) for p in prompts])
        self.assertEqual(engine.kv.num_free, 16)

    def test_eos_and_ignore_eos(self):
        params = SamplingParams(temperature=0, max_tokens=2, forced_token_ids=[2, 4])
        stopped = self.engine().generate([[4, 5]], params)[0]
        self.assertEqual((stopped.token_ids, stopped.finish_reason), ([2], 'stop'))
        params.ignore_eos = True
        continued = self.engine().generate([[4, 5]], params)[0]
        self.assertEqual((continued.token_ids, continued.finish_reason), ([2, 4], 'length'))

    def test_context_limit_finishes_request(self):
        out = self.engine().generate([[4] * 63], SamplingParams(temperature=0, max_tokens=10))[0]
        self.assertEqual((len(out.token_ids), out.finish_reason), (1, 'length'))

    @torch.inference_mode()
    def test_teacher_forced_logprobs_match_hugging_face(self):
        prompt, forced = [4, 5, 6, 7, 8], [9, 10, 11]
        params = SamplingParams(temperature=0, max_tokens=3, forced_token_ids=forced, return_logprobs=True)
        out = self.engine().generate([prompt], params)[0]
        for index, actual in enumerate(out.logprobs):
            ids = torch.tensor([prompt + forced[:index]])
            expected = self.hf(ids).logits[0, -1].log_softmax(-1)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_abort_releases_running_and_waiting_requests(self):
        engine = self.engine(max_num_seqs=1)
        params = SamplingParams(temperature=0, max_tokens=8)
        engine.add_request([4, 5], params, 'a')
        engine.add_request([6, 7], params, 'b')
        engine.step()
        engine.abort('b')
        engine.abort('a')
        self.assertFalse(engine.has_work())
        self.assertFalse(engine.requests)
        self.assertFalse(engine.detok)
        self.assertEqual(engine.kv.num_free, 48)

    def test_validation_rejects_invalid_requests(self):
        engine = self.engine()
        params = SamplingParams(temperature=0, max_tokens=1)
        for prompt in [[], [4] * 64, [999], [-1]]:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                engine.add_request(prompt, params)
        with self.assertRaises(ValueError):
            engine.add_request([4], SamplingParams(max_tokens=0))
        engine.add_request([4], params, 'duplicate')
        with self.assertRaisesRegex(ValueError, 'already active'):
            engine.add_request([5], params, 'duplicate')
        with self.assertRaises(ValueError):
            engine.generate([[4], [5]], [params])

    def test_undersized_cache_rejected(self):
        with self.assertRaisesRegex(ValueError, 'max-length sequence'):
            self.engine(num_blocks=2)

    def test_text_deltas_reconstruct_final_text(self):
        engine = self.engine()
        prompt = 't4 t5 t6'
        rid = engine.add_request(prompt, SamplingParams(temperature=0, max_tokens=5, ignore_eos=True))
        outputs = []
        while engine.has_work():
            outputs.extend(engine.step())
        self.assertTrue(all(o.request_id == rid for o in outputs))
        self.assertEqual(''.join(o.delta for o in outputs), outputs[-1].text)
        self.assertEqual(outputs[-1].text, self.tokenizer.decode(outputs[-1].token_ids, skip_special_tokens=True))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_generation_matches_hugging_face(self):
        prompt = list(range(4, 25))
        out = self.engine(device='mps').generate([prompt], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))[0]
        self.assertEqual(out.token_ids, self.hf_tokens(prompt, 8))


if __name__ == '__main__':
    unittest.main()
