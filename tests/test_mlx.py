import asyncio
import importlib.util
import platform
import tempfile
import unittest

import numpy as np
import torch

from benchmarks.generation_inputs import generation_fixture
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.kv_cache import SamplingParams


AVAILABLE = platform.system() == 'Darwin' and importlib.util.find_spec('mlx') is not None


@unittest.skipUnless(AVAILABLE, 'Optional MLX backend requires Apple Silicon and requirements-mlx.txt')
class MLXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tiny_vllm.mlx_runtime import mx
        if not mx.metal.is_available():
            raise unittest.SkipTest('MLX Metal unavailable')
        cls.mx = mx
        cls.folder = tempfile.TemporaryDirectory()
        cls.hf, _ = generation_fixture(cls.folder.name)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def engine(self, backend='mlx', compiled=False, **kwargs):
        options = dict(model=self.folder.name, backend=backend,
                       device='mps' if backend == 'mlx' else 'cpu', compile_decode=compiled,
                       max_model_len=64, block_size=4, max_num_batched_tokens=8,
                       num_blocks=49, max_num_seqs=3)
        options.update(kwargs)
        return Engine(EngineConfig(**options))

    @torch.inference_mode()
    def test_prefill_logits_match_pytorch(self):
        torch_engine = self.engine('torch', max_num_batched_tokens=64)
        mlx_engine = self.engine(max_num_batched_tokens=64)
        for engine in [torch_engine, mlx_engine]:
            engine.add_request(list(range(4, 25)), SamplingParams(temperature=0, max_tokens=1))
        _, expected = torch_engine.runner.run(torch_engine.scheduler.schedule())
        actual = mlx_engine.runner.run_logits(mlx_engine.scheduler.schedule())
        np.testing.assert_allclose(np.array(actual), expected.numpy(), atol=3e-5, rtol=3e-5)

    def test_eager_and_compiled_generation_match_pytorch(self):
        prompts = [[4, 5, 6], list(range(4, 25)), list(range(4, 39))]
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        expected = [out.token_ids for out in self.engine('torch').generate(prompts, params)]
        for compiled in [False, True]:
            with self.subTest(compiled=compiled):
                outputs = self.engine(compiled=compiled).generate(prompts, params)
                self.assertEqual([out.token_ids for out in outputs], expected)

    @torch.inference_mode()
    def test_teacher_forced_logprobs_match_hugging_face(self):
        prompt, forced = [4, 5, 6, 7, 8], [9, 10, 11, 12]
        params = SamplingParams(temperature=0, max_tokens=4, forced_token_ids=forced, return_logprobs=True)
        for compiled in [False, True]:
            out = self.engine(compiled=compiled).generate([prompt], params)[0]
            for i, actual in enumerate(out.logprobs):
                expected = self.hf(torch.tensor([prompt + forced[:i]])).logits[0, -1].log_softmax(-1)
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_decode_reuses_variant_with_dynamic_positions(self):
        engine = self.engine(compiled=True)
        engine.generate([list(range(4, 21))], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
        self.assertEqual(engine.runner.model.compiled_variants, 1)

    def test_context_bucket_boundary_creates_second_variant(self):
        engine = self.engine(compiled=True)
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        output = engine.generate([list(range(4, 19))], params)[0]
        self.assertEqual(engine.runner.model.compiled_variants, 2)
        expected = self.engine('torch').generate([list(range(4, 19))], params)[0]
        self.assertEqual(output.token_ids, expected.token_ids)

    def test_prefix_reuse_and_release(self):
        engine = self.engine(compiled=True)
        params = SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)
        prompt = list(range(4, 25))
        first = engine.generate([prompt], params)[0]
        second = engine.generate([prompt], params)[0]
        self.assertEqual(first.token_ids, second.token_ids)
        self.assertGreaterEqual(second.num_cached_tokens, 20)
        self.assertEqual(engine.kv.num_free, 48)

    def test_cache_disabled_matches_enabled(self):
        params = SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)
        prompts = [list(range(4, 25)), list(range(4, 25)) + [40]]
        on = self.engine(compiled=True).generate(prompts, params)
        off = self.engine(compiled=True, enable_prefix_caching=False).generate(prompts, params)
        self.assertEqual([out.token_ids for out in on], [out.token_ids for out in off])

    def test_preemption_matches_pytorch(self):
        prompts = [list(range(4, 28)), list(range(30, 50)), list(range(4, 40))]
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        expected = self.engine('torch', num_blocks=17).generate(prompts, params)
        engine = self.engine(compiled=True, num_blocks=17)
        actual = engine.generate(prompts, params)
        self.assertEqual([out.token_ids for out in actual], [out.token_ids for out in expected])
        self.assertEqual(engine.kv.num_free, 16)

    def test_eos_and_context_limit(self):
        engine = self.engine(compiled=True)
        out = engine.generate([[4, 5]], SamplingParams(max_tokens=2, forced_token_ids=[2, 4]))[0]
        self.assertEqual((out.token_ids, out.finish_reason), ([2], 'stop'))
        out = engine.generate([[4] * 63], SamplingParams(temperature=0, max_tokens=10, ignore_eos=True))[0]
        self.assertEqual((len(out.token_ids), out.finish_reason), (1, 'length'))

    def test_filtered_sampling_respects_top_k(self):
        from tiny_vllm.mlx_runner import sample_mlx
        logits = self.mx.array(np.tile([0., 3., 2., -100.], (100, 1)), dtype=self.mx.float32)
        selected = sample_mlx(logits, [SamplingParams(top_k=2)] * 100)
        self.assertLessEqual(set(selected), {1, 2})
        mixed = sample_mlx(self.mx.array([[1., 5., 2.], [5., 1., 2.]]),
                           [SamplingParams(temperature=0), SamplingParams(top_k=1)])
        self.assertEqual(mixed, [1, 0])

    def test_async_worker_uses_compiled_backend(self):
        from tiny_vllm.async_engine import AsyncEngine
        engine = self.engine(compiled=True)
        worker = AsyncEngine(engine)

        async def collect():
            return [out async for out in worker.generate([4, 5, 6], SamplingParams(temperature=0, max_tokens=4))]

        try:
            result = asyncio.run(asyncio.wait_for(collect(), 5))
            self.assertTrue(result[-1].finished)
        finally:
            worker.close()

    def test_compiled_http_stream(self):
        import json
        from fastapi.testclient import TestClient
        from tiny_vllm.async_engine import AsyncEngine
        from tiny_vllm.server import build_app
        worker = AsyncEngine(self.engine(compiled=True))
        with TestClient(build_app(worker, 'mlx-test')) as client:
            response = client.post('/v1/completions', json=dict(prompt=[4, 5, 6], temperature=0,
                                   max_tokens=4, ignore_eos=True, stream=True,
                                   stream_options={'include_usage': True}))
            events = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
            self.assertEqual(events[-1], '[DONE]')
            self.assertEqual(json.loads(events[-2])['usage']['completion_tokens'], 4)
        self.assertFalse(worker.thread.is_alive())


class BackendConfigTests(unittest.TestCase):
    def test_invalid_backend_options_rejected(self):
        with self.assertRaises(ValueError):
            EngineConfig('unused', backend='mlx', device='cpu')
        with self.assertRaises(ValueError):
            EngineConfig('unused', backend='torch', compile_decode=True)


if __name__ == '__main__':
    unittest.main()
