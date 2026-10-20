import asyncio
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.generation_inputs import generation_fixture
from tiny_vllm.async_engine import AsyncEngine
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.kv_cache import SamplingParams


class AsyncEngineTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        generation_fixture(cls.folder.name)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    async def asyncSetUp(self):
        engine = Engine(EngineConfig(model=self.folder.name, max_model_len=64, block_size=4,
                                     num_blocks=49, max_num_batched_tokens=8))
        self.worker = AsyncEngine(engine)

    async def asyncTearDown(self):
        await asyncio.to_thread(self.worker.close)

    async def collect(self, prompt, count=8):
        params = SamplingParams(temperature=0, max_tokens=count, ignore_eos=True)
        return [output async for output in self.worker.generate(prompt, params)]

    async def idle(self):
        for _ in range(200):
            if not self.worker.engine.has_work() and not self.worker.streams:
                return
            await asyncio.sleep(0.005)
        self.fail('Worker did not become idle')

    async def test_concurrent_requests_finish_and_release_pages(self):
        results = await asyncio.wait_for(asyncio.gather(
            self.collect([4, 5, 6]), self.collect(list(range(4, 25))), self.collect([7, 8])), 5)
        self.assertTrue(all(len(outputs[-1].token_ids) == 8 for outputs in results))
        self.assertEqual(len({outputs[-1].request_id for outputs in results}), 3)
        await self.idle()
        self.assertEqual(self.worker.engine.kv.num_free, 48)

    async def test_invalid_admission_does_not_kill_worker(self):
        with self.assertRaises(ValueError):
            await asyncio.wait_for(self.collect([]), 2)
        result = await asyncio.wait_for(self.collect([4, 5]), 2)
        self.assertTrue(result[-1].finished)
        self.assertTrue(self.worker.healthy)

    async def test_closing_generator_aborts_request(self):
        stream = self.worker.generate([4, 5], SamplingParams(temperature=0, max_tokens=60, ignore_eos=True))
        await asyncio.wait_for(anext(stream), 2)
        await stream.aclose()
        await self.idle()
        self.assertEqual(self.worker.engine.kv.num_free, 48)

    async def test_shutdown_stops_thread_and_rejects_new_requests(self):
        await asyncio.to_thread(self.worker.close)
        self.assertFalse(self.worker.thread.is_alive())
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(self.collect([4]), 2)

    async def test_failure_is_delivered_instead_of_hanging(self):
        with patch.object(self.worker.engine, 'step', side_effect=RuntimeError('test failure')):
            with self.assertLogs('tiny_vllm.async_engine', level='ERROR'):
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(self.collect([4, 5]), 2)
        await asyncio.to_thread(self.worker.thread.join, 2)
        self.assertFalse(self.worker.healthy)
        self.assertFalse(self.worker.engine.requests)
        self.assertEqual(self.worker.engine.kv.num_free, 48)


if __name__ == '__main__':
    unittest.main()
