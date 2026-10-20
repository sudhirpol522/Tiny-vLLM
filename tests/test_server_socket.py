import json
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
import torch

from benchmarks.generation_inputs import generation_fixture
from benchmarks.http_inputs import live_server
from tiny_vllm.async_engine import AsyncEngine
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.server import build_app


class SocketStreamingTests(unittest.TestCase):
    def check_stream_and_disconnect(self, device):
        with tempfile.TemporaryDirectory() as folder:
            generation_fixture(folder)
            engine = Engine(EngineConfig(model=folder, device=device, max_model_len=64,
                                         num_blocks=17, max_num_batched_tokens=8))
            original = engine.step
            calls = []

            def slow_step():
                time.sleep(0.01)
                calls.append(1)
                return original()

            with patch.object(engine, 'step', side_effect=slow_step):
                worker = AsyncEngine(engine)
                try:
                    with live_server(build_app(worker, 'test-model')) as url:
                        with httpx.Client(base_url=url, timeout=5) as client:
                            with client.stream('POST', '/v1/completions', json={
                                    'prompt': [4, 5, 6], 'temperature': 0,
                                    'max_tokens': 32, 'ignore_eos': True, 'stream': True}) as response:
                                self.assertEqual(response.status_code, 200)
                                first = next(line for line in response.iter_lines() if line.startswith('data: '))
                                chunk = json.loads(first[6:])
                                self.assertIsNone(chunk['choices'][0]['finish_reason'])
                                self.assertTrue(engine.has_work(), 'Response was buffered until completion')
                            for _ in range(200):
                                if not engine.has_work() and not worker.streams:
                                    break
                                time.sleep(0.005)
                            self.assertFalse(engine.has_work())
                            self.assertFalse(worker.streams)
                            self.assertLess(len(calls), 32, 'Disconnected request ran to completion')
                            self.assertEqual(engine.kv.num_free, 16)
                            self.assertEqual(client.get('/health').status_code, 200)
                finally:
                    worker.close()

    def test_real_socket_streams_early_and_aborts_disconnect(self):
        self.check_stream_and_disconnect('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_socket_stream_and_disconnect(self):
        self.check_stream_and_disconnect('mps')


if __name__ == '__main__':
    unittest.main()
