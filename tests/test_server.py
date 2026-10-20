import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from benchmarks.generation_inputs import generation_fixture
from tiny_vllm.async_engine import AsyncEngine
from tiny_vllm.config import EngineConfig
from tiny_vllm.engine import Engine
from tiny_vllm.server import build_app


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.hf, cls.tokenizer = generation_fixture(cls.folder.name)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        self.engine = Engine(EngineConfig(model=self.folder.name, max_model_len=64,
                                         block_size=4, num_blocks=49, max_num_batched_tokens=8))
        self.worker = AsyncEngine(self.engine)
        self.client = TestClient(build_app(self.worker, 'test-model'))
        self.client.__enter__()
        self.body = dict(model='test-model', prompt='t4 t5 t6', temperature=0,
                         max_tokens=8, ignore_eos=True)

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.assertFalse(self.worker.thread.is_alive())

    def events(self, response):
        return [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]

    def test_health_models_and_metrics(self):
        self.assertEqual(self.client.get('/health').json(), {'status': 'ok'})
        self.assertEqual(self.client.get('/v1/models').json()['data'][0]['id'], 'test-model')
        metrics = self.client.get('/metrics').json()
        self.assertEqual((metrics['running'], metrics['waiting'], metrics['free_kv_blocks']), (0, 0, 48))

    @torch.inference_mode()
    def test_completion_matches_hugging_face(self):
        response = self.client.post('/v1/completions', json=self.body)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        generated = self.hf.generate(torch.tensor([[4, 5, 6]]), max_new_tokens=8,
                                     eos_token_id=None, pad_token_id=0)[0, 3:].tolist()
        self.assertEqual(result['choices'][0]['text'], self.tokenizer.decode(generated, skip_special_tokens=True))
        self.assertEqual(result['choices'][0]['finish_reason'], 'length')
        self.assertEqual(result['usage']['completion_tokens'], 8)
        self.assertEqual(result['usage']['total_tokens'], 11)

    def test_stream_matches_nonstream_and_terminates(self):
        expected = self.client.post('/v1/completions', json=self.body).json()
        response = self.client.post('/v1/completions', json={**self.body, 'stream': True,
                                    'stream_options': {'include_usage': True}})
        self.assertTrue(response.headers['content-type'].startswith('text/event-stream'))
        events = self.events(response)
        self.assertEqual(events[-1], '[DONE]')
        chunks = [json.loads(event) for event in events[:-1]]
        text = ''.join(chunk['choices'][0]['text'] for chunk in chunks if chunk['choices'])
        self.assertEqual(text, expected['choices'][0]['text'])
        self.assertEqual(chunks[-2]['choices'][0]['finish_reason'], 'length')
        self.assertEqual(chunks[-1]['choices'], [])
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 8)
        self.assertEqual(len({chunk['id'] for chunk in chunks}), 1)

    def test_token_id_prompt(self):
        text = self.client.post('/v1/completions', json=self.body).json()
        ids = self.client.post('/v1/completions', json={**self.body, 'prompt': [4, 5, 6]}).json()
        self.assertEqual(ids['choices'][0]['text'], text['choices'][0]['text'])

    def test_chat_stream_has_role_content_and_done(self):
        body = {k: v for k, v in self.body.items() if k != 'prompt'}
        body['messages'] = [{'role': 'user', 'content': [{'type': 'text', 'text': 't4 t5 t6'}]}]
        plain = self.client.post('/v1/chat/completions', json=body)
        self.assertEqual(plain.status_code, 200)
        response = self.client.post('/v1/chat/completions', json={**body, 'stream': True})
        events = self.events(response)
        chunks = [json.loads(event) for event in events[:-1]]
        self.assertEqual(chunks[0]['choices'][0]['delta']['role'], 'assistant')
        self.assertTrue(all(c['object'] == 'chat.completion.chunk' for c in chunks))
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)
        self.assertEqual(text, plain.json()['choices'][0]['message']['content'])
        self.assertEqual(events[-1], '[DONE]')

    def test_invalid_requests_return_errors_and_worker_survives(self):
        cases = [{'prompt': []}, {'prompt': [999]}, {'prompt': [True]}, {'prompt': 't4 ' * 64},
                 {'max_tokens': 0}, {'max_tokens': '3'}, {'temperature': -1}, {'top_p': 0},
                 {'stream': 'yes'}, {'n': 2}, {'unsupported': True}, {'prompt': [['nested']]}]
        for change in cases:
            with self.subTest(change=change):
                response = self.client.post('/v1/completions', json={**self.body, **change})
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.json())
        self.assertEqual(self.client.post('/v1/completions', json={}).status_code, 400)
        self.assertEqual(self.client.post('/v1/completions', content='{bad',
                                         headers={'content-type': 'application/json'}).status_code, 400)
        self.assertEqual(self.client.post('/v1/completions', json=self.body).status_code, 200)

    def test_unknown_model_returns_404(self):
        response = self.client.post('/v1/completions', json={**self.body, 'model': 'other'})
        self.assertEqual(response.status_code, 404)

    def test_context_limit_and_completion_token_alias(self):
        response = self.client.post('/v1/completions', json={**self.body, 'prompt': [4] * 63,
                                    'max_completion_tokens': 20})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['usage']['completion_tokens'], 1)

    def test_prefix_cache_usage_reported(self):
        body = {**self.body, 'prompt': list(range(4, 25))}
        self.client.post('/v1/completions', json=body)
        result = self.client.post('/v1/completions', json=body).json()
        self.assertGreaterEqual(result['usage']['prompt_tokens_details']['cached_tokens'], 20)

    def test_concurrent_http_requests_finish(self):
        def submit(index):
            return self.client.post('/v1/completions', json={**self.body, 'prompt': [4 + index, 9]}).json()
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(submit, range(4)))
        self.assertEqual(len({r['id'] for r in responses}), 4)
        self.assertTrue(all(r['usage']['completion_tokens'] == 8 for r in responses))

    def test_unavailable_worker_returns_503(self):
        self.worker.close()
        self.assertEqual(self.client.get('/health').status_code, 503)
        self.assertEqual(self.client.post('/v1/completions', json=self.body).status_code, 503)

    def test_worker_failure_returns_error_and_marks_health_unavailable(self):
        with patch.object(self.engine, 'step', side_effect=RuntimeError('test failure')):
            with self.assertLogs('tiny_vllm.async_engine', level='ERROR'):
                response = self.client.post('/v1/completions', json=self.body)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get('/health').status_code, 503)

    def test_stream_failure_sends_error_and_done(self):
        with patch.object(self.engine, 'step', side_effect=RuntimeError('test failure')):
            with self.assertLogs('tiny_vllm.async_engine', level='ERROR'):
                response = self.client.post('/v1/completions', json={**self.body, 'stream': True})
        events = self.events(response)
        self.assertIn('error', json.loads(events[0]))
        self.assertEqual(events[-1], '[DONE]')

    def test_unsupported_chat_content_returns_error(self):
        body = {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {}}]}]}
        self.assertEqual(self.client.post('/v1/chat/completions', json=body).status_code, 400)


if __name__ == '__main__':
    unittest.main()
