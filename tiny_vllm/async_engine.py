"""Upstream queue/thread async bridge with shutdown and failure propagation."""
import asyncio
import logging
import queue
import threading
import uuid

from tiny_vllm.engine import Engine
from tiny_vllm.kv_cache import SamplingParams


class AsyncEngine:
    """One worker owns model execution; event loops receive request outputs."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.inbox = queue.Queue()
        self.streams = {}
        self._lock = threading.Lock()
        self._closing = False
        self.failure = None
        self.thread = threading.Thread(target=self._loop, daemon=True, name='tiny-vllm-engine')
        self.thread.start()

    @property
    def healthy(self):
        return not self._closing and self.failure is None and self.thread.is_alive()

    def _send(self, rid, value):
        with self._lock:
            stream = self.streams.get(rid)
        if stream is not None:
            loop, output_queue = stream
            try:
                loop.call_soon_threadsafe(output_queue.put_nowait, value)
            except RuntimeError:  # The client's event loop has already closed.
                self.engine.abort(rid)

    def _fail_all(self, error):
        with self._lock:
            self.failure = error
            ids = list(self.streams)
        for rid in list(self.engine.requests):
            self.engine.abort(rid)
        self.engine.detok.clear()
        for rid in ids:
            self._send(rid, RuntimeError('Generation worker is unavailable'))

    def _loop(self):
        try:
            while True:
                block = not self.engine.has_work()
                while True:
                    try:
                        kind, rid, payload = self.inbox.get(block=block)
                    except queue.Empty:
                        break
                    block = False
                    if kind == 'stop':
                        self._fail_all(RuntimeError('Server shut down'))
                        return
                    if kind == 'add':
                        try:
                            self.engine.add_request(*payload, request_id=rid)
                        except (ValueError, TypeError) as error:
                            self._send(rid, error)
                    else:
                        self.engine.abort(rid)
                for output in self.engine.step():
                    self._send(output.request_id, output)
        except Exception as error:
            logging.getLogger(__name__).exception('Generation worker failed')
            self._fail_all(error)

    async def generate(self, prompt, params: SamplingParams):
        rid = uuid.uuid4().hex
        output_queue = asyncio.Queue()
        with self._lock:
            if not self.healthy:
                raise RuntimeError('Generation worker is unavailable')
            self.streams[rid] = (asyncio.get_running_loop(), output_queue)
            self.inbox.put(('add', rid, (prompt, params)))
        finished = False
        try:
            while not finished:
                output = await output_queue.get()
                if isinstance(output, Exception):
                    raise output
                finished = output.finished
                yield output
        finally:
            with self._lock:
                self.streams.pop(rid, None)
            if not finished:
                self.inbox.put(('abort', rid, None))

    def close(self):
        with self._lock:
            if not self._closing:
                self._closing = True
                self.inbox.put(('stop', '', None))
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Generation worker did not stop within 10 seconds')
