"""Run a loopback Uvicorn server for socket tests and HTTP measurements."""
import socket
import threading
import time
from contextlib import contextmanager

import uvicorn


@contextmanager
def live_server(app):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    address = f'http://127.0.0.1:{listener.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', log_level='error', access_log=False))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    try:
        for _ in range(500):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError('HTTP server failed to start')
            time.sleep(0.01)
        else:
            raise RuntimeError('HTTP server startup timed out')
        yield address
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        if thread.is_alive():
            raise RuntimeError('HTTP server did not stop')
