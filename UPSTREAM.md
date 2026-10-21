# Source notes

Tiny-vLLM's core engine (KV cache manager, scheduler, reference attention,
model definitions, sampling, checkpoint loading, async server) is adapted
from [priyammaz/tinyserve](https://github.com/priyammaz/tinyserve/blob/main/serve.py),
used with the author's permission. The CPU/MPS port, the optional MLX
backend, benchmarking, and the test suite are local additions.
