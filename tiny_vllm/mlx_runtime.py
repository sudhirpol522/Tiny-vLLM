"""Configure numerical precision before initializing the optional MLX runtime."""
import os

# MLX can use reduced-precision matrix units with float32 inputs on the M5.
# Full float32 is our correctness baseline; explicit user overrides are retained.
os.environ.setdefault('MLX_ENABLE_TF32', '0')

import mlx.core as mx
