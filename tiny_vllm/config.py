"""Bounded CPU/MPS configuration for the adapted upstream runner."""
from dataclasses import dataclass

import torch


@dataclass
class EngineConfig:
    model: str
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32
    max_model_len: int = 512
    block_size: int = 16
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 512
    num_blocks: int = 129  # Includes reserved null page 0.
    enable_prefix_caching: bool = True
    seed: int = 0
    backend: str = 'torch'
    compile_decode: bool = False

    def __post_init__(self):
        if self.backend not in {'torch', 'mlx'}:
            raise ValueError('Choose torch or mlx backend')
        if self.backend == 'mlx' and self.device != 'mps':
            raise ValueError('Use device=mps for the MLX Apple GPU backend')
        if self.compile_decode and self.backend != 'mlx':
            raise ValueError('compile_decode requires the MLX backend')
        if self.device not in {'cpu', 'mps'}:
            raise ValueError('Choose cpu or mps')
        for name in ('max_model_len', 'block_size', 'max_num_seqs',
                     'max_num_batched_tokens', 'num_blocks'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_model_len < 2:
            raise ValueError('max_model_len must allow a prompt and an output token')
        if (self.num_blocks - 1) * self.block_size < self.max_model_len:
            raise ValueError('KV pool must hold at least one max-length sequence')
