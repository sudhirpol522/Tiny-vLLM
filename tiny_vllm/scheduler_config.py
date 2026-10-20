"""CPU-only configuration adapter for the three fields the scheduler reads."""
from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    block_size: int = 16
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 512
