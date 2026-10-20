"""Upstream checkpoint loader adapted for CPU/MPS; see UPSTREAM.md."""
import glob
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

from tiny_vllm.model import Model


def load_model(name: str, dtype=torch.float32, max_position: int = 512, device='cpu'):
    """Load local or Hub safetensors; reject incomplete or mismatched weights."""
    device = torch.device(device)
    if device.type not in {'cpu', 'mps'}:
        raise ValueError('This loader currently supports CPU and MPS')
    if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError('Use float32, float16, or bfloat16 weights')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this process')
    path = name if os.path.isdir(name) else snapshot_download(
        name, allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt'])
    cfg = AutoConfig.from_pretrained(path)
    shards = sorted(glob.glob(os.path.join(path, '*.safetensors')))
    if not shards:
        raise FileNotFoundError(f'No safetensors checkpoints in {path}')

    # The original model builds float64 RoPE tables, which MPS cannot construct.
    # Build and load on CPU, then transfer the completed model to the target.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device('cpu'):
            model = Model(cfg, max_position)
    finally:
        torch.set_default_dtype(previous_dtype)
    params = dict(model.named_parameters())
    loaded = set()
    with torch.no_grad():
        for filename in shards:
            with safe_open(filename, 'pt', device='cpu') as checkpoint:
                for key in checkpoint.keys():
                    name_ = key.removeprefix('model.')
                    if name_ in params:
                        if name_ in loaded:
                            raise ValueError(f'Duplicate checkpoint parameter: {name_}')
                        tensor = checkpoint.get_tensor(key)
                        if tensor.shape != params[name_].shape:
                            raise ValueError(f'Checkpoint shape mismatch for {name_}: '
                                             f'{tuple(tensor.shape)} != {tuple(params[name_].shape)}')
                        params[name_].copy_(tensor)
                        loaded.add(name_)
    missing = sorted(set(params) - loaded)
    if missing:
        raise ValueError('Missing checkpoint parameters: ' + ', '.join(missing))
    return model.to(device).eval()
