"""Local random-weight Hugging Face checkpoint fixture; no downloads."""
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM


def checkpoint_fixture(path, sharded=False, tied=False):
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, tie_word_embeddings=tied)
    cfg._attn_implementation = 'eager'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        hf = Qwen3ForCausalLM(cfg).eval()
    cfg.save_pretrained(path)
    state = {k: v.detach().clone() for k, v in hf.state_dict().items()}
    if tied:
        state.pop('lm_head.weight')
    if sharded:
        keys = sorted(state)
        for index, subset in enumerate((keys[::2], keys[1::2])):
            save_file({k: state[k] for k in subset}, str(Path(path) / f'model-{index}.safetensors'))
    else:
        save_file(state, str(Path(path) / 'model.safetensors'))
    return hf
