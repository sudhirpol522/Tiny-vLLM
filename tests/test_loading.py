import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from benchmarks.model_inputs import attach_cache, forward, make_inputs
from tiny_vllm.loading import load_model


from benchmarks.checkpoint_inputs import checkpoint_fixture


class LoadingTests(unittest.TestCase):
    @torch.inference_mode()
    def parity(self, sharded=False, tied=False, device='cpu'):
        with tempfile.TemporaryDirectory() as folder:
            hf = checkpoint_fixture(folder, sharded, tied)
            model = load_model(folder, max_position=64, device=device)
            pages = attach_cache(model, [19])
            inputs = make_inputs([19], [0], pages, device=device, vocab=97)
            actual = forward(model, inputs).cpu()
            expected = hf(inputs[0].cpu()[None]).logits[0]
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
            self.assertFalse(model.training)

    def test_single_file_logits_match_hugging_face(self):
        self.parity()

    def test_sharded_checkpoint_matches_hugging_face(self):
        self.parity(sharded=True)

    def test_tied_embeddings_match_hugging_face(self):
        self.parity(tied=True)

    def test_missing_parameter_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint_fixture(folder)
            file = Path(folder) / 'model.safetensors'
            state = load_file(str(file))
            state.pop('model.layers.0.self_attn.q_proj.weight')
            save_file(state, str(file))
            with self.assertRaisesRegex(ValueError, 'Missing checkpoint parameters'):
                load_model(folder, max_position=64)

    def test_shape_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint_fixture(folder)
            file = Path(folder) / 'model.safetensors'
            state = load_file(str(file))
            state['model.embed_tokens.weight'] = torch.zeros(1, 1)
            save_file(state, str(file))
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                load_model(folder, max_position=64)

    def test_duplicate_parameter_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            hf = checkpoint_fixture(folder)
            save_file({'model.embed_tokens.weight': hf.model.embed_tokens.weight.detach().clone()},
                      str(Path(folder) / 'duplicate.safetensors'))
            with self.assertRaisesRegex(ValueError, 'Duplicate checkpoint parameter'):
                load_model(folder, max_position=64)

    def test_no_weights_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint_fixture(folder)
            (Path(folder) / 'model.safetensors').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'No safetensors'):
                load_model(folder)

    def test_default_dtype_restored_after_model_build_error(self):
        with tempfile.TemporaryDirectory() as folder:
            hf = checkpoint_fixture(folder)
            hf.config.rope_parameters = {'rope_type': 'linear', 'rope_theta': 10000.0, 'factor': 2.0}
            hf.config.save_pretrained(folder)
            before = torch.get_default_dtype()
            with self.assertRaisesRegex(AssertionError, 'plain RoPE'):
                load_model(folder, dtype=torch.float16)
            self.assertEqual(torch.get_default_dtype(), before)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_checkpoint_logits_match_hugging_face(self):
        self.parity(device='mps')


if __name__ == '__main__':
    unittest.main()
