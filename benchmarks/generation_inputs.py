"""A local untrained checkpoint and token-ID vocabulary for generation tests."""
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GenerationConfig, PreTrainedTokenizerFast

from benchmarks.checkpoint_inputs import checkpoint_fixture


def generation_fixture(folder):
    hf = checkpoint_fixture(folder)
    hf.config.eos_token_id = 2
    hf.config.pad_token_id = 0
    hf.config.bos_token_id = 3
    hf.config.save_pretrained(folder)
    hf.generation_config = GenerationConfig.from_model_config(hf.config)
    hf.generation_config.save_pretrained(folder)
    backend = Tokenizer(WordLevel({f't{i}': i for i in range(97)}, unk_token='t1'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='t1',
                                       pad_token='t0', eos_token='t2', bos_token='t3')
    tokenizer.save_pretrained(folder)
    return hf, tokenizer
