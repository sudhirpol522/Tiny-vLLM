# Benchmarks

Measurements were collected on an Apple M5 running macOS 26.6.2. The tensor
benchmarks use Python 3.12.13, PyTorch 2.14.1, float32, and one CPU thread. The
cache and scheduler measurements use Python 3.14.6. MPS samples synchronize
before and after the timed calls. Case order and raw samples are recorded in JSON.

| Component | Method | Results |
| --- | --- | --- |
| KV cache | 1,000 sequential 1,024-token requests; 16-token pages; 0/256/768 shared tokens; 9 runs | [JSON](results/kv_cache.json) |
| Scheduler | 128 simulated requests; 64/256/512/1,024-token prompts; 32 synthetic outputs; 9 runs | [JSON](results/scheduler.json) |
| Attention | 8 query heads, 2 KV heads, head dimension 64; 3 warmups; 7 samples of 5 calls | [JSON](results/attention.json) |
| Model | 3.54M random parameters, 4 layers, hidden size 256; 3 warmups; 7 samples of 5 calls | [JSON](results/model.json) |
| Loading | 86,528-parameter local Hugging Face Qwen3 fixture; 1 warmup; 7 loads | [JSON](results/loading.json) |
| Sampling | 32,768-logit vocabulary, batches of 1/8/32; 3 warmups; 7 samples of 5 calls | [JSON](results/sampling.json) |
| Generation | 86,528 random parameters; vocabulary 97; 16 output tokens/request; 1 warmup, 7 runs | [JSON](results/generation.json) |

The attention control uses already-contiguous K/V with the same SDPA operation;
it is not a serving baseline. Model calls include cache writes and logits for all
query positions. They replay fixed contexts after priming, without sampling or
scheduling. Loading includes config parsing, CPU construction, weight copies,
and device transfer; it excludes imports and downloads and uses a warm filesystem
cache. Filtered sampling uses temperature 0.7, top-k 50, and top-p 0.9, with every
fourth row greedy. Sampling time includes returning Python token IDs.

No pretrained checkpoint was downloaded for these measurements. Local checkpoint
tests compare logits with Hugging Face using the same random weights. Component
timings exclude generation. The generation benchmark measures offline decoding
on this tiny fixture, not pretrained-model performance or network serving.

Generation compares one 32-token prompt, eight mixed-length prompts (16–72
tokens), and eight 32-token prompts sharing their first 24 tokens. Both methods
are greedy with EOS stopping disabled. HF uses eager attention and one padded
static batch; Tiny-vLLM uses an eight-sequence limit and a 64-token step budget.
Prefix metadata is reset before every measured engine run. Engine loading and
cache allocation are excluded; input packing, sampling, and returned token IDs
are included. Tiny-vLLM also incrementally decodes text. Method order alternates
across seven runs; every measured output is checked for token identity.

## Reproduce

From the repository root, after installing requirements:

```bash
python -m benchmarks.kv_cache
python -m benchmarks.scheduler
python -m benchmarks.attention --devices cpu mps
python -m benchmarks.model --devices cpu mps
python -m benchmarks.loading_sampling --devices cpu mps
python -m benchmarks.generation --devices cpu mps
python -m benchmarks.charts
```

Use `--devices cpu` without Apple GPU access. The chart command redraws saved
measurements with Matplotlib; it does not rerun benchmarks. New benchmark runs
overwrite their JSON and PNG files. Refresh the root README table when publishing
new measurements.

## Known limits

The copied scheduler uses FIFO admission. The engine rejects pools that cannot
hold a max-length request and rejects oversized prompts. The scheduler itself
processes running requests in admission order and
optimistically advances token counts. The model supports plain RoPE and uses
global attention metadata, so forward calls must be serialized. Reference
attention gathers dense prefixes and materializes masks instead of using a
custom paged kernel. Checkpoint parity currently covers small Qwen3 fixtures,
including shards and tied embeddings.
