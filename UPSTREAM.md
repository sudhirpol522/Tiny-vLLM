# Source notes

The first Tiny-vLLM component is extracted from
[priyammaz/tinyserve](https://github.com/priyammaz/tinyserve/blob/main/serve.py).
Retrieved on October 20, 2026. The source `serve.py` Git blob is
`e681cb61d9c5701c975ae8daf116d1e4e32e9995`.

`tiny_vllm/kv_cache.py` contains the original `SamplingParams`, `Request`, and
`KVCacheManager` definitions, including their comments. The only additions are a
module docstring, postponed annotation evaluation, and the two standard-library
imports needed to use this component without importing the full serving stack.
`examples/kv_demo.py` contains the original `kv_cache_demo` function with imports
and a module entry point added. The project/package name is Tiny-vLLM; upstream
class names and behavior are preserved.

The benchmark, checks, chart renderer, and README are local additions. This is an
incremental adaptation of tinyserve, not an independently authored inference engine.

The second component, `tiny_vllm/scheduler.py`, contains the original
`ScheduledRequest` and `Scheduler` definitions verbatim from the same source blob.
Only the module header and imports are added. `scheduler_config.py` is a local
three-field adapter (`block_size`, `max_num_seqs`, `max_num_batched_tokens`),
replacing the full engine configuration at this boundary so this stage does not
require PyTorch or CUDA. It does not change scheduler decisions. The scheduler
demo and CPU simulation benchmark are local additions, not upstream model runs.

The third component, `tiny_vllm/attention.py`, contains the original
`AttnMetadata` and `torch_ref_attention` definitions verbatim from the same source
blob. Only a module header and imports are added. There are no changes to the
attention algorithm for CPU or MPS. The CUDA/FlashAttention wrapper is not
included in this component.

The synthetic page-packing helper, float64 dense oracle, pre-gathered SDPA control,
attention demo, tests, benchmark, and chart renderer are local validation tools.
The control follows the reference's mask and head-expansion operations but reads
already-contiguous K/V tensors. It is not copied from another serving engine.
The demo replaces the upstream CUDA-only FlashAttention comparison with an
independent numerical check that can run on a Mac.

The fourth component, `tiny_vllm/model.py`, contains the original `RMSNorm`,
`rotary`, `Attention`, `MLP`, `DecoderLayer`, and `Model` definitions verbatim
from the same source blob. Added imports connect those definitions to the
extracted attention module. `ATTN_META` remains module-global as upstream;
`ATTN_FN` defaults to `torch_ref_attention` instead of the CUDA/FlashAttention
function. The CUDA-only checkpoint loader is not included at this stage.

Local model fixtures construct a small Qwen3-shaped configuration with random
weights, attach per-layer page pools, and supply metadata during a forward call.
They build the original float64 RoPE tables on CPU before moving the model to
MPS. A context manager restores global metadata after each call. These fixtures,
the independent CPU float64 dense-model oracle, model tests, demo, benchmark,
and chart are validation tools; they do not replace the copied model definitions.

The fifth component, `tiny_vllm/sampling.py`, preserves the original `sample`
function verbatim. `tiny_vllm/loading.py` adapts the original `load_model` helper:
it keeps the Hub/local-directory selection, config loading, safetensors shard
loop, `model.` prefix removal, and parameter copies. CPU construction and a final
device transfer replace CUDA-only construction/loading. It also restores the
previous default dtype and rejects missing, duplicate, or shape-mismatched
parameters rather than leaving random weights in an incomplete model.

Checkpoint tests use a local, randomly initialized Hugging Face Qwen3 model,
including sharded weights and tied embeddings, and compare forward logits.
No pretrained model was downloaded for these results. Loading and sampling
benchmarks and the checkpoint demo are local additions. All benchmark charts
are now generated with Matplotlib from the saved JSON measurements.

The sixth component adds `tiny_vllm/runner.py` and `tiny_vllm/engine.py` from the
same upstream source blob. The runner's `_prepare` method, `Output`, and
`Detokenizer` definitions are preserved verbatim. `ModelRunner` construction,
cache allocation, and eager `run` are adapted to CPU/MPS: explicit block counts
replace CUDA memory profiling, CUDA graph paths are omitted, and attention
metadata is assigned in the extracted model module and restored after execution.

`Engine` retains upstream admission, stepping, sampling, incremental decoding,
completion, abort, and offline batch flow. CUDA graph initialization is omitted;
input validation additionally rejects invalid token IDs, duplicate active IDs,
nonpositive output limits, incomplete forced-token sequences, and parameter-list
length mismatches. `EngineConfig` supplies the Mac runner's settings and rejects
cache pools too small for a max-length request.

The local generation fixture adds a synthetic tokenizer to the random Qwen3
checkpoint. Tests compare greedy token sequences and teacher-forced logprobs
against Hugging Face, including chunking, prefix reuse, and preemption. The
offline generation benchmark compares the same random checkpoint using our
reference backend and Hugging Face eager attention. It is not a pretrained-model
or production-serving benchmark.

The seventh component adapts upstream `AsyncEngine` and `build_app` into
`tiny_vllm/async_engine.py` and `tiny_vllm/server.py`. It retains the background
thread/inbox/output-queue architecture, model sampling defaults, completion/chat
response shapes, usage accounting, chat templates, and SSE framing. These
definitions are adapted rather than verbatim copies: lifecycle shutdown,
thread-safe stream registration, failure propagation, structured validation,
disconnect cleanup, optional streaming usage, and initial chat role events are
added. Model ownership labels use Tiny-vLLM.

`tiny_vllm/serve.py` is a local CLI around Uvicorn with one engine worker and a
loopback default. Chat remains text-only; unsupported request fields return an
error. `/metrics` exposes JSON counts, not Prometheus exposition. The synthetic
fixture now includes a minimal test-only chat template. Socket tests verify
early streaming and abort/release on client disconnect on CPU and MPS. The
loopback HTTP benchmark and Matplotlib chart are local validation additions.

No license file was present in the upstream root listing when inspected. No
license is granted here for the upstream code. Reuse/distribution permission
needs to be established before publishing the copied source.
