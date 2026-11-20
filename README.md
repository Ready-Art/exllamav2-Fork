# ReadyArt ExLlamaV2

**Tensor-parallel, speculative-decoding ExLlamaV2 for multi-GPU rigs.**

ReadyArt ExLlamaV2 is a performance-focused fork of [ExLlamaV2](https://github.com/turboderp/exllamav2)
that adds **multi-GPU tensor parallelism**, **MTP (multi-token prediction) speculative decoding**,
and first-class **Gemma-4-31B** support — so a 31B-class model runs fast across several consumer
GPUs (e.g. 4× RTX 3090) instead of being squeezed onto one card.

It keeps everything upstream ExLlamaV2 is good at (EXL2 quantization, the dynamic generator with
batching / prompt caching / K/V dedup, paged attention) and builds on it. Licensed under the same
MIT license as upstream; all original credit to [turboderp](https://github.com/turboderp) and contributors.

**Run it as an OpenAI-compatible server:** pair this fork with
**[ReadyArt/tabbyAPI-Fork](https://github.com/Ready-Art/tabbyAPI-Fork)** (branch `ReadyArt`) — it adds
SP/MP backend selection, MTP drafting, CUDA-graph/FlashInfer controls and Gemma-4 thinking/tool-call
handling on top of this fork. See [OpenAI-compatible server (TabbyAPI)](#openai-compatible-server-tabbyapi).

---

## Release status — alpha/beta proof of concept

> **This is an early alpha/beta release, intended as a proof of concept.** Performance is **not yet
> where we want it to be** — multi-GPU decode and MTP throughput still have significant headroom
> (the tensor-parallel collective path and quantized-GEMM efficiency are the main bottlenecks).
> Expect rough edges, APIs that may change, and speeds below what the hardware is ultimately capable
> of. It is shared to demonstrate the approach and invite feedback, not as a finished, tuned product.

**Model compatibility:** other EXL2 models *should* work, but this fork has only been **built and
tested on [Gemma-4-31B-IT](https://huggingface.co/ReadyArt/Gemma-4-31B-IT-EXL2-6.0BPW-HB16)**. Other
architectures are untested here — treat them as experimental and verify correctness/parity yourself.

**Hardware:** built and tested on **SM86 (Ampere — RTX 3090)** only (see [Requirements](#requirements)).

---

## Development focus & testing environment

> **The majority of the development effort has gone into the MP + CUDA graphs + MTP path** —
> multiprocess tensor parallelism (one worker per GPU), whole-step TP CUDA graphs, and Gemma-4
> MTP speculative decoding. This includes a custom **PCIe P2P collective kernel** that replaces
> NCCL for the small decode/verify all-gathers and all-reduces, which is where most of the
> recent performance work has landed.

- **Tested hardware:** **4× RTX 3090 (sm86) on a PCI switch with P2P enabled**
  (`NCCL_P2P_LEVEL=PXB`), no NVLink. The P2P collective path and the performance numbers in this
  README are validated on exactly this topology.
- **Power-throttled test rig:** the 3090s are **power-limited to 200 W each via
  [LACT](https://github.com/ilya-zlobintsev/LACT)** (stock TDP is 350 W). **Every performance number
  here is measured at that 200 W cap — a full-powered GPU will be significantly faster.** Treat the
  figures as a lower bound, not the ceiling.
- **Other configurations work, but they are not the priority.** Single-process TP (`sp`), the
  `native` paged backend, non-MTP decoding, other GPU architectures, and other model families are
  functional, but they receive far less optimization effort and **will not match the performance
  of the MP + CUDA graphs + MTP path** on the tested rig. Treat them as secondary paths.
- Only **Gemma-4-31B-IT** has been built and tested end-to-end (see [Release status](#release-status--alphabeta-proof-of-concept)).

## Feedback & contributions

- **Want a change?** Open a [GitHub issue](https://github.com/Ready-Art/exllamav2-Fork/issues) —
  bug reports, feature requests, and performance observations are all welcome. Describe what you
  want to see changed and, for perf reports, your hardware/topology and the numbers you measured.
- **Pull requests are welcome.** If you'd like to contribute code, open a
  [PR](https://github.com/Ready-Art/exllamav2-Fork/pulls) against the `ReadyArt` branch with a clear
  description of what you're changing and why.

---

## What this fork adds

| Feature | What it gives you |
|---|---|
| **Multi-GPU tensor parallelism (TP)** | Shard one model across GPUs. Single-process TP (`load_tp`, `--gpu_split`) for simplicity. |
| **Multiprocess backend (MP)** | One worker process per GPU (`TPProcessGroup`) so dispatch overlaps across devices — higher decode throughput than single-process TP. |
| **MTP speculative decoding** | A Gemma-4 *assistant* drafter proposes a window of tokens; the target verifies them in one forward. Works **with TP and MP** (the drafter gathers the target's sharded K/V). |
| **Gemma-4-31B-IT support** | Text-only. Handles the model's two attention geometries (head_dim 256 sliding / 512 full), `k_eq_v` layers, proportional RoPE and per-layer norms. |
| **Sliding-window + TP KV cache** | `ExLlamaV2Cache_SWA` / `ExLlamaV2Cache_TP`: ring buffers for sliding layers make long context fit (≈22.5 GB instead of ≈236 GB at 256k for Gemma-4-31B). |
| **TP CUDA graphs** | Whole-step decode graphs (`tp_graph`) remove per-op launch latency on the tensor-parallel path. |
| **FlashInfer attention (required)** | FlashInfer **≥ 0.7.4** is a hard dependency. On sm86 (RTX 3090) it serves **head_dim 512** on every path — paged decode (`BatchDecodeWithPagedKVCacheWrapper`, including CUDA-graph mode with a per-step re-plan), single-token decode, and prefill — verified cos 1.0 vs torch. Flash-Attention's paged kernel refuses hd512; FlashInfer does not. |
| **Multi-GPU quantization** | `convert.py -d 0,1 -crs on` distributes the measurement/quantization passes across GPUs. See [doc/convert.md](doc/convert.md). |

## Requirements

- **Built and tested on SM86 (Ampere — NVIDIA RTX 3090).** This fork is developed and validated
  on the Ampere `sm_86` architecture — **4× RTX 3090 on a PCI switch with P2P enabled**
  (`NCCL_P2P_LEVEL=PXB`), no NVLink, **power-limited to 200 W each via LACT** (stock 350 W). Other
  architectures may work but are untested. The bundled kernel build targets `TORCH_CUDA_ARCH_LIST="8.6"`.
- **Python 3.12** and a CUDA build of [PyTorch](https://pytorch.org/get-started/locally/).
  Tested combination: **Python 3.12.15, torch 2.15.0+cu132**, CUDA toolkit 13.x with `nvcc` on
  `PATH` (built with nvcc 13.5 against cu132 torch — a minor-version mismatch is only a warning).
- **FlashInfer ≥ 0.7.4** (hard requirement — importing the library raises if it is missing or older):
  ```sh
  uv pip install "flashinfer-python>=0.7.4"
  flashinfer download-kernels      # kernel wheels are CUDA/torch-version specific
  ```
  Re-run `flashinfer download-kernels` after every FlashInfer upgrade.
- **Flash-Attention 2 — required (for now) even with FlashInfer.** The tensor-parallel and
  dynamic-generator paths still call flash-attn kernels directly, so `flash_attn` must be installed
  alongside FlashInfer. Prebuilt wheel for the tested stack (Python 3.12 / torch 2.15 / CUDA 13.2):
  ```sh
  uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v1.5.0/flash_attn-2.8.3+cu132torch2.15-cp312-cp312-linux_x86_64.whl"
  ```
  The wheel must match your Python/torch/CUDA combination — for other builds pick a wheel from the
  [mjun0812 prebuilt-wheels releases](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases).

## Install

Verified on Ubuntu 26.10, 4× RTX 3090 (sm86), Python 3.12.15, torch 2.15.0+cu132, CUDA toolkit 13.5,
FlashInfer 0.7.4, flash-attn 2.8.3 — including end-to-end generation through TabbyAPI (SP MTP
sampling, MP MTP greedy, MP sampling fallback) with coherent, corruption-free multi-thousand-token
outputs. Install torch **before** building the fork: `setup.py` precompiles the CUDA extension
against the installed torch, and `--no-build-isolation` makes the build use it.

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python — it installs
Python 3.12 itself, so no system Python 3.12 package is needed:

```sh
# uv + Python 3.12 environment (tested combination)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv venv --python 3.12
source venv/bin/activate

# CUDA 13.2 torch — install BEFORE building the fork
uv pip install torch==2.15.0+cu132 --index-url https://download.pytorch.org/whl/cu132

# Fork + CUDA extension (nvcc must be on PATH; "8.6" = RTX 3090)
git clone https://github.com/Ready-Art/exllamav2-Fork
cd exllamav2-Fork
uv pip install -r requirements.txt
TORCH_CUDA_ARCH_LIST="8.6" uv pip install . --no-build-isolation

# FlashInfer kernels (hard requirement; re-run after every FlashInfer upgrade)
flashinfer download-kernels

# Flash-Attention 2 — required for now even with FlashInfer (see Requirements)
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v1.5.0/flash_attn-2.8.3+cu132torch2.15-cp312-cp312-linux_x86_64.whl"

# single-GPU sanity check
python test_inference.py -m <path_to_model> -p "Once upon a time,"

# multi-GPU inference (tensor parallel)
python test_inference.py -m <path_to_model> -p "Once upon a time," --gpu_split auto
```

For other torch/CUDA versions, install the matching torch build from the
[PyTorch wheel index](https://download.pytorch.org/whl/) and a matching flash-attn wheel
(see [Requirements](#requirements)); the extension compiles against whatever torch is installed.

## Multi-GPU inference

**Single-process tensor parallelism** — the simplest way to spread a model over GPUs:

```python
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.cache import ExLlamaV2Cache_TP, ExLlamaV2Cache_SWA

config = ExLlamaV2Config(); config.model_dir = "<model>"; config.prepare()
model = ExLlamaV2(config)
model.load_tp([20, 20, 20, 20],                       # VRAM (GiB) per GPU
              expect_cache_tokens=config.max_seq_len,
              expect_cache_base=ExLlamaV2Cache_SWA)    # SWA ring cache for Gemma-4
cache = ExLlamaV2Cache_TP(model, base=ExLlamaV2Cache_SWA, max_seq_len=config.max_seq_len)
```

**Multiprocess backend (MP)** — one worker per GPU for higher decode throughput:

```python
from exllamav2.tp_mp import TPProcessGroup
pg = TPProcessGroup("<model>", [20.0]*4, max_seq_len, max_seq_len, "ExLlamaV2Cache_SWA")
pg.start()
logits = pg.forward(input_ids, 0, n_prompt, 1, True)   # prefill
# ... decode with pg.forward(...) / pg.forward_state(...)
pg.close()
```

**MTP speculative decoding** — through the streaming generator (single-process TP) or the MP driver:

```python
from exllamav2.generator import ExLlamaV2StreamingGenerator
from exllamav2.mtp_drafter import Gemma4AssistantDrafter

drafter = Gemma4AssistantDrafter(model, "<gemma-4-assistant-drafter-dir>",
                                 device=0, num_draft_tokens=6)
gen = ExLlamaV2StreamingGenerator(model, cache, tokenizer,
                                  mtp_drafter=drafter, num_speculative_tokens=6)
gen.begin_stream(input_ids, settings, token_healing=True)
```

For the MP backend use `MTPMPDriver` (`exllamav2/mtp_mp.py`); runnable benchmarks and examples
are in [`mtp/`](mtp/). Every emitted token is sampled from the **target's** verified logits, so
the output distribution stays exactly target-autoregressive on both backends. On **SP**, a sampled
token that deviates from the cached draft/bonus token triggers a rewind-and-recommit of the cache
(BUG 100 fix) so the K/V can never desync from the emitted text. On **MP**, sampling uses
**probabilistic speculative acceptance** (BUG 105) — accepted drafts are kept with the correct
target probability and deviating tokens are resampled from the target's residual distribution,
avoiding the SP recommit cost. Greedy MTP matches plain autoregressive generation except for rare
fp16 near-tie flips between the verify-window and single-token decode attention kernels on long prose.

## Gemma-4-31B-IT

Pre-quantized EXL2 weights ready to run with this fork (6.0 bpw, fp16 head):
**[ReadyArt/Gemma-4-31B-IT-EXL2-6.0BPW-HB16](https://huggingface.co/ReadyArt/Gemma-4-31B-IT-EXL2-6.0BPW-HB16)** on Hugging Face.

`Gemma4ForConditionalGeneration` is supported for **text only** (vision tensors are skipped; passing
an image raises a clear error). The model mixes two attention geometries, which is why several
internals are per-layer:

| | sliding layers (50) | full layers (10) |
|---|---|---|
| `head_dim` | 256 | 512 |
| kv heads | 16 | 4 |
| RoPE | default, theta 10000 | proportional, 0.25, theta 1e6 |
| `v_proj` | present | absent (`attention_k_eq_v`: V is K) |
| KV cache | ring buffer, 1280 tokens | full `max_seq_len` |

Both geometries run on FlashInfer on sm86: the hd512 full layers use the paged decode wrapper
(including CUDA-graph mode) and FlashInfer prefill; the sliding layers use its windowed kernels.

Use the `gemma4` prompt format:
```
<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n
```

**MTP speculative decoding IS supported for Gemma-4 on both TP and MP** — the assistant drafter
cross-attends to the target's K/V, which the TP/MP cache layers gather onto the drafter device
(`gather_target` / `gather_kv_mtp`). On structured prompts MTP gives roughly **1.3–2×** decode
speedup over plain autoregressive generation (acceptance-dependent).

## OpenAI-compatible server (TabbyAPI)

**[ReadyArt/tabbyAPI-Fork](https://github.com/Ready-Art/tabbyAPI-Fork)** (branch `ReadyArt`) integrates
this fork into [TabbyAPI](https://github.com/theroyallab/tabbyAPI), so Gemma-4-31B-IT runs behind an
OpenAI-compatible API with:

- `tensor_parallel_backend: sp` (single-process TP) or `mp` (one worker per GPU),
- MTP drafting via `draft_model.draft_mode: mtp` + the
  [gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant) drafter
  (full sampler on both SP and MP),
- TP CUDA graphs (`tp_graphs`) and FlashInfer controls,
- Gemma-4 **thinking** (thought-channel `reasoning_content`) and **tool calls** (`gemma4` tool
  format), plus `reasoning_effort` on chat completions.

See the TabbyAPI repo's `docs/11.-ReadyArt-ExLlamaV2-Fork.md` for configuration and limitations.

## Conversion & quantization (multi-GPU)

Quantize to EXL2 with `convert.py`; the measurement and quantization passes can be spread across
GPUs with `-d` (device list) and `-crs on` (shard calibration rows, one module replica per device):

```sh
python convert.py -i <hf_fp16_model> -o <work_dir> -cf <out_dir> \
    -b 6.0 -hb 16 -d 0,1 -crs on
```

See **[doc/convert.md](doc/convert.md)** for all arguments, the multi-GPU process, hardware
requirements, and how to run the converted model tensor-parallel.

## Performance (this fork)

> Early proof-of-concept numbers — **below the target**, not a tuned result (see
> [Release status](#release-status--alphabeta-proof-of-concept)). Shared for transparency.

Measured on **4× RTX 3090 (sm86) on a PCI switch with P2P enabled** (`NCCL_P2P_LEVEL=PXB`),
**power-limited to 200 W each via LACT** (stock 350 W), Gemma-4-31B EXL2 6.0 bpw, greedy.
**These are 200 W-cap numbers — full-powered GPUs will be significantly faster:**

| Mode | 4k context | counting prompt |
|---|---:|---:|
| Plain decode (MP) | ~39 tok/s | ~40 tok/s |
| **MTP (MP)** | **~40 tok/s** | **~53 tok/s (≈1.35×)** |

The MP decode/verify path now uses a custom **in-graph PCIe P2P collective kernel** for the small
all-gathers/all-reduces (replacing NCCL's per-op latency on this no-NVLink topology). Recent
long-context measurements on the same rig: **plain decode ~28.9 tok/s at 122k context**, **MTP
~55 tok/s at ~1.3k context**, and a **64k-token prefill in ~97 s** (`EXL2_TP_O_RS=1`, the default).
These are still proof-of-concept numbers with headroom remaining (see Release status).

MTP acceptance ≈ 2.7 tokens/round on mixed prose, ≈ 3.9 on structured prompts, up to 5–6 on
arithmetic. Speedups scale with acceptance; long-context (32k/64k) MTP also matches or beats plain
decode. Numbers vary with prompt, context length and hardware. **Short-context caveat:** at ~512
tokens of context, MP MTP greedy currently runs *slower* than plain MP decode (~27 vs ~37 tok/s
measured through TabbyAPI) — the per-round fixed costs (worker-pipe K/V gather/rewind round trips
and the torch hd512 verify-window attention) dominate when there is little context to amortize
them over. Reducing those round costs is the main MTP optimization target.

**Sampling caveat:** the numbers above are greedy. On **SP**, under sampling (e.g. temperature 1.0)
every deviation from the drafted tokens costs a rewind + recommit forward (the BUG 100 correctness
fix), so MTP throughput drops toward — or below — plain decode at short context (~12 vs ~19 tok/s
SP measured at temp 1.0 / top_p 0.92). On **MP**, sampling now uses **probabilistic speculative
acceptance** (BUG 105), which keeps accepted drafts at the correct target probability and resamples
deviations from the target's residual distribution — recovering sampling throughput without the SP
recommit cost. MTP's speedup is still largest on greedy / low-temperature / structured generation
where acceptance is high.

## Upstream features

The dynamic generator (batching, smart prompt caching, K/V dedup, simplified API), sampling filters,
Q4 cache, and the rest of upstream ExLlamaV2 are retained. See [doc/dynamic.md](doc/dynamic.md) and
the [examples/](examples/) directory. Upstream docs: [ExLlamaV2 wiki](https://github.com/turboderp/exllamav2/wiki).

## License

MIT, as upstream ExLlamaV2. See [LICENSE](LICENSE). This fork is not affiliated with the original
ExLlamaV2 authors; all original credit belongs to them.