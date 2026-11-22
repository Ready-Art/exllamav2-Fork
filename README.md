# ReadyArt ExLlamaV2

**ReadyArt Exllamav2.5 v1.0** — tensor-parallel, speculative-decoding ExLlamaV2 for multi-GPU
rigs, with the **EXL2.5** quantization format (fast-path GEMM + Hadamard/AWQ quality upgrades;
see [EXL2.5 quantization format](#exl25-quantization-format-new)).

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

> **Collective fusion (in progress).** The P2P rendezvous is being fused to cut per-op PCIe
> latency: a warp-parallel flag/ack protocol (one lane per peer) and a graph-safe deferred-ack
> allreduce with per-key double-buffered scratch are landed; an optional fused
> allgather+RMSNorm+residual kernel (`EXL2_TP_FUSE_AGNR=1`) collapses the decode block tail into
> one collective. These are **latency** optimizations — on the 200 W-cap rig the cards are
> compute-limited by quantized-GEMM dequant, so throughput gains are small; the remaining big
> lever is the GEMM dequant path itself. `EXL2_TP_ROW_REPL=1` fuses the **o_proj** block into a
> single allreduce (fp16 logical-order row shard; greedy-parity verified). The down row-parallel
> variant is **disabled** (`EXL2_TP_ROW_REPL_DOWN=0` default — its remap path has a correctness
> bug and an fp16 down shard is VRAM-prohibitive); down stays column-parallel.
> `EXL2_MTP_COST_TRACE=1` prints a per-phase gather cost breakdown for profiling.

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
- **Multi-batch (batch > 1) is experimental.** It is implemented on the MP + CUDA graphs + MTP
  path and validated under true concurrent load at `max_batch_size: 4` (4× RTX 3090, mixed context
  lengths, greedy and sampled, shared-prefix KV cloning). The default remains batch size 1. See
  [Multi-batch](#multi-batch-experimental-mp--mtp).
- **DFlash block-diffusion drafting is experimental.** Implemented on the MP path (batch size 1)
  with first end-to-end runs verified on the tested rig; long-context and multi-turn tool-loop
  validation is in progress. See [DFlash](#dflash-block-diffusion-drafting-experimental-mp).

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
| **DFlash block-diffusion drafting (experimental)** | A [z-lab DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) drafter denoises a whole block (anchor + up to 15 mask tokens) in **one parallel pass**, conditioned on the target's hidden states; FlashInfer attention. **MP only**, batch size 1; see [DFlash](#dflash-block-diffusion-drafting-experimental-mp). |
| **Multi-batch (experimental)** | Slot-based concurrency on the **MP + MTP** path (`TPProcessGroup(max_batch=N)` + `MTPMPBatchDriver`): one cache row per job, batched graphed verify rounds, per-slot append-only prompt caching. Opt-in; see [Multi-batch](#multi-batch-experimental-mp--mtp). |
| **Prompt-cache RAM offload (experimental)** | Tier idle prompt-cache lineages to host RAM under a **whole-server** byte cap, so the VRAM context budget is not a hard limit on cached conversations (API sharing). `TPProcessGroup.offload_row` / `restore_row`; restores the MTP shadow + boundary ring snapshot (thought-stripped and long conversations). Opt-in; see [Prompt-cache RAM offload](#prompt-cache-ram-offload-experimental-mp--mtp). |
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

### GPU count (2 / 3 / 4 validated; 5+ head-count limit)

TP shards the KV heads with a largest-remainder integer split, so **uneven GPU counts work** —
the per-rank head counts need not be equal. On Gemma-4-31B the two attention geometries split
independently (e.g. on **3 GPUs**: sliding kv 16 → 6/5/5, full-attention kv 4 → 2/1/1). The MP
path is validated at **2, 3 and 4 GPUs** (3-GPU load + MTP + multi-batch + prompt-cache offload
tested coherent and corruption-free; 4-GPU regression clean).

Two uneven-split bugs were fixed for non-power-of-two counts: the device scratch estimate now
selects each layer's geometry before sizing (it previously estimated with a stale split and
overflowed at 3 GPUs), and the MTP K/V gather joins each rank's **actual** head width from the
geometry split rather than assuming `total // world`.

**5+ GPUs are limited by the model's KV-head count:** every rank must hold at least one KV head
on every layer. Gemma-4's full-attention layers have only **4** KV heads, so across 5 GPUs one
rank gets zero full-layer heads — the attention path has no KV-head replication for that case,
so 5 GPUs is **not supported for Gemma-4** (the sliding layers' 16 heads would split fine). A
model with ≥ `world_size` KV heads on every layer can use 5+ GPUs with the same generic split /
gather path.

### DFlash block-diffusion drafting (experimental, MP)

**Status: experimental.** First end-to-end runs are verified on the tested rig (Gemma-4-31B-IT,
4× RTX 3090 @ 200 W); long-context and multi-turn tool-loop validation is in progress.

[DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) (z-lab,
[paper](https://arxiv.org/abs/2602.06036)) is a block-diffusion drafter: instead of MTP's
sequential draft steps, it denoises a whole block — one anchor token plus up to 15 mask tokens —
in **one parallel forward**. It conditions on the target's residual streams at six layers
(concat → projection → per-layer K/V), so the draft K/V is derived from target hidden states
captured during prefill/verify; the draft model never re-processes the prompt. Attention is
FlashInfer paged prefill (sliding layers causal + window, the full layer non-causal within the
block). Draft cost is ~flat in the window width — that is where DFlash wins over MTP at large
windows.

- **MP backend only**, `max_batch_size: 1`. The drafter runs on one device (default 0) and needs
  a full fp16 copy of the target lm_head plus its own position-indexed draft K/V cache: ~6.6 GB
  extra on the draft device at 32k cache, ~9.2 GB at 160k (fits 24 GB cards, but tight — prefer
  `cache_size: 32768` unless you need the depth). **On 2-GPU splits set `draft_device` to a
  worker GPU (e.g. `1`)** — the master's doubled weight shard otherwise leaves too little room
  for prefill; a load-time warning points at the freest device if the placement is tight.
- Same verify/accept protocol as MP MTP (`MTPMPDriver`): every emitted token comes from the
  target's verified distribution (greedy or probabilistic acceptance), so output stays exactly
  target-autoregressive. The append-only prompt cache applies too (draft K/V persists per lineage).
- Through TabbyAPI: `draft_model.draft_mode: dflash` (a DFlash checkpoint is also auto-detected
  under `mtp` mode). `draft_num_tokens` defaults to the checkpoint's `block_size - 1` (15);
  smaller windows are supported.
- exllamav2 side: `DFlashDrafter` (`exllamav2/dflash_drafter.py`) + `TPProcessGroup.enable_mtp`;
  greedy parity harness in [`mtp/dflash_test.py`](mtp/dflash_test.py) (DFlash greedy must match
  plain autoregressive output exactly).

### Multi-batch (experimental, MP + MTP)

**Status: experimental.** Validated on the tested 4× RTX 3090 rig under true concurrent load at
batch size 4 (mixed context lengths up to ~15k, greedy and sampled, shared-prefix cloning,
per-slot prompt-cache continuations). The default is batch size 1, which keeps the
single-sequence path unchanged.

The MP + MTP path can run several jobs concurrently with `TPProcessGroup(..., max_batch=N)` and
`MTPMPBatchDriver` (`exllamav2/mtp_mp.py`). Each job owns one **cache row (slot)**:

- Per-row sequence positions / SWA ring origins; the graphed verify window is batched
  (`(B, W+1)` rows, one CUDA graph per batch size) with per-row acceptance, rewind and sampling —
  the same target-autoregressive guarantees as the scalar driver.
- **Full-batch rounds**: each scheduler step drains every active row's decided tokens (a burst of
  `m_acc+1` chunks per row), so every verify round runs at the full active batch size. Staggered
  admissions and variable acceptance cannot desynchronize the rows into solo rounds.
- The drafter cross-attends to all active rows in one batched FlashInfer decode per draft step.
  With `EXL2_MTP_DRAFT_GRAPH=1` the whole W-step draft window is a captured CUDA graph per batch
  size (GPU-resident embed table, on-device argmax/state chaining, per-round FlashInfer plan).
- **Sparse draft head** (`EXL2_SPARSE_DRAFT_HEAD=<columns>`) works on the batch path too: the
  frequency subset is rebuilt at job boundaries, baked into the draft graphs, and probe-gated.
- **Per-slot append-only prompt cache**: a job reuses a slot's K/V when its prompt strictly
  extends that slot's own lineage, restores its prompt-boundary ring snapshot, or shares a long
  prefix with it (edited/regenerated turns rewind to the divergence point — a position
  truncation, no copies).
- **Cross-slot shared-prefix clone**: a new job whose prompt shares a long prefix with another
  slot's lineage (e.g. concurrent requests with the same system prompt / document) copies that
  slot's committed K/V — full layers, SWA ring and the drafter shadow — instead of re-prefilling
  the prefix. The clone is a physical copy: slots never share live K/V afterwards, so concurrent
  conversations cannot cross-contaminate. Knobs: `EXL2_BATCH_KV_CLONE=0` disables,
  `EXL2_BATCH_CLONE_MIN` sets the minimum shared prefix (default 512 tokens).
- **Context-regime graph management**: a verify graph whose kv length leaves its capture envelope
  (±2048) is recaptured *per (B, q_len) shape* — other shapes keep their graphs, FlashInfer
  wrappers and P2P arena regions. Each shape reuses a stable arena epoch across recaptures, so
  mixed-length concurrency does not exhaust the P2P slot arena.
- **Threaded acceptance**: the per-row sampler pipeline (the ext kernels release the GIL) runs in
  a thread pool for penalty-free sampling rows — same kernels, same draw order, same distribution.
- Prefill is serialized per slot (chunked, with decode rounds interleaved between chunks);
  generation is capped so a verify window always fits inside the cache depth.
- Progress line every 50 rounds: `[mtp-mp-batch] … acc/round … emit/round … ms/round
  (gather/draft/verify/accept/rewind) … draft_agree …` (acceptance and draft/target argmax
  agreement; judge speed by total emitted tokens/s across streams). Measured ~100-110 tok/s
  **aggregate** across 4 concurrent streams on open-ended essay prompts at temperature 0.7
  (4× RTX 3090, W=3, `cache_size: 32768`); acceptance-dependent — predictable content and
  greedy sampling run faster.

VRAM scales linearly with `max_batch`: the KV cache and the joined MTP K/V shadow are allocated
per row (~5 GB/row at 160k context for Gemma-4-31B on 4 GPUs; ~1.2 GB/row at 32k). On 24 GB cards,
batch size 4 fits at `cache_size: 32768` on 4 GPUs (master ~18.5 GB with the draft graph's embed
table). **On 2 GPUs each rank holds twice the KV heads**, so halve the per-slot budget again
(e.g. `cache_size: 8192–16384` for batch 4, or a lower-bpw quant — 6.0 bpw weights alone are
~11.7 GB/rank on 2 GPUs). Through TabbyAPI, enable it with `model.max_batch_size` (see the
TabbyAPI fork docs); the exllamav2-side knobs are `TPProcessGroup(max_batch=...)` and the
`forward_batch` / `gather_kv_mtp_batch` / `clone_row` protocol.

### Prompt-cache RAM offload (experimental, MP + MTP)

**Status: experimental.** Row-level tiering of idle prompt-cache lineages to host RAM, so the
VRAM context budget is not a hard cap on how many conversations stay cached (e.g. when sharing
the API). The VRAM rows keep the hot lineages; an idle lineage whose row is needed by a new job
is copied to RAM and a later matching request restores it in milliseconds instead of
re-prefilling. Works at batch 1 and multi-batch; MTP only (DFlash keeps its own draft K/V).

The `TPProcessGroup` primitives (`exllamav2/tp_mp.py`):

- `offload_row(row, pos, wb, entry_id)` → a bundle. Each rank D2Hs its own slice of the row's
  committed K/V (full layers `[0, pos)`, SWA ring valid prefix) into **worker-local host
  storage** — the K/V never crosses the process pipe, only metadata does. The master keeps its
  own slice plus the joined MTP draft shadow row; the prompt-boundary ring snapshot is re-keyed
  to the entry (zero-copy, it is already host-resident). Returns `None` on host-allocation
  failure (the caller skips the offload).
- `restore_row(row, bundle, p=None, attach_snapshot=False)` → H2Ds the stored prefixes back into
  the row at position `p` (default the full committed position), restores the shadow gather
  frontiers (so the next gather pulls only the delta), and optionally re-attaches the boundary
  ring snapshot. `truncate_cache(p)` is the batch-1 partial-prefix rewind.
- `drop_offload_bundle(bundle)` frees the worker-local storage + detached snapshot (idempotent).

Semantics and safety:

- **No VRAM allocation** anywhere in these paths — every copy targets the preallocated cache
  rows / shadow buffers, so offload/restore cannot OOM VRAM and never touch the CUDA graphs
  (they run between rounds, like `clone_row`).
- The **RAM cap is global to the server**, enforced by the scheduler's cold store (TabbyAPI
  `model.prompt_cache_ram_limit_mb`): it sums every rank's stored slice + snapshots + shadow per
  lineage and evicts oldest/least-used first. Per-process RSS also includes uncapped working
  buffers (hot-row ring snapshots, grow-only pinned pack staging, CUDA context, the master's
  drafter host embed), so it exceeds `cap / num_gpus`.
- `cache_size` stays the per-row VRAM depth; RAM extends the *number* of cached lineages, not a
  single conversation's attendable depth (full-attention K/V must be VRAM-resident for attention).
- **Thought-stripped / long conversations**: restoring the full lineage + re-attaching the
  boundary snapshot lets the scheduler rewind the SWA ring to the prompt boundary even after a
  long generation shifted the ring past it, so a re-rendered continuation restores its document
  instead of re-prefilling it.
- Failure-soft: a failed restore drops the entry and re-prefills; the job never fails because of
  the offload system. Through TabbyAPI: `model.prompt_cache_offload: true` (+
  `prompt_cache_ram_limit_mb`); `EXL2_PROMPT_OFFLOAD=0` kill switch, `EXL2_OFFLOAD_DEBUG=1`
  decision logging. See the TabbyAPI fork docs for the config and the global-cap details.

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

## EXL2.5 quantization format (new)

**EXL2.5** is this fork's next-generation quant format: same EXL2 file container, plus
per-tensor opt-in upgrades for **speed** (a branchless fast-path GEMM built for Ampere /
RTX 3090, with a 16-row cap so batch-4 MTP verify reads the weights once) and **quality**
(Hadamard incoherence rotation + AWQ channel protection + Hessian-guided rounding at the
same bits-per-weight). Plain EXL2 models keep loading and running exactly as before.

### The one rule to understand (please read)

> **EXL2.5's new features require the EXL2.5 fast path. Mixed-bpw tensors revert to EXL2.**

Spelled out:

- A tensor converted **with** `--exl25_uniform_bits N` is an **EXL2.5 tensor**: one
  bit-width for the whole tensor, uniform groups, served by the new fast-path kernel.
  Hadamard (`--exl25_hadamard`) and AWQ (`--exl25_awq`) **only work on such tensors** —
  the fast path is what applies their activation transforms, and the loader refuses
  (loudly) to run them anywhere else.
- A tensor converted **without** `--exl25_uniform_bits` is a **classic mixed-bpw EXL2
  tensor** and runs on the **original EXL2 kernel**, byte-for-byte unchanged. Nothing
  about EXL2.5 slows it down or changes its output.
- You can **mix both kinds in one model**. You still choose any model-wide BPW you like:
  either classic per-tensor mixed-bit allocation (EXL2 path), or different uniform
  bit-widths per tensor/type (EXL2.5 path), e.g. attention 6-bit + MLP 4-bit + fp16 head.
  What you cannot do is put Hadamard/AWQ on a mixed-bpw tensor — those require the uniform
  fast path by design.
- `EXL2_EXL25=0` is an A/B kill switch: it serves **plain uniform** tensors with the legacy
  kernel (same weights, slower) so you can measure the speedup. Hadamard/AWQ tensors are
  exempt — they always use the fast path, because the legacy kernel cannot produce correct
  results for them.

### Hadamard requires an fp16 head (HB16)

> **With `--exl25_hadamard`, use `-hb 16`.** If you quantize the head (`-hb 6` etc.), the
> converter **skips Hadamard on the lm_head tensor** (prints a warning; the head is
> quantized unrotated). The output head is the most quality-sensitive tensor in the model;
> keep it fp16 when using Hadamard.

### Converting to EXL2.5

```sh
# Uniform 6-bit fast path + Hadamard + AWQ, fp16 head (recommended quality preset)
python convert.py -i <hf_fp16_model> -o <work_dir> -cf <out_dir> \
    -b 6.0 -hb 16 -x25u 6 -x25h -x25a

# Uniform 4-bit fast path (speed/size preset), no rotation
python convert.py -i <hf_fp16_model> -o <work_dir> -cf <out_dir> \
    -b 4.0 -hb 16 -x25u 4
```

| Flag | Meaning |
|---|---|
| `-x25u / --exl25_uniform_bits N` | EXL2.5 uniform fast path at N bits (2/3/4/5/6/8). 0/absent = classic EXL2. |
| `-x25g / --exl25_group_size G` | Uniform group size along K (32/64/128/256; K must divide). Default 128. Hadamard/AWQ tensors use ≤128. |
| `-x25h / --exl25_hadamard` | K-side Hadamard incoherence: weights pre-rotated at conversion, activations rotated inside the GEMM (free at inference). Lower quant error at the same bpw. Requires `-x25u` and `-hb 16`. |
| `-x25a / --exl25_awq` | AWQ-style per-channel protection from calibration data. Requires `-x25u`. |

The converted model runs on the normal load path — tags are self-describing, no config
switch needed. MoE expert tensors get the same treatment automatically (the fused MoE
kernel serves EXL2.5 experts with routing weights). Engram (hashed n-gram) tables have
their own scalar row format + converter (`python -m exllamav2.conversion.convert_ngram`)
and module (`exllamav2.ngram_embedding`), RAM-resident or disk-streaming — no trellis,
Ampere-friendly.

Validation harnesses (developer): `mtp/exl25_gemm_test.py` (parity m=1..16 + cuBLAS path,
quality A/B, speed A/B vs legacy, CUDA-graph capture/replay), `mtp/exl25_moe_test.py`
(fused MoE parity), `mtp/exl25_ngram_test.py` (engram converter/kernel/hashing parity).

### EXL2.5 validation status (2026-11-22, RTX 3090 / sm86, same-session A/B at 200 W)

- **Correctness: ALL PASS** — GEMM parity m=1..16 + prefill cuBLAS+transform path (fp32
  oracle, fp16 noise floor), CUDA-graph capture/replay m=1/4/16, fused MoE with EXL2.5
  experts (plain + Hadamard/AWQ, rows 1/2/4), engram converter/kernel/hashing (RAM +
  disk-streaming, 4/8-bit). Hadamard butterfly verified against an exact-answer probe.
- **Quality (same bpw, synthetic outlier stress test)**: Hadamard+AWQ cuts quantization
  rel_l2 ~30-45% (e.g. 6-bit 0.0236 → 0.0133; 4-bit 0.095 → 0.054). Real-model perplexity /
  MTP-acceptance measurement pending the first EXL2.5 conversion.
- **Kernel speed vs legacy (plain uniform, identical weights)**: decode m=1 **1.05-1.27×**,
  m=4 up to **2.13×** (6-bit gate/up), m=8 mixed (0.42-2.22× by shape/bit-width), m=16
  currently slower than legacy's two m=8 passes on most shapes — **m=8/16 register-pressure
  / block-size tuning is the next lever** (the 16-row one-pass design is in place; occupancy
  tuning pending). Hadamard/AWQ add an in-kernel butterfly cost on top (quality feature).
- EXL2.5 is **opt-in per tensor**; existing EXL2 models and the legacy kernel are untouched.

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

The MP decode/verify path uses a custom **in-graph PCIe P2P collective kernel** for the small
all-gathers/all-reduces (replacing NCCL's per-op latency on this no-NVLink topology), with a
deferred-ack rendezvous (one graph-start barrier instead of per-op ack spins). Recent
measurements on the same rig (2026-11-21, via tabbyAPI): **MTP ~68 tok/s at ~1.3k context,
temperature-1 sampling**, with `draft_num_tokens: 3` and the whole-window drafter CUDA graph
enabled — launched as `NCCL_P2P_LEVEL=PXB EXL2_MTP_DRAFT_GRAPH=1 python main.py` with
`gpu_split: [22, 22, 22, 22]`, `cache_size: 32768`, `chunk_size: 2048`, `max_batch_size: 1`.
Also: **plain decode ~28.9 tok/s at 122k context** and a **64k-token prefill in ~97 s**
(`EXL2_TP_O_RS=1`, the default). These are still proof-of-concept numbers with headroom
remaining (see Release status).

MTP acceptance ≈ 2.7 tokens/round on mixed prose, ≈ 3.9 on structured prompts, up to 5–6 on
arithmetic. Speedups scale with acceptance; long-context (32k/64k) MTP also matches or beats plain
decode. Numbers vary with prompt, context length and hardware. **Short-context caveat:** at ~512
tokens of context, MP MTP greedy currently runs *slower* than plain MP decode (~27 vs ~37 tok/s
measured through TabbyAPI) — the per-round fixed costs (worker-pipe K/V gather/rewind round trips
and the hd512 verify-window attention) dominate when there is little context to amortize
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