# ReadyArt ExLlamaV2

**Tensor-parallel, speculative-decoding ExLlamaV2 for multi-GPU rigs** — with the **EXL2.5**
quantization format (fast-path GEMM + Hadamard/AWQ quality upgrades).

ReadyArt ExLlamaV2 is a performance-focused fork of [ExLlamaV2](https://github.com/turboderp/exllamav2)
that adds **multi-GPU tensor parallelism**, **MTP (multi-token prediction) speculative decoding**,
and first-class **Gemma-4-31B** support — so a 31B-class model runs fast across several consumer
GPUs (e.g. 4× RTX 3090) instead of being squeezed onto one card. It keeps everything upstream
ExLlamaV2 is good at (EXL2 quantization, the dynamic generator with batching / prompt caching /
K/V dedup, paged attention) and builds on it.

**Run it as an OpenAI-compatible server:** pair this fork with
**[ReadyArt/tabbyAPI-Fork](https://github.com/Ready-Art/tabbyAPI-Fork)** (branch `ReadyArt`) — it adds
SP/MP backend selection, MTP drafting, CUDA-graph/FlashInfer controls and Gemma-4 thinking/tool-call
handling. See [OpenAI-compatible server](#openai-compatible-server-tabbyapi).

Licensed under the same MIT license as upstream; all original credit to
[turboderp](https://github.com/turboderp) and contributors.

---

## Release status — alpha/beta proof of concept

> **This is an early alpha/beta release, intended as a proof of concept.** Expect rough edges and
> APIs that may change. It is shared to demonstrate the approach and invite feedback, not as a
> finished, tuned product.

- **Model compatibility:** built and tested end-to-end on
  **[Gemma-4-31B-IT](https://huggingface.co/ReadyArt/Gemma-4-31B-IT-EXL2-6.0BPW-HB16)** only.
  **[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** (hybrid gated-delta-net backbone),
  **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (hybrid GDN + QSA
  indexed attention + 512-expert MoE + hyper-connections + PLE n-grams + built-in MTP) and
  **[Gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)** (Gemma-4 with a
  parallel 128-expert MoE block) are implemented and **synthetic-parity verified** (HF reference
  logits / block math / MP fork-vs-fork), but **not yet validated on real quantized checkpoints
  on the rig** — treat them as experimental. Other EXL2 models *should* work; other architectures
  are untested here — verify correctness yourself.
- **Hardware:** built and tested on **SM86 (Ampere — RTX 3090)** only. The kernel build targets
  `TORCH_CUDA_ARCH_LIST="8.6"`. Other architectures may work but are untested.
- **Primary path:** the majority of development and optimization goes into
  **MP (multiprocess TP) + CUDA graphs + MTP**, including a custom in-graph **PCIe P2P collective
  kernel** that replaces NCCL for the small decode/verify all-gathers/all-reduces on no-NVLink
  rigs. Single-process TP (`sp`), the `native` paged backend, non-MTP decoding and other model
  families are functional but secondary — they will not match the primary path's performance.
- **Test rig:** 4× RTX 3090 on a PCI switch with P2P enabled (`NCCL_P2P_LEVEL=PXB`), no NVLink,
  **power-limited to 200 W each** via [LACT](https://github.com/ilya-zlobintsev/LACT) (stock 350 W).
  **Every performance number in these docs is a 200 W-cap lower bound — full-powered GPUs are
  significantly faster.**

---

## Quickstart

Tested combination: **Ubuntu 26.10, Python 3.12.15, torch 2.15.0+cu132, CUDA toolkit 13.5
(`nvcc` on `PATH`), FlashInfer 0.7.4, flash-attn 2.8.3**, 4× RTX 3090 (sm86).

Requirements:

- **Python 3.12** and a CUDA build of [PyTorch](https://pytorch.org/get-started/locally/).
  Install torch **before** building the fork — `setup.py` compiles the CUDA extension against it.
- **FlashInfer ≥ 0.7.4** — a hard dependency (the import gate raises if it is missing or older).
  It serves head_dim 512 on sm86, which Flash-Attention's paged kernel refuses.
- **Flash-Attention 2 — still required alongside FlashInfer** (the TP and dynamic-generator paths
  call flash-attn kernels directly; phasing it out is in progress).

```sh
# uv + Python 3.12 environment (uv installs Python itself)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv venv --python 3.12
source venv/bin/activate

# CUDA 13.2 torch — install BEFORE building the fork
uv pip install torch==2.15.0+cu132 --index-url https://download.pytorch.org/whl/cu132

# Fork + CUDA extension ("8.6" = RTX 3090; use your arch, e.g. "8.9" for RTX 4090)
git clone https://github.com/Ready-Art/exllamav2-Fork
cd exllamav2-Fork
uv pip install -r requirements.txt
TORCH_CUDA_ARCH_LIST="8.6" uv pip install . --no-build-isolation

# FlashInfer kernels (hard requirement; re-run after every FlashInfer upgrade)
uv pip install "flashinfer-python>=0.7.4"
flashinfer download-kernels

# Flash-Attention 2 — wheel must match your Python/torch/CUDA combination.
# This one is Python 3.12 / torch 2.15 / CUDA 13.2; others:
# https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v1.5.0/flash_attn-2.8.3+cu132torch2.15-cp312-cp312-linux_x86_64.whl"

# single-GPU sanity check
python test_inference.py -m <path_to_model> -p "Once upon a time,"

# multi-GPU inference (tensor parallel)
python test_inference.py -m <path_to_model> -p "Once upon a time," --gpu_split auto
```

For other torch/CUDA versions, install the matching torch build from the
[PyTorch wheel index](https://download.pytorch.org/whl/) and a matching flash-attn wheel; the
extension compiles against whatever torch is installed.

**Model:** pre-quantized Gemma-4-31B-IT EXL2 weights (6.0 bpw, fp16 head):
[ReadyArt/Gemma-4-31B-IT-EXL2-6.0BPW-HB16](https://huggingface.co/ReadyArt/Gemma-4-31B-IT-EXL2-6.0BPW-HB16).
MTP drafter (939 MB): [google/gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant).

---

## What this fork adds

| Feature | What it gives you |
|---|---|
| **Multi-GPU tensor parallelism (TP)** | Shard one model across GPUs. Single-process TP (`load_tp`, `--gpu_split`) for simplicity. |
| **Multiprocess backend (MP)** | One worker process per GPU (`TPProcessGroup`) so dispatch overlaps across devices — higher decode throughput than single-process TP. |
| **MTP speculative decoding** | A Gemma-4 *assistant* drafter proposes a window of tokens; the target verifies them in one forward. Works with TP and MP, greedy and sampled. |
| **EXL2.5 quantization format** | Per-tensor opt-in upgrades: uniform-bit fast-path GEMM (Ampere-optimized, 16-row cap), Hadamard incoherence + AWQ channel protection for lower quant error at the same bpw. Plain EXL2 models run unchanged. See [EXL2.5](#exl25-quantization-format). |
| **Gemma-4-31B-IT support** | Text-only. Two attention geometries (head_dim 256 sliding / 512 full), `k_eq_v` layers, proportional RoPE, per-layer norms. |
| **Qwen3.8-27B support (experimental)** | Text-only hybrid backbone: gated delta net (linear attention) + output-gated full attention, built-in MTP drafter, TP/MP. Not yet rig-validated. See [Experimental features](#experimental-features). |
| **Qwen3.8-Flash-Next support (experimental)** | Text-only `qwen4_exp`: 4-stream hyper-connections, gated delta net + QSA indexed sparse attention (top-512 blocks), 512-expert MoE + sigmoid shared expert, PLE hashed n-gram embeddings (engram disk/RAM/hybrid storage), built-in MTP. MP: GDN shards by value heads, attention/PLE/HC replicated, MoE replicate/shard/offload. Synthetic-parity verified; real-checkpoint validation pending. |
| **Gemma-4-26B-A4B support (experimental)** | Gemma-4 attention (SWA hd256 / full hd512 `k_eq_v`) plus a parallel per-layer MoE block: 128 experts top-8 with scale/per-expert-scale router, dense MLP branch, fused-3D expert quantization. MP: low-KV-head layers replicate, MoE replicate/shard/offload. Synthetic block-math parity cos 1.0; real-checkpoint validation pending. |
| **Built-in MTP on MP** | The built-in drafters (Qwen3.5/3.8 and Flash-Next) run on the MP path through `BuiltinMTPMPAdapter`: own draft KV on the master rank, target GDN/PLE verify state staged and restored by the distributed rewind. Greedy chain; sampled acceptance resamples from verified target logits. Single-sequence (`max_batch_size 1`). |
| **Sliding-window + TP KV cache** | Ring buffers for sliding layers make long context fit (≈22.5 GB instead of ≈236 GB at 256k for Gemma-4-31B). |
| **TP CUDA graphs** | Whole-step decode/verify graphs remove per-op launch latency on the tensor-parallel path. |
| **FlashInfer attention** | head_dim 512 on every path on sm86 — paged decode (incl. CUDA-graph mode), single-token decode, prefill. |
| **Multi-batch (experimental)** | Slot-based concurrency on MP + MTP, validated at batch 4. See [Experimental features](#experimental-features). |
| **Prompt-cache RAM offload (experimental)** | Tier idle conversations to host RAM under a whole-server cap. See [Experimental features](#experimental-features). |
| **INT8 KV cache (opt-in)** | INT8 K/V shadow + custom flash-decode halves long-context decode KV reads. Default fp16. |
| **Priority scheduler (opt-in)** | Shortest-first admission + reprocess-abuse penalty for shared-API serving. |
| **DFlash drafting (experimental)** | Block-diffusion drafter: a whole block in one parallel pass. See [Experimental features](#experimental-features). |
| **Multi-GPU quantization** | `convert.py -d 0,1,2,3 -crs on` spreads conversion across GPUs; run several conversions in parallel on separate GPUs. See [doc/convert.md](doc/convert.md). |

---

## Performance

Measured on the test rig (4× RTX 3090 sm86, PCIe P2P, **200 W cap per card** — see
[Release status](#release-status--alphabeta-proof-of-concept)), Gemma-4-31B EXL2 6.0 bpw,
through TabbyAPI. Judge MTP by **total emitted tokens/s**, not base decode speed.

| Mode | Throughput | Conditions |
|---|---:|---|
| **MTP (MP), temp-1 sampling** | **~68 tok/s** | ~1.3k context prose, `draft_num_tokens: 3`, draft CUDA graph ON |
| Plain decode (MP) | ~28.9 tok/s | 122k context |
| Prefill | 64k tokens in ~97 s | chunk 2048, `EXL2_TP_O_RS=1` (default) |
| Multi-batch B=4 (experimental) | ~85–110 tok/s aggregate | 4 concurrent streams, temp 0.7, W=3, cache 32768 |
| DFlash (experimental) | ~51 tok/s | W=3, ~1.3k context, 4 GPUs |

The recommended launch (the ~68 tok/s recipe):

```bash
NCCL_P2P_LEVEL=PXB EXL2_MTP_DRAFT_GRAPH=1 python main.py   # in tabbyAPI-Fork
```

with `gpu_split: [22, 22, 22, 22]`, `cache_size: 32768`, `chunk_size: 2048`,
`draft_num_tokens: 3`, `max_batch_size: 1`. `NCCL_P2P_LEVEL=PXB` enables PCIe peer-to-peer across
the PCI switch on the tested box — **the right value depends on your topology** (not needed on
NVLink). The fork's P2P kernel probes `canAccessPeer` and falls back to NCCL automatically.
**Machines without P2P are supported:** forcing `EXL2_TP_P2P=0` (pure NCCL collectives) runs
the MP path correctly — plain decode and MTP, short and long context, verified coherent
(throughput is lower than the P2P kernel, but it works).

Caveats:

- **Acceptance-dependent.** MTP speed scales with how many drafted tokens the target accepts
  (≈2.7 tokens/round on mixed prose, ≈3.9 on structured prompts, up to 5–6 on arithmetic).
  Predictable content and low temperature run faster.
- **Short context:** at ~512 tokens, MP MTP greedy can run slower than plain MP decode — the
  per-round fixed costs (K/V gather/rewind round trips, verify-window attention) dominate when
  there is little context to amortize them over. This is inherent to the fork's round protocol.
- **SP sampling:** on single-process TP, every sampled deviation from the draft costs a
  rewind-and-recommit (correctness guarantee), so sampled MTP throughput drops toward plain
  decode at short context. **MP** uses probabilistic speculative acceptance instead — accepted
  drafts keep the correct target probability, deviations resample from the target's residual
  distribution — no recommit cost. Every emitted token on both backends comes from the target's
  verified logits: the output distribution is exactly target-autoregressive.

---

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

**MTP speculative decoding** — through the streaming generator (SP) or the MP driver:

```python
from exllamav2.generator import ExLlamaV2StreamingGenerator
from exllamav2.mtp_drafter import Gemma4AssistantDrafter

drafter = Gemma4AssistantDrafter(model, "<gemma-4-assistant-drafter-dir>",
                                 device=0, num_draft_tokens=6)
gen = ExLlamaV2StreamingGenerator(model, cache, tokenizer,
                                  mtp_drafter=drafter, num_speculative_tokens=6)
gen.begin_stream(input_ids, settings, token_healing=True)
```

For MP use `MTPMPDriver` (`exllamav2/mtp_mp.py`); runnable benchmarks and examples are in
[`mtp/`](mtp/). Greedy MTP matches plain autoregressive output except for rare fp16 near-tie
flips between the verify-window and single-token decode attention kernels on long prose.

### GPU count (2 / 3 / 4 validated; 5+ head-count limit)

TP shards the KV heads with a largest-remainder integer split, so **uneven GPU counts work**.
On Gemma-4-31B the two attention geometries split independently (e.g. on **3 GPUs**: sliding
kv 16 → 6/5/5, full-attention kv 4 → 2/1/1). The MP path is validated at **2, 3 and 4 GPUs**
(load + MTP + multi-batch + prompt-cache offload, coherent and corruption-free).

**5+ GPUs are limited by the model's KV-head count:** every rank must hold at least one KV head
on every layer. Gemma-4's full-attention layers have only **4** KV heads, so a fifth rank would
hold zero full-layer heads — there is no KV-head replication, so **5 GPUs is not supported for
Gemma-4**. A model with ≥ `world_size` KV heads on every layer can use 5+.

### VRAM stability under multi-turn prompt-cache load

Collective destination buffers are cached for reuse, and the cache key includes the **row
count**. That is required for CUDA graphs (a captured graph bakes buffer addresses), but a
prompt-cache continuation prefill has a *different* tail length on every turn, so caching
those shapes pinned a new ~0.5 GB entry per turn on every rank until the master OOMed
mid-prefill. Large **eager** shapes (`rows > 32` outside graph mode) therefore use fresh
uncached buffers that the caching allocator recycles — the same rule the per-module GEMM/norm
outputs already follow. Graph mode and decode/verify-sized rows keep the stable-address
cache. Prefill is bandwidth-bound, so it loses nothing from the reuse (measured ~1.6%, noise).

If VRAM still climbs on a long session, `EXL2_VRAM_WATCH=1` + `EXL2_VRAM_CENSUS=1` (tabbyAPI)
prints a live-tensor census with per-category deltas, and `EXL2_VRAM_REFS=1` names the object
holding each growing category.

---

## EXL2.5 quantization format

**EXL2.5** is this fork's next-generation quant format: the same EXL2 file container, plus
per-tensor opt-in upgrades for **speed** (a branchless fast-path GEMM built for Ampere / RTX 3090,
with a 16-row cap so batch-4 MTP verify reads the weights once) and **quality** (Hadamard
incoherence rotation + AWQ channel protection + Hessian-guided rounding at the same
bits-per-weight). Plain EXL2 models keep loading and running exactly as before.

### The one rule to understand (please read)

> **EXL2.5's new features require the EXL2.5 fast path. Mixed-bpw tensors revert to EXL2.**

Spelled out:

- A tensor converted **with** `--exl25_uniform_bits N` is an **EXL2.5 tensor**: one bit-width for
  the whole tensor, uniform groups, served by the new fast-path kernel. Hadamard
  (`--exl25_hadamard`) and AWQ (`--exl25_awq`) **only work on such tensors** — the fast path is
  what applies their activation transforms, and the loader refuses (loudly) to run them anywhere
  else.
- A tensor converted **without** `--exl25_uniform_bits` is a **classic mixed-bpw EXL2 tensor** and
  runs on the **original EXL2 kernel**, byte-for-byte unchanged. Nothing about EXL2.5 slows it
  down or changes its output.
- You can **mix both kinds in one model**. You still choose any model-wide BPW you like: either
  classic per-tensor mixed-bit allocation (EXL2 path), or different uniform bit-widths per
  tensor/type (EXL2.5 path), e.g. attention 6-bit + MLP 4-bit + fp16 head. What you cannot do is
  put Hadamard/AWQ on a mixed-bpw tensor — those require the uniform fast path by design.
- `EXL2_EXL25=0` is an A/B kill switch: it serves **plain uniform** tensors with the legacy kernel
  (same weights, slower) so you can measure the speedup. Hadamard/AWQ tensors are exempt — they
  always use the fast path, because the legacy kernel cannot produce correct results for them.

### Hadamard requires an fp16 head (HB16)

> **With `--exl25_hadamard`, use `-hb 16`.** If you quantize the head (`-hb 6` etc.), the converter
> **skips Hadamard on the lm_head tensor** (prints a warning; the head is quantized unrotated).
> The output head is the most quality-sensitive tensor in the model; keep it fp16 when using
> Hadamard.

### Converting to EXL2.5

```sh
# Uniform 6-bit fast path + Hadamard + AWQ, fp16 head (recommended quality preset)
python convert.py -i <hf_fp16_model> -o <work_dir> -cf <out_dir> \
    -b 6.0 -hb 16 -x25u 6 -x25h -x25a -d 0,1,2,3 -crs on

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

The converted model runs on the normal load path — tags are self-describing in the safetensors,
no config switch needed (TabbyAPI picks it up automatically; the fork's CUDA extension must be
built from this branch). MoE expert tensors get the same treatment automatically (the fused MoE
kernel serves EXL2.5 experts with routing weights). Engram (hashed n-gram) tables have their own
scalar row format + converter (`python -m exllamav2.conversion.convert_ngram`) and module
(`exllamav2.ngram_embedding`), RAM-resident or disk-streaming — no trellis, Ampere-friendly.

**Conversion notes:**

- Use a **fresh work directory** for each job (`-o`, or `-nr` to clear an existing one). Do not
  resume a job across converter code changes — the calibration trajectory in the work dir must
  match the code that produced it.
- Several conversions can run **in parallel on separate GPUs from one venv** — see
  [doc/convert.md](doc/convert.md#running-several-conversions-in-parallel).
- After conversion, check the final `calibration perplexity (quant)` line: single/low-double
  digits is healthy; thousands or more means the conversion failed (see doc/convert.md).

Validation harnesses (developer): `mtp/exl25_gemm_test.py` (parity m=1..16 + cuBLAS path, quality
A/B, speed A/B vs legacy, CUDA-graph capture/replay), `mtp/exl25_moe_test.py` (fused MoE parity),
`mtp/exl25_ngram_test.py` (engram converter/kernel/hashing parity).

### EXL2.5 validation status (2026-11-22, RTX 3090 / sm86)

- **Kernel correctness: ALL PASS** — GEMM parity m=1..16 + prefill cuBLAS+transform path (fp32
  oracle, fp16 noise floor), CUDA-graph capture/replay m=1/4/16, fused MoE with EXL2.5 experts
  (plain + Hadamard/AWQ, rows 1/2/4), engram converter/kernel/hashing (RAM + disk-streaming,
  4/8-bit).
- **Real-weight verification:** compiled-checkpoint tensors reconstructed through the runtime
  kernels match the fp16 base at normal 6-bit quant error (rel_l2 ≈ 0.03–0.08), on both the fast
  lane and the prefill path, with real-magnitude calibration activations.
- **Kernel numerics + speed (2026-11-23):** the Hadamard activation rotation accumulates in
  fp32 (fp16 partial sums overflow on large-magnitude activations). The fast-lane dot uses
  fast `hfma2` partials with an **exact power-of-two overflow guard**: activations are
  pre-scaled by 2⁻¹² and the group scale carries 2¹², so a 16-term partial of raw codes is
  bounded ≤ 32768 for *any* fp16 activation — no overflow, no fp32-FMA cost. Parity ALL PASS,
  real-activation canary clean. Same-session greedy A/B on the test rig (4× RTX 3090, 200 W,
  MP + MTP W=3): **EXL2.5 vs EXL2-6.0 ≈ +19% total tok/s at 100k context (70.9 vs 59.4) and
  +13% at 1.3k (96.6 vs 85.7)**; verify round phase 31.6 vs 35.3 ms (100k), 23.2 vs 26.1 ms
  (1.3k). Acceptance is also higher at equal bpw (Hadamard/AWQ quality).
- **Load-path fixes:** the TP loaders retain the AWQ scale for the QMatrix's lifetime
  (user-confirmed coherent on MP + MTP); the o_proj prefill shortcut and the GPTQ loaders
  pass the full EXL2.5-era `make_q_matrix` signature (plain EXL2 / GPTQ MP loads crashed
  with a TypeError before this).
- **Experimental hd512 verify kernel (default OFF):** `EXL2_VERIFY_HD512_SPLIT=1` enables a
  rewritten split-KV small-q GQA attention kernel (register-resident GQA group, cp.async
  double-buffered KV; parity ALL PASS vs fp32 at kv up to 79k). Measured ~60 GB/s — it
  matches FlashInfer's *ragged* prefill but loses to the *paged* prefill the live verify
  path uses (verify 64.9 vs 31.6 ms at 100k), so it stays opt-in/dormant pending a better
  small-q design.
- **Converter fixes:** the Hessian is Hadamard-rotated exactly once (a double rotation had
  cancelled itself, leaving GPTQ error compensation in the wrong space), and tensors that share
  a Hessian (q/k/v, gate/up) now share the AWQ protection scale too. A module-by-module replay
  of the quant pass tracks the fp16 trajectory at cosine ≥ 0.99999 with both single- and
  multi-GPU quantization; per-module reconstruction error improved ~3× vs the original run.
- **Quality (same bpw, synthetic outlier stress test):** Hadamard+AWQ cuts quantization rel_l2
  ~30–45% (e.g. 6-bit 0.0236 → 0.0133; 4-bit 0.095 → 0.054).
- **Kernel speed vs legacy (plain uniform, identical weights, 200 W, same session):** decode
  m=1 **≈1.0–1.28×**, m=4 up to **2.27×** (6-bit down); m=8 mixed by shape (0.5–2.8×); m=16
  still **0.5–0.8×** (occupancy/register tuning is the next lever; the 16-row one-pass design
  is in place, graph capture/replay PASS). Hadamard/AWQ add an in-kernel butterfly cost on
  top (quality feature); at m=4 the transform amortizes and had+awq ≈ plain.
- **TP loader fix (user-confirmed 2026-11-23):** the tensor-parallel loaders stored the AWQ
  scale only in a local variable while the QMatrix kept a raw device pointer to it — the
  scale was freed on return and the GEMMs read recycled memory (gibberish, zero MTP
  acceptance on the MP backend). All five TP construction paths now retain the exact scale
  tensor for the handle's lifetime. A freshly converted checkpoint runs **coherent on
  MP + MTP** after this fix.
- **First checkpoint invalid:** the first EXL2.5 conversion was interrupted by a converter bug
  and resumed; its calibration trajectory was corrupted (composed perplexity ~1e18 while every
  per-module check stayed small), so its weights were rounded against garbage activations.
  Always convert with a **fresh work dir** using the fixed converter. Throughput A/B numbers
  vs EXL2-6.0 are still being measured. EXL2.5 is opt-in per tensor; existing EXL2 models and
  the legacy kernel are untouched.

---

## Experimental features

All three are opt-in on the **MP + MTP** path and validated on the test rig; the default
(batch size 1, no offload, MTP drafting) is the verified single-job path. Configuration details
live in the [TabbyAPI fork docs](https://github.com/Ready-Art/tabbyAPI-Fork/blob/ReadyArt/docs/11.-ReadyArt-ExLlamaV2-Fork.md).

### Qwen3.8-27B / Qwen3.5 hybrid backbone (EXPERIMENTAL — not yet rig-validated)

Support for **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** (architecture
`Qwen3_5ForConditionalGeneration`, text-only). This is a **hybrid linear-attention backbone**,
structurally different from Gemma-4:

- **64 layers**: 3× *gated delta net* (linear attention) then 1× output-gated full attention,
  repeating (16 full-attention layers). The delta-net layers carry **no K/V cache** — a depthwise
  short-conv state (last 3 raw inputs/channel) plus a fp32 recurrent state per value head
  (`ExLlamaV2GatedDeltaNet`, `exllamav2/gdn.py`). The delta-rule scan is pure torch (chunked
  prefill + per-token decode), ported from the HF Qwen3.5/Qwen3-Next reference and cross-checked
  bit-exact against it; **no Triton, no trellis**. A CUDA C++ decode kernel is the follow-up
  perf lever.
- **Full-attention layers**: output-gated q (`q_proj` 2× wide, sigmoid gate), per-head q/k RMSNorm
  (zero-centered `1+w`), partial NEOX RoPE (64 of 256 dims, theta 1e7).
- **Hybrid cache** (`ExLlamaV2Cache_Hybrid`): FP16 K/V on full layers + conv/recurrent state on
  linear layers, with a one-deep backup for single-token rewinds (token healing) and verify
  staging for speculative rejection.
- **TP/MP**: the delta net shards by value-head groups (the recurrence is per-head independent, so
  ranks never exchange recurrent state); q/k shard by key heads, `out_proj` is row-parallel. Both
  the single-process (`sp`) and multiprocess (`mp`) backends are wired; CUDA graphs are disabled
  for hybrid (eager delta-net ops). KV-head count must be ≥ world size on the full layers (4 KV
  heads → up to 4 GPUs).
- **Built-in MTP** (`exllamav2/mtp_qwen35.py`, experimental): the drafter is a single decoder layer
  shipped inside the checkpoint (`mtp.*`), borrowing the target's embedding/head. Enable with
  `draft_model.draft_mode: mtp` and **no** `draft_model_name`. **Greedy decoding only**,
  single-GPU/autosplit only for now (MP fan-out of the verify staging is a follow-up). The
  converter copies `mtp.*` through unquantized (fp16).
- **Converter**: `convert_exl2.py` handles the delta-net projections (`in_proj_q/k/v/z`, `out_proj`
  quantized; `in_proj_b/a` fp16), the raw conv/`A_log`/`dt_bias`/norm tensors, and the `mtp.*`
  passthrough. EXL2.5 (uniform bits, no trellis) is the intended quant format. Vision tower
  (`model.visual.*`) is skipped, same policy as Gemma-4.

**Status:** implemented and syntax/parity-checked (delta-rule math bit-exact vs HF; architecture
recognition tested), but **not yet run on the GPU rig** — treat output as unverified until a
short-context coherence test passes. Quantize with `convert_exl2.py` (e.g. EXL2.5 uniform 4-bit,
fp16 head) and load through the TabbyAPI fork.

### Multi-batch (MP + MTP)

`TPProcessGroup(..., max_batch=N)` + `MTPMPBatchDriver` run several jobs concurrently, one cache
row (slot) per job. Validated under true concurrent load at **batch 4** (mixed context lengths,
greedy and sampled). Highlights:

- Batched graphed verify rounds (`(B, W+1)` rows, one CUDA graph per batch size) with per-row
  acceptance/rewind/sampling; every round runs at the full active batch size (burst-drain
  scheduling — staggered admissions cannot desync rows into solo rounds).
- **Per-slot append-only prompt cache** (strict-prefix continuations, prompt-boundary ring
  restore, partial-prefix reuse for edited turns) plus **cross-slot shared-prefix KV cloning**
  (physical copies — slots never share live K/V, so conversations cannot cross-contaminate).
- Batched drafter with whole-window CUDA graphs (`EXL2_MTP_DRAFT_GRAPH=1`), sparse draft head,
  threaded acceptance (same kernels, same draw order, same distribution).
- VRAM scales per slot (~5 GB/row at 160k context on 4 GPUs; ~1.2 GB/row at 32k). Batch 4 fits
  24 GB cards at `cache_size: 32768` on 4 GPUs; on 2 GPUs halve the per-slot budget again.
- Through TabbyAPI: `model.max_batch_size`. Measured ~85–110 tok/s aggregate across 4 concurrent
  essay streams at temp 0.7 (acceptance-dependent).

### Prompt-cache RAM offload (MP + MTP)

Tier idle prompt-cache lineages to host RAM so the VRAM context budget is not a hard cap on how
many conversations stay cached (e.g. when sharing the API). Each rank stores its own K/V slice
worker-locally (K/V never crosses the process pipe); restores are byte-identical and run between
rounds — **no VRAM allocation, no graph disturbance**. The RAM cap is **global to the whole
server** (master + all workers), enforced by the scheduler's cold store with LRU eviction.
Thought-stripped and long conversations restore their document via the prompt-boundary ring
snapshot. Failure-soft: a failed restore drops the entry and re-prefills.

Through TabbyAPI: `model.prompt_cache_offload: true` + `model.prompt_cache_ram_limit_mb`.
Kill switch `EXL2_PROMPT_OFFLOAD=0`; decision logging `EXL2_OFFLOAD_DEBUG=1`. `cache_size` stays
the per-row VRAM depth — RAM extends the *number* of cached lineages, not a single
conversation's attendable depth.

### INT8 KV cache — PRIMARY storage (opt-in, default fp16)

Ampere has no FP8, so K/V are quantized to symmetric per-(token, head) **INT8 with fp16
scales** and read by custom split-KV attention kernels (GQA, graph-safe, per-row kv_lens +
row_map): **decode** uses a multi-warp CUDA-core kernel (`int8_attn_fast`), and **MTP
verify** on hd512 windows whose per-kv-head query count is 16/32 (Gemma-4 W=1/W=3) uses an
**FA2-style pipelined tensor-core kernel** (`int8_attn_fa2`, `mma.m16n8k16` + `cp.async`);
other verify geometries fall back to the warp kernel.

The FA2 kernel exists because both earlier INT8 verify kernels lost to FlashInfer's fp16
paged prefill at long context — halving the KV bytes does not help when the kernel is
compute-bound per key. The warp kernel is CUDA-core bound (~35 GB/s effective); the HMMA
kernel (`int8_attn_hmma`, `nvcuda::wmma` m16n16k16, HD split across warps) ran QK^T on only
the `hs==0` warp of each query tile, so 6 of 8 warps idled through QK and the phase was ~4x
longer than the PV that followed. `int8_attn_fa2` balances both phases: every warp does QK
for its 8-key sub-tile **and** PV for its 128-column HD slice, K/V stream through
double-buffered `cp.async` INT8 smem tiles and are dequantized to half in registers at
fragment-load time (no half smem tiles — that is what fits hd512 in sm86's 101376 B block
limit), and the cross-warp online softmax reduces only per-row m/l partials, never the full
score tile. `sm_scale*log2(e)` is applied to the fp32 accumulator, not pre-scaled into Q.

Measured same-session on 4x3090, live per-rank geometry (nq8/kvh1, q_len 4), against the
**live** FI fp16 paged-prefill path (not ragged): at kv=100k **0.544 ms vs FI 0.796 ms
(1.46x)**, vs HMMA 2.440 ms (4.49x) and the warp kernel 2.893 ms (5.32x); parity cos
0.99997 including batched row_map. The crossover matters: at kv=25.6k it is still 0.69x FI,
and at 4k roughly parity — the win is in the long-context regime where the KV read dominates.

Kill switches / A-B: `EXL2_INT8_FA2=0` falls back to HMMA, `EXL2_INT8_HMMA=0` then falls back
to the warp kernel. Harness: `mtp/int8_fa2_test.py`.
INT8 is the **only** full-depth KV storage on the
full-attention layers: **no fp16 shadow alongside**, so KV VRAM is **halved (~−50%)** and
the decode/verify KV read is halved with it. fp16 full-depth buffers exist only as
**transient prefill staging** (prefill kernels read fp16): allocated per prefill, each chunk
quantized into the INT8 primary as it lands, freed when prefill ends; continuation prefills
dequantize the INT8 prefix back into staging. SWA rings stay fp16. With MTP, the drafter's
joined KV shadow goes INT8-primary automatically and the gather pipes INT8 directly (half
the pipe bytes). Batch-safe (per-row kv_lens + row_map); clone/offload/restore move INT8.
Full layers with head_dim 256/512; logit-softcap models stay fp16.
Through TabbyAPI: `model.kv_cache_int8: true` (sets `EXL2_KV_INT8`). Default off.

### INT8 MTP drafter (opt-in, default fp16)

Two independent gates for the MTP assistant drafter:

- **`EXL2_MTP_INT8_W`** — weight-only INT8 GEMVs (per-output-row symmetric INT8 + fp16
  scales, custom warp-per-row CUDA kernel; no Triton). Halves the drafter's weight-read
  bandwidth. Kernel parity rel_L2 ≈ 4e-4 vs the dequantized reference; greedy draft ids
  unchanged in parity tests. Measured at 25k ctx (4×3090, 200 W, temp 1, W=3): head GEMV
  1.66× faster, draft phase −31%, acceptance-normalized generation **+12%**.
- **`EXL2_MTP_INT8_KV`** — **INT8-primary** gathered K/V shadow for the full-attention
  layers: the full-depth fp16 shadow is *replaced* by INT8 + fp16 scales (**~−40% shadow
  VRAM**), gather deltas quantize through a small staging buffer, and clone / RAM-offload
  / restore move INT8. The drafter's cross-attention reads it with the split-KV INT8
  flash-decode kernel (graph-safe, per-row device kv_lens, logical→physical row map for
  batched slot subsets). SWA rings stay fp16 (small). At 25k the INT8 attention kernel is
  currently a touch slower than FlashInfer fp16 (−3% generation; kernel tuning pending) —
  the long-context payoff is the halved KV read. **Both gates together: +9.5%**
  acceptance-normalized at 25k.

Through TabbyAPI: `draft_model.mtp_int8_weights: true` / `draft_model.mtp_int8_kv: true`.
Both default off; fp16 paths are unchanged when disabled.

### MoE expert CPU/RAM offload (opt-in, by expert count)

`EXL2_MOE_OFFLOAD=N` (TabbyAPI: `model.moe_offload_experts: N`) keeps the **last N
experts of every MoE layer in pinned host RAM**; `N >= num_experts` offloads them all.
One GPU staging slot per projection is **shared** across the offloaded experts of a
layer (all same-shape), H2D-filled on demand when routing selects a cold expert — so
GPU holds one expert's data, not N. Small per-expert constants (permutations, scale
metadata) stay GPU-resident; the `q_handle` is built against the staging slot once.
Offload forces the per-expert torch loop (the fused `q_moe_mlp` kernel needs every
expert resident): slower decode, much lower VRAM — the trade-off for fitting large MoE
models on small cards. Default 0 = off.

**MP tensor-parallel:** MoE runs on the MP path with **replicated experts** (every rank
holds the full MoE and runs it identically; the residual is gathered/sharded around the
block). Offload moves cold experts to pinned RAM on *each* rank. Resident experts use the
graph-safe fused kernel (TP graphs stay on); offloaded MoE runs eager because its
data-dependent routing + H2D staging cannot be baked into a replayed graph. Verified on a
synthetic Qwen3MoE across 4 GPUs (N=0/4/8 greedy ids identical). Expert column/row TP
*sharding* (less VRAM than replication) is future work.

**Multi-model MP:** the fused TP attention kernel only covers head_dim 256/512 without
qk_norm; other families (Qwen3/Qwen3MoE with qk_norm, Llama head_dim 128, …) automatically
take the generic eager TP attention path (FlashInfer/SDPA, qk_norm-aware). KV-head count
must be ≥ the TP world size (a rank owning zero KV heads is unsupported, as for Gemma-4).
The `head_norm` kernel now supports head_dim < 64 (Qwen3's head_dim 32): it pads the
reduction to a full warp so the shuffle sum is correct (previously half-warp launches
corrupted qk_norm). Verified vs torch RMSNorm for hd 32/64/128/256/512.

**Expert TP sharding (EXPERIMENTAL, opt-in):** `EXL2_MOE_TP_SHARD=1` (default 0 =
replicate) splits each expert's FFN across ranks (column-parallel w1/w3, row-parallel w2)
like a dense MLP — ~world× less expert VRAM/RAM per rank and ~world× less offload PCIe
traffic; works on 2/3/4 GPUs (uneven splits). **Caveat:** a Qwen3-family MP attention
parity gap (prefill logits cos ~0.83 vs single-device, present in replicated MP too) is
still open, so sharded Qwen3 output is not yet verified coherent — keep it opt-in until
that attention bug is fixed (see `newplan.md`). Gemma-4 (the tested family) is unaffected.

### INT8 & offload — quick configuration

All INT8/offload features are **opt-in, default off**; the fp16 paths are byte-identical
when disabled. Each is independent — enable any combination.

| Feature | TabbyAPI option | Env var | Effect | VRAM |
|---|---|---|---|---|
| Target KV INT8 **primary** | `model.kv_cache_int8: true` | `EXL2_KV_INT8=1` | decode + verify read INT8 (full layers) | **−~50% KV** (no fp16 shadow) |
| MTP drafter INT8 **weights** | `draft_model.mtp_int8_weights: true` | `EXL2_MTP_INT8_W=1` | +12% gen @25k | ~neutral |
| MTP KV INT8 **primary** | `draft_model.mtp_int8_kv: true` | `EXL2_MTP_INT8_KV=1` | halved long-ctx draft KV read | **−~40% MTP shadow** |
| MoE expert offload | `model.moe_offload_experts: N` | `EXL2_MOE_OFFLOAD=N` | fit big MoE on small cards | experts → host RAM |
| MoE expert TP sharding (experimental) | — | `EXL2_MOE_TP_SHARD=1` | split experts across ranks | ~world× less expert VRAM/RAM |

Example `config.yml` (MP + MTP, all INT8 features on, MoE offload for a MoE model):

```yaml
model:
  model_name: Gemma-4-31B-IT-EXL25-6U-HAD-AWQ-HB16
  backend: exllamav2
  tensor_parallel: true
  tensor_parallel_backend: mp
  gpu_split: [22, 22, 22, 22]
  cache_size: 160000
  kv_cache_int8: true          # target INT8-primary KV (decode + verify; −KV VRAM)
  # moe_offload_experts: 8     # for MoE models: last 8 experts/layer -> pinned RAM
draft_model:
  draft_mode: mtp
  draft_model_name: gemma-4-31B-it-assistant
  draft_num_tokens: 3
  mtp_draft_graph: true
  mtp_int8_weights: true       # drafter INT8 GEMVs (+12% gen)
  mtp_int8_kv: true            # drafter INT8-primary KV shadow (-40% shadow VRAM)
```

Or purely via environment (no TabbyAPI): `EXL2_KV_INT8=1 EXL2_MTP_INT8_W=1
EXL2_MTP_INT8_KV=1 EXL2_MOE_OFFLOAD=8`.

**Which INT8 KV mode to pick:** the *target* shadow (`kv_cache_int8`) is **speed-only** —
it keeps fp16 primary because FlashInfer prefill/verify still read fp16, so it *adds*
~50% KV VRAM; use it when you want faster base decode and have VRAM headroom. The *MTP*
shadow (`mtp_int8_kv`) is **INT8-primary** — it drops the fp16 copy and *reduces* VRAM
~40%; use it to save drafter VRAM (small short-ctx speed cost until the INT8 attention
kernel is tuned). Making the **target** cache INT8-primary (VRAM-reducing) requires a
competitive INT8 verify/prefill attention kernel and is tracked as future work — see
`newplan.md` (P10b).

### Priority prompt scheduler (opt-in, MP + MTP multi-batch)

Shortest-prompt-first admission with a small micro-batch window, so short requests are not
stuck behind multi-second long prefills. Includes a **reprocess-abuse penalty**: long prompts
that repeatedly FULLY re-prefill from scratch (tracked by prompt fingerprint — prefix hash +
length bucket, no API-key tracking) are deprioritized and held briefly; re-rendered last
replies take the partial-reuse paths and never count. Aging bounds starvation. Through
TabbyAPI: `model.priority_queue: true` (+ `batch_window_ms`, `reprocess_*` options). Default
off. Mid-prefill preemption (`priority_preempt`, **default off**, opt-in) admits short jobs
between the chunks of a running long prefill; the earlier slot-reuse crash chain is fixed
(a slot is free only when it holds no job) and verified under contention.

### Engram (n-gram) table storage modes

For models with hashed n-gram embedding tables (e.g. Qwen3.8-Flash-Next): `EXL2_NGRAM_MODE` =
`disk` (pread per lookup from SSD/NVMe, default), `ram` (full table in host RAM), or `hybrid`
(disk + bounded LRU row cache, `EXL2_NGRAM_HYBRID_MB`, default 1024). Never VRAM-resident.
Through TabbyAPI: `model.ngram_mode` / `model.ngram_hybrid_mb`.

### DFlash block-diffusion drafting (MP, batch 1)

[DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) (z-lab,
[paper](https://arxiv.org/abs/2602.06036)) denoises a whole block — one anchor token plus up to
15 mask tokens — in **one parallel forward**, conditioned on the target's residual streams at
six layers. Draft cost is ~flat in window width, which is where DFlash wins over MTP at large
windows. Same verify/accept protocol as MP MTP (every emitted token comes from the target's
verified distribution). First end-to-end runs verified (~51 tok/s at W=3); long-context and
tool-loop validation in progress.

Through TabbyAPI: `draft_model.draft_mode: dflash` (auto-detected for DFlash checkpoints under
`mtp` mode). The drafter needs ~6.6 GB extra on its device at 32k cache (~9.2 GB at 160k) — on
2-GPU splits set `draft_device` to a worker GPU. exllamav2 side: `DFlashDrafter`
(`exllamav2/dflash_drafter.py`); greedy parity harness `mtp/dflash_test.py`.

---

## Gemma-4-31B-IT

`Gemma4ForConditionalGeneration` is supported for **text only** (vision tensors are skipped;
passing an image raises a clear error). The model mixes two attention geometries, which is why
several internals are per-layer:

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

**Untied-head finetunes are supported:** the head loader prefers a distinct `lm_head.weight`
over the tied `embed_tokens` fallback, and head shards stored with a bare `lm_head.weight` key
(some community finetunes) are remapped automatically. Budget ~2.8 GB extra VRAM on the
head-owning rank for an fp16 untied head.

MTP speculative decoding is supported on both TP and MP — the assistant drafter cross-attends to
the target's K/V, which the TP/MP cache layers gather onto the drafter device.

---

## OpenAI-compatible server (TabbyAPI)

**[ReadyArt/tabbyAPI-Fork](https://github.com/Ready-Art/tabbyAPI-Fork)** (branch `ReadyArt`)
integrates this fork into [TabbyAPI](https://github.com/theroyallab/tabbyAPI), so Gemma-4-31B-IT
runs behind an OpenAI-compatible API with:

- `tensor_parallel_backend: sp` (single-process TP) or `mp` (one worker per GPU),
- MTP drafting via `draft_model.draft_mode: mtp` + the
  [gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant) drafter
  (full sampler on both SP and MP),
- TP CUDA graphs (`tp_graphs`) and FlashInfer controls,
- Gemma-4 **thinking** (thought-channel `reasoning_content`) and **tool calls** (`gemma4` tool
  format), plus `reasoning_effort` on chat completions,
- the experimental features above (multi-batch, prompt-cache RAM offload, DFlash).

See the TabbyAPI repo's
[docs/11.-ReadyArt-ExLlamaV2-Fork.md](https://github.com/Ready-Art/tabbyAPI-Fork/blob/ReadyArt/docs/11.-ReadyArt-ExLlamaV2-Fork.md)
for configuration and limitations.

## Conversion & quantization

Quantize to EXL2 or EXL2.5 with `convert.py`; spread the work across GPUs with `-d` (device list)
and `-crs on` (shard calibration rows), or run several conversions in parallel on separate GPUs:

```sh
python convert.py -i <hf_fp16_model> -o <work_dir> -cf <out_dir> \
    -b 6.0 -hb 16 -d 0,1,2,3 -crs on
```

See **[doc/convert.md](doc/convert.md)** for all arguments, the multi-GPU process, parallel
conversions, hardware requirements, and how to run the converted model tensor-parallel.

## Upstream features

The dynamic generator (batching, smart prompt caching, K/V dedup, simplified API), sampling
filters, Q4 cache, and the rest of upstream ExLlamaV2 are retained. See
[doc/dynamic.md](doc/dynamic.md) and the [examples/](examples/) directory. Upstream docs:
[ExLlamaV2 wiki](https://github.com/turboderp/exllamav2/wiki).

## Feedback & contributions

- **Want a change?** Open a [GitHub issue](https://github.com/Ready-Art/exllamav2-Fork/issues) —
  bug reports, feature requests, and performance observations are all welcome. For perf reports,
  include your hardware/topology and the numbers you measured.
- **Pull requests are welcome.** Open a [PR](https://github.com/Ready-Art/exllamav2-Fork/pulls)
  against the `ReadyArt` branch with a clear description of what you're changing and why.

## License

MIT, as upstream ExLlamaV2. See [LICENSE](LICENSE). This fork is not affiliated with the original
ExLlamaV2 authors; all original credit belongs to them.
