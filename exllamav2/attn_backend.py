"""
Attention backend abstraction for exllamav2.

This module introduces a pluggable attention backend so that FlashInfer can replace
flash-attn-2 as the paged/prefill kernel, while keeping a pure-torch reference path for
shapes FlashInfer cannot serve.

Design notes (see PLAN.md §3):

* exl2's paged cache is already laid out as ``[num_pages, page_size, num_kv_heads, head_dim]``
  after the ``.view(...)`` in ``attn.py:forward_paged``. That is exactly FlashInfer's ``NHD``
  layout, so no transpose is needed.
* FlashInfer takes ``paged_kv_indptr`` / ``paged_kv_indices`` / ``paged_kv_last_page_len``
  instead of a rectangular ``block_table``. The conversion is on the hot path and is cached
  per ``(batch, pages_per_seq)`` shape.
* head_dim 512 (Gemma-4's global attention layers) IS served by FlashInfer >= 0.7.4 on
  sm86 (Ampere, RTX 3090): ``BatchDecodeWithPagedKVCacheWrapper`` and
  ``single_decode_with_kv_cache`` both accept hd512 plans/kernels (verified cos 1.0 vs
  torch). The old "512 -> MLA, refused" claim was wrong for sm86 -- the head_dim > 256
  rejection in FlashInfer's decode.py is a Hopper/Blackwell-only XQA-MTP path.
* ``indptr`` and ``indices`` must be ``int32``. int64 causes indexing errors.
"""

from __future__ import annotations

import os

import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------------------------

# Change98: FlashInfer >= 0.7.4 is a HARD, non-optional requirement of this fork.
# It serves head_dim 512 decode on sm86 (RTX 3090) via single_decode_with_kv_cache and
# the paged decode wrapper (verified empirically on 0.7.4). The old "optional extra"
# soft-import and the EXLLAMA_NO_FLASHINFER bypass are removed: importing exllamav2
# now fails fast if FlashInfer is missing or too old. Verify against the INSTALLED
# version, never against stale repo comments.
FLASHINFER_MIN_VERSION = (0, 7, 4)
flashinfer_error: str | None = None


def _flashinfer_version_tuple(v: str) -> tuple:
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except Exception:
        return (0,)


try:
    import flashinfer
except Exception as e:  # ModuleNotFoundError, or a broken CUDA/kernel install
    raise ImportError(
        "exllamav2-fork requires FlashInfer >= 0.7.4 (hard dependency, not optional). "
        "Install: pip install 'flashinfer-python>=0.7.4' && flashinfer download-kernels. "
        f"Import failed: {type(e).__name__}: {e}"
    ) from e

if _flashinfer_version_tuple(flashinfer.__version__) < FLASHINFER_MIN_VERSION:
    raise ImportError(
        f"exllamav2-fork requires FlashInfer >= 0.7.4 (installed {flashinfer.__version__}). "
        "Upgrade: pip install -U 'flashinfer-python>=0.7.4' && flashinfer download-kernels"
    )

has_flashinfer = True


# Head dims served by FlashInfer's GQA kernels on sm86 (verified on 0.7.4, RTX 3090):
# 64/128/256 plus 512. hd512 decode works via BatchDecodeWithPagedKVCacheWrapper and
# single_decode_with_kv_cache (cos 1.0 vs torch); the old "512 -> MLA, refused" claim
# was wrong for sm86 (the head_dim>256 rejection in FlashInfer's decode.py is a
# Hopper/Blackwell-only XQA-MTP path). Verify against the INSTALLED version, never
# against stale repo comments.
FLASHINFER_GQA_HEAD_DIMS = (64, 128, 256, 512)

# Page sizes FlashInfer accepts at all: page_size % 8 == 0 and
# (128 % page_size == 0 or page_size % 128 == 0). The *fast path* additionally prefers
# 16/32/64/128, which is why PAGED_PAGE_SIZE is 128 (PLAN.md §3.6).
FLASHINFER_FAST_PAGE_SIZES = (16, 32, 64, 128)


def flashinfer_supports_page_size(page_size: int) -> bool:
    return page_size % 8 == 0 and (128 % page_size == 0 or page_size % 128 == 0)


# Plan caching is off by default: correctness first. See FlashInferPaged.plan_and_run.
_PLAN_CACHE_ENABLED = os.environ.get("EXLLAMA_FLASHINFER_PLAN_CACHE", "0") == "1"


# ---------------------------------------------------------------------------------------------
# block_table -> indptr/indices conversion
# ---------------------------------------------------------------------------------------------

def page_table_to_indptr(
    block_index: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    out_indptr: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    out_last_page_len: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert exl2's rectangular page table into FlashInfer's ragged representation.

    :param block_index:
        ``[batch, pages_per_seq]`` int tensor, physical page index per logical page.

    :param cache_seqlens:
        ``[batch]`` int tensor, number of valid KV positions per sequence (including the
        tokens being added in this call).

    :param page_size:
        Tokens per page.

    :return:
        ``(paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len)``, all int32 and on the
        same device as ``block_index``.
    """

    device = block_index.device
    batch = block_index.shape[0]

    pages_per_seq = torch.div(cache_seqlens + page_size - 1, page_size, rounding_mode = "floor")
    pages_per_seq = pages_per_seq.to(torch.int32)

    # indptr: [batch + 1], cumulative page counts
    if out_indptr is not None and out_indptr.numel() >= batch + 1:
        indptr = out_indptr[:batch + 1]
        indptr[0] = 0
        torch.cumsum(pages_per_seq, dim = 0, out = indptr[1:])
    else:
        indptr = torch.zeros(batch + 1, dtype = torch.int32, device = device)
        torch.cumsum(pages_per_seq, dim = 0, out = indptr[1:])

    total_pages = int(indptr[-1].item())

    # indices: flatten the valid prefix of each row of block_index
    if out_indices is not None and out_indices.numel() >= total_pages:
        indices = out_indices[:total_pages]
    else:
        indices = torch.empty(total_pages, dtype = torch.int32, device = device)

    if total_pages > 0:
        ar = torch.arange(block_index.shape[1], device = device, dtype = torch.int32)
        mask = ar.unsqueeze(0) < pages_per_seq.unsqueeze(1)
        indices.copy_(block_index.to(torch.int32)[mask])

    # last_page_len: 1..page_size
    last_page_len = cache_seqlens - (pages_per_seq - 1) * page_size
    last_page_len = last_page_len.to(torch.int32).clamp_(1, page_size)

    if out_last_page_len is not None and out_last_page_len.numel() >= batch:
        out_last_page_len[:batch].copy_(last_page_len)
        last_page_len = out_last_page_len[:batch]

    return indptr, indices, last_page_len


# ---------------------------------------------------------------------------------------------
# Torch / SDPA reference path (also the fallback for unsupported shapes)
# ---------------------------------------------------------------------------------------------

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match Q heads using per-group ``expand()`` views.

    Deliberately *not* ``F.scaled_dot_product_attention(..., enable_gqa = True)``: the cutlass
    memory-efficient SDPA kernel rejects ``enable_gqa`` and silently falls back to the ~5x slower
    math kernel on Ampere. See PLAN.md §8 trap 18 and
    ``exllamav3/modules/attention_fn/torch.py:222-232``.
    """
    if n_rep == 1:
        return x
    batch, heads, slen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, heads, n_rep, slen, head_dim)
    return x.reshape(batch, heads * n_rep, slen, head_dim)


def torch_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_index: torch.Tensor,
    page_size: int,
    sm_scale: float,
    window_left: int = -1,
    logits_soft_cap: float | None = None,
) -> torch.Tensor:
    """
    Paged attention implemented with plain torch matmul. Reference implementation and fallback.

    :param q:
        ``[batch, q_len, num_q_heads, head_dim]``

    :param k_cache / v_cache:
        ``[num_pages, page_size, num_kv_heads, head_dim]``

    :param cache_seqlens:
        ``[batch]`` total KV length per sequence, *including* the q_len new tokens.

    :param block_index:
        ``[batch, pages_per_seq]`` physical page indices.

    :return:
        ``[batch, q_len, num_q_heads, head_dim]``
    """

    batch, q_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    n_rep = num_q_heads // num_kv_heads
    device = q.device

    k_flat = k_cache.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_cache.reshape(-1, num_kv_heads, head_dim)

    slot_offsets = torch.arange(page_size, device = device)

    outputs = []
    for b in range(batch):

        total_len = int(cache_seqlens[b].item())
        n_pages = (total_len + page_size - 1) // page_size
        pages = block_index[b, :n_pages].long()

        gather_idx = (pages.unsqueeze(1) * page_size + slot_offsets.unsqueeze(0)).reshape(-1)
        k = k_flat.index_select(0, gather_idx)[:total_len]
        v = v_flat.index_select(0, gather_idx)[:total_len]

        # [heads, seq, dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        qb = q[b].transpose(0, 1)

        # Sliding window: drop the oldest KV that can never be attended to.
        kv_start = 0
        if window_left >= 0:
            kv_start = max(0, total_len - q_len - window_left)
            if kv_start:
                k = k[:, kv_start:, :]
                v = v[:, kv_start:, :]

        k = repeat_kv(k.unsqueeze(0), n_rep).squeeze(0)
        v = repeat_kv(v.unsqueeze(0), n_rep).squeeze(0)

        # Lower-right aligned causal mask: query i (absolute position
        # total_len - q_len + i) may attend to keys at absolute positions <= that.
        kv_len = k.shape[1]
        qi = torch.arange(q_len, device = device).unsqueeze(1) + (total_len - q_len)
        ki = torch.arange(kv_len, device = device).unsqueeze(0) + kv_start
        # [q_len, kv_len], NOT [1, 1, q_len, kv_len].
        #
        # The batch loop above is already unrolled, so `scores` below is 3-D
        # [num_q_heads, q_len, kv_len] with no batch axis. A 4-D mask broadcasts against it by
        # *prepending* a dimension, silently producing [1, heads, q_len, kv_len] and therefore a
        # 5-D output. That is not a shape error the caller can ignore: the returned tensor came
        # back as [batch, heads, 1, q_len, head_dim] instead of [batch, q_len, heads, head_dim].
        # This path is the reference implementation AND the fallback for Gemma-4's head_dim-512
        # layers, so the corruption was live on every global-attention layer.
        mask = torch.where(ki <= qi, 0.0, float("-inf")).to(torch.float32)  # [q_len, kv_len]

        scores = torch.matmul(qb.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * sm_scale
        if logits_soft_cap:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask
        probs = torch.softmax(scores, dim = -1).to(q.dtype)
        out = torch.matmul(probs, v)

        outputs.append(out.transpose(0, 1))

    return torch.stack(outputs, dim = 0)


# ---------------------------------------------------------------------------------------------
# FlashInfer backend
# ---------------------------------------------------------------------------------------------

class FlashInferPaged:
    """
    Per-device FlashInfer paged-attention wrapper with a shape-keyed plan cache.

    One instance is created per ``ExLlamaV2Attention`` module. The 128 MB workspace buffer is
    shared per device via ``_workspace_cache`` so we do not allocate one per layer.
    """

    # 128 MB, the size FlashInfer's docs recommend for the plan workspace.
    WORKSPACE_BYTES = 128 * 1024 * 1024

    _workspace_cache: dict = {}

    def __init__(self):

        self.prefill_wrapper = None
        self.decode_wrapper = None

        # Persistent int32 buffers, grown on demand
        self._indptr = None
        self._indices = None
        self._last_page_len = None
        self._qo_indptr = None

        # Plan cache: shape key -> True once planned
        self._planned: dict = {}

    # -- workspace ---------------------------------------------------------------------------

    # BUG 117 probe: 4 KB canary allocated in the SAME block right after the workspace
    # (a separate small tensor would land in the small pool, never adjacent). FI should
    # never write past WORKSPACE_BYTES; if a split-KV schedule overflows the workspace,
    # the canary catches it. Checked by TPContext._fi_guard.
    _workspace_canary: dict = {}

    @classmethod
    def _get_workspace(cls, device: torch.device) -> torch.Tensor:
        key = str(device)
        ws = cls._workspace_cache.get(key)
        if ws is None:
            full = torch.zeros(cls.WORKSPACE_BYTES + 4096, dtype = torch.uint8, device = device)
            ws = full[:cls.WORKSPACE_BYTES]
            cls._workspace_canary[key] = full[cls.WORKSPACE_BYTES:]
            cls._workspace_canary[key].fill_(0xA5)
            cls._workspace_cache[key] = ws
        return ws

    # -- capability --------------------------------------------------------------------------

    @staticmethod
    def supports(
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        page_size: int,
        dtype: torch.dtype = torch.half,
    ) -> bool:
        """
        Single policy hook deciding whether a layer may use FlashInfer.

        Includes head_dim 512 (Gemma-4 global layers): served by the paged decode
        wrapper and single_decode on sm86 / FlashInfer >= 0.7.4 (verified cos 1.0).
        """
        if not has_flashinfer:
            return False
        if head_dim not in FLASHINFER_GQA_HEAD_DIMS:
            return False
        if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
            return False
        if not flashinfer_supports_page_size(page_size):
            return False
        if dtype not in (torch.half, torch.bfloat16):
            return False
        return True

    # -- buffers -----------------------------------------------------------------------------

    def _ensure_buffers(self, batch: int, max_pages: int, device: torch.device):

        if self._indptr is None or self._indptr.numel() < batch + 1 or self._indptr.device != device:
            self._indptr = torch.zeros(batch + 1, dtype = torch.int32, device = device)
        if self._indices is None or self._indices.numel() < batch * max_pages or self._indices.device != device:
            self._indices = torch.zeros(batch * max_pages, dtype = torch.int32, device = device)
        if self._last_page_len is None or self._last_page_len.numel() < batch or self._last_page_len.device != device:
            self._last_page_len = torch.zeros(batch, dtype = torch.int32, device = device)

    def _ensure_qo_indptr(self, batch: int, q_len: int, device: torch.device):
        if self._qo_indptr is None or self._qo_indptr.numel() < batch + 1 or self._qo_indptr.device != device:
            self._qo_indptr = torch.zeros(batch + 1, dtype = torch.int32, device = device)
        self._qo_indptr[:batch + 1] = torch.arange(
            0, (batch + 1) * q_len, q_len, dtype = torch.int32, device = device
        )

    # -- main entry point --------------------------------------------------------------------

    def plan_and_run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_index: torch.Tensor,
        page_size: int,
        sm_scale: float,
        window_left: int = -1,
        logits_soft_cap: float | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Run paged attention through FlashInfer.

        :param q:
            ``[batch, q_len, num_q_heads, head_dim]``

        :param k_cache / v_cache:
            ``[num_pages, page_size, num_kv_heads, head_dim]``

        :param cache_seqlens:
            ``[batch]`` total KV length per sequence, including the q_len new tokens.

        :param block_index:
            ``[batch, pages_per_seq]``

        :return:
            ``[batch, q_len, num_q_heads, head_dim]``
        """

        batch, q_len, num_q_heads, head_dim = q.shape
        num_kv_heads = k_cache.shape[2]
        device = q.device

        self._ensure_buffers(batch, block_index.shape[1], device)

        indptr, indices, last_page_len = page_table_to_indptr(
            block_index,
            cache_seqlens,
            page_size,
            out_indptr = self._indptr,
            out_indices = self._indices,
            out_last_page_len = self._last_page_len,
        )

        workspace = self._get_workspace(device)

        # Plan caching.
        #
        # FlashInfer's plan encodes the actual indptr/indices, i.e. the per-sequence page counts.
        # Whether a cached plan stays valid when those buffers are rewritten in place depends on
        # the FlashInfer version, so the safe default is to re-plan on every call. Set
        # EXLLAMA_FLASHINFER_PLAN_CACHE=1 to opt into caching, which is keyed on the exact
        # (batch, total_pages, q_len) shape and therefore only skips work when nothing changed.
        #
        # This must be benchmarked and parity-checked on hardware before being enabled by default.
        # See IMPLEMENTED.md "Needs hardware validation".
        use_plan_cache = _PLAN_CACHE_ENABLED

        plan_key = (
            batch,
            int(indptr[-1].item()),
            page_size,
            window_left,
            logits_soft_cap,
            sm_scale,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_len,
        )

        if q_len == 1:

            if self.decode_wrapper is None:
                self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

            if not use_plan_cache or plan_key not in self._planned:
                self.decode_wrapper.plan(
                    indptr = indptr,
                    indices = indices,
                    last_page_len = last_page_len,
                    num_qo_heads = num_q_heads,
                    num_kv_heads = num_kv_heads,
                    head_dim = head_dim,
                    page_size = page_size,
                    pos_encoding_mode = "NONE",
                    window_left = window_left,
                    logits_soft_cap = logits_soft_cap,
                    sm_scale = sm_scale,
                    q_data_type = q.dtype,
                    kv_data_type = k_cache.dtype,
                )
                if use_plan_cache:
                    self._planned[plan_key] = True

            q_flat = q.view(batch, num_q_heads, head_dim)
            out = self.decode_wrapper.run(q_flat, (k_cache, v_cache))
            return out.view(batch, 1, num_q_heads, head_dim)

        # Prefill (q_len > 1)

        if self.prefill_wrapper is None:
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

        self._ensure_qo_indptr(batch, q_len, device)
        qo_indptr = self._qo_indptr[:batch + 1]

        if not use_plan_cache or plan_key not in self._planned:
            self.prefill_wrapper.plan(
                qo_indptr = qo_indptr,
                paged_kv_indptr = indptr,
                paged_kv_indices = indices,
                paged_kv_last_page_len = last_page_len,
                num_qo_heads = num_q_heads,
                num_kv_heads = num_kv_heads,
                head_dim_qk = head_dim,
                page_size = page_size,
                causal = causal,
                pos_encoding_mode = "NONE",
                sm_scale = sm_scale,
                window_left = window_left,
                logits_soft_cap = logits_soft_cap,
                q_data_type = q.dtype,
                kv_data_type = k_cache.dtype,
            )
            if use_plan_cache:
                self._planned[plan_key] = True

        q_flat = q.reshape(batch * q_len, num_q_heads, head_dim)
        out = self.prefill_wrapper.run(q_flat, (k_cache, v_cache))
        return out.view(batch, q_len, num_q_heads, head_dim)

    def invalidate_plans(self):
        self._planned.clear()


# ---------------------------------------------------------------------------------------------
# Non-paged (no cache) helpers, used by _attn_flash
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# Eager prefill: persistent ragged wrappers + cuDNN for head_dim 512
# ---------------------------------------------------------------------------------------------
#
# Prefill chunk-curve fix (2026-11-20): the stateless single_prefill API builds a fresh
# wrapper and plan per layer per chunk (60 plans/chunk), and on sm86 "auto" resolves to
# the FA2 backend, whose hd512 prefill kernel runs far below tensor-core peak -- the
# measured ms/chunk grew ~linearly with kv_len (~900 ms early -> ~5400 ms at 56k on a
# 75k prompt). Two fixes:
#   1. Persistent per-geometry ragged wrappers; plan once per (q_len, kv_len) -- layers
#      sharing a geometry within a chunk reuse the plan (60 plans/chunk -> 2).
#   2. head_dim > 256 prefill prefers the cuDNN fused engine (backend="cudnn"), which
#      handles large head dims at tensor-core efficiency on Ampere. Probed once with a
#      tiny shape; any failure falls back to FA2 ragged, then to stateless single_prefill.
#      EXL2_PREFILL_CUDNN=0 disables the cuDNN preference.
# Wrappers share the per-device 128 MB workspace with the paged wrappers; all runs are
# sequential on the TP stream (eager prefill is never graph-captured), so sharing is safe.

_ragged_wrappers: dict = {}
_ragged_planned: dict = {}
_cudnn_prefill_ok: bool | None = None


# Prefill cost attribution (EXL2_PREFILL_TRACE): event pairs around the eager prefill
# attention runs, drained once per chunk by model.forward's trace print (same contract
# as tensor_p._coll_events -- queue only, no host sync at record time).
_PREFILL_TRACE = os.environ.get("EXL2_PREFILL_TRACE", "1") != "0"
_attn_events: list = []


class _AttnTimer:
    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        if self.on:
            self.s = torch.cuda.Event(enable_timing = True)
            self.e = torch.cuda.Event(enable_timing = True)
            self.s.record()
        return self

    def __exit__(self, *exc):
        if self.on:
            self.e.record()
            _attn_events.append((self.s, self.e))
        return False


def prefill_trace_attn_ms() -> float:
    if not _attn_events:
        return 0.0
    torch.cuda.synchronize()
    ms = sum(s.elapsed_time(e) for s, e in _attn_events)
    _attn_events.clear()
    return ms


def _ragged_key(q, k, sm_scale, window_left, backend):
    return (str(q.device), q.shape[2], k.shape[2], q.shape[3],
            sm_scale, window_left, backend)


def _evict_ragged_backend(backend):
    for d in (_ragged_wrappers, _ragged_planned):
        for key in [key for key in d if key[-1] == backend]:
            del d[key]


def ragged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    causal: bool = True,
    window_left: int = -1,
    backend: str = "fa2",
) -> torch.Tensor:
    """Persistent-wrapper ragged prefill. q/k/v are [1, seq, heads, head_dim]."""
    key = _ragged_key(q, k, sm_scale, window_left, backend)
    wrapper = _ragged_wrappers.get(key)
    if wrapper is None:
        workspace = FlashInferPaged._get_workspace(q.device)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", backend=backend)
        _ragged_wrappers[key] = wrapper
    ql, kv = q.shape[1], k.shape[1]
    if _ragged_planned.get(key) != (ql, kv, causal):
        qo_indptr = torch.tensor([0, ql], dtype=torch.int32, device=q.device)
        kv_indptr = torch.tensor([0, kv], dtype=torch.int32, device=q.device)
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=q.shape[2],
            num_kv_heads=k.shape[2],
            head_dim_qk=q.shape[3],
            head_dim_vo=v.shape[3],
            causal=causal,
            sm_scale=sm_scale,
            window_left=window_left,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )
        _ragged_planned[key] = (ql, kv, causal)
    with _AttnTimer(_PREFILL_TRACE and ql > 32):
        out = wrapper.run(q[0], k[0], v[0])
    return out.unsqueeze(0)


def cudnn_prefill_ok(device: torch.device) -> bool:
    """One-time tiny-shape probe of the cuDNN ragged prefill backend (hd512)."""
    global _cudnn_prefill_ok
    if _cudnn_prefill_ok is None:
        if os.environ.get("EXL2_PREFILL_CUDNN", "1") == "0":
            _cudnn_prefill_ok = False
        else:
            try:
                z = torch.zeros((1, 2, 8, 512), dtype=torch.half, device=device)
                kv = torch.zeros((1, 2, 1, 512), dtype=torch.half, device=device)
                ragged_prefill(z, kv, kv, 512 ** -0.5, backend="cudnn")
                torch.cuda.synchronize(device)
                _cudnn_prefill_ok = True
            except Exception:
                _cudnn_prefill_ok = False
                _evict_ragged_backend("cudnn")
    return _cudnn_prefill_ok


_hd512_cuda_ok: bool | None = None


_hd512_sweep_done: bool = False


def _hd512_fa2_sweep(device):
    """
    One-time diagnostic: time FI's fa2 ragged hd512 prefill across representative
    shapes and print ms + effective TFLOPS.

    Why: the first probe measured fa2 at ~36 TFLOPS @ q256/kv4096, but the live
    EXL2_PREFILL_TRACE attn column implies ~9.6 TFLOPS @ q2048/kv75k -- a 4x
    collapse with kv length that points at split-KV scheduling / workspace
    underfill (fewer CTAs -> idle SMs), NOT an inherent kernel limit. This sweep
    shows exactly where the falloff happens so the next fix targets the cause.
    Runs once at the first hd512 prefill (~8 s), independent of the kernel gate.
    """
    global _hd512_sweep_done
    if _hd512_sweep_done:
        return
    _hd512_sweep_done = True
    try:
        g = torch.Generator(device = device).manual_seed(1234)
        scale = 512 ** -0.5
        print("[prefill] hd512 fa2 shape sweep (ms / effective TFLOPS):", flush = True)
        for ql, kv, iters in ((256, 4096, 5), (2048, 4096, 3), (2048, 32768, 2), (2048, 75000, 2)):
            q = torch.randn((1, ql, 8, 512), dtype = torch.half, device = device, generator = g) * 0.1
            k = torch.randn((1, kv, 1, 512), dtype = torch.half, device = device, generator = g) * 0.1
            v = torch.randn((1, kv, 1, 512), dtype = torch.half, device = device, generator = g) * 0.1
            ragged_prefill(q, k, v, scale, True, -1, backend = "fa2")
            torch.cuda.synchronize(device)
            s = torch.cuda.Event(enable_timing = True); e = torch.cuda.Event(enable_timing = True)
            s.record()
            for _ in range(iters):
                ragged_prefill(q, k, v, scale, True, -1, backend = "fa2")
            e.record(); torch.cuda.synchronize(device)
            ms = s.elapsed_time(e) / iters
            flops = 4.0 * ql * (kv - ql / 2) * 8 * 512
            print(f"[prefill]   q{ql:>5}/kv{kv:>6}: {ms:8.2f} ms  {flops / (ms * 1e-3) / 1e12:6.1f} TFLOPS", flush = True)
            del q, k, v
    except Exception as ex:
        print(f"[prefill] hd512 sweep failed ({type(ex).__name__}: {ex})", flush = True)


def hd512_cuda_prefill_ok(device) -> bool:
    """
    Gate for the custom wmma hd512 prefill kernel (ext_c.prefill_hd512).

    DEFAULT OFF: v1 measured 2.02 ms vs fa2 0.46 ms @ q256/kv4096 (occupancy-starved
    16-row tiles). Kept in-tree for iteration; EXL2_PREFILL_HD512_CUDA=1 re-runs the
    parity + timing probe at a representative prefill shape (q2048/kv32768) and
    enables only if it beats fa2 there. The fa2 shape sweep runs regardless.
    """
    global _hd512_cuda_ok
    _hd512_fa2_sweep(device)
    if _hd512_cuda_ok is not None:
        return _hd512_cuda_ok
    if os.environ.get("EXL2_PREFILL_HD512_CUDA", "0") == "0":
        _hd512_cuda_ok = False
        return False
    try:
        from exllamav2.ext import ext_c
        g = torch.Generator(device = device).manual_seed(1234)
        scale = 512 ** -0.5
        ok = True
        for ql, kv in ((129, 385), (64, 64), (33, 2049)):
            q = torch.randn((1, ql, 8, 512), dtype = torch.half, device = device, generator = g) * 0.1
            k = torch.randn((1, kv, 1, 512), dtype = torch.half, device = device, generator = g) * 0.1
            v = torch.randn((1, kv, 1, 512), dtype = torch.half, device = device, generator = g) * 0.1
            ref = ragged_prefill(q, k, v, scale, True, -1, backend = "fa2")
            o = torch.empty_like(q)
            ext_c.prefill_hd512(q[0], k[0], v[0], o[0], ql, kv, 8, 1, scale)
            torch.cuda.synchronize(device)
            a, b = o.float().flatten(), ref.float().flatten()
            cos = torch.nn.functional.cosine_similarity(a, b, dim = 0).item()
            err = (a - b).abs().max().item()
            if not (cos > 0.999 and err < 0.01 * ref.float().abs().max().item() + 1e-3):
                print(f"[prefill] hd512 CUDA parity FAIL @ q{ql}/kv{kv} (cos {cos:.5f}, err {err:.4f}); using fa2", flush = True)
                ok = False
                break
        if ok:
            ql, kv = 2048, 32768
            q = torch.randn((1, ql, 8, 512), dtype = torch.half, device = device, generator = g) * 0.1
            k = torch.randn((1, kv, 1, 512), dtype = torch.half, device = device, generator = g) * 0.1
            v = torch.randn((1, kv, 1, 512), dtype = torch.half, device = device, generator = g) * 0.1
            o = torch.empty_like(q)
            for _ in range(2):
                ext_c.prefill_hd512(q[0], k[0], v[0], o[0], ql, kv, 8, 1, scale)
                ragged_prefill(q, k, v, scale, True, -1, backend = "fa2")
            torch.cuda.synchronize(device)
            s = torch.cuda.Event(enable_timing = True); e = torch.cuda.Event(enable_timing = True)
            s.record()
            for _ in range(3):
                ext_c.prefill_hd512(q[0], k[0], v[0], o[0], ql, kv, 8, 1, scale)
            e.record(); torch.cuda.synchronize(device)
            t_cuda = s.elapsed_time(e) / 3
            s.record()
            for _ in range(3):
                ragged_prefill(q, k, v, scale, True, -1, backend = "fa2")
            e.record(); torch.cuda.synchronize(device)
            t_fa2 = s.elapsed_time(e) / 3
            ok = t_cuda < t_fa2
            print(f"[prefill] hd512 CUDA kernel {'ENABLED' if ok else 'slower than fa2, keeping fa2'} "
                  f"(cuda {t_cuda:.2f} ms vs fa2 {t_fa2:.2f} ms @ q2048/kv32768)", flush = True)
            del q, k, v, o
        _hd512_cuda_ok = ok
    except Exception as ex:
        print(f"[prefill] hd512 CUDA probe failed ({type(ex).__name__}: {ex}); using fa2", flush = True)
        _hd512_cuda_ok = False
    return _hd512_cuda_ok


def smart_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    causal: bool = True,
    window_left: int = -1,
) -> torch.Tensor:
    """Eager prefill dispatch: hd512 CUDA (probed, opt-in) -> FA2 ragged -> cuDNN -> single.

    FA2 FIRST for hd512 (BUG 115): the load-time shape sweep (debug99.txt) measured
    FI's fa2 ragged hd512 at 32-43 TFLOPS across q256/kv4k .. q2048/kv75k -- NO
    falloff with kv length. Yet live prefill ran at ~9.6 TFLOPS because this dispatch
    preferred cuDNN whenever its tiny-shape probe passed: on sm86 the probe succeeds
    but the hd512 cuDNN engine is ~4x slower than fa2 ragged at depth. The original
    "cuDNN preference" was anchored to the OLD 2 TFLOPS measurement, which was the
    single_prefill path, not ragged fa2. cuDNN remains a fallback if fa2 ever raises.
    """
    global _cudnn_prefill_ok, _hd512_cuda_ok
    if q.shape[3] > 256 and causal and window_left < 0 and q.dtype == torch.half:
        if hd512_cuda_prefill_ok(q.device):
            try:
                from exllamav2.ext import ext_c
                ql, H, D = q.shape[1], q.shape[2], q.shape[3]
                kv, KVH = k.shape[1], k.shape[2]
                o = torch.empty_like(q)
                with _AttnTimer(_PREFILL_TRACE and ql > 32):
                    ext_c.prefill_hd512(q[0], k[0], v[0], o[0], ql, kv, H, KVH, sm_scale)
                return o
            except Exception:
                _hd512_cuda_ok = False
    try:
        return ragged_prefill(q, k, v, sm_scale, causal, window_left, backend="fa2")
    except Exception:
        _evict_ragged_backend("fa2")
        if q.shape[3] > 256 and cudnn_prefill_ok(q.device):
            try:
                return ragged_prefill(q, k, v, sm_scale, causal, window_left, backend="cudnn")
            except Exception:
                _cudnn_prefill_ok = False
                _evict_ragged_backend("cudnn")
        return single_prefill(q, k, v, sm_scale=sm_scale, causal=causal,
                              window_left=window_left)


def single_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    causal: bool = True,
    window_left: int = -1,
    logits_soft_cap: float | None = None,
) -> torch.Tensor:
    """
    FlashInfer replacement for ``flash_attn_func``.

    :param q / k / v:
        ``[batch, seq, heads, head_dim]`` (the flash-attn layout)

    FlashInfer's ``single_prefill_with_kv_cache`` is a *single-request* API: it takes 3-D
    ``[seq, heads, head_dim]`` tensors and has no batch dimension. Passing the 4-D flash-attn
    tensors straight through does not raise — FlashInfer silently reads the leading batch dim as
    sequence length and returns garbage (measured max error ~6.0 vs SDPA, plus NaN on some
    shapes). Batch is therefore flattened explicitly here and restored on the way out.
    """

    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, \
        f"single_prefill expects [batch, seq, heads, head_dim], got {tuple(q.shape)}"

    batch, seq, _, _ = q.shape

    # FlashInfer handles one request at a time; loop over the batch. Batch is normally 1 here
    # (the generator packs sequences into the sequence dimension), so this is not a hot path.
    outs = []
    for b in range(batch):
        with _AttnTimer(_PREFILL_TRACE and seq > 32):
            outs.append(flashinfer.single_prefill_with_kv_cache(
                q[b].contiguous(),
                k[b].contiguous(),
                v[b].contiguous(),
                causal = causal,
                sm_scale = sm_scale,
                window_left = window_left,
                logits_soft_cap = logits_soft_cap,
            ))

    return outs[0].unsqueeze(0) if batch == 1 else torch.stack(outs, dim = 0)


def varlen_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sm_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """
    FlashInfer replacement for ``flash_attn_varlen_func``, used for block-diagonal layers.

    :param q / k / v:
        ``[total_tokens, heads, head_dim]``
    """
    workspace = FlashInferPaged._get_workspace(q.device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

    indptr = cu_seqlens.to(torch.int32)
    wrapper.plan(
        qo_indptr = indptr,
        kv_indptr = indptr,
        num_qo_heads = q.shape[1],
        num_kv_heads = k.shape[1],
        head_dim_qk = q.shape[2],
        causal = causal,
        sm_scale = sm_scale,
        q_data_type = q.dtype,
        kv_data_type = k.dtype,
    )
    return wrapper.run(q, k, v)
