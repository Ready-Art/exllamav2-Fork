from __future__ import annotations
import os
import torch
from exllamav2.util import get_all_gpu_memory, integer_split
from exllamav2.device import global_streams
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
# from line_profiler import profile
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2
    from exllamav2.attn import ExLlamaV2Attention

BROADCAST_KV = 0
BROADCAST_ID = 1
BROADCAST_VC = 2


def fuse_agnr_enabled(kv_len: int) -> bool:
    """
    Context-adaptive AG+norm+residual fusion. Measured (user regime, temp1/top_p0.92,
    4x3090 200W): +7.8% total MTP tok/s at 100k ctx, but -17% at ~1.3k — the fixed
    collective-latency saving only pays once the round is attention-bound. Default
    "auto": fuse at kv >= 8192. EXL2_TP_FUSE_AGNR=1/0 forces on/off; the threshold
    is tunable via EXL2_TP_FUSE_AGNR_KV.
    """
    import os
    v = os.environ.get("EXL2_TP_FUSE_AGNR", "auto")
    if v == "1":
        return True
    if v == "0":
        return False
    try:
        thr = int(os.environ.get("EXL2_TP_FUSE_AGNR_KV", "8192"))
    except ValueError:
        thr = 8192
    return kv_len >= thr
BROADCAST_RS = 3
BROADCAST_Q = 4
# Hybrid linear-attention (Qwen3.5 gated delta net) splits, in head units.
# GDN_K: key/query heads (nk). GDN_V: value heads (nv). The value split is
# derived from the key split times the GQA group so every rank's value-head
# block maps to a whole number of key heads (the delta rule is per-head
# independent, so ranks never exchange recurrent state).
BROADCAST_GDN_K = 5
BROADCAST_GDN_V = 6

# Decode-time replicated-residual flow (EXL2_TP_DECODE_REPLICATED): forwards with
# q_len <= this threshold keep the residual FULL-width on every rank instead of the
# seqpar hidden shard. Same weights, same math -- but the two tiny per-layer post-norm
# stat all_reduces (the worst latency offenders at ~120/token) disappear, replaced by
# full-width post-norms on the already-gathered projection outputs. Prefill chunks
# (large q_len) keep the seqpar shard flow and its collective-volume win.
DECODE_REPL_MAX_Q = 32


# Prefill cost attribution (EXL2_PREFILL_TRACE): CUDA-event pairs recorded around the
# large rank-mode collectives, drained once per chunk by model.forward's trace print.
# Events are only queued (no host sync here); the drain synchronizes once at the chunk
# boundary, which the chunk loop already pays implicitly. Gated to eager multi-row
# forwards so graph capture/replay never sees an event alloc.
_PREFILL_TRACE = os.environ.get("EXL2_PREFILL_TRACE", "1") != "0"
_coll_events: list = []


class _CollTimer:
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
            _coll_events.append((self.s, self.e))
        return False


def prefill_trace_coll_ms() -> float:
    if not _coll_events:
        return 0.0
    torch.cuda.synchronize()
    ms = sum(s.elapsed_time(e) for s, e in _coll_events)
    _coll_events.clear()
    return ms


def tp_reuse_buffers(cache: dict, specs: list, key_rows: bool = False,
                     eager_large_ok: bool = False):
    """
    Return reusable output buffers for a TP module.

    ``specs`` is a list of (device, rows, cols, dtype). Buffers are keyed by
    (device, cols, dtype) and reallocated only when the row count changes, so the
    cache holds at most one buffer per device per module. BUG 41.

    BUG 110: with ``key_rows=True`` the row count is part of the key, so different q_len
    shapes get DISTINCT, never-reallocated entries. Required for graph-mode collectives:
    a captured graph bakes the buffer address into its nodes, and reallocating it for another
    shape (bonus q_len=1 vs verify q_len=W) frees that address -> cuGraphLaunch segfault.

    BUG 120: with ``eager_large_ok=True`` (callers pass ``not graph_mode``), prefill-sized
    shapes (rows > 32, the MAX_Q_GEMM_ROWS / decode_repl envelope) get FRESH uncached
    buffers instead of replacing the module's cached entry. The per-module cache otherwise
    pins a (chunk_rows x shard_cols) buffer for EVERY projection/norm of EVERY layer for
    the whole prefill -- ~150 MiB/layer at world=2, ~9 GB over 60 layers, which OOMed the
    master mid-prefill on a 2-GPU 22 GB split (debug.txt; at world=4 the half-width shards
    fit the larger headroom, hiding it). Fresh buffers are consumed within the layer and
    freed, so the caching allocator recycles one layer's worth (~200 MiB). Decode/graph
    shapes (rows <= 32, or any rows under graph_mode) keep the cached stable-address
    behavior exactly; large fresh blocks also can't alias the small baked graph buffers
    (separate allocator size pools).
    """
    out = []
    for dev, rows, cols, dtype in specs:
        if eager_large_ok and not key_rows and rows > 32:
            out.append(torch.empty((rows, cols), device = dev, dtype = dtype))
            continue
        key = (dev, rows, cols, dtype) if key_rows else (dev, cols, dtype)
        ent = cache.get(key)
        if ent is None or ent[0] != rows:
            buf = torch.empty((rows, cols), device = dev, dtype = dtype)
            cache[key] = (rows, buf)
        else:
            buf = ent[1]
        out.append(buf)
    return out


class TPContext:

    model: ExLlamaV2

    kv_split: list[tuple[int, int, int]] | None
    kv_split_devs: list[int] | None
    id_split: list[tuple[int, int, int]] | None
    id_split_devs: list[int] | None
    vc_split: list[tuple[int, int, int]] | None
    vc_split_devs: list[int] | None
    rs_split: list[tuple[int, int, int]] | None
    rs_split_devs: list[int] | None
    q_split: list[tuple[int, int, int]] | None
    q_split_devs: list[int] | None

    pinned_temp: list[torch.Tensor] | None

    device: int | None
    all_devs: list[int | None] | None
    num_devices: int | None
    streams: list[int | None] | None

    ext_tp_context: int | None

    sin: list[torch.Tensor | None] | None
    cos: list[torch.Tensor | None] | None


    def __init__(
        self,
        model: ExLlamaV2,
        gpu_split: list[float] | None,
        expect_cache_tokens: int = 0,
        expect_cache_base: type = None,
        max_batch: int = 1
    ):
        self.model = model
        cfg = self.model.config
        # Multi-batch (MP slot scheduler): number of cache rows / independent sequences
        # the device-side position tensors and graph-mode FI wrappers must serve. 1 keeps
        # the verified single-sequence path byte-identical.
        self.max_batch = max(1, int(max_batch))

        assert cfg.arch.lm.supports_tp, \
            f"Tensor-parallel is not supported for {cfg.arch.arch_string}"
        # The %128 granularity is required by the dense-MLP column split. The new
        # MoE backbones run their MLP/expert blocks replicated (or shard experts
        # per-head-group), so the dense intermediate size need not be %128 there
        # (Gemma-4-26B dense 2112, expert 704; Qwen4-Exp expert 640).
        if not getattr(cfg.arch.lm, "gemma4_moe_block", False) and \
                not getattr(cfg.arch.lm, "hc_residual", False):
            assert cfg.intermediate_size % 128 == 0, \
                "Model intermediate size must be divisible by 128"

        self.kv_split = None
        self.kv_split_devs = None
        self.id_split = None
        self.id_split_devs = None
        self.vc_split = None
        self.vc_split_devs = None
        self.rs_split = None
        self.rs_split_devs = None
        self.q_split = None
        self.q_split_devs = None
        self.device = None
        self.all_devs = None
        self.num_devices = None
        self.streams = None
        self.pinned_temp = None
        self.ext_tp_context = None

        self.sin = None
        self.cos = None

        # Reusable destination buffers for broadcast()/gather_dev(). Allocating per call is
        # not permitted inside a CUDA graph capture region. BUG 43.
        self._bc_cache = None
        self._gather_cache = None
        self._ag_cache = None
        # BUG 110: separate, ROW-KEYED, never-evicted caches for graph-mode collectives.
        # The eager caches above key by (dev, cols, dtype) and reallocate when the row count
        # changes -- so the bonus graph (q_len=1) and the verify graph (q_len=W) clobbered
        # each other's captured buffer, freeing memory a graph baked into its nodes ->
        # cuGraphLaunch segfault on SP replay. Graph shapes are only {1, W}, so a row-keyed
        # cache holds a few tiny buffers permanently (each graph keeps its own address);
        # eager prefill keeps the realloc cache so large chunk shapes are not all retained.
        self._bc_cache_graph = None
        self._gather_cache_graph = None

        # CUDA-graph mode. When set, gather() returns a *device* tensor on the primary device
        # instead of a host view of pinned_temp, and skips the host drain. That removes the two
        # things that make the decode step uncapturable. `pos` is the current sequence position
        # as a device int32 tensor, so RoPE and the KV write read it on-device rather than
        # having it baked into kernel arguments at capture time. BUG 43.
        self.graph_mode = False
        self.pos = None
        self.pos_tp = {}

        # Per-geometry state. Populated by define_split()/finalize(); declared here because
        # scratch_space_tp() and tp_split() call get_split() during load, before finalize().
        self.geometry_splits = {}
        self.geometry_keys = []
        self.geometry_key_active = None
        self.ext_tp_contexts = {}

        # Multiprocess mode: the device index this process owns, or None for the
        # single-process path. Set by TPContext.set_rank() from a worker before loading.
        self.rank = None

        # Abortable prefill (MP): the master's TPProcessGroup sets abort_flag to a
        # multiprocessing.Event the API layer signals on client disconnect. model.forward
        # polls it at every prefill chunk boundary and propagates the decision with a tiny
        # MAX all_reduce so ALL ranks abort at the SAME chunk (a local-only check would
        # desync the ranks and hang NCCL). Workers leave it None and contribute 0.
        self.abort_flag = None
        self._abort_dev = None

        # Sequence-parallel (hidden-sharded residual) mode, EXL2_TP_SEQPAR=1. Set by
        # model.load_tp_rank() BEFORE any module loads (the loaders shard norm weights
        # and pick o_proj's parallel layout from it). MP rank-local path only: the
        # residual stream lives as a per-rank hidden shard (BROADCAST_RS split), norms
        # and projection outputs run on shards, and only the column-parallel inputs are
        # allgathered to full width. See attn/mlp.forward_tp_old.
        self.seqpar = False
        # Set by load_tp_rank (MP rank-local path only); single-process TP keeps False.
        self.decode_repl = False
        # PCIe P2P small-collective fast path (P2PComm); None = NCCL everywhere.
        self.p2p = None

        # FlashInfer paged decode wrappers for head_dim > 256 layers in CUDA-graph mode
        # (Gemma-4's hd512 global layers). One wrapper per (device, geometry): all such
        # layers share the per-device head split, so a single per-step plan serves every
        # layer's run() inside the graph. plan() is host work and must run OUTSIDE the
        # capture, before each replay; run() is captured. FlashInfer >= 0.7.4 serves
        # hd512 on sm86 (verified cos 1.0); the contiguous per-device KV buffer is viewed
        # as 128-token pages. See attn.py forward_tp_old / model.py _tp_graph_step.
        self.fi_decode = {}

        # BUG 107: FlashInfer paged PREFILL wrappers for the graphed MTP verify window
        # (q_len > 1). Same graph-safe pattern as fi_decode (use_cuda_graph=True, backend
        # "fa2", fixed indptr/indices/last_page_len buffers refreshed by a per-step plan
        # outside the capture), but with a fixed qo_indptr = [0, q_len, 2*q_len, ...] so
        # the captured multi-token attention replays at every round. One wrapper per
        # (device, geometry, q_len, batch). Replaces the full-cache-depth SDPA that made
        # the graphed verify ~250 ms.
        self.fi_prefill = {}

        # Multi-batch SWA verify: one BatchDecode wrapper per (device, geometry, q_len,
        # batch) serving B*q_len per-row windowed requests (the B==1 path keeps the
        # original per-row fi_decode wrappers). Same fixed-buffer graph contract.
        self.fi_decode_rows = {}

        self.define_split(gpu_split, expect_cache_tokens, expect_cache_base)


    def unload(self):

        if self.ext_tp_context is not None:
            ext_c.free_tp_context(self.ext_tp_context)
            self.ext_tp_context = None


    def all_devices(self) -> list[int]:

        # Multiprocess mode: this process owns exactly one device. Returning the full set
        # would make wait_streams()/_sync_all_devices() touch devices with no context here.
        if self.rank is not None:
            return [self.rank]

        devs = set([d for d, _, _ in self.kv_split])
        devs |= set([d for d, _, _ in self.id_split])
        devs |= set([d for d, _, _ in self.vc_split])
        return sorted(devs)

    # ------------------------------------------------------------------ hd>256 graph decode

    FI_DECODE_PAGE = 128

    def _fi_batch_meta(self, entry, kv_lens, row_map, npages, page):
        """Refresh a graph-mode wrapper's fixed metadata buffers for one batched step.

        Per-row page lists are contiguous runs ``row_map[b]*npages + [0, pages_valid_b)``
        in the (B_max*npages, page, HKV, D) page view of the multi-row KV buffer. indptr /
        last_page_len are written in place; the page indices are staged through a separate
        buffer (plan() copies them into the registered fixed buffer, whose address the
        captured graph holds). All writes run on the TP stream -- plan()'s host readback
        is ordered after them (BUG 114 contract).
        """
        B = len(kv_lens)
        pages_valid = [(l + page - 1) // page for l in kv_lens]
        total = sum(pages_valid)
        # Indices address the FULL page view (row_map[b]*npages + page), so their valid
        # range is (max physical row + 1)*npages -- NOT the B*npages entry count. The
        # fi-guard bounds check uses this (subset batches using the last slot otherwise
        # false-positive as corruption).
        _rm = row_map if row_map is not None else list(range(B))
        entry["pages_bound"] = (max(_rm) + 1) * npages
        idx_host = torch.empty(total, dtype = torch.int32)
        ip = [0]
        o = 0
        for b in range(B):
            pv = pages_valid[b]
            base = row_map[b] * npages
            idx_host[o:o + pv] = torch.arange(base, base + pv, dtype = torch.int32)
            o += pv
            ip.append(o)
        entry["stage"][:total].copy_(idx_host, non_blocking = True)
        entry["indptr"].copy_(torch.tensor(ip, dtype = torch.int32), non_blocking = True)
        entry["last_page_len"].copy_(torch.tensor(
            [kv_lens[b] - (pages_valid[b] - 1) * page for b in range(B)],
            dtype = torch.int32), non_blocking = True)
        return total

    def fi_decode_plan(self, key, kv_lens, k_cache_dev: torch.Tensor,
                       row_map = None) -> bool:
        """
        Plan the shared per-device FlashInfer paged decode wrapper for this step.

        Host-side work: must run OUTSIDE any CUDA graph capture, before warmup/capture
        and before every replay. The contiguous ``(B_max, S, HKV, D)`` TP KV buffer is
        served as ``B_max * (S // 128)`` pages; each batch row's valid prefix length is
        dynamic via indptr/last_page_len, which the per-step re-plan refreshes (the
        graph-safe hd512 pattern).

        :param key: ``(dev, num_qo_heads, num_kv_heads, head_dim, sm_scale, window_left, B)``
            ``window_left`` is the sliding window (>=0) for SWA layers, else -1. Including
            it in the key gives SWA hd256 layers their own wrapper distinct from full hd512.
            B (the batch size) is part of the key because FlashInfer's graph mode freezes
            the batch at wrapper construction (plan() rejects a different batch size).
        :param kv_lens: per-batch-row valid KV prefix including the token being decoded.
            For SWA layers this is the ring-relative fill, not the absolute position.
        :param k_cache_dev: this device's KV buffer (for depth/dtype).
        :param row_map: cache row per batch row (default identity).
        :return: True if the wrapper is ready, False if the shape cannot be served.
        """
        import flashinfer
        from exllamav2 import attn_backend

        # 7-tuple (dev, nq, nkv, hd, scale, window_left, B) for plain decode, or
        # 8-tuple (..., row_index, B) for the B==1 per-row MTP verify wrappers: the
        # row index only discriminates the dict entry (one wrapper per verify row);
        # the plan parameters are the first six elements and the batch is the last.
        if len(key) == 7:
            dev, nq, nkv, hd, scale, window_left, B = key
        else:
            dev, nq, nkv, hd, scale, window_left, _row, B = key
        page = self.FI_DECODE_PAGE
        depth = k_cache_dev.shape[1]
        if depth % page != 0 or B < 1 or len(kv_lens) != B:
            return False
        if any(l < 1 or l > depth for l in kv_lens):
            return False
        if row_map is None:
            row_map = list(range(B))

        # BUG 114: EVERYTHING that touches the fixed metadata buffers -- creation,
        # the per-step scalar writes, and plan() (which reads them back to host) --
        # must run on the SAME stream. The old code created/wrote them on the caller's
        # (legacy default) stream and planned on the non-blocking TP stream: the legacy
        # default stream does not synchronize with non-blocking streams, so plan's D2H
        # readback raced the init/metadata H2D writes and read uninitialized memory
        # (garbage indptr -> scheduler asserts / graph-mode row-count errors, debug30.txt).
        entry = self.fi_decode.get(key)
        prev_dev = torch.cuda.current_device()
        prev_stream = torch.cuda.current_stream(dev)
        torch.cuda.set_device(dev)
        torch.cuda.set_stream(self.streams_dev[dev])
        try:
            self._fi_guard("decode-plan", max(kv_lens), 1)
            if entry is None:
                workspace = attn_backend.FlashInferPaged._get_workspace(torch.device("cuda", dev))
                npages = depth // page
                # BUG 103 (proper fix): use FlashInfer's native CUDA-graph mode. In non-graph
                # mode plan() reallocates internal metadata/split-KV buffers as kv_len changes,
                # staling the addresses a captured graph run() holds (variable-onset garbage at
                # long context). use_cuda_graph=True freezes those buffers: plan() copies the
                # dynamic indptr/indices/last_page_len into the fixed buffers below instead of
                # reallocating, and split-KV intermediates stay in the fixed workspace. This
                # keeps split-KV parallelism (full hd512 decode speed) AND graph-safe addresses.
                # backend="fa2" is explicit: under cuda graphs "auto" resolves to cuDNN, which
                # is not the verified hd512/sm86 path. Batch B is fixed per wrapper.
                indices = torch.empty(B * npages, dtype = torch.int32, device = dev)
                indptr = torch.zeros(B + 1, dtype = torch.int32, device = dev)
                last_page_len = torch.zeros(B, dtype = torch.int32, device = dev)
                entry = {
                    "wrapper": flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        workspace, "NHD",
                        use_cuda_graph = True,
                        backend = "fa2",
                        paged_kv_indptr_buffer = indptr,
                        paged_kv_indices_buffer = indices,
                        paged_kv_last_page_len_buffer = last_page_len,
                    ),
                    "npages": npages,
                    "indices": indices,
                    "indptr": indptr,
                    "last_page_len": last_page_len,
                    "stage": torch.empty(B * npages, dtype = torch.int32, device = dev),
                    "ptrs": (indptr.data_ptr(), last_page_len.data_ptr(), indices.data_ptr()),
                }
                self.fi_decode[key] = entry

            total = self._fi_batch_meta(entry, kv_lens, row_map, entry["npages"], page)
            entry["wrapper"].plan(
                indptr = entry["indptr"],
                indices = entry["stage"][:total],
                last_page_len = entry["last_page_len"],
                num_qo_heads = nq,
                num_kv_heads = nkv,
                head_dim = hd,
                page_size = page,
                pos_encoding_mode = "NONE",
                window_left = window_left,
                logits_soft_cap = None,
                sm_scale = scale,
                q_data_type = torch.half,
                kv_data_type = k_cache_dev.dtype,
                # BUG 103: split-KV stays ENABLED (full parallel reduction). Graph safety
                # comes from use_cuda_graph=True above (fixed buffers), not from disabling
                # split-KV -- disabling it cost long-context decode speed unnecessarily.
                # BUG 117 A/B knob: EXL2_FI_SPLITKV=0 disables it (default unchanged) to
                # test whether the long-context split-KV schedule is the OOB writer.
                disable_split_kv = os.environ.get("EXL2_FI_SPLITKV", "1") == "0",
            )
        finally:
            torch.cuda.set_stream(prev_stream)
            torch.cuda.set_device(prev_dev)
        return True

    def fi_decode_run(self, key, q: torch.Tensor, k_cache_dev: torch.Tensor, v_cache_dev: torch.Tensor):
        """
        Capturable hd>256 decode attention against the per-step plan. ``(B, 1, H, D)`` query,
        contiguous per-device KV buffers; returns ``(B, 1, H * D)`` like _attn_torch.
        """
        entry = self.fi_decode[key]
        page = self.FI_DECODE_PAGE
        npages = entry["npages"]
        nkv = k_cache_dev.shape[2]
        hd = q.shape[3]
        B = q.shape[0]
        k_pages = k_cache_dev.view(-1, page, nkv, hd)
        v_pages = v_cache_dev.view(-1, page, nkv, hd)
        # run() expects 3-D q (batch, num_qo_heads, head_dim); a 2-D (H, D) tensor is
        # misread as q_len_per_req=H by the non-tensor-core decode kernel.
        out = entry["wrapper"].run(q.view(B, q.shape[2], hd), (k_pages, v_pages))
        return out.view(B, 1, q.shape[2] * hd)

    def fi_prefill_plan(self, key, kv_lens, k_cache_dev: torch.Tensor,
                        row_map = None) -> bool:
        """
        BUG 107: plan the shared per-device FlashInfer paged PREFILL wrapper for the
        graphed MTP verify window (q_len > 1). Host-side work, OUTSIDE the capture, before
        warmup/capture and before every replay -- exactly the fi_decode_plan pattern.

        :param key: ``(dev, num_qo_heads, num_kv_heads, head_dim, sm_scale, window_left, q_len, B)``
            q_len and B are part of the key: the fixed query shape is baked into qo_indptr
            and FlashInfer graph mode freezes the batch at construction.
        :param kv_lens: per-batch-row valid KV prefix including the q_len verify tokens
            (ring-relative for SWA layers, absolute for full layers).
        :param row_map: cache row per batch row (default identity).
        :return: True if the wrapper is ready, False if the shape cannot be served.
        """
        import flashinfer
        from exllamav2 import attn_backend

        dev, nq, nkv, hd, scale, window_left, q_len, B = key
        page = self.FI_DECODE_PAGE
        depth = k_cache_dev.shape[1]
        if depth % page != 0 or q_len < 1 or B < 1 or len(kv_lens) != B:
            return False
        if any(l < 1 or l > depth for l in kv_lens):
            return False
        if row_map is None:
            row_map = list(range(B))

        # BUG 114: same-stream contract as fi_decode_plan -- buffer creation, the
        # per-step scalar writes and plan()'s host readback all run on the TP stream.
        # Creating/writing on the caller's legacy default stream while planning on the
        # non-blocking TP stream let plan read uninitialized indptr memory (debug30.txt:
        # garbage qo_indptr/kv_indptr, "rows ... cannot exceed ... initialization 3").
        entry = self.fi_prefill.get(key)
        prev_dev = torch.cuda.current_device()
        prev_stream = torch.cuda.current_stream(dev)
        torch.cuda.set_device(dev)
        torch.cuda.set_stream(self.streams_dev[dev])
        try:
            self._fi_guard("prefill-plan", max(kv_lens), q_len)
            if entry is None:
                workspace = attn_backend.FlashInferPaged._get_workspace(torch.device("cuda", dev))
                npages = depth // page
                # use_cuda_graph=True + backend="fa2": plan() refreshes the FA2 scheduling
                # metadata in fixed buffers (the BUG 103 decode fix), so a captured run()
                # replays correctly as kv_len grows. "auto" would resolve to cuDNN under
                # graphs, which is not the verified hd512/sm86 path. B / q_len fixed.
                qo_indptr = torch.arange(B + 1, dtype = torch.int32, device = dev) * q_len
                indices = torch.empty(B * npages, dtype = torch.int32, device = dev)
                indptr = torch.zeros(B + 1, dtype = torch.int32, device = dev)
                last_page_len = torch.zeros(B, dtype = torch.int32, device = dev)
                entry = {
                    "wrapper": flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                        workspace, "NHD",
                        use_cuda_graph = True,
                        backend = "fa2",
                        qo_indptr_buf = qo_indptr,
                        paged_kv_indptr_buf = indptr,
                        paged_kv_indices_buf = indices,
                        paged_kv_last_page_len_buf = last_page_len,
                    ),
                    "npages": npages,
                    "qo_indptr": qo_indptr,
                    "indices": indices,
                    "indptr": indptr,
                    "last_page_len": last_page_len,
                    "stage": torch.empty(B * npages, dtype = torch.int32, device = dev),
                    "ptrs": (qo_indptr.data_ptr(), indptr.data_ptr(),
                             last_page_len.data_ptr(), indices.data_ptr()),
                }
                self.fi_prefill[key] = entry

            total = self._fi_batch_meta(entry, kv_lens, row_map, entry["npages"], page)
            entry["wrapper"].plan(
                qo_indptr = entry["qo_indptr"],
                paged_kv_indptr = entry["indptr"],
                paged_kv_indices = entry["stage"][:total],
                paged_kv_last_page_len = entry["last_page_len"],
                num_qo_heads = nq,
                num_kv_heads = nkv,
                head_dim_qk = hd,
                page_size = page,
                causal = True,
                pos_encoding_mode = "NONE",
                window_left = window_left,
                logits_soft_cap = None,
                sm_scale = scale,
                q_data_type = torch.half,
                kv_data_type = k_cache_dev.dtype,
                disable_split_kv = os.environ.get("EXL2_FI_SPLITKV", "1") == "0",
            )
        finally:
            torch.cuda.set_stream(prev_stream)
            torch.cuda.set_device(prev_dev)
        return True

    def fi_prefill_run(self, key, q: torch.Tensor, k_cache_dev: torch.Tensor, v_cache_dev: torch.Tensor):
        """
        Capturable multi-token (verify-window) attention against the per-step prefill plan.
        ``(B, q_len, H, D)`` query, contiguous per-device KV buffers; returns
        ``(B, q_len, H * D)`` like _attn_torch.
        """
        entry = self.fi_prefill[key]
        page = self.FI_DECODE_PAGE
        nkv = k_cache_dev.shape[2]
        hd = q.shape[3]
        B, q_len = q.shape[0], q.shape[1]
        k_pages = k_cache_dev.view(-1, page, nkv, hd)
        v_pages = v_cache_dev.view(-1, page, nkv, hd)
        out = entry["wrapper"].run(q.view(B * q_len, q.shape[2], hd), (k_pages, v_pages))
        return out.view(B, q_len, q.shape[2] * hd)

    def fi_decode_rows_plan(self, key, kv_lens, rows, k_cache_dev: torch.Tensor) -> bool:
        """
        Batched per-row windowed DECODE for the graphed SWA verify window (q_len > 1,
        B > 1): one BatchDecode wrapper serving B*q_len "requests", request (b, i) being
        row b's query i attending to its causal ring prefix. The B == 1 path keeps the
        original per-row fi_decode wrappers (verified state); this is its multi-batch
        generalization. Same fixed-buffer graph contract as fi_decode_plan.

        :param key: ``(dev, num_qo_heads, num_kv_heads, head_dim, sm_scale, window_left, q_len, B)``
        :param kv_lens: flat list of B*q_len valid (ring-relative, clamped) KV lengths.
        :param rows: flat list of B*q_len cache rows (row_map[b] repeated q_len times).
        """
        import flashinfer
        from exllamav2 import attn_backend

        dev, nq, nkv, hd, scale, window_left, q_len, B = key
        page = self.FI_DECODE_PAGE
        depth = k_cache_dev.shape[1]
        R = B * q_len
        if depth % page != 0 or q_len < 1 or B < 1 or len(kv_lens) != R or len(rows) != R:
            return False
        if any(l < 1 or l > depth for l in kv_lens):
            return False

        entry = self.fi_decode_rows.get(key)
        prev_dev = torch.cuda.current_device()
        prev_stream = torch.cuda.current_stream(dev)
        torch.cuda.set_device(dev)
        torch.cuda.set_stream(self.streams_dev[dev])
        try:
            self._fi_guard("rows-plan", max(kv_lens), q_len)
            if entry is None:
                workspace = attn_backend.FlashInferPaged._get_workspace(torch.device("cuda", dev))
                npages = depth // page
                indices = torch.empty(R * npages, dtype = torch.int32, device = dev)
                indptr = torch.zeros(R + 1, dtype = torch.int32, device = dev)
                last_page_len = torch.zeros(R, dtype = torch.int32, device = dev)
                entry = {
                    "wrapper": flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        workspace, "NHD",
                        use_cuda_graph = True,
                        backend = "fa2",
                        paged_kv_indptr_buffer = indptr,
                        paged_kv_indices_buffer = indices,
                        paged_kv_last_page_len_buffer = last_page_len,
                    ),
                    "npages": npages,
                    "indices": indices,
                    "indptr": indptr,
                    "last_page_len": last_page_len,
                    "stage": torch.empty(R * npages, dtype = torch.int32, device = dev),
                    "ptrs": (indptr.data_ptr(), last_page_len.data_ptr(), indices.data_ptr()),
                }
                self.fi_decode_rows[key] = entry

            total = self._fi_batch_meta(entry, kv_lens, rows, entry["npages"], page)
            entry["wrapper"].plan(
                indptr = entry["indptr"],
                indices = entry["stage"][:total],
                last_page_len = entry["last_page_len"],
                num_qo_heads = nq,
                num_kv_heads = nkv,
                head_dim = hd,
                page_size = page,
                pos_encoding_mode = "NONE",
                window_left = window_left,
                logits_soft_cap = None,
                sm_scale = scale,
                q_data_type = torch.half,
                kv_data_type = k_cache_dev.dtype,
                disable_split_kv = os.environ.get("EXL2_FI_SPLITKV", "1") == "0",
            )
        finally:
            torch.cuda.set_stream(prev_stream)
            torch.cuda.set_device(prev_dev)
        return True

    def fi_decode_rows_run(self, key, q: torch.Tensor, k_cache_dev: torch.Tensor, v_cache_dev: torch.Tensor):
        """Capturable batched per-row verify attention. ``(B, q_len, H, D)`` query ->
        ``(B, q_len, H * D)`` output (request order (b, i) matches the plan)."""
        entry = self.fi_decode_rows[key]
        page = self.FI_DECODE_PAGE
        nkv = k_cache_dev.shape[2]
        hd = q.shape[3]
        B, q_len = q.shape[0], q.shape[1]
        k_pages = k_cache_dev.view(-1, page, nkv, hd)
        v_pages = v_cache_dev.view(-1, page, nkv, hd)
        out = entry["wrapper"].run(q.view(B * q_len, q.shape[2], hd), (k_pages, v_pages))
        return out.view(B, q_len, q.shape[2] * hd)

    def _fi_guard(self, phase: str, kv_len: int, q_len: int):
        """
        BUG 117 instrumented guard: validate the fixed FI plan buffers before plan().

        log.txt (122k MTP): the q_len=3 verify wrapper's qo_indptr read back as
        garbage ("rows 721464845 ... cannot exceed ... initialization 3") on the
        first verify after a long prefill, while the q_len=2 / decode wrappers
        stayed intact. These buffers are created once and never rewritten (except
        indptr[1]/last_page_len, which plan() refreshes), so garbage means an OOB
        device write landed on them or their storage was reused. This guard checks
        every entry's invariants + pointer stability, prints the evidence (which
        buffer, got vs want, phase), and repairs IN PLACE -- the captured graphs
        hold these addresses, so recreating an entry would stale them. A changed
        data_ptr (storage reuse) cannot be repaired and raises with the evidence.
        EXL2_FI_GUARD=0 disables; only active at kv_len > 20000 (the corruption
        regime) so short-context runs carry no per-plan sync cost.
        """
        if kv_len <= 20000 or os.environ.get("EXL2_FI_GUARD", "1") == "0":
            return
        from exllamav2 import attn_backend
        # Truthful readbacks: the TP streams are non-blocking, so drain the device
        # first. (If the corruption is an ordering race, this sync also masks it --
        # a clean run with the guard on vs a crash with EXL2_FI_GUARD=0 is itself
        # the evidence for that class.)
        torch.cuda.synchronize()
        bad = []
        reuse = []
        for kind, d in (("decode", self.fi_decode), ("prefill", self.fi_prefill),
                        ("rows", self.fi_decode_rows)):
            for key, entry in d.items():
                npages = entry["npages"]
                ind = entry["indptr"]
                lpl = entry["last_page_len"]
                idx = entry["indices"]
                ptrs = (ind.data_ptr(), lpl.data_ptr(), idx.data_ptr())
                if kind == "prefill":
                    ptrs = (entry["qo_indptr"].data_ptr(),) + ptrs
                if "ptrs" in entry and entry["ptrs"] != ptrs:
                    reuse.append((kind, key, entry["ptrs"], ptrs))
                    continue
                # Batch size: decode key (..., B); prefill/rows key (..., q_len, B).
                B = key[-1]
                nreq = len(lpl)
                meta = torch.cat([ind, lpl]).cpu()
                ip = meta[:nreq + 1].tolist()
                lp = meta[nreq + 1:].tolist()
                if kind == "prefill":
                    q_len = key[6]
                    qo = entry["qo_indptr"].cpu()
                    want_qo = [i * q_len for i in range(B + 1)]
                    if qo.tolist() != want_qo:
                        bad.append((kind, key, "qo_indptr", entry["qo_indptr"].data_ptr(),
                                    qo.tolist(), want_qo))
                # Bounds are per-REQUEST (rows wrappers serve B*q_len requests) and
                # per-BUFFER (indices address cache rows via row_map, so values can
                # exceed B*npages when the batch is a subset of the allocated rows).
                ok_ip = (ip[0] == 0 and len(ip) == nreq + 1
                         and all(1 <= ip[i + 1] - ip[i] <= npages for i in range(nreq))
                         and ip[-1] <= nreq * npages)
                if not ok_ip:
                    bad.append((kind, key, "indptr", ind.data_ptr(), ip,
                                f"0-start, per-row 1..{npages} pages, total <= {nreq * npages}"))
                if not all(1 <= x <= 128 for x in lp):
                    bad.append((kind, key, "last_page_len", lpl.data_ptr(), lp, "1..128"))
                if ok_ip and ip[-1] > 0:
                    iv = idx[:ip[-1]].cpu()
                    bound = entry.get("pages_bound", idx.numel())
                    if int(iv.min()) < 0 or int(iv.max()) >= bound:
                        bad.append((kind, key, "indices", idx.data_ptr(),
                                    [int(iv.min()), int(iv.max())], f"0..{bound - 1}"))
        ws_bad = []
        for wkey, canary in attn_backend.FlashInferPaged._workspace_canary.items():
            tail = canary[:1024].cpu()
            if not bool((tail == 0xA5).all()):
                nz = int((tail != 0xA5).sum())
                ws_bad.append((wkey, nz, tail[:8].tolist()))
        if not bad and not reuse and not ws_bad:
            return
        rank = self.rank if self.rank is not None else 0
        print(f"[fi-guard] CORRUPTION rank={rank} phase={phase} kv_len={kv_len} q_len={q_len}",
              flush=True)
        for kind, key, name, ptr, got, want in bad:
            print(f"[fi-guard]   {kind} key={key} buf={name} ptr={hex(ptr)} got={got} want={want}",
                  flush=True)
        for kind, key, old, new in reuse:
            print(f"[fi-guard]   {kind} key={key} STORAGE REUSE ptrs {old} -> {new}", flush=True)
        for wkey, nz, sample in ws_bad:
            print(f"[fi-guard]   WORKSPACE {wkey} tail overwritten ({nz}/1024 bytes) sample={sample}",
                  flush=True)
        if reuse:
            raise RuntimeError("[fi-guard] FI buffer storage reuse -- cannot repair in place "
                               "(captured graphs hold the old addresses); evidence above")
        _dicts = {"decode": self.fi_decode, "prefill": self.fi_prefill,
                  "rows": self.fi_decode_rows}
        if any(key[-1] > 1 for _, key, _, _, _, _ in bad):
            raise RuntimeError("[fi-guard] batched FI metadata corruption -- cannot repair "
                               "without the step's kv_lens/row_map; evidence above")
        for kind, key, name, ptr, got, want in bad:
            entry = _dicts[kind][key]
            if name == "qo_indptr":
                entry["qo_indptr"].copy_(torch.tensor([0, key[6]], dtype=torch.int32,
                                                      device=entry["qo_indptr"].device))
            elif name == "indptr":
                entry["indptr"][0] = 0   # [1] is refreshed by the plan about to run
            elif name == "last_page_len":
                entry["last_page_len"][0] = 1
            elif name == "indices":
                entry["indices"].copy_(torch.arange(entry["npages"], dtype=torch.int32,
                                                    device=entry["indices"].device))
        for wkey, _, _ in ws_bad:
            attn_backend.FlashInferPaged._workspace_canary[wkey].fill_(0xA5)
        print("[fi-guard]   repaired in place (graph addresses preserved); continuing", flush=True)


    def fi_decode_order_before_replay(self):
        """
        Order dev0's replay launch after every device's plan work: record an event on each
        device's TP stream (where fi_decode_plan ran) and make dev0's stream wait on them.
        The graph's internal fork/join events then propagate that ordering to every device
        branch, so no replayed run() can read a half-written plan.
        """
        dev0 = self.device
        s0 = self.streams_dev[dev0]
        for dev in self.all_devices():
            if dev == dev0:
                continue
            ev = torch.cuda.Event()
            ev.record(self.streams_dev[dev])
            s0.wait_event(ev)


    def define_split(
        self,
        gpu_split: list[float] | None,
        expect_cache_tokens: int,
        expect_cache_base: type
    ):
        cfg = self.model.config

        if gpu_split is None:
            gpu_memory = get_all_gpu_memory()
            gpu_split = [0] * (max(gpu_memory.keys()) + 1)
            for k, v in gpu_memory.items():
                gpu_split[k] = v["free"]
        else:
            gpu_split = [gs * 1024 for gs in gpu_split]

        # Q and KV splits

        kv_split = integer_split(cfg.num_key_value_heads, gpu_split)
        q_split = [s * cfg.num_key_value_groups for s in kv_split]
        attn_ratio = [s / cfg.num_key_value_heads for s in kv_split]

        # Subtract size of cache according to KV split

        if not expect_cache_tokens:
            expect_cache_tokens = cfg.max_seq_len * cfg.max_batch_size
        if expect_cache_base == sys.modules["exllamav2.cache"].ExLlamaV2Cache_8bit:
            bytes_per_element = 1
        elif expect_cache_base == sys.modules["exllamav2.cache"].ExLlamaV2Cache_Q8:
            bytes_per_element = 8.5/8
        elif expect_cache_base == sys.modules["exllamav2.cache"].ExLlamaV2Cache_Q6:
            bytes_per_element = 6.5/8
        elif expect_cache_base == sys.modules["exllamav2.cache"].ExLlamaV2Cache_Q4:
            bytes_per_element = 4.5/8
        else:
            bytes_per_element = 2

        # Per-layer geometry: Gemma-4 mixes head_dim 256/512 and 16/4 kv heads, so summing
        # num_key_value_heads * head_dim over layers is not the same as multiplying by
        # num_hidden_layers. PLAN.md §2.2.
        # Sliding-window layers only ever hold a ring of
        #   ceil((sliding_window + overp) / page_size) * page_size
        # tokens, not the full context. Sizing them at `expect_cache_tokens` over-reserves by
        # orders of magnitude: for Gemma-4 at 64k that is 59 GB instead of ~1.4 GB, which
        # zeroes the per-device split and then divides by zero in integer_split().
        # PLAN.md §4.0.
        _sw = getattr(cfg, "sliding_window", 0) or 0
        _page = 128
        _overp = getattr(cfg, "sliding_window_overp", None)
        if _overp is None:
            # Keep in sync with ExLlamaV2Cache_SWA: the ring must hold the sliding window
            # PLUS a full prefill chunk, or the chunk loop's shift drops the window prefix.
            _overp = max(2 * _page, cfg.max_input_len)
        _ring = (((_sw + _overp) + _page - 1) // _page) * _page if _sw else 0

        kv_elements = 0
        for _li in range(cfg.num_hidden_layers):
            _hd, _, _kvh, _full = cfg.get_layer_attn_geometry(_li)
            # Hybrid linear-attention layers carry no K/V (kvh == 0); their recurrent
            # state is tiny (per-head, context-independent) and reserved below instead.
            if _kvh == 0:
                continue
            # Multi-batch: every cache row holds its own ring, so the SWA reservation
            # scales with max_batch too (full layers scale via expect_cache_tokens).
            _tokens = expect_cache_tokens if (_full or not _ring) \
                else min(_ring * self.max_batch, expect_cache_tokens)
            kv_elements += _kvh * _hd * _tokens
        cache_size = 2 * bytes_per_element * kv_elements
        gpu_split = [max(0, gs - int(cache_size * r / 1024**2)) for gs, r in zip(gpu_split, attn_ratio)]

        # Subtract size of attn layers

        ExLlamaV2Attention_ = sys.modules["exllamav2.attn"].ExLlamaV2Attention
        for module in self.model.modules:
            if not isinstance(module, ExLlamaV2Attention_):
                continue
            wfp = module.weight_footprint()
            gpu_split = [max(0, gs - int(wfp * r / 1024**2)) for gs, r in zip(gpu_split, attn_ratio)]

        # Hybrid (Qwen3.5): GDN splits. Value heads derive from key heads times the
        # GQA group so each rank's value block maps to whole key heads. Subtract GDN
        # weight + recurrent-state footprint by the value-head ratio before the
        # MLP/vocab splits are derived.
        self.gdn_k_split = None
        self.gdn_k_split_devs = None
        self.gdn_v_split = None
        self.gdn_v_split_devs = None
        if getattr(cfg, "hybrid_linear_attn", False):
            _group = cfg.linear_num_value_heads // cfg.linear_num_key_heads
            gdn_k_split = integer_split(cfg.linear_num_key_heads, gpu_split)
            gdn_v_split = [s * _group for s in gdn_k_split]
            gdn_ratio = [s / cfg.linear_num_value_heads for s in gdn_v_split]
            ExLlamaV2GatedDeltaNet_ = sys.modules["exllamav2.gdn"].ExLlamaV2GatedDeltaNet
            for module in self.model.modules:
                if not isinstance(module, ExLlamaV2GatedDeltaNet_):
                    continue
                wfp = module.weight_footprint()
                gpu_split = [max(0, gs - int(wfp * r / 1024**2)) for gs, r in zip(gpu_split, gdn_ratio)]
            # Recurrent state: fp32 (nv, dk, dv) + fp16 conv (conv_dim, K-1), x2 for
            # the rewind backup, per batch row. Context-independent.
            _nk, _nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
            _dk, _dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
            _conv_dim = 2 * _nk * _dk + _nv * _dv
            _state_bytes = self.max_batch * (
                4 * _nv * _dk * _dv + 2 * _conv_dim * (cfg.linear_conv_kernel_dim - 1)) * 2
            _n_linear = sum(1 for _t in cfg.layer_types if _t == "linear_attention")
            _state_bytes *= _n_linear
            gpu_split = [max(0, gs - int(_state_bytes * r / 1024**2)) for gs, r in zip(gpu_split, gdn_ratio)]

        # MLP split

        id_split = [s * 128 for s in integer_split(cfg.intermediate_size // 128, gpu_split, 2)]
        rs_split = [s * 32 for s in integer_split(cfg.hidden_size // 32, gpu_split, 8)]

        # Vocab split

        vc_split = [s * 32 for s in integer_split((cfg.vocab_size + 31) // 32, gpu_split, 16)]

        def set_split(raw_split):
            b = 0
            split = []
            devs = []
            for d, s in enumerate(raw_split):
                a = b
                b = a + s
                if s:
                    split.append((d, a, b))
                    devs.append(d)
            return split, devs

        self.kv_split, self.kv_split_devs = set_split(kv_split)
        self.id_split, self.id_split_devs = set_split(id_split)
        self.vc_split, self.vc_split_devs = set_split(vc_split)
        self.rs_split, self.rs_split_devs = set_split(rs_split)
        self.q_split, self.q_split_devs = set_split(q_split)
        if getattr(cfg, "hybrid_linear_attn", False):
            self.gdn_k_split, self.gdn_k_split_devs = set_split(gdn_k_split)
            self.gdn_v_split, self.gdn_v_split_devs = set_split(gdn_v_split)

        # One split set per distinct attention geometry.
        #
        # For every architecture except Gemma-4 there is exactly one geometry, so this
        # produces a single entry and the code path is identical to before. For Gemma-4 it
        # produces two: (256, 32, 16, False) and (512, 32, 4, True). Each gets its own
        # kv/q split, derived from the same post-subtraction `gpu_split`, so the per-device
        # head counts are correct for that geometry (e.g. on 3 GPUs: 16 kv heads -> 6/5/5,
        # 4 kv heads -> 2/1/1).
        #
        # The C++ ExtTPContext stores one split set, but `ext_tp_context` is read from the
        # Python attribute on every call, so one context per geometry can be built and the
        # active one swapped per layer by `set_geometry()`. No C++ change is required.
        self.geometry_splits = {}
        self.geometry_keys = []
        _hybrid = getattr(cfg, "hybrid_linear_attn", False)
        _full_geo_key = None
        for _li in range(cfg.num_hidden_layers):
            _hd, _qh, _kvh, _full = cfg.get_layer_attn_geometry(_li)
            if _hybrid and _kvh == 0:
                # Linear-attention layers have no attention geometry; route them to
                # the full-attention geometry so set_geometry() selects the shared
                # RS/ID/VC splits (GDN uses its own GDN_K/GDN_V splits).
                if _full_geo_key is None:
                    for _lj in range(cfg.num_hidden_layers):
                        _g = cfg.get_layer_attn_geometry(_lj)
                        if _g[2] > 0:
                            _full_geo_key = _g
                            break
                self.geometry_keys.append(_full_geo_key)
            else:
                self.geometry_keys.append((_hd, _qh, _kvh, _full))
        for _key in dict.fromkeys(self.geometry_keys):
            _hd, _qh, _kvh, _full = _key
            _kv = integer_split(_kvh, gpu_split)
            _nkg = _qh // _kvh
            _q = [s * _nkg for s in _kv]
            self.geometry_splits[_key] = {
                "kv": set_split(_kv),
                "q": set_split(_q),
                "id": set_split(id_split),
                "rs": set_split(rs_split),
                "vc": set_split(vc_split),
            }

        self.all_devs = self.all_devices()
        self.device = self.all_devs[0]
        self.num_devices = max(self.all_devs) + 1


    def finalize(self):
        cfg = self.model.config

        if self.rank is not None:
            # BUG 59: multiprocess mode never stages through host memory -- broadcast() is
            # the identity and gather() is a NCCL all_gather -- so the full pinned buffer
            # would be hundreds of MB of dead pinned memory per process. Keep a token-sized
            # one so any code path that still touches pinned_temp stays in bounds.
            size = 4096
        else:
            size = (cfg.max_output_len if cfg.max_output_len is not None else cfg.max_input_len) * cfg.vocab_size
            size = max(size, cfg.max_input_len * cfg.intermediate_size)

        self.pinned_temp = [
            torch.empty((size,), dtype = torch.half, pin_memory = True)
            for _ in range(2)
        ]

        devices = self.all_devices()
        max_device = max(devices)

        # BUG 60: size by num_devices, not max_device + 1, and leave unowned slots as None.
        # global_streams only contains devices whose ExLlamaV2DeviceContext was built, which
        # in a worker is exactly one -- indexing it for any other device raises KeyError.
        n_slots = self.num_devices if self.num_devices is not None else (max_device + 1)
        # 0, not None: make_tp_context() takes Sequence[int] and pybind11 rejects None.
        # The C++ side only indexes ctx->streams[dev] for devs in the filtered split.
        self.streams = [0] * n_slots
        # torch Stream wrappers for the same streams, needed to record/wait events on them
        # during CUDA-graph capture (BUG 43).
        self.streams_dev = {}
        for idx in devices:
            s = global_streams.get(idx)
            if s is None:
                continue
            self.streams[idx] = s.cuda_stream
            self.streams_dev[idx] = s

        # One C++ context per distinct geometry (see define_split). The default context is
        # the first geometry's, so single-geometry models behave exactly as before.
        self.ext_tp_contexts = {}
        for _key, _sp in self.geometry_splits.items():
            # BUG 60: in worker mode hand the C++ context only this rank's slice, so it
            # never indexes ctx->streams at a device this process does not own.
            if self.rank is not None:
                def _own(lst):
                    return [(d, a, b) for d, a, b in lst if d == self.rank]
                _kv, _id, _vc, _rs, _q = (_own(_sp["kv"][0]), _own(_sp["id"][0]),
                                          _own(_sp["vc"][0]), _own(_sp["rs"][0]),
                                          _own(_sp["q"][0]))
            else:
                _kv, _id, _vc, _rs, _q = (_sp["kv"][0], _sp["id"][0], _sp["vc"][0],
                                          _sp["rs"][0], _sp["q"][0])
            self.ext_tp_contexts[_key] = ext_c.make_tp_context(
                _kv, _id, _vc, _rs, _q,
                self.pinned_temp,
                self.streams
            )
        self.geometry_key_active = None
        self.ext_tp_context = self.ext_tp_contexts[self.geometry_keys[0]]

        # Device-side sequence position for CUDA-graph decode (BUG 43). Allocated once here so
        # its address is stable for the lifetime of the model: a captured graph holds this
        # pointer, and reallocating it would silently make the graph read freed memory.
        #
        # One per device. kv_write and the attention mask are built on every device, and a
        # CUDA kernel can only dereference a pointer into its own device's memory, so a single
        # dev0-resident `pos` made devices 1-3 read garbage -- which silently corrupted the KV
        # write on those devices (the graph-vs-eager divergence at step 2).
        # Multi-batch: one position per cache row (pos[b] is sequence b's write position;
        # rope_ reads past_lens[blockIdx.z], kv_write reads pos[b]). At max_batch == 1 this
        # is the original (1,) tensor and every legacy fill/read is unchanged.
        _mb = self.max_batch
        self.pos = torch.zeros((_mb,), dtype = torch.int32, device = self.device)
        self.pos_tp = {}
        for _dev in devices:
            self.pos_tp[_dev] = self.pos if _dev == self.device else \
                torch.zeros((_mb,), dtype = torch.int32, device = _dev)

        # BUG 94: ring-relative KV write position for the sliding-window layers, one per
        # device (same per-device rationale as `pos`). The SWA attention writes its new K/V
        # at ring position `past_len - window_beg` and reads the window from the ring; the
        # absolute `pos` would run off the end of the 1280-deep ring once the sequence
        # grows past it (the 16k prefill/decode crash).
        self.ring_pos = torch.zeros((_mb,), dtype = torch.int32, device = self.device)
        self.ring_pos_tp = {}
        for _dev in devices:
            self.ring_pos_tp[_dev] = self.ring_pos if _dev == self.device else \
                torch.zeros((_mb,), dtype = torch.int32, device = _dev)

        # Multi-batch slot mapping: batch row b of a graphed step writes cache row
        # row_map[b] (the scheduler compacts any active slot subset into the batch).
        # Identity by default; refreshed before every replay like pos. kv_write reads it
        # on-device; the FI plans build their page indices from it host-side.
        self.row_map = torch.arange(_mb, dtype = torch.int32, device = self.device)
        self.row_map_tp = {}
        for _dev in devices:
            self.row_map_tp[_dev] = self.row_map if _dev == self.device else \
                torch.arange(_mb, dtype = torch.int32, device = _dev)

        # Per-device cache_seqlens for flash_attn_with_kvcache, one tensor per device per
        # attention geometry. These live here, not on attn_params, for the same reason `pos` does:
        # a captured graph holds their addresses, so they must be allocated once and updated in
        # place. attn_params is rebuilt on every forward() call, so anything cached on it is
        # reallocated each step and the graph keeps reading the capture-time copy -- which froze
        # every replay at the capture position (the graph-vs-eager divergence at step 2).
        self.past_len_tp = {}
        for _key in dict.fromkeys(self.geometry_keys):
            _sp = self.geometry_splits[_key]["kv"][0]
            if self.rank is not None:
                _sp = [(d, a, b) for d, a, b in _sp if d == self.rank]
            self.past_len_tp[_key] = [
                torch.zeros((_mb,), dtype = torch.int32, device = _dev)
                for _dev, _, _ in _sp
            ]

    def get_past_len_tp(self, geometry_key):
        """Stable per-device cache_seqlens tensors for this geometry."""
        return self.past_len_tp.get(geometry_key)

    def _fill_vec(self, _t, value):
        """Write an int scalar (legacy: every entry) or a per-row list into a device
        position tensor, on the CURRENT stream. The list form writes only the first
        len(value) entries -- the active batch rows; kernels index by batch row."""
        if isinstance(value, int):
            _t.fill_(value)
        else:
            n = len(value)
            stage = getattr(self, "_pos_stage", None)
            if stage is None or stage.numel() < self.max_batch:
                stage = self._pos_stage = torch.empty(self.max_batch, dtype = torch.int32)
            stage[:n].copy_(torch.tensor(value, dtype = torch.int32))
            _t[:n].copy_(stage[:n], non_blocking = True)

    def _set_vec_all_devs(self, tp_dict, value):
        """Issue _fill_vec on every device's TP stream and order dev0's graph-launch
        stream after all of them (the set_pos ordering contract, BUG 43)."""
        evs = []
        for _dev, _t in tp_dict.items():
            with torch.cuda.device(_dev):
                _s = self.streams_dev[_dev]
                _prev = torch.cuda.current_stream(_dev)
                torch.cuda.set_stream(_s)
                try:
                    self._fill_vec(_t, value)
                    _ev = torch.cuda.Event()
                    _ev.record(_s)
                    evs.append((_dev, _ev))
                finally:
                    torch.cuda.set_stream(_prev)
        with torch.cuda.device(self.device):
            _s0 = self.streams_dev[self.device]
            for _dev, _ev in evs:
                if _dev == self.device:
                    continue
                _s0.wait_event(_ev)

    def set_pos(self, value):
        """Set the device-side sequence position on every TP device, ordered before replay.

        ``value`` is an int (single-sequence legacy path: fills the whole tensor) or a
        per-batch-row list (multi-batch: pos[b] = sequence b's position).

        Each device needs its own copy because the kernels that read it (kv_write and the
        attention mask) run on that device and can only dereference their own memory.

        Ordering matters as much as the value. A plain ``fill_`` on a cuda:N tensor is issued on
        that device's *default* stream, which is unrelated to the TP stream the graph launches on,
        so the graph could read the previous token's position. That is the graph-vs-eager KV-write
        divergence. Each fill is therefore issued on that device's TP stream and dev0's TP stream
        (the one the graph is launched on) waits on an event recorded after it.
        """
        self._set_vec_all_devs(self.pos_tp, value)

    def set_ring_pos(self, value):
        """BUG 94: like set_pos, but for the SWA ring write position (see ring_pos)."""
        self._set_vec_all_devs(self.ring_pos_tp, value)

    def set_row_map(self, value):
        """Multi-batch: cache row per batch row, ordered before replay (see row_map)."""
        self._set_vec_all_devs(self.row_map_tp, value)

    def fill_tp(self, tensors, value):
        """Fill one int tensor per device with ``value`` (int scalar or per-batch-row
        list), ordered before the graph launch.

        ``tensors`` is a list parallel to ``get_split(BROADCAST_KV)``, i.e. one entry per device
        (entries may be None). These are the ``past_len_tp`` tensors handed to
        ``flash_attn_with_kvcache`` as ``cache_seqlens``.

        A bare ``fill_`` on a cuda:N tensor is issued on that device's *default* stream, which is
        unrelated to the TP stream the graph is launched on, so the graph can read the previous
        token's length. That is the same ordering bug as ``set_pos`` and it is what kept every
        replay attending to a frozen sequence length. Each fill is issued on that device's TP
        stream, and dev0's TP stream waits on an event recorded after it.
        """
        evs = []
        for _t in tensors:
            if _t is None:
                continue
            _dev = _t.device.index if isinstance(_t.device, torch.device) else int(_t.device)
            with torch.cuda.device(_dev):
                _s = self.streams_dev.get(_dev)
                if _s is None:
                    self._fill_vec(_t, value)
                    continue
                _prev = torch.cuda.current_stream(_dev)
                torch.cuda.set_stream(_s)
                try:
                    self._fill_vec(_t, value)
                    _ev = torch.cuda.Event()
                    _ev.record(_s)
                    evs.append((_dev, _ev))
                finally:
                    torch.cuda.set_stream(_prev)

        with torch.cuda.device(self.device):
            _s0 = self.streams_dev[self.device]
            for _dev, _ev in evs:
                if _dev == self.device:
                    continue
                _s0.wait_event(_ev)


    def geometry_key(self, layer_idx: int):
        """Attention geometry key for a layer: (head_dim, num_q_heads, num_kv_heads, is_full)."""
        return self.geometry_keys[layer_idx]

    def set_geometry(self, layer_idx: int):
        """
        Select the split set and C++ context matching this layer's attention geometry.

        No-op for single-geometry models, where every layer maps to the same entry.

        IMPORTANT: this must work *before* ``finalize()``. ``tp_split()`` and
        ``scratch_space_tp()`` both run during loading, when ``ext_tp_contexts`` has not been
        built yet. An earlier version returned early whenever ``ext_tp_context`` was None,
        which silently left every layer split with the FIRST geometry's ratios. For Gemma-4
        that gave layer 5 (4 kv heads, head_dim 512) the 4-kv-heads-per-device split inherited
        from layer 0 (16 kv heads), so ``k_proj`` was cut into four 2048-wide slices from a
        2048-wide tensor and the GEMM read past the end of the weight buffer. See Stage L
        bug 21 in IMPLEMENTED.md.
        """
        key = self.geometry_keys[layer_idx]
        if key == self.geometry_key_active:
            return
        self.geometry_key_active = key
        sp = self.geometry_splits[key]
        self.kv_split, self.kv_split_devs = sp["kv"]
        self.q_split, self.q_split_devs = sp["q"]
        # Contexts only exist after finalize(); during loading the split sets are all that is
        # needed, and `ext_tp_context` is read from this attribute at call time anyway.
        if self.ext_tp_contexts and key in self.ext_tp_contexts:
            self.ext_tp_context = self.ext_tp_contexts[key]

    def set_rank(self, rank: int | None):
        """
        Enter multiprocess mode for this process, owning device ``rank``.

        Must be called before any module loads, because get_split() starts filtering
        immediately and the loaders size their shards from it.
        """
        self.rank = rank

        # BUG 76: all_devs / device / num_devices were computed in define_split() while
        # rank was still None, so they describe the FULL device set. Every worker would then
        # think its primary device is 0 -- but load_tp_rank only allocates a context for its
        # own rank, so get_device_context(0) returns None on ranks 1-3.
        if rank is not None:
            self.all_devs = [rank]
            self.device = rank
            # Keep num_devices as the GLOBAL device count: it sizes per-device lists that are
            # indexed by global rank (out_features_tp etc.), and the loaders rely on it.
            self.num_devices = max(
                [d for d, _, _ in (self.kv_split or [])] + [rank]
            ) + 1

    def is_worker(self) -> bool:
        return self.rank is not None

    def get_split(self, broadcast_type: int):

        # GDN splits are global (not per-geometry): the delta-net head layout is
        # uniform across all linear layers.
        if broadcast_type == BROADCAST_GDN_K:
            full = self.gdn_k_split
        elif broadcast_type == BROADCAST_GDN_V:
            full = self.gdn_v_split
        elif self.geometry_key_active is not None:
            sp = self.geometry_splits[self.geometry_key_active]
            if broadcast_type == BROADCAST_KV: full = sp["kv"][0]
            elif broadcast_type == BROADCAST_Q: full = sp["q"][0]
            elif broadcast_type == BROADCAST_ID: full = sp["id"][0]
            elif broadcast_type == BROADCAST_RS: full = sp["rs"][0]
            elif broadcast_type == BROADCAST_VC: full = sp["vc"][0]
            else: full = None
        else:
            full = [
                self.kv_split,
                self.id_split,
                self.vc_split,
                self.rs_split,
                self.q_split
            ][broadcast_type]

        # Multiprocess mode: this process owns exactly one slice. Returning a one-element
        # list keeps (a, b) ABSOLUTE, so the shard geometry matches the single-process
        # split exactly and every existing `for dev, a, b in split` loop iterates once
        # over the right device with the right column range.
        if self.rank is not None:
            for d, a, b in full:
                if d == self.rank:
                    return [(d, a, b)]
            return []
        return full


    def get_split_full(self, broadcast_type: int):
        """
        The UNFILTERED split, i.e. every rank's slice.

        get_split() returns only this process's slice in worker mode. The collectives still need
        the whole picture: an all-gather has to know how wide every rank's contribution is, and
        the C++ contexts are built from the full geometry. BUG 71.
        """
        if broadcast_type == BROADCAST_GDN_K: return self.gdn_k_split
        if broadcast_type == BROADCAST_GDN_V: return self.gdn_v_split
        if self.geometry_key_active is not None:
            sp = self.geometry_splits[self.geometry_key_active]
            if broadcast_type == BROADCAST_KV: return sp["kv"][0]
            if broadcast_type == BROADCAST_Q: return sp["q"][0]
            if broadcast_type == BROADCAST_ID: return sp["id"][0]
            if broadcast_type == BROADCAST_RS: return sp["rs"][0]
            if broadcast_type == BROADCAST_VC: return sp["vc"][0]
        return [
            self.kv_split,
            self.id_split,
            self.vc_split,
            self.rs_split,
            self.q_split
        ][broadcast_type]

    def get_devs(self, broadcast_type: int):

        if broadcast_type == BROADCAST_GDN_K: return self.gdn_k_split_devs
        if broadcast_type == BROADCAST_GDN_V: return self.gdn_v_split_devs
        return [
            self.kv_split_devs,
            self.id_split_devs,
            self.vc_split_devs,
            self.rs_split_devs,
            self.q_split_devs
        ][broadcast_type]


    def get_temp_tensors_bc(self, rows: int, dtype: torch.dtype, broadcast_type: int, dim: int = 1):

        split = self.get_split(broadcast_type)
        dim = split[-1][2] * dim
        return [torch.empty((rows, dim), device = dev, dtype = dtype) for dev, _, _ in split]


    def get_temp_tensors_bc_s(self, rows: int, esize: int, broadcast_type: int, dim: int = 1):

        scratch = [0] * self.num_devices
        split = self.get_split(broadcast_type)
        dim = split[-1][2] * dim
        for dev, _, _ in split:
            scratch[dev] = rows * dim * esize
        return scratch


    def get_temp_tensors(self, rows: int, dtype: torch.dtype, broadcast_type: int, dim: int = 1):

        split = self.get_split(broadcast_type)
        return [torch.empty((rows, (b - a) * dim), device = dev, dtype = dtype) for dev, a, b in split]


    def get_temp_tensors_s(self, rows: int, esize: int, broadcast_type: int, dim: int = 1):

        scratch = [0] * self.num_devices
        split = self.get_split(broadcast_type)
        for dev, a, b in split:
            scratch[dev] = rows * (b - a) * dim * esize
        return scratch


    def _sync_all_devices(self):
        """
        Wait for every TP device's stream to drain.

        The pinned staging buffers in ``pinned_temp`` are shared by all devices and are used
        for asynchronous D2H (gather) and H2D (broadcast) copies. Nothing else orders those
        copies against the host-side reads and writes that follow, so both boundaries must be
        synchronised explicitly. BUG 35.

        Per-device, not a bare ``torch.cuda.synchronize()``: a gather writes from every
        device, and a plain synchronize only waits on the current one.
        """
        for dev in self.all_devices():
            torch.cuda.synchronize(dev)

    def get_pinned(self, buffer: int, batch_size: int, q_len: int, dim: int):

        pt = self.pinned_temp[buffer][:batch_size * q_len * dim]
        pt = pt.view(batch_size, q_len, dim)
        return pt


    def broadcast(
        self,
        buffer: int,
        input_tensor: torch.Tensor,
        broadcast_type: int,
        dim: int = 1
    ):
        # Multiprocess mode: every rank already holds the full-width residual stream (it is
        # reconstructed by gather()'s all_gather), so a broadcast is the identity. This is
        # the whole point -- it deletes the pinned D2H/H2D round trip from the hot path.
        if self.rank is not None:
            return [input_tensor]

        split = self.get_split(broadcast_type)

        # BUG 40: no host-side drain here any more. tp_broadcast now records an event on the
        # source stream after its D2H and makes every target stream wait on it before its H2D
        # (ext_tp.cpp), so the pinned buffer is correctly ordered without the host blocking.
        # The bug-35 sync is still required in gather(), which hands a host view to the caller.

        # Reuse the per-device destination buffers. Allocating here would put a
        # torch.empty on the hot path of every module (and, more importantly, is not
        # allowed inside a CUDA graph capture region). BUG 110: in graph mode use the
        # row-keyed, never-evicted graph cache so the bonus (q_len=1) and verify (q_len=W)
        # graphs keep distinct stable buffers; eager mode keeps the realloc cache.
        _specs = [(idx, input_tensor.shape[0], input_tensor.shape[1], input_tensor.dtype)
                  for idx, _, _ in split]
        if self.graph_mode:
            if self._bc_cache_graph is None:
                self._bc_cache_graph = {}
            bc_tensors = tp_reuse_buffers(self._bc_cache_graph, _specs, key_rows = True)
        else:
            if self._bc_cache is None:
                self._bc_cache = {}
            bc_tensors = tp_reuse_buffers(self._bc_cache, _specs, eager_large_ok = True)

        ext_c.tp_broadcast(
            self.ext_tp_context,
            buffer,
            input_tensor,
            broadcast_type,
            bc_tensors,
            dim,
            -1
        )

        return bc_tensors


    def gather(
        self,
        buffer: int,
        inputs: list[torch.Tensor],
        broadcast_type: int,
        dim: int = 1

    ):
        # Multiprocess mode: one NCCL all_gather along dim 1 reassembles the full width.
        # `inputs` holds this rank's shard only.
        if self.rank is not None:
            return self._gather_nccl(inputs[0], broadcast_type, dim)

        split = self.get_split(broadcast_type)

        # CUDA-graph mode: keep the result on the primary device and skip the host drain.
        # Both the drain and the host view are illegal inside a capture region, and the host
        # round trip is what forces the next module to re-broadcast from pinned memory.
        # BUG 43.
        if self.graph_mode:
            if self._gather_cache_graph is None:
                self._gather_cache_graph = {}
            return self.gather_dev(
                buffer, inputs, broadcast_type, dim, cache = self._gather_cache_graph
            )

        ext_c.tp_gather(
            self.ext_tp_context,
            buffer,
            inputs,
            broadcast_type,
            [],
            -1,
            dim,
            -1
        )

        # BUG 35: tp_gather only *issues* the D2H copies. The caller reads the host view
        # returned below, so they must have landed first.
        self._sync_all_devices()

        pt = self.pinned_temp[buffer][:split[-1][2] * dim * inputs[0].shape[0]]
        pt = pt.view(inputs[0].shape[0], split[-1][2] * dim)
        return pt

    def _gather_nccl(self, local: torch.Tensor, broadcast_type: int, dim: int = 1,
                     tag: int = 0) -> torch.Tensor:
        """
        All-gather the per-rank shards of a column-parallel projection into full width.

        Each rank contributes (rows, (b - a) * dim). The shards are contiguous column
        ranges of the full tensor, so concatenating them in rank order along dim 1
        reproduces the single-process result exactly.

        ``tag`` separates the two callers that can otherwise collide on the cache key:
        ``allgather`` (BROADCAST_Q in, full width out, feeding o_proj) and ``gather``
        (BROADCAST_Q in, full width out, the layer output) have the same broadcast_type,
        row count and width, so without a tag they would share one buffer and the second
        call would clobber the first's result.
        """
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return local

        # BUG 71: the FULL split. get_split() is filtered to this rank in worker mode, so
        # using it here made len(split) == 1 and skipped the gather entirely, leaving the
        # residual stream at shard width.
        split = self.get_split_full(broadcast_type)
        if len(split) <= 1:
            # Only one rank owns a slice of this split (e.g. a geometry where the other
            # ranks get zero heads). Nothing to gather.
            return local

        # BUG 78: a REPLICATED split (BROADCAST_RS) gives every rank the FULL width --
        # [(0,0,W), (1,0,W), ...] with identical ranges. All-gathering it would produce a
        # 4W-wide tensor and corrupt everything downstream. The local tensor is already the
        # full result, so return it as-is.
        _ranges = set((a, b) for _, a, b in split)
        if len(_ranges) == 1:
            return local

        world = dist.get_world_size()
        if world == 1:
            return local

        widths = [0] * world
        for d, a, b in split:
            widths[d] = (b - a) * dim

        # all_gather needs a contiguous input.
        if not local.is_contiguous():
            local = local.contiguous()

        rows = local.shape[0]
        dtype = local.dtype

        # PCIe P2P fast path: small equal-split gathers (decode/verify sizes) run as
        # one-sided P2P writes + flag spins (~10-20 us vs NCCL's ~70 us). Large
        # prefill gathers and uneven splits stay on NCCL below. Symmetric decision
        # on every rank (same shapes, same enabled flag agreed at init).
        _p2p = self.p2p
        if _p2p is not None and _p2p.enabled \
                and len(set(widths)) == 1 and widths[0] == local.shape[1]:
            _out = _p2p.allgather(local, (tag, broadcast_type, widths[0], dtype),
                                  (rows, world * widths[0]))
            if _out is not None:
                return _out

        if self._ag_cache is None:
            self._ag_cache = {}

        # BUG 82: cache the output buffer and the per-rank view list on the context,
        # keyed by (broadcast_type, rows, width, dtype), and skip the pad/trim entirely
        # when every rank has the same width.
        #
        # Measured, 4x 3090 over PCIe (6.3 GB/s/rank achievable):
        #   prefill rows=1032, w=8192   8.264 ms -> 8.132 ms   (1.6% -- noise)
        #   decode  rows=1,    w=8192   0.135 ms -> 0.072 ms   (47%)
        # So this is NOT a prefill fix. Prefill is bandwidth-bound: the collectives move
        # ~3.4 GB/rank and already run at ~90% of the PCIe ceiling, which is why the
        # fresh `parts` list and `torch.cat` cost almost nothing there. The win is at
        # decode, where the tensors are tiny, the collective is launch-bound rather than
        # bandwidth-bound, and there are ~180 of them per token.
        #
        # Reuse is safe because the cache key includes broadcast_type and ``tag``, and no
        # module gathers the same (type, width) twice before consuming the previous result.
        #
        # ALIASING CONTRACT -- the returned tensor is a cached buffer. It is double-buffered
        # and alternated per key, because the caller can still be reading the previous
        # result when the next gather runs: a layer's output IS the next layer's residual
        # stream, and it is consumed by add_residual after the following layer has already
        # issued its own gathers. Two buffers give a full layer of slack, which is more than
        # the module order needs (per layer: attn[allgather Q tag=1] -> attn[gather Q tag=0]
        # -> mlp[allgather ID] -> mlp[gather RS]; a buffer written by one of those is read
        # only before the next write of the SAME key, two steps later).
        dev = local.device
        if not local.is_contiguous():
            local = local.contiguous()
        # BUG 122: ``rows`` is part of the key and entries are NEVER evicted, so every
        # distinct prefill row count pinned a permanent ~0.5 GB entry (2 assembled buffers
        # + 2*world part buffers at chunk 4096). Prompt-cache continuation prefills have a
        # different tail length every turn, so the cache grew without bound until the
        # master OOMed mid-prefill (debug1000.txt: dev0 22.3 GB allocated while prefilling
        # only 797 tokens). Same class as BUG 120, in the allgather path that never got the
        # eager_large_ok treatment: large EAGER shapes take fresh uncached buffers, consumed
        # within the layer and recycled by the caching allocator. Graph mode and
        # decode/verify-sized rows (<= 32) keep the row-keyed stable-address cache whose
        # addresses the captured graphs bake. Fresh buffers also remove the aliasing
        # hazard entirely (nothing is reused), and prefill gains nothing from the reuse
        # anyway (measured 1.6% = noise, see above).
        _eager_large = rows > 32 and not self.graph_mode
        if len(set(widths)) == 1 and widths[0] == local.shape[1]:
            w = widths[0]
            key = (tag, broadcast_type, rows, w, dtype, dev)
            ent = None if _eager_large else self._ag_cache.get(key)
            if ent is None:
                _nbuf = 1 if _eager_large else 2
                bufs = [torch.empty((rows, world * w), dtype = dtype, device = dev)
                        for _ in range(_nbuf)]
                # BUG 101: all_gather outputs MUST be contiguous. Column slices of the
                # assembled buffer are strided views, which torch rejects ("Tensors must
                # be contiguous"), so gather into standalone per-rank tensors and assemble
                # with device-local slice copies afterwards. The copies run at HBM speed
                # (~1 TB/s) vs the PCIe collective (~6 GB/s), so they add negligible cost.
                parts = [[torch.empty((rows, w), dtype = dtype, device = dev)
                          for _ in range(world)] for _ in range(_nbuf)]
                ent = [bufs, parts, 0]
                if not _eager_large:
                    self._ag_cache[key] = ent
            bufs, parts, i = ent
            out = bufs[i]
            ent[2] = 1 - i
            # NCCL all_gather requires non-overlapping input and output. If the caller
            # handed us a view of the buffer we are about to write, copy it out first.
            if local.data_ptr() >= out.data_ptr() and \
               local.data_ptr() < out.data_ptr() + out.numel() * out.element_size():
                local = local.clone()
            with _CollTimer(_PREFILL_TRACE and rows > 32 and not self.graph_mode):
                dist.all_gather(parts[i], local)
            for r in range(world):
                out[:, r * w:(r + 1) * w].copy_(parts[i][r])
            return out

        # Uneven widths (a geometry where heads do not divide evenly between ranks):
        # pad to the max width, gather, then trim each rank back to its real width.
        max_w = max(widths)
        key = (tag, broadcast_type, rows, max_w, dtype, dev)
        ent = None if _eager_large else self._ag_cache.get(key)
        if ent is None:
            padded = torch.empty((rows, max_w), dtype = dtype, device = dev)
            parts = [torch.empty_like(padded) for _ in range(world)]
            if not _eager_large:
                self._ag_cache[key] = (padded, parts)
        else:
            padded, parts = ent
        padded.zero_()
        padded[:, :local.shape[1]] = local
        with _CollTimer(_PREFILL_TRACE and rows > 32 and not self.graph_mode):
            dist.all_gather(parts, padded)
        trimmed = [parts[d][:, :widths[d]] for d in range(world) if widths[d] > 0]
        return torch.cat(trimmed, dim = 1)

    # ------------------------------------------------------------------ seqpar residual

    def shard_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequence-parallel: slice a full-width hidden tensor to this rank's residual shard.

        The RS split partitions the hidden dim; every rank keeps only its slice of the
        residual stream. Pure view, no collective -- the embedding is replicated, so each
        rank can slice its own range locally.
        """
        split = self.get_split(BROADCAST_RS)
        assert len(split) == 1, "shard_residual requires rank-local (MP) mode"
        _, a, b = split[0]
        return x[..., a:b]

    def gather_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequence-parallel: allgather a residual shard back to full hidden width.

        Used at the two points where full width is structurally required: before a
        column-parallel projection's input norm (q/k/v/gate/up consume the whole hidden
        dim) and before the final norm / lm_head / MTP state extraction. Tag 2 keeps the
        cached destination distinct from the projection gathers (tags 0/1). The result is
        a cached, double-buffered device tensor (graph-stable, BUG 82/110 pattern).
        """
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        full = self._gather_nccl(x2, BROADCAST_RS, 1, tag = 2)
        return full.view(*shape[:-1], full.shape[-1])

    def ag_norm_res(self, shard: torch.Tensor, tag: int, weight: torch.Tensor,
                    residual: torch.Tensor, eps: float):
        """
        Fused decode_repl block tail: allgather this rank's RS output shard to full hidden
        width, RMSNorm (full weight), add full residual -- one P2P kernel instead of
        AG + norm + add. Returns the full-width fp16 result, or None to fall back to the
        unfused path (P2P off / unsupported shape / non-fp16). ``tag`` separates the attn
        and mlp call sites (distinct arena slots + output buffers). The output buffer is
        cached per (tag, rows, hidden) and graph-stable (BUG 82/110 pattern).
        """
        p2p = self.p2p
        if p2p is None or not p2p.enabled:
            return None
        if shard.dtype != torch.half or residual.dtype != torch.half \
                or weight.dtype != torch.half:
            return None
        split = self.get_split_full(BROADCAST_RS)
        widths = [(b - a) for _, a, b in split]
        if len(set(widths)) != 1 or shard.shape[1] != widths[0]:
            return None
        rows = shard.shape[0]
        hidden = widths[0] * len(split)
        key = (tag, rows, hidden)
        bufs = getattr(self, "_agnr_cache", None)
        if bufs is None:
            bufs = self._agnr_cache = {}
        # BUG 122 (same class as _ag_cache): row-keyed and never evicted, so large eager
        # prefill shapes must not enter it -- fresh buffer per call, recycled after use.
        _eager_large = rows > 32 and not self.graph_mode
        out = None if _eager_large else bufs.get(key)
        if out is None:
            with torch.inference_mode(False):
                out = torch.empty((rows, hidden), dtype=torch.half, device=shard.device)
            if not _eager_large:
                bufs[key] = out
        return p2p.ag_norm_res(shard, (tag, BROADCAST_RS, widths[0], torch.half),
                               (rows, hidden), weight, residual, out, eps)

    def reduce_scatter_hidden(self, partial: torch.Tensor) -> torch.Tensor:
        """
        Sequence-parallel row-parallel reduction: sum full-width partial outputs across
        ranks and return this rank's residual (RS) shard.

        ``partial`` is (rows, hidden_size) -- the partial sum a row-parallel projection
        (down_proj / o_proj) produced from this rank's input shard. The RS split partitions
        the hidden dim, so the reduce-scatter runs on the TRANSPOSED (hidden, rows) view:
        reduce_scatter_tensor splits dim 0, which must be the hidden dim. The two transpose
        copies move rows*hidden*2 bytes at HBM speed (~0.02 ms at prefill shapes), against
        the PCIe collective they enable.

        Volume: (world-1)/world * rows*hidden*2 per rank -- 4x less than allgathering the
        intermediate activations (width 4*hidden) that the column-parallel down_proj needs.

        Uneven RS splits (gpu_split-derived ranges are not always equal) fall back to
        all_reduce + local slice: still less volume than the intermediate allgather, and
        always aligned with get_split_full(BROADCAST_RS).

        Graph-capturable: NCCL reduce_scatter/all_reduce are the same primitives the
        captured decode already uses; the transpose copies allocate from the graph pool.
        """
        import torch.distributed as dist
        assert self.rank is not None, "reduce_scatter_hidden requires rank-local (MP) mode"
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return partial
        rows, h = partial.shape
        split = self.get_split_full(BROADCAST_RS)
        widths = [b - a for _, a, b in split]
        world = dist.get_world_size()
        _trace = _PREFILL_TRACE and rows > 32 and not self.graph_mode
        if len(widths) == world and len(set(widths)) == 1 and h == world * widths[0]:
            pt = partial.t().contiguous()
            out = torch.empty((widths[0], rows), dtype = partial.dtype, device = partial.device)
            with _CollTimer(_trace):
                dist.reduce_scatter_tensor(out, pt)
            return out.t().contiguous()
        with _CollTimer(_trace):
            dist.all_reduce(partial, op = dist.ReduceOp.SUM)
        _, a, b = self.get_split(BROADCAST_RS)[0]
        return partial[:, a:b].contiguous()

    def all_reduce_hidden(self, partial: torch.Tensor, key: str = "__ar__") -> torch.Tensor:
        """
        Full-width SUM all-reduce of a row-parallel projection's (rows, hidden) partial,
        for the decode_repl row-parallel path (EXL2_TP_ROW_REPL): replaces the reassembly
        allgather + RS-shard allgather (2 collectives) with ONE allreduce. Uses the PCIe
        P2P allreduce (per-key slotted scratch, deferred-ack, graph-safe) when available,
        else NCCL. ``key`` separates call sites (o_proj vs down) so their captured scratch
        regions never alias. In-place on the P2P path; returns the reduced full tensor.
        """
        import torch.distributed as dist
        assert self.rank is not None, "all_reduce_hidden requires rank-local (MP) mode"
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return partial
        p2p = self.p2p
        if p2p is not None and p2p.enabled and p2p.allreduce(partial, key):
            return partial
        with _CollTimer(_PREFILL_TRACE and partial.shape[0] > 32 and not self.graph_mode):
            dist.all_reduce(partial, op = dist.ReduceOp.SUM)
        return partial

    def gather_dev(
        self,
        buffer: int,
        inputs: list[torch.Tensor],
        broadcast_type: int,
        dim: int = 1,
        cache: dict | None = None
    ) -> torch.Tensor:
        """
        Gather the per-device slices into one full-width tensor that stays **on device 0**.

        This is the CUDA-graph-safe counterpart of ``gather()``. ``gather()`` returns a *host*
        view of ``pinned_temp``, which forces two things that make the decode step
        uncapturable:

        1. ``_sync_all_devices()`` -- a host drain in the middle of the step.
        2. A host round trip per module: the caller reads the pinned buffer, and the next
           module broadcasts it back to the devices. ~120 of those per token.

        ``tp_gather_barrier`` already lands the assembled result in its ``targets`` device
        tensors and orders the D2H against the H2D with CUDA events, so no host involvement is
        needed at all. This wrapper just returns that device tensor instead of the pinned one.

        ``cache`` (optional) supplies reusable output buffers keyed by
        ``(device, rows, cols, dtype)`` so repeated calls reuse the same addresses, which is
        required for graph capture.
        """
        split = self.get_split(broadcast_type)
        rows = inputs[0].shape[0]
        cols = split[-1][2] * dim
        dtype = inputs[0].dtype

        if cache is not None:
            # BUG 110: gather_dev is the graph-mode path; key by rows so each captured
            # q_len shape keeps its own stable buffer (never reallocated for another shape).
            targets = tp_reuse_buffers(
                cache,
                [(dev, rows, cols, dtype) for dev, _, _ in split],
                key_rows = True
            )
        else:
            targets = [
                torch.empty((rows, cols), device = dev, dtype = dtype)
                for dev, _, _ in split
            ]

        ext_c.tp_gather(
            self.ext_tp_context,
            buffer,
            inputs,
            broadcast_type,
            targets,
            broadcast_type,
            dim,
            -1
        )

        # No host sync: the D2H/H2D pair inside tp_gather_barrier is ordered by the
        # cross-device event barrier, and the result is consumed on-device.
        return targets[0]

    # @profile
    def allgather(
        self,
        buffer,
        inputs: list[torch.Tensor],
        broadcast_type_g: int,
        broadcast_type_b: int,
        dim: int = 1
    ):
        if self.rank is not None:
            return [self._gather_nccl(inputs[0], broadcast_type_g, dim, tag = 1)]

        # split_g = self.get_split(broadcast_type_g)
        split_b = self.get_split(broadcast_type_b)
        sh = (inputs[0].shape[0], split_b[-1][-1] * dim)
        dtype = inputs[0].dtype

        bc_tensors = [torch.empty(sh, device = dev, dtype = dtype) for dev, _, _ in split_b]

        ext_c.tp_gather(
            self.ext_tp_context,
            buffer,
            inputs,
            broadcast_type_g,
            bc_tensors,
            broadcast_type_b,
            dim,
            -1
        )

        return bc_tensors


    def copy_pinned(
        self,
        buffer: int,
        inputs: torch.Tensor
    ):
        pt = self.pinned_temp[buffer][:inputs.numel()]
        pt = pt.view(inputs.shape)
        pt.copy_(inputs)
        return pt


    def all_reduce_split(
        self,
        tensors: list[torch.Tensor],
        broadcast_type: int,
        dim: int = 1
    ):
        """
        Sum ``tensors`` across devices, where each device holds only its own slice of the
        output (a partial sum over its input slice). Returns one full-width tensor.

        This is the reduce half of Megatron row-parallelism. exl2's fused TP kernel only
        implements column-parallel projections, so a row-parallel layer (one whose
        ``out_features`` is not ``num_heads * head_dim``, e.g. Gemma-4's ``o_proj``) needs
        this in Python. PLAN.md §2.5.
        """
        split = self.get_split(broadcast_type)
        dev0 = split[0][0]
        context = self.model.get_device_context(dev0)
        torch.cuda.set_stream(context.stream)

        # Start from device 0's slice, then add every other device's slice into it.
        acc = tensors[0].clone()
        for idx in range(1, len(split)):
            dev, a, b = split[idx]
            ctx_d = self.model.get_device_context(dev)
            torch.cuda.set_stream(ctx_d.stream)
            part = tensors[idx].to(dev0, non_blocking = True)
            acc.add_(part)
            part = None

        torch.cuda.set_stream(context.stream)
        return acc

    def add_residual(
        self,
        target: list[torch.Tensor],
        source: list[torch.Tensor],
        broadcast_type: int,
        dim: int = 1
    ):
        split = self.get_split(broadcast_type)

        for idx, (dev, a, b) in enumerate(split):
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)
            target[idx].add_(source[idx][:, a * dim : b * dim])


    def wait_streams(self, device: int | None = None):
        """
        Wait for the TP step to finish.

        ``device`` limits the wait to the device that owns the result. The caller only ever
        needs the logits, which live on the primary device, and the gather's event chain has
        already ordered every other device's contribution into it. Draining all four devices
        plus a global ``torch.cuda.synchronize()`` on every token was pure host stall: it
        pegged one core at 100% of wall time while the profiler showed 0.39 ms/token of real
        aten work.
        """
        if self.rank is not None:
            # One device per process: drain only our own stream.
            global_streams[self.rank].synchronize()
            return
        if device is not None:
            global_streams[device].synchronize()
            return
        for dev in self.all_devs:
            s = global_streams[dev]
            s.synchronize()
        torch.cuda.synchronize()


    def get_sin_cos(self):
        if self.sin is not None:
            return self.sin, self.cos
        self.sin = []
        self.cos = []
        for dev in range(self.num_devices):
            if dev in self.all_devs:
                devctx = self.model.get_device_context(dev)
                self.sin.append(torch.cat(devctx.sin, 0))
                self.cos.append(torch.cat(devctx.cos, 0))
            else:
                self.sin.append(none_tensor)
                self.cos.append(none_tensor)
        return self.sin, self.cos


    def begin_scratch_alloc_tp(self):

        for devctx in self.model.device_context:
            if devctx:
                devctx.begin_scratch_alloc()


    def _devctx_or_none(self, dev: int):
        """
        Return this device's context, or None if this process does not own that device.

        In the single-threaded TP path every device has a context and this is just
        get_device_context(). In a multiprocess worker only one device does, and the scratch helpers
        must skip the rest instead of dereferencing None.
        """
        contexts = self.model.device_context
        if contexts and dev < len(contexts) and contexts[dev] is not None:
            return contexts[dev]
        return None


    def get_scratch_slice_tp_bc(self, rows: int, dtype: torch.dtype, broadcast_type: int, dim: int = 1):

        split = self.get_split(broadcast_type)
        dim = split[-1][2] * dim
        if dtype == torch.half: esize = 2
        if dtype == torch.float: esize = 4

        tensors = []
        for dev, _, _ in split:
            devctx = self._devctx_or_none(dev)
            if devctx is None:
                # A single-rank worker: keep the list aligned with the split but do not allocate.
                tensors.append(None)
                continue
            size_bytes = rows * dim * esize
            tensor = devctx.get_scratch_slice(size_bytes)
            tensor = tensor.view(rows, dim)
            tensors.append(tensor)

        return tensors


    def get_scratch_slice_tp(self, rows: int, dtype: torch.dtype, broadcast_type: int, dim: int = 1):

        split = self.get_split(broadcast_type)
        if dtype == torch.half: esize = 2
        if dtype == torch.float: esize = 4

        tensors = []
        for dev, a, b in split:
            devctx = self._devctx_or_none(dev)
            if devctx is None:
                tensors.append(None)
                continue
            size_bytes = rows * (b - a) * dim * esize
            tensor = devctx.get_scratch_slice(size_bytes)
            tensor = tensor.view(rows, (b - a) * dim)
            tensors.append(tensor)

        return tensors


    def reserve_scratch(self, scratch: list[int]):

        for dev, s in enumerate(scratch):
            if s == 0: continue
            devctx = self._devctx_or_none(dev)
            if devctx is None: continue
            devctx.get_scratch_slice(s)
