"""
CUDA graph capture/replay for the tensor-parallel decode step.

Why this exists
---------------
TP decode is launch-latency bound, not bandwidth bound: the per-token payloads are 10-43 KB
and the step issues ~240 modules' worth of small kernels plus host-staged collectives. The
measured symptom is GPUs sitting at 10-15% while one CPU core runs at 101%. Capturing the
whole step into one graph removes essentially all of that per-kernel host submission cost.

Why the capture uses torch's raw CUDAGraph API rather than ``torch.cuda.graph()``
--------------------------------------------------------------------------------
The ``torch.cuda.graph()`` context manager refuses multi-device capture: it asserts that
capture begins and ends on the same stream, and work issued to a non-capture stream
invalidates it. ``torch.cuda.CUDAGraph.capture_begin()/capture_end()`` do not have that
restriction, and they still install torch's private graph memory pool, so allocations made
inside the capture region are owned by the graph and are not recycled by the caching allocator.
That last part matters: with a bare ``cudaStreamBeginCapture`` the allocator knows nothing about
the capture, recycles the graph's buffers, and the replayed graph silently reads stale memory
(which presents as fluent-looking garbage -- the bug-39 signature).

What had to change to make capture legal
---------------------------------------
Three things in the existing TP path were illegal inside a capture region, and each has a
device-side replacement (BUG 43):

1. ``TPContext.gather()`` called ``_sync_all_devices()`` -- a host drain. In graph mode it
   returns a device tensor instead and skips the drain (``gather_dev``).
2. ``gather()`` returned a *host* view of ``pinned_temp``, so the next module re-broadcast it
   to the devices: a host round trip per module. Graph mode keeps the residual stream on the
   primary device.
3. ``past_len`` was a Python int baked into the RoPE kernel args and the KV-cache write slice.
   Graph mode reads it from a device int32 tensor (``TPContext.pos``) instead, so one captured
   graph replays correctly at every step.

Design
------
* The *decode* step (q_len == 1) is captured, plus two MTP shapes (Change98): the bonus
  commit (q_len == 1, also returns the pre-final-norm state via a static buffer) and the
  fixed-window verify forward (q_len == W, marked explicitly by the caller). Prefill keeps
  the eager path: its shapes vary with the prompt, and it is not the bottleneck.
* The graph is keyed on (batch, q_len, state, last_id_only). A shape change invalidates it.
* Inputs are written into static buffers before replay; the graph is position-agnostic because
  every position-dependent value comes from ``TPContext.pos``.
* ``EXL2_TP_GRAPHS=0`` disables the whole thing and restores the eager path exactly.
"""

from __future__ import annotations

import faulthandler
import gc
import os
import torch

from exllamav2.ext import exllamav2_ext as ext_c

# BUG 110: dump the Python stack of every thread on SIGSEGV so a graph-replay crash
# pinpoint the offending runner/key instead of only the native cuGraphLaunch frame.
faulthandler.enable()


def graphs_enabled() -> bool:
    """Whether TP decode should be captured. On by default; EXL2_TP_GRAPHS=0 disables."""
    return os.environ.get("EXL2_TP_GRAPHS", "1") != "0"


class TPGraphRunner:
    """
    Captures the TP decode step once and replays it per token.

    The caller supplies a zero-argument ``step_fn`` that runs the whole decode step eagerly
    (writing its result into buffers it owns). ``step_fn`` is called a few times for warmup, once
    inside the capture region, and then never again -- replay is used from then on.
    """

    def __init__(self, step_fn, device: int, key: tuple, tp_context):
        self.step_fn = step_fn
        self.device = device
        self.key = key
        self.tp_context = tp_context
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.num_replays = 0
        # BUG 117: kv length (past_len + seq_len) at capture; the caller recaptures when a
        # step falls outside this envelope (the graph-mode attention launch is capture-baked).
        self.capture_kv = 0
        # Set by the caller so capture() can restore the KV cache position after its warmup
        # iterations. The warmup runs the real step, which advances cache.current_seq_len and
        # writes KV rows; without restoring it the cache is left several tokens ahead and every
        # subsequent replay writes KV at the wrong row.
        self.cache = None
        # Change98: per-runner static buffers. The graph holds their addresses, so they are
        # allocated once at capture and never reallocated. input_buf is the staged embedding;
        # logits is the capture-time output tensor the graph rewrites on every replay;
        # state_buf (MTP bonus graph only) receives the pre-final-norm hidden state.
        self.input_buf = None
        self.logits = None
        self.state_buf = None

    def capture(self):
        """
        Warm up, then capture. Raises on failure.

        Warmup matters: cuBLAS workspaces, the GEMM autotuner and the torch allocator's pools
        must all be settled before capture, because a first-touch allocation inside the capture
        region is an error, and the GEMM autotuner host-synchronises (which is illegal there).

        Capture happens on dev0's *persistent* TP stream (``ctx.streams_dev[dev0]``), not on a
        freshly created one. That is the whole trick: the TP modules already do
        ``torch.cuda.set_stream(context.stream)`` before every kernel, and ``context.stream`` for
        dev0 is exactly this stream. Capturing here therefore records dev0's torch work and
        dev0's C++ kernel launches without remapping anything.

        Capturing on a *different* stream was the original bug. dev0's kernels still went to
        ``streams_dev[dev0]``, so ``tp_cross_device_barrier`` recorded ``sync_events[dev0]`` on a
        stream outside the capture, and every captured device stream that waited on that event
        aborted with "dependency created on uncaptured work in another stream".
        """
        dev0 = self.tp_context.device
        ctx = self.tp_context

        # Capture on dev0's real TP stream. It is a non-default stream (created with
        # priority -100 in device.py), which is what torch's capture API requires.
        self.capture_stream = ctx.streams_dev[dev0]

        # Deferred-ack P2P protocol: graphed allgathers skip the per-op ack spin, so
        # every execution of the step (warmup iterations AND the captured body) must
        # start with the barrier that waits for all peers' acks of the previous round.
        _p2p = getattr(ctx, "p2p", None)
        if _p2p is not None and _p2p.enabled:
            _inner = self.step_fn
            def _barried_step():
                _p2p.barrier()
                _inner()
            self.step_fn = _barried_step

        # The warmup below runs the real step, which advances the KV cache position and writes
        # KV rows. Save the position so it can be restored afterwards: the caller expects the
        # cache to be exactly where it was, and a stale position makes every replay write KV at
        # the wrong row (the graph-first divergence at step 2).
        _saved_seq_len = self.cache.current_seq_len if self.cache is not None else None

        # Warm up on the real per-device streams, with dev0's stream current so the
        # torch-current-stream kernels warm up where they will run during capture.
        prev = torch.cuda.current_stream(dev0)
        torch.cuda.set_stream(self.capture_stream)
        try:
            for _ in range(3):
                self.step_fn()
        finally:
            torch.cuda.set_stream(prev)
        torch.cuda.synchronize(dev0)

        # Undo the warmup's cache advance before capturing, so the captured step runs at the
        # caller's real position and the graph's KV writes land where the caller expects.
        if _saved_seq_len is not None:
            self.cache.current_seq_len = _saved_seq_len

        # Capture. dev0's work is already on the capture stream. The other devices' streams are
        # forked from it with an event and joined back afterwards, which is what brings their
        # work into the graph.
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(dev0)
        # BUG 100: drop stale Python temporaries BEFORE capture. With the cudaMallocAsync
        # backend, a pre-capture tensor freed inside the capture region triggers
        # "freeAsync() was called on an uncaptured allocation" and can invalidate the
        # capture or corrupt replay; GC timing made this fire non-deterministically.
        gc.collect()
        torch.cuda.synchronize(dev0)
        torch.cuda.set_stream(self.capture_stream)

        # BUG 110: force the graph-mode collective destination buffers to be allocated
        # INSIDE the capture region (graph-pool-owned). The warmup above populated the
        # row-keyed graph caches with REGULAR (pre-capture) allocations; reusing them during
        # capture bakes pre-capture addresses into the graph nodes, and a later reallocation
        # frees that memory -> cuGraphLaunch segfault on replay (seen on SP's multi-device
        # capture). Emptying the caches here makes capture allocate fresh pool-owned buffers
        # that live as long as the graph. Previously captured graphs keep their OWN pool
        # memory alive independently, so dropping their cache entries does not free them.
        for _gc in (getattr(ctx, "_bc_cache_graph", None),
                    getattr(ctx, "_gather_cache_graph", None)):
            if _gc is not None:
                _gc.clear()
        # BUG 121: same treatment for the per-module GEMM/norm output graph caches
        # (linear/rmsnorm _tp_out_cache_graph): clear so each capture allocates fresh
        # pool-owned buffers. Previously captured graphs keep their own pool memory
        # alive independently, so dropping their cache entries does not free them.
        for _m in ctx.model.modules:
            _gc = getattr(_m, "_tp_out_cache_graph", None)
            if _gc is not None:
                _gc.clear()

        # BUG 110 (SP multi-device): capture_begin pools allocations on the CURRENT device
        # (dev0) only. In the single-process TP path the step also allocates on dev1-3; without
        # routing those to the graph pool they are REGULAR allocations that get freed/recycled
        # after capture, so a later replay dereferences freed memory -> cuGraphLaunch segfault
        # (the bonus graph crashed on its second replay, after the verify capture recycled its
        # dev1-3 buffers). Explicitly begin allocation-to-pool on every other device for the
        # duration of the capture so ALL graph buffers are pool-owned (live as long as the graph).
        _pool = torch.cuda.graph_pool_handle()
        _pooled_devs = [d for d in ctx.all_devices() if d != dev0]
        for _d in _pooled_devs:
            torch._C._cuda_beginAllocateToPool(_d, _pool)

        try:
            self.graph.capture_begin(pool = _pool)

            # Tell the C++ context which stream the capture is on, so the host-staged D2H in
            # tp_broadcast is issued on a stream that belongs to the capture rather than on the
            # torch current stream (which would invalidate it).
            for _c in (ctx.ext_tp_contexts.values() if ctx.ext_tp_contexts
                       else [ctx.ext_tp_context]):
                if _c:
                    ext_c.tp_set_capture_stream(_c, self.capture_stream.cuda_stream)

            # Fork the per-device TP streams from the capture stream.
            fork_ev = torch.cuda.Event()
            fork_ev.record(self.capture_stream)
            for dev in ctx.all_devices():
                if dev == dev0:
                    continue
                ctx.streams_dev[dev].wait_event(fork_ev)

            try:
                self.step_fn()
            finally:
                # Join every forked stream back before ending the capture.
                for dev in ctx.all_devices():
                    if dev == dev0:
                        continue
                    join_ev = torch.cuda.Event()
                    join_ev.record(ctx.streams_dev[dev])
                    self.capture_stream.wait_event(join_ev)

            # The modules call set_stream() per device and do not restore it, so the current
            # stream after step_fn() is whichever device ran last. torch's capture_end()
            # requires the capture to end on the stream it began on, so make the capture
            # stream current again before ending it.
            torch.cuda.set_stream(self.capture_stream)
            self.graph.capture_end()
        except Exception:
            # BUG 100: an invalidated capture can leave the stream in capture mode
            # (capture_end throws before cudaStreamEndCapture runs), so every later
            # eager kernel would be recorded instead of executed and the host would
            # hang. Force the stream out of capture mode and reset the cross-device
            # counter protocol so the eager fallback is always usable.
            try:
                self.graph.capture_end()
            except Exception:
                pass
            ext_c.tp_force_end_capture(self.capture_stream.cuda_stream)
            for _c in (ctx.ext_tp_contexts.values() if ctx.ext_tp_contexts
                       else [ctx.ext_tp_context]):
                if _c:
                    ext_c.tp_reset_counters(_c)
            self.graph = None
            raise
        finally:
            # BUG 110: stop routing dev1-3 allocations to the pool (capture_begin/end handles
            # dev0 internally). Always run, even if capture failed, so the allocators recover.
            for _d in _pooled_devs:
                try:
                    torch._C._cuda_endAllocateToPool(_d, _pool)
                except Exception:
                    pass
            for _c in (ctx.ext_tp_contexts.values() if ctx.ext_tp_contexts
                       else [ctx.ext_tp_context]):
                if _c:
                    ext_c.tp_set_capture_stream(_c, 0)
            torch.cuda.set_stream(prev)

        # The capture region ran the step once more, so restore the position again.
        if _saved_seq_len is not None:
            self.cache.current_seq_len = _saved_seq_len

        torch.cuda.synchronize(dev0)
        print(f"[TP graph] captured key={self.key} dev={dev0}", flush=True)

    def release(self):
        """
        Drop the captured graph and the KV-cache reference.

        BUG 47: a runner holds a hard reference to the cache it was captured against, and
        the graph's nodes point into that cache's buffers. When the caller switches to a new
        cache the runner must be released, or replay dereferences freed memory (illegal
        memory access on 2 GPUs; segfault in cuGraphLaunch when the old cache is collected).
        """
        self.graph = None
        self.cache = None
        self.step_fn = None
        self.num_replays = 0
        self.input_buf = None
        self.logits = None
        self.state_buf = None

    def replay(self):
        if self.num_replays == 0:
            print(f"[TP graph] first replay key={self.key}", flush=True)
        self.graph.replay()
        self.num_replays += 1


class TPGraphManager:
    """
    Owns the per-shape TP decode graphs for one model.

    Kept deliberately small: one graph per static decode shape, created lazily on the first
    decode step of that shape, and dropped if the shape changes.
    """

    def __init__(self, model):
        self.model = model
        self.runners: dict[tuple, TPGraphRunner] = {}
        self.enabled = graphs_enabled()
        self.disabled_reason: str | None = None
        # Set once a capture has failed, so we do not retry (and re-fail) every token.
        self.failed = False

    def key_for(self, batch_size: int, q_len: int, state: int = 0,
                last_id_only: bool = True) -> tuple:
        # The state mode separates the plain decode graph (0) from the MTP bonus-commit
        # graph (1: same (1, 1) shape, but also copies the pre-final-norm hidden state
        # into a static buffer) and the bonus-eliminated verify graph (2: copies the
        # pre-final-norm state of EVERY row, so the driver can seed the next round's
        # drafter from the last accepted draft's state without a bonus forward).
        # last_id_only is part of the key because it changes whether step_fn narrows
        # the residual before the head.
        return (batch_size, q_len, int(state), bool(last_id_only))

    def can_capture(self, batch_size: int, q_len: int, loras,
                    allow_multi: bool = False) -> bool:
        """Capture is valid for the single-token decode step, and for the fixed-shape
        MTP verify window when ``allow_multi`` is set (the caller marks that forward
        explicitly; arbitrary prefill chunks are never captured). No LoRA."""
        if not self.enabled or self.failed:
            return False
        if q_len != 1 and not allow_multi:
            return False
        if loras:
            return False
        # BUG 45: the fused TP attention kernel (``tp_attn_forward_``, used by models WITHOUT
        # per-layer attention geometry -- Llama2 and friends) is not capturable: it invalidates the
        # capture on the very first layer, so every attempt aborted with
        # cudaErrorStreamCaptureInvalidated and fell back to eager anyway. Failing up front is
        # cheaper and quieter than a failed capture per process.
        #
        # Gemma-4 sets ``tp_per_layer_geometry`` (it mixes head_dim 256/16-kv-head and
        # head_dim 512/4-kv-head layers), which routes attention through ``forward_tp_old`` -- the
        # pure-torch path this graph was built and verified against. That is the supported case.
        cfg = self.model.config
        if not bool(getattr(cfg.arch.lm, "tp_per_layer_geometry", False)):
            self.disabled_reason = (
                "fused TP attention path (no tp_per_layer_geometry) is not graph-capturable"
            )
            self.failed = True
            return False
        return True

    def get(self, batch_size: int, q_len: int, cache = None, state: int = 0,
            last_id_only: bool = True) -> TPGraphRunner | None:
        """
        Return the runner for this shape, but ONLY if it was captured against ``cache``.

        BUG 47: the key used to be ``(batch_size, q_len)`` alone, so a runner survived a
        change of KV cache. A captured graph has the cache's buffer addresses baked into its
        nodes, so replaying it against a *different* cache writes KV into the old (possibly
        freed) tensors and reads attention state from them. That is what produced

          * ``CUDA error: an illegal memory access`` in ``tp_context.wait_streams()`` on
            2 GPUs, and
          * a hard segfault inside ``cuGraphLaunch`` when a second sequence reused the
            process (the old cache had been freed, so the graph pointed at unmapped memory).

        It also silently corrupted output: the second sequence attended to the first
        sequence's KV rows, which is why short prompts produced repetition loops while the
        first sequence in a process was always fine.

        The cache identity is therefore part of the validity check, and a runner captured
        against a different cache is dropped (and its graph released) so it gets recaptured.
        """
        runner = self.runners.get(self.key_for(batch_size, q_len, state, last_id_only))
        if runner is None:
            return None
        if cache is not None and runner.cache is not cache:
            # Stale: captured against a cache that is no longer the one in use.
            self.drop(batch_size, q_len, state, last_id_only)
            return None
        return runner

    def add(self, batch_size: int, q_len: int, runner: TPGraphRunner,
            state: int = 0, last_id_only: bool = True):
        self.runners[self.key_for(batch_size, q_len, state, last_id_only)] = runner

    def drop(self, batch_size: int, q_len: int, state: int = 0,
             last_id_only: bool = True):
        """Discard a runner and release its captured graph and memory pool."""
        runner = self.runners.pop(self.key_for(batch_size, q_len, state, last_id_only), None)
        if runner is not None:
            runner.release()

    def clear(self):
        """BUG 117: release every runner (graphs stay ENABLED) on a context-regime change,
        so all shapes recapture at the new kv length instead of replaying short-capture
        graphs far outside their valid envelope."""
        for k in list(self.runners.keys()):
            self.runners.pop(k).release()
        self.runners.clear()

    def disable(self, reason: str):
        self.failed = True
        self.disabled_reason = reason
        for k in list(self.runners.keys()):
            self.runners.pop(k).release()
        self.runners.clear()
