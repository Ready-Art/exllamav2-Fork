"""
exllamav2/tp_mp.py

Multiprocess tensor-parallel backend: one Python process per GPU.

WHY THIS EXISTS
---------------
EXL2 TP runs all four GPUs from a single Python thread. Every module forward is a Python
call that issues work to device i, then to device i+1, ... serially, and every TP
collective is a C++ call made from that same thread. The result is that the four GPUs can
never be busy at the same time: the host is the critical path.

Measured on 4x RTX 3090, Gemma-4-31B EXL2 6.0bpw, decode:

    wall                13.31 ms/token
    GPU utilisation     27% / 10% / 26% / 14%   (nvidia-smi, during the decode loop)
    CPU                 93% of ONE core, 96% of the machine idle
    aten self time      0.39 ms/token  (so the core is spinning in CUDA sync, not working)

exllamav3 gets roughly double the decode throughput on the same hardware because it does
not do this: `Model.create_tp_context()` spawns one worker process per device, each with
its own CUDA context, its own module shards and its own dispatch thread, and the ranks
exchange tensors through NCCL. Four dispatch threads instead of one.

This module brings the same structure to exllamav2:

  * one spawned worker process per device (spawn, not fork: CUDA cannot be forked safely)
  * each worker loads only its own shard of the model
  * a `torch.distributed` NCCL process group per worker for collectives
  * the parent keeps device 0 in-process (the "master") so the logits are already local
  * commands and small metadata go over a Pipe; tensors move through NCCL

The parent drives the workers with a fan-out dispatch: every worker gets the command
*before* the parent runs its own slice, so the master's blocking work overlaps the
workers' work instead of following it.
"""

import os
import sys
import signal
import threading
import traceback
import uuid
from multiprocessing import Process, Pipe

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Worker-side context
# ---------------------------------------------------------------------------

def _init_pg(device: int, world_size: int, rank: int, init_method: str, uuid_hex: str):
    """Initialise this worker's NCCL process group."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = init_method.rsplit(":", 1)[-1]
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend = "nccl",
        init_method = init_method,
        world_size = world_size,
        rank = rank,
        timeout = __import__("datetime").timedelta(seconds = 120),
    )
    return {"device": device, "rank": rank, "world_size": world_size}


def _parent_watchdog(parent_pid):
    """BUG 92: exit this worker if its parent dies.

    A daemon spawn worker is NOT killed when its parent is SIGKILLed -- it is
    reparented to init, stays parked in conn.recv() below, and keeps its GPU
    shard pinned (the 9-12 GiB of 'leftover' VRAM seen after force-killed
    tests). Poll the parent PID; if it changes, the parent is gone, so exit
    immediately and release the VRAM.
    """
    import time
    while True:
        time.sleep(2)
        try:
            if os.getppid() != parent_pid:
                os._exit(0)
        except Exception:
            os._exit(0)


def mp_worker(conn, device, world_size, rank, init_method, uuid_hex, model_dir, gpu_split,
              max_seq_len, cache_tokens, cache_base_name):
    """
    Worker process entry point. Owns one GPU for the lifetime of the model.

    Protocol: the parent sends (cmd, args) tuples and expects one reply per command.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threading.Thread(target = _parent_watchdog, args = (os.getppid(),), daemon = True).start()

    try:
        torch.cuda.set_device(device)
        ctx = _init_pg(device, world_size, rank, init_method, uuid_hex)

        from exllamav2 import ExLlamaV2, ExLlamaV2Config
        from exllamav2.cache import ExLlamaV2Cache_SWA

        cfg = ExLlamaV2Config()
        cfg.model_dir = model_dir
        cfg.prepare()
        cfg.arch_compat_overrides()
        cfg.max_seq_len = max_seq_len

        model = ExLlamaV2(cfg)
        # Load ONLY this rank's shard. load_tp() would materialise every module in full on
        # this process, which defeats the entire purpose (the model does not fit on one card).
        # load_tp_rank() reads just this rank's byte ranges out of the checkpoint and leaves
        # every sharded linear in single-device state. Verified bit-identical to
        # load_tp()+tp_split() in order100.md.
        model.load_tp_rank(rank, gpu_split,
                           expect_cache_tokens = cache_tokens,
                           expect_cache_base = ExLlamaV2Cache_SWA,
                           load_head = (rank == 0))

        # Per-rank KV cache: this rank's slice of the KV split, under each layer's own
        # geometry. ExLlamaV2Cache_TP would build all four devices' caches.
        from exllamav2.cache import tp_rank_cache
        cache = tp_rank_cache(model, rank, base = ExLlamaV2Cache_SWA,
                              max_seq_len = max_seq_len)

        ctx["model"] = model
        ctx["cfg"] = cfg
        ctx["cache"] = cache
        conn.send(("ready", None))
    except Exception as e:
        conn.send(("error", "".join(traceback.format_exception(e))))
        return

    # Command loop
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg == "quit":
            # Do NOT synchronize / destroy_process_group here: with peers exiting at
            # different times the NCCL teardown blocks indefinitely (the shutdown
            # freeze that left VRAM-holding processes behind). Process exit releases
            # the CUDA context and VRAM; the parent escalates to SIGTERM/SIGKILL if
            # exit stalls, and the watchdog below bounds it no matter what.
            def _quit_watchdog():
                import time
                time.sleep(5)
                os._exit(0)
            threading.Thread(target = _quit_watchdog, daemon = True).start()
            break

        cmd, args = msg
        try:
            if cmd == "forward":
                # args: (input_ids, past_len, seq_len, batch_size, last_id_only,
                #        graph_verify[, return_row_states])
                result = _worker_forward(ctx, *args)
                conn.send(("ok", result))
            elif cmd == "forward-state":
                # MTP: args: (input_ids, last_id_only, cpu_logits). Fan-out collective:
                # every rank runs its slice; only rank 0 returns (logits, last_state).
                result = _worker_forward_state(ctx, *args)
                conn.send(("ok", result))
            elif cmd == "set-seqlen":
                # MTP: distributed rewind. All ranks must move current_seq_len together
                # (the SWA ring shift is a deterministic function of it -- BUG 94).
                ctx["cache"].current_seq_len = args[0]
                if args[0] == 0:
                    ctx["cache"].window_beg = 0   # full reset (new sequence)
                conn.send(("ok", None))
            elif cmd == "reset-cache":
                # MTP validation: zero current_seq_len + window_beg on this rank.
                ctx["cache"].current_seq_len = 0
                ctx["cache"].window_beg = 0
                conn.send(("ok", None))
            elif cmd == "kv-slice":
                # MTP: return this rank's K/V rows [start:end] of one layer (the parent
                # joins the four ranks' slices along the kv-head axis).
                conn.send(("ok", _worker_kv_slice(ctx, *args)))
            elif cmd == "kv-pack":
                # MTP gather batching: both drafter source layers' K/V deltas packed
                # into one pinned CPU tensor (see _worker_kv_pack).
                conn.send(("ok", _worker_kv_pack(ctx, *args)))
            elif cmd == "ring-snapshot":
                # Prompt-boundary prompt cache: save this rank's SWA ring + bookkeeping.
                ctx["ring_backup"] = _clone_swa_ring(ctx["cache"])
                ctx["ring_backup_wb"] = ctx["cache"].window_beg
                conn.send(("ok", None))
            elif cmd == "ring-restore":
                # Restore the saved ring and rewind to the snapshot position. Exact:
                # the ring content IS the state the original prefill left at that
                # position -- no recompute (SWA K/V cannot be recomputed from a
                # truncated start: attention windows would regress to the sequence
                # start). Full-layer K/V above the position is ignored via seq_len.
                _restore_swa_ring(ctx["cache"], ctx["ring_backup"], args,
                                  ctx["ring_backup_wb"])
                conn.send(("ok", None))
            elif cmd == "sync":
                torch.cuda.synchronize()
                conn.send(("ok", None))
            elif cmd == "barrier":
                dist.barrier()
                conn.send(("ok", None))
            elif cmd == "exec":
                fn, a = args
                conn.send(("ok", fn(ctx, *a)))
            else:
                conn.send(("error", "unknown command " + str(cmd)))
        except Exception as e:
            # BUG 74: also write to stderr. The parent is usually blocked in a collective
            # waiting for this rank, so it never reads the pipe reply -- without this the
            # traceback is lost and the run just hangs.
            tb = "".join(traceback.format_exception(e))
            sys.stderr.write("[rank%d] worker error:\n%s\n" % (rank, tb))
            sys.stderr.flush()
            conn.send(("error", tb))


def _worker_forward_state(ctx, input_ids, last_id_only, cpu_logits):
    """
    MTP: one forward that ALSO returns the pre-final-norm hidden state.

    Only rank 0 (the master, which owns lm_head) returns a value; the other ranks
    run the same collective and return None.

    Change98: single-token state requests (the MTP bonus commit) now RUN through the
    CUDA-graph path -- the bonus graph copies the pre-final-norm state into a static
    buffer inside the capture. The returned state/logits are that runner's static
    buffers: callers must consume (copy / serialize) them before the next replay.
    Multi-token state requests (prefill) still take the eager path.
    """
    model = ctx["model"]
    cache = ctx["cache"]
    dev = ctx["device"]
    if input_ids.device.index != dev:
        input_ids = input_ids.to(dev)
    r = model.forward(input_ids, cache = cache, last_id_only = last_id_only,
                      return_last_state = True, cpu_logits = cpu_logits)
    if r is None:
        return None
    if isinstance(r, tuple):
        logits, state = r
    else:
        logits, state = r, None
    if ctx["rank"] != 0:
        return None
    return (logits, state)


def _worker_kv_slice(ctx, layer_idx, start, end):
    """MTP: this rank's K/V rows [start:end] of one target layer, on CPU.

    Returns 3-D ``(S, kvh, hd)`` tensors -- the same layout the single-process
    drafter path builds (``key_states[layer][0, :len]``) -- so the parent can join
    the four ranks along the kv-head axis (dim 1) and feed FlashInfer directly.
    """
    cache = ctx["cache"]
    width = cache.key_states[layer_idx].shape[1]
    start = max(0, min(start, width))
    end = max(start, min(end, width))
    k = cache.key_states[layer_idx][0, start:end].detach().cpu()
    v = cache.value_states[layer_idx][0, start:end].detach().cpu()
    return (k, v)


def _clone_swa_ring(cache):
    """Pinned-CPU copy of every SWA layer's ring (the ring-depth buffers).

    Was a device-side clone: at ring = window + chunk (2304+ slots) that is ~0.5 GB
    of GPU per rank held for the whole job, which OOMs the master alongside the hb16
    head + drafter + 160k KV (debug2.txt). Host pinned memory is free here (125 GB
    RAM); the snapshot pays one ~30 ms D2H at job end and the restore one H2D, both
    negligible against seconds of prefill.

    inference_mode: the cache buffers are inference tensors (allocated under
    inference_mode at load); these helpers run OUTSIDE model.forward's
    @torch.inference_mode() scope (prompt-cache decision block), where inplace
    ops on inference tensors raise. Nesting the mode is reentrant-safe."""
    ring = cache.kv_state_size
    with torch.inference_mode():
        out = {}
        for l in range(len(cache.key_states)):
            if cache.key_states[l].shape[1] != ring:
                continue
            kc, vc = cache.key_states[l], cache.value_states[l]
            k = torch.empty(kc.shape, dtype = kc.dtype, pin_memory = True)
            v = torch.empty(vc.shape, dtype = vc.dtype, pin_memory = True)
            k.copy_(kc, non_blocking = True)
            v.copy_(vc, non_blocking = True)
            out[l] = (k, v)
        # The caller may overwrite the rings (next prefill) as soon as this returns;
        # every async D2H above must have landed first. One sync for the whole batch.
        torch.cuda.synchronize()
        return out


def _restore_swa_ring(cache, backup, seq, wb):
    with torch.inference_mode():
        for l, (k, v) in backup.items():
            cache.key_states[l].copy_(k, non_blocking = True)
            cache.value_states[l].copy_(v, non_blocking = True)
        torch.cuda.synchronize()
    cache.current_seq_len = seq
    cache.window_beg = wb


def _worker_kv_pack(ctx, lf, fl0, fl1, ls, sl0, sl1):
    """MTP gather batching: this rank's K/V deltas for BOTH drafter source layers in
    ONE pinned CPU tensor (one pipe message instead of two commands + two replies).

    Flat layout: [K_full | V_full | K_swa | V_swa], each (rows, kvh_local, hd). The
    parent knows every shape from the request args and unpacks zero-copy views.
    Grow-only pinned staging (the first, full-length gather is the largest; steady
    deltas are a handful of rows). The four D2H copies are async on the worker's
    stream with a single sync, instead of four synchronous .cpu() round trips.
    """
    cache = ctx["cache"]

    def clamp(layer, a, b):
        width = cache.key_states[layer].shape[1]
        a = max(0, min(a, width))
        b = max(a, min(b, width))
        return a, b

    fl0, fl1 = clamp(lf, fl0, fl1)
    sl0, sl1 = clamp(ls, sl0, sl1)
    kf = cache.key_states[lf][0, fl0:fl1].detach()
    vf = cache.value_states[lf][0, fl0:fl1].detach()
    ks = cache.key_states[ls][0, sl0:sl1].detach()
    vs = cache.value_states[ls][0, sl0:sl1].detach()
    n = kf.numel() + vf.numel() + ks.numel() + vs.numel()
    stag = ctx.get("kv_pack_staging")
    if stag is None or stag.numel() < n:
        stag = ctx["kv_pack_staging"] = torch.empty(n, dtype = torch.half, pin_memory = True)
    o = 0
    for t in (kf, vf, ks, vs):
        stag[o:o + t.numel()].view(t.shape).copy_(t, non_blocking = True)
        o += t.numel()
    torch.cuda.current_stream().synchronize()
    # Safe to return a view of the reusable staging: the pipe pickler copies CPU tensor
    # data into shared memory at send time (same mechanism the old kv-slice path used).
    return stag[:n]


def _worker_forward(ctx, input_ids, past_len, seq_len, batch_size, last_id_only,
                    graph_verify = False, return_row_states = False):
    """
    Run this rank's share of one forward pass.

    Every rank runs the whole module stack; each one only computes its own column-parallel
    shard and the collectives reassemble the full width. Only rank 0 has lm_head, so only
    rank 0 returns logits.

    ``graph_verify`` marks the fixed-shape MTP verify window so forward_chunk may capture /
    replay it as a CUDA graph (Change98). Every rank must agree on it: the capture is a
    collective step, and a rank taking the eager path while the others replay would deadlock.

    ``return_row_states`` (bonus-forward elimination): also return the pre-final-norm hidden
    state of EVERY processed row, so the MTP driver can seed the next round's drafter from
    the last accepted draft's verify state instead of paying a separate bonus forward.
    Rank 0 returns ``(logits, row_states)``; the other ranks return None. Every rank must
    agree on this flag too -- it changes the captured graph's state mode.
    """
    model = ctx["model"]
    cache = ctx["cache"]

    # BUG 63: the embedding table is device-resident in a worker (tp_load_rank moves it off
    # the host), so the token ids have to be on that device too.
    dev = ctx["device"]
    if input_ids.device.index != dev:
        input_ids = input_ids.to(dev)

    r = model.forward(
        input_ids,
        cache = cache,
        last_id_only = last_id_only,
        graph_verify = graph_verify,
        return_row_states = return_row_states,
    )
    if r is None:
        return None
    if return_row_states:
        if ctx["rank"] != 0:
            return None
        logits, states = r
        return (logits, states)
    if isinstance(r, tuple):
        r = r[0]
    if isinstance(r, dict):
        return r.get("logits", None)
    return r


# ---------------------------------------------------------------------------
# Parent-side handle
# ---------------------------------------------------------------------------

class TPWorkerHandle:
    """Parent-side proxy for one worker process."""

    def __init__(self, device, conn, proc):
        self.device = device
        self.conn = conn
        self.proc = proc

    def send(self, cmd, args = None):
        self.conn.send((cmd, args))

    def recv(self):
        status, payload = self.conn.recv()
        if status == "error":
            raise RuntimeError("TP worker %d failed:\n%s" % (self.device, payload))
        return payload

    def call(self, cmd, args = None):
        self.send(cmd, args)
        return self.recv()

    def close(self):
        try:
            self.conn.send("quit")
        except Exception:
            pass
        self.join_and_escalate()

    def join_and_escalate(self):
        self.proc.join(timeout = 5)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout = 5)
        if self.proc.is_alive():
            # BUG 92: a worker parked inside an NCCL collective may not exit on
            # SIGTERM either; escalate so close() never leaves a VRAM-holding
            # process behind.
            os.kill(self.proc.pid, signal.SIGKILL)
            self.proc.join(timeout = 5)
        try:
            self.conn.close()
        except Exception:
            pass


from exllamav2.mtp_mp import MTPMPDriver   # MTP driver for the MP backend

class TPProcessGroup:
    """
    Spawns and owns one worker process per GPU.

    `master_device` stays in the parent process so the logits are already local and the
    sampling step needs no extra hop.
    """

    def __init__(self, model_dir, gpu_split, max_seq_len, cache_tokens, cache_base_name,
                 master_device = 0, verbose = False):
        import multiprocessing
        multiprocessing.set_start_method("spawn", force = True)
        torch.multiprocessing.set_sharing_strategy("file_system")

        self.model_dir = model_dir
        self.gpu_split = gpu_split
        self.max_seq_len = max_seq_len
        self.cache_tokens = cache_tokens
        self.cache_base_name = cache_base_name
        self.master_device = master_device
        self.verbose = verbose
        self.workers: dict[int, TPWorkerHandle] = {}
        self.master_ctx = None
        self.model = None
        self.cache = None
        # MTP: drafter (single-device, parent process) + its draft window. See
        # exllamav2/mtp_mp.py (MTPMPDriver) for the round protocol.
        self.mtp_drafter = None
        self.num_speculative_tokens = 6
        self._mtp_kv = None
        self._init_method = None
        self._world_size = len(gpu_split)
        # Client-disconnect abort: set by the API layer (job.cancel), polled by
        # model.forward at every prefill chunk boundary on the master rank and
        # broadcast to the workers with a MAX all_reduce (same-boundary abort).
        self.abort_flag = multiprocessing.Event()
        # Prompt-boundary ring snapshot (prompt cache restore path): every rank's
        # SWA ring at the end of the lineage's first prompt, ~ring x kv heads x hd
        # x 2 x 2B x n_swa_layers per rank (~262 MB for Gemma-4 on 4 GPUs).
        self._ring_backup_local = None
        self._ring_backup_seq = -1
        self._ring_backup_wb = 0

    def start(self):
        port = _find_free_port()
        self._init_method = "tcp://127.0.0.1:%d" % port
        uuid_hex = uuid.uuid4().hex

        for rank, device in enumerate(range(len(self.gpu_split))):
            if device == self.master_device:
                continue
            parent_conn, child_conn = Pipe()
            proc = Process(
                target = mp_worker,
                args = (child_conn, device, self._world_size, rank, self._init_method,
                        uuid_hex, self.model_dir, self.gpu_split, self.max_seq_len,
                        self.cache_tokens, self.cache_base_name),
                daemon = True,
            )
            proc.start()
            self.workers[device] = TPWorkerHandle(device, parent_conn, proc)

        # The master also joins the process group, so collectives work.
        self.master_ctx = _init_pg(self.master_device, self._world_size,
                                  self.master_device, self._init_method, uuid_hex)

        # The master loads its own shard in-process, exactly as the workers do. It keeps
        # lm_head, so the logits are already local and sampling needs no extra hop.
        from exllamav2 import ExLlamaV2, ExLlamaV2Config
        from exllamav2.cache import ExLlamaV2Cache_SWA, tp_rank_cache

        cfg = ExLlamaV2Config()
        cfg.model_dir = self.model_dir
        cfg.prepare()
        cfg.arch_compat_overrides()
        cfg.max_seq_len = self.max_seq_len

        model = ExLlamaV2(cfg)
        model.load_tp_rank(self.master_device, self.gpu_split,
                           expect_cache_tokens = self.cache_tokens,
                           expect_cache_base = ExLlamaV2Cache_SWA,
                           load_head = True)
        cache = tp_rank_cache(model, self.master_device, base = ExLlamaV2Cache_SWA,
                              max_seq_len = self.max_seq_len)

        self.model = model
        self.cache = cache
        # _worker_forward() reads the model and cache out of the ctx dict, so the master's
        # copies have to live there too, not just on this object.
        self.master_ctx["model"] = model
        self.master_ctx["cfg"] = cfg
        self.master_ctx["cache"] = cache
        # Abortable prefill: only the master polls the API-layer event; the workers
        # contribute 0 to the per-chunk MAX all_reduce (see model.forward).
        model.tp_context.abort_flag = self.abort_flag

        # Wait for every worker to finish loading.
        for device, w in sorted(self.workers.items()):
            w.recv()

    # ------------------------------------------------------------------ MTP
    @property
    def config(self):
        return self.model.config

    @property
    def modules(self):
        return self.model.modules

    @property
    def head_layer_idx(self):
        return self.model.head_layer_idx

    def get_device_context(self, device_idx, scratch = True):
        return self.model.get_device_context(device_idx, scratch)

    def enable_mtp(self, drafter, num_draft_tokens = 6):
        """Attach the MTP drafter so :meth:`forward_state` / :meth:`gather_kv_mtp`
        serve the draft-verify loop. The drafter is a parent-process, single-device
        model; it reads the target's sharded K/V via :meth:`gather_kv_mtp`."""
        self.mtp_drafter = drafter
        self.num_speculative_tokens = int(num_draft_tokens)
        # Allocate the joined KV shadow NOW, at load time with a clean allocator --
        # not lazily on the first gather after a long prefill (debug10.txt OOM).
        self._mtp_kv = None
        self._alloc_mtp_kv()
        if os.environ.get("EXL2_VRAM_TRACE", "1") != "0":
            self._vram_breakdown()

        # Cold-start fix (results.md, 2026-11-20): FlashInfer JIT-compiles each kernel
        # geometry on FIRST use. The first server run paid ~33 s in prefill (1315 tokens
        # at 40 T/s) plus ~19.4 s inside the first draft phase (the running draft average
        # fits C/N + 11.9 ms with C ≈ 19.4 s at every checkpoint) -- about a minute to
        # first text. The second run started fast ONLY because the JIT disk cache was
        # warm, not because of EXL2_TP_SEQPAR. Compile at LOAD time instead: a 2-token
        # dummy prefill (target FI prefill kernels) + one draft step (drafter hd512/hd256
        # single_decode kernels), then reset the cache. EXL2_MTP_WARMUP=0 skips; failure
        # is non-fatal (the first request would just pay the JIT as before).
        if os.environ.get("EXL2_MTP_WARMUP", "1") != "0":
            try:
                import time as _time
                _t0 = _time.perf_counter()
                print("[mtp] warmup: pre-compiling FlashInfer kernels "
                      "(slow only on the very first load)...", flush=True)
                dummy = torch.tensor([[0, 0]], dtype = torch.long)
                _logits, _state = self.forward_state(dummy, last_id_only = True,
                                                     cpu_logits = True)
                drafter.gather_target(self)
                # last_state is (1, q_len, H) from the eager multi-token prefill -- the
                # drafter consumes only the LAST row (same slice mtp_mp.begin() applies).
                drafter.draft_window(torch.tensor([[0]], dtype = torch.long),
                                     _state[:, -1:, :].float().cpu(), 2, 1)
                self.mtp_reset_cache()
                print(f"[mtp] warmup done in {_time.perf_counter() - _t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[mtp] warmup skipped ({type(e).__name__}: {e}); first request "
                      f"will pay FlashInfer JIT", flush=True)
                try:
                    self.mtp_reset_cache()
                except Exception:
                    pass

    def forward_state(self, input_ids, last_id_only = True, cpu_logits = True):
        """Distributed forward that returns ``(logits, last_state)``.

        ``last_state`` is the pre-final-norm hidden state of the last processed
        token (what the MTP drafter consumes as its round base). Every rank runs
        the collective; only the master produces the value.
        """
        for device, w in sorted(self.workers.items()):
            w.send("forward-state", (input_ids, last_id_only, cpu_logits))
        out = _worker_forward_state(self.master_ctx, input_ids, last_id_only, cpu_logits)
        for device, w in sorted(self.workers.items()):
            w.recv()
        return out

    def mtp_set_seq_len(self, n):
        """Distributed rewind: moves the master cache AND every worker cache to ``n``.
        The SWA ring shift is a deterministic function of ``current_seq_len`` (BUG 94),
        so moving all ranks in lockstep keeps every ring in sync with no extra traffic."""
        for device, w in sorted(self.workers.items()):
            w.send("set-seqlen", (n,))
            w.recv()
        self.cache.current_seq_len = n

    def snapshot_prompt_ring(self):
        """Save every rank's SWA ring + window_beg at the current position (the end
        of a fresh prompt prefill). Enables exact restore to this prompt boundary
        when a later request re-renders the generated history differently (e.g. a
        client stripping thought channels): the full-layer K/V prefix stays live,
        the ring is copied back, and only the re-rendered tail is prefilled."""
        seq = self.cache.current_seq_len
        for device, w in sorted(self.workers.items()):
            w.send("ring-snapshot", None)
        self._ring_backup_local = _clone_swa_ring(self.cache)
        self._ring_backup_wb = self.cache.window_beg
        for device, w in sorted(self.workers.items()):
            w.recv()
        self._ring_backup_seq = seq

    def restore_prompt_ring(self, seq):
        """Rewind every rank to the snapshot position and copy the saved rings back.
        Returns False if no snapshot exists for exactly this position."""
        if self._ring_backup_local is None or self._ring_backup_seq != seq:
            return False
        for device, w in sorted(self.workers.items()):
            w.send("ring-restore", seq)
        _restore_swa_ring(self.cache, self._ring_backup_local, seq, self._ring_backup_wb)
        for device, w in sorted(self.workers.items()):
            w.recv()
        # the drafter's incremental gather shadow is stale after a rewind -- reset the
        # bookkeeping and re-zero, but KEEP the buffers (post-prefill realloc OOMs)
        self._reset_mtp_kv()
        return True

    def mtp_reset_cache(self):
        """Distributed full cache reset (current_seq_len = window_beg = 0 on every rank).
        Used by the same-process validation to re-prefill cleanly between runs."""
        for device, w in sorted(self.workers.items()):
            w.send("reset-cache", None)
            w.recv()
        self.cache.current_seq_len = 0
        self.cache.window_beg = 0
        # reset the MTP gather shadow so the next gather starts fresh -- keep the
        # buffers allocated (freeing them forces a post-prefill realloc; debug10.txt)
        self._reset_mtp_kv()
        # a stale disconnect abort from a previous job must not kill this one
        self.abort_flag.clear()

    def _vram_breakdown(self):
        """One-time load-time VRAM accounting (EXL2_VRAM_TRACE=0 disables). Prints the
        master process's per-device torch footprint and the big opt-in component sizes
        (o_rs fp16 shards, MTP KV shadow, drafter weights, KV cache) so an OOM can be
        attributed to a specific component instead of guessed at."""
        try:
            from exllamav2.attn import ExLlamaV2Attention
            m = self.model
            mb = lambda n: n / 1024 ** 2
            o_rs = sum(a._o_fp16_row.numel() * a._o_fp16_row.element_size()
                       for a in m.modules if isinstance(a, ExLlamaV2Attention)
                       and getattr(a, "_o_fp16_row", None) is not None)
            shadow = sum(t.numel() * t.element_size() for t in self._mtp_kv.values()
                         if torch.is_tensor(t))
            draft = 0
            d = self.mtp_drafter
            for obj in [d] + list(getattr(d, "layers", [])):
                for v in vars(obj).values():
                    if torch.is_tensor(v):
                        draft += v.numel() * v.element_size()
            kv = 0
            for ks in (self.cache.key_states, self.cache.value_states):
                for t in ks:
                    if torch.is_tensor(t):
                        kv += t.numel() * t.element_size()
                    elif isinstance(t, (list, tuple)):
                        kv += sum(x.numel() * x.element_size() for x in t if torch.is_tensor(x))
            ndev = torch.cuda.device_count()
            per = " ".join(f"dev{i}={mb(torch.cuda.memory_allocated(i)):.0f}" for i in range(ndev))
            print(f"[vram] master torch alloc MiB: {per} (reserved dev0="
                  f"{mb(torch.cuda.memory_reserved(0)):.0f})", flush=True)
            print(f"[vram] components MiB: o_rs_fp16={mb(o_rs):.0f} mtp_shadow={mb(shadow):.0f} "
                  f"drafter={mb(draft):.0f} kv_cache={mb(kv):.0f}", flush=True)
        except Exception as e:
            print(f"[vram] breakdown skipped ({type(e).__name__}: {e})", flush=True)

    def _alloc_mtp_kv(self):
        """Allocate the drafter's joined KV shadow (fixed-size, zero-initialized).

        Called at LOAD time (enable_mtp): the 2x626 MiB full-KV buffers must come from
        a clean allocator. The old lazy path allocated them on the first gather AFTER
        a long prefill, when the caching allocator was full and fragmented -- the V
        buffer's 626 MiB request OOMed the master with 387 MiB free (debug10.txt).
        Resets REUSE these buffers (_reset_mtp_kv) instead of freeing/reallocating.

        Change98: zero-init (not empty). The draft graph's attention matmul reads the
        FULL buffer depth; unwritten tail rows must be finite (0) or NaN/inf garbage
        poisons the softmax. Zero rows score 0 and get masked out.
        """
        d = self.mtp_drafter
        mc = self.cache
        dev = d.dev
        # inference_mode(FALSE): enable_mtp runs under the loader's InferenceMode, but
        # gather_kv_mtp mutates these buffers IN PLACE on the job thread OUTSIDE it
        # (ring-shift mirror copy_, delta H2D copy_). Inference tensors forbid that
        # (debug11.txt); the old lazy allocation created them outside InferenceMode as
        # normal tensors -- preserve exactly that semantics here at load time.
        with torch.inference_mode(False):
            self._mtp_kv = {
                "fl": 0,
                "sl": 0,      # absolute seq_len at the last SWA gather
                "wb": 0,      # window_beg at the last SWA gather
                # 3-D (S, kvh, hd) -- identical layout to the single-process drafter.
                "kf": torch.zeros((self.max_seq_len, d.num_global_kv_heads, d.global_head_dim),
                                  dtype = torch.half, device = dev),
                "vf": torch.zeros((self.max_seq_len, d.num_global_kv_heads, d.global_head_dim),
                                  dtype = torch.half, device = dev),
                "ks": torch.zeros((mc.kv_state_size, d.num_kv_heads, d.head_dim),
                                  dtype = torch.half, device = dev),
                "vs": torch.zeros((mc.kv_state_size, d.num_kv_heads, d.head_dim),
                                  dtype = torch.half, device = dev),
            }
        return self._mtp_kv

    def _reset_mtp_kv(self):
        """Reset the gather bookkeeping and re-zero the shadow, KEEPING the buffers.
        Freeing them here (the old `_mtp_kv = None`) forced a post-prefill reallocation
        of 1.25 GB at the worst possible allocator state (debug10.txt OOM). Re-zeroing
        ~1.3 GB costs ~1.5 ms per job -- nothing against seconds of prefill."""
        st = self._mtp_kv
        if st is None:
            return
        with torch.inference_mode():
            for _k in ("kf", "vf", "ks", "vs"):
                st[_k].zero_()
        st["fl"] = 0
        st["sl"] = 0
        st["wb"] = 0

    def gather_kv_mtp(self):
        """Give the drafter the target's last full (59) and sliding (58) K/V, joined
        along the kv-head axis, on the drafter's device.

        Returns ``(k_full, v_full, k_swa, v_swa, seq_len, window_beg)``.

        INCREMENTAL: the full-attention cache is append-only, so after the first
        (full-length) gather only the NEW rows since the previous gather are pulled
        from each rank (a handful of rows per round, not the whole 128k context).
        Change98: the SWA ring is incremental too. The ring shift is a physical
        left-copy of whole pages (cache.shift_ring), so the parent mirrors the same
        shift on its joined shadow buffers (a cheap device copy) and then pulls only
        the rows appended since the last gather -- a handful of rows per round
        instead of re-pulling the whole ~10 MB/rank valid prefix. A full rotation
        (shift == ring) leaves the ring data unchanged and forces a prefix re-pull;
        non-monotonic bookkeeping (not expected in the round protocol) also falls
        back to a full re-pull. All pipe requests are issued up front and all
        replies collected afterwards, so the workers' GPU->CPU copies overlap
        instead of serialising.
        """
        import torch
        assert self.mtp_drafter is not None, "call enable_mtp(drafter) first"
        d = self.mtp_drafter
        lf, ls = d.t_last_full, d.t_last_swa
        mc = self.cache
        dev = d.dev
        seq_len = mc.current_seq_len
        window_beg = getattr(mc, "window_beg", 0)
        swa_len = max(0, min(seq_len - window_beg, mc.kv_state_size))

        st = self._mtp_kv
        if st is None:
            st = self._alloc_mtp_kv()

        # ---- SWA ring: mirror the page shift on the shadow, pull only new rows ----
        ring = mc.kv_state_size
        swa_shift = window_beg - st["wb"]
        swa_start = max(st["sl"], window_beg) - window_beg   # ring slot of first new row
        if swa_shift < 0 or seq_len < st["sl"] or swa_start > swa_len:
            # Defensive full re-pull: bookkeeping moved backwards (not expected in the
            # round protocol -- the committed seq_len advances by m_acc + 1 >= 1 per
            # round, and window_beg is monotonic within a sequence).
            swa_shift = 0
            swa_start = 0
        if 0 < swa_shift < ring:
            # Mirror cache.shift_ring's physical left-copy on the joined shadow.
            st["ks"][:ring - swa_shift].copy_(st["ks"][swa_shift:].clone())
            st["vs"][:ring - swa_shift].copy_(st["vs"][swa_shift:].clone())
        # swa_shift == ring is a full rotation: the ring data is unchanged and the valid
        # prefix consists only of rows written after it (window_beg > st["sl"] forces
        # swa_start == 0), so the delta pull below re-covers the whole prefix.

        # ---- packed pull: ONE message per worker (was 2 commands + 2 replies each) ----
        # Each worker packs [K_full | V_full | K_swa | V_swa] deltas into one pinned CPU
        # tensor (async D2H + single sync); the parent unpacks zero-copy views, joins the
        # ranks along the kv-head axis on CPU (a few KB), and copies straight into the
        # device shadow -- 4 H2D total instead of 8 H2D + 4 device cats + 4 assigns.
        world = len(self.workers) + 1
        assert d.num_global_kv_heads % world == 0 and d.num_kv_heads % world == 0, \
            "kv-pack: kv heads must divide evenly across ranks"
        fkh = d.num_global_kv_heads // world
        skh = d.num_kv_heads // world
        df = seq_len - st["fl"]
        ds = swa_len - swa_start
        # issue every request first (workers copy in parallel), then collect
        for device, w in sorted(self.workers.items()):
            w.send("kv-pack", (lf, st["fl"], seq_len, ls, swa_start, swa_len))
        # master's own slices, on CPU like the workers' so the join is uniform
        full_k = [mc.key_states[lf][0, st["fl"]:seq_len].detach().cpu()]
        full_v = [mc.value_states[lf][0, st["fl"]:seq_len].detach().cpu()]
        swa_k = [mc.key_states[ls][0, swa_start:swa_len].detach().cpu()]
        swa_v = [mc.value_states[ls][0, swa_start:swa_len].detach().cpu()]
        for device, w in sorted(self.workers.items()):
            flat = w.recv()
            o = 0
            for rows, kh, hd, acc in ((df, fkh, d.global_head_dim, full_k),
                                      (df, fkh, d.global_head_dim, full_v),
                                      (ds, skh, d.head_dim, swa_k),
                                      (ds, skh, d.head_dim, swa_v)):
                n = rows * kh * hd
                acc.append(flat[o:o + n].view(rows, kh, hd))
                o += n
        if df > 0:
            st["kf"][st["fl"]:seq_len].copy_(torch.cat(full_k, dim = 1))
            st["vf"][st["fl"]:seq_len].copy_(torch.cat(full_v, dim = 1))
        # (after a rewind, st["fl"] < seq_len and the delta above re-writes from there)
        st["fl"] = seq_len

        if ds > 0:
            st["ks"][swa_start:swa_len].copy_(torch.cat(swa_k, dim = 1))
            st["vs"][swa_start:swa_len].copy_(torch.cat(swa_v, dim = 1))
        st["sl"] = seq_len
        st["wb"] = window_beg

        return (st["kf"][:seq_len], st["vf"][:seq_len],
                st["ks"][:swa_len], st["vs"][:swa_len],
                seq_len, window_beg)

    def forward(self, input_ids, past_len, seq_len, batch_size, last_id_only,
                graph_verify = False, return_row_states = False):
        """
        Fan out the step, then run the master's own share.

        Every worker is given the command BEFORE the master runs its slice, so the workers'
        work overlaps the master's instead of following it. That overlap is the point of the
        whole exercise: four dispatch threads instead of one.

        ``graph_verify=True`` marks the MTP verify window (fixed q_len) as graph-capturable
        on every rank (Change98).

        ``return_row_states=True`` (bonus-forward elimination) returns ``(logits, row_states)``
        from the master: the pre-final-norm hidden state of every processed row. Only the
        master produces a value (workers return None), so no CUDA tensor crosses a pipe.
        The states are the verify graph's static buffer -- consume them before the next
        replay (the driver copies them to CPU immediately).
        """
        for device, w in sorted(self.workers.items()):
            w.send("forward", (input_ids, past_len, seq_len, batch_size, last_id_only,
                               graph_verify, return_row_states))

        # Master's own share, overlapped with the workers.
        logits = _worker_forward(self.master_ctx, input_ids, past_len, seq_len,
                                 batch_size, last_id_only, graph_verify, return_row_states)

        # Collect the workers' replies so their errors surface here rather than as a hang.
        for device, w in sorted(self.workers.items()):
            w.recv()

        return logits

    def close(self):
        # Stop any in-flight prefill at its next chunk boundary so workers reach the
        # quit command promptly instead of finishing a multi-minute job first.
        try:
            self.abort_flag.set()
        except Exception:
            pass
        # Broadcast quit to ALL workers BEFORE joining any. The old sequential
        # quit/join left worker 1 blocked in NCCL teardown waiting for peers that
        # had not been told to quit yet -- every shutdown paid the full timeout
        # escalation per worker, and a stuck teardown froze the whole exit.
        for w in self.workers.values():
            try:
                w.conn.send("quit")
            except Exception:
                pass
        for w in self.workers.values():
            w.join_and_escalate()
        self.workers = {}
        # BUG 92: the master's NCCL teardown hangs indefinitely once the worker
        # ranks have already destroyed their comms and exited -- the main thread
        # was observed parked in destroy_process_group() after the test printed
        # its result, holding ~12 GiB on GPU 0 forever. Run it on a daemon
        # thread with a bounded join so a hung teardown can never block process
        # exit; CUDA context teardown at interpreter shutdown releases the VRAM.
        def _destroy():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        t = threading.Thread(target = _destroy, daemon = True)
        t.start()
        t.join(timeout = 10)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.close()


def _find_free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
