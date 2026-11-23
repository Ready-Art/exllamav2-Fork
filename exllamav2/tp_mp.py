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
              max_seq_len, cache_tokens, cache_base_name, max_batch = 1):
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
                           load_head = (rank == 0),
                           max_batch = max_batch)

        # Per-rank KV cache: this rank's slice of the KV split, under each layer's own
        # geometry. ExLlamaV2Cache_TP would build all four devices' caches.
        from exllamav2.cache import tp_rank_cache
        cache = tp_rank_cache(model, rank, base = ExLlamaV2Cache_SWA,
                              max_seq_len = max_seq_len, batch_size = max_batch)

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
                #        graph_verify, return_row_states[, return_row_aux])
                result = _worker_forward(ctx, *args)
                conn.send(("ok", result))
            elif cmd == "forward-state":
                # MTP: args: (input_ids, last_id_only, cpu_logits[, return_row_aux]).
                # Fan-out collective: every rank runs its slice; only rank 0 returns
                # (logits, last_state[, aux]). DFlash aux shards come from every rank.
                result = _worker_forward_state(ctx, *args)
                conn.send(("ok", result))
            elif cmd == "set-aux-taps":
                # DFlash: the module indices whose input residual is captured on
                # return_row_aux forwards. Must match on every rank (the capture is
                # collective; a divergent tap set would desync the graph state mode).
                ctx["model"]._dflash_aux_taps = list(args[0])
                conn.send(("ok", None))
            elif cmd == "head-shard":
                # DFlash: this rank's vocab-parallel lm_head shard (CPU fp16), so the
                # parent can assemble the full head for the drafter's logits GEMM.
                m = ctx["model"]
                head = m.modules[m.head_layer_idx]
                conn.send(("ok", head.linear.weight.detach().cpu()))
            elif cmd == "rope-tables":
                # MTP drafter on this worker's GPU: lend the parent our RoPE tables
                # (CUDA IPC, zero-copy). They stay alive in this rank's persistent
                # device context; the parent caches the mapped tensors in a shim.
                m = ctx["model"]
                dc = m.get_device_context(ctx["rank"], scratch = False)
                conn.send(("ok", (dc.sin, dc.cos)))
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
            elif cmd == "forward-batch":
                # Multi-batch graphed step: (input_ids, seq_lens, window_begs, row_map,
                # last_id_only, graph_verify, return_row_states). The per-row lists are
                # the master's authoritative bookkeeping; the ring prep inside
                # forward_tp_batch mutates window_begs identically on every rank.
                result = _worker_forward_batch(ctx, *args)
                conn.send(("ok", result))
            elif cmd == "forward-row":
                # Eager prefill chunk into ONE cache row (row view): (row, input_ids,
                # last_id_only, preprocess_only).
                result = _worker_forward_row(ctx, *args)
                conn.send(("ok", result))
            elif cmd == "forward-state-row":
                # Solo decode-shaped forward on one row returning (logits, last_state):
                # (row, input_ids, last_id_only). MTP heal / recommit fallback.
                result = _worker_forward_state_row(ctx, *args)
                conn.send(("ok", result))
            elif cmd == "set-seqlens":
                # Distributed per-row rewind: replace the whole seq_lens list.
                ctx["cache"].seq_lens[:] = args[0]
                conn.send(("ok", None))
            elif cmd == "set-seqlen-row":
                ctx["cache"].seq_lens[args[0]] = args[1]
                conn.send(("ok", None))
            elif cmd == "reset-row":
                # Full reset of ONE slot (new lineage): position + ring origin zeroed.
                # Ring content beyond the (zero) valid prefix is never read.
                ctx["cache"].seq_lens[args[0]] = 0
                ctx["cache"].window_begs[args[0]] = 0
                conn.send(("ok", None))
            elif cmd == "truncate-row":
                # Partial prefix reuse: rewind ONE slot to position p (>= its ring
                # origin). K/V beyond p is ignored until overwritten -- full layers
                # by seq_len, SWA by the ring fill. No copies, no ring shift.
                r, p = args
                assert p >= ctx["cache"].window_begs[r]
                ctx["cache"].seq_lens[r] = p
                conn.send(("ok", None))
            elif cmd == "clone-row":
                # Shared-prefix prompt cache: copy src row's committed K/V state at
                # position p into dst row -- full layers [0, p), SWA ring prefix
                # [0, p - window_begs[src]) at the SAME ring origin. A physical copy
                # (slots never share live K/V). Bookkeeping adopts the source's origin.
                src, dst, p = args
                cache = ctx["cache"]
                wb = cache.window_begs[src]
                n = p - wb
                with torch.inference_mode():
                    for l in range(len(cache.key_states)):
                        kc, vc = cache.key_states[l], cache.value_states[l]
                        if kc is None:
                            continue
                        if kc.shape[1] == cache.kv_state_size:
                            kc[dst, :n].copy_(kc[src, :n])
                            vc[dst, :n].copy_(vc[src, :n])
                        else:
                            kc[dst, :p].copy_(kc[src, :p])
                            vc[dst, :p].copy_(vc[src, :p])
                cache.seq_lens[dst] = p
                cache.window_begs[dst] = wb
                _requant_shadow_row(cache, dst, p)
                conn.send(("ok", None))
            elif cmd == "kv-pack-batch":
                # Multi-batch MTP gather: per-row deltas of both drafter source layers,
                # packed into one pinned CPU tensor (see _worker_kv_pack_batch).
                conn.send(("ok", _worker_kv_pack_batch(ctx, *args)))
            elif cmd == "ring-snapshot-row":
                # Per-slot prompt-boundary ring snapshot (pinned CPU, row-scoped).
                r = args[0]
                backups = ctx.setdefault("ring_backup_rows", {})
                backups[r] = _clone_swa_ring(ctx["cache"].row_view(r))
                ctx.setdefault("ring_backup_rows_wb", {})[r] = ctx["cache"].window_begs[r]
                conn.send(("ok", None))
            elif cmd == "ring-restore-row":
                r, seq = args
                _restore_swa_ring(ctx["cache"].row_view(r),
                                  ctx["ring_backup_rows"][r], seq,
                                  ctx["ring_backup_rows_wb"][r])
                conn.send(("ok", None))
            elif cmd == "offload-row":
                # Prompt-cache RAM offload: D2H this rank's slice of ONE idle row
                # into worker-local host storage (the KV never crosses the pipe --
                # each rank keeps its own slice; the parent stores only metadata +
                # its own master slice + the MTP shadow). The row's prompt-boundary
                # ring snapshot, if any, is re-keyed to the entry id (zero-copy).
                # args: (row, pos, wb, entry_id). Reply: (meta, nbytes, snap_bytes)
                # or None when host allocation fails (parent skips the offload).
                row, pos, wb, entry_id = args
                try:
                    flat, meta = _pack_row_offload(ctx, row, pos, wb)
                    store = flat.clone()   # pageable storage; staging is reusable
                except Exception:
                    conn.send(("ok", None))
                    continue
                ctx.setdefault("cold_bundles", {})[entry_id] = (store, meta)
                snap_bytes = _detach_ring_snapshot(ctx, row, entry_id)
                conn.send(("ok", (meta, store.numel() * store.element_size(), snap_bytes)))
            elif cmd == "restore-row":
                # H2D a stored offload bundle back into ONE idle row, truncated to
                # position p (>= the bundle's ring origin). args:
                # (row, entry_id, p, wb, attach_snapshot). The bundle's meta carries
                # the ORIGINAL packed shapes; only the [0, p) / [0, p - wb) prefixes
                # are copied (the rest of the row is unread until overwritten).
                row, entry_id, p, wb, attach_snapshot = args
                store, meta = ctx["cold_bundles"][entry_id]
                _restore_row_offload(ctx["cache"], row, store, meta, p, wb)
                _requant_shadow_row(ctx["cache"], row, p)
                if attach_snapshot:
                    _attach_ring_snapshot(ctx, row, entry_id)
                conn.send(("ok", None))
            elif cmd == "drop-cold-bundle":
                # Free one offloaded lineage's worker-local storage + detached ring
                # snapshot (RAM-cap eviction / entry consumed by a full restore).
                entry_id = args[0]
                ctx.get("cold_bundles", {}).pop(entry_id, None)
                ctx.get("ring_backup_cold", {}).pop(entry_id, None)
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


def _worker_forward_state(ctx, input_ids, last_id_only, cpu_logits, return_row_aux = False):
    """
    MTP: one forward that ALSO returns the pre-final-norm hidden state.

    Only rank 0 (the master, which owns lm_head) returns a value; the other ranks
    run the same collective and return None.

    Change98: single-token state requests (the MTP bonus commit) now RUN through the
    CUDA-graph path -- the bonus graph copies the pre-final-norm state into a static
    buffer inside the capture. The returned state/logits are that runner's static
    buffers: callers must consume (copy / serialize) them before the next replay.
    Multi-token state requests (prefill) still take the eager path.

    DFlash (return_row_aux): also return the per-row aux states (the target-layer
    residual taps, concatenated). Prefill is seqpar-sharded, so EVERY rank returns
    its (rows, n_features * hidden_local) shard on CPU and the parent joins them;
    a full-width step (decode_repl) returns rank 0's (rows, n_features * hidden)
    only. The return is (logits, state, aux) on rank 0, aux-or-None on workers.
    """
    model = ctx["model"]
    cache = ctx["cache"]
    dev = ctx["device"]
    if input_ids.device.index != dev:
        input_ids = input_ids.to(dev)
    r = model.forward(input_ids, cache = cache, last_id_only = last_id_only,
                      return_last_state = True, cpu_logits = cpu_logits,
                      return_row_aux = return_row_aux)
    if r is None:
        return None
    if return_row_aux:
        logits, state, aux = r
        aux = aux[0].detach().cpu() if aux is not None else None
        if ctx["rank"] != 0:
            return _aux_shard_or_none(model, aux)
        return (logits, state, aux)
    if isinstance(r, tuple):
        logits, state = r
    else:
        logits, state = r, None
    if ctx["rank"] != 0:
        return None
    return (logits, state)


def _aux_shard_or_none(model, aux):
    """DFlash pipe protocol: a worker pipes its aux shard only when the states are
    seqpar-sharded (width < n_features * hidden). Full-width steps (decode_repl
    verify) are rank-0-only -- workers return None and the parent uses rank 0's."""
    if aux is None:
        return None
    full_w = len(model._dflash_aux_taps) * model.config.hidden_size
    return aux if aux.shape[-1] < full_w else None


def _join_aux(aux0, worker_auxes, n_features, hidden_size):
    """Join the ranks' DFlash aux shards into (rows, n_features * hidden).

    Each rank's tensor is (rows, n_features * hidden_local): the concat of its
    per-tap hidden shards. Viewing as (rows, n_features, hidden_local) and
    concatenating along dim 2 in RANK order reconstructs every tap's full hidden
    state (the RS split assigns contiguous rank-ordered ranges, set_split).
    Full-width rank-0 states (all workers None) pass through unchanged.
    """
    full_w = n_features * hidden_size
    if all(a is None for a in worker_auxes):
        assert aux0.shape[-1] == full_w, \
            f"DFlash aux: rank-0 width {aux0.shape[-1]} != full {full_w}"
        return aux0
    parts = [aux0] + list(worker_auxes)
    rows = aux0.shape[0]
    shards = [p.view(rows, n_features, p.shape[-1] // n_features) for p in parts]
    full = torch.cat(shards, dim = 2)
    assert full.shape[2] == hidden_size, \
        f"DFlash aux shard join width {full.shape[2]} != hidden {hidden_size}"
    return full.reshape(rows, full_w)


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


def _requant_shadow_row(cache, row, n):
    """
    Re-quantize the INT8 KV shadow for a row whose fp16 KV was just cloned/restored.
    The shadow is derived state: a raw fp16 copy (clone-row, restore-row) leaves it
    stale, and the int8 decode would read garbage for the copied prefix.
    """
    sh = getattr(cache, "kv_i8", None)
    if not sh or n <= 0:
        return
    from exllamav2.ext import exllamav2_ext as ext_c
    with torch.inference_mode():
        for l, (ki, vi, ks, vs) in sh.items():
            kc, vc = cache.key_states[l], cache.value_states[l]
            if kc is None:
                continue
            dev = kc.device
            pos = torch.zeros(1, dtype = torch.int32, device = dev)
            rmap = torch.tensor([row], dtype = torch.int32, device = dev)
            ext_c.kv_quant_shadow(kc, vc, ki, vi, ks, vs, pos, 1, n,
                                  kc.shape[2], kc.shape[3], rmap)


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


# ---------------------------------------------------------------------------
# Prompt-cache RAM offload (opt-in): row-level D2H / H2D of committed K/V.
#
# Each rank stores its OWN slice of an offloaded lineage in host RAM (worker:
# ctx["cold_bundles"], master: the parent's bundle dict). The KV never crosses
# the process pipe -- only metadata does. The MTP drafter's joined shadow row
# and the prompt-boundary ring snapshots travel with the lineage (the snapshots
# are already host-resident per rank; offload/restore just re-keys ownership).
#
# VRAM safety: every copy targets the PREALLOCATED cache rows / shadow buffers.
# Nothing here allocates device memory, so offload/restore cannot OOM VRAM.
# Host safety: storage allocation failures abort the offload gracefully (the
# scheduler falls back to a full re-prefill); the RAM cap is enforced by the
# scheduler's cold store BEFORE it calls offload_row.
# ---------------------------------------------------------------------------

def _pack_row_offload(ctx, row, pos, wb):
    """D2H one idle cache row's committed K/V into a flat pinned staging tensor.

    Full-attention layers: [0, pos). SWA layers: the ring's valid prefix
    [0, min(pos - wb, ring)) -- the stale tail beyond the fill is never read
    (attention is bounded by swa_len; the same invariant clone_row relies on).
    Layout: per layer in index order, K block then V block, each (rows, kvh, hd).
    meta = (pos, wb, [(layer, rows, kvh, hd), ...]) lets the restore unpack the
    ORIGINAL shapes even when restoring a truncated prefix p < pos.

    Grow-only pinned staging (one allocation amortized across offloads), async
    copies + a single sync -- the kv-pack pattern. The caller clones the returned
    view into storage before the staging is reused."""
    cache = ctx["cache"]
    ring = cache.kv_state_size
    n = max(0, min(pos - wb, ring))
    blocks = []
    total = 0
    for l in range(len(cache.key_states)):
        kc = cache.key_states[l]
        if kc is None:
            continue
        rows = n if kc.shape[1] == ring else pos
        if rows <= 0:
            continue
        kh, hd = kc.shape[2], kc.shape[3]
        blocks.append((l, rows, kh, hd))
        total += 2 * rows * kh * hd
    stag = ctx.get("offload_staging")
    if stag is None or stag.numel() < total:
        stag = ctx["offload_staging"] = torch.empty(max(total, 1), dtype = torch.half,
                                                    pin_memory = True)
    o = 0
    for (l, rows, kh, hd) in blocks:
        for t in (cache.key_states[l][row, :rows], cache.value_states[l][row, :rows]):
            stag[o:o + t.numel()].view(t.shape).copy_(t, non_blocking = True)
            o += t.numel()
    torch.cuda.current_stream().synchronize()
    return stag[:total], (pos, wb, blocks)


def _restore_row_offload(cache, row, flat, meta, p, wb):
    """H2D an offloaded bundle's prefixes into one idle row at position p.

    Unpacks with the ORIGINAL packed shapes (meta), copies full layers [0, p)
    and the SWA ring prefix [0, p - wb), then adopts the bookkeeping. At
    batch_size 1 the scalar current_seq_len / window_beg fields are synced too
    (the verified single-sequence path reads those, not the row lists)."""
    pos0, wb0, blocks = meta
    assert wb <= p <= pos0, f"restore_row_offload: bad position {p} (wb {wb}, pos0 {pos0})"
    ring = cache.kv_state_size
    o = 0
    with torch.inference_mode():
        for (l, rows, kh, hd) in blocks:
            kc, vc = cache.key_states[l], cache.value_states[l]
            take = min(p - wb, rows) if kc.shape[1] == ring else min(p, rows)
            k = flat[o:o + rows * kh * hd].view(rows, kh, hd)
            o += rows * kh * hd
            v = flat[o:o + rows * kh * hd].view(rows, kh, hd)
            o += rows * kh * hd
            if take > 0:
                kc[row, :take].copy_(k[:take], non_blocking = True)
                vc[row, :take].copy_(v[:take], non_blocking = True)
        torch.cuda.current_stream().synchronize()
    cache.seq_lens[row] = p
    cache.window_begs[row] = wb
    if cache.batch_size == 1:
        cache.current_seq_len = p
        cache.window_beg = wb


def _detach_ring_snapshot(ctx, row, entry_id):
    """Move a row's prompt-boundary ring snapshot into cold storage keyed by the
    offload entry id (zero-copy -- the snapshot is already pinned host RAM).
    Returns the snapshot's byte size (0 when the row had none). Handles both the
    per-row batch snapshots and the scalar batch-1 snapshot."""
    cache = ctx["cache"]
    cold = ctx.setdefault("ring_backup_cold", {})
    backups = ctx.get("ring_backup_rows", {})
    if row in backups:
        backup = backups.pop(row)
        wb = ctx["ring_backup_rows_wb"].pop(row)
        seq = ctx.get("ring_backup_rows_seq", {}).pop(row, None)
    elif cache.batch_size == 1 and ctx.get("ring_backup") is not None:
        backup = ctx.pop("ring_backup")
        wb = ctx.pop("ring_backup_wb")
        seq = None   # the scalar restore takes the position as an argument
    else:
        return 0
    cold[entry_id] = (backup, wb, seq)
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in backup.values())


def _attach_ring_snapshot(ctx, row, entry_id):
    """Move a detached ring snapshot back to a row (full-lineage restore). The
    snapshot keeps its ORIGINAL prompt-boundary position -- exactly the state a
    hot row would carry, so the thought-stripping restore path works unchanged."""
    cold = ctx.get("ring_backup_cold", {})
    if entry_id not in cold:
        return
    backup, wb, seq = cold.pop(entry_id)
    if ctx["cache"].batch_size == 1:
        ctx["ring_backup"] = backup
        ctx["ring_backup_wb"] = wb
    else:
        ctx.setdefault("ring_backup_rows", {})[row] = backup
        ctx.setdefault("ring_backup_rows_wb", {})[row] = wb
        if seq is not None:
            ctx.setdefault("ring_backup_rows_seq", {})[row] = seq


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
                    graph_verify = False, return_row_states = False, return_row_aux = False):
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

    ``return_row_aux`` (DFlash): return the per-row aux states instead of the final-norm
    states -- rank 0 ``(logits, aux)`` (aux is (rows, n_features*hidden), CPU); workers
    their aux shard when seqpar-sharded, else None (see _aux_shard_or_none). The verify
    window is decode_repl (full width), so this is normally rank-0-only; the graph's aux
    buffer is static -- the parent consumes it before the next replay.
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
        return_row_aux = return_row_aux,
    )
    if r is None:
        return None
    if return_row_aux:
        logits, _state, aux = r
        aux = aux[0].detach().cpu() if aux is not None else None
        if ctx["rank"] != 0:
            return _aux_shard_or_none(model, aux)
        return (logits, aux)
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


def _worker_forward_batch(ctx, input_ids, seq_lens, window_begs, row_map,
                          last_id_only, graph_verify, return_row_states):
    """Multi-batch graphed step on this rank. The master's per-row bookkeeping is
    authoritative: adopt it, then run the same deterministic ring prep + graphed step
    every rank runs. Only rank 0 returns (logits[, row_states])."""
    model = ctx["model"]
    cache = ctx["cache"]
    dev = ctx["device"]
    if input_ids.device.index != dev:
        input_ids = input_ids.to(dev)
    # Per-BATCH-ROW lists (length B, the round's active subset): adopt them into the
    # cache rows named by row_map, leaving every other slot's bookkeeping intact.
    # forward_tp_batch persists its ring shifts back into cache.window_begs[r].
    seq_lens = list(seq_lens)
    wb = list(window_begs)
    row_map = list(row_map)
    for b, r in enumerate(row_map):
        cache.seq_lens[r] = seq_lens[b]
        cache.window_begs[r] = wb[b]
    r = model.forward_tp_batch(
        input_ids, cache, seq_lens, wb, row_map,
        last_id_only = last_id_only, graph_verify = graph_verify,
        return_row_states = return_row_states,
    )
    if ctx["rank"] != 0:
        return None
    return r


def _worker_forward_row(ctx, row, input_ids, last_id_only, preprocess_only):
    """Eager prefill chunk into one cache row via its persistent row view. The scalar
    chunk loop / SWA ring protocol runs unchanged, scoped to the row. Graphs are
    disabled: a row view is a different cache identity than the batched graphs' full
    cache, and prefill is eager by design."""
    model = ctx["model"]
    cache = ctx["cache"]
    dev = ctx["device"]
    if input_ids.device.index != dev:
        input_ids = input_ids.to(dev)
    r = model.forward(input_ids, cache = cache.row_view(row),
                      last_id_only = last_id_only, preprocess_only = preprocess_only,
                      use_graphs = False)
    if ctx["rank"] != 0:
        return None
    return r


def _worker_forward_state_row(ctx, row, input_ids, last_id_only):
    """Solo decode-shaped forward on one row returning (logits, last_state) on rank 0.
    Eager (use_graphs=False): the heal/recommit path must not capture row-view graphs
    that would thrash the batched graphs' cache identity."""
    model = ctx["model"]
    cache = ctx["cache"]
    dev = ctx["device"]
    if input_ids.device.index != dev:
        input_ids = input_ids.to(dev)
    r = model.forward(input_ids, cache = cache.row_view(row),
                      last_id_only = last_id_only, return_last_state = True,
                      cpu_logits = True, use_graphs = False)
    if ctx["rank"] != 0:
        return None
    if isinstance(r, tuple):
        return r
    return (r, None)


def _worker_kv_pack_batch(ctx, lf, ls, specs):
    """Multi-batch MTP gather: this rank's per-row K/V deltas for both drafter source
    layers in ONE pinned CPU tensor. ``specs`` is a list of (row, fl0, fl1, sl0, sl1)
    in the parent's request order; the flat layout concatenates each row's
    [K_full | V_full | K_swa | V_swa] block, so the parent unpacks zero-copy views
    with the shapes it already knows."""
    cache = ctx["cache"]

    def clamp(layer, a, b):
        width = cache.key_states[layer].shape[1]
        a = max(0, min(a, width))
        b = max(a, min(b, width))
        return a, b

    parts = []
    n = 0
    for row, fl0, fl1, sl0, sl1 in specs:
        fl0, fl1 = clamp(lf, fl0, fl1)
        sl0, sl1 = clamp(ls, sl0, sl1)
        for t in (cache.key_states[lf][row, fl0:fl1],
                  cache.value_states[lf][row, fl0:fl1],
                  cache.key_states[ls][row, sl0:sl1],
                  cache.value_states[ls][row, sl0:sl1]):
            parts.append(t.detach())
            n += t.numel()
    stag = ctx.get("kv_pack_staging")
    if stag is None or stag.numel() < n:
        stag = ctx["kv_pack_staging"] = torch.empty(max(n, 1), dtype = torch.half,
                                                    pin_memory = True)
    o = 0
    for t in parts:
        stag[o:o + t.numel()].view(t.shape).copy_(t, non_blocking = True)
        o += t.numel()
    torch.cuda.current_stream().synchronize()
    return stag[:n]


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
                 master_device = 0, verbose = False, max_batch = 1):
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
        # Multi-batch (slot scheduler): number of independent cache rows. The KV cache
        # and the MTP shadows scale linearly with this; 1 keeps the verified single-
        # sequence path untouched.
        self.max_batch = max(1, int(max_batch))
        self.workers: dict[int, TPWorkerHandle] = {}
        self.master_ctx = None
        self.model = None
        self.cache = None
        # MTP: drafter (single-device, parent process) + its draft window. See
        # exllamav2/mtp_mp.py (MTPMPDriver) for the round protocol.
        self.mtp_drafter = None
        self.num_speculative_tokens = 6
        self._mtp_kv = None
        # RoPE-table shims for drafters on NON-master devices (get_device_context).
        self._rope_ctx_shims = {}
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
                        self.cache_tokens * self.max_batch, self.cache_base_name,
                        self.max_batch),
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
                           expect_cache_tokens = self.cache_tokens * self.max_batch,
                           expect_cache_base = ExLlamaV2Cache_SWA,
                           load_head = True,
                           max_batch = self.max_batch)
        cache = tp_rank_cache(model, self.master_device, base = ExLlamaV2Cache_SWA,
                              max_seq_len = self.max_seq_len, batch_size = self.max_batch)

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
        if device_idx == -1:
            return None
        if device_idx == self.master_device:
            return self.model.get_device_context(device_idx, scratch)
        # Drafter on a WORKER GPU (draft_device != master): the master model's
        # device_context list is [None] * devs with a real context only at its own
        # rank (load_tp_rank), so forwarding would return None (fix.txt crash).
        # The only consumer is the MTP drafter's _get_rope (scratch=False), which
        # needs ctx.sin/cos on the drafter device. Borrow the owning worker's
        # tables via CUDA IPC -- zero-copy (the fork already pipes CUDA tensors
        # for states); they live in the worker's persistent device context, so
        # the mappings stay valid for the group's lifetime. No extra VRAM.
        shim = self._rope_ctx_shims.get(device_idx)
        if shim is None:
            import types
            sin, cos = self.workers[device_idx].call("rope-tables")
            shim = types.SimpleNamespace(sin = sin, cos = cos, ready = True)
            self._rope_ctx_shims[device_idx] = shim
        return shim

    def enable_mtp(self, drafter, num_draft_tokens = 6):
        """Attach the MTP drafter so :meth:`forward_state` / :meth:`gather_kv_mtp`
        serve the draft-verify loop. The drafter is a parent-process, single-device
        model; it reads the target's sharded K/V via :meth:`gather_kv_mtp`.

        DFlash drafters (is_dflash) take a different wiring: no K/V shadow (the
        draft KV is derived from target hidden states, owned by the drafter),
        aux-tap capture configured on every rank, and a full lm_head copy
        assembled from the vocab shards for the draft logits GEMM."""
        self.mtp_drafter = drafter
        self.num_speculative_tokens = int(num_draft_tokens)
        if getattr(drafter, "is_dflash", False):
            self._enable_dflash(drafter)
            return
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
                if self.max_batch == 1:
                    drafter.gather_target(self)
                    # last_state is (1, q_len, H) from the eager multi-token prefill -- the
                    # drafter consumes only the LAST row (same slice mtp_mp.begin() applies).
                    drafter.draft_window(torch.tensor([[0]], dtype = torch.long),
                                         _state[:, -1:, :].float().cpu(), 2, 1)
                # At max_batch > 1 the joined shadow is 4-D (per-row); the scalar
                # gather/draft above assumes the 3-D layout. The batch warmup below
                # compiles the drafter kernels through the batched path instead.
                self.mtp_reset_cache()
                print(f"[mtp] warmup done in {_time.perf_counter() - _t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[mtp] warmup skipped ({type(e).__name__}: {e}); first request "
                      f"will pay FlashInfer JIT", flush=True)
                try:
                    self.mtp_reset_cache()
                except Exception:
                    pass

        # Multi-batch warmup: pre-capture EVERY batch size's decode + verify graphs
        # and JIT the batched drafter kernels at load, so the first concurrent request
        # doesn't pay for them. With EXL2_MTP_DRAFT_GRAPH=1 the whole-window batch
        # draft graph is captured here too (kv-length-safe: replanned per round).
        if self.max_batch > 1 and os.environ.get("EXL2_MTP_WARMUP", "1") != "0":
            try:
                import time as _time
                _t0 = _time.perf_counter()
                print("[mtp] batch warmup: pre-compiling multi-batch kernels "
                      f"(B=1..{self.max_batch})...", flush=True)
                W = self.num_speculative_tokens
                draft_graph = os.environ.get("EXL2_MTP_DRAFT_GRAPH", "0") == "1"
                for B in range(1, self.max_batch + 1):
                    dummy = torch.zeros((B, 1), dtype = torch.long)
                    self.forward_batch(dummy, [0] * B, list(range(B)), last_id_only = True)
                    # Verify shape (B, W+1) with per-row states: captures the round
                    # graph and JITs the q_len>1 attention kernels (paged prefill +
                    # batched SWA decode-rows).
                    self.forward_batch(torch.zeros((B, W + 1), dtype = torch.long),
                                       [1] * B, list(range(B)), last_id_only = False,
                                       graph_verify = True, return_row_states = True)
                    self.mtp_set_seq_lens([1] * B + [0] * (self.max_batch - B))
                    self.gather_kv_mtp_batch(list(range(B)))
                    drafter.gather_target_batch(list(range(B)))
                    drafter.draft_window_batch(
                        torch.zeros((B, 1), dtype = torch.long),
                        torch.zeros((B, 1, drafter.backbone_hidden_size)),
                        [1] * B, 1)
                    if draft_graph:
                        drafter.draft_window_batch(
                            torch.zeros((B, 1), dtype = torch.long),
                            torch.zeros((B, 1, drafter.backbone_hidden_size)),
                            [1] * B, W)
                    for r in range(B):
                        self.mtp_reset_row(r)
                print(f"[mtp] batch warmup done in {_time.perf_counter() - _t0:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"[mtp] batch warmup skipped ({type(e).__name__}: {e}); first "
                      f"concurrent request will pay the capture/JIT", flush=True)
                try:
                    for r in range(self.max_batch):
                        self.mtp_reset_row(r)
                except Exception:
                    pass

    def _enable_dflash(self, drafter):
        """DFlash wiring: aux taps on every rank, the full lm_head copy (vocab
        shards gathered), the drafter's KV allocation, and load-time warmup of
        the aux path + the drafter's FlashInfer prefill kernels."""
        assert self.max_batch == 1, "DFlash currently requires max_batch_size 1"
        # Aux taps: decoder layer L's output is the residual BEFORE the first
        # module of layer L+1 runs. Derive the indices from the module keys --
        # Gemma-4 uses THREE modules per layer (attn, mlp, layer_scalar), so a
        # fixed stride would tap the wrong stream. tap = last module of L, + 1.
        import re
        last_of_layer = {}
        for i, m in enumerate(self.model.modules[: self.model.head_layer_idx]):
            mt = re.search(r"\.layers\.(\d+)(?:\.|$)", m.key)
            if mt:
                last_of_layer[int(mt.group(1))] = i
        assert all(L in last_of_layer for L in drafter.target_layer_ids), \
            "DFlash aux taps: target layer modules not found"
        taps = [last_of_layer[L] + 1 for L in drafter.target_layer_ids]
        self.model._dflash_aux_taps = taps
        for device, w in sorted(self.workers.items()):
            w.send("set-aux-taps", (taps,))
        for device, w in sorted(self.workers.items()):
            w.recv()

        # Full lm_head for the draft logits GEMM. Vocab-parallel head: gather every
        # rank's shard (the VC split assigns contiguous rank-ordered row ranges);
        # replicated head (EXL2_TP_VOCAB_HEAD=0): the master already holds it all.
        head_mod = self.model.modules[self.model.head_layer_idx]
        if getattr(head_mod, "_tp_vocab_parallel", False):
            shards = [head_mod.linear.weight.detach().cpu()]
            for device, w in sorted(self.workers.items()):
                shards.append(w.call("head-shard"))
            weight = torch.cat(shards, dim = 0)
        else:
            weight = head_mod.linear.weight.detach().clone()
        drafter.set_head(weight, head_mod.padding)

        # Draft KV at LOAD time with a clean allocator (the _alloc_mtp_kv lesson:
        # lazy allocation after a long prefill OOMs on a fragmented allocator).
        drafter.alloc_kv(self.max_seq_len)
        self._mtp_kv = None   # DFlash keeps no MTP gather shadow
        if os.environ.get("EXL2_VRAM_TRACE", "1") != "0":
            self._vram_breakdown()

        # VRAM fit check: the drafter stack (~6.4 GB @32k cache) is master-only by
        # default. On 2-GPU splits the doubled weight shard leaves <1 GB on the
        # master for prefill transients -> first-chunk OOM. The drafter is fully
        # device-parameterized; a worker GPU usually has the room. Warn, don't
        # auto-move (placement stays the user's knob).
        free_b, _ = torch.cuda.mem_get_info(drafter.dev)
        if free_b < 1536 * 2**20:
            best = max(range(torch.cuda.device_count()),
                       key = lambda i: torch.cuda.mem_get_info(i)[0])
            best_free = torch.cuda.mem_get_info(best)[0]
            print(f"[dflash] WARNING: only {free_b / 2**20:.0f} MiB free on "
                  f"cuda:{drafter.device} after drafter allocation -- prefill may "
                  f"OOM. Consider draft_device: {best} "
                  f"({best_free / 2**20:.0f} MiB free there).", flush = True)

        # Cold-start fix (same pattern as MTP): FlashInfer JIT-compiles each kernel
        # geometry on first use; pay it at load, not inside the first request.
        if os.environ.get("EXL2_MTP_WARMUP", "1") != "0":
            try:
                import time as _time
                _t0 = _time.perf_counter()
                print("[dflash] warmup: pre-compiling FlashInfer kernels "
                      "(slow only on the very first load)...", flush=True)
                dummy = torch.tensor([[0, 0]], dtype = torch.long)
                _logits, _state, _aux = self.forward_state(
                    dummy, last_id_only = True, cpu_logits = True, return_row_aux = True)
                drafter.feed_context(_aux, 0)
                drafter.draft_window(torch.tensor([[0]], dtype = torch.long), None, 2, 1)
                self.mtp_reset_cache()
                print(f"[dflash] warmup done in {_time.perf_counter() - _t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[dflash] warmup skipped ({type(e).__name__}: {e}); first request "
                      f"will pay FlashInfer JIT", flush=True)
                try:
                    self.mtp_reset_cache()
                except Exception:
                    pass

    def forward_state(self, input_ids, last_id_only = True, cpu_logits = True,
                      return_row_aux = False):
        """Distributed forward that returns ``(logits, last_state)``.

        ``last_state`` is the pre-final-norm hidden state of the last processed
        token (what the MTP drafter consumes as its round base). Every rank runs
        the collective; only the master produces the value.

        DFlash (return_row_aux=True): returns ``(logits, last_state, aux)`` where
        ``aux`` is (rows, n_features * hidden) on CPU -- the ranks' seqpar shards
        joined along the hidden dim (prefill), or the master's full-width states
        (decode-shaped steps). Consumed by DFlashDrafter.feed_context.
        """
        for device, w in sorted(self.workers.items()):
            w.send("forward-state", (input_ids, last_id_only, cpu_logits, return_row_aux))
        out = _worker_forward_state(self.master_ctx, input_ids, last_id_only, cpu_logits,
                                    return_row_aux)
        worker_aux = [w.recv() for _, w in sorted(self.workers.items())]
        if return_row_aux and out is not None:
            logits, state, aux0 = out
            aux = _join_aux(aux0, worker_aux, len(self.model._dflash_aux_taps),
                            self.config.hidden_size)
            return logits, state, aux
        return out

    def mtp_set_seq_len(self, n):
        """Distributed rewind: moves the master cache AND every worker cache to ``n``.
        The SWA ring shift is a deterministic function of ``current_seq_len`` (BUG 94),
        so moving all ranks in lockstep keeps every ring in sync with no extra traffic."""
        for device, w in sorted(self.workers.items()):
            w.send("set-seqlen", (n,))
            w.recv()
        self.cache.current_seq_len = n
        # DFlash: the draft context length tracks the committed target length.
        # Round rewinds land at or above ctx_len (no-op); external trims shrink it.
        if getattr(self.mtp_drafter, "is_dflash", False):
            self.mtp_drafter.trim(n)

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
        # DFlash: the draft KV for the restored prefix is still valid (same lineage,
        # written when those tokens were first committed); only trim the length.
        if getattr(self.mtp_drafter, "is_dflash", False):
            self.mtp_drafter.trim(seq)
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
        # DFlash: new lineage -- drop the draft context (buffers kept).
        if getattr(self.mtp_drafter, "is_dflash", False):
            self.mtp_drafter.reset()
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
                         if torch.is_tensor(t)) if self._mtp_kv is not None else 0
            draft = 0
            d = self.mtp_drafter
            for obj in [d] + list(getattr(d, "layers", [])):
                for v in vars(obj).values():
                    # CUDA only: the drafter also holds a full CPU embedding copy
                    # (DFlash/MTP), which must not inflate the GPU attribution.
                    if torch.is_tensor(v) and v.is_cuda:
                        draft += v.numel() * v.element_size()
            dflash_kv = 0
            if getattr(d, "is_dflash", False) and d.k_cache is not None:
                dflash_kv = sum(t.numel() * t.element_size()
                                for t in d.k_cache + d.v_cache)
                draft += dflash_kv
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
                  f"drafter={mb(draft):.0f} dflash_kv={mb(dflash_kv):.0f} "
                  f"kv_cache={mb(kv):.0f}", flush=True)
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
        # Page-align the full-layer shadow depth: the drafter's graphed FI attention
        # page-views these buffers (depth % 128 == 0 required, same as the cache).
        sdepth = ((self.max_seq_len + 127) // 128) * 128
        # inference_mode(FALSE): enable_mtp runs under the loader's InferenceMode, but
        # gather_kv_mtp mutates these buffers IN PLACE on the job thread OUTSIDE it
        # (ring-shift mirror copy_, delta H2D copy_). Inference tensors forbid that
        # (debug11.txt); the old lazy allocation created them outside InferenceMode as
        # normal tensors -- preserve exactly that semantics here at load time.
        with torch.inference_mode(False):
            if self.max_batch > 1:
                # Multi-batch: one shadow row per cache slot (4-D). VRAM scales linearly
                # with max_batch (~1.25 GB/row at 160k for Gemma-4's joined source layers);
                # preallocated here at load with a clean allocator, never lazily.
                B = self.max_batch
                self._mtp_kv = {
                    "fl": 0, "sl": 0, "wb": 0,   # legacy scalar keys (unused at B > 1)
                    "fls": [0] * B,              # per-row full-layer gather frontier
                    "sls": [0] * B,              # per-row absolute seq_len at last SWA gather
                    "wbs": [0] * B,              # per-row window_beg at last SWA gather
                    "kf": torch.zeros((B, sdepth, d.num_global_kv_heads, d.global_head_dim),
                                      dtype = torch.half, device = dev),
                    "vf": torch.zeros((B, sdepth, d.num_global_kv_heads, d.global_head_dim),
                                      dtype = torch.half, device = dev),
                    "ks": torch.zeros((B, mc.kv_state_size, d.num_kv_heads, d.head_dim),
                                      dtype = torch.half, device = dev),
                    "vs": torch.zeros((B, mc.kv_state_size, d.num_kv_heads, d.head_dim),
                                      dtype = torch.half, device = dev),
                }
            else:
                self._mtp_kv = {
                    "fl": 0,
                    "sl": 0,      # absolute seq_len at the last SWA gather
                    "wb": 0,      # window_beg at the last SWA gather
                    # 3-D (S, kvh, hd) -- identical layout to the single-process drafter.
                    "kf": torch.zeros((sdepth, d.num_global_kv_heads, d.global_head_dim),
                                      dtype = torch.half, device = dev),
                    "vf": torch.zeros((sdepth, d.num_global_kv_heads, d.global_head_dim),
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
        if "fls" in st:
            B = len(st["fls"])
            st["fls"] = [0] * B
            st["sls"] = [0] * B
            st["wbs"] = [0] * B

    def _reset_mtp_kv_row(self, r):
        """Re-zero ONE slot's shadow rows + bookkeeping (slot lineage reset). Keeps the
        buffers allocated; other slots' gathered K/V is untouched."""
        st = self._mtp_kv
        if st is None or "fls" not in st:
            return
        with torch.inference_mode():
            for _k in ("kf", "vf", "ks", "vs"):
                st[_k][r].zero_()
        st["fls"][r] = 0
        st["sls"][r] = 0
        st["wbs"][r] = 0

    def _kv_head_widths(self, layer_idx):
        """Per-rank KV head counts for one layer, from the target's geometry split.

        The TP cache shards KV heads with integer_split (largest-remainder), so on
        non-power-of-two GPU counts the ranks hold DIFFERENT widths (Gemma-4 on 3
        GPUs: full kv 4 -> 2/1/1, SWA kv 16 -> 6/5/5). The MTP gather joins the
        ranks' actual slices along the head axis, so it must unpack each rank's real
        width -- not total // world (which only matched when every rank held the
        same count). A rank absent from the split holds 0 heads of that geometry
        (e.g. full kv 4 across 5 GPUs -> 1/1/1/1/0); callers use .get(dev, 0)."""
        hd, qh, kvh, full = self.config.get_layer_attn_geometry(layer_idx)
        split = self.model.tp_context.geometry_splits[(hd, qh, kvh, full)]["kv"][0]
        return {dev: b - a for dev, a, b in split}

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

        # Opt-in sub-phase cost trace (EXL2_MTP_COST_TRACE=1): separates the mirror
        # device copy, the master D2H + pipe sends, the worker-reply wait (worker pack
        # + stream sync), and the shadow H2D -- pins where the long-ctx gather cost
        # lives. No overhead when unset.
        _trace = os.environ.get("EXL2_MTP_COST_TRACE", "0") == "1"
        if _trace:
            import time as _time
            _tc = _time.perf_counter
            _t0 = _tc()
            _tm = {}

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
        if _trace:
            _tm["mirror"] = _tc() - _t0
            _t0 = _tc()

        # ---- packed pull: ONE message per worker (was 2 commands + 2 replies each) ----
        # Each worker packs [K_full | V_full | K_swa | V_swa] deltas into one pinned CPU
        # tensor (async D2H + single sync); the parent unpacks zero-copy views, joins the
        # ranks along the kv-head axis on CPU (a few KB), and copies straight into the
        # device shadow -- 4 H2D total instead of 8 H2D + 4 device cats + 4 assigns.
        # Per-rank KV head widths (uneven splits on 3/5 GPUs: integer_split gives e.g.
        # full kv 4 -> 2/1/1, SWA kv 16 -> 6/5/5). The worker packs its ACTUAL slice;
        # the parent must unpack each rank's real width and join in device order, not
        # assume total // world. The master (device 0) slice is appended directly.
        fw = self._kv_head_widths(lf)
        sw = self._kv_head_widths(ls)
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
        if _trace:
            _tm["cpu"] = _tc() - _t0
            _t0 = _tc()
        for device, w in sorted(self.workers.items()):
            flat = w.recv()
            o = 0
            fkh, skh = fw.get(device, 0), sw.get(device, 0)
            for rows, kh, hd, acc in ((df, fkh, d.global_head_dim, full_k),
                                      (df, fkh, d.global_head_dim, full_v),
                                      (ds, skh, d.head_dim, swa_k),
                                      (ds, skh, d.head_dim, swa_v)):
                n = rows * kh * hd
                acc.append(flat[o:o + n].view(rows, kh, hd))
                o += n
        if _trace:
            _tm["recv"] = _tc() - _t0
            _t0 = _tc()
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

        if _trace:
            _tm["h2d"] = _tc() - _t0
            tr = getattr(self, "_kv_trace", None)
            if tr is None:
                tr = self._kv_trace = {"n": 0, "mirror": 0.0, "cpu": 0.0,
                                       "recv": 0.0, "h2d": 0.0, "df": 0, "ds": 0,
                                       "shifts": 0, "repulls": 0}
            tr["n"] += 1
            for _k in ("mirror", "cpu", "recv", "h2d"):
                tr[_k] += _tm[_k]
            tr["df"] += df
            tr["ds"] += ds
            tr["shifts"] += 1 if swa_shift > 0 else 0
            tr["repulls"] += 1 if (swa_start == 0 and ds == swa_len and swa_len > 0) else 0
            if tr["n"] % 50 == 0:
                _n = tr["n"]
                print(f"[kv-trace] {_n} gathers ctx={seq_len} ms/gather "
                      f"mirror={tr['mirror']/_n*1000:.2f} cpu={tr['cpu']/_n*1000:.2f} "
                      f"recv={tr['recv']/_n*1000:.2f} h2d={tr['h2d']/_n*1000:.2f} "
                      f"avg_df={tr['df']/_n:.1f} avg_ds={tr['ds']/_n:.1f} "
                      f"shifts={tr['shifts']} repulls={tr['repulls']}", flush=True)

        return (st["kf"][:seq_len], st["vf"][:seq_len],
                st["ks"][:swa_len], st["vs"][:swa_len],
                seq_len, window_beg)

    def gather_kv_mtp_batch(self, rows):
        """Multi-batch incremental gather: refresh the joined shadow rows for ``rows``
        (the active verify batch). Same delta protocol as gather_kv_mtp, per row: mirror
        each row's SWA page shift on its shadow, pull only the new full/SWA rows, ONE
        packed pipe message per worker covering every row. The drafter then reads the
        4-D shadows directly (gather_target_batch)."""
        assert self.mtp_drafter is not None, "call enable_mtp(drafter) first"
        d = self.mtp_drafter
        lf, ls = d.t_last_full, d.t_last_swa
        mc = self.cache
        st = self._mtp_kv
        assert st is not None and st["kf"].dim() == 4, "batch gather needs max_batch > 1"
        ring = mc.kv_state_size
        # Per-rank KV head widths (uneven splits on 3/5 GPUs); see gather_kv_mtp.
        fw = self._kv_head_widths(lf)
        sw = self._kv_head_widths(ls)

        specs, meta = [], []
        for r in rows:
            seq_len = mc.seq_lens[r]
            # Fail loudly at the source if a row ever runs past the cache depth: the
            # worker's pack clamps to the buffer width while the parent's delta math
            # doesn't, so an overflow desyncs the flat layout (cryptic view error).
            assert seq_len <= mc.max_seq_len, \
                f"gather_kv_mtp_batch: row {r} position {seq_len} exceeds cache depth " \
                f"{mc.max_seq_len} (generation must be capped at admission)"
            wb = mc.window_begs[r]
            swa_len = max(0, min(seq_len - wb, ring))
            swa_shift = wb - st["wbs"][r]
            swa_start = max(st["sls"][r], wb) - wb
            if swa_shift < 0 or seq_len < st["sls"][r] or swa_start > swa_len:
                # Defensive full re-pull for this row (bookkeeping moved backwards).
                swa_shift = 0
                swa_start = 0
            if 0 < swa_shift < ring:
                st["ks"][r, :ring - swa_shift].copy_(st["ks"][r, swa_shift:].clone())
                st["vs"][r, :ring - swa_shift].copy_(st["vs"][r, swa_shift:].clone())
            df = seq_len - st["fls"][r]
            fl0 = st["fls"][r]
            if df < 0:
                # Row reused for a SHORTER prompt than its last gather (slot reset /
                # mid-prefill preemption): re-pull the full layer from 0, mirroring the
                # SWA backward guard above. Without this, view(df<0, ...) crashes.
                fl0 = 0
                df = seq_len
            ds = swa_len - swa_start
            specs.append((r, fl0, seq_len, swa_start, swa_len))
            meta.append((seq_len, swa_len, swa_start, df, ds))

        for device, w in sorted(self.workers.items()):
            w.send("kv-pack-batch", (lf, ls, specs))

        parts = []
        for (r, fl0, fl1, sl0, sl1) in specs:
            parts.append([
                [mc.key_states[lf][r, fl0:fl1].detach().cpu()],
                [mc.value_states[lf][r, fl0:fl1].detach().cpu()],
                [mc.key_states[ls][r, sl0:sl1].detach().cpu()],
                [mc.value_states[ls][r, sl0:sl1].detach().cpu()],
            ])
        for device, w in sorted(self.workers.items()):
            flat = w.recv()
            o = 0
            fkh, skh = fw.get(device, 0), sw.get(device, 0)
            for bi in range(len(specs)):
                df, ds = meta[bi][3], meta[bi][4]
                for rows_n, kh, hd, acc in ((df, fkh, d.global_head_dim, parts[bi][0]),
                                            (df, fkh, d.global_head_dim, parts[bi][1]),
                                            (ds, skh, d.head_dim, parts[bi][2]),
                                            (ds, skh, d.head_dim, parts[bi][3])):
                    n = rows_n * kh * hd
                    acc.append(flat[o:o + n].view(rows_n, kh, hd))
                    o += n
        for bi, r in enumerate(rows):
            seq_len, swa_len, swa_start, df, ds = meta[bi]
            fl0 = specs[bi][1]
            if df > 0:
                st["kf"][r, fl0:seq_len].copy_(torch.cat(parts[bi][0], dim = 1))
                st["vf"][r, fl0:seq_len].copy_(torch.cat(parts[bi][1], dim = 1))
            st["fls"][r] = seq_len
            if ds > 0:
                st["ks"][r, swa_start:swa_len].copy_(torch.cat(parts[bi][2], dim = 1))
                st["vs"][r, swa_start:swa_len].copy_(torch.cat(parts[bi][3], dim = 1))
            st["sls"][r] = seq_len
            st["wbs"][r] = mc.window_begs[r]
        return st

    def forward(self, input_ids, past_len, seq_len, batch_size, last_id_only,
                graph_verify = False, return_row_states = False, return_row_aux = False):
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
                               graph_verify, return_row_states, return_row_aux))

        # Master's own share, overlapped with the workers.
        logits = _worker_forward(self.master_ctx, input_ids, past_len, seq_len,
                                 batch_size, last_id_only, graph_verify, return_row_states,
                                 return_row_aux)

        # Collect the workers' replies so their errors surface here rather than as a hang.
        worker_out = [w.recv() for _, w in sorted(self.workers.items())]

        if return_row_aux and logits is not None:
            # DFlash verify: (logits, aux) with the ranks' shards joined when the
            # step ran seqpar-sharded (decode_repl off); normally full-width from
            # the master alone. aux is the verify graph's STATIC buffer, already
            # copied to CPU by the worker wrapper -- consume before next replay.
            lg, aux0 = logits
            aux = _join_aux(aux0, worker_out, len(self.model._dflash_aux_taps),
                            self.config.hidden_size)
            return lg, aux

        return logits

    # ------------------------------------------------------------------ multi-batch

    def row_view(self, row):
        """Persistent row-scoped cache facade (eager prefill / solo streaming runs)."""
        return self.cache.row_view(row)

    def forward_batch(self, input_ids, seq_lens, row_map, last_id_only = True,
                      graph_verify = False, return_row_states = False):
        """One graphed multi-batch step. ``seq_lens``/``row_map`` are per-BATCH-ROW
        (length B = the round's active subset, which may be smaller than max_batch);
        ``row_map[b]`` names batch row b's cache row. Every rank adopts the values into
        those cache rows only -- other slots' bookkeeping is untouched. The ring prep
        inside forward_tp_batch persists its shifts into cache.window_begs[r] on every
        rank (deterministic, no extra traffic). Returns the master's logits (and
        row_states when requested); workers return None."""
        seq_lens = list(seq_lens)
        row_map = list(row_map)
        wb_rows = [self.cache.window_begs[r] for r in row_map]
        for b, r in enumerate(row_map):
            self.cache.seq_lens[r] = seq_lens[b]
        for device, w in sorted(self.workers.items()):
            w.send("forward-batch", (input_ids, seq_lens, list(wb_rows), row_map,
                                     last_id_only, graph_verify, return_row_states))
        out = _worker_forward_batch(self.master_ctx, input_ids, seq_lens,
                                    wb_rows, row_map,
                                    last_id_only, graph_verify, return_row_states)
        for device, w in sorted(self.workers.items()):
            w.recv()
        return out

    def forward_row(self, row, input_ids, last_id_only = True, preprocess_only = False):
        """Eager prefill chunk into one cache row (the scheduler drives the chunk loop
        so decode rounds can interleave between chunks). Returns the master's logits
        (None when preprocess_only)."""
        for device, w in sorted(self.workers.items()):
            w.send("forward-row", (row, input_ids, last_id_only, preprocess_only))
        out = _worker_forward_row(self.master_ctx, row, input_ids, last_id_only,
                                  preprocess_only)
        for device, w in sorted(self.workers.items()):
            w.recv()
        return out

    def forward_state_row(self, row, input_ids, last_id_only = True):
        """Solo decode-shaped forward on one row returning (logits, last_state) — the
        MTP heal / recommit path. Eager on every rank (no row-view graph captures)."""
        for device, w in sorted(self.workers.items()):
            w.send("forward-state-row", (row, input_ids, last_id_only))
        out = _worker_forward_state_row(self.master_ctx, row, input_ids, last_id_only)
        for device, w in sorted(self.workers.items()):
            w.recv()
        return out

    def mtp_set_seq_lens(self, seq_lens):
        """Distributed per-row rewind: replace every rank's seq_lens list at once."""
        seq_lens = list(seq_lens)
        for device, w in sorted(self.workers.items()):
            w.send("set-seqlens", (seq_lens,))
            w.recv()
        self.cache.seq_lens[:] = seq_lens

    def mtp_set_seq_len_row(self, row, n):
        for device, w in sorted(self.workers.items()):
            w.send("set-seqlen-row", (row, n))
            w.recv()
        self.cache.seq_lens[row] = n

    def mtp_reset_row(self, row):
        """Full reset of one slot (new lineage): position + ring origin + shadow row.
        Other slots are untouched; a stale disconnect abort must not leak across jobs."""
        for device, w in sorted(self.workers.items()):
            w.send("reset-row", (row,))
            w.recv()
        self.cache.seq_lens[row] = 0
        self.cache.window_begs[row] = 0
        self._reset_mtp_kv_row(row)
        self.abort_flag.clear()

    def clone_row(self, src, dst, p):
        """Shared-prefix prompt cache: copy slot ``src``'s committed K/V state at
        position ``p`` into slot ``dst`` on EVERY rank (full layers [0, p), SWA ring
        prefix at src's window origin), then clone the drafter's joined shadow row.
        A physical copy -- slots never share live K/V afterwards, so the isolation
        invariant holds. The caller (slot scheduler) runs between rounds, so src's
        [0, p) prefix is committed and stable.

        The shadow lags the cache by the last round's accepted drafts (gather runs
        at the pre-verify frontier): clone only up to src's gather frontier and let
        the next gather_kv_mtp_batch pull the small delta through the normal
        incremental protocol -- no full-length re-pull across the pipes."""
        mc = self.cache
        assert 0 < p <= mc.seq_lens[src], f"clone_row: bad position {p}"
        wb = mc.window_begs[src]
        assert p >= wb, f"clone_row: prefix {p} below src ring origin {wb}"
        for device, w in sorted(self.workers.items()):
            w.send("clone-row", (src, dst, p))
        with torch.inference_mode():
            for l in range(len(mc.key_states)):
                kc, vc = mc.key_states[l], mc.value_states[l]
                if kc is None:
                    continue
                if kc.shape[1] == mc.kv_state_size:
                    n = p - wb
                    kc[dst, :n].copy_(kc[src, :n])
                    vc[dst, :n].copy_(vc[src, :n])
                else:
                    kc[dst, :p].copy_(kc[src, :p])
                    vc[dst, :p].copy_(vc[src, :p])
        mc.seq_lens[dst] = p
        mc.window_begs[dst] = wb
        st = self._mtp_kv
        if st is not None and "fls" in st:
            # Re-zero dst's shadow first (its tail may hold a stale lineage; the
            # zero-init contract keeps unwritten rows finite), then copy the source's
            # valid prefix up to its gather frontier.
            self._reset_mtp_kv_row(dst)
            f = min(p, st["fls"][src])
            swb = st["wbs"][src]
            n = max(0, min(f - swb, st["ks"].shape[1]))
            with torch.inference_mode():
                st["kf"][dst, :f].copy_(st["kf"][src, :f])
                st["vf"][dst, :f].copy_(st["vf"][src, :f])
                if n > 0:
                    st["ks"][dst, :n].copy_(st["ks"][src, :n])
                    st["vs"][dst, :n].copy_(st["vs"][src, :n])
            st["fls"][dst] = f
            st["sls"][dst] = f
            st["wbs"][dst] = swb
        for device, w in sorted(self.workers.items()):
            w.recv()

    def truncate_row(self, row, p):
        """Partial prefix reuse: rewind slot ``row`` to position ``p`` on every rank,
        keeping its K/V (full layers ignore rows >= p via seq_len; the SWA ring's valid
        fill shrinks to p - window_beg). The drafter shadow's gather frontiers truncate
        the same way -- its gathered prefix [0, min(fls, p)) stays valid, so the next
        gather pulls only the suffix delta. No copies, no ring shift; p must sit inside
        the live ring (>= window_beg)."""
        mc = self.cache
        assert mc.window_begs[row] <= p <= mc.seq_lens[row], \
            f"truncate_row: bad position {p}"
        for device, w in sorted(self.workers.items()):
            w.send("truncate-row", (row, p))
        mc.seq_lens[row] = p
        st = self._mtp_kv
        if st is not None and "fls" in st:
            st["fls"][row] = min(st["fls"][row], p)
            st["sls"][row] = min(st["sls"][row], p)
        for device, w in sorted(self.workers.items()):
            w.recv()

    def truncate_cache(self, p):
        """Scalar (batch-1) partial prefix reuse: rewind to ``p`` on every rank,
        keeping the K/V. The scalar current_seq_len / window_beg fields are the
        batch-1 path's authoritative bookkeeping (the per-row lists are not
        maintained by the scalar forwards), so this moves those -- not the rows.
        p must sit inside the live ring (>= window_beg); the ring is untouched
        (its valid fill shrinks to p - window_beg). The scalar shadow frontiers
        truncate so the next gather pulls only the suffix delta."""
        mc = self.cache
        assert mc.window_beg <= p <= mc.current_seq_len, \
            f"truncate_cache: bad position {p}"
        for device, w in sorted(self.workers.items()):
            w.send("set-seqlen", (p,))
            w.recv()
        mc.current_seq_len = p
        st = self._mtp_kv
        if st is not None and "fls" not in st:
            st["fl"] = min(st["fl"], p)
            st["sl"] = min(st["sl"], p)

    def snapshot_prompt_ring_row(self, row):
        """Per-slot prompt-boundary ring snapshot (pinned CPU on every rank)."""
        seq = self.cache.seq_lens[row]
        for device, w in sorted(self.workers.items()):
            w.send("ring-snapshot-row", (row,))
        backups = getattr(self, "_ring_backup_rows", None)
        if backups is None:
            backups = self._ring_backup_rows = {}
            self._ring_backup_rows_wb = {}
            self._ring_backup_rows_seq = {}
        backups[row] = _clone_swa_ring(self.row_view(row))
        self._ring_backup_rows_wb[row] = self.cache.window_begs[row]
        for device, w in sorted(self.workers.items()):
            w.recv()
        self._ring_backup_rows_seq[row] = seq

    def restore_prompt_ring_row(self, row, seq):
        """Restore slot ``row`` to its snapshot position (exact ring copy-back).
        Returns False if no snapshot exists for exactly this position."""
        backups = getattr(self, "_ring_backup_rows", None)
        if backups is None or row not in backups or self._ring_backup_rows_seq[row] != seq:
            return False
        for device, w in sorted(self.workers.items()):
            w.send("ring-restore-row", (row, seq))
        _restore_swa_ring(self.row_view(row), backups[row], seq,
                          self._ring_backup_rows_wb[row])
        for device, w in sorted(self.workers.items()):
            w.recv()
        self._reset_mtp_kv_row(row)
        return True

    # ------------------------------------------------------------------ prompt-cache RAM offload

    def offload_row(self, row, pos = None, wb = None, entry_id = None):
        """Offload ONE idle row's committed K/V state to host RAM (opt-in prompt
        cache tiering). Returns a bundle dict the scheduler stores; the KV slices
        stay rank-local (workers keep theirs in ctx["cold_bundles"], the master's
        flat copy is in the bundle), so nothing large crosses the pipe. The MTP
        drafter's joined shadow row and the prompt-boundary ring snapshot travel
        with the lineage. Returns None when host allocation fails anywhere (the
        caller skips the offload; any partial worker storage is cleaned up).

        pos/wb default to the row's batch bookkeeping; the scalar batch-1 path
        passes cache.current_seq_len / window_beg explicitly (it owns those).
        NEVER allocates device memory -- VRAM cannot OOM from this path."""
        mc = self.cache
        if pos is None:
            pos = mc.seq_lens[row]
        if wb is None:
            wb = mc.window_begs[row]
        assert 0 < pos <= mc.max_seq_len and 0 <= wb <= pos, \
            f"offload_row: bad position {pos} / window origin {wb}"
        if entry_id is None:
            entry_id = uuid.uuid4().hex
        for device, w in sorted(self.workers.items()):
            w.send("offload-row", (row, pos, wb, entry_id))
        try:
            flat, meta = _pack_row_offload(self.master_ctx, row, pos, wb)
            master_flat = flat.clone()
        except Exception:
            master_flat = None
        metas = {}
        nbytes = 0
        snap_bytes = 0
        failed = master_flat is None
        for device, w in sorted(self.workers.items()):
            r = w.recv()
            if r is None:
                failed = True
                continue
            wmeta, wn, wsn = r
            metas[device] = wmeta
            nbytes += wn
            snap_bytes += wsn
        if failed:
            self.drop_offload_bundle({"entry_id": entry_id})
            return None
        nbytes += master_flat.numel() * master_flat.element_size()

        # Master's own ring snapshot (kept on the pg object, not master_ctx).
        local_snap = None
        local_snap_bytes = 0
        backups = getattr(self, "_ring_backup_rows", None)
        if backups is not None and row in backups:
            backup = backups.pop(row)
            local_snap = (backup, self._ring_backup_rows_wb.pop(row),
                          self._ring_backup_rows_seq.pop(row))
        elif self.max_batch == 1 and self._ring_backup_local is not None:
            local_snap = (self._ring_backup_local, self._ring_backup_wb,
                          self._ring_backup_seq)
            self._ring_backup_local = None
            self._ring_backup_seq = -1
        if local_snap is not None:
            local_snap_bytes = sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for k, v in local_snap[0].values())

        shadow = self._save_shadow_row(row)
        shadow_bytes = 0
        if shadow is not None:
            shadow_bytes = sum(t.numel() * t.element_size()
                               for t in shadow.values() if torch.is_tensor(t))
        return {
            "entry_id": entry_id,
            "ranks": metas,
            "master_flat": master_flat,
            "master_meta": meta,
            "pos": pos,
            "wb": wb,
            "local_snap": local_snap,
            "shadow": shadow,
            "bytes": nbytes + snap_bytes + local_snap_bytes + shadow_bytes,
        }

    def restore_row(self, row, bundle, p = None, attach_snapshot = False):
        """Restore an offloaded lineage into ONE idle row at position p (default:
        the bundle's full committed position). p must sit inside the bundle's ring
        (>= bundle wb). Every rank H2Ds its own stored slice; the master restores
        its slice + the MTP shadow row; attach_snapshot=True also moves the
        prompt-boundary ring snapshot back to the row (full-lineage adoption --
        partial prefixes keep it in cold storage). Copies target preallocated
        buffers only: no device allocation, no graph interaction (runs between
        rounds, like clone_row). Raises on worker failure -- the caller drops the
        bundle and falls back to a full re-prefill."""
        if p is None:
            p = bundle["pos"]
        wb = bundle["wb"]
        assert wb <= p <= bundle["pos"], \
            f"restore_row: bad position {p} (wb {wb}, pos {bundle['pos']})"
        eid = bundle["entry_id"]
        for device, w in sorted(self.workers.items()):
            w.send("restore-row", (row, eid, p, wb, attach_snapshot))
        _restore_row_offload(self.cache, row, bundle["master_flat"],
                             bundle["master_meta"], p, wb)
        if attach_snapshot and bundle.get("local_snap") is not None:
            backup, swb, sseq = bundle["local_snap"]
            if self.max_batch > 1:
                self._ring_backup_rows[row] = backup
                self._ring_backup_rows_wb[row] = swb
                if sseq is not None:
                    self._ring_backup_rows_seq[row] = sseq
            else:
                self._ring_backup_local = backup
                self._ring_backup_wb = swb
                self._ring_backup_seq = sseq if sseq is not None else -1
            bundle["local_snap"] = None   # ownership moved to the row
        self._restore_shadow_row(row, bundle.get("shadow"), p)
        for device, w in sorted(self.workers.items()):
            w.recv()
        return True

    def drop_offload_bundle(self, bundle):
        """Free an offloaded lineage: worker-local KV storage + detached ring
        snapshots on every rank, then the parent's own tensors (GC). Idempotent;
        safe on partially-created bundles (offload failure cleanup)."""
        eid = bundle.get("entry_id")
        if eid is not None:
            for device, w in sorted(self.workers.items()):
                try:
                    w.send("drop-cold-bundle", (eid,))
                except Exception:
                    pass
            for device, w in sorted(self.workers.items()):
                try:
                    w.recv()
                except Exception:
                    pass
        bundle["master_flat"] = None
        bundle["local_snap"] = None
        bundle["shadow"] = None

    def _save_shadow_row(self, row):
        """D2H the MTP drafter's joined shadow prefix for one row (the gathered
        K/V of its two source layers) + the gather frontiers. Restoring these lets
        the next gather pull only the small post-restore delta instead of a full
        context re-pull through the pipes. None when there is no shadow (DFlash)."""
        st = self._mtp_kv
        if st is None:
            return None
        ring = self.cache.kv_state_size
        batched = "fls" in st
        if batched:
            fl, sl, wb = st["fls"][row], st["sls"][row], st["wbs"][row]
            n = max(0, min(sl - wb, ring))
            out = {
                "kf": st["kf"][row, :fl].detach().cpu(),
                "vf": st["vf"][row, :fl].detach().cpu(),
                "ks": st["ks"][row, :n].detach().cpu(),
                "vs": st["vs"][row, :n].detach().cpu(),
            }
        else:
            fl, sl, wb = st["fl"], st["sl"], st["wb"]
            n = max(0, min(sl - wb, ring))
            out = {
                "kf": st["kf"][:fl].detach().cpu(),
                "vf": st["vf"][:fl].detach().cpu(),
                "ks": st["ks"][:n].detach().cpu(),
                "vs": st["vs"][:n].detach().cpu(),
            }
        out.update(fl = fl, sl = sl, wb = wb, batched = batched)
        return out

    def _restore_shadow_row(self, row, shadow, p):
        """H2D a saved shadow prefix back, truncated to position p (the clone_row
        frontier contract: gather frontiers land at min(p, fl), the next gather
        pulls the delta). Re-zeros the row first -- the zero-init contract keeps
        unwritten tail rows finite for the drafter's full-depth attention reads."""
        st = self._mtp_kv
        if st is None or shadow is None:
            return
        ring = self.cache.kv_state_size
        f = min(p, shadow["fl"])
        n = max(0, min(f - shadow["wb"], ring))
        if shadow["batched"]:
            self._reset_mtp_kv_row(row)
            with torch.inference_mode():
                if f > 0:
                    st["kf"][row, :f].copy_(shadow["kf"][:f])
                    st["vf"][row, :f].copy_(shadow["vf"][:f])
                if n > 0:
                    st["ks"][row, :n].copy_(shadow["ks"][:n])
                    st["vs"][row, :n].copy_(shadow["vs"][:n])
            st["fls"][row] = f
            st["sls"][row] = f
            st["wbs"][row] = shadow["wb"]
        else:
            self._reset_mtp_kv()
            with torch.inference_mode():
                if f > 0:
                    st["kf"][:f].copy_(shadow["kf"][:f])
                    st["vf"][:f].copy_(shadow["vf"][:f])
                if n > 0:
                    st["ks"][:n].copy_(shadow["ks"][:n])
                    st["vs"][:n].copy_(shadow["vs"][:n])
            st["fl"] = f
            st["sl"] = f
            st["wb"] = shadow["wb"]

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
