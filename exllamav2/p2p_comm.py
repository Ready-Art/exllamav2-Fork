"""
Low-latency PCIe P2P collectives for the MP decode path (EXL2_TP_P2P, default on).

NCCL small-message latency (~70 us/op over PXB, no NVLink) dominates base decode:
~240 collectives/token after decode-repl. This module replaces the SMALL ones
(<= 1 MB: decode allgathers, post-norm stat all-reduces, embedding all-reduce)
with one-sided P2P writes into IPC-shared peer arenas + system-scope flag spins
(~10-20 us). Large prefill collectives stay on NCCL (bandwidth-bound, NCCL wins).

NOT a hard requirement: every step is probed and agreed across ranks; any failure
(no P2P capability, IPC blocked, probe mismatch) disables the path and everything
falls back to NCCL. EXL2_TP_P2P=0 forces NCCL. Single-node only by construction
(cudaDeviceCanAccessPeer fails across hosts).

Graph safety: all protocol state (peer pointers, flags, round counter) lives in
the fixed arena; the round counter is device-incremented, so capture/replay needs
no host state. Per-call tensor pointers are graph-pool-stable, exactly like the
captured NCCL ops they replace. All ranks execute the same collective sequence in
lockstep on one stream (the TP forward contract).
"""

import os
import torch


class P2PComm:
    SLOT_BYTES = 1 << 20
    # 32 double-buffered slots (32 MB arena). The model uses ~14; headroom avoids
    # silent slot-exhaustion fallback (which would drop a collective to NCCL mid-graph).
    # 64 slots: eager (~12) + each captured graph gets its OWN permanent slot region
    # (~12/graph x decode/verify/bonus + shape variants). Per-graph isolation is what
    # makes in-graph P2P safe -- shared keys across persistent graphs otherwise bake
    # overlapping slots and clobber each other on replay (the gibberish bug).
    NSLOTS = 64

    def __init__(self, rank: int, world: int, n_slots: int = 64):
        self.rank = rank
        self.world = world
        # Multi-batch captures more persistent graphs (decode/verify per batch size),
        # each owning a permanent slot region; the caller raises this above the default
        # 64 when max_batch > 1 so later graphs don't exhaust the arena and silently
        # fall back to NCCL collectives mid-capture.
        self.NSLOTS = int(n_slots)
        self.enabled = False
        self.arena = None
        self.peer_ptrs = None
        self._slots: dict = {}
        self._alt: dict = {}
        self._next_slot = 0
        # Slot epoch: 0 = eager; each graph capture bumps to a fresh permanent region.
        self._epoch = 0

        if os.environ.get("EXL2_TP_P2P", "1") == "0":
            return
        if world < 2:
            return
        import torch.distributed as dist
        from exllamav2.ext import ext_c
        if not (dist.is_available() and dist.is_initialized()):
            return

        def _agree(ok: bool, why: str) -> bool:
            # Group-wide MIN agreement over NCCL: either EVERY rank enables P2P or
            # NONE does. Asymmetric enablement would deadlock the flag protocol.
            t = torch.tensor([1 if ok else 0], dtype = torch.int32, device = rank)
            dist.all_reduce(t, op = dist.ReduceOp.MIN)
            if t.item() != 1:
                if not ok:
                    print(f"[p2p] rank {rank}: {why}, NCCL fallback", flush = True)
                return False
            return True

        # 1) Capability check + agreement BEFORE any P2P kernel runs.
        try:
            local_ok = ext_c.p2p_enable_peer_access(rank, world)
        except Exception as e:
            local_ok = False
            print(f"[p2p] rank {rank}: peer access check failed ({e})", flush = True)
        if not _agree(local_ok, "peer access unavailable"):
            return

        # 2) Arena + IPC handle exchange. Handles are HEX strings: the TCPStore
        # utf-8-decodes values, so raw binary handles raise UnicodeDecodeError.
        setup_ok = True
        try:
            self.ar_off = self.NSLOTS * self.SLOT_BYTES
            # Per-key double-buffered AR scratch (2 regions x world x SLOT_BYTES per key),
            # mirroring the allgather _slot contract: each AR call site (key) owns a pair
            # of regions and alternates between them, so a peer's round-r+1 contribution
            # cannot clobber a still-unread round-r region of the SAME key -> the per-op
            # ack spin can be deferred inside graphs (graph-start barrier covers the
            # cross-replay hazard). A GLOBAL counter would alias distinct keys onto the
            # same region across capture warmup vs capture (the ROW_REPL parity bug).
            self.ar_region_bytes = world * self.SLOT_BYTES
            self._ar_nkeys = 8
            self.flags_off = self.ar_off + self._ar_nkeys * 2 * self.ar_region_bytes
            self.acks_off = self.flags_off + world * 8
            self.counter_off = self.acks_off + world * 8
            self.ptrs_off = self.counter_off + 8
            total = self.ptrs_off + world * 8
            self._ar_slots = {}
            self._ar_alt = {}
            self._ar_next = 0
            self.arena = torch.empty(total, dtype = torch.uint8, device = rank)
            self.arena.zero_()
            torch.cuda.synchronize(rank)

            handle = ext_c.p2p_ipc_handle(self.arena)
            store = dist.distributed_c10d._get_default_store()
            store.set(f"exl2_p2p_h_{rank}", handle.hex())
            dist.barrier()
            peers = []
            for p in range(world):
                if p == rank:
                    peers.append(self.arena.data_ptr())
                else:
                    v = store.get(f"exl2_p2p_h_{p}")
                    if isinstance(v, bytes):
                        v = v.decode("ascii")
                    peers.append(ext_c.p2p_ipc_open(bytes.fromhex(v)))
            self.peer_ptrs = torch.tensor(peers, dtype = torch.int64, device = rank)
            self.peer_ptrs_dev = self.peer_ptrs.to(rank)
        except Exception as e:
            setup_ok = False
            print(f"[p2p] rank {rank}: arena/IPC setup failed ({type(e).__name__}: {e})",
                  flush = True)
        # Agree BEFORE the probe: a rank with no arena must not be waited on by
        # the others' probe kernels (flag-spin deadlock).
        if not _agree(setup_ok, "arena/IPC setup failed"):
            return

        # 3) Correctness probe (eager, outside any graph), then final agreement.
        try:
            probe_ok = self._probe()
        except Exception as e:
            probe_ok = False
            print(f"[p2p] rank {rank}: probe failed ({type(e).__name__}: {e})", flush = True)
        if not _agree(probe_ok, "probe mismatch"):
            return

        self.enabled = True
        print(f"[p2p] rank {rank}: PCIe P2P collectives ENABLED "
              f"({self.NSLOTS} x 1 MB slots, world {world})", flush = True)

    def begin_graph(self, epoch: int | None = None) -> int:
        """Start a permanent slot region for a graph capture (warmup + capture
        share it). Slots are NEVER reused across epochs, so persistent graphs can't
        clobber each other's baked arena addresses on replay.

        A caller recapturing the SAME graph shape passes that shape's stable epoch:
        the released graph no longer references the region, and the recapture's
        collective keys map to the same slots, so the region is safely reused.
        Without this every recapture leaks a fresh region and a multi-batch server
        (envelope recaptures at job boundaries) exhausts the arena -- _slot returns
        None and collectives silently drop to NCCL mid-capture."""
        if epoch is None:
            self._epoch += 1
        else:
            self._epoch = int(epoch)
        return self._epoch

    def end_graph(self):
        self._epoch = 0

    def _slot(self, key) -> int | None:
        # Double-buffered pairs, mirroring the NCCL _ag_cache aliasing contract:
        # a caller may still be reading the previous result of the SAME key when
        # the next gather runs. Alternate between the pair's two slots.
        # Keyed by (epoch, key): each captured graph owns a distinct slot region.
        k = (self._epoch, key)
        s = self._slots.get(k)
        if s is None:
            if self._next_slot + 2 > self.NSLOTS:
                return None
            s = self._next_slot
            self._next_slot += 2
            self._slots[k] = s
            self._alt[k] = 0
        self._alt[k] ^= 1
        return s + self._alt[k]

    def _probe(self) -> bool:
        # Uses the UNGATED raw paths: self.enabled is still False here (it is set
        # only after this probe passes) -- the public wrappers would short-circuit.
        n = 16
        shard = torch.full((n,), float(self.rank + 1), dtype = torch.half, device = self.rank)
        out = self._ag_raw(shard, ("__probe_ag__",), (n * self.world,))
        if out is None:
            print(f"[p2p] rank {self.rank}: probe AG slot unavailable", flush = True)
            return False
        torch.cuda.synchronize(self.rank)
        expect = torch.cat([torch.full((n,), float(p + 1), dtype = torch.half,
                                       device = self.rank) for p in range(self.world)])
        if not torch.equal(out, expect):
            print(f"[p2p] rank {self.rank}: probe AG mismatch: got {out[:8].tolist()} "
                  f"expect {expect[:8].tolist()}", flush = True)
            return False
        t = torch.ones(n, dtype = torch.float32, device = self.rank)
        if not self._ar_raw(t):
            print(f"[p2p] rank {self.rank}: probe AR refused", flush = True)
            return False
        torch.cuda.synchronize(self.rank)
        ok = bool(torch.equal(t, torch.full((n,), float(self.world),
                                            dtype = torch.float32, device = self.rank)))
        if not ok:
            print(f"[p2p] rank {self.rank}: probe AR mismatch: got {t[:4].tolist()} "
                  f"expect {float(self.world)}", flush = True)
        return ok

    def barrier(self):
        """Graph-start rendezvous of the deferred-ack protocol: wait until every peer
        has acked the round counter's current value (the previous replay's last
        collective) before this replay's first one-sided write. Captured as the first
        P2P node of every graph and run before each warmup iteration; reads the device
        counter at replay time, so one captured node serves every replay."""
        if not self.enabled:
            return
        from exllamav2.ext import exllamav2_ext as ext_c
        ext_c.p2p_barrier(self.arena.data_ptr(), self.peer_ptrs_dev.data_ptr(),
                          self.acks_off, self.counter_off, self.world, self.rank)

    def allgather(self, shard: torch.Tensor, key, full_shape):
        """
        Equal-split allgather of ``shard`` (this rank's contiguous slice) into a
        full-width tensor. Returns a view of the arena slot (consumed before the
        next collective on the same stream, same contract as the NCCL buffers),
        or None if the shape does not fit a slot (caller falls back to NCCL).
        """
        if not self.enabled:
            return None
        return self._ag_raw(shard, key, full_shape)

    def _ag_raw(self, shard: torch.Tensor, key, full_shape):
        from exllamav2.ext import ext_c
        shard_bytes = shard.numel() * shard.element_size()
        full_numel = 1
        for d in full_shape:
            full_numel *= d
        if shard_bytes * self.world > self.SLOT_BYTES or shard_bytes == 0:
            return None
        slot = self._slot(key)
        if slot is None:
            return None
        if not shard.is_contiguous():
            shard = shard.contiguous()
        # NCCL all_gather(dim=1) layout: result row i interleaves the ranks' row i
        # ([rank0_row_i | rank1_row_i | ...]). row_bytes is one row of this rank's
        # (rows, width) shard; the kernel scatters it at (i*world + rank)*row_bytes.
        # A 1-D shard (the probe) is a single row. rows == 1 reduces to the old
        # contiguous rank-block write, so decode is byte-identical; multi-row
        # verify windows were row-scrambled by the rank-block layout (gibberish).
        rows = full_shape[0] if len(full_shape) > 1 else 1
        if rows < 1 or shard_bytes % rows != 0:
            return None
        row_bytes = shard_bytes // rows
        # Eager ops (epoch 0) keep the full per-op ack rendezvous; graphed ops skip
        # the ack spin -- the graph-start barrier() covers the cross-replay hazard
        # (see the deferred-ack comment in p2p.cu).
        ext_c.p2p_allgather(
            self.arena.data_ptr(), self.peer_ptrs_dev.data_ptr(), shard.data_ptr(),
            slot * self.SLOT_BYTES, shard_bytes, row_bytes, self.world, self.rank,
            self.flags_off, self.acks_off, self.counter_off,
            1 if self._epoch == 0 else 0)
        off = slot * self.SLOT_BYTES
        nbytes = full_numel * shard.element_size()
        return self.arena[off:off + nbytes].view(shard.dtype).reshape(full_shape)

    def ag_norm_res(self, shard: torch.Tensor, key, full_shape, weight, residual, out, eps):
        """
        Fused allgather + RMSNorm + residual add for the decode_repl block tail. Gathers
        this rank's RS output shard to full width (same arena slot/protocol as allgather),
        then out = RMSNorm(gathered, weight, eps) + residual in ONE kernel. ``out`` is a
        caller-owned graph-stable (rows, hidden) fp16 buffer. Returns ``out`` or None if
        the shape/dtype does not fit (caller falls back to the unfused AG+norm+add).
        """
        if not self.enabled:
            return None
        from exllamav2.ext import exllamav2_ext as ext_c
        if shard.dtype != torch.half or weight.dtype != torch.half \
                or residual.dtype != torch.half or out.dtype != torch.half:
            return None
        shard_bytes = shard.numel() * shard.element_size()
        full_numel = 1
        for d in full_shape:
            full_numel *= d
        if shard_bytes * self.world > self.SLOT_BYTES or shard_bytes == 0:
            return None
        slot = self._slot(key)
        if slot is None:
            return None
        if not shard.is_contiguous():
            shard = shard.contiguous()
        rows = full_shape[0] if len(full_shape) > 1 else 1
        if rows < 1 or shard_bytes % rows != 0:
            return None
        row_bytes = shard_bytes // rows
        hidden = full_shape[-1]
        if not weight.is_contiguous():
            weight = weight.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        ext_c.p2p_ag_norm_res(
            self.arena.data_ptr(), self.peer_ptrs_dev.data_ptr(), shard.data_ptr(),
            slot * self.SLOT_BYTES, shard_bytes, row_bytes, self.world, self.rank,
            self.flags_off, self.acks_off, self.counter_off,
            1 if self._epoch == 0 else 0,
            weight, residual, out, rows, hidden, eps)
        return out

    def allreduce(self, t: torch.Tensor, key="__ar__") -> bool:
        """
        In-place SUM all-reduce of a small half/float32 tensor. Returns False if
        the tensor is too large or unsupported (caller falls back to NCCL). ``key``
        identifies the call site for the per-key double-buffered scratch (graph-safe).
        """
        if not self.enabled:
            return False
        return self._ar_raw(t, key)

    def _ar_raw(self, t: torch.Tensor, key="__ar__") -> bool:
        from exllamav2.ext import ext_c
        if t.dtype == torch.half:
            dtype = 0
        elif t.dtype == torch.float32:
            dtype = 1
        else:
            return False
        elems = t.numel()
        if elems * t.element_size() > self.SLOT_BYTES or elems == 0:
            return False
        if not t.is_contiguous():
            return False
        # Per-key double-buffered region (epoch-scoped like the AG slots): each captured
        # graph owns distinct regions; alternate the pair per call for deferred-ack safety.
        k = (self._epoch, key)
        s = self._ar_slots.get(k)
        if s is None:
            if self._ar_next >= self._ar_nkeys:
                return False
            s = self._ar_next
            self._ar_next += 1
            self._ar_slots[k] = s
            self._ar_alt[k] = 0
        self._ar_alt[k] ^= 1
        scratch_off = self.ar_off + (s * 2 + self._ar_alt[k]) * self.ar_region_bytes
        ext_c.p2p_allreduce(
            self.arena.data_ptr(), self.peer_ptrs_dev.data_ptr(), t.data_ptr(),
            scratch_off, self.SLOT_BYTES // t.element_size(), elems,
            self.world, self.rank, self.flags_off, self.acks_off, self.counter_off, dtype,
            1 if self._epoch == 0 else 0)
        return True