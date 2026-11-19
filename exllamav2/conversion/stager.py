"""
Pinned-memory weight staging for the EXL2 quantizer.

Port of EXL3's ``_WeightStager`` (exllamav3/modules/quant/exl3_lib/quantize.py:1944) and its
double-buffered pinned arena (``_get_stage_buf``, line 1930), adapted to EXL2's direction of travel.
See quant-plan.md §6.8 (M5).

Why this exists
---------------
EXL2's ``AdaptiveGPTQ.prepare()`` currently does

    self.weights = self.layer.weight.data.cpu().T.clone().float().contiguous()

at a point where ``self.layer.weight.data`` is already resident in VRAM (``module.load()`` has run). So
this is a **device-to-host** copy, and it is a *synchronous, pageable* one: the D2H copy cannot be issued
as DMA, so the driver has to bounce it through an internal staging buffer, and the caller blocks for the whole
transfer before it can do anything else with the tensor.

Two things are wrong with that:

  * **No overlap.** The transfer of tensor ``t`` is on the critical path. Nothing else happens during it.
  * **Pageable, not pinned.** Pageable D2H runs at roughly half the bandwidth of pinned D2H on most
    chipsets, because the driver cannot DMA straight into the destination pages.

The stager fixes both. It copies into a *pinned* (page-locked) host buffer, so the copy is real DMA, and it
issues that copy one tensor ahead on a side stream, so the transfer of tensor ``t+1`` overlaps the
regularization and quantization of tensor ``t``.

Note the direction: EXL3 stages **host → device** (its checkpoints are read from disk and cast on the device).
EXL2 stages **device → host** (its modules are loaded to VRAM first). The mechanism is the same; only the
stream direction and the final cast differ. The fp32 cast is done on the host, after the transfer, exactly as the
old code did it, so the resulting tensor is **bit-identical** to the old path. M5 is a pure wall-clock change.

This is a single-GPU win as well as a multi-GPU one. It is worth having even if the box has one card.
"""

from __future__ import annotations

import threading

import torch


_stage_bufs = {}
_stage_bufs_lock = threading.Lock()


def _get_stage_buf(owner: int, slot: int, nbytes: int):
    """
    Double-buffered pinned arena, one buffer per (owner, slot). The buffer is grown in place when a larger
    tensor arrives; it is never shrunk, because the largest tensor in a model is usually seen early (attention/MLP
    of the first layer) and reallocating pinned memory is expensive.

    ``owner`` must identify the *thread* using the buffer. Two worker threads sharing a pinned buffer would
    overwrite each other's staging data between the host memcpy and the async D2H copy, which silently corrupts the
    weights of whichever tensor loses the race. Keying on the thread ident is what makes the stager safe to share
    across the multi-GPU worker pool. quant-plan.md §6.4 / §6.8.
    """
    key = (owner, slot)
    with _stage_bufs_lock:
        buf = _stage_bufs.get(key)
        if buf is None or buf[0].numel() < nbytes:
            buf = (torch.empty(nbytes, dtype = torch.uint8, pin_memory = True),
                   torch.cuda.Event())
            _stage_bufs[key] = buf
        return buf


def free_stage_bufs():
    """Release the pinned arena. Call at the end of a conversion job."""
    global _stage_bufs
    with _stage_bufs_lock:
        _stage_bufs = {}


class WeightStager:
    """
    Stages device-resident weights into pinned host memory on a side stream, one tensor ahead of
    consumption, so the copy overlaps the previous tensor's regularization and quantization.

    Usage, in a loop over tensors::

        stager = WeightStager()
        stager.prefetch(0, w0)                  # warm the pipeline
        for t, w in enumerate(weights):
            if t + 1 < len(weights): stager.prefetch(t + 1, weights[t + 1])
            w_host = stager.get(t, w)            # fp32 on host, shape (in, out), waits only if needed
            ... regularize / quantize w_host ...

    ``get()`` is idempotent with respect to ``prefetch()``: calling it on a tensor that was never
    prefetched simply stages it inline. That keeps the fast path and the fallback path identical in behaviour,
    so an A/B against the old code differs only in timing, never in output.
    """

    def __init__(self, device: torch.device | int | str | None = None):
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # All mutable state is per-thread. One WeightStager instance is shared by every worker in the multi-GPU
        # pool, so a plain attribute would be raced on by all of them: two threads would pick the same slot, one
        # would overwrite the other's pinned buffer between the host memcpy and the async D2H copy, and the weights
        # of whichever tensor lost would be silently corrupted. quant-plan.md §6.4 / §6.8.
        self._local = threading.local()

    # -- per-thread state ---------------------------------------------------------------------

    @property
    def owner(self) -> int:
        return threading.get_ident()

    @property
    def slot(self) -> int:
        return getattr(self._local, "slot", 0)

    @slot.setter
    def slot(self, v: int):
        self._local.slot = v

    @property
    def pending(self) -> dict:
        if not hasattr(self._local, "pending"):
            self._local.pending = {}
        return self._local.pending

    @property
    def copy_stream(self):
        # One stream per thread: a shared stream would serialize the copies anyway, and the stream is created
        # lazily because it needs a current device.
        if not hasattr(self._local, "stream"):
            self._local.stream = None
        return self._local.stream

    @copy_stream.setter
    def copy_stream(self, v):
        self._local.stream = v

    def _stream(self, w: torch.Tensor):
        if self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream(device = w.device)
        return self.copy_stream

    def prefetch(self, t: int, w: torch.Tensor):
        """Begin staging tensor ``t`` to pinned host memory. Safe to call twice; the second is a no-op."""
        if t in self.pending:
            return

        # Already on the host: pin it so the consumer gets the same fast path.
        if not w.is_cuda:
            nbytes = w.numel() * w.element_size()
            buf, _ = _get_stage_buf(self.owner, self.slot, nbytes)
            self.slot ^= 1
            pin = buf[:nbytes].view(w.dtype).view(w.shape)
            pin.copy_(w)
            self.pending[t] = (pin, None)
            return

        nbytes = w.numel() * w.element_size()
        buf, reuse_ev = _get_stage_buf(self.owner, self.slot, nbytes)
        self.slot ^= 1

        # The buffer's previous async copy must have drained before the host side refills it.
        reuse_ev.synchronize()

        pin = buf[:nbytes].view(w.dtype).view(w.shape)
        stream = self._stream(w)

        # Order the copy stream after whatever produced `w`. A side stream has no implicit ordering against the default
        # stream, so without this the D2H copy can read `w` before the producer's kernel has finished writing it — silent
        # weight corruption with no exception. `reuse_ev` below only orders this buffer's *previous* copy, not the
        # producer. Stage K bug 10.
        producer_ev = torch.cuda.Event()
        producer_ev.record(torch.cuda.current_stream(w.device))
        stream.wait_event(producer_ev)

        with torch.cuda.stream(stream):
            pin.copy_(w, non_blocking = True)
            reuse_ev.record(stream)

        ev = torch.cuda.Event()
        ev.record(stream)
        self.pending[t] = (pin, ev)

    def get(self, t: int, w: torch.Tensor) -> torch.Tensor:
        """
        Return tensor ``t`` as a contiguous fp32 host tensor of shape ``(in_features, out_features)``,
        waiting for its copy only if it has not landed yet.
        """
        self.prefetch(t, w)
        pin, ev = self.pending.pop(t)
        if ev is not None:
            ev.synchronize()
        # Transpose + cast on the host, exactly as the old inline path did. Same values, same dtype,
        # same shape, so the quantizer sees a bit-identical input.
        return pin.T.float().contiguous()

    def release(self, t: int):
        """Drop a staged tensor without consuming it (e.g. on an error path)."""
        self.pending.pop(t, None)

    def drain(self):
        """Drop everything staged. Does not synchronize; callers that need ordering must do it."""
        self.pending.clear()
