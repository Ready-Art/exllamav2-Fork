"""
Cross-tensor parallel scheduling for the EXL2 quantizer.

Port of EXL3's ``quantize_linears_parallel()`` (exllamav3/conversion/convert_model.py:736) and its
worker-exception handling (line 770-778), adapted to EXL2's module structure.
See quant-plan.md §6.4 (M2).

What this does, and what it deliberately does not do
--------------------------------------------------
EXL2's ``quant()`` walks modules one at a time and, inside each module, quantizes each projection
sequentially. But the four attention projections and the three MLP projections of one layer are **independent
tensors**: they share a Hessian, but not a recursion. So they can be quantized on different GPUs at the same
time, one thread each.

This module provides the scheduling half only:

  * ``bin_pack()`` — greedy largest-remaining-capacity assignment of work items to devices, weighted by
    ``AutoSplit`` ratios.
  * ``run_on_devices()`` — one worker thread per bucket, each pinned to its device with ``torch.cuda.device()``,
    with worker exceptions collected and re-raised.

**Two things are NOT done here, on purpose:**

1. **No tensor is ever split.** ``find_params()`` computes ``qscale_max_t = torch.max(base_scale)`` — a single
   scalar over *all* columns (adaptivegptq.py:48). Splitting a tensor by column would give each shard a different
   ``qscale_max`` and silently change the quantization. Splitting by row is equally wrong before Phase 4, because
   ``quantize_range`` is a ``for c = a; c < b; c++`` loop where row ``c`` reads the error written by row ``c-1``
   (cpp/quantize_func.cpp:31-57). The unit of parallel work is therefore **one whole tensor**, never a slice.
   quant-plan.md §6.3 / §6.4 / R10.

2. **No Hessian work is done in a worker thread.** ``prepare()`` mutates ``self.hessian`` in place and then drops
   the reference, and q/k/v share one ``hessian_inv`` via ``reuse_h``. The caller must run every
   ``prepare_hessian()`` / ``reuse_h_hessian()`` serially on the primary device *before* dispatching tensors here.
   quant-plan.md §6.4 / R11.
"""

from __future__ import annotations

import threading
import time
import traceback

import torch

from exllamav2.conversion.devices import auto_split, device_ratios


def bin_pack(items, devices: list[int], ratios: list[float] | None, size_fn):
    """
    Greedy LPT bin-pack ``items`` onto ``devices`` by size, with per-device capacity proportional to ``ratios``.
    Each item goes to the bucket with the most remaining capacity (EXL3 convert_model.py:750-762).

    **Items are sorted largest-first before packing.** Greedy "largest remaining capacity" is only a good approximation
    under the LPT (Longest Processing Time) rule; walking the caller's insertion order instead can be arbitrarily bad. For
    Gemma-4 attention the caller's order is q, k, v, o with sizes 44, 22, 22, 44, so insertion order puts both 44s on
    device 0 and both 22s on device 1 — a 2:1 split on a homogeneous box. Sorting gives 66/66. Stage K bug 9.

    Sorting is output-neutral: a tensor's quantized values do not depend on which device runs the recursion.

    Returns a list of buckets, one per device, each a list of items.
    """
    if not devices:
        return []

    total = sum(size_fn(it) for it in items)
    if ratios:
        s = sum(ratios)
        capacity = [total * r / s for r in ratios] if s > 0 else [total / len(devices)] * len(devices)
    else:
        capacity = [total / len(devices)] * len(devices)

    buckets = [[] for _ in devices]
    for it in sorted(items, key = size_fn, reverse = True):
        sz = size_fn(it)
        best = max(range(len(capacity)), key = lambda x: capacity[x])
        capacity[best] -= sz
        buckets[best].append(it)

    return buckets


def run_on_devices(buckets, devices: list[int], worker_fn):
    """
    Run ``worker_fn(device_index, bucket)`` on one thread per non-empty bucket, each pinned to its device.

    A worker that raises must abort the whole job. A thread that dies mid-module would otherwise leave its
    linears unquantized while the job carries on, compiling a broken model at the end with only a warning in the
    scrollback. EXL3 fixed exactly this bug (convert_model.py:770-778); the fix is copied here.
    """
    errors = []
    started = []

    def guarded(device_idx, bucket):
        try:
            with torch.cuda.device(device_idx):
                with torch.inference_mode():
                    worker_fn(device_idx, bucket)
        except BaseException as e:
            errors.append((device_idx, e, traceback.format_exc()))

    for dev, bucket in zip(devices, buckets):
        if not bucket:
            continue
        t = threading.Thread(target = guarded, args = (dev, bucket), daemon = True)
        t.start()
        started.append(t)

    for t in started:
        t.join()

    if errors:
        dev, e, tb = errors[0]
        print(f" !! Quantization worker on device {dev} failed, aborting job")
        print(tb)
        raise e


def quantize_tensor_set(tasks, devices: list[int], kind: str, run_one, size_fn):
    """
    Quantize a set of independent tensors across ``devices``.

    ``tasks`` is a list of opaque items; ``run_one(task, device_index)`` performs the whole per-tensor
    quantization (staging, GPTQ recursion, packing) for one item on one device. ``size_fn(task)`` returns a
    relative work size used only for balancing.

    With one device this runs serially in the caller's thread and produces exactly the same result as the old
    code path, so a single-GPU box is unaffected by the presence of this module.
    """
    if not tasks:
        return

    devices = devices or [0]
    if len(devices) <= 1:
        dev = devices[0]
        for t in tasks:
            run_one(t, dev)
        return

    ratios = device_ratios(kind, devices)

    # Per-device accounting for the AutoSplit EMA. Work is measured in weight elements, which is the right proxy:
    # the GPTQ recursion cost is linear in numel for a fixed group size.
    acc = {d: [0, 0.0] for d in devices}
    acc_lock = threading.Lock()

    def worker(dev, bucket):
        t0 = time.time()
        for t in bucket:
            run_one(t, dev)
        elapsed = time.time() - t0
        work = sum(size_fn(t) for t in bucket)
        with acc_lock:
            acc[dev][0] += work
            acc[dev][1] += elapsed

    buckets = bin_pack(tasks, devices, ratios, size_fn)
    run_on_devices(buckets, devices, worker)

    for dev, (work, elapsed) in acc.items():
        if work > 0:
            auto_split.report(kind, dev, work, elapsed)
