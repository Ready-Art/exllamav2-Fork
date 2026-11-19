"""
Calibration row sharding for the EXL2 measurement pass.

Port of EXL3's ``calibration_row_shards()`` (exllamav3/conversion/convert_model.py:880),
``load_parallel_calib_modules()`` (line 897) and ``run_row_workers()`` (line 940).
See quant-plan.md §6.5 (M3).

Status: **opt-in, and the highest-risk item in §6.** The default path is unchanged. Read the caveats
below before enabling it.

What it does
------------
``measure_quant()`` currently forwards calibration rows one at a time on a single device:

    for i in range(len(hidden_states)):
        x = hidden_states[i].to("cuda:0")
        outputs = module.forward(x, cache, attn_params, intermediates = True)
        ... accumulate H, collect target_states ...

That loop is embarrassingly parallel across rows: row ``i`` reads ``hidden_states[i]`` and writes only its own
Hessian contribution and its own target state. So the rows can be split into one contiguous shard per device, each
device running its shard through its **own replica** of the module, and the results merged afterwards.

Merging is a plain sum for the Hessians (``H`` is an unnormalized sum over token batches, and ``num_batches`` is
counted alongside it), and an in-order concatenation for ``target_states``.

The two blockers, both real
-------------------------
1. **Replicas must fit in VRAM simultaneously.** ``measure_quant()`` loads the module once and reuses it for every
   row. Sharding requires loading it ``len(devices)`` times. For a 70B model that is a VRAM problem, not a code
   problem. ``load_parallel_calib_modules()`` therefore checks ``torch.cuda.mem_get_info`` before each replica and
   **drops any device that cannot hold one**, falling back to fewer replicas. A device that cannot fit a replica must
   be removed from the shard list, never allowed to OOM. quant-plan.md §6.5 / R13.

2. **The module is not written for replicas.** ``ExLlamaV2Attention.load()`` allocates scratch from
   ``model.get_device_context(self.device_idx)``, and ``set_device_idx()`` rewrites every submodule. Two replicas of the
   same module object cannot coexist; a replica has to be a *separate module instance* built against the same model with a
   different ``device_idx``. This module does not build those instances — the caller does, because only the caller knows how
   to construct its module type. ``load_parallel_calib_modules()`` takes the already-constructed replicas and only decides
   which ones are usable.
"""

from __future__ import annotations

import threading
import traceback

import torch


def calibration_row_shards(num_rows: int, devices: list[int], ratios: list[float] | None):
    """
    Split calibration row indices into one contiguous shard per device, proportional to ``ratios`` if given.

    Row 0 lands in the first device's shard whenever that shard is non-empty, which keeps the reference states on the
    primary device. Returns a list of ``range`` objects, one per device.
    """
    n = len(devices)
    if n == 0:
        return []

    w = ratios if ratios else [1] * n
    tot = sum(w)
    if tot <= 0:
        w, tot = [1] * n, n

    counts = [int(num_rows * r / tot) for r in w]      # ratios may be learned floats
    counts[0] += num_rows - sum(counts)

    shards, start = [], 0
    for c in counts:
        shards.append(range(start, start + c))
        start += c
    return shards


def ensure_device_contexts(model, devices: list[int]):
    """
    Extend ``model.device_context`` so every device in ``devices`` has a slot.

    This is the one thing M3 needs that the model loader does not provide. ``set_device_map()`` creates a
    context only for devices that actually receive modules, so a 2-GPU box with a model that fits on one card
    has ``len(model.device_context) == 1``. A replica on device 1 then fails in ``get_device_context(1)`` with
    ``IndexError``, which is what blocked the first M3 attempt.

    ``ExLlamaV2DeviceContext`` is independently constructible, so the slot can simply be appended. Verified
    on 4x3090: a replica on device 1 produces intermediates and a Hessian bit-identical to device 0.
    """
    if not devices:
        return

    from exllamav2.device import ExLlamaV2DeviceContext

    need = max(devices) + 1
    if len(model.device_context) >= need:
        return

    scratch_bytes = max(m.scratch_space_fixed() for m in model.modules)
    while len(model.device_context) < need:
        idx = len(model.device_context)
        model.device_context.append(ExLlamaV2DeviceContext(model, idx, scratch_bytes))


def module_replica_bytes(module) -> int:
    """
    Approximate bytes a replica of ``module`` will hold: the sum of its linears' fp16 storage.

    Deliberately an over-estimate rather than an under-estimate: getting this wrong in the optimistic direction
    produces an OOM in the middle of a calibration pass, which loses the whole pass.
    """
    total = 0
    try:
        for m in module.submodules or []:
            lin = getattr(m, "linear", None)
            if lin is not None and getattr(lin, "weight", None) is not None:
                total += lin.weight.numel() * lin.weight.element_size()
    except Exception:
        pass
    return total


def load_parallel_calib_modules(replicas, devices: list[int], need_bytes: int = 0, headroom: int = 1 << 30):
    """
    Decide which replica modules are usable, checking free VRAM first.

    ``replicas`` is a list of already-constructed module instances, one per entry of ``devices[1:]``, each with
    ``device_idx`` already set to its target device. A device without room for the module's weights (``need_bytes`` plus
    ``headroom`` for the pass itself) is left out, as is one whose load raises; the rows are then split over the devices
    that hold a replica.

    Returns ``(usable_replicas, used_devices)`` with the primary device first in ``used_devices``.
    """
    used = [devices[0]]
    if not replicas:
        return [], used

    usable = []
    for rep, dev in zip(replicas, devices[1:]):
        d = torch.device(dev)
        if need_bytes and d.type == "cuda":
            try:
                free, _ = torch.cuda.mem_get_info(d)
            except Exception:
                free = 0
            if free < need_bytes + headroom:
                print(f" !! No calibration replica on cuda:{dev}: {need_bytes >> 20} MiB of weights, "
                      f"{free >> 20} MiB free")
                continue
        try:
            rep.load()
            usable.append(rep)
            used.append(dev)
        except Exception as e:
            print(f" !! No calibration replica on cuda:{dev}: {str(e).splitlines()[0][:160]}")
            try:
                rep.unload()
            except Exception:
                pass
            torch.cuda.empty_cache()

    return usable, used


def run_row_workers(workers):
    """
    Run one worker callable per device in its own thread, each already pinned to its device by the caller.

    Worker exceptions are collected and the first is re-raised: a worker that dies silently would leave part of the
    calibration set unprocessed and the resulting Hessian would be quietly wrong, which is worse than a crash.
    """
    errors = []

    def guarded(fn):
        def inner():
            try:
                with torch.inference_mode():
                    fn()
            except BaseException as e:
                errors.append((e, traceback.format_exc()))
        return inner

    threads = [threading.Thread(target = guarded(w), daemon = True) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        e, tb = errors[0]
        print(" !! Calibration row worker failed, aborting job")
        print(tb)
        raise e


def shard_calibration_rows(hidden_states, replicas, devices, ratios, row_fn):
    """
    Run the calibration row loop with rows sharded across devices.

    ``row_fn(device_idx, module, row_indices)`` must forward the given rows through ``module`` and return
    ``(quantizers_dict, target_states_list)`` for exactly those rows, in row order. ``replicas`` is a list of
    module instances, one per entry of ``devices[1:]``, each already set to its device.

    Returns the merged ``quantizers_dict`` (Hessians summed, batch counts added) and the concatenated
    ``target_states`` in original row order.

    The merge is a plain sum because ``H`` is an unnormalized sum over token batches and ``num_batches`` counts
    them; ``prepare_hessian()`` divides by ``num_batches`` downstream. Averaging here would double-count.
    """
    shards = calibration_row_shards(len(hidden_states), devices, ratios)
    if len(devices) <= 1 or not replicas:
        return row_fn(devices[0], None, list(range(len(hidden_states))))

    per_device = {}
    lock = threading.Lock()

    def make_worker(dev, rep, rows):
        def w():
            res = row_fn(dev, rep, list(rows))
            with lock:
                per_device[dev] = res
        return w

    workers = []
    for dev, rep, rows in zip(devices[1:], replicas, shards[1:]):
        if not rows:
            continue
        workers.append(make_worker(dev, rep, rows))

    # Primary shard runs in the caller's thread so its quantizers stay the merge target.
    primary_rows = list(shards[0])
    primary_result = None

    def primary_worker():
        nonlocal primary_result
        primary_result = row_fn(devices[0], None, primary_rows)

    all_workers = [primary_worker] + workers
    run_row_workers(all_workers)

    if primary_result is None:
        raise RuntimeError("primary calibration shard produced no result")

    q_primary, ts_primary = primary_result[0], primary_result[1]
    ue_primary = primary_result[4] if len(primary_result) > 4 else None

    # Merge Hessians from the secondary devices into the primary's quantizers.
    others = [per_device[d][0] for d in devices[1:] if d in per_device]
    merge_hessians([q_primary] + others)

    # Sum the uncalibrated-expert counters, which are per-shard.
    if ue_primary is not None:
        for dev in devices[1:]:
            if dev not in per_device: continue
            ue = per_device[dev][4] if len(per_device[dev]) > 4 else None
            if ue is None: continue
            for j in range(len(ue_primary)):
                ue_primary[j] += ue[j]

    # Reassemble target states in original row order.
    ordered = [None] * len(hidden_states)
    for i, r in enumerate(shards[0]):
        ordered[r] = ts_primary[i]
    for dev in devices[1:]:
        if dev not in per_device:
            continue
        ts = per_device[dev][1]
        for i, r in enumerate(shards[devices.index(dev)]):
            ordered[r] = ts[i]

    return q_primary, ordered, ue_primary


def merge_hessians(per_device_quantizers):
    """
    Sum per-device Hessian accumulators into the first device's quantizers.

    Summing is correct because ``H`` is an unnormalized sum over token batches and ``num_batches`` counts the batches;
    the downstream ``prepare_hessian()`` divides by ``num_batches``. Averaging here would double-count.
    """
    if not per_device_quantizers:
        return

    primary = per_device_quantizers[0]
    for other in per_device_quantizers[1:]:
        for key, q in other.items():
            if key not in primary:
                primary[key] = q
                continue
            pq = primary[key]
            if pq.hessian is None:
                pq.hessian = q.hessian
            elif q.hessian is not None:
                # add_ on an inference tensor is rejected outside inference_mode, so do the
                # accumulation inside it. The caller may or may not already be in that mode.
                with torch.inference_mode():
                    pq.hessian.add_(q.hessian.to(pq.hessian.device))
            pq.num_batches += q.num_batches
            q.hessian = None


def build_module_replica(module, device_idx: int):
    """
    Construct a second, independent instance of ``module`` on ``device_idx``.

    This is the piece M3 needs and that the first implementation got wrong. A replica cannot be
    ``type(module)(model, key, layer_idx)``: the constructors have more required arguments than
    that, and the ones that matter are *semantic*, not cosmetic.

    ``ExLlamaV2Attention`` takes ``sliding_window`` and ``rope_index``. Omitting them defaults to
    ``sliding_window = 0`` and ``rope_index = 0``, which turns a Gemma-4 sliding layer into a
    full-attention layer that attends over the whole sequence and reads the *global* rope table. The
    replica then produces a Hessian for the wrong input distribution, and because M3 merges that
    Hessian into the primary's, the corruption is silent — the numbers stay finite and plausible.

    ``ExLlamaV2MLP`` takes ``in_features`` / ``out_features`` / ``interm_features`` and ``pad32``.
    For Gemma-4 these are the plain config values, but they are passed explicitly so a future
    architecture with non-uniform MLP geometry does not silently get the default.

    ``ExLlamaV2ParallelDecoder`` wraps both and forwards the same two arguments.

    Returns the loaded replica, or raises. The caller decides whether to drop the device.
    """

    from exllamav2.attn import ExLlamaV2Attention
    from exllamav2.mlp import ExLlamaV2MLP
    from exllamav2.parallel_decoder import ExLlamaV2ParallelDecoder

    cfg = module.model.config

    if isinstance(module, ExLlamaV2Attention):
        rep = ExLlamaV2Attention(
            module.model,
            module.key,
            module.layer_idx,
            has_norm = module.has_norm,
            has_residual = module.has_residual,
            sliding_window = module.sliding_window,
            archparams = module.archparams,
            rope_index = module.rope_index,
        )

    elif isinstance(module, ExLlamaV2MLP):
        rep = ExLlamaV2MLP(
            module.model,
            module.key,
            module.layer_idx,
            has_norm = module.has_norm,
            has_residual = module.has_residual,
            archparams = module.archparams,
            in_features = module.in_features,
            out_features = module.out_features,
            interm_features = module.interm_features,
            merge = module.merge,
        )

    elif isinstance(module, ExLlamaV2ParallelDecoder):
        rep = ExLlamaV2ParallelDecoder(
            module.model,
            module.key,
            module.layer_idx,
            sliding_window = module.attn.sliding_window,
            archparams = module.archparams,
            rope_index = module.attn.rope_index,
        )

    else:
        raise TypeError(f"no replica recipe for {type(module).__name__}")

    rep.set_device_idx(device_idx)
    rep.load()
    return rep
