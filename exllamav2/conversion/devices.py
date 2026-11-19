"""
Device selection and learned throughput ratios for multi-GPU quantization.

Port of EXL3's ``arch_prior_speed()`` / ``AutoSplit`` (exllamav3/modules/quant/exl3_lib/quantize.py:187-240),
adapted to EXL2's conversion pipeline. See quant-plan.md §6.7 (M4).

The point of this module is *not* to make anything faster by itself. It exists so that every
split decision in the conversion (cross-tensor bin packing in schedule.py, calibration row shards in
measure.py) is proportional to how fast each device actually is, instead of assuming all devices are
equal. On a homogeneous box it degenerates to an even split; on a mixed Ampere + Ada box it puts twice
the work on the Ada card from the first tensor onward, before any measurement exists.
"""

from __future__ import annotations

import threading
from functools import lru_cache


@lru_cache(maxsize = 64)
def arch_prior_speed(device: int) -> float:
    """
    Initial relative quant-throughput guess by architecture, used to seed device splits before any
    measurements exist. From measured conversion throughput: Ampere lands at about half of Ada, which is
    about 80% of Blackwell (Hopper assumed equal to Blackwell); anything older is pessimistically
    assumed half of Ampere.

    Returns a unit-less relative speed. Only ratios between devices matter.
    """

    import torch

    if torch.version.hip:
        return 1.0                                     # no measurements (HIP's major is the gfx generation)

    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return 1.0

    if major >= 9: return 5.0                            # Hopper, Blackwell
    if major == 8: return 4.0 if minor == 9 else 2.0     # Ada / Ampere
    return 1.0


def resolve_devices(devices = None) -> list[int]:
    """
    Normalize a user-supplied device list (indices, or "cuda:N" strings, or None) into a list of
    integer CUDA indices. ``None`` means "every visible device", which for a single-GPU box is ``[0]``.
    """

    import torch

    n = torch.cuda.device_count()
    if n == 0:
        return []

    if devices is None:
        return list(range(n))

    if isinstance(devices, (int, str)):
        devices = [devices]

    out = []
    for d in devices:
        if isinstance(d, str):
            d = d.strip()
            if d.startswith("cuda:"):
                d = d[len("cuda:"):]
            d = int(d)
        d = int(d)
        if d < 0 or d >= n:
            raise ValueError(f"Device index {d} out of range (visible devices: 0..{n - 1})")
        if d not in out:
            out.append(d)

    return out or list(range(n))


class AutoSplit:
    """
    Learned per-device speed ratios for heterogeneous multi-GPU conversion splits.

    Each workload kind (cross-tensor quant threads, calibration row workers) is tracked separately,
    since their per-unit costs differ; within a kind, only the ratios between devices matter. Speeds are
    EMA-damped so one noisy module (JIT warmup, thermal excursions) doesn't swing the split. Until every
    active device has a measurement for the kind, the split falls back to the per-architecture prior.

    Ported from exllamav3/modules/quant/exl3_lib/quantize.py:200.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.speeds = {}   # kind -> {device: ema work/sec}

    def report(self, kind: str, device: int, work: float, elapsed_s: float):
        """
        Record one measurement. ``elapsed_s < 0.05`` is discarded: single calls are ms-scale and a
        device would otherwise be over-weighted by one tiny tensor (EXL3 quantize.py:213).
        """
        if work <= 0 or elapsed_s < 0.05:
            return
        s = work / elapsed_s
        with self.lock:
            d = self.speeds.setdefault(kind, {})
            prev = d.get(device)
            d[device] = s if prev is None else 0.7 * prev + 0.3 * s

    def ratios(self, kind: str, devices: list) -> list:
        """
        Relative speed for each device in ``devices``. Measured speeds when every device has a sample,
        architecture priors otherwise. Mixing measured speeds with unit-less priors would skew the split, so
        it is one or the other.
        """
        with self.lock:
            d = self.speeds.get(kind)
            if d is not None and all(dev in d for dev in devices):
                return [d[dev] for dev in devices]
        return [arch_prior_speed(dev) for dev in devices]

    def has_measured(self, kind: str, devices: list) -> bool:
        with self.lock:
            d = self.speeds.get(kind)
            return d is not None and all(dev in d for dev in devices)

    def report_dict(self) -> dict:
        with self.lock:
            return {k: dict(v) for k, v in self.speeds.items()}


# Process-wide instance, shared by the quantize scheduler and the calibration sharder.
auto_split = AutoSplit()


def device_ratios(kind: str, devices: list[int]) -> list[float]:
    """Convenience wrapper: ratios for ``devices`` under workload ``kind``."""
    if not devices:
        return []
    return auto_split.ratios(kind, devices)


def split_sizes(total: int, devices: list[int], ratios: list[float] | None = None) -> list[int]:
    """
    Split ``total`` units of work across ``devices`` proportional to ``ratios`` (or evenly when ratios
    is None). The remainder is added to the last shard so the sizes always sum to ``total``. Negative
    shards (too many devices, or a tensor too small to divide) are folded backwards into the previous
    shard, which is what EXL3 does at quantize.py:290-293.
    """
    n = len(devices)
    if n == 0:
        return []

    if ratios:
        s = sum(ratios)
        if s <= 0:
            sizes = [total // n] * n
        else:
            sizes = [int(total * r / s) for r in ratios]
    else:
        sizes = [total // n] * n

    sizes[-1] += total - sum(sizes)

    for i in range(n - 2, -1, -1):
        if sizes[i + 1] < 0:
            sizes[i] += sizes[i + 1]
            sizes[i + 1] = 0

    return sizes
