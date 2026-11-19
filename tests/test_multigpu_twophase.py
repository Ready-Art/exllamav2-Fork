"""
Measure the REAL two-phase path, matching quant_attn/quant_mlp exactly.

Phase 1: every prepare_hessian(), serial, on the primary device.
Phase 2: stage + quantize + pack, parallel across devices.

This is the shape the production code uses, so this is the honest speedup number.
The earlier e2e test put prepare_hessian() inside the worker, which parallelizes work
the design deliberately keeps serial.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
from exllamav2.conversion.qparams import QParams
from exllamav2.conversion.devices import resolve_devices
from exllamav2.conversion.schedule import quantize_tensor_set
from exllamav2.conversion.stager import WeightStager, free_stage_bufs


def make_layer(rows, cols, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn((cols, rows), generator=g, dtype=torch.float32).to(torch.half)
    lin = nn.Linear(rows, cols, False, device="meta", dtype=torch.float16)
    lin.weight = nn.Parameter(w.to(device))
    return lin


def make_hessian(rows, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed + 9999)
    a = torch.randn((512, rows), generator=g, dtype=torch.float32)
    return (a.T @ a) / 512


def build(rows, cols, n, qp, device=0):
    """Build layers + quantizers with Hessians already accumulated."""
    layers, quantizers = [], []
    for i in range(n):
        lin = make_layer(rows, cols, torch.device(f"cuda:{device}"), i)
        lq = AdaptiveGPTQ(lin)
        lq.hessian = make_hessian(rows, i, torch.device(f"cuda:{device}"))
        lq.num_batches = 1
        lq.hessian_device = device
        lq.quant_device = device
        lq.output_device = device
        layers.append(lin)
        quantizers.append(lq)
    return layers, quantizers


def run_serial(rows, cols, n, qp, devices):
    layers, quantizers = build(rows, cols, n, qp, devices[0])
    dev = devices[0]
    torch.cuda.synchronize()
    t0 = time.time()
    for lq in quantizers:
        lq.hessian_device = dev
        lq.prepare_hessian()                    # phase 1, serial
    for i, lq in enumerate(quantizers):
        lq.quant_device = dev
        lq.output_device = dev
        lq.stage_weights(None, i)                # phase 2, serial
        lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
        lq.quantize(keep_qweight=True, apply=False)
        with torch.inference_mode():
            lq.pack("t", qp)
    torch.cuda.synchronize()
    return time.time() - t0


def run_parallel(rows, cols, n, qp, devices):
    layers, quantizers = build(rows, cols, n, qp, devices[0])
    primary = devices[0]
    torch.cuda.synchronize()
    t0 = time.time()
    for lq in quantizers:
        lq.hessian_device = primary
        lq.prepare_hessian()                    # phase 1, serial, primary only
    stager = WeightStager(primary)
    def run_one(i, device):
        lq = quantizers[i]
        lq.quant_device = device
        lq.output_device = device
        lq.stage_weights(stager, i)              # phase 2, parallel
        lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
        lq.quantize(keep_qweight=True, apply=False)
        with torch.inference_mode():
            lq.pack("t", qp)
    quantize_tensor_set(list(range(n)), devices, "prof", run_one,
                        lambda i: layers[i].weight.numel())
    torch.cuda.synchronize()
    return time.time() - t0


def main():
    rows = cols = 4096
    n = 8
    qp = QParams.from_dict({"bits": [4.0], "group_size": {4.0: 64}, "bits_prop": [1.0], "scale_bits": 4})
    all_devs = resolve_devices(None)
    print(f"{n} tensors {rows}x{cols}, visible devices {all_devs}\n")

    t1 = run_serial(rows, cols, n, qp, all_devs[:1])
    print(f"1 device : {t1:6.2f} s")
    for k in (2, 3, 4):
        if len(all_devs) < k: continue
        tk = run_parallel(rows, cols, n, qp, all_devs[:k])
        print(f"{k} devices: {tk:6.2f} s   speedup {t1/tk:.2f}x")
    free_stage_bufs()


if __name__ == "__main__":
    main()
