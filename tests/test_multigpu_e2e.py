"""
End-to-end test of the multi-GPU quantizer on real model weights.

Two things must hold, in this order of importance:

1. IDENTITY. Quantizing a tensor with N devices must produce a bit-identical packed
   tensor to quantizing it with 1 device. M2 splits by *tensor*, never by column or row,
   so there is no numerical reason for any difference. If this fails, the split is wrong.

2. SPEEDUP. With 2+ devices the wall clock must drop. This is the only thing M2 is for.

Run:  python tests/test_multigpu_e2e.py [--devices 0,1] [--tensors N]
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
from exllamav2.conversion.qparams import QParams
from exllamav2.conversion.devices import resolve_devices
from exllamav2.conversion.schedule import quantize_tensor_set
from exllamav2.conversion.stager import WeightStager, free_stage_bufs

MODEL = os.environ.get("MODEL_DIR", "/home/westfox35/exl2work/models/gemma-4-31B-it")


def make_layer(rows, cols, device, seed):
    """A synthetic linear with the same shape as a real projection."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn((cols, rows), generator=g, dtype=torch.float32).to(torch.half)
    lin = nn.Linear(rows, cols, False, device="meta", dtype=torch.float16)
    lin.weight = nn.Parameter(w.to(device))
    return lin


def make_hessian(rows, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed + 9999)
    autu = torch.randn((512, rows), generator=g, dtype=torch.float32)
    return (autu.T @ autu) / 512


def quantize_one(lin, H, device, stager=None, key=0, qp=None):
    """
    Quantize one linear exactly as conversion/quantize.py does: prepare the Hessian
    serially, then stage weights, run the recursion, and pack.
    """
    lq = AdaptiveGPTQ(lin)
    lq.hessian = H.to(device)
    lq.num_batches = 1
    lq.hessian_device = device
    lq.quant_device = device
    lq.output_device = device

    lq.prepare_hessian()
    if stager is not None:
        lq.stage_weights(stager, key)
    else:
        lq.stage_weights(None, key)

    lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
    lq.quantize(keep_qweight=True, apply=False)
    # pack() does in-place updates on inference tensors, so it must run inside
    # inference_mode, exactly as quant() does in the real pipeline.
    with torch.inference_mode():
        packed = lq.pack("test", qp)
    return packed, lq


def compare_packed(a, b):
    """Bit-exact comparison of two packed dicts."""
    keys = sorted(set(a) | set(b))
    diffs = {}
    for k in keys:
        if k not in a or k not in b:
            diffs[k] = "MISSING"
            continue
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            diffs[k] = f"shape {ta.shape} vs {tb.shape}"
        elif ta.dtype != tb.dtype:
            diffs[k] = f"dtype {ta.dtype} vs {tb.dtype}"
        elif not torch.equal(ta.cpu(), tb.cpu()):
            d = (ta.cpu().float() - tb.cpu().float()).abs().max().item()
            diffs[k] = f"maxdiff {d}"
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=2048)
    ap.add_argument("--ntensors", type=int, default=4)
    args = ap.parse_args()

    devices = resolve_devices([int(x) for x in args.devices.split(",")])
    print(f"devices: {devices}")
    if len(devices) < 2:
        print("Need at least 2 devices for this test.")
        return

    qp = QParams.from_dict({"bits": [4.0], "group_size": {4.0: 64}, "bits_prop": [1.0], "scale_bits": 4})

    # Build the same set of tensors twice: one set for the serial run, one for the parallel run.
    print(f"\nbuilding {args.ntensors} tensors of {args.rows}x{args.cols} ...")
    layers = []
    hessians = []
    for i in range(args.ntensors):
        layers.append(make_layer(args.rows, args.cols, torch.device("cuda:0"), i))
        hessians.append(make_hessian(args.rows, i, torch.device("cuda:0")))

    # --- Serial baseline on one device ---
    print("\n--- serial (1 device) ---")
    torch.cuda.synchronize()
    t0 = time.time()
    serial = []
    for i in range(args.ntensors):
        packed, _ = quantize_one(layers[i], hessians[i], devices[0], qp=qp)
        serial.append(packed)
        torch.cuda.synchronize()
    t_serial = time.time() - t0
    print(f"serial: {t_serial:.2f} s")

    # --- Parallel across devices ---
    print(f"\n--- parallel ({len(devices)} devices) ---")
    stager = WeightStager(devices[0])
    results = [None] * args.ntensors

    def run_one(task, device):
        i = task
        packed, _ = quantize_one(layers[i], hessians[i], device, stager=stager, key=i, qp=qp)
        results[i] = packed

    torch.cuda.synchronize()
    t0 = time.time()
    quantize_tensor_set(list(range(args.ntensors)), devices, "e2e", run_one,
                        lambda i: layers[i].weight.numel())
    torch.cuda.synchronize()
    t_par = time.time() - t0
    print(f"parallel: {t_par:.2f} s")

    # --- Identity check ---
    print("\n--- identity ---")
    ok = True
    for i in range(args.ntensors):
        d = compare_packed(serial[i], results[i])
        if d:
            ok = False
            print(f"FAIL  tensor {i}: {d}")
        else:
            print(f"PASS  tensor {i}: bit-identical")
    free_stage_bufs()

    print(f"\nserial {t_serial:.2f}s  parallel {t_par:.2f}s  speedup {t_serial / t_par:.2f}x")
    if ok:
        print("RESULT: parallel output is bit-identical to serial.")
    else:
        print("RESULT: parallel output DIFFERS from serial — the split is wrong.")


if __name__ == "__main__":
    main()
