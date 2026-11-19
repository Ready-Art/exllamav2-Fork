"""
Measure where the time goes in a per-tensor quantization, to explain the speedup ceiling.

M2 parallelizes only the weight half (stage + GPTQ recursion + pack). The Hessian half
(prepare_hessian) is serial by design, because q/k/v share one hessian_inv and prepare()
mutates it in place. quant-plan.md §6.4 / R11.

So the achievable speedup is bounded by Amdahl's law on the Hessian fraction. This script
measures that fraction directly so the ceiling is a number, not a guess.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
from exllamav2.conversion.qparams import QParams
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


def main():
    rows = cols = 4096
    n = 8
    dev = 0
    qp = QParams.from_dict({"bits": [4.0], "group_size": {4.0: 64}, "bits_prop": [1.0], "scale_bits": 4})

    print(f"{n} tensors of {rows}x{cols}, device {dev}\n")

    t_hess = 0.0
    t_stage = 0.0
    t_quant = 0.0
    t_pack = 0.0

    for i in range(n):
        lin = make_layer(rows, cols, torch.device(f"cuda:{dev}"), i)
        H = make_hessian(rows, i, torch.device(f"cuda:{dev}"))

        lq = AdaptiveGPTQ(lin)
        lq.hessian = H
        lq.num_batches = 1
        lq.hessian_device = dev
        lq.quant_device = dev
        lq.output_device = dev

        torch.cuda.synchronize()
        t0 = time.time()
        lq.prepare_hessian()
        torch.cuda.synchronize()
        t_hess += time.time() - t0

        st = WeightStager(dev)
        torch.cuda.synchronize()
        t0 = time.time()
        lq.stage_weights(st, i)
        torch.cuda.synchronize()
        t_stage += time.time() - t0

        lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
        torch.cuda.synchronize()
        t0 = time.time()
        lq.quantize(keep_qweight=True, apply=False)
        torch.cuda.synchronize()
        t_quant += time.time() - t0

        with torch.inference_mode():
            torch.cuda.synchronize()
            t0 = time.time()
            lq.pack("test", qp)
            torch.cuda.synchronize()
            t_pack += time.time() - t0

        lq.drop_buffers()
        free_stage_bufs()

    tot = t_hess + t_stage + t_quant + t_pack
    print(f"hessian (serial)  : {t_hess:6.2f} s  {100*t_hess/tot:5.1f}%")
    print(f"stage   (parallel): {t_stage:6.2f} s  {100*t_stage/tot:5.1f}%")
    print(f"quantize(parallel): {t_quant:6.2f} s  {100*t_quant/tot:5.1f}%")
    print(f"pack    (parallel): {t_pack:6.2f} s  {100*t_pack/tot:5.1f}%")
    print(f"total             : {tot:6.2f} s")

    par = t_stage + t_quant + t_pack
    print(f"\nserial fraction   : {100*t_hess/tot:5.1f}%")
    print(f"parallel fraction : {100*par/tot:5.1f}%")
    for nd in (2, 3, 4):
        amdahl = 1 / (t_hess/tot + par/tot/nd)
        print(f"  Amdahl ceiling with {nd} devices: {amdahl:.2f}x")


if __name__ == "__main__":
    main()
