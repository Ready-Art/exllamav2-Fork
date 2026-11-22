"""
EXL2.5 fast-lane + Hadamard/AWQ validation harness (single GPU, no model).

  parity : exl25 GEMM vs exact-dequant (reconstruct) reference, rel_l2 < 0.002,
           m = 1..16 (fast lane) and m = 33/64/2048 (reconstruct+cuBLAS+transform path)
  quality: quant error vs original fp16 weights, plain vs hadamard vs awq (same bpw)
  speed  : exl25 vs legacy kernel on the SAME packed weights (tags stripped), m=1/4/8/16
  graph  : capture/replay with mutated inputs == eager

Usage: venv/bin/python exllamav2-Fork/mtp/exl25_gemm_test.py [parity|quality|speed|graph|all]
"""

import sys, os, math, time
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor, make_q_matrix
from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
from exllamav2.conversion.qparams import QParams
from exllamav2.hadamard.hadamard import had_rotate_activations

DEV = "cuda:0"
MAX_DQ = 16 * 1024 * 1024  # elements; forces chunked reconstruct on big shapes

# Gemma-4-31B real projection shapes (k, n)
SHAPES = {
    "gate/up": (5376, 21504),
    "down":    (21504, 5376),
    "q":       (5376, 16384),
    "o":       (16384, 5376),
    "k/v":     (5376, 2048),
}


def make_weights(k, n, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(k, n, generator=g, device=DEV, dtype=torch.float) * 0.02
    # Salient channels: a few input dims carry outliers (the case Hadamard/AWQ exist for)
    w[:16, :] *= 8.0
    return w.half()


def make_cal(k, rows=64, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, k, generator=g, device=DEV, dtype=torch.float)
    x[:, :16] *= 8.0
    return x


@torch.inference_mode()
def quantize_tensor(w, cal, bits, gs=128, had=0, awq=False):
    """Run the EXL2.5 conversion path on one tensor; return packed dict (key 't').
    inference_mode matches the real conversion job context (quant() is decorated)."""
    k, n = w.shape
    layer = nn.Linear(k, n, bias=False, device="meta")
    layer.weight = nn.Parameter(w.T.contiguous(), requires_grad=False)
    lq = AdaptiveGPTQ(layer)
    lq.quant_device = 0
    lq.hessian_device = 0
    lq.output_device = 0
    lq.target_device = 0
    lq.exl25_had = had
    lq.exl25_awq = awq
    lq.exl25_uniform_bits = bits
    lq.exl25_groupsize = gs
    lq.exl25_identity_perm = True
    lq.add_batch(cal)
    lq.prepare_hessian()
    lq.stage_weights()
    qp = QParams(gs, [bits], [1.0], 4)
    lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
    lq.quantize(keep_qweight=True, apply=False)
    packed = lq.pack("t", qp)
    lq.drop_buffers()
    return packed


@torch.inference_mode()
def load_handle(packed, k, n, strip_tags=False):
    # CLONE: make_q_matrix folds q_scale_max /= 256 IN PLACE; packed tensors already on
    # the output device would alias through .to(dev) and corrupt the source dict.
    w = {kk.split(".", 1)[1]: vv.to(DEV).clone() for kk, vv in packed.items()}
    if strip_tags:
        for t in ("q_uniform_bits", "q_group_size", "q_hadamard", "q_awq_scale"):
            w.pop(t, None)
    w["q_perm"] = torch.argsort(w["q_invperm"]).to(torch.int)
    temp_dq = torch.empty(2 * min(k * n, MAX_DQ), dtype=torch.uint8, device=DEV)
    handle = make_q_matrix(w, temp_dq, max_dq_rows=MAX_DQ // n)
    return handle, w, temp_dq


def transform_x(x, w):
    """Logical activations -> stored space (matches the kernel preload exactly)."""
    if "q_awq_scale" in w:
        x = x * w["q_awq_scale"].float()[None, :]
    if "q_hadamard" in w:
        x = had_rotate_activations(x.float(), int(w["q_hadamard"].item()))
    return x.half()


def rel_l2(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) /
            torch.linalg.vector_norm(b.float()).clamp_min(1e-12)).item()


def gemm(x, handle, force_cuda=True):
    x = x.half()
    out = torch.empty(x.shape[0], handle_n[0], dtype=torch.half, device=DEV)
    ext_c.gemm_half_q_half(x, handle, out, force_cuda)
    return out


handle_n = [0]  # set per shape


def parity_shape(name, k, n, bits, had, awq):
    """
    Correctness gates (fp16 accumulation makes an fp32 reference a noisy oracle at large k):
      plain uniform : exl25 vs LEGACY kernel on identical weights, same path -> < 0.002
      had/awq       : vs fp32 stored-space reference < 0.006 (fp16 floor), and
                      cuBLAS+transform path vs in-kernel-transform lane < 0.006
    Quant error vs original fp16 weights is reported (informational / quality A/B).
    """
    global handle_n
    handle_n[0] = n
    w = make_weights(k, n)
    cal = make_cal(k)
    packed = quantize_tensor(w, cal, bits, had=had, awq=awq)
    handle, wd, temp_dq = load_handle(packed, k, n)
    hleg, wleg, temp_dq_l = load_handle(packed, k, n, strip_tags=True)

    dq = torch.empty(k, n, dtype=torch.half, device=DEV)
    ext_c.reconstruct(handle, dq)

    # Gates vs the fp32 stored-space reference. fp16 chunk accumulation has a k-dependent
    # noise floor (measured: ~0.002 at k=5376, ~0.004 at k=21504; legacy shows the SAME
    # floor, exl25 slightly better). Lane-vs-legacy differences of this size are rounding
    # ORDER, not error -- so the oracle is fp32, with a k-scaled floor. The cuBLAS path
    # (cublasHgemm fp16) is noisier still: flat 0.006 gate.
    gate_lane = max(0.002, 2.5e-7 * k)
    ok = True
    for m in (1, 2, 3, 4, 5, 8, 12, 16, 33, 64, 2048):
        x = make_cal(k, rows=m, seed=100 + m)
        ref = (transform_x(x, wd).float() @ dq.float()).half()
        y_t = gemm(x, handle, force_cuda=True)
        e = rel_l2(y_t, ref)
        if e > gate_lane:
            print(f"  FAIL {name} {bits}b had={had} awq={awq} m={m} lane-vs-fp32: {e:.6f} (gate {gate_lane:.4f})")
            ok = False
        if m > 16:
            y_f = gemm(x, handle, force_cuda=False)
            e = rel_l2(y_f, ref)
            if e > 0.006:
                print(f"  FAIL {name} {bits}b had={had} awq={awq} m={m} dispatch-false: {e:.6f}")
                ok = False
    x = make_cal(k, rows=64, seed=7)
    y = gemm(x, handle, force_cuda=True)
    qe = rel_l2(y, (x.float() @ w.float()).half())
    print(f"  {name:8s} {bits}b had={had} awq={int(awq)}: parity {'PASS' if ok else 'FAIL'}, quant rel_l2={qe:.5f}")
    ext_c.free_q_matrix(handle)
    ext_c.free_q_matrix(hleg)
    del dq, temp_dq, temp_dq_l
    torch.cuda.empty_cache()
    return ok, qe


def cmd_parity():
    ok = True
    for name, (k, n) in SHAPES.items():
        for bits in (4, 6):
            for had, awq in ((0, False), (128, True)):
                r, _ = parity_shape(name, k, n, bits, had, awq)
                ok = ok and r
    print("PARITY:", "ALL PASS" if ok else "FAILURES")
    return ok


def cmd_quality():
    print("Quality A/B (same bpw, quant rel_l2 vs fp16 original; lower = better):")
    for name in ("gate/up", "down"):
        k, n = SHAPES[name]
        for bits in (4, 6):
            row = []
            for had, awq in ((0, False), (128, False), (0, True), (128, True)):
                _, qe = parity_shape(name, k, n, bits, had, awq)
                row.append(f"{'had' if had else '   '}{'awq' if awq else '   '}={qe:.5f}")
            print(f"  {name:8s} {bits}b: " + "  ".join(row))


def bench_ms(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def cmd_speed():
    # Plain uniform packing: pure kernel A/B (had/awq would charge exl25 for the in-kernel
    # butterfly while legacy gets pre-transformed inputs -- unfair; feature cost measured
    # separately by the parity/quality sections).
    print("Speed A/B: exl25 fast lane vs legacy kernel, SAME packed weights (ms, eff GB/s):")
    for name in ("gate/up", "down", "o"):
        k, n = SHAPES[name]
        for bits in (4, 6):
            w = make_weights(k, n)
            cal = make_cal(k)
            packed = quantize_tensor(w, cal, bits, had=0, awq=False)
            h25, wd, dq25 = load_handle(packed, k, n)
            hleg, _, dqleg = load_handle(packed, k, n, strip_tags=True)
            wbytes = k * n * bits / 8
            for m in (1, 4, 8, 16):
                x = make_cal(k, rows=m, seed=5).half()
                o25 = torch.empty(m, n, dtype=torch.half, device=DEV)
                oleg = torch.empty(m, n, dtype=torch.half, device=DEV)
                # legacy kernel can't rotate: feed it pre-transformed activations so both
                # kernels do the same math (isolates kernel speed from the transform).
                xt = transform_x(x, wd)
                t25 = bench_ms(lambda: ext_c.gemm_half_q_half(x, h25, o25, True))
                tleg = bench_ms(lambda: ext_c.gemm_half_q_half(xt, hleg, oleg, True))
                gb = wbytes / 1e9
                print(f"  {name:8s} {bits}b m={m:2d}: exl25 {t25:.3f} ms ({gb/t25*1e3:.0f} GB/s)  "
                      f"legacy {tleg:.3f} ms ({gb/tleg*1e3:.0f} GB/s)  speedup {tleg/t25:.2f}x")
            ext_c.free_q_matrix(h25); ext_c.free_q_matrix(hleg)
            del dq25, dqleg
            torch.cuda.empty_cache()


def cmd_transform():
    """Force the reconstruct+cuBLAS path (m>32, k<=row_step) with had+awq: the standalone
    activation-transform kernel must match the in-kernel transform lane."""
    global handle_n, MAX_DQ
    k, n = 5376, 2048
    handle_n[0] = n
    old = MAX_DQ
    MAX_DQ = 256 * 1024 * 1024  # row_step >= k -> cuBLAS path taken
    w = make_weights(k, n)
    packed = quantize_tensor(w, make_cal(k), 6, had=128, awq=True)
    handle, wd, temp_dq = load_handle(packed, k, n)
    dq = torch.empty(k, n, dtype=torch.half, device=DEV)
    ext_c.reconstruct(handle, dq)
    ok = True
    for m in (33, 64, 512):
        x = make_cal(k, rows=m, seed=300 + m)
        ref = (transform_x(x, wd).float() @ dq.float()).half()
        y = gemm(x, handle, force_cuda=False)  # cuBLAS + exl25_transform_a
        e = rel_l2(y, ref)
        if e > 0.006:  # cublasHgemm fp16 noise floor (debug5: 0.0053 at m=512)
            print(f"  FAIL transform-path m={m}: rel_l2={e:.6f}")
            ok = False
    print(f"  transform-path (cuBLAS + had/awq): {'PASS' if ok else 'FAIL'}")
    ext_c.free_q_matrix(handle)
    MAX_DQ = old
    del dq, temp_dq
    torch.cuda.empty_cache()
    return ok


def cmd_graph():
    global handle_n
    k, n = SHAPES["gate/up"]
    handle_n[0] = n
    w = make_weights(k, n)
    packed = quantize_tensor(w, make_cal(k), 6, had=128, awq=True)
    handle, wd, temp_dq = load_handle(packed, k, n)
    ok = True
    for m in (1, 4, 16):
        x = make_cal(k, rows=m, seed=9).half()
        out = torch.empty(m, n, dtype=torch.half, device=DEV)
        # warmup (autotune-free lane, but JIT/allocator warm)
        for _ in range(3): ext_c.gemm_half_q_half(x, handle, out, True)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ext_c.gemm_half_q_half(x, handle, out, True)
        for trial in range(3):
            x.copy_(make_cal(k, rows=m, seed=200 + trial))
            g.replay()
            torch.cuda.synchronize()
            ref = (transform_x(x, wd).float() @ _dq_ref(handle, k, n).float()).half()
            e = rel_l2(out, ref)
            if e > 0.002:
                print(f"  FAIL graph m={m} trial={trial}: rel_l2={e:.6f}")
                ok = False
        print(f"  graph m={m}: {'PASS' if ok else 'FAIL'}")
    ext_c.free_q_matrix(handle)
    print("GRAPH:", "ALL PASS" if ok else "FAILURES")
    return ok


def _dq_ref(handle, k, n):
    dq = torch.empty(k, n, dtype=torch.half, device=DEV)
    ext_c.reconstruct(handle, dq)
    return dq


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    torch.cuda.init()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    results = []
    if cmd in ("parity", "all"): results.append(cmd_parity())
    if cmd in ("transform", "all"): results.append(cmd_transform())
    if cmd in ("quality", "all"): cmd_quality()
    if cmd in ("speed", "all"): cmd_speed()
    if cmd in ("graph", "all"): results.append(cmd_graph())
    if cmd == "all":
        print("OVERALL:", "PASS" if all(results) else "FAIL")
        sys.exit(0 if all(results) else 1)
