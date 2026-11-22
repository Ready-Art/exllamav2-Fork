import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exllamav2.ext import exllamav2_ext as ext_c
from exl25_gemm_test import quantize_tensor, load_handle, make_weights, make_cal, rel_l2, DEV

# (k, n): kn heuristic -> 128 for narrow n / short k, 256 for big grids
for (k, n) in ((256, 128), (5376, 2048), (5376, 21504), (1024, 1024)):
    w = make_weights(k, n, seed=0)
    cal = make_cal(k, rows=32, seed=1)
    packed = quantize_tensor(w, cal, 4)
    h25, wd, b25 = load_handle(packed, k, n)
    packed2 = quantize_tensor(w, cal, 4)
    hl, wl, bl = load_handle(packed2, k, n, strip_tags=True)
    dq = torch.empty(k, n, dtype=torch.half, device=DEV)
    ext_c.reconstruct(h25, dq)
    x = make_cal(k, rows=1, seed=5).half()
    ref = (x.float() @ dq.float()).half()
    y25 = torch.empty(1, n, dtype=torch.half, device=DEV)
    yl = torch.empty(1, n, dtype=torch.half, device=DEV)
    ext_c.gemm_half_q_half(x, h25, y25, True)
    ext_c.gemm_half_q_half(x, hl, yl, True)
    print(f"k={k:5d} n={n:5d}: exl25 {rel_l2(y25, ref):.6f}  legacy {rel_l2(yl, ref):.6f}")
    ext_c.free_q_matrix(h25); ext_c.free_q_matrix(hl)
    del dq, b25, bl
    torch.cuda.empty_cache()
