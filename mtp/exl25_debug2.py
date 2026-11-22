import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exllamav2.ext import exllamav2_ext as ext_c
from exl25_gemm_test import quantize_tensor, load_handle, make_weights, make_cal, rel_l2, transform_x, DEV

k, n, bits = 5376, 21504, 4
w = make_weights(k, n, seed=0)
cal = make_cal(k, rows=64, seed=1)
packed = quantize_tensor(w, cal, bits)
handle, wd, temp_dq = load_handle(packed, k, n)
dq = torch.empty(k, n, dtype=torch.half, device=DEV)
ext_c.reconstruct(handle, dq)
print("reconstruct vs W:", rel_l2(dq, w))

for m in (1, 4, 16):
    x = make_cal(k, rows=m, seed=100 + m)
    ref = (transform_x(x, wd).float() @ dq.float()).half()
    y = torch.empty(m, n, dtype=torch.half, device=DEV)
    ext_c.gemm_half_q_half(x.half(), handle, y, True)
    print(f"m={m}: kernel vs fp32-ref rel_l2 = {rel_l2(y, ref):.6f}")
    ref16 = (x.half() @ dq)
    print(f"m={m}: kernel vs fp16-ref rel_l2 = {rel_l2(y, ref16):.6f}")
    # legacy on a FRESH packed copy (no aliasing)
    packed2 = quantize_tensor(w, cal, bits)
    hl, wl, dql = load_handle(packed2, k, n, strip_tags=True)
    yl = torch.empty(m, n, dtype=torch.half, device=DEV)
    ext_c.gemm_half_q_half(x.half(), hl, yl, True)
    print(f"m={m}: legacy vs fp32-ref rel_l2 = {rel_l2(yl, ref):.6f}; exl25 vs legacy = {rel_l2(y, yl):.6f}")
    ext_c.free_q_matrix(hl)
    del dql
ext_c.free_q_matrix(handle)
