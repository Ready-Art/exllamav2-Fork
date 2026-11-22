import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exllamav2.ext import exllamav2_ext as ext_c
import exl25_gemm_test as T
from exl25_gemm_test import quantize_tensor, load_handle, make_weights, make_cal, rel_l2, transform_x, DEV

# (a) exl25 lane vs legacy lane, down shape (k=21504), m=64 -- same accumulation scheme
k, n = 21504, 5376
w = make_weights(k, n, seed=0)
packed = quantize_tensor(w, make_cal(k), 6)
h25, wd, b25 = load_handle(packed, k, n)
packed2 = quantize_tensor(w, make_cal(k), 6)
hl, wl, bl = load_handle(packed2, k, n, strip_tags=True)
x = make_cal(k, rows=64, seed=164).half()
y25 = torch.empty(64, n, dtype=torch.half, device=DEV)
yl = torch.empty(64, n, dtype=torch.half, device=DEV)
ext_c.gemm_half_q_half(x, h25, y25, True)
ext_c.gemm_half_q_half(x, hl, yl, True)
dq = torch.empty(k, n, dtype=torch.half, device=DEV)
ext_c.reconstruct(h25, dq)
ref = (x.float() @ dq.float()).half()
print(f"(a) down m=64: exl25-vs-legacy {rel_l2(y25, yl):.6f} | exl25-vs-fp32 {rel_l2(y25, ref):.6f} | legacy-vs-fp32 {rel_l2(yl, ref):.6f}")
ext_c.free_q_matrix(h25); ext_c.free_q_matrix(hl)
del dq, b25, bl
torch.cuda.empty_cache()

# (b) transform-path (cuBLAS) vs fast lane, had+awq, k/v shape m=512
T.MAX_DQ = 256 * 1024 * 1024
k, n = 5376, 2048
w = make_weights(k, n, seed=0)
packed = quantize_tensor(w, make_cal(k), 6, had=128, awq=True)
h, wd, bq = load_handle(packed, k, n)
x = make_cal(k, rows=512, seed=812).half()
yc = torch.empty(512, n, dtype=torch.half, device=DEV)
yk = torch.empty(512, n, dtype=torch.half, device=DEV)
ext_c.gemm_half_q_half(x, h, yc, False)  # cuBLAS + transform
ext_c.gemm_half_q_half(x, h, yk, True)   # fast lane
dq = torch.empty(k, n, dtype=torch.half, device=DEV)
ext_c.reconstruct(h, dq)
ref = (transform_x(x, wd).float() @ dq.float()).half()
print(f"(b) k/v m=512 had+awq: cublas-vs-lane {rel_l2(yc, yk):.6f} | cublas-vs-fp32 {rel_l2(yc, ref):.6f} | lane-vs-fp32 {rel_l2(yk, ref):.6f}")
ext_c.free_q_matrix(h)
