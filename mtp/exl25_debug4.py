import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor, make_q_matrix, make_group_map

DEV = "cuda:0"
k, n, bits, gs = 256, 512, 4, 128

# Hand-crafted EXL2.5 tensor: all q=9 -> dq = 1*scale; group scales 0.04 / 0.08.
# q_scale_max raw = 256*scale (load folds /256); stored qs nibble = 15 -> (15+1)^2/256 = 1.
qrows = k * bits // 32
qw = torch.full((qrows, n), 0x99999999 - (1 << 32), dtype=torch.int32, device=DEV)  # all nibbles 9
qscale = torch.full((k // gs, n // 8), -1, dtype=torch.int32, device=DEV)  # all qs=15
qscale_max = torch.tensor([256 * 0.04, 256 * 0.08], dtype=torch.half, device=DEV)
qgroups = torch.tensor([bits, 0, bits, qrows // 2], dtype=torch.short, device=DEV)
invperm = torch.arange(k, dtype=torch.int32, device=DEV)
gmap = make_group_map(qgroups.cpu(), qrows).to(DEV)

def build(tags):
    w = dict(q_weight=qw.clone(), q_invperm=invperm.clone(), q_scale=qscale.clone(),
             q_scale_max=qscale_max.clone(), q_groups=qgroups.clone(), q_group_map=gmap.clone())
    if tags:
        w["q_uniform_bits"] = torch.tensor(bits, dtype=torch.int32)
        w["q_group_size"] = torch.tensor(gs, dtype=torch.int32)
    else:
        w["q_perm"] = invperm.clone()
    tdq = torch.empty(2 * k * n, dtype=torch.uint8, device=DEV)
    return make_q_matrix(w, tdq, max_dq_rows=k), w

h25, w25 = build(True)
hl, wl = build(False)
dq = torch.empty(k, n, dtype=torch.half, device=DEV)
ext_c.reconstruct(h25, dq)
print("dq[0,0], dq[128,0] (expect 0.04, 0.08):", dq[0, 0].item(), dq[128, 0].item())

x = torch.ones(1, k, dtype=torch.half, device=DEV)
ref = (x.float() @ dq.float()).half()
print("ref[0,:4] (expect 15.36):", ref[0, :4].tolist())
y25 = torch.empty(1, n, dtype=torch.half, device=DEV)
yl = torch.empty(1, n, dtype=torch.half, device=DEV)
ext_c.gemm_half_q_half(x, h25, y25, True)
ext_c.gemm_half_q_half(x, hl, yl, True)
print("y25[0,:8]:", y25[0, :8].tolist())
print("yleg[0,:8]:", yl[0, :8].tolist())
print("y25 unique:", y25.unique().tolist()[:10])
print("yleg unique:", yl.unique().tolist()[:10])

# one-hot slot mapping probe (post-fix: all ratios must be 1.0)
ratios = []
for j in list(range(0, 16)) + list(range(120, 129)):
    xh = torch.zeros(1, k, dtype=torch.half, device=DEV); xh[0, j] = 1.0
    a = torch.empty(1, n, dtype=torch.half, device=DEV)
    b = torch.empty(1, n, dtype=torch.half, device=DEV)
    ext_c.gemm_half_q_half(xh, h25, a, True)
    ext_c.gemm_half_q_half(xh, hl, b, True)
    ratios.append(round(a[0, 0].item() / b[0, 0].item(), 3))
print("j=0..15 ratios y25/yleg:", ratios[:16])
print("j=120..128 ratios:", ratios[16:])

# EXACT Hadamard test: x = all ones -> x @ H_int = 128 at every 128-block start, 0 elsewhere
# (H_int row 0 is all +1; every other row sums to 0). Expected y = 128 * sum_b dq[b*128, :].
wh = dict(q_weight=qw.clone(), q_invperm=invperm.clone(), q_scale=qscale.clone(),
          q_scale_max=qscale_max.clone(), q_groups=qgroups.clone(), q_group_map=gmap.clone(),
          q_uniform_bits=torch.tensor(bits, dtype=torch.int32),
          q_group_size=torch.tensor(gs, dtype=torch.int32),
          q_hadamard=torch.tensor(128, dtype=torch.int32))
tdq = torch.empty(2 * k * n, dtype=torch.uint8, device=DEV)
hh = make_q_matrix(wh, tdq, max_dq_rows=k)
xones = torch.ones(1, k, dtype=torch.half, device=DEV)
yh = torch.empty(1, n, dtype=torch.half, device=DEV)
ext_c.gemm_half_q_half(xones, hh, yh, True)
expected = 128.0 * dq[::128, :].float().sum(dim=0)
err = (yh[0].float() - expected).abs().max().item()
print(f"HADAMARD exact test: max abs err = {err:.6f} (expect ~0; scale {expected[0].item():.2f})")
ext_c.free_q_matrix(hh)
