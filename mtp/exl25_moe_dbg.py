import sys, os
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exl25_gemm_test import quantize_tensor, load_handle, transform_x, rel_l2, make_weights, make_cal, DEV
from exl25_moe_test import rmsnorm, HIDDEN, INTER, E, TOPK, MAXROWS, MAX_DQ

bits, had, awq, rows = 6, 0, False, 1
g = torch.Generator(device=DEV).manual_seed(42)
norm_w = (torch.rand(HIDDEN, generator=g, device=DEV) * 0.5 + 0.75).half()
gate = (torch.randn(E, HIDDEN, generator=g, device=DEV) * 0.02).half()
handles, dqs, tags = [], [], []
for e in range(E):
    for (k, n) in ((HIDDEN, INTER), (INTER, HIDDEN), (HIDDEN, INTER)):
        w = make_weights(k, n, seed=1000 + e * 3 + len(handles))
        cal = make_cal(k, rows=32, seed=2000 + e)
        packed = quantize_tensor(w, cal, bits, had=had, awq=awq)
        h, wd, dqbuf = load_handle(packed, k, n)
        dq = torch.empty(k, n, dtype=torch.half, device=DEV)
        ext_c.reconstruct(h, dq)
        handles.append(h); dqs.append(dq); tags.append(wd)

def buf(n): return torch.zeros(n, dtype=torch.uint8, device=DEV)
logits_buf = buf(MAXROWS*E*2+128)
state_buf = buf(MAXROWS*HIDDEN*2+128)
a_buf = buf(MAXROWS*INTER*2+128)
b_buf = buf(MAXROWS*INTER*2+128)
moe = ext_c.make_q_moe_mlp(norm_w, none_tensor, True, 1e-6, gate, E, TOPK,
    [handles[e*3+0] for e in range(E)], [handles[e*3+1] for e in range(E)], [handles[e*3+2] for e in range(E)],
    state_buf, buf(MAXROWS*HIDDEN*2+128), a_buf,
    b_buf, logits_buf, buf(2*min(HIDDEN*INTER, MAX_DQ)), MAXROWS, False)

x = make_cal(HIDDEN, rows=rows, seed=77).half()
x_orig = x.clone()
xn = rmsnorm(x_orig, norm_w)
probs = F.softmax(xn.float() @ gate.float().T, dim=-1)
rw, sel = torch.topk(probs, TOPK, dim=-1)
rw = (rw / rw.sum(-1, keepdim=True)).half()
contrib = {}
for e in range(E):
    ridx, kidx = torch.where(sel == e)
    if ridx.numel() == 0: continue
    xe = xn[ridx]
    dq1, dq2, dq3 = dqs[e*3+0], dqs[e*3+1], dqs[e*3+2]
    t1, t2, t3 = tags[e*3+0], tags[e*3+1], tags[e*3+2]
    go = (transform_x(xe, t1).float() @ dq1.float()).half()
    uo = (transform_x(xe, t3).float() @ dq3.float()).half()
    hh = F.silu(go.float()) * uo.float()
    yy = (transform_x(hh, t2).float() @ dq2.float()).half()
    contrib[e] = (yy.float() * rw[ridx, kidx, None].float())

ext_c.q_moe_mlp_forward_(moe, x)
torch.cuda.synchronize()
got = (x - x_orig).float()
print("sel:", sel.tolist(), "rw:", rw.float().tolist())
kl = logits_buf[:E*2].view(torch.float16).float()
print("kernel temp_logits:", kl.tolist())
kn = state_buf[:HIDDEN*2].view(torch.float16)
print("kernel norm vs ref norm rel_l2:", rel_l2(kn, xn[0]))
kw = {e: kl[e].item() for e in range(E)}
c0, c4 = contrib[0][0], contrib[4][0]
print("kernel-weighted combo rel_l2:", rel_l2(got, kw[0]*c0 + kw[4]*c4))
print("got norm:", got.norm().item(), "correct norm:", (rw[0,0].item()*c0 + rw[0,1].item()*c4).norm().item())
# temp_a after forward = last selected expert's (4) activation output
xe = xn[:1]
go4 = (transform_x(xe, tags[4*3+0]).float() @ dqs[4*3+0].float()).half()
uo4 = (transform_x(xe, tags[4*3+2]).float() @ dqs[4*3+2].float()).half()
h4 = (F.silu(go4.float()) * uo4.float()).half()
ka = a_buf[:INTER*2].view(torch.float16)
kb = b_buf[:INTER*2].view(torch.float16)
print("expert4 act temp_a vs ref silu*up rel_l2:", rel_l2(ka, h4[0]))
print("expert4 temp_b vs ref up rel_l2:", rel_l2(kb, uo4[0]))
c0, c4 = contrib[0][0], contrib[4][0]
w0, w4 = rw[0,0].item(), rw[0,1].item()
cands = {
    "w0*c0+w4*c4 (correct)": w0*c0 + w4*c4,
    "c0+c4 (no routing mul)": c0 + c4,
    "w0*c0 only": w0*c0,
    "w4*c4 only": w4*c4,
    "c0 only": c0,
    "c4 only": c4,
    "w0*c0+c4": w0*c0 + c4,
    "c0+w4*c4": c0 + w4*c4,
}
for name, c in cands.items():
    print(f"  {name:28s} rel_l2={rel_l2(got, c):.6f}")
