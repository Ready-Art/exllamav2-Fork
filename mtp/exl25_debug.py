import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exllamav2.ext import exllamav2_ext as ext_c
from exl25_gemm_test import load_handle, make_weights, make_cal, rel_l2, DEV
from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
from exllamav2.conversion.qparams import QParams
from torch import nn

k, n, bits, gs = 256, 128, 4, 128
w = make_weights(k, n, seed=0)
cal = make_cal(k, rows=32, seed=1)

def stats(t):
    t = t.float()
    return f"mean={t.mean().item():.6f} std={t.std().item():.6f} absmax={t.abs().max().item():.6f}"

@torch.inference_mode()
def quant_dbg(w, cal, bits, identity_perm, tag):
    layer = nn.Linear(k, n, bias=False, device="meta")
    layer.weight = nn.Parameter(w.T.contiguous(), requires_grad=False)
    lq = AdaptiveGPTQ(layer)
    lq.exl25_identity_perm = identity_perm
    lq.exl25_uniform_bits = bits if identity_perm else 0
    lq.exl25_groupsize = gs
    lq.add_batch(cal)
    lq.prepare_hessian()
    lq.stage_weights()
    print(f"[{tag}] staged weights: {stats(lq.weights)}")
    print(f"[{tag}] perm[:8]: {lq.perm[:8].tolist()}")
    qp = QParams(gs, [bits], [1.0], 4)
    lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
    lq.quantize(keep_qweight=True, apply=False)
    print(f"[{tag}] lq.quant (dequant fp32): {stats(lq.quant)}")
    print(f"[{tag}] lq.quant vs W[perm]: {rel_l2(lq.quant, w[lq.perm.to(w.device)])}")
    print(f"[{tag}] qscale: {stats(lq.qscale)}  qscale_max: {stats(lq.qscale_max)}")
    packed = lq.pack("t", qp)
    print(f"[{tag}] packed q_scale_max: {stats(packed['t.q_scale_max'])} dev={packed['t.q_scale_max'].device}")
    h, wd, dqbuf = load_handle(packed, k, n)
    print(f"[{tag}] loaded q_scale_max: {stats(wd['q_scale_max'])}")
    dq = torch.empty(k, n, dtype=torch.half, device=DEV)
    ext_c.reconstruct(h, dq)
    print(f"[{tag}] reconstruct: {stats(dq)}  vs W: {rel_l2(dq, w)}")
    ext_c.free_q_matrix(h)
    return packed

quant_dbg(w, cal, bits, False, "act-order")
quant_dbg(w, cal, bits, True, "identity")
