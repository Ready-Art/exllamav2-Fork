"""
EXL2.5 MoE validation: fused q_moe_mlp C++ path with fast-lane (+Hadamard/AWQ) experts.

Proves the format's MoE support end to end: per-expert EXL2.5 tags flow through the fused
kernel's gemm_half_q_half_cuda calls (routing-weight early-exit + mul_r_weights contract),
and the in-kernel activation transform applies to expert inputs AND the silu*up intermediate
fed to w2. Parity vs a torch reference using the SAME dequantized (stored-space) weights.

Usage: venv/bin/python exllamav2-Fork/mtp/exl25_moe_test.py
"""

import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exl25_gemm_test import quantize_tensor, load_handle, transform_x, rel_l2, make_weights, make_cal, DEV

HIDDEN = 512
INTER = 1024
E = 8
TOPK = 2
MAXROWS = 8
MAX_DQ = 16 * 1024 * 1024


def rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).half()


def run_case(bits, had, awq, rows):
    g = torch.Generator(device=DEV).manual_seed(42)
    norm_w = (torch.rand(HIDDEN, generator=g, device=DEV) * 0.5 + 0.75).half()
    gate = (torch.randn(E, HIDDEN, generator=g, device=DEV) * 0.02).half()

    handles, dqs, tags, packed_all = [], [], [], []
    for e in range(E):
        for (k, n) in ((HIDDEN, INTER), (INTER, HIDDEN), (HIDDEN, INTER)):  # w1, w2, w3
            w = make_weights(k, n, seed=1000 + e * 3 + len(handles))
            cal = make_cal(k, rows=32, seed=2000 + e)
            packed = quantize_tensor(w, cal, bits, had=had, awq=awq)
            h, wd, dqbuf = load_handle(packed, k, n)
            dq = torch.empty(k, n, dtype=torch.half, device=DEV)
            ext_c.reconstruct(h, dq)
            handles.append(h); dqs.append(dq); tags.append(wd); packed_all.append((w, dqbuf))

    def buf(nbytes):
        return torch.zeros(nbytes, dtype=torch.uint8, device=DEV)

    # KEEP REFERENCES: QMoEMLP holds raw pointers to these scratch buffers; anonymous
    # temporaries would be freed and reused by the allocator (dangling-pointer corruption).
    scratch = [buf(MAXROWS * HIDDEN * 2 + 128), buf(MAXROWS * HIDDEN * 2 + 128),
               buf(MAXROWS * INTER * 2 + 128), buf(MAXROWS * INTER * 2 + 128),
               buf(MAXROWS * E * 2 + 128), buf(2 * min(HIDDEN * INTER, MAX_DQ))]
    moe = ext_c.make_q_moe_mlp(
        norm_w, none_tensor, True, 1e-6, gate, E, TOPK,
        [handles[e * 3 + 0] for e in range(E)],
        [handles[e * 3 + 1] for e in range(E)],
        [handles[e * 3 + 2] for e in range(E)],
        *scratch, MAXROWS, False)

    x = make_cal(HIDDEN, rows=rows, seed=77).half()
    x_orig = x.clone()

    # torch reference in stored space (same quantized weights)
    xn = rmsnorm(x_orig, norm_w)
    logits = (xn.float() @ gate.float().T)
    probs = F.softmax(logits, dim=-1)
    rw, sel = torch.topk(probs, TOPK, dim=-1)
    rw = (rw / rw.sum(-1, keepdim=True)).half()
    ref = x_orig.clone()
    for e in range(E):
        ridx, kidx = torch.where(sel == e)
        if ridx.numel() == 0: continue
        xe = xn[ridx]
        dq1, dq2, dq3 = dqs[e * 3 + 0], dqs[e * 3 + 1], dqs[e * 3 + 2]
        t1, t2, t3 = tags[e * 3 + 0], tags[e * 3 + 1], tags[e * 3 + 2]
        gate_out = (transform_x(xe, t1).float() @ dq1.float()).half()
        up_out = (transform_x(xe, t3).float() @ dq3.float()).half()
        h = F.silu(gate_out.float()) * up_out.float()
        y = (transform_x(h, t2).float() @ dq2.float()).half()
        ref.index_add_(0, ridx, (y.float() * rw[ridx, kidx, None].float()).half())

    ext_c.q_moe_mlp_forward_(moe, x)
    torch.cuda.synchronize()
    e = rel_l2(x, ref)
    ok = e < 0.005
    print(f"  MoE {bits}b had={had} awq={int(awq)} rows={rows}: rel_l2={e:.6f} {'PASS' if ok else 'FAIL'}")
    if not ok:
        got = (x - x_orig).float()
        csum = torch.zeros_like(got)
        for ee in range(E):
            ridx, kidx = torch.where(sel == ee)
            if ridx.numel() == 0: continue
            xe = xn[ridx]
            dq1, dq2, dq3 = dqs[ee*3+0], dqs[ee*3+1], dqs[ee*3+2]
            t1, t2, t3 = tags[ee*3+0], tags[ee*3+1], tags[ee*3+2]
            go = (transform_x(xe, t1).float() @ dq1.float()).half()
            uo = (transform_x(xe, t3).float() @ dq3.float()).half()
            hh = F.silu(go.float()) * uo.float()
            yy = (transform_x(hh, t2).float() @ dq2.float()).half()
            csum.index_add_(0, ridx, yy.float() * rw[ridx, kidx, None].float())
        print("    got-vs-recomputed-contrib:", rel_l2(got, csum))
        print("    ref-vs-x_orig-csum:", rel_l2(ref, x_orig.float() + csum))
        print("    ref sel:", sel.tolist(), "rw:", rw.float().tolist())
        xn2 = rmsnorm(x_orig, norm_w)
        lg2 = (xn2.float() @ gate.float().T)
        pr2 = F.softmax(lg2, dim=-1)
        rw2, sel2 = torch.topk(pr2, TOPK, dim=-1)
        print("    recompute sel:", sel2.tolist())
        for ee in range(E):
            ridx, kidx = torch.where(sel == ee)
            if ridx.numel() == 0: continue
            xe = xn2[ridx]
            dq1, dq2, dq3 = dqs[ee * 3 + 0], dqs[ee * 3 + 1], dqs[ee * 3 + 2]
            t1, t2, t3 = tags[ee * 3 + 0], tags[ee * 3 + 1], tags[ee * 3 + 2]
            g_o = (transform_x(xe, t1).float() @ dq1.float()).half()
            u_o = (transform_x(xe, t3).float() @ dq3.float()).half()
            hh = F.silu(g_o.float()) * u_o.float()
            yy = (transform_x(hh, t2).float() @ dq2.float()).half()
            print(f"    expert {ee}: rows {ridx.tolist()} contrib_norm {yy.float().norm().item():.4f}")

    ext_c.free_q_moe_mlp(moe)
    for h in handles: ext_c.free_q_matrix(h)
    del dqs, packed_all
    torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    torch.cuda.init()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    ok = True
    # optional filter: bits,had,rows (e.g. "6,0,1") to run one case in a fresh process
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    for bits in (4, 6):
        for had, awq in ((0, False), (128, True)):
            for rows in (1, 2, 4):
                if filt and tuple(int(x) for x in filt.split(",")) != (bits, had, rows):
                    continue
                ok = run_case(bits, had, awq, rows) and ok
    print("MOE:", "ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
