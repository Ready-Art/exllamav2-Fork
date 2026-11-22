"""Parity + benchmark for the split-KV hd512 verify attention kernel (ext_c.verify_hd512).

Single GPU, no TP/model needed. Validates correctness vs an fp32 reference and times
the kernel against FlashInfer's single_prefill (ragged fa2) and paged-prefill wrapper
at the MTP verify shapes (small q_len, long kv, GQA).

Run after rebuilding the ext:
    cd ~/exl2work && venv/bin/python exllamav2-Fork/mtp/verify_hd512_test.py
"""
import torch
import flashinfer
from exllamav2.ext import exllamav2_ext as ext_c

torch.manual_seed(0)
dev = torch.device("cuda:0")


def reference(q, k, v, sm_scale):
    # fp32 lower-right causal GQA attention. q (ql,nq,512), k/v (kv,nkv,512).
    ql, nq, hd = q.shape
    kv, nkv, _ = k.shape
    group = nq // nkv
    qf = q.float()
    kf = k.float()
    vf = v.float()
    out = torch.empty(ql, nq, hd, device=q.device, dtype=torch.float32)
    rows = torch.arange(ql, device=q.device).unsqueeze(1)
    cols = torch.arange(kv, device=q.device).unsqueeze(0)
    mask = cols <= (kv - ql + rows)            # lower-right causal
    for h in range(nq):
        kvh = h // group
        s = (qf[:, h, :] @ kf[:, kvh, :].transpose(0, 1)) * sm_scale
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[:, h, :] = p @ vf[:, kvh, :]
    return out


def run_custom(q, k, v, po, pl, o, splits, sm_scale, kv_len_dev):
    ql, nq, hd = q.shape
    kv, nkv, _ = k.shape
    ext_c.verify_hd512(q, k, v, o, po, pl, kv_len_dev, ql, nq, nkv, splits, sm_scale)
    return o


def make(ql, kv, nq, nkv, hd=512):
    q = torch.randn(ql, nq, hd, device=dev, dtype=torch.half) * 0.5
    k = torch.randn(kv, nkv, hd, device=dev, dtype=torch.half) * 0.5
    v = torch.randn(kv, nkv, hd, device=dev, dtype=torch.half) * 0.5
    return q, k, v


def parity():
    sm_scale = 1.0
    print("== PARITY (rel_l2 vs fp32 reference; PASS < 0.01) ==")
    allok = True
    for (nq, nkv) in ((8, 1), (32, 4)):
        for ql in (2, 3, 4):
            for kv in (1315, 4096, 32768, 79067):
                q, k, v = make(ql, kv, nq, nkv)
                kv_len_dev = torch.tensor([kv], device=dev, dtype=torch.int32)
                ref = reference(q, k, v, sm_scale)
                for splits in (1, 8, 32):
                    po = torch.empty(splits, ql, nq, 512, device=dev, dtype=torch.float32)
                    pl = torch.empty(splits, ql, nq, 2, device=dev, dtype=torch.float32)
                    o = torch.empty_like(q)
                    run_custom(q, k, v, po, pl, o, splits, sm_scale, kv_len_dev)
                    err = (o.float() - ref).norm() / ref.norm()
                    ok = err.item() < 0.01
                    allok &= ok
                    if not ok or splits == 32:
                        print(f"  nq{nq}/nkv{nkv} q{ql} kv{kv} s{splits}: rel_l2={err.item():.5f} "
                              f"{'PASS' if ok else 'FAIL'}")
    print("PARITY:", "ALL PASS" if allok else "FAILURES PRESENT")
    return allok


def bench():
    sm_scale = 1.0
    print("== BENCH (ms/call; GB/s = KV-bytes-read-once / time) ==")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    for (nq, nkv) in ((8, 1), (32, 4)):
        for ql in (3,):
            for kv in (1315, 32768, 79067):
                q, k, v = make(ql, kv, nq, nkv)
                kv_len_dev = torch.tensor([kv], device=dev, dtype=torch.int32)
                kv_bytes = kv * nkv * 512 * 2 * 2
                splits = max(1, min(64, kv // 2048))
                po = torch.empty(splits, ql, nq, 512, device=dev, dtype=torch.float32)
                pl = torch.empty(splits, ql, nq, 2, device=dev, dtype=torch.float32)
                o = torch.empty_like(q)

                def timeit(fn, n=50):
                    for _ in range(10):
                        fn()
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
                    s.record()
                    for _ in range(n):
                        fn()
                    e.record(); torch.cuda.synchronize()
                    return s.elapsed_time(e) / n

                t_cust = timeit(lambda: run_custom(q, k, v, po, pl, o, splits, sm_scale, kv_len_dev))
                t_single = timeit(lambda: flashinfer.single_prefill_with_kv_cache(
                    q, k, v, causal=True, sm_scale=sm_scale))
                print(f"  nq{nq}/nkv{nkv} q{ql} kv{kv} s{splits}: "
                      f"custom={t_cust:.3f}ms ({kv_bytes/t_cust/1e6:.0f} GB/s)  "
                      f"fi_single={t_single:.3f}ms ({kv_bytes/t_single/1e6:.0f} GB/s)  "
                      f"speedup={t_single/t_cust:.2f}x")


if __name__ == "__main__":
    ok = parity()
    bench()