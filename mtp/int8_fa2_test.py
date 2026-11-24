"""FA2-style INT8 verify kernel: parity + SAME-SESSION A/B vs the live FI fp16
paged-prefill kernel, the HD-split HMMA kernel and the CUDA-core warp kernel.

Geometries: the live per-rank shape (nq8/kvh1 -> NQ_rt=32) and nq16/kvh4 (NQ_rt=16).
All four paths are timed interleaved in one process at the current power state.

Run: venv/bin/python mtp/int8_fa2_test.py
"""
import os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEV = torch.device("cuda", 0)
HD, QL = 512, 4
SM_SCALE = 1.0 / (HD ** 0.5)


def quant_kv(k, v):
    ak = k.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    av = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    sk = (ak / 127.0).half()
    sv = (av / 127.0).half()
    qk = torch.round(k / sk).clamp(-127, 127).to(torch.int8)
    qv = torch.round(v / sv).clamp(-127, 127).to(torch.int8)
    return qk.contiguous(), qv.contiguous(), sk.squeeze(-1).contiguous(), sv.squeeze(-1).contiguous()


def torch_ref(q, k, v, kv_len, nq, nkv):
    g = nq // nkv
    kk = k[:kv_len].repeat_interleave(g, dim=1).float()
    vv = v[:kv_len].repeat_interleave(g, dim=1).float()
    out = []
    for r in range(QL):
        lim = kv_len - QL + r + 1
        s = torch.einsum("hd,khd->hk", q[r].float(), kk[:lim]) * SM_SCALE
        p = torch.softmax(s, dim=-1)
        out.append(torch.einsum("hk,khd->hd", p, vv[:lim]))
    return torch.stack(out)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.flatten(), dim=0).item()


def timeit(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    from exllamav2.ext import exllamav2_ext as ext_c
    import flashinfer
    torch.manual_seed(0)
    EMPTY = torch.empty(0, dtype=torch.int32, device=DEV)
    page = 128
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)

    for (nq, nkv) in ((8, 1), (16, 4)):
        nq_rt = (nq // nkv) * QL
        print(f"\n=== nq={nq} nkv={nkv} q_len={QL} -> NQ_rt={nq_rt} ===", flush=True)
        for kv_len in (4096, 25600, 100000):
            depth = ((kv_len + 127) // 128) * 128
            k = (torch.randn(depth, nkv, HD, device=DEV, dtype=torch.half) * 0.5)
            v = (torch.randn(depth, nkv, HD, device=DEV, dtype=torch.half) * 0.5)
            k[kv_len:] = 0; v[kv_len:] = 0
            k8, v8, ks, vs = quant_kv(k, v)
            k8b, v8b, ksb, vsb = (t.unsqueeze(0) for t in (k8, v8, ks, vs))
            q = (torch.randn(1, QL, nq, HD, device=DEV, dtype=torch.half) * 0.5)
            kvl = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
            ref = torch_ref(q[0], k, v, kv_len, nq, nkv)
            hr = nq * QL

            # live FI fp16 paged prefill
            wrap = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
            npages, pv = depth // page, (kv_len + page - 1) // page
            wrap.plan(
                qo_indptr=torch.tensor([0, QL], dtype=torch.int32, device=DEV),
                paged_kv_indptr=torch.tensor([0, pv], dtype=torch.int32, device=DEV),
                paged_kv_indices=torch.arange(pv, dtype=torch.int32, device=DEV),
                paged_kv_last_page_len=torch.tensor([kv_len - (pv - 1) * page],
                                                    dtype=torch.int32, device=DEV),
                num_qo_heads=nq, num_kv_heads=nkv, head_dim_qk=HD, page_size=page,
                causal=True, pos_encoding_mode="NONE", sm_scale=SM_SCALE,
                q_data_type=torch.half, kv_data_type=torch.half)
            kp, vp = k.view(npages, page, nkv, HD), v.view(npages, page, nkv, HD)
            fi_cos = cos(wrap.run(q[0], (kp, vp)), ref)

            results = {}
            for name in ("fa2", "hmma", "fast"):
                fn_ext = getattr(ext_c, "int8_attn_" + name)
                best = None
                for splits in (64, 128, 256):
                    po = torch.empty(1, splits, hr, HD, device=DEV, dtype=torch.float32)
                    pl = torch.empty(1, splits, hr, 2, device=DEV, dtype=torch.float32)
                    o = torch.empty(1, QL, nq, HD, device=DEV, dtype=torch.half)
                    call = lambda: fn_ext(q, k8b, v8b, ksb, vsb, o, po, pl, kvl,
                                          1, QL, nq, nkv, splits, SM_SCALE, HD, EMPTY)
                    call()
                    c = cos(o[0], ref)
                    ms = timeit(call)
                    if best is None or ms < best[1]:
                        best = (splits, ms, c)
                results[name] = best
                sp, ms, c = best
                gbs = kv_len * nkv * HD * 2 / 1e9 / (ms / 1000)
                print(f"  kv={kv_len:6d} {name:5s} splits={sp:3d} {ms:7.3f} ms "
                      f"({gbs:5.0f} GB/s int8) cos={c:.6f}", flush=True)
                assert c > 0.999, f"{name} parity FAIL kv={kv_len}: cos {c}"

            fi_ms = timeit(lambda: wrap.run(q[0], (kp, vp)))
            fi_gbs = kv_len * nkv * HD * 2 * 2 / 1e9 / (fi_ms / 1000)
            print(f"  kv={kv_len:6d} FI fp16        {fi_ms:7.3f} ms ({fi_gbs:5.0f} GB/s) "
                  f"cos={fi_cos:.6f}", flush=True)
            f2, hm, fa = results["fa2"][1], results["hmma"][1], results["fast"][1]
            print(f"  kv={kv_len:6d} -> fa2 vs FI {fi_ms/f2:.2f}x | vs hmma {hm/f2:.2f}x | "
                  f"vs warp {fa/f2:.2f}x", flush=True)

    # row_map B=2 parity (physical rows swapped, different kv lengths)
    nq, nkv = 8, 1
    kv_len = 4096
    depth = ((kv_len + 127) // 128) * 128
    k = (torch.randn(2, depth, nkv, HD, device=DEV, dtype=torch.half) * 0.5)
    v = (torch.randn(2, depth, nkv, HD, device=DEV, dtype=torch.half) * 0.5)
    k[:, kv_len:] = 0; v[:, kv_len:] = 0
    qa = [quant_kv(k[i], v[i]) for i in range(2)]
    k8b = torch.stack([qa[i][0] for i in range(2)])
    v8b = torch.stack([qa[i][1] for i in range(2)])
    ksb = torch.stack([qa[i][2] for i in range(2)])
    vsb = torch.stack([qa[i][3] for i in range(2)])
    q2 = (torch.randn(2, QL, nq, HD, device=DEV, dtype=torch.half) * 0.5)
    kvl2 = torch.tensor([kv_len, kv_len - 1000], dtype=torch.int32, device=DEV)
    rmap = torch.tensor([1, 0], dtype=torch.int32, device=DEV)
    po = torch.empty(2, 128, nq * QL, HD, device=DEV, dtype=torch.float32)
    pl = torch.empty(2, 128, nq * QL, 2, device=DEV, dtype=torch.float32)
    o = torch.empty(2, QL, nq, HD, device=DEV, dtype=torch.half)
    ext_c.int8_attn_fa2(q2, k8b, v8b, ksb, vsb, o, po, pl, kvl2, 2, QL, nq, nkv,
                        128, SM_SCALE, HD, rmap)
    c0 = cos(o[0], torch_ref(q2[0], k[1], v[1], kv_len, nq, nkv))
    c1 = cos(o[1], torch_ref(q2[1], k[0], v[0], kv_len - 1000, nq, nkv))
    assert c0 > 0.999 and c1 > 0.999, f"row_map parity FAIL: {c0} {c1}"
    print(f"\nrow_map B=2 parity PASS (cos {c0:.6f} / {c1:.6f})", flush=True)
    print("INT8_FA2_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
