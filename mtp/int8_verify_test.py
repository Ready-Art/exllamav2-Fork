"""INT8 verify kernel microbench vs the LIVE FI paged-prefill fp16 kernel.
Synthetic KV (no model): hd512, GQA nq16/kvh4, q_len 4 (W=3 verify window).
Gate (user P4 rule): int8_verify must beat FI in ms BEFORE integration.

Run: venv/bin/python mtp/int8_verify_test.py
"""
import os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("NCCL_P2P_LEVEL", "PXB")

DEV = torch.device("cuda", 0)
NQ, NKV, HD, QL = 16, 4, 512, 4
SM_SCALE = 1.0 / (HD ** 0.5)


def quant_kv(k, v):
    # k/v (depth, kvh, hd) fp16 -> int8 + fp16 scales (per token,head)
    ak = k.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    av = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    sk = (ak / 127.0).half()
    sv = (av / 127.0).half()
    qk = torch.round(k / sk).clamp(-127, 127).to(torch.int8)
    qv = torch.round(v / sv).clamp(-127, 127).to(torch.int8)
    return qk.contiguous(), qv.contiguous(), sk.squeeze(-1).contiguous(), sv.squeeze(-1).contiguous()


def torch_ref(q, k, v, kv_len):
    # q (QL, NQ, HD); k/v (kv_len, NKV, HD). Causal: row r sees keys < kv_len-QL+r+1.
    g = NQ // NKV
    kk = k[:kv_len].repeat_interleave(g, dim=1).float()   # (kv_len, NQ, HD)
    vv = v[:kv_len].repeat_interleave(g, dim=1).float()
    out = []
    for r in range(QL):
        lim = kv_len - QL + r + 1
        s = torch.einsum("hd,khd->hk", q[r].float(), kk[:lim]) * SM_SCALE
        p = torch.softmax(s, dim=-1)
        out.append(torch.einsum("hk,khd->hd", p, vv[:lim]))
    return torch.stack(out)


def main():
    from exllamav2.ext import exllamav2_ext as ext_c
    import flashinfer
    torch.manual_seed(0)
    EMPTY = torch.empty(0, dtype=torch.int32, device=DEV)
    page = 128
    for kv_len in (4096, 16384, 25600):
        depth = ((kv_len + 127) // 128) * 128
        k = (torch.randn(depth, NKV, HD, device=DEV, dtype=torch.half) * 0.5)
        v = (torch.randn(depth, NKV, HD, device=DEV, dtype=torch.half) * 0.5)
        k[kv_len:] = 0; v[kv_len:] = 0
        k8, v8, ks, vs = quant_kv(k, v)
        q = (torch.randn(1, QL, NQ, HD, device=DEV, dtype=torch.half) * 0.5)
        kvl = torch.tensor([kv_len], dtype=torch.int32, device=DEV)

        ref = torch_ref(q[0], k, v, kv_len)

        # FI paged prefill fp16 (the live verify kernel)
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
        wrap = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
        npages = depth // page
        pv = (kv_len + page - 1) // page
        wrap.plan(
            qo_indptr=torch.tensor([0, QL], dtype=torch.int32, device=DEV),
            paged_kv_indptr=torch.tensor([0, pv], dtype=torch.int32, device=DEV),
            paged_kv_indices=torch.arange(pv, dtype=torch.int32, device=DEV),
            paged_kv_last_page_len=torch.tensor([kv_len - (pv - 1) * page],
                                                dtype=torch.int32, device=DEV),
            num_qo_heads=NQ, num_kv_heads=NKV, head_dim_qk=HD, page_size=page,
            causal=True, pos_encoding_mode="NONE", sm_scale=SM_SCALE,
            q_data_type=torch.half, kv_data_type=torch.half)
        kp = k.view(npages, page, NKV, HD)
        vp = v.view(npages, page, NKV, HD)
        fi_out = wrap.run(q[0], (kp, vp))
        fi_cos = torch.nn.functional.cosine_similarity(
            fi_out.float().flatten(), ref.flatten(), dim=0).item()

        best = None
        for splits in (64, 128, 256, 512):
            po = torch.empty(1, splits, NQ * QL, HD, device=DEV, dtype=torch.float32)
            pl = torch.empty(1, splits, NQ * QL, 2, device=DEV, dtype=torch.float32)
            o = torch.empty(1, QL, NQ, HD, device=DEV, dtype=torch.half)
            ext_c.int8_verify(q, k8.unsqueeze(0), v8.unsqueeze(0), ks.unsqueeze(0),
                              vs.unsqueeze(0), o, po, pl, kvl, 1, QL, NQ, NKV,
                              splits, SM_SCALE, HD, EMPTY)
            cos = torch.nn.functional.cosine_similarity(
                o[0].float().flatten(), ref.flatten(), dim=0).item()
            for _ in range(10):
                ext_c.int8_verify(q, k8.unsqueeze(0), v8.unsqueeze(0), ks.unsqueeze(0),
                                  vs.unsqueeze(0), o, po, pl, kvl, 1, QL, NQ, NKV,
                                  splits, SM_SCALE, HD, EMPTY)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                ext_c.int8_verify(q, k8.unsqueeze(0), v8.unsqueeze(0), ks.unsqueeze(0),
                                  vs.unsqueeze(0), o, po, pl, kvl, 1, QL, NQ, NKV,
                                  splits, SM_SCALE, HD, EMPTY)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 100 * 1000
            gbs = kv_len * NKV * HD * 2 / 1e9 / (ms / 1000)
            if best is None or ms < best[1]:
                best = (splits, ms, cos, gbs)
        # Fast variant (multi-warp CTAs + U=4 unroll) -- the INT8-primary reader.
        best_fast = None
        for splits in (64, 128, 256, 512):
            po = torch.empty(1, splits, NQ * QL, HD, device=DEV, dtype=torch.float32)
            pl = torch.empty(1, splits, NQ * QL, 2, device=DEV, dtype=torch.float32)
            o = torch.empty(1, QL, NQ, HD, device=DEV, dtype=torch.half)
            ext_c.int8_attn_fast(q, k8.unsqueeze(0), v8.unsqueeze(0), ks.unsqueeze(0),
                                 vs.unsqueeze(0), o, po, pl, kvl, 1, QL, NQ, NKV,
                                 splits, SM_SCALE, HD, EMPTY)
            cos = torch.nn.functional.cosine_similarity(
                o[0].float().flatten(), ref.flatten(), dim=0).item()
            assert cos > 0.999, f"int8_attn_fast parity FAIL kv={kv_len}: cos {cos}"
            for _ in range(10):
                ext_c.int8_attn_fast(q, k8.unsqueeze(0), v8.unsqueeze(0), ks.unsqueeze(0),
                                     vs.unsqueeze(0), o, po, pl, kvl, 1, QL, NQ, NKV,
                                     splits, SM_SCALE, HD, EMPTY)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                ext_c.int8_attn_fast(q, k8.unsqueeze(0), v8.unsqueeze(0), ks.unsqueeze(0),
                                     vs.unsqueeze(0), o, po, pl, kvl, 1, QL, NQ, NKV,
                                     splits, SM_SCALE, HD, EMPTY)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 100 * 1000
            gbs = kv_len * NKV * HD * 2 / 1e9 / (ms / 1000)
            if best_fast is None or ms < best_fast[1]:
                best_fast = (splits, ms, cos, gbs)
        for _ in range(10):
            wrap.run(q[0], (kp, vp))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            wrap.run(q[0], (kp, vp))
        torch.cuda.synchronize()
        fi_ms = (time.perf_counter() - t0) / 100 * 1000
        fi_gbs = kv_len * NKV * HD * 2 * 2 / 1e9 / (fi_ms / 1000)
        sp, ms, cos, gbs = best
        spf, msf, cosf, gbsf = best_fast
        print(f"kv={kv_len:6d} FI fp16 {fi_ms:.3f} ms ({fi_gbs:.0f} GB/s, cos {fi_cos:.5f}) | "
              f"int8 splits={sp} {ms:.3f} ms ({gbs:.0f} GB/s, cos {cos:.5f}) | "
              f"speedup {fi_ms/ms:.2f}x", flush=True)
        print(f"         FAST splits={spf} {msf:.3f} ms ({gbsf:.0f} GB/s, cos {cosf:.5f}) | "
              f"vs FI {fi_ms/msf:.2f}x | vs old int8 {ms/msf:.2f}x", flush=True)
        assert cos > 0.999, f"int8_verify parity FAIL kv={kv_len}: cos {cos}"
    # Batched row_map parity: B=2 over physical rows [1, 0] (subset-slot geometry).
    kv_len = 4096
    depth = ((kv_len + 127) // 128) * 128
    k = (torch.randn(2, depth, NKV, HD, device=DEV, dtype=torch.half) * 0.5)
    v = (torch.randn(2, depth, NKV, HD, device=DEV, dtype=torch.half) * 0.5)
    k[:, kv_len:] = 0; v[:, kv_len:] = 0
    k8, v8, ks, vs = quant_kv(k[0], v[0])
    k8b = torch.stack([k8, quant_kv(k[1], v[1])[0]])
    v8b = torch.stack([v8, quant_kv(k[1], v[1])[1]])
    ksb = torch.stack([ks, quant_kv(k[1], v[1])[2]])
    vsb = torch.stack([vs, quant_kv(k[1], v[1])[3]])
    q2 = (torch.randn(2, QL, NQ, HD, device=DEV, dtype=torch.half) * 0.5)
    kvl2 = torch.tensor([kv_len, kv_len - 1000], dtype=torch.int32, device=DEV)
    rmap = torch.tensor([1, 0], dtype=torch.int32, device=DEV)
    po = torch.empty(2, 128, NQ * QL, HD, device=DEV, dtype=torch.float32)
    pl = torch.empty(2, 128, NQ * QL, 2, device=DEV, dtype=torch.float32)
    o = torch.empty(2, QL, NQ, HD, device=DEV, dtype=torch.half)
    ext_c.int8_attn_fast(q2, k8b, v8b, ksb, vsb, o, po, pl, kvl2, 2, QL, NQ, NKV,
                         128, SM_SCALE, HD, rmap)
    ref0 = torch_ref(q2[0], k[1], v[1], kv_len)
    ref1 = torch_ref(q2[1], k[0], v[0], kv_len - 1000)
    c0 = torch.nn.functional.cosine_similarity(o[0].float().flatten(), ref0.flatten(), dim=0).item()
    c1 = torch.nn.functional.cosine_similarity(o[1].float().flatten(), ref1.flatten(), dim=0).item()
    assert c0 > 0.999 and c1 > 0.999, f"row_map parity FAIL: {c0} {c1}"
    print(f"row_map B=2 parity PASS (cos {c0:.5f} / {c1:.5f})", flush=True)
    print("INT8_VERIFY_MICROBENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
