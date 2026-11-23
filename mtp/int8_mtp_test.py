"""INT8 MTP parity + agreement harness. ONE mode per fresh subprocess (a second
process-group init in the same process crashes). KV mode is fixed by the environment
at shadow allocation (EXL2_MTP_INT8_KV=1 -> INT8-primary, VRAM-saving):

  scalar: max_batch=1 pg -- GEMV kernel parity, draft ids (fp16 weights -> INT8 weights),
          GEMV microbench.
  batch:  max_batch=2 pg -- GEMV kernel parity, int8_decode row_map parity (KV=1 only),
          batched draft ids.

Run twice (KV=0 and KV=1) and compare the printed fp16-weight draft ids across runs
to measure the KV-quantization effect on drafts. Within a run, the weight toggle
measures the INT8-weight effect.

Run: venv/bin/python mtp/int8_mtp_test.py <model_dir> <ctx> <scalar|batch>
"""
import os, sys, time, torch
os.environ.setdefault("EXL2_MTP_INT8_KV", "0")
os.environ["EXL2_MTP_INT8_W"] = "0"    # start fp16; toggled below
os.environ.setdefault("EXL2_MTP_DRAFT_GRAPH", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "PXB")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DRAFT = "/home/westfox35/exl2work/models/gemma-4-31B-it-assistant"
W = 3


def build_prompt(tok, n):
    t = ("The migration patterns of arctic terns span pole to pole, navigating by sun, "
         "stars and magnetic field. Register allocation maps virtual registers onto the "
         "physical register file via graph colouring or linear scan. ")
    ids = tok.encode("<bos><|turn>user\n", encode_special_tokens=True)
    while ids.shape[-1] < n:
        ids = torch.cat([ids, tok.encode(t, encode_special_tokens=False)], dim=-1)
    return ids[:, :n]


def rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9))


def gemv_parity(d, ext_c):
    ws = [("pre", d.pre_projection), ("post", d.post_projection), ("head", d.head)]
    for i, L in enumerate(d.layers):
        ws += [(f"L{i}.q", L.q_proj), (f"L{i}.o", L.o_proj),
               (f"L{i}.gate", L.gate), (f"L{i}.up", L.up), (f"L{i}.down", L.down)]
    worst_k, worst_q = 0.0, 0.0
    for name, w in ws:
        w8, s = d._quant_rows(w)
        deq = w8.float() * s.float().unsqueeze(1)   # exact dequant reference
        for m in (1, 2, 4):
            x = (torch.randn(m, w.shape[1], device=d.dev, dtype=torch.half) * 0.1)
            got = ext_c.mtp_int8_gemm(x, w8, s)
            rk = rel_l2(got, x @ deq.to(torch.half).t())
            rq = rel_l2(got, x @ w.t())
            worst_k = max(worst_k, rk)
            worst_q = max(worst_q, rq)
            assert rk < 2e-3, f"GEMV KERNEL FAIL {name} m={m}: rel_l2={rk}"
    print(f"[1] mtp_int8_gemm kernel parity ALL PASS (worst vs dequant {worst_k:.6f}; "
          f"worst quantization error vs fp16 {worst_q:.5f})", flush=True)


def set_weights_int8(d, on):
    d._int8_w = on
    if on:
        d._quantize_weights()
        if d.sparse is not None and d.sparse.subset is not None:
            d.sparse.int8 = True
            d.sparse.subset8, d.sparse.subset_s = d._quant_rows(d.sparse.subset)
    elif d.sparse is not None:
        d.sparse.int8 = False
        d.sparse.subset8 = d.sparse.subset_s = None


def main():
    from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.tp_mp import TPProcessGroup
    from exllamav2.mtp_drafter import Gemma4AssistantDrafter
    from exllamav2.ext import exllamav2_ext as ext_c

    MD, CTX, MODE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    KV = os.environ["EXL2_MTP_INT8_KV"]
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    maxseq = CTX + 1024
    c.max_seq_len = maxseq; c.chunk_size = 2048; c.max_input_len = 2048
    tok = ExLlamaV2Tokenizer(c)
    ids = build_prompt(tok, CTX)
    mb = 1 if MODE == "scalar" else 2
    pg = TPProcessGroup(MD, [22.0] * 4, maxseq, maxseq, "ExLlamaV2Cache_SWA", max_batch=mb)
    pg.start()
    d = Gemma4AssistantDrafter(pg, DRAFT, device=0, num_draft_tokens=W,
                               sparse_head_columns=16384)
    pg.enable_mtp(d, W)
    gemv_parity(d, ext_c)
    print(f"[mode] {MODE} KV={KV} i8p={bool((pg._mtp_kv or {}).get('i8p'))}", flush=True)

    if MODE == "scalar":
        lg, stt = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        pg.gather_kv_mtp()
        d.gather_target(pg.cache)
        pos = pg.cache.current_seq_len
        bonus = lg[:, -1:, :].argmax(dim=-1).cpu()
        base_state = stt[:, -1:, :].float().cpu()

        def draft():
            d.invalidate_draft_graph()
            return d.draft_window(bonus, base_state, pos, W)[0].tolist()

        ref = draft()
        set_weights_int8(d, True)
        both = draft()
        rf, bf = ref[0], both[0]
        ab = sum(a == b for a, b in zip(rf, bf))
        print(f"[3] scalar drafts: fp16w={rf} int8w={bf} agree={ab}/{W}", flush=True)

        for name, w in (("head", d.head), ("gate", d.layers[0].gate)):
            w8, s = d._quant_rows(w)
            x = torch.randn(1, w.shape[1], device=d.dev, dtype=torch.half)
            for fn, label in ((lambda: x @ w.t(), "fp16"),
                              (lambda: ext_c.mtp_int8_gemm(x, w8, s), "int8")):
                for _ in range(20): fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(200): fn()
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / 200 * 1000
                gb = w.numel() * (1 if label == "int8" else 2) / 1e9
                print(f"[4] {name} {label}: {ms:.3f} ms  {gb/ms*1000:.0f} GB/s", flush=True)
        ok = ab >= W - 1
    else:
        ids2 = build_prompt(tok, max(512, CTX // 3))
        lg0, st0 = pg.forward_state_row(0, ids, last_id_only=True)
        lg1, st1 = pg.forward_state_row(1, ids2, last_id_only=True)
        pg.gather_kv_mtp_batch([0, 1])
        d.gather_target_batch([0, 1])
        st = pg._mtp_kv
        if "kf8" in st:
            nq, hd, nkv = d.num_q_heads, d.global_head_dim, d.num_global_kv_heads
            depth = st["kf8"].shape[1]
            q = torch.randn(2, nq, hd, device=d.dev, dtype=torch.half) * 0.1
            kvl = torch.tensor([pg.cache.seq_lens[0], pg.cache.seq_lens[1]],
                               dtype=torch.int32, device=d.dev)
            rmap = torch.tensor([0, 1], dtype=torch.int32, device=d.dev)
            (po, pl), splits = d._i8_partials(2, nq, hd, depth)
            o = torch.empty(2, nq, hd, device=d.dev, dtype=torch.half)
            ext_c.int8_decode(q, st["kf8"], st["vf8"], st["kfs"], st["vfs"],
                              o, po, pl, kvl, 2, nq, nkv, splits, 1.0, hd, rmap)
            for b in range(2):
                (po1, pl1), sp1 = d._i8_partials(1, nq, hd, depth)
                o1 = torch.empty(1, nq, hd, device=d.dev, dtype=torch.half)
                ext_c.int8_decode(q[b:b+1], st["kf8"], st["vf8"], st["kfs"], st["vfs"],
                                  o1, po1, pl1, kvl[b:b+1], 1, nq, nkv, sp1, 1.0, hd,
                                  torch.tensor([b], dtype=torch.int32, device=d.dev))
                r = rel_l2(o[b:b+1], o1)
                assert r < 1e-3, f"row_map parity FAIL row {b}: {r}"
            rmap_rev = torch.tensor([1, 0], dtype=torch.int32, device=d.dev)
            o_rev = torch.empty(2, nq, hd, device=d.dev, dtype=torch.half)
            ext_c.int8_decode(q, st["kf8"], st["vf8"], st["kfs"], st["vfs"],
                              o_rev, po, pl, kvl, 2, nq, nkv, splits, 1.0, hd, rmap_rev)
            for b in range(2):
                (po1, pl1), sp1 = d._i8_partials(1, nq, hd, depth)
                o1 = torch.empty(1, nq, hd, device=d.dev, dtype=torch.half)
                ext_c.int8_decode(q[b:b+1], st["kf8"], st["vf8"], st["kfs"], st["vfs"],
                                  o1, po1, pl1, kvl[b:b+1], 1, nq, nkv, sp1, 1.0, hd,
                                  torch.tensor([1 - b], dtype=torch.int32, device=d.dev))
                r = rel_l2(o_rev[b:b+1], o1)
                assert r < 1e-3, f"reversed row_map parity FAIL row {b}: {r}"
            print("[2] int8_decode row_map parity ALL PASS", flush=True)

        pos_b = [pg.cache.seq_lens[0], pg.cache.seq_lens[1]]
        bt = torch.cat([lg0[:, -1:, :].argmax(-1), lg1[:, -1:, :].argmax(-1)], dim=0).cpu()
        bs = torch.cat([st0[:, -1:, :].float().cpu(), st1[:, -1:, :].float().cpu()], dim=0)

        def draft():
            d.invalidate_draft_graph()
            return d.draft_window_batch(bt, bs, pos_b, W).tolist()

        ref = draft()
        set_weights_int8(d, True)
        both = draft()
        ab = sum(a == b for ra, rb in zip(ref, both) for a, b in zip(ra, rb))
        print(f"[3b] batch drafts: fp16w={ref} int8w={both} agree={ab}/{2*W}", flush=True)
        ok = ab >= 2*W - 2

    print(f"VERDICT: {'PASS' if ok else 'CHECK'} (int8-weight draft agreement within 1 token/row)",
          flush=True)
    pg.close()


if __name__ == "__main__":
    main()
