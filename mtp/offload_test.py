"""Prompt-cache RAM offload correctness harness (fork MP+MTP, opt-in tiering).

Drives a TPProcessGroup directly (no server) and proves the offload/restore
mechanics preserve the EXACT K/V state:

  scalar  batch-1 path: prefill A + generate, fingerprint the row K/V + MTP shadow,
          offload to RAM, overwrite the row with a different prompt B, restore A,
          assert byte-identical K/V + shadow + bookkeeping, then continue generation
          (greedy agreement vs a no-offload reference + coherence). Also a partial
          (truncated) restore.
  batch   max_batch=4: fill 4 rows, decode rounds, finish; offload rows 0/1, reset,
          refill, restore row 0 full + row 1 partial; assert K/V fingerprints, then
          arm the rows and run batched rounds (coherence + incremental gather).

KV equality is the rigorous proof: identical bytes + identical bookkeeping means the
forward is the same function. Greedy token agreement is reported (fp16 atomicAdd
near-ties can flip a token run-to-run; >=90% agreement + coherent text is the pass
bar, byte equality must be EXACT).

Usage (from ~/exl2work, fresh subprocess per phase -- 2nd pg init crashes):
  venv/bin/python exllamav2-Fork/mtp/offload_test.py scalar
  venv/bin/python exllamav2-Fork/mtp/offload_test.py batch
Env: NGPU (default 4), DRAFT_DIR (drafter checkpoint), EXL2_MTP_DRAFT_GRAPH=1 optional.
"""
import os, sys, time, json, torch

MD = "models/Gemma-4-31B-IT-EXL2-6.0BPW-HB16"
_dflt = "models/gemma-4-31B-it-assistant"
_hf = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
DRAFT_DIR = os.environ.get("DRAFT_DIR", _dflt if os.path.isdir(_dflt) else _hf)
NGPU = int(os.environ.get("NGPU", "4"))
PHASE = sys.argv[1] if len(sys.argv) > 1 else "scalar"
W = 3
MAXSEQ = 2048          # page-aligned; prompts ~1300-1500 + generation headroom
NGEN = 40

from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup
from exllamav2.mtp_mp import MTPMPDriver, MTPMPBatchDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter
from exllamav2.generator.sampler import ExLlamaV2Sampler

GREEDY = ExLlamaV2Sampler.Settings()
GREEDY.top_k = 1

FILLER = ("The history of computing is a long and intricate story that begins with "
          "mechanical calculators and passes through punched cards, vacuum tubes, "
          "transistors and integrated circuits. Each layer hid the complexity of the "
          "one below it, and each new layer created new kinds of complexity. ")


def build_prompt(tok, n, tag):
    ids = tok.encode(f"<bos><|turn>user\n{tag} ", encode_special_tokens=True)
    while ids.shape[-1] < n:
        ids = torch.cat([ids, tok.encode(FILLER, encode_special_tokens=False)], dim=-1)
    ids = ids[:, :n]
    return torch.cat([ids, tok.encode("<turn|>\n<|turn>model\n",
                                      encode_special_tokens=True)], dim=-1)


def _sum16(t):
    return int(t.contiguous().view(torch.int16).to(torch.int64).sum().item())


def fingerprint(pg, row, pos, wb):
    """Deterministic per-layer checksums of a row's committed K/V + the MTP shadow."""
    mc = pg.cache
    ring = mc.kv_state_size
    n = max(0, min(pos - wb, ring))
    fp = {}
    for l in range(len(mc.key_states)):
        kc = mc.key_states[l]
        if kc is None:
            continue
        rows = n if kc.shape[1] == ring else pos
        fp[("k", l)] = _sum16(kc[row, :rows])
        fp[("v", l)] = _sum16(mc.value_states[l][row, :rows])
    st = pg._mtp_kv
    if st is not None:
        if "fls" in st:
            fl, sl, swb = st["fls"][row], st["sls"][row], st["wbs"][row]
            sn = max(0, min(sl - swb, ring))
            fp[("skf",)] = _sum16(st["kf"][row, :fl])
            fp[("svf",)] = _sum16(st["vf"][row, :fl])
            fp[("sks",)] = _sum16(st["ks"][row, :sn])
            fp[("svs",)] = _sum16(st["vs"][row, :sn])
            fp["front"] = (fl, sl, swb)
        else:
            fl, sl, swb = st["fl"], st["sl"], st["wb"]
            sn = max(0, min(sl - swb, ring))
            fp[("skf",)] = _sum16(st["kf"][:fl])
            fp[("svf",)] = _sum16(st["vf"][:fl])
            fp[("sks",)] = _sum16(st["ks"][:sn])
            fp[("svs",)] = _sum16(st["vs"][:sn])
            fp["front"] = (fl, sl, swb)
    return fp


def fp_equal(a, b, upto=None):
    """Compare fingerprints; upto limits full-layer rows to a prefix (partial restore).
    SWA/shadow checksums are prefix-dependent, so for partial restores we recompute
    the reference at the truncated position instead (caller passes matching refs)."""
    if a.keys() != b.keys():
        return False, "key set differs"
    for k in a:
        if a[k] != b[k]:
            return False, f"{k}: {a[k]} != {b[k]}"
    return True, ""


def load(max_batch):
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = MAXSEQ
    tok = ExLlamaV2Tokenizer(c)
    pg = TPProcessGroup(MD, [22.0] * NGPU, MAXSEQ, MAXSEQ, "ExLlamaV2Cache_SWA",
                        max_batch=max_batch)
    pg.start()
    drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=W)
    pg.enable_mtp(drafter, W)
    return pg, drafter, tok


def gen_scalar(pg, drafter, tok, ids, n):
    logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
    drv = MTPMPDriver(pg, drafter, window=W)
    drv.begin(ids, logits, state)
    out = []
    seq = ids.clone()
    for _ in range(n):
        t = drv.step(GREEDY, tok, seq)
        out.append(t)
        seq = torch.cat([seq, torch.tensor([[t]], dtype=torch.long)], dim=-1)
    return out, seq, drv


def scalar_phase():
    pg, drafter, tok = load(1)
    ok = True
    ids_a = build_prompt(tok, 1300, "Document alpha.")
    ids_b = build_prompt(tok, 900, "Document beta, a different subject entirely.")

    # --- generate 40 tokens on A; fingerprint + offload the pos40 state FIRST
    #     (before any further generation shifts the SWA ring), then generate the
    #     40-token REFERENCE continuation forward from pos40. ---
    out1, seq1, drv = gen_scalar(pg, drafter, tok, ids_a, NGEN)
    pos40 = pg.cache.current_seq_len
    wb_a = pg.cache.window_beg
    fp_a = fingerprint(pg, 0, pos40, wb_a)
    t0 = time.perf_counter()
    bundle = pg.offload_row(0, pos=pos40, wb=wb_a, entry_id="scalar-a")
    off_ms = (time.perf_counter() - t0) * 1000
    assert bundle is not None, "offload returned None"
    print(f"[scalar] A: prompt {ids_a.shape[-1]}, pos40={pos40}, wb={wb_a}; "
          f"offload {bundle['bytes']/2**20:.1f} MB in {off_ms:.0f} ms", flush=True)
    ref, seq_ref, _ = gen_scalar_continue(pg, drafter, tok, seq1, NGEN)
    pos80 = pg.cache.current_seq_len
    print(f"[scalar] reference continuation generated (pos80={pos80})", flush=True)

    # --- overwrite with B, restore A ---
    pg.mtp_reset_cache()
    gen_scalar(pg, drafter, tok, ids_b, 8)   # overwrites row 0
    fp_b = fingerprint(pg, 0, pg.cache.current_seq_len, pg.cache.window_beg)
    same, _ = fp_equal(fp_a, fp_b)
    assert not same, "row B fingerprint equals A -- overwrite did not happen"

    t0 = time.perf_counter()
    pg.restore_row(0, bundle, attach_snapshot=True)
    res_ms = (time.perf_counter() - t0) * 1000
    fp_r = fingerprint(pg, 0, pg.cache.current_seq_len, pg.cache.window_beg)
    same, why = fp_equal(fp_a, fp_r)
    print(f"[scalar] restore in {res_ms:.0f} ms; K/V+shadow byte-equal: {same} {why}",
          flush=True)
    ok &= same
    assert pg.cache.current_seq_len == pos40 and pg.cache.window_beg == wb_a
    assert pg.cache.seq_lens[0] == pos40

    # --- continue from the restored state; greedy agreement vs reference ---
    pg.mtp_set_seq_len(pos40 - 1)
    bl, bs = pg.forward_state(seq1[:, -1:], last_id_only=True, cpu_logits=True)
    drv2 = MTPMPDriver(pg, drafter, window=W)
    drv2.begin(seq1, bl, bs)
    out2 = []
    seq2 = seq1.clone()
    for _ in range(NGEN):
        t = drv2.step(GREEDY, tok, seq2)
        out2.append(t)
        seq2 = torch.cat([seq2, torch.tensor([[t]], dtype=torch.long)], dim=-1)
    agree = sum(a == b for a, b in zip(ref, out2)) / NGEN
    text = tok.decode(torch.tensor([out2]))
    degen = text.count("la la") > 10 or text.count("inging") > 5
    print(f"[scalar] continuation greedy agreement {agree*100:.0f}% ; "
          f"degenerate={degen}; tail={text[-60:]!r}", flush=True)
    ok &= agree >= 0.9 and not degen

    # --- partial (truncated) restore: restore-at-p must equal full-restore-then-
    #     truncate-to-p (two independent code paths producing the same prefix state) ---
    pg.mtp_reset_cache()
    gen_scalar(pg, drafter, tok, ids_b, 8)
    p = pos40 - 200
    pg.restore_row(0, bundle, attach_snapshot=False)   # full restore at pos40
    pg.truncate_cache(p)                               # in-place truncation
    fp_trunc = fingerprint(pg, 0, p, wb_a)
    pg.restore_row(0, bundle, p=p, attach_snapshot=False)   # direct partial restore
    fp_part = fingerprint(pg, 0, p, wb_a)
    same, why = fp_equal(fp_trunc, fp_part)
    print(f"[scalar] partial restore p={p}; prefix byte-equal: {same} {why}", flush=True)
    ok &= same
    assert pg.cache.current_seq_len == p
    # continue from the truncated prefix with a divergent suffix
    suffix = tok.encode(" A new conclusion follows.", encode_special_tokens=False)
    seqp = torch.cat([seq1[:, :p], suffix], dim=-1)
    lg, st = pg.forward_state(suffix, last_id_only=True, cpu_logits=True)
    drvp = MTPMPDriver(pg, drafter, window=W)
    drvp.begin(seqp, lg, st)
    outp = [drvp.step(GREEDY, tok, seqp) for _ in range(16)]
    textp = tok.decode(torch.tensor([outp]))
    degenp = textp.count("la la") > 10 or textp.count("inging") > 5
    print(f"[scalar] partial continuation degenerate={degenp}; tail={textp[-60:]!r}",
          flush=True)
    ok &= not degenp

    pg.drop_offload_bundle(bundle)
    pg.close()
    print(json.dumps({"phase": "scalar", "PASS": bool(ok)}), flush=True)
    sys.exit(0 if ok else 1)


def gen_scalar_continue(pg, drafter, tok, seq, n):
    """Continue generation from the CURRENT cache position (seq tracks the ids)."""
    pg.mtp_set_seq_len(seq.shape[-1] - 1)
    bl, bs = pg.forward_state(seq[:, -1:], last_id_only=True, cpu_logits=True)
    drv = MTPMPDriver(pg, drafter, window=W)
    drv.begin(seq, bl, bs)
    out = []
    for _ in range(n):
        t = drv.step(GREEDY, tok, seq)
        out.append(t)
        seq = torch.cat([seq, torch.tensor([[t]], dtype=torch.long)], dim=-1)
    return out, seq, drv


def batch_phase():
    pg, drafter, tok = load(4)
    ok = True
    driver = MTPMPBatchDriver(pg, drafter, window=W)
    prompts = [build_prompt(tok, 1200 + 100 * i, f"Batch document {i}.") for i in range(4)]
    lineages = []
    for r, ids in enumerate(prompts):
        lg, st = pg.forward_state_row(r, ids, last_id_only=True)
        driver.begin_row(r, ids, lg, st, settings=GREEDY, tokenizer=tok,
                         seq_ids=ids.clone())
    # a few batched rounds so every row advances + gathers
    for _ in range(6):
        active = list(range(4))
        smap = {r: GREEDY for r in active}
        seqmap = {r: prompts[r].clone() for r in active}
        # track emitted ids per row for lineage fidelity
        toks = driver.step_batch(active, smap, tok, seqmap, {})
        for r in active:
            for t in toks.get(r, ()):
                prompts[r] = torch.cat(
                    [prompts[r], torch.tensor([[t]], dtype=torch.long)], dim=-1)
    for r in range(4):
        driver.finish_row(r)
        lineages.append((prompts[r].clone(), pg.cache.seq_lens[r], pg.cache.window_begs[r]))
    fp0 = fingerprint(pg, 0, lineages[0][1], lineages[0][2])
    fp1 = fingerprint(pg, 1, lineages[1][1], lineages[1][2])
    print(f"[batch] filled 4 rows; pos={[l[1] for l in lineages]}", flush=True)

    # offload rows 0 and 1, reset, refill with different prompts
    b0 = pg.offload_row(0, pos=lineages[0][1], wb=lineages[0][2], entry_id="batch-0")
    b1 = pg.offload_row(1, pos=lineages[1][1], wb=lineages[1][2], entry_id="batch-1")
    assert b0 is not None and b1 is not None
    pg.mtp_reset_row(0); pg.mtp_reset_row(1)
    for r in (0, 1):
        nid = build_prompt(tok, 700, f"Replacement prompt {r}.")
        lg, st = pg.forward_state_row(r, nid, last_id_only=True)
        driver.begin_row(r, nid, lg, st, settings=GREEDY, tokenizer=tok, seq_ids=nid.clone())
    driver.finish_row(0); driver.finish_row(1)

    # full restore row 0
    t0 = time.perf_counter()
    pg.restore_row(0, b0, attach_snapshot=True)
    ms = (time.perf_counter() - t0) * 1000
    fp0r = fingerprint(pg, 0, pg.cache.seq_lens[0], pg.cache.window_begs[0])
    same0, why0 = fp_equal(fp0, fp0r)
    print(f"[batch] row0 full restore {ms:.0f} ms; byte-equal: {same0} {why0}", flush=True)
    ok &= same0
    # partial restore row 1: restore-at-p == full-restore-then-truncate-to-p
    p1 = lineages[1][1] - 150
    pg.restore_row(1, b1, attach_snapshot=False)       # full
    pg.truncate_row(1, p1)                             # in-place truncation
    fp1trunc = fingerprint(pg, 1, p1, lineages[1][2])
    pg.restore_row(1, b1, p=p1, attach_snapshot=False)  # direct partial
    fp1part = fingerprint(pg, 1, p1, lineages[1][2])
    same1, why1 = fp_equal(fp1trunc, fp1part)
    print(f"[batch] row1 partial restore p={p1}; byte-equal: {same1} {why1}", flush=True)
    ok &= same1

    # arm restored rows + run batched rounds (incremental gather must work)
    ids0 = lineages[0][0]
    pg.mtp_set_seq_len_row(0, ids0.shape[-1] - 1)
    lg0, st0 = pg.forward_state_row(0, ids0[:, -1:], last_id_only=True)
    driver.begin_row(0, ids0, lg0, st0, settings=GREEDY, tokenizer=tok, seq_ids=ids0.clone())
    ids1 = torch.cat([lineages[1][0][:, :p1],
                      tok.encode(" Revised ending.", encode_special_tokens=False)], dim=-1)
    lg1, st1 = pg.forward_state_row(1, ids1[:, p1:], last_id_only=True)
    driver.begin_row(1, ids1, lg1, st1, settings=GREEDY, tokenizer=tok, seq_ids=ids1.clone())
    seqs = {0: ids0.clone(), 1: ids1.clone()}
    emitted = {0: [], 1: []}
    for _ in range(8):
        active = [0, 1]
        toks = driver.step_batch(active, {r: GREEDY for r in active}, tok, seqs, {})
        for r in active:
            for t in toks.get(r, ()):
                emitted[r].append(t)
                seqs[r] = torch.cat([seqs[r], torch.tensor([[t]], dtype=torch.long)], dim=-1)
    for r in (0, 1):
        driver.finish_row(r)
        text = tok.decode(torch.tensor([emitted[r]])) if emitted[r] else ""
        degen = text.count("la la") > 10 or text.count("inging") > 5
        print(f"[batch] row{r} post-restore {len(emitted[r])} tok; degenerate={degen}; "
              f"tail={text[-60:]!r}", flush=True)
        ok &= len(emitted[r]) > 0 and not degen

    pg.drop_offload_bundle(b0)
    pg.drop_offload_bundle(b1)
    pg.close()
    print(json.dumps({"phase": "batch", "PASS": bool(ok)}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if PHASE == "scalar":
        scalar_phase()
    elif PHASE == "batch":
        batch_phase()
    else:
        print(f"unknown phase {PHASE}")
        sys.exit(2)
