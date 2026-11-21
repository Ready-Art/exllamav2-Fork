"""DFlash validation: greedy output parity vs plain MP baseline + acceptance/perf.

DFlash drafts are greedy (argmax of the block-diffusion logits) and the driver's
greedy acceptance compares them against the target's verified argmax, so a correct
implementation emits EXACTLY the plain greedy autoregressive sequence -- any
mismatch means the aux-state / draft-KV mechanics are wrong (positions, taps,
RoPE, mask geometry), not a sampling difference.

Each run gets its OWN subprocess (a second NCCL process-group init in the same
process is not safe on this fork). Run from ~/exl2work:

    python exllamav2-Fork/mtp/dflash_test.py both [window]     # parity + stats
    python exllamav2-Fork/mtp/dflash_test.py dflash [window]   # single run
    python exllamav2-Fork/mtp/dflash_test.py baseline

Env: DFLASH_MODEL_DIR / DFLASH_DRAFT_DIR / DFLASH_MAXSEQ override the defaults.
"""
import os
import subprocess
import sys
import time

import torch

MD = os.environ.get("DFLASH_MODEL_DIR", "models/Gemma-4-31B-IT-EXL2-6.0BPW-HB16")
DRAFT_DIR = os.environ.get("DFLASH_DRAFT_DIR", "models/gemma-4-31B-it-DFlash")
MAXSEQ = int(os.environ.get("DFLASH_MAXSEQ", "4096"))

CASES = [
    ("What is 2+2? Answer in one sentence.", 64),
    ("Explain what a hash map is and when to use one, in three short paragraphs.", 300),
]


def build_prompt(tok, q):
    return tok.encode(f"<bos><|turn>user\n{q}<turn|>\n<|turn>model\n",
                      encode_special_tokens=True)


def make_pg(with_drafter, window):
    from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.tp_mp import TPProcessGroup
    c = ExLlamaV2Config()
    c.model_dir = MD
    c.prepare()
    c.arch_compat_overrides()
    c.max_seq_len = MAXSEQ
    tok = ExLlamaV2Tokenizer(c)
    pg = TPProcessGroup(MD, [20.0] * 4, MAXSEQ, MAXSEQ, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = None
    if with_drafter:
        from exllamav2.dflash_drafter import DFlashDrafter
        drafter = DFlashDrafter(pg, DRAFT_DIR, device=0,
                                num_draft_tokens=window if window > 0 else None)
        pg.enable_mtp(drafter, drafter.num_draft_tokens)
    return pg, tok, drafter


def run_baseline(pg, tok, ids, nmax):
    np_ = ids.shape[-1]
    logits = pg.forward(ids, 0, np_, 1, True)
    nxt = int(torch.argmax(logits[0, -1].float()))
    past = np_
    out_ids = [nxt]
    t0 = time.perf_counter()
    while len(out_ids) < nmax:
        cur = torch.tensor([[nxt]], dtype=torch.long)
        lg = pg.forward(cur, past, 1, 1, True)
        nxt = int(torch.argmax(lg[0, -1].float()))
        out_ids.append(nxt)
        past += 1
    dt = time.perf_counter() - t0
    return out_ids, dt


def run_dflash(pg, tok, drafter, ids, nmax, window):
    from exllamav2.mtp_mp import MTPMPDriver
    np_ = ids.shape[-1]
    t0 = time.perf_counter()
    logits, state, aux = pg.forward_state(ids, last_id_only=True, cpu_logits=True,
                                          return_row_aux=True)
    t_prefill = time.perf_counter() - t0
    drv = MTPMPDriver(pg, drafter, window=drafter.num_draft_tokens,
                      verbose=(os.environ.get("MTP_VERBOSE") == "1"))
    drv.begin(ids, logits, state, aux=aux, aux_p0=0)
    out_ids = []
    t0 = time.perf_counter()
    while len(out_ids) < nmax:
        out_ids.append(drv.step())
    dt = time.perf_counter() - t0
    return out_ids, dt, t_prefill, drv


def single_run(mode, window):
    pg, tok, drafter = make_pg(mode == "dflash", window)
    try:
        for ci, (q, nmax) in enumerate(CASES):
            ids = build_prompt(tok, q)
            if mode == "baseline":
                out_ids, dt = run_baseline(pg, tok, ids, nmax)
                stats = {}
                t_prefill = 0.0
            else:
                out_ids, dt, t_prefill, drv = run_dflash(pg, tok, drafter, ids, nmax, window)
                stats = drv.stats()
                stats["ms_per_round"] = drv.total_ms / max(1, drv.mtp_rounds)
            text = tok.decode(torch.tensor([out_ids]))[0]
            print(f"=== CASE {ci} BEGIN ===", flush=True)
            print(text, flush=True)
            print(f"=== CASE {ci} END ===", flush=True)
            print(f"[stats case {ci}] tokens={len(out_ids)} gen_s={dt:.3f} "
                  f"tok_s={len(out_ids)/dt:.2f} prefill_s={t_prefill:.3f} "
                  f"prompt_tokens={ids.shape[-1]} {stats}", flush=True)
            # Fresh lineage per case (the prompt cache is a tabby-layer feature).
            pg.mtp_reset_cache()
    finally:
        pg.close()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if mode in ("baseline", "dflash"):
        single_run(mode, window)
        return
    # both: one subprocess per run (fresh NCCL init each), then compare.
    outputs = {}
    for m in ("baseline", "dflash"):
        print(f"\n########## {m} run ##########", flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), m, str(window)],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            print(f"*** {m} run FAILED (rc={r.returncode}) ***", flush=True)
            sys.exit(1)
        sys.stderr.write(r.stderr)
        outputs[m] = r.stdout
    ok = True
    for ci in range(len(CASES)):
        def extract(s):
            a = s.index(f"=== CASE {ci} BEGIN ===")
            b = s.index(f"=== CASE {ci} END ===")
            return s[a:b]
        match = extract(outputs["baseline"]) == extract(outputs["dflash"])
        ok &= match
        print(f"case {ci}: {'MATCH' if match else '*** MISMATCH ***'}", flush=True)
    print("\nALL MATCH" if ok else "\nFAILURES PRESENT", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
