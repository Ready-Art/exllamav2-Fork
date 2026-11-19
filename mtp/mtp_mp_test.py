"""MP-backend MTP validation: output parity vs plain baseline (both greedy).
Mirrors mtp/mtp_stream_test.py but over the multiprocess TPProcessGroup backend.
Each run gets its OWN fresh process group (clean cache), matching the SP test.
Run from ~/exl2work:  python exllamav2-fork/mtp/mtp_mp_test.py [window]
"""
import os, sys, time, torch

MD = "models/gemma-4-31b-exl2-6.0bpw-hb16-2gpu"
DRAFT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
MAXSEQ = 4096
WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 6

from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup, MTPMPDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter


def build_prompt(tok, q):
    return tok.encode(f"<bos><|turn>user\n{q}<turn|>\n<|turn>model\n",
                      encode_special_tokens=True)


CASES = [
    ("What is 2+2?", 100),
    ("Explain what a hash map is and when to use one, in three short paragraphs.", 300),
]


def make_pg(use_mtp):
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = MAXSEQ
    tok = ExLlamaV2Tokenizer(c)
    pg = TPProcessGroup(MD, [20.0] * 4, MAXSEQ, MAXSEQ, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = None
    if use_mtp:
        drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
        pg.enable_mtp(drafter, WINDOW)
    return pg, tok, drafter


def run_baseline(pg, tok, ids, nmax):
    """Plain greedy AR decode (no MTP)."""
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
    return "".join(tok.decode(torch.tensor(out_ids))), len(out_ids), dt


def run_mtp(pg, tok, drafter, ids, nmax):
    np_ = ids.shape[-1]
    logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
    drv = MTPMPDriver(pg, drafter, window=WINDOW, verbose=(os.environ.get("MTP_VERBOSE") == "1"))
    drv.begin(ids, logits, state)
    out_ids = []
    t0 = time.perf_counter()
    while len(out_ids) < nmax:
        out_ids.append(drv.step())
    dt = time.perf_counter() - t0
    return "".join(tok.decode(torch.tensor(out_ids))), len(out_ids), dt, drv


def main():
    ok_all = True
    for q, nmax in CASES:
        print(f"\n=== case: {q[:44]!r}... (up to {nmax} out) ===", flush=True)
        # baseline: own fresh group
        pg, tok, _ = make_pg(False)
        ids = build_prompt(tok, q)
        base, bn, bdt = run_baseline(pg, tok, ids, nmax)
        pg.close()
        # MTP: own fresh group
        pg, tok, drafter = make_pg(True)
        ids = build_prompt(tok, q)
        mtp, mn, mdt, drv = run_mtp(pg, tok, drafter, ids, nmax)
        pg.close()
        match = (base == mtp)
        ok_all &= match
        st = drv.stats()
        print(f"baseline: {bn} tok {bdt:.2f}s = {bn/bdt:6.1f} tok/s", flush=True)
        print(f"MTP w={WINDOW}: {mn} tok {mdt:.2f}s = {mn/mdt:6.1f} tok/s   "
              f"speedup {bdt/mdt:4.2f}x   {st}", flush=True)
        print(f"  base tail: ...{base[-60:]!r}", flush=True)
        print(f"  mtp  tail: ...{mtp[-60:]!r}", flush=True)
        print("  MATCH" if match else "  *** MISMATCH ***", flush=True)

    print("\nALL MATCH" if ok_all else "\nFAILURES PRESENT", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
