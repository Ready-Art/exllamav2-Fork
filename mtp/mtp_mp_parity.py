"""Same-process MTP-vs-baseline parity for the MP backend (one TPProcessGroup).

Controls for BUG 90a's cross-process non-determinism by running EVERYTHING in a
single process: the prefill state is deterministic within a process, so a
baseline run and an MTP run of the SAME prompt must produce IDENTICAL text if the
MTP integration is numerically faithful (rewinds, bonus commit, KV gather all
correct). Also includes a baseline self-consistency control (two baseline runs).

Per case:  base -> reset -> base(control) -> reset -> mtp
  CONTROL : base_run1 == base_run2   (in-process determinism / reset works)
  PARITY  : base_run1 == mtp_run     (MTP == baseline)

Run from ~/exl2work:  python exllamav2-fork/mtp/mtp_mp_parity.py [window]
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

CASES = [
    ("What is 2+2?", 64),
    ("Explain what a hash map is and when to use one, in three short paragraphs.", 128),
]


def run_baseline(pg, tok, ids, nmax):
    np_ = ids.shape[-1]
    logits = pg.forward(ids, 0, np_, 1, True)
    nxt = int(torch.argmax(logits[0, -1].float()))
    past = np_
    out_ids = [nxt]
    while len(out_ids) < nmax:
        cur = torch.tensor([[nxt]], dtype=torch.long)
        lg = pg.forward(cur, past, 1, 1, True)
        nxt = int(torch.argmax(lg[0, -1].float()))
        out_ids.append(nxt)
        past += 1
    return "".join(tok.decode(torch.tensor(out_ids))), len(out_ids)


def run_mtp(pg, tok, drafter, ids, nmax):
    np_ = ids.shape[-1]
    logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
    drv = MTPMPDriver(pg, drafter, window=WINDOW)
    drv.begin(ids, logits, state)
    out_ids = []
    while len(out_ids) < nmax:
        out_ids.append(drv.step())
    return "".join(tok.decode(torch.tensor(out_ids))), len(out_ids), drv.stats()


def main():
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = MAXSEQ
    tok = ExLlamaV2Tokenizer(c)
    pg = TPProcessGroup(MD, [20.0] * 4, MAXSEQ, MAXSEQ, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
    pg.enable_mtp(drafter, WINDOW)
    print("[loaded]", flush=True)

    all_ctrl = all_par = True
    for ci, (q, nmax) in enumerate(CASES):
        ids = tok.encode(f"<bos><|turn>user\n{q}<turn|>\n<|turn>model\n",
                         encode_special_tokens=True)
        print(f"\n=== case {ci}: {q[:44]!r} (prompt {ids.shape[-1]}, up to {nmax}) ===", flush=True)

        pg.mtp_reset_cache()
        t0 = time.perf_counter()
        b1, n1 = run_baseline(pg, tok, ids, nmax)
        db1 = time.perf_counter() - t0

        pg.mtp_reset_cache()
        b2, _ = run_baseline(pg, tok, ids, nmax)

        pg.mtp_reset_cache()
        t0 = time.perf_counter()
        m, nm, stats = run_mtp(pg, tok, drafter, ids, nmax)
        dm = time.perf_counter() - t0

        ctrl = (b1 == b2)
        par = (b1 == m)
        all_ctrl &= ctrl
        all_par &= par
        print(f"  base1 : {n1} tok {db1:.2f}s = {n1/db1:6.1f} tok/s  tail ...{b1[-50:]!r}", flush=True)
        print(f"  base2 : {'CONTROL MATCH' if ctrl else '*** CONTROL MISMATCH ***'}", flush=True)
        print(f"  mtp   : {nm} tok {dm:.2f}s = {nm/dm:6.1f} tok/s  speedup {db1/dm:.2f}x  {stats}  tail ...{m[-50:]!r}", flush=True)
        print(f"  PARITY: {'MATCH' if par else '*** MISMATCH ***'}", flush=True)

    print("\nCONTROL (baseline self-consistency):", "PASS" if all_ctrl else "FAIL", flush=True)
    print("PARITY  (MTP == baseline):           ", "PASS" if all_par else "FAIL", flush=True)
    pg.close()
    sys.exit(0 if (all_ctrl and all_par) else 1)


if __name__ == "__main__":
    main()
