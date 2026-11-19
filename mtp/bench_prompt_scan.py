"""Prompt scan for a CLEAN long-context benchmark regime (BUG 90a content-dependence).

For each candidate prompt at 4k context: baseline x2 (determinism + cleanliness)
and MTP x1 (acceptance). Prints one RESULT json line per (prompt, mode).
Usage:  python bench_prompt_scan.py [window]
Run from ~/exl2work.
"""
import os, sys, re, time, json, torch

MD = "models/gemma-4-31b-exl2-6.0bpw-hb16-2gpu"
DRAFT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
MAXSEQ = 4096
WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NOUT = 120

from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup, MTPMPDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter

PROMPTS = [
    "What is 2+2?",
    "Count from 1 to 20, one number per line.",
    "List the first ten prime numbers.",
    "Write a haiku about the ocean.",
    "Explain photosynthesis in two short sentences.",
    "What are the capital cities of France and Germany?",
]


def loop_free(text):
    """True if no 1-4 token n-gram repeats 4+ times consecutively."""
    toks = text.split()
    for n in (1, 2, 3, 4):
        i = 0
        while i < len(toks) - 4 * n:
            gram = tuple(toks[i:i + n])
            j = i + n
            rep = 1
            while j + n <= len(toks) and tuple(toks[j:j + n]) == gram:
                rep += 1
                j += n
            if rep >= 4:
                return False
            i = j if rep > 1 else i + n
    return True


def run_one(pg, tok, ids, nmax, use_mtp, drafter):
    np_ = ids.shape[-1]
    if use_mtp:
        logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        drv = MTPMPDriver(pg, drafter, window=WINDOW)
        drv.begin(ids, logits, state)
        out_ids = []
        t0 = time.perf_counter()
        while len(out_ids) < nmax:
            out_ids.append(drv.step())
        dt = time.perf_counter() - t0
        stats = drv.stats()
    else:
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
        stats = {}
    text = "".join(tok.decode(torch.tensor(out_ids)))
    return text, len(out_ids), dt, stats


def main():
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = MAXSEQ
    tok = ExLlamaV2Tokenizer(c)
    pg = TPProcessGroup(MD, [20.0] * 4, MAXSEQ, MAXSEQ, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
    pg.enable_mtp(drafter, WINDOW)
    print("[loaded]", flush=True)

    for pi, q in enumerate(PROMPTS):
        ids = tok.encode(f"<bos><|turn>user\n{q}<turn|>\n<|turn>model\n",
                         encode_special_tokens=True)
        texts = []
        for mode in ("base", "base2", "mtp"):
            pg.mtp_reset_cache()
            use_mtp = (mode == "mtp")
            text, n, dt, stats = run_one(pg, tok, ids, NOUT, use_mtp, drafter)
            texts.append(text)
            print("RESULT " + json.dumps({
                "pi": pi, "q": q, "mode": mode, "n": n, "gen_tps": round(n / dt, 2),
                "loop_free": loop_free(text), "stats": stats,
                "head": text[:60], "tail": text[-40:],
            }), flush=True)
        det = (texts[0] == texts[1])
        print(f"[scan] p{pi} det={det} base_loopfree={loop_free(texts[0])} "
              f"mtp_loopfree={loop_free(texts[2])}", flush=True)
    pg.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
