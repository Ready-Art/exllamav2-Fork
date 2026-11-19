"""MP-backend MTP single-case runner (ONE process group per invocation).

Args:  <mode> <case_idx> [window]
    mode     : base | mtp
    case_idx : index into CASES
    window   : draft window (default 6)

Prints one machine-readable line:  RESULT <json>
with the full decoded output text (for base-vs-mtp parity) plus stats.
Builds a single TPProcessGroup, runs one case, exits. Run from ~/exl2work.
"""
import os, sys, time, json, torch

MD = "models/gemma-4-31b-exl2-6.0bpw-hb16-2gpu"
DRAFT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
MAXSEQ = 4096

MODE = sys.argv[1]
CASE_IDX = int(sys.argv[2])
WINDOW = int(sys.argv[3]) if len(sys.argv) > 3 else 6

from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup, MTPMPDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter

CASES = [
    ("What is 2+2?", 100),
    ("Explain what a hash map is and when to use one, in three short paragraphs.", 300),
]


def main():
    q, nmax = CASES[CASE_IDX]
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = MAXSEQ
    tok = ExLlamaV2Tokenizer(c)
    ids = tok.encode(f"<bos><|turn>user\n{q}<turn|>\n<|turn>model\n",
                     encode_special_tokens=True)
    np_ = ids.shape[-1]

    use_mtp = (MODE == "mtp")
    t0 = time.time()
    pg = TPProcessGroup(MD, [20.0] * 4, MAXSEQ, MAXSEQ, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = None
    if use_mtp:
        drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
        pg.enable_mtp(drafter, WINDOW)
    load_s = time.time() - t0

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
    degenerate = (text.count("la la") > 10 or text.count("inging") > 5
                  or text.count("C.") > 3)
    pg.close()
    print("RESULT " + json.dumps({
        "mode": MODE, "case": CASE_IDX, "q": q, "n_prompt": np_, "n_out": len(out_ids),
        "gen_s": round(dt, 3), "gen_tps": round(len(out_ids) / dt, 3),
        "load_s": round(load_s, 1), "degenerate": degenerate, "mtp_stats": stats,
        "text": text,
    }), flush=True)


if __name__ == "__main__":
    main()
