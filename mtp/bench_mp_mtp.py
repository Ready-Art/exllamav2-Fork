"""Benchmark MTP on the MP (process-per-GPU) backend: prefill + generation tok/s.

Contexts: 32k / 64k / 128k. For each context, a FRESH TPProcessGroup is built and
run twice -- plain baseline, then MTP (window 6) -- so the numbers are directly
comparable and every run starts from a clean cache. A short (unmeasured) warmup
decode runs inside the measured group so CUDA-graph capture / first-gather costs
are paid before the timed section, identically for both paths.

Usage (from ~/exl2work):  python mtp/bench_mp_mtp.py [contexts_csv] [ngen] [warm]
  e.g.  python mtp/bench_mp_mtp.py 32768,65536,131072 192 8
"""
import os, sys, time, json, torch

MD = "models/gemma-4-31b-exl2-6.0bpw-hb16-2gpu"
DRAFT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
NGPU = 4
WINDOW = 6

CTXS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else
                         ["32768", "65536", "131072"])]
NGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 192
NWARM = int(sys.argv[3]) if len(sys.argv) > 3 else 8

from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup, MTPMPDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter

FILLER = ("The history of computing is a long and intricate story that begins with "
          "mechanical calculators and passes through punched cards, vacuum tubes, "
          "transistors and integrated circuits. ")


def build_prompt(tok, nprefill):
    ids = tok.encode("<bos><|turn>user\n", encode_special_tokens=True)
    while ids.shape[-1] < nprefill:
        ids = torch.cat([ids, tok.encode(FILLER, encode_special_tokens=False)], dim=-1)
    ids = ids[:, :nprefill]
    return torch.cat([ids, tok.encode("<turn|>\n<|turn>model\n",
                                      encode_special_tokens=True)], dim=-1)


def run_once(maxseq, nprefill, ngen, use_mtp):
    """Fresh process group; returns a result dict."""
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = maxseq
    tok = ExLlamaV2Tokenizer(c)
    ids = build_prompt(tok, nprefill)
    n_prompt = ids.shape[-1]

    t0 = time.time()
    pg = TPProcessGroup(MD, [20.0] * NGPU, maxseq, maxseq, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = None
    if use_mtp:
        drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
        pg.enable_mtp(drafter, WINDOW)
    load_s = time.time() - t0

    # ---- prefill (measured) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if use_mtp:
        logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        drv = MTPMPDriver(pg, drafter, window=WINDOW)
        drv.begin(ids, logits, state)
        nxt = int(torch.argmax(logits[0, -1].float()))
    else:
        logits = pg.forward(ids, 0, n_prompt, 1, True)
        nxt = int(torch.argmax(logits[0, -1].float()))
        past = n_prompt
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    pp_tps = (n_prompt - 1) / t_prefill

    # ---- warmup decode (unmeasured): graph capture + first MTP gather ----
    for _ in range(NWARM):
        if use_mtp:
            nxt = drv.step()
        else:
            cur = torch.tensor([[nxt]], dtype=torch.long)
            lg = pg.forward(cur, past, 1, 1, True)
            nxt = int(torch.argmax(lg[0, -1].float()))
            past += 1
    torch.cuda.synchronize()

    # ---- generation (measured) ----
    t0 = time.perf_counter()
    out_ids = []
    if use_mtp:
        while len(out_ids) < ngen:
            out_ids.append(drv.step())
    else:
        n = 0
        while n < ngen:
            cur = torch.tensor([[nxt]], dtype=torch.long)
            lg = pg.forward(cur, past, 1, 1, True)
            nxt = int(torch.argmax(lg[0, -1].float()))
            out_ids.append(nxt)
            past += 1
            n += 1
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0
    gen_tps = len(out_ids) / t_gen

    text = tok.decode(torch.tensor(out_ids))
    degenerate = (text.count("la la") > 10 or text.count("inging") > 5
                  or text.count("C.") > 3)
    st = drv.stats() if use_mtp else {}
    pg.close()
    return {"ctx": nprefill, "use_mtp": use_mtp, "load_s": round(load_s, 1),
            "n_prompt": n_prompt, "pp_tps": round(pp_tps, 2),
            "pp_ms": round(t_prefill * 1000, 1), "gen_tps": round(gen_tps, 2),
            "gen_ms_tok": round(t_gen / len(out_ids) * 1000, 2), "gen_n": len(out_ids),
            "degenerate": degenerate, "mtp_stats": st, "tail": text[-48:]}


def main():
    results = []
    for ctx in CTXS:
        maxseq = ctx + NGEN + NWARM + 256
        for use_mtp in (False, True):
            tag = "MTP " if use_mtp else "base"
            print(f"\n===== ctx={ctx} {tag} maxseq={maxseq} =====", flush=True)
            r = run_once(maxseq, ctx, NGEN, use_mtp)
            print(json.dumps(r), flush=True)
            results.append(r)
    print("\n==== SUMMARY ====", flush=True)
    for r in results:
        print(f"ctx={r['ctx']:>6} {'MTP' if r['use_mtp'] else 'base'}  "
              f"PP {r['pp_tps']:>8.2f} tok/s ({r['pp_ms']:>8.1f} ms)   "
              f"GEN {r['gen_tps']:>6.2f} tok/s ({r['gen_ms_tok']:>7.2f} ms/tok)"
              f"{'  ***DEGEN***' if r['degenerate'] else ''}", flush=True)
    with open("bench_mp_mtp_results.json", "w") as f:
        json.dump({"ctxs": CTXS, "ngen": NGEN, "warm": NWARM, "window": WINDOW,
                   "results": results}, f, indent=2)
    print("wrote bench_mp_mtp_results.json", flush=True)


if __name__ == "__main__":
    main()
