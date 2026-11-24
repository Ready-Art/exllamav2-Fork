"""INT8-PRIMARY KV end-to-end test (MP TP4, Gemma-4). Fresh process per run.

Usage: python mtp/int8_primary_test.py <ctx> <ngen> [--int8 0|1] [--base] [--cont]
  --int8 1  : EXL2_KV_INT8=1 (INT8 primary, no fp16 shadow)
  --base    : plain decode (no MTP) instead of the MTP draft/verify loop
  --cont    : after the first decode, truncate to ctx//2 and prefill a suffix
              (exercises the INT8->fp16 continuation dequant staging), decode more

Prints RESULT json (gen tok/s, coherence, tail) and VRAM json (nvidia-smi per-GPU
samples at load / post-prefill / steady-decode). A/B by running twice with
--int8 0 / --int8 1 in separate processes and comparing.
"""
import os, sys, time, json, subprocess, threading

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
CTX = int(_args[0])
NGEN = int(_args[1])
INT8 = "1" if "--int8" in sys.argv and sys.argv[sys.argv.index("--int8") + 1] == "1" else "0"
BASE = "--base" in sys.argv
CONT = "--cont" in sys.argv
os.environ["EXL2_KV_INT8"] = INT8
os.environ.setdefault("EXL2_VRAM_TRACE", "1")

import torch
from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup, MTPMPDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter

MD = "/home/westfox35/exl2work/models/Gemma-4-31B-IT-EXL25-6U-HAD-AWQ-HB16"
DRAFT_DIR = "/home/westfox35/exl2work/models/gemma-4-31B-it-assistant"
SPLIT = [19.0, 22.0, 22.0, 22.0]
WINDOW = 3
FILLER = ("The history of computing is a long and intricate story that begins with "
          "mechanical calculators and passes through punched cards, vacuum tubes, "
          "transistors and integrated circuits. ")


def smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return [int(x) for x in out.split()]
    except Exception:
        return []


def build_prompt(tok, nprefill):
    ids = tok.encode("<bos><|turn>user\n", encode_special_tokens=True)
    while ids.shape[-1] < nprefill:
        ids = torch.cat([ids, tok.encode(FILLER, encode_special_tokens=False)], dim=-1)
    ids = ids[:, :nprefill]
    return torch.cat([ids, tok.encode("<turn|>\n<|turn>model\n",
                                      encode_special_tokens=True)], dim=-1)


def degenerate(text):
    return (text.count("la la") > 10 or text.count("inging") > 5
            or text.count("C.") > 3 or text.count(" the the") > 5)


def main():
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = CTX + NGEN + 1024
    tok = ExLlamaV2Tokenizer(c)
    ids = build_prompt(tok, CTX)
    n_prompt = ids.shape[-1]

    vram = {"load_pre": smi()}
    t0 = time.time()
    pg = TPProcessGroup(MD, SPLIT, c.max_seq_len, c.max_seq_len, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = None
    if not BASE:
        drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
        pg.enable_mtp(drafter, WINDOW)
    load_s = time.time() - t0
    vram["load_post"] = smi()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if not BASE:
        logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        drv = MTPMPDriver(pg, drafter, window=WINDOW)
        drv.begin(ids, logits, state)
    else:
        logits = pg.forward(ids, 0, n_prompt, 1, True)
        nxt = int(torch.argmax(logits[0, -1].float()))
        past = n_prompt
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    vram["post_prefill"] = smi()

    def decode(n):
        out = []
        if not BASE:
            while len(out) < n:
                out.append(drv.step())
        else:
            nonlocal nxt, past
            while len(out) < n:
                cur = torch.tensor([[nxt]], dtype=torch.long)
                lg = pg.forward(cur, past, 1, 1, True)
                nxt = int(torch.argmax(lg[0, -1].float()))
                out.append(nxt)
                past += 1
        return out

    decode(8)  # warmup (graph captures)
    torch.cuda.synchronize()
    vram["steady_decode"] = smi()
    t0 = time.perf_counter()
    out_ids = decode(NGEN)
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0
    text = tok.decode(torch.tensor(out_ids)) if out_ids else ""

    cont_info = None
    if CONT:
        # Continuation prefill from a truncated position: the INT8 primary holds
        # the prefix; begin_prefill_staging must dequantize it into fp16 staging.
        # p must sit inside the live SWA ring (>= window_beg): truncate near the end.
        p = pg.cache.current_seq_len - 256
        pg.truncate_cache(p)
        full2 = build_prompt(tok, CTX + 128)
        suffix = full2[:, p:]
        t0 = time.perf_counter()
        if not BASE:
            lg2, st2 = pg.forward_state(suffix, last_id_only=True, cpu_logits=True)
            # Driver protocol (as tabby): FULL prompt ids for position tracking,
            # suffix-prefill logits/state for the base.
            drv.begin(full2, lg2, st2)
        else:
            lg2 = pg.forward(suffix, p, suffix.shape[-1], 1, True)
            nxt = int(torch.argmax(lg2[0, -1].float()))
            past = p + suffix.shape[-1]
        torch.cuda.synchronize()
        t_cont = time.perf_counter() - t0
        out2 = decode(64)
        text2 = tok.decode(torch.tensor(out2)) if out2 else ""
        cont_info = {"cont_prefill_ms": round(t_cont * 1000, 1),
                     "cont_degenerate": degenerate(text2), "cont_tail": text2[-48:]}

    stats = drv.stats() if not BASE else {}
    res = {
        "int8": INT8, "base": BASE, "ctx": CTX, "maxseq": c.max_seq_len,
        "n_prompt": n_prompt, "pp_tps": round((n_prompt - 1) / t_prefill, 2),
        "pp_ms": round(t_prefill * 1000, 1),
        "gen_n": len(out_ids), "gen_tps": round(len(out_ids) / t_gen, 2),
        "load_s": round(load_s, 1), "degenerate": degenerate(text),
        "mtp_stats": stats, "tail": text[-64:], "cont": cont_info,
    }
    print("RESULT " + json.dumps(res), flush=True)
    print("VRAM " + json.dumps(vram), flush=True)
    pg.close()


if __name__ == "__main__":
    main()
