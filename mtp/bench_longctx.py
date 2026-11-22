"""Focused long-ctx MTP bench: one fresh TPProcessGroup, prefill to CTX, then generate
NGEN tokens with MTP window W. Reports total gen tok/s AND a warm-window (rounds>=40)
phase breakdown so graph-capture contamination is excluded.

Usage (from ~/exl2work):
  venv/bin/python exllamav2-Fork/mtp/bench_longctx.py CTX W NGEN
  e.g. ... bench_longctx.py 79067 2 400
"""
import os, sys, time, json, torch

MD = "models/Gemma-4-31B-IT-EXL2-6.0BPW-HB16"
DRAFT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
NGPU = 4

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 79067
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
NGEN = int(sys.argv[3]) if len(sys.argv) > 3 else 400
NWARM = 8

from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup, MTPMPDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter
from exllamav2.generator.sampler import ExLlamaV2Sampler

SETTINGS = ExLlamaV2Sampler.Settings()
SETTINGS.temperature = 1.0
SETTINGS.top_p = 0.95
SETTINGS.top_k = 64

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


def main():
    maxseq = CTX + NGEN + NWARM + 256
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = maxseq
    tok = ExLlamaV2Tokenizer(c)
    ids = build_prompt(tok, CTX)
    n_prompt = ids.shape[-1]

    pg = TPProcessGroup(MD, [22.0] * NGPU, maxseq, maxseq, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=W)
    pg.enable_mtp(drafter, W)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
    drv = MTPMPDriver(pg, drafter, window=W)
    drv.begin(ids, logits, state)
    torch.cuda.synchronize()
    pp_ms = (time.perf_counter() - t0) * 1000

    for _ in range(NWARM):
        drv.step(SETTINGS, tok)
    torch.cuda.synchronize()

    # snapshot cumulative timers/acc at the warm boundary
    phases = ("t_gather", "t_draft", "t_verify", "t_rewind", "t_bonus", "t_accept")
    snap_r = drv.total_rounds
    snap_a = drv.total_acc
    snap_t = {k: getattr(drv, k) for k in phases}

    t0 = time.perf_counter()
    out = []
    while len(out) < NGEN:
        out.append(drv.step(SETTINGS, tok))
    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t0) * 1000
    gen_tps = len(out) / (gen_ms / 1000)

    dr = drv.total_rounds - snap_r
    warm = {k: (getattr(drv, k) - snap_t[k]) / dr * 1000 for k in phases}
    wacc = (drv.total_acc - snap_a) / dr
    text = tok.decode(torch.tensor(out))
    degen = (text.count("la la") > 10 or text.count("inging") > 5)
    print(json.dumps({
        "ctx": CTX, "W": W, "n_prompt": n_prompt, "pp_ms": round(pp_ms, 1),
        "gen_tps": round(gen_tps, 2), "gen_ms_tok": round(gen_ms / len(out), 2),
        "warm_rounds": dr, "warm_acc_round": round(wacc, 3),
        "warm_emit_round": round(wacc + 1, 3),
        "warm_ms_round": {k: round(v, 2) for k, v in warm.items()},
        "warm_round_total_ms": round(sum(warm.values()), 2),
        "degenerate": degen, "tail": text[-48:],
    }, indent=2), flush=True)
    pg.close()


if __name__ == "__main__":
    main()