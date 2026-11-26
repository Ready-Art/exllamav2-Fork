"""Degeneration isolation (user-authorized GPU self-test 2026-11-26).

One mode per subprocess (fresh TP group). Greedy long generation; reports a
repetition score (fraction of tail tokens in a repeating cycle; high => loop).

  --mode base_graphs   : plain target decode, CUDA graphs ON
  --mode base_eager    : plain target decode, graphs disabled
  --mode mtp_graphs    : MTP greedy acceptance, graphs ON
  --mode mtp_eager     : MTP greedy acceptance, graphs disabled

Usage: venv/bin/python exllamav2-Fork/mtp/degen_isolate.py --mode mtp_graphs
"""
import argparse
import os
import time

os.environ.setdefault("EXL2_KV_INT8", "1")
os.environ.setdefault("EXL2_TP_GRAPH_KV_ENVELOPE", "512")
os.environ["EXL2_BATCH_DRAFT_XCHECK"] = "0"

import torch
from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup
from exllamav2.mtp_mp import MTPMPDriver, MTPMPBatchDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter
from exllamav2.generator.sampler import ExLlamaV2Sampler

MD = "/home/westfox35/exl2work/models/Gemma-4-31B-IT-EXL25-6U-HAD-AWQ-HB16"
DRAFT = "/home/westfox35/exl2work/models/gemma-4-31B-it-assistant"
NGPU = 4
MAXSEQ = 16384
W = 3
NGEN = 500

GREEDY = ExLlamaV2Sampler.Settings()
GREEDY.top_k = 1

TEMP = ExLlamaV2Sampler.Settings()
TEMP.temperature = 0.7
TEMP.top_k = 64
TEMP.top_p = 0.9

FILLER = ("The history of computing is a long and intricate story that begins with "
          "mechanical calculators and passes through punched cards, vacuum tubes, "
          "transistors and integrated circuits. Each layer hid the complexity of the "
          "one below it, and each new layer created new kinds of complexity. ")


def build_prompt(tok, n):
    ids = tok.encode("<bos><|turn>user\nWrite a varied story about astronauts. ",
                     encode_special_tokens=True)
    while ids.shape[-1] < n:
        ids = torch.cat([ids, tok.encode(FILLER, encode_special_tokens=False)], dim=-1)
    ids = ids[:, :n]
    return torch.cat([ids, tok.encode("<turn|>\n<|turn>model\n",
                                      encode_special_tokens=True)], dim=-1)


def rep_score(ids):
    ids = list(ids)
    n = len(ids)
    if n < 40:
        return 0.0
    tail = ids[n // 2:]
    best = 0.0
    for period in range(1, 24):
        match = sum(1 for i in range(period, len(tail)) if tail[i] == tail[i - period])
        best = max(best, match / max(1, len(tail) - period))
    return best


def main():
    global NGEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=1300)
    ap.add_argument("--ngen", type=int, default=NGEN)
    args = ap.parse_args()
    NGEN = args.ngen
    graphs = args.mode.endswith("graphs")
    use_mtp = args.mode.startswith("mtp")
    use_batch = "batch" in args.mode
    settings = TEMP if "temp" in args.mode else GREEDY

    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = MAXSEQ
    tok = ExLlamaV2Tokenizer(c)
    pg = TPProcessGroup(MD, [18.0, 22.0, 22.0, 22.0], MAXSEQ, MAXSEQ, "ExLlamaV2Cache_SWA",
                        max_batch=(2 if use_batch else 1), max_input_len=4096)
    pg.start()
    if not graphs:
        pg.model.tp_graph.disable("degen_isolate eager mode")
    drafter = Gemma4AssistantDrafter(pg, DRAFT, device=0, num_draft_tokens=W)
    pg.enable_mtp(drafter, W)

    ids = build_prompt(tok, args.prompt_tokens)
    print(f"[{args.mode}] prompt {ids.shape[-1]} tokens, graphs={graphs}, mtp={use_mtp}",
          flush=True)
    t0 = time.perf_counter()
    out = []
    if use_mtp and use_batch:
        logits, state = pg.forward_state_row(0, ids, last_id_only=True)
        drv = MTPMPBatchDriver(pg, drafter, window=W)
        seq = ids.clone()
        drv.begin_row(0, ids, logits, state, settings, tok, seq)
        while len(out) < NGEN:
            toks = drv.step_batch([0], {0: settings}, tok, {0: seq})
            for t in toks.get(0, []):
                out.append(t)
                seq = torch.cat([seq, torch.tensor([[t]], dtype=torch.long)], dim=-1)
                if len(out) >= NGEN:
                    break
    elif use_mtp:
        logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        drv = MTPMPDriver(pg, drafter, window=W)
        drv.begin(ids, logits, state)
        seq = ids.clone()
        for _ in range(NGEN):
            t = drv.step(settings, tok, seq)
            out.append(t)
            seq = torch.cat([seq, torch.tensor([[t]], dtype=torch.long)], dim=-1)
    else:
        logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        seq = ids.clone()
        for _ in range(NGEN):
            t = int(logits[0, -1].argmax().item())
            out.append(t)
            seq = torch.cat([seq, torch.tensor([[t]], dtype=torch.long)], dim=-1)
            logits, state = pg.forward_state(seq[:, -1:], last_id_only=True, cpu_logits=True)
    dt = time.perf_counter() - t0
    rs = rep_score(out)
    txt = tok.decode(torch.tensor(out[-40:], dtype=torch.long).unsqueeze(0))
    print(f"[{args.mode}] rep_score={rs:.3f} time={dt:.1f}s t/s={NGEN/dt:.1f}", flush=True)
    print(f"[{args.mode}] tail_text={txt!r}", flush=True)
    print(f"[{args.mode}] tail_ids={out[-30:]}", flush=True)
    pg.close()


if __name__ == "__main__":
    main()
