"""
Single-run MTP window bench (fresh subprocess per config — second process-group
init in one interpreter crashes). Matches the tabby test shape: 4x3090 split
[22,22,22,22], cache 32768, ctx ~1315 prose, temp-1 sampled MTP, greedy base.

Usage (from ~/exl2work):
  NCCL_P2P_LEVEL=PXB PYTORCH_CUDA_ALLOC_CONF=backend:native \
    venv/bin/python exllamav2-Fork/mtp/bench_window.py WINDOW MODE [CTX] [NGEN] [WARM]
  WINDOW: draft tokens (0 for base), MODE: mtp|base
"""
import os, sys, time, json, torch

MD = "models/Gemma-4-31B-IT-EXL2-6.0BPW-HB16"
DRAFT = "models/gemma-4-31B-it-assistant"
SPLIT = [22.0] * 4
CACHE = 32768

WINDOW = int(sys.argv[1])
MODE = sys.argv[2]
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 1315
NGEN = int(sys.argv[4]) if len(sys.argv) > 4 else 320
NWARM = int(sys.argv[5]) if len(sys.argv) > 5 else 16

from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup, MTPMPDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter
from exllamav2.generator.sampler import ExLlamaV2Sampler

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

    t0 = time.time()
    pg = TPProcessGroup(MD, SPLIT, maxseq, CACHE, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = None
    if MODE == "mtp":
        drafter = Gemma4AssistantDrafter(pg, DRAFT, device=0, num_draft_tokens=WINDOW)
        pg.enable_mtp(drafter, WINDOW)
    load_s = time.time() - t0

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 1.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if MODE == "mtp":
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

    for _ in range(NWARM):
        if MODE == "mtp":
            nxt = drv.step(settings=settings, tokenizer=tok, sequence_ids=ids)
        else:
            cur = torch.tensor([[nxt]], dtype=torch.long)
            lg = pg.forward(cur, past, 1, 1, True)
            nxt = int(torch.argmax(lg[0, -1].float()))
            past += 1
    torch.cuda.synchronize()

    prof = None
    if os.environ.get("BENCH_CPROFILE"):
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    t0 = time.perf_counter()
    out_ids = []
    if MODE == "mtp":
        while len(out_ids) < NGEN:
            out_ids.append(drv.step(settings=settings, tokenizer=tok, sequence_ids=ids))
    else:
        while len(out_ids) < NGEN:
            cur = torch.tensor([[nxt]], dtype=torch.long)
            lg = pg.forward(cur, past, 1, 1, True)
            nxt = int(torch.argmax(lg[0, -1].float()))
            out_ids.append(nxt)
            past += 1
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0
    if prof is not None:
        prof.disable()
        import pstats, io
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(35)
        print(s.getvalue(), flush=True)

    text = tok.decode(torch.tensor(out_ids))
    st = drv.stats() if MODE == "mtp" else {}
    r = {"window": WINDOW, "mode": MODE, "ctx": CTX, "load_s": round(load_s, 1),
         "n_prompt": n_prompt, "pp_tps": round((n_prompt - 1) / t_prefill, 2),
         "gen_tps": round(len(out_ids) / t_gen, 2),
         "gen_ms_tok": round(t_gen / len(out_ids) * 1000, 2),
         "mtp_stats": st, "tail": text[-48:]}
    print("RESULT " + json.dumps(r), flush=True)
    pg.close()


if __name__ == "__main__":
    main()
