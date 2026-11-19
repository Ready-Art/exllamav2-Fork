"""MTP single-context benchmark. DEFAULT backend: mp (fastest measured direct loop).

Args:  <mode> <ctx> <ngen> [warm] [--prompt TEXT] [--backend mp|sp]
    mode : base | mtp
    ctx  : prompt context length (filler prompt) OR exact prompt length
    ngen : generated tokens to measure
    warm : unmeasured warmup decode steps (default 8)
    --prompt TEXT : use a real chat prompt (clean-regime benching); the
         <ctx> arg is ignored except for max_seq_len sizing (ctx + ngen + warm)
    --backend mp|sp : DEFAULT mp = multiprocess TPProcessGroup + MTPMPDriver (tight direct
         loop, ~39 tok/s base @4k). sp = single-process TP (load_tp) + CUDA graphs via the
         STREAMING generator (opt-in). NOTE: SP raw forward is faster (~75 tok/s) but the
         streaming generator adds ~30 ms/token host overhead, measuring SLOWER (~23 base /
         31 mtp @4k) than MP. A direct-loop SP MTP driver is needed to realize SP's speed;
         see MP_DECODE_ANALYSIS.md.

Prints one machine-readable line:  RESULT <json>
with prefill tok/s, generation tok/s, and (for mtp) acceptance stats.
Run from ~/exl2work.
"""
import os, sys, time, json, torch

MD = "models/gemma-4-31b-exl2-6.0bpw-hb16-2gpu"
DRAFT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
NGPU = 4
WINDOW = 6

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_PROMPT_ARGS = [a for a in sys.argv[1:] if a.startswith("--prompt")]
MODE = _args[0]
CTX = int(_args[1])
NGEN = int(_args[2])
NWARM = int(_args[3]) if len(_args) > 3 else 8
CUSTOM_PROMPT = None
if _PROMPT_ARGS:
    rest = sys.argv[1:]
    i = rest.index("--prompt")
    CUSTOM_PROMPT = rest[i + 1]
BACKEND = "mp"
if "--backend" in sys.argv:
    BACKEND = sys.argv[sys.argv.index("--backend") + 1]

from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
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


def make_inputs():
    if CUSTOM_PROMPT is not None:
        c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
        c.max_seq_len = max(CTX + NGEN + NWARM + 256, 4096)
        tok = ExLlamaV2Tokenizer(c)
        ids = tok.encode(f"<bos><|turn>user\n{CUSTOM_PROMPT}<turn|>\n<|turn>model\n",
                         encode_special_tokens=True)
    else:
        c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
        c.max_seq_len = CTX + NGEN + NWARM + 256
        tok = ExLlamaV2Tokenizer(c)
        ids = build_prompt(tok, CTX)
    return c, tok, ids


def emit(c, n_prompt, pp_tps, t_prefill, out_ids, t_gen, load_s, stats, tok):
    text = tok.decode(torch.tensor(out_ids)) if out_ids else ""
    degenerate = (text.count("la la") > 10 or text.count("inging") > 5
                  or text.count("C.") > 3)
    gen_tps = len(out_ids) / t_gen if t_gen > 0 else 0.0
    print("RESULT " + json.dumps({
        "mode": MODE, "backend": BACKEND, "ctx": CTX, "maxseq": c.max_seq_len,
        "n_prompt": n_prompt, "pp_tps": round(pp_tps, 2), "pp_ms": round(t_prefill * 1000, 1),
        "gen_n": len(out_ids), "gen_tps": round(gen_tps, 2),
        "gen_ms_tok": round(t_gen / max(1, len(out_ids)) * 1000, 2),
        "load_s": round(load_s, 1), "degenerate": degenerate,
        "mtp_stats": stats, "tail": text[-48:],
    }), flush=True)


# --------------------------------------------------------------------- SP + graphs (default)
def main_sp():
    from exllamav2 import ExLlamaV2
    from exllamav2.cache import ExLlamaV2Cache_SWA, ExLlamaV2Cache_TP
    from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
    c, tok, ids = make_inputs()
    n_prompt = ids.shape[-1]
    use_mtp = (MODE == "mtp")

    t0 = time.time()
    m = ExLlamaV2(c)
    m.load_tp([20.0] * NGPU, expect_cache_tokens=c.max_seq_len,
              expect_cache_base=ExLlamaV2Cache_SWA)
    drafter = Gemma4AssistantDrafter(m, DRAFT_DIR, device=0, num_draft_tokens=WINDOW) if use_mtp else None
    load_s = time.time() - t0

    cache = ExLlamaV2Cache_TP(m, base=ExLlamaV2Cache_SWA, max_seq_len=c.max_seq_len)
    kw = dict(mtp_drafter=drafter, num_speculative_tokens=WINDOW) if use_mtp else {}
    g = ExLlamaV2StreamingGenerator(m, cache, tok, **kw)
    g.set_stop_conditions([106, 1])
    settings = ExLlamaV2Sampler.Settings(); settings.temperature = 0.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.begin_stream(ids, settings, token_healing=True)   # prefill (+ first-token healing)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    pp_tps = (n_prompt - 1) / t_prefill if t_prefill > 0 else 0.0

    def take(n):
        got = []
        while len(got) < n:
            ch, eos, chunk = g.stream()
            if chunk is not None and chunk.numel():
                got.extend(chunk[0].tolist())
            if eos:
                break
        return got

    take(NWARM)                                          # warmup: graph capture, unmeasured
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_ids = take(NGEN)
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0

    stats = {}
    if use_mtp:
        eff, acc, tot, dr, ac = g.get_sd_stats()
        rounds = dr / WINDOW if dr else 0
        stats = {"backend": "sp", "accepted": ac, "drafted": dr,
                 "acc_per_round": round(ac / rounds, 3) if rounds else 0.0,
                 "emitted_per_round": round(tot / rounds, 3) if rounds else 0.0,
                 "accuracy": round(acc, 4), "efficiency": round(eff, 4)}
    emit(c, n_prompt, pp_tps, t_prefill, out_ids, t_gen, load_s, stats, tok)


# --------------------------------------------------------------------- MP (opt-in)
def main_mp():
    from exllamav2.tp_mp import TPProcessGroup, MTPMPDriver
    c, tok, ids = make_inputs()
    n_prompt = ids.shape[-1]
    use_mtp = (MODE == "mtp")
    t0 = time.time()
    pg = TPProcessGroup(MD, [20.0] * NGPU, c.max_seq_len, c.max_seq_len, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = None
    if use_mtp:
        drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
        pg.enable_mtp(drafter, WINDOW)
    load_s = time.time() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if use_mtp:
        logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        drv = MTPMPDriver(pg, drafter, window=WINDOW)
        drv.begin(ids, logits, state)
    else:
        logits = pg.forward(ids, 0, n_prompt, 1, True)
        nxt = int(torch.argmax(logits[0, -1].float()))
        past = n_prompt
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    pp_tps = (n_prompt - 1) / t_prefill

    for _ in range(NWARM):
        if use_mtp:
            nxt = drv.step()
        else:
            cur = torch.tensor([[nxt]], dtype=torch.long)
            lg = pg.forward(cur, past, 1, 1, True)
            nxt = int(torch.argmax(lg[0, -1].float()))
            past += 1
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    out_ids = []
    if use_mtp:
        while len(out_ids) < NGEN:
            out_ids.append(drv.step())
    else:
        n = 0
        while n < NGEN:
            cur = torch.tensor([[nxt]], dtype=torch.long)
            lg = pg.forward(cur, past, 1, 1, True)
            nxt = int(torch.argmax(lg[0, -1].float()))
            out_ids.append(nxt)
            past += 1
            n += 1
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0

    stats = drv.stats() if use_mtp else {}
    pg.close()
    emit(c, n_prompt, pp_tps, t_prefill, out_ids, t_gen, load_s, stats, tok)


def main():
    if BACKEND == "mp":
        main_mp()
    else:
        main_sp()


if __name__ == "__main__":
    main()
