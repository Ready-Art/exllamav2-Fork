"""Single-config runner for the fused AG+norm+residual P2P kernel validation.
ONE fresh process group per invocation (second init in-process crashes). Writes a
JSON result (incl. tokens) to OUTFILE. Driven by fuse_agnr_run.sh which spawns a
subprocess per config and compares.

Usage: venv/bin/python fuse_agnr_test.py CTX W NGEN FUSE SAMPLED OUTFILE
"""
import os, sys, time, json, torch

MD = "models/Gemma-4-31B-IT-EXL2-6.0BPW-HB16"
DRAFT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
NGPU = 4
FILLER = ("The history of computing is a long and intricate story that begins with "
          "mechanical calculators and passes through punched cards, vacuum tubes, "
          "transistors and integrated circuits. ")

CTX = int(sys.argv[1]); W = int(sys.argv[2]); NGEN = int(sys.argv[3])
FUSE = int(sys.argv[4]); SAMPLED = int(sys.argv[5]); OUTFILE = sys.argv[6]
_FLAG = os.environ.get("FLAG_NAME", "EXL2_TP_FUSE_AGNR")
os.environ[_FLAG] = "1" if FUSE else "0"

from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.tp_mp import TPProcessGroup, MTPMPDriver
from exllamav2.mtp_drafter import Gemma4AssistantDrafter
from exllamav2.generator.sampler import ExLlamaV2Sampler


def build_prompt(tok, nprefill):
    ids = tok.encode("<bos><|turn>user\n", encode_special_tokens=True)
    while ids.shape[-1] < nprefill:
        ids = torch.cat([ids, tok.encode(FILLER, encode_special_tokens=False)], dim=-1)
    ids = ids[:, :nprefill]
    return torch.cat([ids, tok.encode("<turn|>\n<|turn>model\n",
                                      encode_special_tokens=True)], dim=-1)


def main():
    maxseq = CTX + NGEN + 256
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = maxseq
    tok = ExLlamaV2Tokenizer(c)
    ids = build_prompt(tok, CTX)
    pg = TPProcessGroup(MD, [22.0] * NGPU, maxseq, maxseq, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=W)
    pg.enable_mtp(drafter, W)
    logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
    drv = MTPMPDriver(pg, drafter, window=W)
    drv.begin(ids, logits, state)
    settings = None
    if SAMPLED:
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 1.0; settings.top_p = 0.95; settings.top_k = 64
    for _ in range(8):
        drv.step(settings, tok)
    phases = ("t_gather", "t_draft", "t_verify", "t_rewind", "t_bonus", "t_accept")
    snap_r, snap_a = drv.total_rounds, drv.total_acc
    snap_t = {k: getattr(drv, k) for k in phases}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = []
    while len(out) < NGEN:
        out.append(drv.step(settings, tok))
    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t0) * 1000
    dr = drv.total_rounds - snap_r
    warm = {k: round((getattr(drv, k) - snap_t[k]) / dr * 1000, 2) for k in phases}
    res = {"ctx": CTX, "W": W, "fuse": FUSE, "sampled": SAMPLED,
           "gen_tps": round(len(out) / (gen_ms / 1000), 2),
           "warm_acc": round((drv.total_acc - snap_a) / dr, 3),
           "warm_ms": warm, "tokens": out}
    with open(OUTFILE, "w") as f:
        json.dump(res, f)
    print(json.dumps({k: v for k, v in res.items() if k != "tokens"}), flush=True)
    pg.close()


if __name__ == "__main__":
    main()