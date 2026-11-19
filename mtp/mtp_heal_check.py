"""Healed MTP-MP single case: checks output is clean + measures gen speed/acceptance.
Run from ~/exl2work:  python mtp_heal_check.py [window]
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


def main():
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = MAXSEQ
    tok = ExLlamaV2Tokenizer(c)
    pg = TPProcessGroup(MD, [20.0] * 4, MAXSEQ, MAXSEQ, "ExLlamaV2Cache_SWA")
    pg.start()
    drafter = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
    pg.enable_mtp(drafter, WINDOW)

    for q, nmax in [("What is 2+2?", 64),
                    ("Explain what a hash map is and when to use one, in three short paragraphs.", 128)]:
        ids = tok.encode(f"<bos><|turn>user\n{q}<turn|>\n<|turn>model\n",
                         encode_special_tokens=True)
        np_ = ids.shape[-1]
        logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        drv = MTPMPDriver(pg, drafter, window=WINDOW, heal=True)
        drv.begin(ids, logits, state)
        out_ids = []
        t0 = time.perf_counter()
        while len(out_ids) < nmax:
            out_ids.append(drv.step())
        dt = time.perf_counter() - t0
        text = "".join(tok.decode(torch.tensor(out_ids)))
        degenerate = (text.count("la la") > 10 or text.count("inging") > 5
                      or text.count(" de ") > 5 or "lL lL" in text)
        print(f"\n[{q[:40]!r}] {len(out_ids)} tok {dt:.2f}s = {len(out_ids)/dt:.1f} tok/s "
              f"degen={degenerate} {drv.stats()}", flush=True)
        print(f"  head={text[:120]!r}", flush=True)
        print(f"  tail={text[-80:]!r}", flush=True)
        pg.mtp_reset_cache()
    pg.close()


if __name__ == "__main__":
    main()
