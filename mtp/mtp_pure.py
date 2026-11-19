"""PURE MTP run (no baseline), DEFAULT env (graphs on). Validates the main deliverable
with the settings a user would actually run. Hash-map + arithmetic prompts."""
import torch, time, os, sys
# NOTE: do NOT set EXL2_TP_GRAPHS -> default (on), matching real usage.
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.cache import ExLlamaV2Cache_SWA, ExLlamaV2Cache_TP
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
from exllamav2.mtp_drafter import Gemma4AssistantDrafter

MD = "models/gemma-4-31b-exl2-6.0bpw-hb16-2gpu"
DRAFT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
MAXSEQ = 4096
WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 6

c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
c.max_seq_len = MAXSEQ
m = ExLlamaV2(c); m.load_tp([20, 20, 20, 20], expect_cache_tokens=MAXSEQ, expect_cache_base=ExLlamaV2Cache_SWA)
tok = ExLlamaV2Tokenizer(c)
print("loaded target", flush=True)
t0 = time.time()
drafter = Gemma4AssistantDrafter(m, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
print(f"loaded drafter in {time.time()-t0:.1f}s", flush=True)

CASES = [
    ("What is 2+2?", 60),
    ("Explain what a hash map is and when to use one, in three short paragraphs.", 200),
]
ok_all = True
for q, nmax in CASES:
    ids = tok.encode(f"<bos><|turn>user\n{q}<turn|>\n<|turn>model\n", encode_special_tokens=True)
    cache = ExLlamaV2Cache_TP(m, base=ExLlamaV2Cache_SWA, max_seq_len=MAXSEQ)
    g = ExLlamaV2StreamingGenerator(m, cache, tok, mtp_drafter=drafter, num_speculative_tokens=WINDOW)
    s = ExLlamaV2Sampler.Settings(); s.temperature = 0.0
    g.set_stop_conditions([106, 1])
    g.begin_stream(ids, s, token_healing=True)
    out, toks = [], 0
    t0 = time.time()
    while True:
        ch, eos, chunk_ids = g.stream()
        out.append(ch)
        toks += int(chunk_ids.shape[-1]) if chunk_ids is not None and chunk_ids.numel() else 0
        if eos or toks >= nmax:
            break
    dt = time.time() - t0
    r = "".join(out)
    eff, acc, tt, td, ac = g.get_sd_stats()
    loop = r.count("la la") > 10 or r.count("inging") > 5 or r.count("C.") > 3
    ok_all &= not loop
    print(f"\n[{q[:40]!r}] {toks} tok {dt:.2f}s = {toks/dt:.1f} tok/s  acc/round={ac/max(tt,1)*WINDOW:.2f}{'  ***DEGEN***' if loop else ''}", flush=True)
    print(f"  head={r[:120]!r}", flush=True)
    print(f"  tail={r[-80:]!r}", flush=True)
print("\nALL CLEAN" if ok_all else "\nDEGENERATION DETECTED")
sys.exit(0 if ok_all else 1)
