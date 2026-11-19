"""
MTP integration test through the real streaming API:
  ExLlamaV2StreamingGenerator(model, cache, tokenizer, mtp_drafter=drafter)
Compares MTP output vs plain baseline (both greedy) and measures throughput.
Token healing is left ON (default), which exercises the drafter's self-healing
re-commit path. Run from ~/exl2work:  python mtp_stream_test.py [window]
"""
import os, sys, time, torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.cache import ExLlamaV2Cache_SWA, ExLlamaV2Cache_TP
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
from exllamav2.mtp_drafter import Gemma4AssistantDrafter

MD = "models/gemma-4-31b-exl2-6.0bpw-hb16-2gpu"
DRAFT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant"
    "/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1")
MAXSEQ = 4096
SPLIT = [20, 20, 20, 20]
WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 6

c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
c.max_seq_len = MAXSEQ
m = ExLlamaV2(c)
m.load_tp(SPLIT, expect_cache_tokens=MAXSEQ, expect_cache_base=ExLlamaV2Cache_SWA)
tok = ExLlamaV2Tokenizer(c)
print("loaded target", flush=True)
t0 = time.perf_counter()
drafter = Gemma4AssistantDrafter(m, DRAFT_DIR, device = 0, num_draft_tokens = WINDOW)
print(f"loaded drafter in {time.perf_counter()-t0:.1f}s", flush=True)

CASES = [
    ("What is 2+2?", 100),
    ("Explain what a hash map is and when to use one, in three short paragraphs.", 300),
]

def run(use_mtp, ids, nmax, settings):
    cache = ExLlamaV2Cache_TP(m, base = ExLlamaV2Cache_SWA, max_seq_len = MAXSEQ)
    kw = {}
    if use_mtp:
        kw = dict(mtp_drafter = drafter, num_speculative_tokens = WINDOW)
    g = ExLlamaV2StreamingGenerator(m, cache, tok, **kw)
    g.set_stop_conditions([106, 1])
    g.begin_stream(ids, settings, token_healing = True)
    out, toks = [], 0
    t0 = time.perf_counter()
    while True:
        ch, eos, chunk_ids = g.stream()
        out.append(ch)
        toks += int(chunk_ids.shape[-1]) if chunk_ids is not None and chunk_ids.numel() else 0
        if eos or toks >= nmax:
            break
    dt = time.perf_counter() - t0
    return "".join(out), toks, dt, g

ok_all = True
for q, nmax in CASES:
    ids = tok.encode(f"<bos><|turn>user\n{q}<turn|>\n<|turn>model\n", encode_special_tokens = True)
    s0 = ExLlamaV2Sampler.Settings(); s0.temperature = 0.0
    s1 = ExLlamaV2Sampler.Settings(); s1.temperature = 0.0
    print(f"\n=== case: {q[:44]!r}... (prompt {ids.shape[-1]} tok, up to {nmax} out) ===", flush=True)
    base, bt, bdt, bg = run(False, ids, nmax, s0)
    eff_b, acc_b, tot_b, dr_b, ac_b = bg.get_sd_stats()
    mtp, mt, mdt, mg = run(True, ids, nmax, s1)
    eff_m, acc_m, tot_m, dr_m, ac_m = mg.get_sd_stats()
    match = (base == mtp)
    ok_all &= match
    print(f"baseline: {bt} tok {bdt:.2f}s = {bt/bdt:6.1f} tok/s   sd={bg.get_sd_stats()}", flush=True)
    print(f"MTP w={WINDOW}: {mt} tok {mdt:.2f}s = {mt/mdt:6.1f} tok/s   speedup {bdt/mdt:4.2f}x   sd={mg.get_sd_stats()}", flush=True)
    print(f"  base tail: ...{base[-60:]!r}", flush=True)
    print(f"  mtp  tail: ...{mtp[-60:]!r}", flush=True)
    print("  MATCH" if match else "  *** MISMATCH ***", flush=True)

print("\nALL MATCH" if ok_all else "\nFAILURES PRESENT")
sys.exit(0 if ok_all else 1)
