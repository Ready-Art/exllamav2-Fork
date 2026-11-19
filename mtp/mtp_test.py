"""
Standalone MTP (Gemma4 assistant) test: load the TP Gemma-4 target + the single-device
drafter, run the draft-verify loop for a few rounds, and compare the emitted text against
the target's plain greedy baseline. Confirms (a) the drafter produces tokens the target
accepts, and (b) the output matches the no-draft baseline exactly (greedy spec-decode is
output-identical when the drafter is greedy and the target verifies).

Run from ~/exl2work:  python mtp_test.py [window]
"""
import os, sys, time, faulthandler, torch
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

def load_target():
    c = ExLlamaV2Config(); c.model_dir = MD; c.prepare(); c.arch_compat_overrides()
    c.max_seq_len = MAXSEQ
    m = ExLlamaV2(c)
    m.load_tp(SPLIT, expect_cache_tokens=MAXSEQ, expect_cache_base=ExLlamaV2Cache_SWA)
    tok = ExLlamaV2Tokenizer(c)
    return m, tok

def make_prompt(tok, q):
    return tok.encode(f"<bos><|turn>user\n{q}<turn|>\n<|turn>model\n", encode_special_tokens=True)

def baseline(m, tok, ids, n):
    cache = ExLlamaV2Cache_TP(m, base=ExLlamaV2Cache_SWA, max_seq_len=MAXSEQ)
    g = ExLlamaV2StreamingGenerator(m, cache, tok)
    s = ExLlamaV2Sampler.Settings(); s.temperature = 0.0
    g.set_stop_conditions([106, 1])
    g.begin_stream(ids, s)
    out = []
    for _ in range(n):
        ch, eos, _ = g.stream()
        out.append(ch)
        if eos: break
    return "".join(out)

def mtp_gen(m, tok, drafter, ids, n, window):
    cache = ExLlamaV2Cache_TP(m, base=ExLlamaV2Cache_SWA, max_seq_len=MAXSEQ)
    stop = {106, 1}
    # prefill target (all but last token); then forward the last prompt token to get the
    # bonus-candidate logits (prediction for position pos) and its pre-final-norm state.
    m.forward(ids[:, :-1], cache, preprocess_only=True)
    dev_logits, last_state = m.forward(ids[:, -1:], cache, return_last_state=True)
    last_state = last_state[0, -1, :].view(1, 1, -1)  # (1,1,backbone)
    last_tok = ids[:, -1:].clone()                    # (1,1)
    emitted = []
    accepted = total_draft = 0
    rounds = 0
    t0 = time.perf_counter()
    while len(emitted) < n:
        rounds += 1
        pos = cache.current_seq_len                    # cache holds positions 0..pos-1
        # 1) gather target KV for the drafter (once per round)
        drafter.gather_target(cache)
        # 2) draft a window (greedy); the drafter's query sits at frontier position pos
        draft = drafter.draft_window(last_tok, last_state, pos, window)   # (1, window)
        total_draft += draft.shape[-1]
        # 3) verify: target forward over the draft tokens (positions pos..pos+window-1).
        #    The target's embedding is CPU-resident in TP mode, so token ids must be on CPU.
        vlogits = m.forward(draft.cpu(), cache)                       # (1, window, vocab)
        # target_preds[k] = the target's greedy prediction for position pos+k:
        #   k=0 comes from the previous round's bonus forward (dev_logits);
        #   k>=1 comes from vlogits[0, k-1] (the target processing draft[k-1]).
        target_preds = [int(torch.argmax(dev_logits[0, 0], dim = -1))]
        for i in range(window):
            target_preds.append(int(torch.argmax(vlogits[0, i], dim = -1)))
        # 4) accept the longest prefix of the draft that the target agrees with
        m_acc = 0
        for k in range(window):
            if target_preds[k] == int(draft[0, k]):
                m_acc = k + 1
            else:
                break
        # emit the accepted drafts + the bonus (target_preds[m_acc])
        for i in range(m_acc + 1):
            emitted.append(target_preds[i])
        accepted += m_acc
        if os.environ.get("MTP_VERBOSE"):
            print(f"  round {rounds}: pos={pos} m_acc={m_acc} last={emitted[-1]}", flush=True)
        if int(emitted[-1]) in stop:
            break
        if len(emitted) >= n:
            break
        # 5) next round: rewind the cache to just after the last accepted draft, then
        #    forward the bonus token to write its KV and get its pre-final-norm state.
        cache.current_seq_len = pos + m_acc
        bonus = torch.tensor([[target_preds[m_acc]]], dtype=torch.long)   # CPU
        dev_logits, last_state = m.forward(bonus, cache, return_last_state=True)
        last_state = last_state[0, -1, :].view(1, 1, -1)
        last_tok = bonus
    dt = time.perf_counter() - t0
    text = tok.decode(torch.tensor(emitted[:n], dtype=torch.long).view(1,-1))[0]
    return text, accepted, total_draft, rounds, dt

if os.environ.get("MTP_WATCHDOG"):
    # Dump every thread's stack if the run is still alive after this many seconds.
    faulthandler.dump_traceback_later(int(os.environ.get("MTP_WATCHDOG", 180)), exit = True)

m, tok = load_target()
print(f"loaded target; loading drafter from {DRAFT_DIR}", flush=True)
t0=time.perf_counter()
drafter = Gemma4AssistantDrafter(m, DRAFT_DIR, device=0, num_draft_tokens=WINDOW)
print(f"drafter loaded in {time.perf_counter()-t0:.1f}s, params={drafter.footprint()/2/1e6:.1f}M", flush=True)

q = "What is 2+2?"
ids = make_prompt(tok, q)
print(f"\nprompt: {q!r}  ({ids.shape[-1]} tok)  window={WINDOW}\n", flush=True)

print("=== baseline (no draft) ===", flush=True)
t0=time.perf_counter(); base = baseline(m, tok, ids, 48); tb=time.perf_counter()-t0
print(repr(base), f"  ({48} tok in {tb:.2f}s = {48/tb:.1f} tok/s)", flush=True)

print("\n=== MTP (draft window=%d) ===" % WINDOW, flush=True)
mtp, acc, td, rounds, dt = mtp_gen(m, tok, drafter, ids, 48, WINDOW)
print(repr(mtp), f"  ({48} tok in {dt:.2f}s = {48/dt:.1f} tok/s)", flush=True)
print(f"  accepted={acc} total_draft={td} rounds={rounds} acc/round={acc/rounds:.2f}", flush=True)
print(f"  speedup vs baseline: {tb/dt:.2f}x", flush=True)
print("\nMATCH" if mtp == base else f"\nMISMATCH:\n  base={base!r}\n  mtp ={mtp!r}")
