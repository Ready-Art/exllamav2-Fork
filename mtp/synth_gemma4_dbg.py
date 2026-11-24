import os, sys, json
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from synth_newmodels_test import make_gemma4_moe_synth, WORK, G4
from safetensors.torch import load_file

MOE = "dense" not in sys.argv
d = os.path.join(WORK, "gemma4moe" if MOE else "gemma4dense")
make_gemma4_moe_synth(d, moe=MOE)
torch.manual_seed(1234)
ids = torch.randint(2, 256, (1, 40))

from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
raw = json.load(open(os.path.join(d, "config.json")))
cfg = Gemma4Config(**raw)
hm = Gemma4ForConditionalGeneration(cfg)
sd = load_file(os.path.join(d, "model.safetensors"))
hm.load_state_dict(sd, strict=False)
hm.tie_weights()
hm = hm.cuda().eval()
if "fp16" in sys.argv:
    hm = hm.half()
with torch.inference_mode():
    r = hm(ids.cuda(), output_hidden_states=True)
hf_hs = [h.float().cpu() for h in r.hidden_states]
print("HF hidden states:", len(hf_hs), "logits", r.logits.shape)

from exllamav2 import ExLlamaV2, ExLlamaV2Config
from exllamav2.cache import ExLlamaV2Cache_SWA
fc = ExLlamaV2Config(d)
fc.max_seq_len = 512
m = ExLlamaV2(fc)
m.load()
cache = ExLlamaV2Cache_SWA(m, max_seq_len=512)
stash = {}
for i, mod in enumerate(m.modules):
    of = mod.forward
    def wrap(x, *a, _of=of, _i=i, **k):
        rr = _of(x, *a, **k)
        t = rr["hidden_states"] if isinstance(rr, dict) else rr
        stash[_i] = t.detach().float().cpu()
        return rr
    mod.forward = wrap
with torch.inference_mode():
    logits = m.forward(ids, cache=cache, last_id_only=True)
    a = logits.float().cpu().flatten(); b = r.logits[:, -1:, :].float().cpu().flatten()
    print("PREFILL-40 last-row cos=", torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    stash_prefill = {k: v.clone() for k, v in stash.items()}
    stride = 3  # attn + mlp(/block) + layer_scalar on every Gemma-4 layer
    for li in range(G4["layers"]):
        idx = 1 + li*stride + (stride - 1)
        fa = stash_prefill[idx][:, -1:, :].flatten()
        fb = hf_hs[li+1][:, -1:, :].flatten()
        print(f"  prefill layer{li} last-row cos=", torch.nn.functional.cosine_similarity(fa, fb, dim=0).item())
    fk_logits = [logits.float().cpu()]
    cur = ids[:, -1:]
    hf_cur = ids[:, -1:].cuda()
    hf_past = r.past_key_values
    for s in range(8):
        logits = m.forward(cur, cache=cache, last_id_only=True)
        fk_logits.append(logits.float().cpu())
        cur = logits.argmax(-1).cpu()
        hr = hm(hf_cur, past_key_values=hf_past, use_cache=True)
        hf_past = hr.past_key_values
        hf_logits_step = hr.logits[:, -1:, :].float().cpu()
        a = fk_logits[-1].flatten(); b = hf_logits_step.flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        print(f"step{s} cos={cos:.5f} fk_argmax={cur.item()} hf_argmax={hf_logits_step.argmax(-1).item()}")
        hf_cur = hr.logits[:, -1:, :].argmax(-1)
print("fork modules:", [(i, mod.key, mod.name) for i, mod in enumerate(m.modules)][:8])

def cmp(a, b, name):
    a = a.flatten().float(); b = b.flatten().float()
    n = min(a.numel(), b.numel())
    cos = torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0).item()
    rel = ((a[:n]-b[:n]).norm()/(b[:n].norm()+1e-9)).item()
    print(f"{name:40s} cos={cos:.5f} rel={rel:.5f} shapes {tuple(a.shape)} {tuple(b.shape)}")

# embedding
cmp(stash[0], hf_hs[0], "embed")
# after layer 0 = module index 3 (attn0=1, moe0=2, scalar0=3)
for li in range(min(3, G4["layers"])):
    idx = 1 + li*3 + 2
    cmp(stash[idx], hf_hs[li+1], f"layer{li} out (mod {idx})")
cmp(logits.float().cpu(), r.logits.float().cpu(), "logits")
