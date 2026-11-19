"""
Prove which attention backend the conversion forward passes actually use.

The conversion pipeline (measure.py, quantize.py) runs module.forward() to build Hessians and
target states. If FlashInfer is enabled but silently falling back to the torch path, every
timing number from the multi-GPU work is measured against the wrong baseline.

This script loads real modules from the model and counts which branch fires.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config
from exllamav2 import attn_backend as ab
from exllamav2.attn import ExLlamaV2Attention, use_flashinfer

MODEL = os.environ.get("MODEL_DIR", "/home/westfox35/exl2work/models/gemma-4-31B-it")

counts = {"flashinfer": 0, "flash_attn": 0, "torch": 0}
orig_plan = ab.FlashInferPaged.plan_and_run
orig_torch = ab.torch_paged_attention

def plan_and_run(self, *a, **kw):
    counts["flashinfer"] += 1
    return orig_plan(self, *a, **kw)

def torch_paged(*a, **kw):
    counts["torch"] += 1
    return orig_torch(*a, **kw)

ab.FlashInferPaged.plan_and_run = plan_and_run
ab.torch_paged_attention = torch_paged

cfg = ExLlamaV2Config()
cfg.model_dir = MODEL
cfg.prepare()
cfg.arch_compat_overrides()
cfg.max_output_len = 16

print(f"has_flashinfer         : {ab.has_flashinfer}")
print(f"use_flashinfer(cfg)    : {use_flashinfer(cfg)}")
print(f"no_flashinfer          : {getattr(cfg, 'no_flashinfer', None)}")
print(f"no_flash_attn           : {cfg.no_flash_attn}")

model = ExLlamaV2(cfg)
model.load(lazy=True)

# Find the first two attention modules: Gemma-4 alternates sliding-window (head_dim 256)
# and full attention (head_dim 512), so we want one of each.

attn_modules = [m for m in model.modules if isinstance(m, ExLlamaV2Attention)]
print(f"attention modules       : {len(attn_modules)}")

picked = []
seen_dims = set()
for m in attn_modules:
    if m.head_dim not in seen_dims:
        seen_dims.add(m.head_dim)
        picked.append(m)
    if len(picked) >= 2: break

for m in picked:
    print(f"\n--- {m.key}  head_dim={m.head_dim}  q_heads={m.num_attention_heads}  kv_heads={m.num_key_value_heads} ---")
    print(f"    supports(): {ab.FlashInferPaged.supports(m.head_dim, m.num_attention_heads, m.num_key_value_heads, 256, torch.half)}")

    m.load()
    x = torch.randn((1, 8, cfg.hidden_size), dtype=torch.half, device=m.device())
    params = ExLlamaV2Attention.Params(1, 8, 0, None, None)
    before = dict(counts)
    with torch.inference_mode():
        m.forward(x, None, params)
    delta = {k: counts[k] - before[k] for k in counts}
    hit = [k for k, v in delta.items() if v > 0]
    print(f"    backend hit: {hit}  {delta}")
    m.unload()
    torch.cuda.empty_cache()

print(f"\ntotals: {counts}")
if counts["flashinfer"] > 0:
    print("RESULT: FlashInfer IS on the conversion forward path.")
else:
    print("RESULT: FlashInfer is NOT being hit — conversion runs on the fallback path.")
