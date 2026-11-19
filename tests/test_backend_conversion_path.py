"""
Prove which attention backend the CONVERSION forward passes actually use.

The conversion pipeline calls module.forward(x, cache, attn_params, intermediates=True)
to capture post_norm / attn_output / pre_down for Hessian accumulation.

attn.py:1419 reads:

    if self.q_handle is None or intermediates:
        return self.forward_torch(...)

so intermediates=True routes through forward_torch, NOT forward_paged. But forward_torch
still selects `_attn_flash` when FlashInfer or flash-attn is available, and `_attn_flash`
calls FlashInfer's single_prefill / varlen_prefill for supported shapes. So FlashInfer *is*
on the conversion path — through a different door than the paged one.

This script instruments the three real attention kernels and reports which fires.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config
from exllamav2 import attn_backend as ab
from exllamav2.attn import ExLlamaV2Attention, use_flashinfer

MODEL = os.environ.get("MODEL_DIR", "/home/westfox35/exl2work/models/gemma-4-31B-it")

counts = {}
def bump(k): counts[k] = counts.get(k, 0) + 1

# FlashInfer non-paged entry points
for name in ["single_prefill", "varlen_prefill"]:
    orig = getattr(ab, name)
    def make(orig, name):
        def f(*a, **kw):
            bump(f"flashinfer.{name}")
            return orig(*a, **kw)
        return f
    setattr(ab, name, make(orig, name))

# FlashInfer paged entry point
orig_plan = ab.FlashInferPaged.plan_and_run
def plan_and_run(self, *a, **kw):
    bump("flashinfer.paged")
    return orig_plan(self, *a, **kw)
ab.FlashInferPaged.plan_and_run = plan_and_run

# Pure-torch reference
orig_torch = ab.torch_paged_attention
def torch_paged(*a, **kw):
    bump("torch")
    return orig_torch(*a, **kw)
ab.torch_paged_attention = torch_paged

orig_attn_torch = ExLlamaV2Attention._attn_torch
def _attn_torch(self, *a, **kw):
    bump("_attn_torch")
    return orig_attn_torch(self, *a, **kw)
ExLlamaV2Attention._attn_torch = _attn_torch

cfg = ExLlamaV2Config()
cfg.model_dir = MODEL
cfg.prepare()
cfg.arch_compat_overrides()
cfg.max_output_len = 16

print(f"has_flashinfer      : {ab.has_flashinfer}")
print(f"use_flashinfer(cfg) : {use_flashinfer(cfg)}")

model = ExLlamaV2(cfg)
model.load(lazy=True)
attn_modules = [m for m in model.modules if isinstance(m, ExLlamaV2Attention)]

# One layer FlashInfer can serve (head_dim 256) and one it cannot (head_dim 512)
targets = []
for hd in (256, 512):
    m = next((m for m in attn_modules if m.head_dim == hd), None)
    if m: targets.append(m)

for m in targets:
    print(f"\n--- {m.key}  head_dim={m.head_dim} ---")
    print(f"    supports(): {ab.FlashInferPaged.supports(m.head_dim, m.num_attention_heads, m.num_key_value_heads, 256, torch.half)}")
    m.load()
    x = torch.randn((1, 64, cfg.hidden_size), dtype=torch.half, device=m.device())
    params = ExLlamaV2Attention.Params(1, 64, 0, None, None)

    before = dict(counts)
    with torch.inference_mode():
        m.forward(x, None, params, intermediates=True)   # exactly the conversion call
    delta = {k: counts.get(k, 0) - before.get(k, 0) for k in set(counts) | set(before)}
    delta = {k: v for k, v in delta.items() if v}
    print(f"    conversion call -> {delta}")

    m.unload()
    torch.cuda.empty_cache()

print(f"\ntotals: {counts}")
fi = sum(v for k, v in counts.items() if k.startswith("flashinfer"))
print()
if fi > 0:
    print("RESULT: FlashInfer IS on the conversion forward path.")
else:
    print("RESULT: FlashInfer is NOT used during conversion.")
