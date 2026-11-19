"""
Probe 2: can a module replica on device 1 run a forward pass and produce a
Hessian contribution identical to the same rows on device 0?

This is the real M3 question. If the Hessian from a sharded row set matches the
Hessian from the same rows run serially, row sharding is correct.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config
from exllamav2.model import ExLlamaV2Attention
from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
from exllamav2.conversion.calib import ensure_device_contexts

MODEL = os.environ.get("MODEL_DIR", "/home/westfox35/exl2work/models/gemma-4-31B-it")

cfg = ExLlamaV2Config()
cfg.model_dir = MODEL
cfg.prepare()
cfg.arch_compat_overrides()
cfg.max_output_len = 16

model = ExLlamaV2(cfg)
model.load(lazy=True)

# The model loader only creates a device_context slot for devices that actually received modules,
# so a model that fits on one card has len(model.device_context) == 1 and a replica on device 1
# dies in get_device_context(1) with IndexError. This is the M3 blocker described in
# quant-plan.md §6.5; ensure_device_contexts() appends the missing slots.
ensure_device_contexts(model, [0, 1])

attn = [m for m in model.modules if isinstance(m, ExLlamaV2Attention)]
src = next(m for m in attn if m.device_idx == 0)
print(f"source: {src.key}  head_dim={src.head_dim}")

# Build a replica on device 1
rep = ExLlamaV2Attention(model, src.key, src.layer_idx,
                          sliding_window=src.sliding_window, rope_index=src.rope_index)
rep.set_device_idx(1)

src.load()
rep.load()
print("both loaded")

# Same input rows, forwarded on each device
g = torch.Generator(device="cpu").manual_seed(1234)
x = torch.randn((1, 32, cfg.hidden_size), generator=g, dtype=torch.float32).to(torch.half)
params0 = ExLlamaV2Attention.Params(1, 32, 0, None, None)
params1 = ExLlamaV2Attention.Params(1, 32, 0, None, None)

with torch.inference_mode():
    o0 = src.forward(x.to("cuda:0"), None, params0, intermediates=True)
    o1 = rep.forward(x.to("cuda:1"), None, params1, intermediates=True)

print("\nintermediate keys:", sorted(o0.keys()))
for k in sorted(o0.keys()):
    a = o0[k].float().cpu()
    b = o1[k].float().cpu()
    d = (a - b).abs().max().item()
    print(f"  {k:24s} shape {tuple(a.shape)}  maxdiff {d:.3e}")

# Hessian from each
q0 = AdaptiveGPTQ(src.q_proj.linear)
q1 = AdaptiveGPTQ(rep.q_proj.linear)
q0.hessian_device = 0
q1.hessian_device = 1
q0.add_batch(o0["post_norm"])
q1.add_batch(o1["post_norm"])

h0 = q0.hessian.float().cpu()
h1 = q1.hessian.float().cpu()
print(f"\nhessian maxdiff: {(h0 - h1).abs().max().item():.3e}")
print(f"hessian allclose: {torch.allclose(h0, h1, rtol=1e-5, atol=1e-5)}")

src.unload()
rep.unload()
