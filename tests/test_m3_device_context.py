"""
Probe 3: extend model.device_context to cover every device, then run a replica
forward pass on device 1 and compare Hessians with device 0.

If this works, M3 is implementable: the only thing missing is a device context per
device, which ExLlamaV2DeviceContext can supply directly.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config
from exllamav2.model import ExLlamaV2Attention
from exllamav2.device import ExLlamaV2DeviceContext
from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ

MODEL = os.environ.get("MODEL_DIR", "/home/westfox35/exl2work/models/gemma-4-31B-it")

cfg = ExLlamaV2Config()
cfg.model_dir = MODEL
cfg.prepare()
cfg.arch_compat_overrides()
cfg.max_output_len = 16

model = ExLlamaV2(cfg)
model.load(lazy=True)

attn = [m for m in model.modules if isinstance(m, ExLlamaV2Attention)]
src = next(m for m in attn if m.device_idx == 0)
print(f"source: {src.key}  head_dim={src.head_dim}")
print(f"device_context slots before: {len(model.device_context)}")

# Extend device_context so every device has a slot. This is what M3 needs.
scratch_bytes = max(m.scratch_space_fixed() for m in model.modules)
print(f"scratch_bytes: {scratch_bytes >> 20} MiB")
while len(model.device_context) < torch.cuda.device_count():
    idx = len(model.device_context)
    model.device_context.append(ExLlamaV2DeviceContext(model, idx, scratch_bytes))
print(f"device_context slots after: {len(model.device_context)}")

rep = ExLlamaV2Attention(model, src.key, src.layer_idx,
                         sliding_window=src.sliding_window, rope_index=src.rope_index)
rep.set_device_idx(1)

src.load()
rep.load()
print("both loaded")

g = torch.Generator(device="cpu").manual_seed(1234)
x = torch.randn((1, 32, cfg.hidden_size), generator=g, dtype=torch.float32).to(torch.half)

with torch.inference_mode():
    o0 = src.forward(x.to("cuda:0"), None, ExLlamaV2Attention.Params(1, 32, 0, None, None), intermediates=True)
    o1 = rep.forward(x.to("cuda:1"), None, ExLlamaV2Attention.Params(1, 32, 0, None, None), intermediates=True)

print("\nintermediates:")
for k in sorted(o0.keys()):
    a = o0[k].float().cpu()
    b = o1[k].float().cpu()
    print(f"  {k:24s} shape {tuple(a.shape)}  maxdiff {(a - b).abs().max().item():.3e}")

q0 = AdaptiveGPTQ(src.q_proj.linear); q0.hessian_device = 0
q1 = AdaptiveGPTQ(rep.q_proj.linear); q1.hessian_device = 1
q0.add_batch(o0["post_norm"])
q1.add_batch(o1["post_norm"])

h0 = q0.hessian.float().cpu()
h1 = q1.hessian.float().cpu()
print(f"\nhessian maxdiff: {(h0 - h1).abs().max().item():.3e}")
print(f"hessian allclose(rtol=1e-4): {torch.allclose(h0, h1, rtol=1e-4, atol=1e-4)}")
print(f"hessian max abs: {h0.abs().max().item():.3e}")

src.unload()
rep.unload()
