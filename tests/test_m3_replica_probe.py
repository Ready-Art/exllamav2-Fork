"""
Probe: can a module be replicated onto a second device?

M3 (calibration row sharding) needs one module replica per device. This script
answers the question the plan flags as the blocker: does constructing a second
ExLlamaV2Attention against the same model, with device_idx = 1, actually work?

Two things have to hold:
  1. model.device_context[1] must exist, or get_device_context(1) must create it.
  2. module.load() must succeed on device 1 while the device-0 module is loaded.

If either fails, M3 cannot be implemented without changing the model loader, and
that is a much larger change than the plan assumes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config
from exllamav2.model import ExLlamaV2Attention, ExLlamaV2MLP

MODEL = os.environ.get("MODEL_DIR", "/home/westfox35/exl2work/models/gemma-4-31B-it")

cfg = ExLlamaV2Config()
cfg.model_dir = MODEL
cfg.prepare()
cfg.arch_compat_overrides()
cfg.max_output_len = 16

# Load lazily with a split that covers two devices, so device_context has two slots.
model = ExLlamaV2(cfg)
try:
    model.load(gpu_split=[12, 12], lazy=True)
    print("loaded with gpu_split=[12,12]")
except Exception as e:
    print(f"gpu_split=[12,12] failed: {e}")
    model = ExLlamaV2(cfg)
    model.load(lazy=True)
    print("loaded with default split")

print(f"device_context slots: {len(model.device_context)}")
for i, dc in enumerate(model.device_context):
    print(f"  [{i}] {'None' if dc is None else 'ready=' + str(dc.ready)}")

attn = [m for m in model.modules if isinstance(m, ExLlamaV2Attention)]
print(f"\nattention modules: {len(attn)}")

# Find a module assigned to device 0 that we can replicate.
src = next((m for m in attn if m.device_idx == 0), None)
if src is None:
    print("no attention module on device 0")
    sys.exit()

print(f"\nsource module: {src.key}  device_idx={src.device_idx}  head_dim={src.head_dim}")
print(f"weight footprint: {src.weight_footprint() >> 20} MiB")

# --- Attempt 1: construct a replica with device_idx = 1 ---
print("\n--- attempt: replicate onto device 1 ---")
try:
    rep = ExLlamaV2Attention(model, src.key, src.layer_idx,
                             sliding_window=src.sliding_window, rope_index=src.rope_index)
    rep.set_device_idx(1)
    print("  constructed + set_device_idx(1): OK")

    # Does the model have a device context for 1?
    try:
        dc = model.get_device_context(1)
        print(f"  get_device_context(1): {'None' if dc is None else 'ok'}")
    except Exception as e:
        print(f"  get_device_context(1) FAILED: {type(e).__name__}: {e}")

    src.load()
    print("  source loaded on device 0: OK")
    try:
        rep.load()
        print("  replica loaded on device 1: OK")
        ok = True
    except Exception as e:
        print(f"  replica load FAILED: {type(e).__name__}: {e}")
        ok = False
    src.unload()
    try:
        rep.unload()
    except Exception:
        pass
except Exception as e:
    print(f"  construct FAILED: {type(e).__name__}: {e}")
