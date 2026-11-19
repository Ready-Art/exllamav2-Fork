"""
M3 correctness: sharded calibration rows must produce the same Hessian as serial.

The merge is a plain sum because H is an unnormalized sum over token batches.
If the sharded Hessian differs from the serial one, the shard or the merge is wrong.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config
from exllamav2.model import ExLlamaV2Attention
from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
from exllamav2.conversion.calib import (
    shard_calibration_rows, ensure_device_contexts, module_replica_bytes, calibration_row_shards
)

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

devices = [0, 1]
ensure_device_contexts(model, devices)
print(f"device_context slots: {len(model.device_context)}")

rep = ExLlamaV2Attention(model, src.key, src.layer_idx,
                         sliding_window=src.sliding_window, rope_index=src.rope_index)
rep.set_device_idx(1)
src.load()
rep.load()
print(f"replica bytes: {module_replica_bytes(src) >> 20} MiB")

# 8 calibration rows
g = torch.Generator(device="cpu").manual_seed(7)
rows = [torch.randn((1, 16, cfg.hidden_size), generator=g, dtype=torch.float32).to(torch.half)
        for _ in range(8)]

def make_quantizers():
    q = AdaptiveGPTQ(src.q_proj.linear)
    q.hessian_device = 0
    return {"q_proj": q}

def row_fn(device_idx, active, row_indices):
    """Forward the given rows through ``active`` and accumulate H."""
    if active is None: active = src
    qz = make_quantizers()
    ts = []
    for i in row_indices:
        x = rows[i].to(torch.device(device_idx))
        o = active.forward(x, None, ExLlamaV2Attention.Params(1, 16, 0, None, None), intermediates=True)
        qz["q_proj"].add_batch(o["post_norm"])
        ts.append(o["hidden_states"].to("cpu"))
    return qz, ts, [], [], None

# --- serial reference ---
q_ser, ts_ser, _, _, _ = row_fn(0, src, list(range(len(rows))))
h_ser = q_ser["q_proj"].hessian.float().cpu()
print(f"\nserial   hessian: shape {tuple(h_ser.shape)}  maxabs {h_ser.abs().max().item():.4e}  batches {q_ser['q_proj'].num_batches}")

# --- sharded ---
q_sh, ts_sh, ue = shard_calibration_rows(rows, [rep], devices, None, row_fn)
h_sh = q_sh["q_proj"].hessian.float().cpu()
print(f"sharded  hessian: shape {tuple(h_sh.shape)}  maxabs {h_sh.abs().max().item():.4e}  batches {q_sh['q_proj'].num_batches}")

d = (h_ser - h_sh).abs().max().item()
rel = d / max(h_ser.abs().max().item(), 1e-30)
print(f"\nhessian maxdiff: {d:.4e}   relative: {rel:.4e}")
print(f"hessian allclose(rtol=1e-4): {torch.allclose(h_ser, h_sh, rtol=1e-4, atol=1e-4)}")

print(f"\ntarget states: {len(ts_sh)} rows")
ok = all(torch.equal(ts_ser[i], ts_sh[i]) for i in range(len(rows)))
print(f"target states bit-identical: {ok}")
for i in range(len(rows)):
    if not torch.equal(ts_ser[i], ts_sh[i]):
        print(f"  row {i} maxdiff {(ts_ser[i].float()-ts_sh[i].float()).abs().max().item():.3e}")

src.unload()
rep.unload()
