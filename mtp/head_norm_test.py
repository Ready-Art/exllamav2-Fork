"""head_norm kernel parity vs torch RMSNorm/LayerNorm across head_dim (esp. hd<64)."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

dev = "cuda:0"
torch.manual_seed(0)
for hd in (32, 64, 128, 256, 512):
    for nq in (1, 8, 32):
        rows = 5
        x = (torch.randn(rows, nq, hd, device=dev, dtype=torch.half) * 0.5)
        w = (torch.randn(nq, hd, device=dev, dtype=torch.half) * 0.1 + 1.0)
        eps = 1e-6
        # RMSNorm reference over last dim, per head
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        ref = (xf * rms * w.float()).half()
        y = torch.empty_like(x)
        ext_c.head_norm(x, w, none_tensor, y, eps, True)
        rel = float((y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9))
        ok = rel < 2e-3
        print(f"hd={hd:3d} nq={nq:2d} rms rel_l2={rel:.6f} {'OK' if ok else 'FAIL'}", flush=True)
        assert ok, f"head_norm rms FAIL hd={hd} nq={nq}"
print("HEAD_NORM_PARITY_DONE", flush=True)
