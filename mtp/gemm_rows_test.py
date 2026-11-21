"""
EXL2 q_gemm row-count parity + timing sweep (m8 kernel validation).

Single GPU, no TP, no full model load: builds ONE real QMatrix from a target
checkpoint tensor (first mlp.gate_proj found), reconstructs it to fp16 as the
reference, then for m = 1..8, 12, 16:
  - correctness: gemm_half_q_half (force_cuda) vs fp32 matmul on the reference
  - timing: CUDA-event ms/call after warmup (the m=5 jump diagnostic)

Run:  venv/bin/python exllamav2-Fork/mtp/gemm_rows_test.py [checkpoint_dir]
Exit code 0 = all rows within tolerance.
"""

import os
import sys
import glob
import torch

DEFAULT_CKPT = os.path.expanduser(
    "~/exl2work/models/Gemma-4-31B-IT-EXL2-6.0BPW-HB16")

REL_TOL = 0.02   # fp16 per-group accumulation + half atomicAdd across k-blocks
COS_TOL = 0.999


def find_gate_proj(ckpt):
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
    assert files, f"no safetensors in {ckpt}"
    for path in files:
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            cands = [k for k in keys
                     if k.endswith(".q_weight") and "gate_proj" in k]
            if not cands:
                cands = [k for k in keys if k.endswith(".q_weight")]
            if not cands:
                continue
            prefix = cands[0][: -len("q_weight")]
            w = {}
            for suffix in ("q_weight", "q_perm", "q_invperm", "q_scale",
                           "q_scale_max", "q_groups", "bias"):
                k = prefix + suffix
                if k in keys:
                    w[suffix] = f.get_tensor(k)
            print(f"[gemm-rows] tensor: {prefix}  ({path})", flush=True)
            return w
    raise RuntimeError("no .q_weight tensor found in checkpoint")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    dev = "cuda:0"
    torch.manual_seed(1234)

    from exllamav2 import ext
    from exllamav2.ext import exllamav2_ext as ext_c

    w = find_gate_proj(ckpt)
    # EXL2 checkpoints store only q_invperm; load_weight synthesizes q_perm
    if "q_perm" not in w and "q_invperm" in w:
        w["q_perm"] = torch.argsort(w["q_invperm"]).to(torch.int)
    # Optional TP column shard (raw-tensor slice, same as tp_split_col):
    # argv[2] = shard width (e.g. 5376 for 4-GPU, 10752 for 2-GPU gate_proj)
    shard_n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if shard_n:
        assert shard_n % 8 == 0 and shard_n <= w["q_weight"].shape[1]
        w["q_weight"] = w["q_weight"][:, :shard_n].contiguous()
        w["q_scale"] = w["q_scale"][:, :shard_n // 8].contiguous()
        if "bias" in w:
            w["bias"] = w["bias"][:shard_n].contiguous()
        print(f"[gemm-rows] column shard n={shard_n}", flush=True)
    for k in w:
        w[k] = w[k].to(dev)
    if w["q_groups"].dtype != torch.int16:
        w["q_groups"] = w["q_groups"].short()

    q_weight = w["q_weight"]
    n = q_weight.shape[1]
    groups = w["q_scale"].shape[0]
    # q_groups is flat [bits, qrow_start, ...] pairs (see make_group_map_py);
    # height k = sum of logical rows per group
    gr = w["q_groups"].tolist()
    num_groups = len(gr) // 2
    num_qrows = q_weight.shape[0]
    k = 0
    for i in range(num_groups):
        bits = gr[i * 2]
        qrows = (gr[i * 2 + 3] - gr[i * 2 + 1]) if i < num_groups - 1 \
            else (num_qrows - gr[i * 2 + 1])
        k += qrows * 32 // bits
    print(f"[gemm-rows] shape k={k} n={n} groups={groups}", flush=True)

    temp_dq = torch.empty(k * n, dtype=torch.half, device=dev)
    handle = ext.make_q_matrix(w, temp_dq, prescale=1, max_dq_rows=k)

    dq = torch.empty((k, n), dtype=torch.half, device=dev)
    ext_c.reconstruct(handle, dq)
    dq_f = dq.float()

    print(f"[gemm-rows] {'m':>3} {'ms/call':>9} {'rel_l2':>10} {'cos':>9}",
          flush=True)
    ok = True
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16):
        a = torch.randn(m, k, dtype=torch.half, device=dev)
        c = torch.empty(m, n, dtype=torch.half, device=dev)
        ref = a.float() @ dq_f

        ext_c.gemm_half_q_half(a, handle, c, True)
        torch.cuda.synchronize()
        cf = c.float()
        rel = (cf - ref).norm().item() / max(ref.norm().item(), 1e-12)
        cos = torch.nn.functional.cosine_similarity(
            cf.flatten(), ref.flatten(), dim=0).item()

        for _ in range(30):
            ext_c.gemm_half_q_half(a, handle, c, True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            ext_c.gemm_half_q_half(a, handle, c, True)
        stop.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(stop) / 100

        good = rel < REL_TOL and cos > COS_TOL
        ok = ok and good
        print(f"[gemm-rows] {m:>3} {ms:>9.4f} {rel:>10.5f} {cos:>9.6f}"
              f"  {'OK' if good else 'FAIL'}", flush=True)

    print(f"[gemm-rows] {'ALL PASS' if ok else 'FAILURES PRESENT'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
