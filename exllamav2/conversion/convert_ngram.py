"""
EXL2.5 engram converter: hashed n-gram embedding table -> scalar uniform-bit rows.

Reads a source table (<key>.weight or <key>.shard_N.weight safetensors) plus the model's
hash params, computes a per-head bias (sampled row mean), quantizes each row symmetrically
to 4 or 8 bits around that bias, and writes the EXL2.5 engram format consumed by
exllamav2.ngram_embedding.ExLlamaV2NGramEmbedding:

    <key>.q_rows            int32 (num_rows, head_dim*bits/32)
    <key>.q_row_scales      fp16  (num_rows,)
    <key>.head_offsets      int64 (num_heads,)
    <key>.head_vocab_sizes  int64 (num_heads,)
    <key>.layer_multipliers int64 (ngram_size,)
    <key>.head_bias         fp16  (num_heads, head_dim)

NO trellis: rows are plain scalar quantization (lookup is bandwidth-bound; dequant is a
few int ops per value, sm86-friendly). 8-bit ~= DeepSeek's native FP8 row quality at the
same byte count; 4-bit halves the table.

Hash params are taken from the source checkpoint (<parent>.ngram_heads_offsets /
.ngram_heads_vocab_sizes / .layer_multipliers, or <key>.head_*). If absent, they are
derived (per-head prime moduli after --vocab, fixed prime multipliers) -- suitable for
tests/synthetic tables; a trained model MUST supply its own params.

Usage:
  python -m exllamav2.conversion.convert_ngram -i MODEL_DIR -o OUT_DIR -k model.ple_embedding \
      --bits 8 --ngram-size 4 --heads-per-ngram 8 --ple-embed-dim 3840 --eos 0 [--vocab 151936]
"""

import argparse
import json
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def find_tensor_files(model_dir):
    files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    index = {}
    for path in files:
        with safe_open(path, framework = "pt", device = "cpu") as f:
            for k in f.keys():
                index[k] = path
    return index


def is_prime(v):
    if v < 2: return False
    if v % 2 == 0: return v == 2
    for d in range(3, int(v ** 0.5) + 1, 2):
        if v % d == 0: return False
    return True


def nth_prime_after(start, count):
    p = start
    out = []
    for _ in range(count):
        p += 1
        while not is_prime(p): p += 1
        out.append(p)
    return out


def pack_rows(q, bits):
    # q: (N, dim) integer quantized values in [0, 2^bits); natural bit order
    n, dim = q.shape
    vpw = 32 // bits
    assert dim % vpw == 0
    q = q.view(n, dim // vpw, vpw).to(torch.int64)
    shifts = torch.arange(vpw, dtype = torch.int64, device = q.device) * bits
    words = (q << shifts).sum(dim = -1)
    return words.to(torch.int32).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_dir", required = True)
    ap.add_argument("-o", "--out_dir", required = True)
    ap.add_argument("-k", "--key", required = True)
    ap.add_argument("--bits", type = int, default = 8, choices = [4, 8])
    ap.add_argument("--ngram-size", type = int, required = True)
    ap.add_argument("--heads-per-ngram", type = int, required = True)
    ap.add_argument("--ple-embed-dim", type = int, required = True)
    ap.add_argument("--head-dim", type = int, default = 160)
    ap.add_argument("--eos", type = int, default = 0)
    ap.add_argument("--vocab", type = int, default = 151936, help = "base for derived prime moduli")
    ap.add_argument("--device", type = int, default = 0)
    ap.add_argument("--chunk", type = int, default = 65536)
    ap.add_argument("--bias-sample-rows", type = int, default = 131072)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.device}")
    num_heads = (args.ngram_size - 1) * args.heads_per_ngram
    assert args.ple_embed_dim == num_heads * args.head_dim
    zero = 1 << (args.bits - 1)
    maxq = (1 << args.bits) - 1
    denom = zero - 1

    index = find_tensor_files(args.in_dir)
    key = args.key
    parent = key.rsplit(".", 1)[0]

    # Source table shards, in order
    shard_keys = sorted([k for k in index if k == key + ".weight" or
                         (k.startswith(key + ".shard_") and k.endswith(".weight"))],
                        key = lambda k: (0 if k == key + ".weight" else int(k.split(".shard_")[1].split(".")[0])))
    assert shard_keys, f"No source table tensors for {key}"

    # Hash params
    def find_aux(*names):
        for n in names:
            if n in index:
                with safe_open(index[n], framework = "pt", device = "cpu") as f:
                    return f.get_tensor(n)
        return None

    head_offsets = find_aux(parent + ".ngram_heads_offsets", key + ".head_offsets")
    head_vocab_sizes = find_aux(parent + ".ngram_heads_vocab_sizes", key + ".head_vocab_sizes")
    layer_multipliers = find_aux(parent + ".layer_multipliers", key + ".layer_multipliers")
    if head_offsets is None:
        sizes = torch.tensor(nth_prime_after(args.vocab, num_heads), dtype = torch.long)
        head_vocab_sizes = sizes
        head_offsets = sizes.cumsum(0) - sizes
        layer_multipliers = torch.tensor(nth_prime_after(1000000, args.ngram_size), dtype = torch.long)
        print(" !! Hash params not in checkpoint -- derived (test/synthetic use only)")
    head_offsets = head_offsets.long()
    head_vocab_sizes = head_vocab_sizes.long()
    layer_multipliers = layer_multipliers.long()
    num_rows = int(head_offsets[-1].item() + head_vocab_sizes[-1].item())

    # Row reader across shards
    handles = {}
    shard_ranges = []
    base = 0
    for sk in shard_keys:
        f = safe_open(index[sk], framework = "pt", device = "cpu")
        handles[sk] = f
        t = f.get_slice(sk)
        n = t.get_shape()[0]
        shard_ranges.append((base, base + n, sk))
        base += n
    assert base == num_rows, f"source rows {base} != hash-param rows {num_rows}"

    def read_rows(a, b):
        parts = []
        for lo, hi, sk in shard_ranges:
            if b <= lo or a >= hi: continue
            ra, rb = max(a, lo) - lo, min(b, hi) - lo
            parts.append(handles[sk].get_slice(sk)[ra:rb])
        return torch.cat(parts, dim = 0) if len(parts) > 1 else parts[0]

    def head_of(a, b):
        idx = torch.arange(a, b)
        return (torch.searchsorted(head_offsets, idx, right = True) - 1).clamp(0, num_heads - 1)

    # Per-head bias: sampled row mean (exl3 convention)
    print(f" -- Computing per-head bias ({num_heads} heads)")
    bias = torch.zeros(num_heads, args.head_dim, dtype = torch.float)
    for h in range(num_heads):
        off = int(head_offsets[h].item())
        size = int(head_vocab_sizes[h].item())
        n = min(size, args.bias_sample_rows)
        step = max(1, size // n)
        idx = torch.arange(0, size, step)[:n] + off
        rows = read_rows(int(idx[0].item()), int(idx[-1].item()) + 1)
        rows = rows.index_select(0, idx - int(idx[0].item())).float()
        bias[h] = rows.mean(dim = 0)
    bias = bias.half()

    # Quantize + pack, streaming
    print(f" -- Quantizing {num_rows} rows x {args.head_dim} @ {args.bits} bit")
    packed_chunks = []
    scale_chunks = []
    for a in range(0, num_rows, args.chunk):
        b = min(a + args.chunk, num_rows)
        rows = read_rows(a, b).to(dev).float()
        heads = head_of(a, b)
        resid = rows - bias[heads].to(dev).float()
        scale = resid.abs().amax(dim = 1).clamp_min(1e-12) / denom
        q = torch.clamp(torch.round(resid / scale[:, None]) + zero, 0, maxq).to(torch.int64)
        packed_chunks.append(pack_rows(q, args.bits).cpu())
        scale_chunks.append(scale.half().cpu())
        if (a // args.chunk) % 64 == 0:
            print(f"    {b}/{num_rows} rows")

    os.makedirs(args.out_dir, exist_ok = True)
    out = {
        key + ".q_rows": torch.cat(packed_chunks, dim = 0),
        key + ".q_row_scales": torch.cat(scale_chunks, dim = 0),
        key + ".head_offsets": head_offsets,
        key + ".head_vocab_sizes": head_vocab_sizes,
        key + ".layer_multipliers": layer_multipliers,
        key + ".head_bias": bias,
    }
    out_path = os.path.join(args.out_dir, "ngram-00001.safetensors")
    save_file(out, out_path)
    cfg = {
        "key": key, "ngram_size": args.ngram_size, "heads_per_ngram": args.heads_per_ngram,
        "ple_embed_dim": args.ple_embed_dim, "head_dim": args.head_dim,
        "bits": args.bits, "eos_token_id": args.eos, "num_rows": num_rows,
    }
    with open(os.path.join(args.out_dir, "ngram_config.json"), "w") as f:
        json.dump(cfg, f, indent = 4)
    print(f" -- Wrote {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
