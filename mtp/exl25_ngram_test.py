"""
EXL2.5 engram validation (single GPU, synthetic table):
  1. converter round-trip: source fp16 table -> EXL2.5 packed rows (4 and 8 bit)
  2. dequant kernel parity vs torch decode of the packed format (rel_l2 < 1e-3)
  3. module forward (RAM + disk-streaming modes) vs original source rows (quant error)
  4. hashing/eos-segmentation vs an independent naive reference

Usage: venv/bin/python exllamav2-Fork/mtp/exl25_ngram_test.py
"""

import sys, os, json, subprocess, tempfile, types
import torch
from safetensors.torch import save_file
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exllamav2.ext import exllamav2_ext as ext_c
from exllamav2.ngram_embedding import ExLlamaV2NGramEmbedding

DEV = "cuda:0"
KEY = "model.ple_embedding"
PARENT = "model"
NGRAM = 4
HPN = 2
HEAD_DIM = 160
NUM_HEADS = (NGRAM - 1) * HPN
PLE = NUM_HEADS * HEAD_DIM
EOS = 0


def is_prime(v):
    if v < 2: return False
    for d in range(2, int(v ** 0.5) + 1):
        if v % d == 0: return False
    return True


def primes_after(start, count):
    p, out = start, []
    for _ in range(count):
        p += 1
        while not is_prime(p): p += 1
        out.append(p)
    return out


def rel_l2(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) /
            torch.linalg.vector_norm(b.float()).clamp_min(1e-12)).item()


def build_source(tmp):
    g = torch.Generator().manual_seed(5)
    sizes = torch.tensor(primes_after(1000, NUM_HEADS), dtype=torch.long)
    offsets = sizes.cumsum(0) - sizes
    mults = torch.tensor(primes_after(100000, NGRAM), dtype=torch.long)
    num_rows = int(offsets[-1] + sizes[-1])
    rows = torch.randn(num_rows, HEAD_DIM, generator=g) * 0.05
    # per-head structure: mean shift + a few outlier dims (bias + scale must absorb these)
    for h in range(NUM_HEADS):
        a, b = int(offsets[h]), int(offsets[h] + sizes[h])
        rows[a:b] += torch.randn(1, HEAD_DIM, generator=g) * 0.3
        rows[a:b, :4] *= 6.0
    rows = rows.half()
    src = os.path.join(tmp, "src")
    os.makedirs(src, exist_ok=True)
    save_file({
        KEY + ".weight": rows,
        PARENT + ".ngram_heads_offsets": offsets,
        PARENT + ".ngram_heads_vocab_sizes": sizes,
        PARENT + ".layer_multipliers": mults,
    }, os.path.join(src, "model.safetensors"))
    return src, rows, offsets, sizes, mults


def run_converter(src, out, bits):
    r = subprocess.run([sys.executable, "-m", "exllamav2.conversion.convert_ngram",
                        "-i", src, "-o", out, "-k", KEY, "--bits", str(bits),
                        "--ngram-size", str(NGRAM), "--heads-per-ngram", str(HPN),
                        "--ple-embed-dim", str(PLE), "--eos", str(EOS), "--device", "0"],
                       capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError("convert_ngram failed")


def load_module(out, stream):
    path = os.path.join(out, "ngram-00001.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        tmap = {k: path for k in f.keys()}
    stub = types.SimpleNamespace(config=types.SimpleNamespace(
        tensor_file_map=tmap, arch=types.SimpleNamespace(keymap={}, lm=None)))
    m = ExLlamaV2NGramEmbedding(stub, KEY, NGRAM, HPN, PLE, EOS,
                                bits=int(json.load(open(os.path.join(out, "ngram_config.json")))["bits"]),
                                head_dim=HEAD_DIM, stream_from_disk=stream, archparams=object())
    m.device_idx = 0
    m.load()
    return m


def naive_ids(history, out_len, offsets, sizes, mults):
    """Independent hashing reference (per-position loops, explicit eos segmentation)."""
    bsz, L = history.shape
    ctx = NGRAM - 1
    ids = torch.zeros(bsz, out_len, NUM_HEADS, dtype=torch.long)
    for b in range(bsz):
        for i, p in enumerate(range(L - out_len, L)):
            last_eos = -1
            # strictly BEFORE p: the eos token's own ngram still sees its real context
            # (exl3 convention: cummax shifted by one)
            for j in range(p - 1, -1, -1):
                if history[b, j].item() == EOS: last_eos = j; break
            for ng in range(2, NGRAM + 1):
                toks = []
                for s in range(ng):
                    src = p - s
                    toks.append(history[b, src].item() if src > last_eos else EOS)
                mixed = toks[0] * int(mults[0])
                for s in range(1, ng):
                    mixed ^= toks[s] * int(mults[s])
                lo = (ng - 2) * HPN
                for h in range(lo, lo + HPN):
                    ids[b, i, h] = mixed % int(sizes[h]) + int(offsets[h])
    return ids


def torch_dequant(packed, scales, heads, bias, bits):
    zero = 1 << (bits - 1)
    vpw = 32 // bits
    u, words = packed.shape
    w = packed.to(torch.int64)
    shifts = torch.arange(vpw, dtype=torch.int64) * bits
    q = ((w.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).view(u, -1)
    return ((q - zero).half() * scales[:, None] + bias[heads.long()])


def main():
    torch.cuda.init()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        src, src_rows, offsets, sizes, mults = build_source(tmp)
        g = torch.Generator().manual_seed(9)
        history = torch.randint(1, 900, (2, NGRAM - 1 + 24), generator=g)
        history[0, 10] = EOS  # mid-sequence eos boundary
        history[1, 3] = EOS

        # hashing reference (mode-independent)
        m_tmp = None
        for bits in (8, 4):
            out = os.path.join(tmp, f"out{bits}")
            run_converter(src, out, bits)
            with safe_open(os.path.join(out, "ngram-00001.safetensors"), framework="pt", device="cpu") as f:
                packed = f.get_tensor(KEY + ".q_rows")
                scales = f.get_tensor(KEY + ".q_row_scales")
                bias = f.get_tensor(KEY + ".head_bias")

            for stream in (False, True):
                m = load_module(out, stream)
                # hashing vs naive reference
                ids = m.compute_ngram_ids(history, 24)
                ref_ids = naive_ids(history, 24, offsets, sizes, mults)
                hash_ok = torch.equal(ids, ref_ids)
                if not hash_ok:
                    print(f"  FAIL {bits}b stream={stream}: hash ids mismatch")
                    print("    module offsets:", m.head_offsets.tolist())
                    print("    source offsets:", offsets.tolist())
                    print("    module sizes:", m.head_vocab_sizes.tolist())
                    print("    module mults:", m.layer_multipliers.tolist(), "source:", mults.tolist())
                    bad = (ids != ref_ids).nonzero()[:5]
                    print("    first bad positions:", bad.tolist())
                    ok = False
                # kernel parity vs torch decode of packed rows (all rows, chunked)
                heads_all = (torch.searchsorted(offsets, torch.arange(packed.shape[0]), right=True) - 1).to(torch.int32)
                for a in range(0, packed.shape[0], 4096):
                    b = min(a + 4096, packed.shape[0])
                    p = packed[a:b].to(DEV); s = scales[a:b].to(DEV)
                    h = heads_all[a:b].to(DEV); bi = bias.to(DEV)
                    got = torch.empty(b - a, HEAD_DIM, dtype=torch.half, device=DEV)
                    ext_c.ngram_dequant_rows(p, s, h, bi, got, bits)
                    ref = torch_dequant(packed[a:b], scales[a:b], heads_all[a:b], bias, bits).to(DEV)
                    e = rel_l2(got, ref)
                    if e > 1e-3:
                        print(f"  FAIL {bits}b kernel parity chunk {a}: {e:.6f}")
                        ok = False
                # end-to-end forward vs original source rows
                y = m.forward(history)
                ref_rows = src_rows[ref_ids.reshape(-1)].view(2, 24, PLE).to(DEV)
                qe = rel_l2(y, ref_rows)
                # synthetic rows carry 6x outliers on 4/160 dims -> per-row scales are
                # outlier-dominated; real trained embeddings are much smoother.
                gate = 0.06 if bits == 8 else 0.10
                good = qe < gate and hash_ok
                ok = ok and good
                print(f"  {bits}b stream={int(stream)}: hash {'PASS' if hash_ok else 'FAIL'}, "
                      f"forward quant rel_l2={qe:.5f} {'PASS' if good else 'FAIL'}")
                m.unload()
    print("ENGRAM:", "ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
