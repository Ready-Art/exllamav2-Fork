"""P9 MoE expert offload functional test.

  make : build a tiny random-weight Mixtral (2 layers, 8 experts) + cal parquet
  run  : load with EXL2_MOE_OFFLOAD=N, greedy-generate, print ids + VRAM

Run N=0/4/8 in fresh processes: generated ids must be IDENTICAL (staging copies the
exact quantized weights) and VRAM must drop as N grows.
"""
import json, os, sys, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
SYNTH = "/tmp/synth_moe"


def make():
    from safetensors.torch import save_file
    import pandas as pd
    torch.manual_seed(42)
    # Qwen3MoE: supports_tp=True AND is_moe=True in the fork (Mixtral has no TP path).
    # V must cover the borrowed Gemma tokenizer's id range (262k), else calibration
    # embedding lookups device-assert.
    # Dims large enough that every TP split (KV heads, ID//128, RS//32, Q) covers all
    # ranks at 2/3/4-way with the fork's min granularity: H=1024 -> RS 32 units,
    # I=2048 -> ID 16 units, NKV=4 >= world, NH=32 (NH*HD == H).
    H, I, L, E, V, NH, NKV, HD = 1024, 2048, 2, 8, 262144, 32, 4, 32
    cfg = {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "hidden_size": H, "moe_intermediate_size": I, "intermediate_size": I,
        "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": NKV,
        "head_dim": HD, "num_experts": E, "num_experts_per_tok": 2,
        "decoder_sparse_step": 1, "norm_topk_prob": True, "mlp_only_layers": [],
        "vocab_size": V, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
        "rope_theta": 1e6, "torch_dtype": "float16", "tie_word_embeddings": False,
        "attention_bias": False, "attention_dropout": 0.0, "initializer_range": 0.02,
    }
    os.makedirs(SYNTH, exist_ok=True)
    json.dump(cfg, open(f"{SYNTH}/config.json", "w"), indent=1)
    w = {}
    def rn(*s): return (torch.randn(*s) * 0.02).half()
    w["model.embed_tokens.weight"] = rn(V, H)
    w["model.norm.weight"] = torch.ones(H).half()
    w["lm_head.weight"] = rn(V, H)
    for i in range(L):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = torch.ones(H).half()
        w[p + "post_attention_layernorm.weight"] = torch.ones(H).half()
        w[p + "self_attn.q_proj.weight"] = rn(NH * HD, H)
        w[p + "self_attn.k_proj.weight"] = rn(NKV * HD, H)
        w[p + "self_attn.v_proj.weight"] = rn(NKV * HD, H)
        w[p + "self_attn.o_proj.weight"] = rn(H, NH * HD)
        w[p + "self_attn.q_norm.weight"] = torch.ones(HD).half()
        w[p + "self_attn.k_norm.weight"] = torch.ones(HD).half()
        w[p + "mlp.gate.weight"] = rn(E, H)
        for e in range(E):
            ep = p + f"mlp.experts.{e}."
            w[ep + "gate_proj.weight"] = rn(I, H)
            w[ep + "up_proj.weight"] = rn(I, H)
            w[ep + "down_proj.weight"] = rn(H, I)
    save_file(w, f"{SYNTH}/model.safetensors")
    # tokenizer: borrow a HF fast tokenizer json from the Gemma EXL2.5 model
    src = "/home/westfox35/exl2work/models/Gemma-4-31B-IT-EXL25-6U-HAD-AWQ-HB16"
    import shutil
    for f in ("tokenizer.json", "tokenizer_config.json"):
        if os.path.exists(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", f"{SYNTH}/{f}")
    txt = ["The quick brown fox jumps over the lazy dog. " * 24,
           "Register allocation maps virtual registers onto physical files. " * 24,
           "Ocean circulation is driven by density gradients. " * 24,
           "Quantum entanglement correlates measurement outcomes. " * 24] * 4
    pd.DataFrame({"text": txt}).to_parquet("/tmp/synth_cal.parquet")
    print("synth model + cal written", flush=True)


def run():
    from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer, ExLlamaV2
    from exllamav2.cache import ExLlamaV2Cache
    N = os.environ.get("EXL2_MOE_OFFLOAD", "0")
    os.environ["EXL2_MOE_DBG"] = "sd"
    os.environ["EXL2_ATTN_DBG"] = "sd"
    c = ExLlamaV2Config(); c.model_dir = SYNTH + "_exl2"; c.prepare()
    c.max_seq_len = 1024
    tok = ExLlamaV2Tokenizer(c)
    m = ExLlamaV2(c)
    m.load()
    cache = ExLlamaV2Cache(m, max_seq_len = 1024)
    ids = tok.encode("The quick brown fox jumps over the lazy dog.", encode_special_tokens = True)
    logits = m.forward(ids, cache = cache)
    pf = logits[0, -1].float()
    pf_sum = float(pf.sum()); pf_abs = float(pf.abs().sum())
    pf_top = torch.topk(pf, 5).indices.tolist()
    torch.save(pf.cpu(), f"/tmp/pf_sd_{N}.pt")
    gen = []
    lg = logits[:, -1:, :]
    for _ in range(24):
        t = lg.argmax(dim = -1).cpu()   # fork keeps token embeddings on CPU (embed_cpu)
        gen.append(int(t[0, 0]))
        lg = m.forward(t, cache = cache, last_id_only = True)
    vram = torch.cuda.memory_allocated() / 2**20
    print(f"SD OFFLOAD={N} vram={vram:.1f}MiB PREFILL sum={pf_sum:.4f} abs={pf_abs:.4f} "
          f"top5={pf_top} gen={gen}", flush=True)


def run_mp():
    # MP path (replicated MoE per rank). Mixtral has no per-layer geometry -> SEQPAR=0.
    os.environ.setdefault("EXL2_TP_SEQPAR", "0")
    from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.tp_mp import TPProcessGroup
    N = os.environ.get("EXL2_MOE_OFFLOAD", "0")
    c = ExLlamaV2Config(); c.model_dir = SYNTH + "_exl2"; c.prepare()
    c.max_seq_len = 1024; c.chunk_size = 256; c.max_input_len = 256
    tok = ExLlamaV2Tokenizer(c)
    ngpu = int(os.environ.get("MOE_TEST_GPUS", "4"))
    _mtag = f"g{ngpu}_{'sh' if os.environ.get('EXL2_MOE_TP_SHARD','auto')!='0' else 'repl'}"
    os.environ["EXL2_MOE_DBG"] = _mtag
    os.environ["EXL2_ATTN_DBG"] = _mtag
    pg = TPProcessGroup(SYNTH + "_exl2", [2.0] * ngpu, 1024, 1024, "ExLlamaV2Cache")
    pg.start()
    ids = tok.encode("The quick brown fox jumps over the lazy dog.", encode_special_tokens = True)
    lg = pg.forward(ids, 0, ids.shape[-1], 1, True)
    # Prefill-logits parity signature (deterministic for the same input; greedy decode
    # ids on random weights are chaotic, so compare THESE across shard/replicate/SD).
    pf = lg[0, -1].float()
    pf_sum = float(pf.sum())
    pf_top = torch.topk(pf, 5).indices.tolist()
    pf_abs = float(pf.abs().sum())
    _tag = f"g{ngpu}_{'sh' if os.environ.get('EXL2_MOE_TP_SHARD','auto')!='0' else 'repl'}_{N}"
    torch.save(pf.cpu(), f"/tmp/pf_{_tag}.pt")
    gen = []
    nxt = int(pf.argmax())
    past = ids.shape[-1]
    gen.append(nxt)
    for _ in range(24):
        lg = pg.forward(torch.tensor([[nxt]]), past, 1, 1, True)
        nxt = int(lg[0, -1].argmax()); past += 1; gen.append(nxt)
    print(f"MP GPUS={ngpu} OFFLOAD={N} SHARD={os.environ.get('EXL2_MOE_TP_SHARD','auto')} "
          f"PREFILL sum={pf_sum:.4f} abs={pf_abs:.4f} top5={pf_top} gen={gen}", flush=True)
    pg.close()


def cmp():
    import glob
    sd = torch.load("/tmp/pf_sd_0.pt")
    for f in sorted(glob.glob("/tmp/pf_g*.pt")):
        x = torch.load(f)
        rel = float((x - sd).norm() / sd.norm().clamp_min(1e-9))
        cos = float(torch.nn.functional.cosine_similarity(x, sd, dim=0))
        print(f"{os.path.basename(f):24s} rel_l2={rel:.5f} cos={cos:.6f}", flush=True)
    if os.path.exists("/tmp/moe_in_sd.pt"):
        s = torch.load("/tmp/moe_in_sd.pt")
        for f in sorted(glob.glob("/tmp/moe_in_g*.pt")):
            x = torch.load(f)
            rel = float((x - s).norm() / s.norm().clamp_min(1e-9))
            cos = float(torch.nn.functional.cosine_similarity(x.flatten(), s.flatten(), dim=0))
            print(f"MOE-IN {os.path.basename(f):20s} rel_l2={rel:.5f} cos={cos:.6f}", flush=True)


def cmp_attn():
    import glob
    s = torch.load("/tmp/attn_out_sd.pt")
    for f in sorted(glob.glob("/tmp/attn_out_g*.pt")):
        x = torch.load(f)
        rel = float((x - s).norm() / s.norm().clamp_min(1e-9))
        cos = float(torch.nn.functional.cosine_similarity(x.flatten(), s.flatten(), dim=0))
        print(f"ATTN-OUT {os.path.basename(f):22s} rel_l2={rel:.5f} cos={cos:.6f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    elif sys.argv[1] == "mp":
        run_mp()
    elif sys.argv[1] == "cmp":
        cmp()
    elif sys.argv[1] == "cmp_attn":
        cmp_attn()
    else:
        run()
