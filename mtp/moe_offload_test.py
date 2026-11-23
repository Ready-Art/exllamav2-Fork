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
    # V must cover the borrowed Gemma tokenizer's id range (262k), else calibration
    # embedding lookups device-assert.
    H, I, L, E, V, NH, NKV = 256, 512, 2, 8, 262144, 8, 2
    cfg = {
        "architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
        "hidden_size": H, "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "num_local_experts": E, "num_experts_per_tok": 2, "vocab_size": V,
        "max_position_embeddings": 4096, "rms_norm_eps": 1e-5, "rope_theta": 1e6,
        "torch_dtype": "float16", "tie_word_embeddings": False,
        "sliding_window": None, "attention_dropout": 0.0, "initializer_range": 0.02,
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
        w[p + "self_attn.q_proj.weight"] = rn(NH * 32, H)
        w[p + "self_attn.k_proj.weight"] = rn(NKV * 32, H)
        w[p + "self_attn.v_proj.weight"] = rn(NKV * 32, H)
        w[p + "self_attn.o_proj.weight"] = rn(H, NH * 32)
        w[p + "block_sparse_moe.gate.weight"] = rn(E, H)
        for e in range(E):
            ep = p + f"block_sparse_moe.experts.{e}."
            w[ep + "w1.weight"] = rn(I, H)
            w[ep + "w2.weight"] = rn(H, I)
            w[ep + "w3.weight"] = rn(I, H)
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
    c = ExLlamaV2Config(); c.model_dir = SYNTH + "_exl2"; c.prepare()
    c.max_seq_len = 1024
    tok = ExLlamaV2Tokenizer(c)
    m = ExLlamaV2(c)
    m.load()
    cache = ExLlamaV2Cache(m, max_seq_len = 1024)
    ids = tok.encode("The quick brown fox jumps over the lazy dog.", encode_special_tokens = True)
    logits = m.forward(ids, cache = cache)
    gen = []
    lg = logits[:, -1:, :]
    for _ in range(24):
        t = lg.argmax(dim = -1).cpu()   # fork keeps token embeddings on CPU (embed_cpu)
        gen.append(int(t[0, 0]))
        lg = m.forward(t, cache = cache, last_id_only = True)
    vram = torch.cuda.memory_allocated() / 2**20
    print(f"OFFLOAD={N} vram={vram:.1f}MiB logits_sum={float(logits.float().sum()):.4f} "
          f"gen={gen}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    else:
        run()
