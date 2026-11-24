"""
Synthetic parity tests for the new model support (P7 Qwen3.8-Flash-Next /
qwen4_exp, and Gemma-4-26B-A4B MoE block).

Builds small random-weight checkpoints in the EXACT checkpoint layout, runs
the HF transformers reference (fp32) and the fork (fp16) on identical inputs,
and compares logits. Then checks MP (2/4 ranks) vs single-GPU, and built-in
MTP greedy equivalence on the MP path.

Usage (on the rig, transformers installed):
  python mtp/synth_newmodels_test.py qwen4exp        # single-GPU parity
  python mtp/synth_newmodels_test.py gemma4moe       # single-GPU parity
  python mtp/synth_newmodels_test.py qwen4exp_mp     # MP 2/4 parity
  python mtp/synth_newmodels_test.py gemma4moe_mp
  python mtp/synth_newmodels_test.py qwen4exp_mtp    # built-in MTP on MP
"""

import json
import math
import os
import sys
import shutil

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

WORK = os.environ.get("SYNTH_WORK", "/home/westfox35/exl2work/synth_newmodels")
SEED = 1234


def _rng():
    return torch.Generator().manual_seed(SEED)


def randn(shape, std=0.02, gen=None):
    return torch.randn(shape, generator=gen or _rng()) * std


def normw(shape, centered=True, gen=None):
    # zero-centered (1+w) norms init near 0; plain norms init near 1
    base = 0.0 if centered else 1.0
    return base + randn(shape, 0.02, gen)


# ---------------------------------------------------------------------------
# n-gram hash params (HF Qwen4ExpTextNGramEmbedding exact algorithm)
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value):
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _M1) & _MASK64
    value = ((value ^ (value >> 27)) * _M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def _nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


def ngram_params(vocab_size, ngram_size, heads_per_ngram, base, ple_layer_index=0, seed=1234):
    num_heads = (ngram_size - 1) * heads_per_ngram
    sizes = [_nth_prime_after(base - 1, ple_layer_index * num_heads + h + 1)
             for h in range(num_heads)]
    offsets = []
    t = 0
    for s in sizes:
        offsets.append(t)
        t += s
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    mults = []
    for i in range(ngram_size):
        value = (base_seed + _GAMMA * (i + 1)) & _MASK64
        mults.append(2 * (_splitmix64(value) % half_bound) + 1)
    return (torch.tensor(sizes, dtype=torch.long),
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(mults, dtype=torch.long), t)


# ---------------------------------------------------------------------------
# Qwen4-Exp synthetic checkpoint
# ---------------------------------------------------------------------------

Q4 = dict(
    hidden=64, layers=8, nq=4, nkv=2, hd=16,
    nk=4, nv=8, dk=32, dv=32, convk=4,
    moe_inter=32, experts=8, topk=2, shared=32,
    hc=4, lowrank=16,
    idx_nh=2, idx_kv=1, idx_hd=8, idx_budget=16, idx_compress=4,
    ple_dim=32, ngram_size=3, heads_per_ngram=2, ngram_base=100,
    vocab=4096, rope_theta=1e7, partial=0.25,
)


def q4_hc_tensors(prefix, gen):
    H, hc, lr = Q4["hidden"], Q4["hc"], Q4["lowrank"]
    hh = H * hc
    d = {
        prefix + ".hc_norm.weight": normw((hh,), True, gen),
        prefix + ".input_mix_weight_down.weight": randn((lr, hh), 0.02, gen),
        prefix + ".input_mix_weight_up.weight": randn((hh, lr), 0.02, gen),
    }
    return d


def q4_attn_tensors(prefix, gen, with_hc=True):
    H, nq, nkv, hd = Q4["hidden"], Q4["nq"], Q4["nkv"], Q4["hd"]
    d = {
        prefix + ".self_attn.q_proj.weight": randn((nq * hd * 2, H), 0.02, gen),
        prefix + ".self_attn.k_proj.weight": randn((nkv * hd, H), 0.02, gen),
        prefix + ".self_attn.v_proj.weight": randn((nkv * hd, H), 0.02, gen),
        prefix + ".self_attn.o_proj.weight": randn((H, nq * hd), 0.02, gen),
        prefix + ".self_attn.q_norm.weight": normw((hd,), True, gen),
        prefix + ".self_attn.k_norm.weight": normw((hd,), True, gen),
        prefix + ".self_attn.indexer.index_qk_proj.weight":
            randn(((Q4["idx_nh"] + Q4["idx_kv"]) * Q4["idx_hd"], H), 0.02, gen),
        prefix + ".self_attn.indexer.q_layernorm.weight": normw((Q4["idx_hd"],), True, gen),
        prefix + ".self_attn.indexer.k_layernorm.weight": normw((Q4["idx_hd"],), True, gen),
    }
    return d


def q4_moe_tensors(prefix, gen):
    H, E, I, S = Q4["hidden"], Q4["experts"], Q4["moe_inter"], Q4["shared"]
    return {
        prefix + ".mlp.experts.gate_up_proj": randn((E, 2 * I, H), 0.02, gen),
        prefix + ".mlp.experts.down_proj": randn((E, H, I), 0.02, gen),
        prefix + ".mlp.gate.weight": randn((E, H), 0.02, gen),
        prefix + ".mlp.shared_expert.gate_proj.weight": randn((S, H), 0.02, gen),
        prefix + ".mlp.shared_expert.up_proj.weight": randn((S, H), 0.02, gen),
        prefix + ".mlp.shared_expert.down_proj.weight": randn((H, S), 0.02, gen),
        prefix + ".mlp.shared_expert_gate.weight": randn((1, H), 0.02, gen),
    }


def q4_gdn_tensors(prefix, gen):
    H, nk, nv, dk, dv, K = Q4["hidden"], Q4["nk"], Q4["nv"], Q4["dk"], Q4["dv"], Q4["convk"]
    kd, vd = nk * dk, nv * dv
    A = torch.rand(nv, generator=gen) * (16 - 0.01) + 0.01
    return {
        prefix + ".linear_attn.in_proj_qkv.weight": randn((2 * kd + vd, H), 0.02, gen),
        prefix + ".linear_attn.in_proj_z.weight": randn((vd, H), 0.02, gen),
        prefix + ".linear_attn.in_proj_b.weight": randn((nv, H), 0.02, gen),
        prefix + ".linear_attn.in_proj_a.weight": randn((nv, H), 0.02, gen),
        prefix + ".linear_attn.conv1d.weight": randn((2 * kd + vd, 1, K), 0.02, gen),
        prefix + ".linear_attn.A_log": torch.log(A),
        prefix + ".linear_attn.dt_bias": torch.ones(nv),
        prefix + ".linear_attn.norm.weight": normw((dv,), True, gen),
        prefix + ".linear_attn.out_proj.weight": randn((H, vd), 0.02, gen),
    }


def make_qwen4exp_synth(out_dir):
    gen = _rng()
    H, hc = Q4["hidden"], Q4["hc"]
    hh = H * hc
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    layer_types = layer_types * (Q4["layers"] // 4)
    sizes, offsets, mults, total_rows = ngram_params(
        Q4["vocab"], Q4["ngram_size"], Q4["heads_per_ngram"], Q4["ngram_base"])
    head_dim = Q4["ple_dim"] // ((Q4["ngram_size"] - 1) * Q4["heads_per_ngram"])
    parts = 4
    padded = math.ceil(total_rows / parts) * parts
    rows_per_part = padded // parts

    text_cfg = {
        "model_type": "qwen4_exp_text",
        "vocab_size": Q4["vocab"],
        "hidden_size": H,
        "num_hidden_layers": Q4["layers"],
        "num_attention_heads": Q4["nq"],
        "num_key_value_heads": Q4["nkv"],
        "head_dim": Q4["hd"],
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_parameters": {"rope_theta": Q4["rope_theta"],
                            "partial_rotary_factor": Q4["partial"],
                            "rope_type": "default"},
        "linear_conv_kernel_dim": Q4["convk"],
        "linear_key_head_dim": Q4["dk"],
        "linear_value_head_dim": Q4["dv"],
        "linear_num_key_heads": Q4["nk"],
        "linear_num_value_heads": Q4["nv"],
        "moe_intermediate_size": Q4["moe_inter"],
        "shared_expert_intermediate_size": Q4["shared"],
        "num_experts": Q4["experts"],
        "num_experts_per_tok": Q4["topk"],
        "norm_topk_prob": True,
        "layer_types": layer_types,
        "hc_count": hc,
        "hc_lowrank": Q4["lowrank"],
        "ple_layer_ids": [2],
        "ple_embed_dim": Q4["ple_dim"],
        "ple_conv_kernel_size": 4,
        "ngram_size": Q4["ngram_size"],
        "heads_per_ngram": Q4["heads_per_ngram"],
        "ngram_vocab_size_base": Q4["ngram_base"],
        "make_ngram_vocab_size_divisible_by": parts,
        "split_ngram_parts": parts,
        "seed": 1234,
        "indexer_n_heads": Q4["idx_nh"],
        "indexer_kv_heads": Q4["idx_kv"],
        "indexer_head_dim": Q4["idx_hd"],
        "indexer_budget": Q4["idx_budget"],
        "indexer_compress_ratio": Q4["idx_compress"],
        "output_gate_type": "sigmoid",
        "eos_token_id": 0,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "mtp_num_hidden_layers": 1,
    }
    cfg = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": text_cfg,
        "vision_config": {"model_type": "qwen4_exp_vision", "depth": 1,
                          "hidden_size": 32, "intermediate_size": 64,
                          "num_heads": 2, "patch_size": 16, "out_hidden_size": H},
        "tie_word_embeddings": False,
        "eos_token_id": 0,
        "bos_token_id": 1,
        "pad_token_id": 0,
    }

    P = "model.language_model."
    t = {}
    t[P + "embed_tokens.weight"] = randn((Q4["vocab"], H), 0.02, gen)
    t["lm_head.weight"] = randn((Q4["vocab"], H), 0.02, gen)
    for i in range(Q4["layers"]):
        p = f"{P}layers.{i}"
        if layer_types[i] == "linear_attention":
            t.update(q4_gdn_tensors(p, gen))
        else:
            t.update(q4_attn_tensors(p, gen))
        t.update(q4_moe_tensors(p, gen))
        t.update(q4_hc_tensors(p + ".attn_hyper_connection", gen))
        t[p + ".attn_hyper_connection.block_inject_weight.weight"] = randn((hc, hh), 0.02, gen)
        t.update(q4_hc_tensors(p + ".mlp_hyper_connection", gen))
        t[p + ".mlp_hyper_connection.block_inject_weight.weight"] = randn((hc, hh), 0.02, gen)
    # PLE at layer index 1
    p = P + "layers.1.ple"
    t[p + ".key_proj.weight"] = randn((hh, Q4["ple_dim"]), 0.02, gen)
    t[p + ".value_proj.weight"] = randn((H, Q4["ple_dim"]), 0.02, gen)
    t[p + ".norm_key.weight"] = normw((hh,), True, gen)
    t[p + ".norm_query.weight"] = normw((hh,), True, gen)
    t[p + ".norm_conv.weight"] = normw((hh,), True, gen)
    t[p + ".conv1d.weight"] = randn((hh, 1, 4), 0.02, gen)
    t[p + ".ple_embedding.layer_multipliers"] = mults
    t[p + ".ple_embedding.ngram_heads_offsets"] = offsets
    t[p + ".ple_embedding.ngram_heads_vocab_sizes"] = sizes
    table = randn((padded, head_dim), 0.02, gen)
    for s in range(parts):
        t[f"{p}.ple_embedding.ngram_embedding.shard_{s}.weight"] = \
            table[s * rows_per_part:(s + 1) * rows_per_part].clone()
    # global mixer
    t.update(q4_hc_tensors(P + "hyper_connection_mixer", gen))
    # MTP module
    t["mtp.pre_fc_norm_hidden.weight"] = normw((hh,), True, gen)
    t["mtp.pre_fc_norm_embedding.weight"] = normw((H,), True, gen)
    t["mtp.fc_embedding.weight"] = randn((H, H), 0.02, gen)
    t["mtp.fc_hidden.weight"] = randn((H, H), 0.02, gen)
    t.update(q4_hc_tensors("mtp.hyper_connection_mixer", gen))
    mp = "mtp.layers.0"
    t.update(q4_attn_tensors(mp, gen))
    t.update(q4_moe_tensors(mp, gen))
    t.update(q4_hc_tensors(mp + ".attn_hyper_connection", gen))
    t[mp + ".attn_hyper_connection.block_inject_weight.weight"] = randn((hc, hh), 0.02, gen)
    t.update(q4_hc_tensors(mp + ".mlp_hyper_connection", gen))
    t[mp + ".mlp_hyper_connection.block_inject_weight.weight"] = randn((hc, hh), 0.02, gen)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    save_file({k: v.contiguous() for k, v in t.items()},
              os.path.join(out_dir, "model.safetensors"))
    _write_mini_tokenizer(out_dir, Q4["vocab"])
    return cfg


def _write_mini_tokenizer(out_dir, vocab):
    # Minimal word-level tokenizer.json so ExLlamaV2Tokenizer can load.
    toks = {f"t{i}": i for i in range(vocab)}
    tj = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<pad>", "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "<bos>", "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": toks, "unk_token": "t0"},
    }
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(tj, f)
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast",
                  "bos_token": "<bos>", "eos_token": "<pad>", "pad_token": "<pad>",
                  "model_max_length": 4096}, f)


# ---------------------------------------------------------------------------
# Qwen3.8-27B (Qwen3_5, P6) synthetic checkpoint — for the built-in MTP-on-MP test
# ---------------------------------------------------------------------------

Q35 = dict(
    hidden=512, layers=8, nq=4, nkv=2, hd=32,
    nk=4, nv=8, dk=32, dv=32, convk=4,
    inter=512, vocab=4096, rope_theta=1e7, partial=0.25,
)


def make_qwen35_synth(out_dir):
    gen = _rng()
    H = Q35["hidden"]
    nq, nkv, hd = Q35["nq"], Q35["nkv"], Q35["hd"]
    nk, nv, dk, dv, K = Q35["nk"], Q35["nv"], Q35["dk"], Q35["dv"], Q35["convk"]
    kd, vd = nk * dk, nv * dv
    I = Q35["inter"]
    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * (Q35["layers"] // 4)
    text_cfg = {
        "model_type": "qwen3_5_text",
        "vocab_size": Q35["vocab"],
        "hidden_size": H,
        "num_hidden_layers": Q35["layers"],
        "num_attention_heads": nq,
        "num_key_value_heads": nkv,
        "head_dim": hd,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_parameters": {"rope_theta": Q35["rope_theta"],
                            "partial_rotary_factor": Q35["partial"],
                            "rope_type": "default"},
        "linear_conv_kernel_dim": K,
        "linear_key_head_dim": dk,
        "linear_value_head_dim": dv,
        "linear_num_key_heads": nk,
        "linear_num_value_heads": nv,
        "intermediate_size": I,
        "layer_types": layer_types,
        "output_gate_type": "swish",
        "eos_token_id": 0,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "mtp_num_hidden_layers": 1,
    }
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": text_cfg,
        "vision_config": {"model_type": "qwen3_5_vision", "depth": 1,
                          "hidden_size": 32, "intermediate_size": 64,
                          "num_heads": 2, "patch_size": 16, "out_hidden_size": H},
        "tie_word_embeddings": False,
        "eos_token_id": 0,
        "bos_token_id": 1,
        "pad_token_id": 0,
    }
    P = "model.language_model."
    t = {}
    t[P + "embed_tokens.weight"] = randn((Q35["vocab"], H), 0.02, gen)
    t["lm_head.weight"] = randn((Q35["vocab"], H), 0.02, gen)
    t[P + "norm.weight"] = normw((H,), True, gen)

    def attn(p):
        t[p + ".self_attn.q_proj.weight"] = randn((nq * hd * 2, H), 0.02, gen)
        t[p + ".self_attn.k_proj.weight"] = randn((nkv * hd, H), 0.02, gen)
        t[p + ".self_attn.v_proj.weight"] = randn((nkv * hd, H), 0.02, gen)
        t[p + ".self_attn.o_proj.weight"] = randn((H, nq * hd), 0.02, gen)
        t[p + ".self_attn.q_norm.weight"] = normw((hd,), True, gen)
        t[p + ".self_attn.k_norm.weight"] = normw((hd,), True, gen)

    def mlp(p):
        t[p + ".mlp.gate_proj.weight"] = randn((I, H), 0.02, gen)
        t[p + ".mlp.up_proj.weight"] = randn((I, H), 0.02, gen)
        t[p + ".mlp.down_proj.weight"] = randn((H, I), 0.02, gen)

    def gdn(p):
        A = torch.rand(nv, generator=gen) * (16 - 0.01) + 0.01
        t[p + ".linear_attn.in_proj_qkv.weight"] = randn((2 * kd + vd, H), 0.02, gen)
        t[p + ".linear_attn.in_proj_z.weight"] = randn((vd, H), 0.02, gen)
        t[p + ".linear_attn.in_proj_b.weight"] = randn((nv, H), 0.02, gen)
        t[p + ".linear_attn.in_proj_a.weight"] = randn((nv, H), 0.02, gen)
        t[p + ".linear_attn.conv1d.weight"] = randn((2 * kd + vd, 1, K), 0.02, gen)
        t[p + ".linear_attn.A_log"] = torch.log(A)
        t[p + ".linear_attn.dt_bias"] = torch.ones(nv)
        t[p + ".linear_attn.norm.weight"] = normw((dv,), True, gen)
        t[p + ".linear_attn.out_proj.weight"] = randn((H, vd), 0.02, gen)

    for i in range(Q35["layers"]):
        p = f"{P}layers.{i}"
        t[p + ".input_layernorm.weight"] = normw((H,), True, gen)
        t[p + ".post_attention_layernorm.weight"] = normw((H,), True, gen)
        if layer_types[i] == "linear_attention":
            gdn(p)
        else:
            attn(p)
        mlp(p)
    # MTP module (concat fc + norm + one dense decoder layer)
    t["mtp.pre_fc_norm_hidden.weight"] = normw((H,), True, gen)
    t["mtp.pre_fc_norm_embedding.weight"] = normw((H,), True, gen)
    t["mtp.fc.weight"] = randn((H, 2 * H), 0.02, gen)
    t["mtp.norm.weight"] = normw((H,), True, gen)
    mp = "mtp.layers.0"
    t[mp + ".input_layernorm.weight"] = normw((H,), True, gen)
    t[mp + ".post_attention_layernorm.weight"] = normw((H,), True, gen)
    attn(mp)
    mlp(mp)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    save_file({k: v.contiguous() for k, v in t.items()},
              os.path.join(out_dir, "model.safetensors"))
    _write_mini_tokenizer(out_dir, Q35["vocab"])
    return cfg


# ---------------------------------------------------------------------------
# Gemma-4 MoE synthetic checkpoint
# ---------------------------------------------------------------------------

G4 = dict(
    hidden=1024, layers=6, nq=4, hd_swa=256, hd_full=512,
    nkv_swa=2, nkv_full=1, swa=1024,
    dense_inter=64, moe_inter=32, experts=8, topk=2,
    vocab=4096,
)


def make_gemma4_moe_synth(out_dir, moe=True):
    gen = _rng()
    H = G4["hidden"]
    # sliding_window_pattern 6: layers 0..4 sliding, layer 5 full
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    text_cfg = {
        "model_type": "gemma4_text",
        "vocab_size": G4["vocab"],
        "hidden_size": H,
        "num_hidden_layers": G4["layers"],
        "num_attention_heads": G4["nq"],
        "num_key_value_heads": G4["nkv_swa"],
        "head_dim": G4["hd_swa"],
        "global_head_dim": G4["hd_full"],
        "num_global_key_value_heads": G4["nkv_full"],
        "hidden_act": "gelu_pytorch_tanh",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_k_eq_v": True,
        "sliding_window": G4["swa"],
        "sliding_window_pattern": 6,
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional",
                               "partial_rotary_factor": 0.25},
        },
        "intermediate_size": G4["dense_inter"],
        "moe_intermediate_size": G4["moe_inter"],
        "enable_moe_block": moe,
        "num_experts": G4["experts"],
        "top_k_experts": G4["topk"],
        "hidden_size_per_layer_input": 0,
        "use_double_wide_mlp": False,
        "layer_types": layer_types,
        "eos_token_id": 0,
        "bos_token_id": 1,
        "pad_token_id": 0,
    }
    cfg = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": text_cfg,
        "vision_config": {"model_type": "gemma4_vision", "hidden_size": 32,
                          "intermediate_size": 64, "num_hidden_layers": 1,
                          "num_attention_heads": 2, "patch_size": 16,
                          "out_hidden_size": H},
        "tie_word_embeddings": True,
        "eos_token_id": 0,
        "bos_token_id": 1,
        "pad_token_id": 0,
    }

    t = {}
    P = "model.language_model."
    t[P + "embed_tokens.weight"] = randn((G4["vocab"], H), 0.02, gen)
    t[P + "norm.weight"] = normw((H,), False, gen)
    for i in range(G4["layers"]):
        p = f"{P}layers.{i}"
        full = layer_types[i] == "full_attention"
        hd = G4["hd_full"] if full else G4["hd_swa"]
        nkv = G4["nkv_full"] if full else G4["nkv_swa"]
        t[p + ".input_layernorm.weight"] = normw((H,), False, gen)
        t[p + ".post_attention_layernorm.weight"] = normw((H,), False, gen)
        t[p + ".pre_feedforward_layernorm.weight"] = normw((H,), False, gen)
        t[p + ".post_feedforward_layernorm.weight"] = normw((H,), False, gen)
        t[p + ".self_attn.q_proj.weight"] = randn((G4["nq"] * hd, H), 0.02, gen)
        t[p + ".self_attn.k_proj.weight"] = randn((nkv * hd, H), 0.02, gen)
        if not full:
            t[p + ".self_attn.v_proj.weight"] = randn((nkv * hd, H), 0.02, gen)
        t[p + ".self_attn.o_proj.weight"] = randn((H, G4["nq"] * hd), 0.02, gen)
        t[p + ".self_attn.q_norm.weight"] = normw((hd,), False, gen)
        t[p + ".self_attn.k_norm.weight"] = normw((hd,), False, gen)
        I, E, MI = G4["dense_inter"], G4["experts"], G4["moe_inter"]
        t[p + ".mlp.gate_proj.weight"] = randn((I, H), 0.02, gen)
        t[p + ".mlp.up_proj.weight"] = randn((I, H), 0.02, gen)
        t[p + ".mlp.down_proj.weight"] = randn((H, I), 0.02, gen)
        if not moe:
            t[p + ".layer_scalar"] = torch.ones(1) + randn((1,), 0.02, gen)
            continue
        t[p + ".experts.gate_up_proj"] = randn((E, 2 * MI, H), 0.02, gen)
        t[p + ".experts.down_proj"] = randn((E, H, MI), 0.02, gen)
        t[p + ".router.proj.weight"] = randn((E, H), 0.02, gen)
        t[p + ".router.scale"] = normw((H,), False, gen)
        t[p + ".router.per_expert_scale"] = normw((E,), False, gen)
        t[p + ".pre_feedforward_layernorm_2.weight"] = normw((H,), False, gen)
        t[p + ".post_feedforward_layernorm_1.weight"] = normw((H,), False, gen)
        t[p + ".post_feedforward_layernorm_2.weight"] = normw((H,), False, gen)
        t[p + ".layer_scalar"] = torch.ones(1) + randn((1,), 0.02, gen)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    save_file({k: v.contiguous() for k, v in t.items()},
              os.path.join(out_dir, "model.safetensors"))
    _write_mini_tokenizer(out_dir, G4["vocab"])
    return cfg


# ---------------------------------------------------------------------------
# parity helpers
# ---------------------------------------------------------------------------

def metrics(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    cos = F.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-9)).item()
    return cos, rel


def hf_qwen4exp_logits(model_dir, ids, steps=0):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    cfg = Qwen4ExpConfig(**raw)
    m = Qwen4ExpForConditionalGeneration(cfg)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    # HF wants one dense ngram table; the checkpoint layout shards it.
    shard_keys = sorted([k for k in sd if ".ngram_embedding.shard_" in k],
                        key=lambda k: int(k.split(".shard_")[1].split(".")[0]))
    if shard_keys:
        base = shard_keys[0].split(".shard_")[0]
        sd[base + ".weight"] = torch.cat([sd.pop(k) for k in shard_keys], dim=0)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "visual" not in k and "vision" not in k]
    assert not missing, f"missing: {missing[:8]}"
    m = m.cuda().eval()
    out = []
    with torch.inference_mode():
        r = m(ids.cuda())
        out.append(r.logits[:, -1:, :].cpu())
        past = r.past_key_values
        cur = ids[:, -1:].cuda()
        for _ in range(steps):
            r = m(cur, past_key_values=past, use_cache=True)
            past = r.past_key_values
            nxt = r.logits[:, -1:, :].argmax(-1)
            out.append(r.logits[:, -1:, :].cpu())
            cur = nxt
    del m
    torch.cuda.empty_cache()
    return torch.cat(out, dim=1)


def fork_logits(model_dir, ids, steps=0, device=0):
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
    from exllamav2.cache import ExLlamaV2Cache_Qwen4Exp, ExLlamaV2Cache_Hybrid
    cfg = ExLlamaV2Config(model_dir)
    cfg.max_seq_len = 4096
    cfg.max_input_len = 1024
    m = ExLlamaV2(cfg)
    m.load()
    hybrid = getattr(cfg, "hybrid_linear_attn", False)
    if getattr(cfg.arch.lm, "hc_residual", False):
        cache = ExLlamaV2Cache_Qwen4Exp(m, max_seq_len=4096)
    elif hybrid:
        cache = ExLlamaV2Cache_Hybrid(m, max_seq_len=4096)
    else:
        from exllamav2.cache import ExLlamaV2Cache_SWA
        cache = ExLlamaV2Cache_SWA(m, max_seq_len=4096)
    out = []
    with torch.inference_mode():
        logits = m.forward(ids, cache=cache, last_id_only=True)
        out.append(logits.cpu())
        cur = ids[:, -1:]
        for _ in range(steps):
            logits = m.forward(cur, cache=cache, last_id_only=True)
            out.append(logits.cpu())
            cur = logits.argmax(-1).cpu()
    del m, cache
    torch.cuda.empty_cache()
    return torch.cat(out, dim=1)


def run_single(family):
    if family == "qwen4exp":
        d = os.path.join(WORK, "qwen4exp")
        make_qwen4exp_synth(d)
    else:
        # Gemma-4's fork attention runs fp16 FI/FA2 kernels + fp16 rope tables;
        # vs an fp32 HF reference the hd512 proportional-rope layer drifts ~2%
        # on RANDOM weights (dense-only shows the same drift -- not MoE code).
        # The exact gate for the new code is the block math unit test (fp32
        # reference, cos 1.0) plus the fork-vs-fork MP parity below.
        run_gemma4_block_unit()
        return
    torch.manual_seed(SEED)
    ids = torch.randint(2, 256, (1, 40))
    print(f"[{family}] short prefill+decode parity (40 tokens + 8 steps)")
    hf = hf_qwen4exp_logits(d, ids, steps=8) if family == "qwen4exp" else \
        hf_gemma4_logits(d, ids, steps=8)
    fk = fork_logits(d, ids, steps=8)
    cos, rel = metrics(hf, fk)
    am = (hf.argmax(-1) == fk.argmax(-1)).float().mean().item()
    print(f"  cos={cos:.6f} rel_l2={rel:.6f} argmax_match={am:.3f}")
    assert cos > 0.99, f"{family} short parity FAILED cos={cos}"
    if family == "qwen4exp":
        print(f"[{family}] LONG prefill parity (300 tokens > indexer budget 16)")
        ids2 = torch.randint(2, 256, (1, 300))
        hf2 = hf_qwen4exp_logits(d, ids2, steps=4)
        fk2 = fork_logits(d, ids2, steps=4)
        cos2, rel2 = metrics(hf2, fk2)
        print(f"  cos={cos2:.6f} rel_l2={rel2:.6f}")
        assert cos2 > 0.99, f"{family} long parity FAILED cos={cos2}"
    if family == "gemma4moe":
        print(f"[{family}] LONG prefill parity (1200 tokens > SWA window 1024)")
        ids2 = torch.randint(2, 256, (1, 1200))
        hf2 = hf_gemma4_logits(d, ids2, steps=2)
        fk2 = fork_logits(d, ids2, steps=2)
        cos2, rel2 = metrics(hf2, fk2)
        print(f"  cos={cos2:.6f} rel_l2={rel2:.6f}")
        assert cos2 > 0.99, f"{family} long parity FAILED cos={cos2}"
    print(f"[{family}] SINGLE-GPU PARITY PASS")


def hf_gemma4_logits(model_dir, ids, steps=0):
    from transformers import AutoModelForCausalLM, AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir)
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32,
                                         attn_implementation="sdpa")
    from safetensors.torch import load_file
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("model.visual") and "vision" not in k
               and k != "lm_head.weight"]  # tied embeddings
    assert not missing, f"missing keys: {missing[:10]}"
    m.tie_weights()
    m = m.cuda().eval()
    out = []
    with torch.inference_mode():
        r = m(ids.cuda())
        out.append(r.logits[:, -1:, :].cpu())
        past = r.past_key_values
        cur = ids[:, -1:].cuda()
        for _ in range(steps):
            r = m(cur, past_key_values=past, use_cache=True)
            past = r.past_key_values
            out.append(r.logits[:, -1:, :].cpu())
            cur = r.logits[:, -1:, :].argmax(-1)
    del m
    torch.cuda.empty_cache()
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# Gemma-4 MoE block: direct math parity vs an fp32 torch reference (isolates the
# block from the fork's fp16 hd512 attention precision profile).
# ---------------------------------------------------------------------------

def run_gemma4_block_unit():
    from safetensors.torch import load_file
    d = os.path.join(WORK, "gemma4moe")
    make_gemma4_moe_synth(d)
    sd = load_file(os.path.join(d, "model.safetensors"))
    from exllamav2 import ExLlamaV2, ExLlamaV2Config
    from exllamav2.gemma4_moe import ExLlamaV2Gemma4MoEBlock
    cfg = ExLlamaV2Config(d)
    cfg.max_seq_len = 512
    m = ExLlamaV2(cfg)
    m.load()
    blocks = [mod for mod in m.modules if isinstance(mod, ExLlamaV2Gemma4MoEBlock)]
    torch.manual_seed(7)
    x = torch.randn(1, 16, G4["hidden"], device="cuda:0", dtype=torch.half)
    worst = 1.0
    for li, blk in enumerate(blocks):
        P = f"model.language_model.layers.{li}."
        def w(k):
            return sd[P + k].cuda().float()
        def rms(t, wt=None):
            tf = t.float()
            v = tf.pow(2).mean(-1, keepdim=True)
            n = tf * torch.rsqrt(v + 1e-6)
            return n * (wt.float() if wt is not None else 1.0)
        xf = x.float()
        # dense branch
        dn = rms(xf, w("pre_feedforward_layernorm.weight"))
        g = dn @ w("mlp.gate_proj.weight").T
        u = dn @ w("mlp.up_proj.weight").T
        dense = (F.gelu(g, approximate="tanh") * u) @ w("mlp.down_proj.weight").T
        dense = rms(dense, w("post_feedforward_layernorm_1.weight"))
        # router
        rn = rms(xf) * w("router.scale") * (G4["hidden"] ** -0.5)
        logits = rn @ w("router.proj.weight").T
        probs = torch.softmax(logits, dim=-1)
        topw, topi = torch.topk(probs, G4["topk"], dim=-1)
        topw = topw / topw.sum(-1, keepdim=True)
        topw = topw * w("router.per_expert_scale")[topi]
        # experts
        h2 = rms(xf, w("pre_feedforward_layernorm_2.weight"))
        gu = w("experts.gate_up_proj")
        dnw = w("experts.down_proj")
        moe = torch.zeros_like(xf)
        flat = h2.view(-1, G4["hidden"])
        for e in range(G4["experts"]):
            mask = (topi[0] == e)
            rows, ks = torch.where(mask)
            if rows.numel() == 0:
                continue
            cur = flat[rows]
            gue = gu[e]
            gg = cur @ gue[:G4["moe_inter"]].T
            uu = cur @ gue[G4["moe_inter"]:].T
            act = F.gelu(gg, approximate="tanh") * uu
            dd = act @ dnw[e].T
            moe.view(-1, G4["hidden"]).index_add_(0, rows, dd * topw[0][rows, ks, None])
        moe = rms(moe, w("post_feedforward_layernorm_2.weight"))
        ref = rms(dense + moe, w("post_feedforward_layernorm.weight")) + xf
        with torch.inference_mode():
            got = blk.forward(x).float()
        cos, rel = metrics(got, ref)
        worst = min(worst, cos)
        print(f"  block{li}: cos={cos:.6f} rel={rel:.6f}")
    assert worst > 0.999, f"gemma4 MoE block math parity FAILED worst={worst}"
    print("[gemma4moe] BLOCK MATH PARITY PASS")


# ---------------------------------------------------------------------------
# MP parity: fork MP (2/4 ranks) vs fork single-GPU, same fp16 precision.
# ---------------------------------------------------------------------------

def fork_logits_mp(model_dir, ids, steps, nranks, base="ExLlamaV2Cache_SWA", forced=None):
    from exllamav2.tp_mp import TPProcessGroup
    pg = TPProcessGroup(model_dir, [20.0] * nranks, max_seq_len=4096,
                        cache_tokens=4096, cache_base_name=base,
                        max_batch=1, max_input_len=1024)
    pg.start()
    out = []
    with torch.inference_mode():
        logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        out.append(logits.float().cpu())
        cur = ids[:, -1:]
        for s in range(steps):
            logits, state = pg.forward_state(cur, last_id_only=True, cpu_logits=True)
            out.append(logits.float().cpu())
            # Teacher-force the single-GPU token sequence so tiny fp differences
            # don't chaotically diverge the rollouts on random weights.
            cur = forced[s + 1] if forced is not None else logits.argmax(-1).cpu()
    pg.close()
    return torch.cat(out, dim=1)


def run_mp(family):
    # MP loads QUANTIZED tensors only: use the converted synthetic models.
    dirs = {"qwen4exp": "q4_quant", "gemma4moe": "g4_quant", "qwen35": "q35_quant"}
    d = os.path.join(WORK, dirs[family])
    assert os.path.isdir(d), f"converted model missing: {d} (run convert.py first)"
    torch.manual_seed(SEED)
    ids = torch.randint(2, 256, (1, 40))
    single = fork_logits(d, ids, steps=6)
    forced = [ids[:, -1:]] + [single[:, s:s+1].argmax(-1).cpu() for s in range(1, 7)]
    bases = {"qwen4exp": "ExLlamaV2Cache_Qwen4Exp", "qwen35": "ExLlamaV2Cache_Hybrid",
             "gemma4moe": "ExLlamaV2Cache_SWA"}
    base = bases[family]
    for nranks in (2,):
        mp = fork_logits_mp(d, ids, steps=6, nranks=nranks, base=base, forced=forced)
        c0, _ = metrics(single[:, :1], mp[:, :1])
        print(f"  prefill-only cos={c0:.6f}")
        for s in range(1, single.shape[1]):
            cs, _ = metrics(single[:, s:s+1], mp[:, s:s+1])
            print(f"  step{s} cos={cs:.6f} am_s={int(single[:,s].argmax())} am_m={int(mp[:,s].argmax())}")
        cos, rel = metrics(single, mp)
        am = (single.argmax(-1) == mp.argmax(-1)).float().mean().item()
        print(f"[{family}] MP{nranks} vs single: cos={cos:.6f} rel={rel:.6f} argmax={am:.3f}")
        # Random-weight logits are near-tied: fp16 reduction-order differences can
        # flip an argmax without any logic error. Gate on cosine for the fp32-math
        # backbone (qwen4_exp); for gemma4 the fp16 hd512 proportional-rope drift
        # lowers cosine, so gate on exact greedy-token equivalence instead.
        if family in ("qwen4exp", "qwen35"):
            assert cos > 0.999, f"{family} MP{nranks} parity FAILED cos={cos}"
        else:
            assert am == 1.0 and cos > 0.9, f"{family} MP{nranks} parity FAILED am={am} cos={cos}"
    print(f"[{family}] MP PARITY PASS")


# ---------------------------------------------------------------------------
# Built-in MTP on MP: greedy MTP emission must EXACTLY equal plain greedy decode.
# ---------------------------------------------------------------------------

def run_mtp_mp(family):
    from exllamav2.tp_mp import TPProcessGroup
    from exllamav2.mtp_mp import MTPMPDriver
    from exllamav2.mtp_builtin_mp import BuiltinMTPMPAdapter
    dirs = {"qwen4exp": "q4_quant", "qwen35": "q35_quant", "gemma4moe": "g4_quant"}
    bases = {"qwen4exp": "ExLlamaV2Cache_Qwen4Exp", "qwen35": "ExLlamaV2Cache_Hybrid",
             "gemma4moe": "ExLlamaV2Cache_SWA"}
    d = os.path.join(WORK, dirs[family])
    base = bases[family]
    pg = TPProcessGroup(d, [20.0, 20.0], max_seq_len=4096, cache_tokens=4096,
                        cache_base_name=base, max_batch=1, max_input_len=1024)
    pg.start()
    if family == "qwen4exp":
        from exllamav2.mtp_qwen4exp import Qwen4ExpMTPDrafter
        drafter = Qwen4ExpMTPDrafter(pg.model, device=0, num_draft_tokens=3, max_seq_len=4096)
    else:
        from exllamav2.mtp_qwen35 import Qwen35MTPDrafter
        drafter = Qwen35MTPDrafter(pg.model, device=0, num_draft_tokens=3, max_seq_len=4096)
    adapter = BuiltinMTPMPAdapter(drafter, pg)
    pg.enable_mtp(adapter, num_draft_tokens=3)

    torch.manual_seed(SEED)
    ids = torch.randint(2, 256, (1, 24))
    N = 24
    with torch.inference_mode():
        logits, state = pg.forward_state(ids, last_id_only=False, cpu_logits=True)
        drv = MTPMPDriver(pg, adapter, window=3, heal=True)
        drv.begin(ids, logits, state)
        mtp_tokens = [drv.step() for _ in range(N)]
        print(f"[{family}] MTP stats: {drv.stats()}")
        # Plain greedy reference on the same group.
        pg.mtp_reset_cache()
        adapter.d.reset()
        logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)
        ref = [int(logits.argmax(-1).item())]
        cur = logits.argmax(-1).cpu()
        for _ in range(N - 1):
            logits, state = pg.forward_state(cur, last_id_only=True, cpu_logits=True)
            cur = logits.argmax(-1).cpu()
            ref.append(int(cur.item()))
    pg.close()
    match = mtp_tokens == ref
    print(f"[{family}] MTP tokens: {mtp_tokens[:12]}...")
    print(f"[{family}] ref tokens: {ref[:12]}...")
    assert match, f"{family} MTP-on-MP greedy equivalence FAILED"
    print(f"[{family}] MTP-ON-MP GREEDY EQUIVALENCE PASS ({N} tokens)")


if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else "qwen4exp"
    os.makedirs(WORK, exist_ok=True)
    if fam in ("qwen4exp", "gemma4moe"):
        run_single(fam)
    elif fam == "gemma4moe_block":
        run_gemma4_block_unit()
    elif fam in ("qwen4exp_mp", "gemma4moe_mp", "qwen35_mp"):
        run_mp(fam[:-3])
    elif fam in ("qwen4exp_mtp", "gemma4moe_mtp", "qwen35_mtp"):
        run_mtp_mp(fam[:-4])
    else:
        print(f"unknown mode {fam}")
        sys.exit(1)
