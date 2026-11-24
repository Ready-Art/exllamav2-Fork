from exllamav2.conversion.qparams import QParams
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
import math
import itertools
import time
from exllamav2.conversion.bot_status import print_stage

def optimize(job, save_fn, model):

    cfg = model.config
    km = cfg.arch.lm.keys

    has_gate = cfg.arch.lm.mlp_gate
    if has_gate: mlp_key_gate = km["mlp_gate"]
    mlp_key_up = km["mlp_up"]
    mlp_key_down = km["mlp_down"]

    norm_interval = (1.5, 3.5)
    norm_2ndstage = 0.15
    anneal_temp_max = 2
    anneal_temp_min = 0.0001
    anneal_cooling_factor = 0.995
    anneal_iter = 1000
    anneal_samples = 80
    anneal_stages = 3

    # `layers_root` is relative to lm_prefix (Gemma-4 uses "layers", not "model.layers").
    # PLAN.md §2.1.
    layers_root = getattr(cfg.arch.lm, "layers_root", "model.layers")
    first_q_layer = 0
    while not model.modules[first_q_layer].key.startswith(cfg.arch.lm_prefix + layers_root):
        first_q_layer += 1

    # max_step_size = 2
    # first_layer_bias = 4
    # bias_layers = 2
    # bias_iter = 0

    # Hybrid backbone (Qwen3.5): layer 0 is a linear-attention (gated delta net) layer
    # with no self_attn keys. Reference shapes come from the first FULL-attention layer;
    # per-layer numel below is read from the actual modules, so the fallbacks here are
    # display-only. newplan.md P6.
    _hybrid = getattr(cfg, "hybrid_linear_attn", False)
    _ref_layer = 0
    if _hybrid:
        for _li in range(cfg.num_hidden_layers):
            if cfg.layer_types[_li] == "full_attention":
                _ref_layer = _li
                break
    key = cfg.arch.lm_prefix + layers_root + "." + str(_ref_layer)
    key_q = key + km["attn_q"]
    key_k = key + km["attn_k"]
    key_v = key + km["attn_v"]
    key_o = key + km["attn_o"]

    # attention_k_eq_v: the full-attention layers have no v_proj at all (Gemma-4).
    # Layer 0 is a sliding layer, so v_proj exists there, but guard anyway so a
    # checkpoint whose first layer is a full layer does not crash here.
    has_v_proj = key_v in model.modules_dict

    if not cfg.arch.lm.is_moe:
        if has_gate: key_g = key + mlp_key_gate
        key_u = key + mlp_key_up
        key_d = key + mlp_key_down
        mlp_mode = "mlp"
    else:
        if has_gate: key_g = key + mlp_key_gate.replace("*", "0")
        key_u = key + mlp_key_up.replace("*", "0")
        key_d = key + mlp_key_down.replace("*", "0")
        mlp_mode = "block_sparse_moe"

    num_experts = cfg.num_experts if cfg.num_experts is not None else 1
    shape_q = model.modules_dict[key_q].matrix_shape()
    shape_k = model.modules_dict[key_k].matrix_shape()
    shape_v = model.modules_dict[key_v].matrix_shape() if has_v_proj else None
    shape_o = model.modules_dict[key_o].matrix_shape()
    shape_g = model.modules_dict[key_g].matrix_shape() if has_gate else None
    shape_u = model.modules_dict[key_u].matrix_shape()
    shape_d = model.modules_dict[key_d].matrix_shape()
    numel_q = shape_q[0] * shape_q[1]
    numel_k = shape_k[0] * shape_k[1]
    numel_v = shape_v[0] * shape_v[1] if has_v_proj else 0
    numel_o = shape_o[0] * shape_o[1]
    numel_g = shape_g[0] * shape_g[1] * num_experts if has_gate else 0
    numel_u = shape_u[0] * shape_u[1] * num_experts
    numel_d = shape_d[0] * shape_d[1] * num_experts
    numel_attn = numel_q + numel_k + numel_v + numel_o
    numel_mlp = numel_g + numel_u + numel_d

    def _attn_mode(layer_i):
        # Measurement / strategy key suffix for a layer's token mixer.
        if _hybrid and cfg.layer_types[layer_i] == "linear_attention":
            return "gdn"
        return "self_attn"

    # Combined size of hidden layers

    num_layers = cfg.num_hidden_layers

    # Modules per layer. Normally attn + mlp = 2, but Gemma-4 inserts an
    # ExLlamaV2LayerScalar after each MLP, making it 3. Counting 2 would slice
    # the module list short and silently drop the last layers' weights from the
    # bitrate budget, producing a quant that misses its target bpw.
    # PLAN.md §2.6.
    modules_per_layer = 3 if getattr(cfg.arch.lm, "layer_scalar", False) else 2
    num_modules = num_layers * modules_per_layer
    if getattr(cfg.arch.lm, "hc_residual", False):
        # Qwen4-Exp layers carry 6+ modules (hyper-connection mix/inject pairs,
        # PLE), so the fixed-stride slice does not describe the quantized set.
        # Sum the actual quantizable projections instead; the fp16 pass-through
        # modules (hc/PLE/indexer/router/shared) are outside the bitrate budget.
        from exllamav2.gdn import ExLlamaV2GatedDeltaNet
        from exllamav2.moe_mlp import ExLlamaV2MoEMLP
        numel = 0
        for m in model.modules:
            if isinstance(m, ExLlamaV2Attention):
                numel += m.q_proj.numel() + m.k_proj.numel() + m.o_proj.numel()
                if m.v_proj is not None: numel += m.v_proj.numel()
            elif isinstance(m, ExLlamaV2GatedDeltaNet):
                numel += sum(p.numel() for p in (m.in_proj_q, m.in_proj_k, m.in_proj_v,
                                                 m.in_proj_z, m.out_proj))
            elif isinstance(m, ExLlamaV2MoEMLP):
                numel += sum(l.numel() for l in m.w1 + m.w2 + m.w3)
    else:
        numel = sum(m.numel() for m in model.modules[first_q_layer : num_modules + first_q_layer])

    target_bpw = job["bits"]
    weight_budget = int(numel * target_bpw)

    # Compile options

    measurement = job["measurement"]
    slots = []
    params = []

    for i in range(num_layers):
        if cfg.arch.lm.parallel_decoder_blocks:
            m1 = measurement[cfg.arch.lm_prefix + layers_root + "." + str(i) + ".parallel_decoder"]["attn"]
            m2 = measurement[cfg.arch.lm_prefix + layers_root + "." + str(i) + ".parallel_decoder"]["mlp"]
        else:
            m1 = measurement[cfg.arch.lm_prefix + layers_root + "." + str(i) + "." + _attn_mode(i)]
            m2 = measurement[cfg.arch.lm_prefix + layers_root + "." + str(i) + "." + mlp_mode]
        for m in [m1, m2]:
            slot = []
            param = []
            for opt in m:
                o = (int(opt["total_bits"]), 1 - opt["accuracy"])
                slot.append(o)
                param.append(opt)
            slots.append(slot)
            params.append(param)

    # Find some solutions

    last_update = 0
    m = float("inf")
    p = float("inf")
    c = weight_budget
    for i in range(anneal_stages * anneal_samples):
        if time.time() - last_update > 1 or i == anneal_samples - 1:
            print(f" -- Optimizing: {i + 1:4}/{anneal_stages * anneal_samples:4}")
            print_stage(job, "Optimizing", i + 1, anneal_stages * anneal_samples)
            last_update = time.time()

        if i < anneal_samples:
            t = i / (anneal_samples - 1)
            norm = (1 - t) * norm_interval[0] + t * norm_interval[1]

        elif i < anneal_samples * 2:
            if i == anneal_samples:
                norm_a = bestnorm - norm_2ndstage / 2
                norm_b = bestnorm + norm_2ndstage / 2
            t = i / (anneal_samples - 1) - 1
            norm = (1 - t) * norm_a + t * norm_b

        else:
            norm = bestnorm

        s_, si_, p_, c_, m_ = ext_c.sim_anneal(
            slots,
            weight_budget,
            anneal_temp_max,
            anneal_cooling_factor,
            anneal_temp_min,
            anneal_iter,
            norm
        )

        if i < anneal_samples * 2:
            if m_ < m:
                m = m_
                bestnorm = norm
        else:
            if p_ < p:
                s, si, p, c, m = s_, si_, p_, c_, m_

    # Tweak solution in case there is some budget left over

    while True:
        repeat = False
        for i in range(len(si)):
            if si[i] < len(slots[i]) - 1:
                delta_c = slots[i][si[i] + 1][0] - slots[i][si[i]][0]
                if c + delta_c <= weight_budget:
                    c += delta_c
                    si[i] = si[i] + 1
                    repeat = True
        if not repeat: break

    solution_idx = si
    print(f" -- max(err): {m:.6f}")
    print(f" -- error_norm: {bestnorm:.6f}")

    # Save strategy

    print(" -- Quantization strategy:")

    logerr = 0
    maxerr = 0
    job["strategy"] = {}
    for layer_ in range(num_layers):

        k1 = cfg.arch.lm_prefix + layers_root + "." + str(layer_) + "." + _attn_mode(layer_)
        k2 = cfg.arch.lm_prefix + layers_root + "." + str(layer_) + "." + mlp_mode
        p1 = params[layer_ * 2][solution_idx[layer_ * 2]]
        p2 = params[layer_ * 2 + 1][solution_idx[layer_ * 2 + 1]]

        # Per-layer numel. Gemma-4's full-attention layers have a different geometry
        # (head_dim 512, 4 kv heads) from the sliding layers (256, 16), so a single
        # numel_attn taken from layer 0 would print the wrong bpw for 10 of 60 layers.
        # PLAN.md §2.2.
        if cfg.arch.lm.parallel_decoder_blocks:
            n1 = sum(m.numel() for m in model.modules_dict[k1].submodules)
            n2 = sum(m.numel() for m in model.modules_dict[k2].submodules)
        else:
            attn_mod = model.modules_dict.get(k1)
            mlp_mod = model.modules_dict.get(k2)
            n1 = attn_mod.numel() if attn_mod is not None else numel_attn
            n2 = mlp_mod.numel() if mlp_mod is not None else numel_mlp

        for (k, p, n) in zip((k1, k2), (p1, p2), (n1, n2)):
            job["strategy"][k] = p
            bpw = p["total_bits"] / n
            err = 1 - p["accuracy"]
            print(f" --   {k:50} {bpw:1.4f} bpw - exp. error: {err:1.8f}")
            logerr += math.log(err)
            maxerr = max(err, maxerr)

    print(f" -- sum(log(err)): {logerr:.6f}")
    print(f" -- max(err): {maxerr:.6f}")

    xx = 0