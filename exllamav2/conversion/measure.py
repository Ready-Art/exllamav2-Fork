from exllamav2.model import \
(
    ExLlamaV2Embedding,
    ExLlamaV2PosEmbedding,
    ExLlamaV2Attention,
    ExLlamaV2GatedDeltaNet,
    ExLlamaV2MLP,
    ExLlamaV2MoEMLP,
    ExLlamaV2Linear,
    ExLlamaV2RMSNorm,
    ExLlamaV2LayerNorm,
    ExLlamaV2ParallelDecoder
)

from exllamav2.layer_scalar import ExLlamaV2LayerScalar
from exllamav2.qwen4_exp import ExLlamaV2HyperConnection, ExLlamaV2PLELayer
from exllamav2.gemma4_moe import ExLlamaV2Gemma4MoEBlock
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav2.conversion.qparams import QParams, qparams_headoptions, qparams_attn, qparams_mlp, qparams_gdn, get_qparams_reduced
from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
import torch
from torch import nn
import os, time, math, json
import torch.nn.functional as F
import gc
from exllamav2.conversion.bot_status import print_stage
from exllamav2.conversion.devices import resolve_devices

# graceful exiting
import signal
import sys

interrupted = False

def signal_handler(signal, frame):
    global interrupted
    if interrupted:
        print("\nGracefully exiting...")
        sys.exit(0)
    else:
        interrupted = True
        print("\nCTRL-C again to quit or type 'exit'. You can always resume the process at a later time.")
        user_input = input("\nPress Enter to continue processing or type 'exit' to quit: ").strip().lower()
        if user_input == 'exit':
            print("Gracefully exiting...")
            sys.exit(0)
        interrupted = False

signal.signal(signal.SIGINT, signal_handler)

def list_live_tensors():

    tensors = {}
    gc.collect()
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                d = str(obj.size()) + ", " + str(obj.dtype) + ", " + str(obj.device)
                if d in tensors.keys():
                    tensors[d] += 1
                else:
                    tensors[d] = 1
        except:
            pass

    print("-----------")
    for k, v in tensors.items():
        print(f"{v} : {k}")


# Get initial token embeddings

def embeddings(job, save_fn, model, measure = False):

    print_stage(job, "Embeddings", 0, 1)

    module = model.modules[0]
    assert isinstance(module, ExLlamaV2Embedding)

    with safe_open(job["cal_filename"], framework = "pt", device = "cpu") as f:
        input_ids = f.get_tensor("input_ids")

    module.load()
    input_ids[input_ids >= module.native_vocab_size] = 0
    hidden_state = module.forward(input_ids, negative_ids_noise = True)
    module.unload()

    embeddings_dict = { f"row.{i:05}": hidden_state[i:i+1, :, :].contiguous() for i in range(hidden_state.shape[0]) }
    save_file(embeddings_dict, os.path.join(job["out_dir"], "hidden_states.safetensors"))

    print_stage(job, "Embeddings", 1, 1)


# Test quantization options

def test_quant(source: ExLlamaV2Linear,
               lq: AdaptiveGPTQ,
               qparams: list):

    variants = []
    variants_bits = []

    original = nn.Linear(source.in_features, source.out_features, source.has_bias, device = "meta", dtype = torch.float16)
    original.weight = nn.Parameter(source.linear.weight.clone())
    if source.has_bias: original.bias.weight = nn.Parameter(source.linear.bias.clone())

    for qp in qparams:

        lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
        lq.quantize()
        quantized = lq.apply_temp()
        quantized.to("cpu")

        variants.append(quantized)
        total_bits = qp.total_bits(quantized.weight.T.shape, original.bias.weight.shape if source.has_bias else None)
        variants_bits.append(total_bits)

        numel = quantized.weight.numel()
        if source.has_bias: numel += original.bias.numel()
        bpw = total_bits / numel
        desc = qp.desc

        print(f" -- {source.key:50} {desc:50} {bpw:2.2f} bpw")

    return variants, variants_bits


def test_error(module, hidden_states, target_states, cache, attn_params):

    rfn_sum = torch.tensor(0.0).cuda()
    rfn_count = 0
    for x, xref in zip(hidden_states, target_states):
        x = x.cuda()
        xref = xref.cuda()
        xtest = module.forward(x, cache, attn_params)
        if not module.model.config.arch.lm.residual_stream_fp32:
            xtest.clamp_(-65504, 65504)
        xtest = xtest[0].float()
        xref = xref[0].float()
        rfn_sum += torch.linalg.norm(xtest - xref, 'fro') / torch.linalg.norm(xref, 'fro')
        rfn_count += 1

    return max(1e-6, 1 - (rfn_sum.item() / rfn_count))


def measure_attn(module, hidden_states, target_states, quantizers, cache, attn_params, keep_q = False):

    qjobs, qmaps = get_qparams_reduced(qparams_attn)
    results = []

    # attention_k_eq_v: the full-attention layers have no v_proj at all (Gemma-4),
    # because V is K. Skip the v_proj measurement and let `v` track `k`.
    # PLAN.md §2.5.
    has_v = module.v_proj is not None

    quantizers["q_proj"].prepare()
    quantizers["k_proj"].reuse_h(quantizers["q_proj"])
    if has_v:
        quantizers["v_proj"].reuse_h(quantizers["q_proj"])
    quantizers["o_proj"].prepare()

    options_q, bits_q = test_quant(module.q_proj, quantizers["q_proj"], qjobs[0])
    options_k, bits_k = test_quant(module.k_proj, quantizers["k_proj"], qjobs[1])
    if has_v:
        options_v, bits_v = test_quant(module.v_proj, quantizers["v_proj"], qjobs[2])
    else:
        # V is K, so it shares K's options and bit cost exactly.
        #
        # NOTE: `bits_v` must be indexed with the *v* index from qmaps, not the k index. The two
        # lists are built independently by get_qparams_reduced (one per module position), so they
        # are not guaranteed to be the same length. Aliasing them here made `bits_v[v]` raise
        # IndexError on the first k-eq-v layer whenever the v list was shorter than the k list.
        # The v index is valid for the k list because both are derived from the same qparams
        # ordering, so index the k list with v.
        options_v, bits_v = options_k, bits_k
    options_o, bits_o = test_quant(module.o_proj, quantizers["o_proj"], qjobs[3])

    total_numel = module.q_proj.numel()
    total_numel += module.k_proj.numel()
    if has_v:
        total_numel += module.v_proj.numel()
    total_numel += module.o_proj.numel()

    max_accuracy = 0.0
    (q_, k_, v_, o_) = (-1, -1, -1, -1)
    for (q, k, v, o) in qmaps:

        if q != q_: module.q_proj.linear.weight = nn.Parameter(options_q[q].weight.cuda())
        if k != k_: module.k_proj.linear.weight = nn.Parameter(options_k[k].weight.cuda())
        if has_v and v != v_: module.v_proj.linear.weight = nn.Parameter(options_v[v].weight.cuda())
        if o != o_: module.o_proj.linear.weight = nn.Parameter(options_o[o].weight.cuda())
        (q_, k_, v_, o_) = (q, k, v, o)

        total_bits = bits_q[q]
        total_bits += bits_k[k]
        # For k-eq-v layers V *is* K, so its bit cost is exactly K's. Use the k index here:
        # the v index is only meaningful for the v option list, which does not exist on these
        # layers and is not guaranteed to be the same length as the k list.
        total_bits += bits_v[v] if has_v else bits_k[k]
        total_bits += bits_o[o]
        total_bpw = total_bits / total_numel

        accuracy = test_error(module, hidden_states, target_states, cache, attn_params)
        print(f" -- {total_bpw:1.4f} bpw  accuracy: {accuracy:1.8f}")

        max_accuracy = max(accuracy, max_accuracy)

        torch.cuda.empty_cache()

        r = { "accuracy": accuracy,
              "total_bits": total_bits,
              "q_proj": qjobs[0][q].get_dict(),
              "k_proj": qjobs[1][k].get_dict(),
              "o_proj": qjobs[3][o].get_dict() }
        # No v_proj entry for attention_k_eq_v layers: V is K, so it inherits K's
        # strategy. PLAN.md §2.5.
        if has_v:
            r["v_proj"] = qjobs[2][v].get_dict()
        results.append(r)

    if max_accuracy < 0.1:
        print(" ## Measurement/inference error (1)")
        os._exit(1)

    for x in ["k_proj", "v_proj", "o_proj"] + (["q_proj"] if not keep_q else []):
        if x in quantizers:
            del quantizers[x]

    return results


def measure_gdn(module, hidden_states, target_states, quantizers, cache, attn_params):
    """
    Measure the gated-delta-net projections (Qwen3.5, newplan.md P6). q/k/v/z share
    the post-norm Hessian; out_proj is measured against the gated-norm output.
    """

    qjobs, qmaps = get_qparams_reduced(qparams_gdn)
    results = []

    quantizers["in_proj_q"].prepare()
    quantizers["in_proj_k"].reuse_h(quantizers["in_proj_q"])
    quantizers["in_proj_v"].reuse_h(quantizers["in_proj_q"])
    quantizers["in_proj_z"].reuse_h(quantizers["in_proj_q"])
    quantizers["out_proj"].prepare()

    options_q, bits_q = test_quant(module.in_proj_q, quantizers["in_proj_q"], qjobs[0])
    options_k, bits_k = test_quant(module.in_proj_k, quantizers["in_proj_k"], qjobs[1])
    options_v, bits_v = test_quant(module.in_proj_v, quantizers["in_proj_v"], qjobs[2])
    options_z, bits_z = test_quant(module.in_proj_z, quantizers["in_proj_z"], qjobs[3])
    options_o, bits_o = test_quant(module.out_proj, quantizers["out_proj"], qjobs[4])

    total_numel = module.in_proj_q.numel() + module.in_proj_k.numel() + \
        module.in_proj_v.numel() + module.in_proj_z.numel() + module.out_proj.numel()

    max_accuracy = 0.0
    (q_, k_, v_, z_, o_) = (-1, -1, -1, -1, -1)
    for (q, k, v, z, o) in qmaps:

        if q != q_: module.in_proj_q.linear.weight = nn.Parameter(options_q[q].weight.cuda())
        if k != k_: module.in_proj_k.linear.weight = nn.Parameter(options_k[k].weight.cuda())
        if v != v_: module.in_proj_v.linear.weight = nn.Parameter(options_v[v].weight.cuda())
        if z != z_: module.in_proj_z.linear.weight = nn.Parameter(options_z[z].weight.cuda())
        if o != o_: module.out_proj.linear.weight = nn.Parameter(options_o[o].weight.cuda())
        (q_, k_, v_, z_, o_) = (q, k, v, z, o)

        total_bits = bits_q[q] + bits_k[k] + bits_v[v] + bits_z[z] + bits_o[o]
        total_bpw = total_bits / total_numel

        accuracy = test_error(module, hidden_states, target_states, cache, attn_params)
        print(f" -- {total_bpw:1.4f} bpw  accuracy: {accuracy:1.8f}")

        max_accuracy = max(accuracy, max_accuracy)

        torch.cuda.empty_cache()

        r = { "accuracy": accuracy,
              "total_bits": total_bits,
              "in_proj_q": qjobs[0][q].get_dict(),
              "in_proj_k": qjobs[1][k].get_dict(),
              "in_proj_v": qjobs[2][v].get_dict(),
              "in_proj_z": qjobs[3][z].get_dict(),
              "out_proj": qjobs[4][o].get_dict() }
        results.append(r)

    if max_accuracy < 0.1:
        print(" ## Measurement/inference error (1)")
        os._exit(1)

    for x in ["in_proj_k", "in_proj_v", "in_proj_z", "out_proj", "in_proj_q"]:
        if x in quantizers:
            del quantizers[x]

    return results


def measure_mlp(module, hidden_states, target_states, quantizers, cache, attn_params, reuse_h_up_proj = None):

    has_gate = module.model.config.arch.lm.mlp_gate

    qjobs, qmaps = get_qparams_reduced(qparams_mlp, not has_gate)
    results = []

    if reuse_h_up_proj is not None:
        quantizers["up_proj"].reuse_h(quantizers[reuse_h_up_proj])
    else:
        quantizers["up_proj"].prepare()
    if has_gate: quantizers["gate_proj"].reuse_h(quantizers["up_proj"])
    quantizers["down_proj"].prepare()

    options_g, bits_g = test_quant(module.gate_proj, quantizers[f"gate_proj"], qjobs[0]) if has_gate else (None, None)
    options_u, bits_u = test_quant(module.up_proj, quantizers[f"up_proj"], qjobs[1])
    options_d, bits_d = test_quant(module.down_proj, quantizers[f"down_proj"], qjobs[2])

    total_numel = module.gate_proj.numel() if has_gate else 0
    total_numel += module.up_proj.numel()
    total_numel += module.down_proj.numel()

    max_accuracy = 0.0
    if has_gate:

        (g_, u_, d_) = (-1, -1, -1)
        for (g, u, d) in qmaps:

            if g != g_: module.gate_proj.linear.weight = nn.Parameter(options_g[g].weight.cuda())
            if u != u_: module.up_proj.linear.weight = nn.Parameter(options_u[u].weight.cuda())
            if d != d_: module.down_proj.linear.weight = nn.Parameter(options_d[d].weight.cuda())
            (g_, u_, d_) = (g, u, d)

            total_bits = bits_g[g]
            total_bits += bits_u[u]
            total_bits += bits_d[d]
            total_bpw = total_bits / total_numel

            accuracy = test_error(module, hidden_states, target_states, cache, attn_params)
            print(f" -- {total_bpw:1.4f} bpw  accuracy: {accuracy:1.8f}")

            max_accuracy = max(accuracy, max_accuracy)

            torch.cuda.empty_cache()

            r = { "accuracy": accuracy,
                  "total_bits": total_bits,
                  "gate_proj": qjobs[0][g].get_dict(),
                  "up_proj": qjobs[1][u].get_dict(),
                  "down_proj": qjobs[2][d].get_dict() }
            results.append(r)

    else:

        (u_, d_) = (-1, -1)
        for (u , d) in qmaps:

            if u != u_: module.up_proj.linear.weight = nn.Parameter(options_u[u].weight.cuda())
            if d != d_: module.down_proj.linear.weight = nn.Parameter(options_d[d].weight.cuda())
            (u_, d_) = (u, d)

            total_bits = bits_u[u]
            total_bits += bits_d[d]
            total_bpw = total_bits / total_numel

            accuracy = test_error(module, hidden_states, target_states, cache, attn_params)
            print(f" -- {total_bpw:1.4f} bpw  accuracy: {accuracy:1.8f}")

            max_accuracy = max(accuracy, max_accuracy)

            torch.cuda.empty_cache()

            r = { "accuracy": accuracy,
                  "total_bits": total_bits,
                  "up_proj": qjobs[1][u].get_dict(),
                  "down_proj": qjobs[2][d].get_dict() }
            results.append(r)

    if max_accuracy < 0.1:
        print(" ## Measurement/inference error (1)")
        os._exit(1)

    for x in ["up_proj", "down_proj", "gate_proj"]:
        if x in quantizers:
            del quantizers[x]

    return results


def measure_moe_mlp(module, hidden_states, target_states, quantizers, cache, attn_mask):

    qjobs, qmaps = get_qparams_reduced(qparams_mlp)
    num_experts = module.model.config.num_experts
    results = []

    quantizers["w1.0"].prepare()
    for i in range(num_experts):
        if i > 0: quantizers[f"w1.{i}"].reuse_h(quantizers["w1.0"])
        quantizers[f"w3.{i}"].reuse_h(quantizers["w1.0"])
        quantizers[f"w2.{i}"].prepare()

    options_g, bits_g = [], []
    options_u, bits_u = [], []
    options_d, bits_d = [], []
    for i in range(num_experts):
        options_g_, bits_g_ = test_quant(module.w1[i], quantizers[f"w1.{i}"], qjobs[0])
        del quantizers[f"w1.{i}"]
        options_u_, bits_u_ = test_quant(module.w3[i], quantizers[f"w3.{i}"], qjobs[1])
        del quantizers[f"w3.{i}"]
        options_d_, bits_d_ = test_quant(module.w2[i], quantizers[f"w2.{i}"], qjobs[2])
        del quantizers[f"w2.{i}"]
        options_g.append(options_g_)
        options_u.append(options_u_)
        options_d.append(options_d_)
        bits_g.append(bits_g_)
        bits_u.append(bits_u_)
        bits_d.append(bits_d_)

    quantizers.clear()
    gc.collect()
    torch.cuda.empty_cache()

    total_numel = sum(module.w1[i].numel() for i in range(num_experts))
    total_numel += sum(module.w3[i].numel() for i in range(num_experts))
    total_numel += sum(module.w2[i].numel() for i in range(num_experts))

    max_accuracy = 0.0
    (g_, u_, d_) = (-1, -1, -1)
    for (g, u, d) in qmaps:

        for i in range(num_experts):
            if g != g_: module.w1[i].linear.weight = nn.Parameter(options_g[i][g].weight.cuda())
            if u != u_: module.w3[i].linear.weight = nn.Parameter(options_u[i][u].weight.cuda())
            if d != d_: module.w2[i].linear.weight = nn.Parameter(options_d[i][d].weight.cuda())
        (g_, u_, d_) = (g, u, d)

        total_bits = sum(bits_g[i][g] for i in range(num_experts))
        total_bits += sum(bits_u[i][u] for i in range(num_experts))
        total_bits += sum(bits_d[i][d] for i in range(num_experts))
        total_bpw = total_bits / total_numel

        accuracy = test_error(module, hidden_states, target_states, cache, attn_mask)
        print(f" -- {total_bpw:1.4f} bpw  accuracy: {accuracy:1.8f}")

        max_accuracy = max(accuracy, max_accuracy)

        torch.cuda.empty_cache()

        r = { "accuracy": accuracy,
              "total_bits": total_bits,
              "w1": qjobs[0][g].get_dict(),
              "w3": qjobs[1][u].get_dict(),
              "w2": qjobs[2][d].get_dict() }
        results.append(r)

    if max_accuracy < 0.1:
        print(" ## Measurement/inference error (1)")
        os._exit(1)

    return results


def measure_parallel_decoder(module, hidden_states, target_states_attn, target_states_mlp, quantizers, cache, attn_params,
                             primary_device = "cuda:0"):

    for i in range(len(hidden_states)):
        hidden_states[i] = hidden_states[i].cpu()

    print(f" -- Sublayer: {module.key}.self_attn")
    results_attn = measure_attn(module.attn, hidden_states, target_states_attn, quantizers, cache, attn_params, keep_q = True)

    module.attn.unload()
    gc.collect()
    torch.cuda.empty_cache()

    print(f" -- Sublayer: {module.key}.mlp")
    results_mlp = measure_mlp(module.mlp, hidden_states, target_states_mlp, quantizers, cache, attn_params, "q_proj")

    for i in range(len(hidden_states)):
        hidden_states[i] = hidden_states[i].to(primary_device)

    r = { "attn": results_attn,
          "mlp": results_mlp }
    return r


# helpful status box for insights around conversions
def get_remaining_time_str(estimated_time_remaining):
    remaining_minutes = int(estimated_time_remaining // 60)
    remaining_seconds = int(estimated_time_remaining % 60)
    return f"{remaining_minutes}min {remaining_seconds}sec"

def format_line(label, box_width):
    return f"| {label.ljust(box_width - 3)}|"

def print_status_box(*content_lines):
    max_content_width = max(len(line) for line in content_lines)
    box_width = max_content_width + 4 

    print('-' * box_width)
    for line in content_lines:
        print(format_line(line, box_width))
    print('-' * box_width)

def _clone_quantizers(quantizers):
    """Fresh AdaptiveGPTQ accumulators for a replica shard, one per key."""
    from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
    out = {}
    for k, q in quantizers.items():
        nq = AdaptiveGPTQ(q.layer)
        nq.hessian_device = q.hessian_device
        out[k] = nq
    return out


@torch.inference_mode()
def measure_quant(job, save_fn, model, hidden_state_offload_layers):

    # vars for status box
    time_spent_list = []  
    rolling_window_size = 10 # (increase to average over larger window)
    completed_steps = 0  
    accuracy_sum = 0  
    accuracy_count = 0  
    overall_rolling_accuracy = 0  

    last_snapshot_time = time.time()
    snapshot_interval_s = 180

    temp_filename = os.path.join(job["out_dir"], "hidden_states_temp.safetensors")
    states_filename = os.path.join(job["out_dir"], "hidden_states.safetensors")
    measurement = job.get("measurement", {})

    # Quantize

    last_ckpt_layer_name = "None"

    if not "last_module_idx" in job:
        job["last_module_idx"] = 0
    else:
        i = job["last_module_idx"]
        if i < len(model.modules):
            last_ckpt_layer_name = f"{model.modules[i].key} ({model.modules[i].name})"
            print(f" -- Resuming from layer: {last_ckpt_layer_name}")

    # vars to support status box
    total_modules = len(model.modules)  
    last_module_idx = job["last_module_idx"]  # resume tracking steps where it stopped previously
    remaining_steps = total_modules - last_module_idx  

    # Devices this pass may use. The first is the primary and owns the module load; the rest are
    # only used when calibration row sharding is enabled. quant-plan.md §6.5 (M3).
    devices = resolve_devices(job.get("devices"))
    if not devices: devices = [0]
    primary_device = torch.device(devices[0])
    primary_device_str = f"cuda:{devices[0]}"

    hidden_states = []
    with safe_open(states_filename, framework = "pt", device = "cpu") as f:
        for i, k in enumerate(sorted(f.keys())):
            t = f.get_tensor(k)
            hidden_states.append(t.to(primary_device) if i < hidden_state_offload_layers else t)

    index = job["last_module_idx"]
    while True:

        print_stage(job, "Measuring", index, len(model.modules))

        # sig handler should catch it faster in most cases
        if interrupted:
            print("Measurement process was interrupted. Please decide:")
            if interrupted:
                print("Exiting after saving the current state.")
                job["measurement"] = measurement.copy()
                job["last_module_idx"] = index
                save_fn()
                return "interrupted"
            else:
                print("Resuming the process.")

        index += 1
        if index >= len(model.modules): break

        # Timer

        begin_time = time.time()

        # Prepare module

        module = model.modules[index]
        module.load()

        print(f" -- Layer: {module.key} ({module.name})")

        # Create quantizers

        quantizers = {}

        if isinstance(module, ExLlamaV2Attention):
            mode = "self_attn"
            quantizers["q_proj"] = AdaptiveGPTQ(module.q_proj.linear)
            quantizers["k_proj"] = AdaptiveGPTQ(module.k_proj.linear)
            # v_proj is None for attention_k_eq_v layers (Gemma-4 full attention).
            # PLAN.md §2.5.
            if module.v_proj is not None:
                quantizers["v_proj"] = AdaptiveGPTQ(module.v_proj.linear)
            quantizers["o_proj"] = AdaptiveGPTQ(module.o_proj.linear)

        elif isinstance(module, ExLlamaV2GatedDeltaNet):
            # Qwen3.5 gated delta net (newplan.md P6). in_proj_b/a stay fp16 (gate logits).
            mode = "gdn"
            quantizers["in_proj_q"] = AdaptiveGPTQ(module.in_proj_q.linear)
            quantizers["in_proj_k"] = AdaptiveGPTQ(module.in_proj_k.linear)
            quantizers["in_proj_v"] = AdaptiveGPTQ(module.in_proj_v.linear)
            quantizers["in_proj_z"] = AdaptiveGPTQ(module.in_proj_z.linear)
            quantizers["out_proj"] = AdaptiveGPTQ(module.out_proj.linear)

        elif isinstance(module, ExLlamaV2MLP):
            mode = "mlp"
            has_gate = module.model.config.arch.lm.mlp_gate
            if has_gate: quantizers["gate_proj"] = AdaptiveGPTQ(module.gate_proj.linear)
            quantizers["up_proj"] = AdaptiveGPTQ(module.up_proj.linear)
            quantizers["down_proj"] = AdaptiveGPTQ(module.down_proj.linear)

        elif isinstance(module, ExLlamaV2Gemma4MoEBlock):
            # Gemma-4-26B-A4B: quantize the routed experts; the dense branch,
            # router and norms stay fp16 (compile copies them through).
            mode = "block_sparse_moe"
            for i in range(model.config.num_experts):
                quantizers[f"w1.{i}"] = AdaptiveGPTQ(module.w1[i].linear)
                quantizers[f"w3.{i}"] = AdaptiveGPTQ(module.w3[i].linear)
                quantizers[f"w2.{i}"] = AdaptiveGPTQ(module.w2[i].linear)

        elif isinstance(module, ExLlamaV2MoEMLP):
            mode = "block_sparse_moe"
            for i in range(model.config.num_experts):
                quantizers[f"w1.{i}"] = AdaptiveGPTQ(module.w1[i].linear)
                quantizers[f"w3.{i}"] = AdaptiveGPTQ(module.w3[i].linear)
                quantizers[f"w2.{i}"] = AdaptiveGPTQ(module.w2[i].linear)

        elif isinstance(module, ExLlamaV2ParallelDecoder):
            mode = "parallel_decoder"
            quantizers["q_proj"] = AdaptiveGPTQ(module.attn.q_proj.linear)
            quantizers["k_proj"] = AdaptiveGPTQ(module.attn.k_proj.linear)
            quantizers["v_proj"] = AdaptiveGPTQ(module.attn.v_proj.linear)
            quantizers["o_proj"] = AdaptiveGPTQ(module.attn.o_proj.linear)
            has_gate = module.model.config.arch.lm.mlp_gate
            if has_gate: quantizers["gate_proj"] = AdaptiveGPTQ(module.mlp.gate_proj.linear)
            quantizers["up_proj"] = AdaptiveGPTQ(module.mlp.up_proj.linear)
            quantizers["down_proj"] = AdaptiveGPTQ(module.mlp.down_proj.linear)

        elif isinstance(module, ExLlamaV2Linear):
            mode = "linear"
            # Don't measure head layer

        elif isinstance(module, ExLlamaV2RMSNorm) or isinstance(module, ExLlamaV2LayerNorm):
            mode = "norm"

        elif isinstance(module, ExLlamaV2LayerScalar):
            # 1-element pass-through tensor: no calibration, no Hessian.
            # PLAN.md §2.6.
            mode = "layer_scalar"

        elif isinstance(module, ExLlamaV2PosEmbedding):
            mode = "pos_emb"

        elif isinstance(module, ExLlamaV2HyperConnection):
            # fp16 pass-through: run the forward to propagate the (width-changing)
            # residual, no quantizers. compile.py copies the raw weights.
            mode = "hc"

        elif isinstance(module, ExLlamaV2PLELayer):
            # fp16 pass-through. The PLE contribution depends on the calibration
            # token ids, which the module loop does not carry; calibration runs
            # without it (identity), compile.py copies the raw weights and the
            # n-gram table is converted by convert_ngram.py.
            mode = "ple"

        # Reference forward pass

        # layer_scalar has no forward pass to run: it is a 1-element multiplier that
        # compile.py copies through. Skip it entirely rather than trying to measure it.
        if mode == "layer_scalar":
            module.unload()
            torch.cuda.empty_cache()
            continue

        cache = None
        attn_params = ExLlamaV2Attention.Params(1, hidden_states[0].shape[1], 0, None, None) \
            if mode in ["self_attn", "parallel_decoder"] else None

        target_states = []
        target_states_attn = []
        target_states_mlp = []

        uncalibrated_experts = [0 for _ in range(model.config.num_experts)] if mode == "block_sparse_moe" else None

        # The calibration row loop, factored out so it can be sharded across devices.
        # ``active`` is the module to forward through: the loaded module for the primary
        # shard, a replica for a secondary one. Returns (quantizers, target_states,
        # target_states_attn, target_states_mlp) for exactly the given rows.
        def run_rows(device_idx, active, rows):

            dev = torch.device(device_idx)
            dev_str = f"cuda:{device_idx}"

            # ``active is None`` means "the primary shard, which is the module already loaded".
            # shard_calibration_rows() passes None for the primary because it does not hold a
            # reference to the module; without this the primary shard would get *cloned*
            # quantizers and its Hessian contributions would never reach the real ones.
            if active is None:
                active = module

            qz = quantizers if active is module else _clone_quantizers(quantizers)
            ap = attn_params
            ts, ts_attn, ts_mlp = [], [], []
            ue_count = [0 for _ in range(model.config.num_experts)] if mode == "block_sparse_moe" else None

            for i in rows:

                x = hidden_states[i].to(dev)
                if torch.isnan(x).any():
                    print(" ## Measurement/inference error (2)")
                    os._exit(1)
                if torch.isinf(x).any():
                    print(" ## Measurement/inference error (3)")
                    os._exit(1)

                if mode == "ple":
                    outputs = { "hidden_states": x }
                else:
                    outputs = active.forward(x, cache, attn_params, intermediates = True)
                target_device = dev_str if i < hidden_state_offload_layers else "cpu"

                for k, v in outputs.items():
                    num_inf = torch.isinf(v).sum()
                    num_nan = torch.isnan(v).sum()
                    p_inf = num_inf / v.numel()
                    p_nan = num_nan / v.numel()

                    if num_nan:
                        print(f" ## Measurement/inference error (2): {k}")
                        print(f" ## NaN elements in output states row {i}: {num_nan} / {v.numel()} = {p_nan*100:.2f}%")
                        os._exit(1)
                    if num_inf:
                        if p_inf < 0.005:
                            print(f" !! Measurement/inference warning (3): {k}")
                            print(f" !! inf elements in output states row {i}: {num_inf} / {v.numel()} = {p_inf*100:.2f}%")
                            print(f" !! clamping state")
                            v.clamp_(-65504, 65504)
                        else:
                            print(f" ## Measurement/inference error (3): {k}")
                            print(f" ## inf elements in output states row {i}: {num_inf} / {v.numel()} = {p_inf*100:.2f}%")
                            print(f" ## Number of inf elements above threshold, aborting")
                            os._exit(1)

                # Hessians

                if mode == "self_attn":
                    qz["q_proj"].add_batch(outputs["post_norm"])  # Reuse H for K and V
                    qz["o_proj"].add_batch(outputs["attn_output"])
                    ts.append(outputs["hidden_states"].to(target_device))

                if mode == "gdn":
                    qz["in_proj_q"].add_batch(outputs["post_norm"])  # Reuse H for k/v/z
                    qz["out_proj"].add_batch(outputs["pre_out"])
                    ts.append(outputs["hidden_states"].to(target_device))

                if mode == "mlp":
                    qz["up_proj"].add_batch(outputs["post_norm"])  # Reuse H for gate_proj
                    qz["down_proj"].add_batch(outputs["pre_down"])
                    ts.append(outputs["hidden_states"].to(target_device))

                if mode == "block_sparse_moe":
                    for j in range(model.config.num_experts):
                        if f"pre_down.{j}" in outputs:
                            if j == 0: qz[f"w1.{j}"].add_batch(outputs["post_norm"])
                            qz[f"w2.{j}"].add_batch(outputs[f"pre_down.{j}"])
                            if outputs[f"pre_down.{j}"].shape[0] < outputs["post_norm"].shape[0] / 10:
                                ue_count[j] += 1
                        else:
                            ue_count[j] += 1
                    ts.append(outputs["hidden_states"].to(target_device))

                if mode == "parallel_decoder":
                    qz["q_proj"].add_batch(outputs["post_norm"])  # Reuse H for K, V, up_proj and gate_proj
                    qz["o_proj"].add_batch(outputs["attn_output"])
                    qz["down_proj"].add_batch(outputs["pre_down"])
                    hidden_states[i] = outputs["post_norm"]
                    ts_attn.append(outputs["hidden_states_attn"].to(target_device))
                    ts_mlp.append(outputs["hidden_states_mlp"].to(target_device))
                    ts.append(outputs["hidden_states"].to(target_device))

                if mode == "pos_emb":
                    ts.append(outputs["hidden_states"].to(target_device))

                if mode in ("hc", "ple"):
                    ts.append(outputs["hidden_states"].to(target_device))

            # NOTE: this return must be at the function body level, NOT inside the row loop.
            # It was indented into the loop by the reindent that fixed the dedent bug, which
            # made run_rows() return after the first row. shard_calibration_rows() then saw a
            # 1-element target_states list and raised
            #   IndexError: list index out of range
            # at ordered[r] = ts_primary[i].
            return qz, ts, ts_attn, ts_mlp, ue_count

        # Row sharding (M3). Off by default: it needs one module replica per device, and a
        # device that cannot hold one is dropped rather than allowed to OOM.
        # quant-plan.md §6.5.
        # NOTE: parallel_decoder is deliberately excluded. shard_calibration_rows() returns
        # only (quantizers, target_states, ue_count) — it does not produce the
        # hidden_states_attn / hidden_states_mlp split states that
        # measure_parallel_decoder() requires, so sharding that mode would feed it None.
        # Gemma-4 does not use parallel_decoder_blocks, but the guard is here so a future
        # architecture that does cannot silently get a None target.
        if job.get("calib_row_shards") and len(devices) > 1 and mode in ("self_attn", "mlp"):
            from exllamav2.conversion.calib import (
                shard_calibration_rows, ensure_device_contexts, module_replica_bytes, build_module_replica
            )
            ensure_device_contexts(model, devices)
            need = module_replica_bytes(module)
            replicas = []
            for d in devices[1:]:
                # Refuse to start a replica that cannot fit rather than letting it OOM mid-pass:
                # an OOM here loses the whole calibration pass, not just one row.
                free_b, _ = torch.cuda.mem_get_info(d)
                if need and free_b < need * 1.10:
                    print(f" !! No calibration replica on cuda:{d}: needs {need / 2**30:.1f} GiB, "
                          f"{free_b / 2**30:.1f} GiB free")
                    continue
                try:
                    # build_module_replica passes sliding_window / rope_index / MLP geometry
                    # through. type(module)(model, key, layer_idx) does NOT: it silently gives a
                    # sliding layer a full-attention replica reading the global rope table.
                    r = build_module_replica(module, d)
                    replicas.append(r)
                except Exception as e:
                    print(f" !! No calibration replica on cuda:{d}: {str(e).splitlines()[0][:120]}")
            # The sharded path returns 3 values, not 5: it does not produce the
            # attn/mlp split states. Bind them so the parallel_decoder call below cannot
            # hit a NameError.
            target_states_attn = None
            target_states_mlp = None
            try:
                if replicas:
                    quantizers, target_states, ue_count = shard_calibration_rows(
                        hidden_states, replicas, devices, None, run_rows
                    )
                else:
                    quantizers, target_states, _, _, ue_count = run_rows(
                        devices[0], module, list(range(len(hidden_states)))
                    )
            finally:
                # Free every replica before the next module loads. Without this each layer
                # leaves a full copy of itself resident on the secondary device and the pass
                # OOMs a few layers in.
                for r in replicas:
                    try:
                        r.unload()
                    except Exception:
                        pass
                del replicas
                gc.collect()
                for d in devices[1:]:
                    with torch.cuda.device(d):
                        torch.cuda.empty_cache()
        else:
            quantizers, target_states, target_states_attn, target_states_mlp, ue_count = \
                run_rows(devices[0], module, list(range(len(hidden_states))))

        # For MoE layers, warn if any layer received less than 10% of a calibration batch

        if mode == "block_sparse_moe":
            for j in range(model.config.num_experts):
                ue = ue_count[j]
                if ue > len(hidden_states) * 0.20:
                    print(f" !! Warning: w2.{j} has less than 10% calibration for {ue}/{len(hidden_states)} rows")

        # Measurement

        m = None

        if mode == "self_attn":
            m = measure_attn(module, hidden_states, target_states, quantizers, cache, attn_params)

        if mode == "gdn":
            m = measure_gdn(module, hidden_states, target_states, quantizers, cache, attn_params)

        if mode == "mlp":
            m = measure_mlp(module, hidden_states, target_states, quantizers, cache, attn_params)

        if mode == "block_sparse_moe":
            m = measure_moe_mlp(module, hidden_states, target_states, quantizers, cache, attn_params)

        if mode == "parallel_decoder":
            m = measure_parallel_decoder(module, hidden_states, target_states_attn, target_states_mlp, quantizers, cache, attn_params,
                                         primary_device = primary_device_str)
            target_states_attn = None
            target_states_mlp = None

        quantizers = None

        measurement[module.key + "." + mode] = m

        # # track overall accuracy for status box
        # if m is not None and len(m) > 0:
        #     layer_accuracies = [result['accuracy'] for result in m]
        #     layer_accuracy_sum = sum(layer_accuracies)
        #     layer_accuracy_count = len(layer_accuracies)
        #
        #     accuracy_sum += layer_accuracy_sum
        #     accuracy_count += layer_accuracy_count
        #     overall_rolling_accuracy = accuracy_sum / accuracy_count

        # Unload module

        module.unload()
        gc.collect()
        torch.cuda.empty_cache()

        # Advance

        hidden_states = target_states

        # Timing and status box

        end_time = time.time()
        duration = end_time - begin_time

        time_spent_list.append(duration)
        if len(time_spent_list) > rolling_window_size:
            time_spent_list.pop(0)
        average_time_per_step = sum(time_spent_list) / len(time_spent_list)

        remaining_steps = total_modules - index
        estimated_time_remaining = average_time_per_step * remaining_steps
        completed_steps = index

        completed_module_name_str = f"Measured: {module.key} ({module.name})"
        duration_str = f"Duration: {duration:.2f} seconds"
        completed_step_str = f"Completed step: {completed_steps}/{total_modules}"
        avg_time_str = f"Avg time / step (rolling): {average_time_per_step:.2f} seconds"
        remaining_time_str = f"Estimated remaining time: {get_remaining_time_str(estimated_time_remaining)}"
        # overall_accuracy_str = f"Overall avg accuracy: {overall_rolling_accuracy:.8f}" if accuracy_count > 0 else ""
        last_ckpt_str = f"Last checkpoint layer: {last_ckpt_layer_name}"

        content_lines = [completed_module_name_str,
                         duration_str,
                         completed_step_str,
                         avg_time_str,
                         remaining_time_str,
                         last_ckpt_str]

        # if accuracy_count > 0:
        #     content_lines.append(overall_accuracy_str)

        print_status_box(*content_lines)

        # Checkpoint

        time_since_snapshot = time.time() - last_snapshot_time
        if time_since_snapshot > snapshot_interval_s or index == len(model.modules) - 1:

            print(" -- Saving checkpoint...")

            save_dict = {f"row.{idx:05}": h for idx, h in enumerate(hidden_states)}
            save_file(save_dict, temp_filename)
            save_dict = None

            job["invalid"] = True
            save_fn()

            os.replace(temp_filename, states_filename)

            job["measurement"] = measurement.copy()
            job["last_module_idx"] = index

            last_ckpt_layer_name = f"{module.key} ({module.name})"

            del job["invalid"]
            save_fn()

            last_snapshot_time = time.time()

    print_stage(job, "Measuring", len(model.modules), len(model.modules))

    # Export measurement

    exp_measurement = { "measurement": job["measurement"],
                        "last_module_idx": job["last_module_idx"] }

    measurement_files = [os.path.join(job["out_dir"], "measurement.json")]
    if job["output_measurement"] is not None:
        measurement_files += [job["output_measurement"]]
        print(f" -- Writing {job['output_measurement']}")

    for filename in measurement_files:
        with open(filename, "w", encoding = "utf8") as f:
            f.write(json.dumps(exp_measurement, indent = 4))

    return "completed"  # graceful exiting 
