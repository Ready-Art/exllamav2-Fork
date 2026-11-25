from exllamav2.model import \
(
    ExLlamaV2Embedding,
    ExLlamaV2PosEmbedding,
    ExLlamaV2Attention,
    ExLlamaV2GatedDeltaNet,
    ExLlamaV2MLP,
    ExLlamaV2MoEMLP,
    ExLlamaV2ParallelDecoder,
    ExLlamaV2Linear,
    ExLlamaV2RMSNorm,
    ExLlamaV2LayerNorm
)

from safetensors import safe_open
from safetensors.torch import save_file
from exllamav2.layer_scalar import ExLlamaV2LayerScalar
from exllamav2.qwen4_exp import ExLlamaV2HyperConnection, ExLlamaV2PLELayer
from exllamav2.gemma4_moe import ExLlamaV2Gemma4MoEBlock
from exllamav2.conversion.qparams import QParams, qparams_headoptions, qparams_attn, qparams_mlp, get_qparams_reduced
from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ
import torch
from torch import nn
import os, time, math, json
import torch.nn.functional as F
import gc
from exllamav2.conversion.bot_status import print_stage
from exllamav2.conversion.devices import resolve_devices
from exllamav2.conversion.schedule import quantize_tensor_set
from exllamav2.conversion.stager import WeightStager, free_stage_bufs
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

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


# Quantize

def _configure_exl25(lq: AdaptiveGPTQ, job: dict):
    """
    Apply job["exl25"] options to a fresh quantizer. Must run BEFORE prepare_hessian
    (Hadamard congruence, AWQ scale, identity permutation). Options: uniform_bits
    (2..8, fast lane), group_size (default 128), hadamard (bool, K-side incoherence;
    requires uniform_bits), awq (bool).
    """
    ex = job.get("exl25") or {}
    if not ex: return
    had = 128 if ex.get("hadamard") else 0
    ub = int(ex.get("uniform_bits") or 0)
    if had and not ub:
        raise ValueError("exl25: hadamard requires uniform_bits (the fast lane carries the rotation)")
    if ub and ub not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"exl25: unsupported uniform_bits {ub}")
    # The K-side Hadamard block needs the input (K) dim divisible by the block.
    # Some MoE down_proj intermediate sizes are not (e.g. Gemma-4-26B I=704,
    # 704 % 128 = 64) -- had_rotate_hessian would assert. Skip the rotation for
    # just those tensors: they still quantize on the uniform-bit fast lane (and
    # keep AWQ), merely without the Hadamard quality boost. Per-tensor q_hadamard
    # tag is absent -> the kernel runs the plain uniform path for them.
    if had and lq.rows % had != 0:
        print(f" !! EXL2.5: hadamard skipped for {lq.layer.key} "
              f"(K={lq.rows} not divisible by {had}); uniform {ub}-bit fast lane kept", flush=True)
        had = 0
    lq.exl25_had = had
    lq.exl25_awq = bool(ex.get("awq"))
    lq.exl25_uniform_bits = ub
    lq.exl25_groupsize = int(ex.get("group_size", 128))
    lq.exl25_identity_perm = bool(ub or had)


def quant_linear(job: dict,
                 source: ExLlamaV2Linear,
                 lq: AdaptiveGPTQ or None,
                 qparams: dict,
                 drop = False,
                 rtn = False,
                 device: int = 0,
                 stager = None,
                 stager_key: int = 0):
    """
    Quantize one linear end to end: stage weights, run the GPTQ recursion, pack, save, and verify.

    ``device`` is the CUDA index this tensor is quantized on. It is set on the quantizer as both the compute
    device and the output device, so nothing is bounced back to device 0 just to be written out.
    quant-plan.md §6.4 (M2).

    ``stager`` / ``stager_key`` optionally route the device-to-host weight copy through pinned memory on a side
    stream. quant-plan.md §6.8 (M5).
    """

    qp = QParams.from_dict(qparams)

    # EXL2.5 uniform fast lane: single bit-width, uniform groups (the per-tensor mixed-bit
    # strategy is overridden; arbitrary BPW is preserved across tensors via uniform_bits).
    if getattr(lq, "exl25_uniform_bits", 0):
        qp = QParams(lq.exl25_groupsize, [lq.exl25_uniform_bits], [1.0], qp.scale_bits)

    print(f" -- Linear: {source.key} -> {qp.get_desc()}, {qp.bpw(source.linear.weight.T.shape):.2f} bpw")

    # Pin this tensor to its assigned device. quant_device is the compute device, output_device the gather
    # target for the packed tensors. Both default to 0, which reproduces the single-GPU behaviour exactly.

    lq.quant_device = device
    lq.output_device = device

    # The dequantized weight is a live model tensor, not a file-bound one, so it has to land back on the device
    # that owns the module. Multi-GPU quantization spreads the work but leaves the model's device map untouched,
    # so `source.device_idx` and `device` are the same only on a single-GPU box. Getting this wrong makes the
    # post-quantization forward pass matmul a cuda:N weight against a cuda:M activation. Stage K bug 8.

    src_idx = getattr(source, "device_idx", None)
    lq.target_device = src_idx if isinstance(src_idx, int) and src_idx >= 0 else device

    # Stage weights (M5). This is the only difference from the old inline .cpu().T.clone().float() path,
    # and it produces a bit-identical tensor.

    if stager is not None:
        lq.stage_weights(stager, stager_key)

    # Quantize

    lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
    if rtn:
        lq.quantize_rtn_inplace(keep_qweight = True, apply = True)
    else:
        lq.quantize(keep_qweight = True, apply = True)

    # Pack and save quantized layer

    packed_dict = lq.pack(source.key, qp)
    tensorfile = os.path.join(job["out_dir"], "out_tensor/" + source.key + ".safetensors")
    save_file(packed_dict, tensorfile)

    # Drop buffers from quantizer to free VRAM

    if drop: lq.drop_buffers()

    # Don't reconstruct RTN layers

    if rtn: return

    # Reconstruct from packed layer

    recons_linear = ExLlamaV2Linear(source.model, source.key, source.in_features, source.out_features, source.has_bias)
    recons_linear.device_idx = source.device_idx
    recons_dict = {}
    recons_keys = ["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups"]
    recons_keys += [k for k in ("q_uniform_bits", "q_group_size", "q_hadamard", "q_awq_scale")
                    if source.key + "." + k in packed_dict]
    if source.has_bias: recons_keys += ["bias"]
    r_device = packed_dict[source.key + ".q_weight"].device
    recons_linear.set_device_idx(r_device.index)
    for k in recons_keys:
        recons_dict[k] = packed_dict[source.key + "." + k].to(r_device)
    recons_dict["q_perm"] = torch.argsort(recons_dict["q_invperm"]).to(torch.int)
    recons_linear.load(recons_dict, device_context = False)

    # Sanity test to ensure reconstructed matrix matches unpacked matrix

    # Compare on the reconstruction's device: the weight now lives on the owning module's device, which differs
    # from the worker device under multi-GPU. `.to()` also makes this a copy rather than a view of the live weight,
    # which the in-place `sub_` below would otherwise mutate. Stage K bug 8.
    recons_w = recons_linear.get_weight_tensor_dq()

    # EXL2.5 had/awq: reconstruct() returns the STORED space (H_int @ S @ W / block).
    # Bring it back to logical space before the sanity checks AND the write-back below --
    # otherwise diff1 compares rotated vs logical weights (fatal false positive) and the
    # live model would keep rotated weights, corrupting every later calibration forward.
    if getattr(lq, "exl25_had", 0) or getattr(lq, "awq_inv_scale", None) is not None:
        recons_w = lq.stored_to_logical(recons_w.float()).half()

    quant_w = source.linear.weight.T.to(recons_w.device)

    try:
        if quant_w.numel() <= 1e9:
            ident = torch.eye(recons_linear.in_features, dtype = torch.half, device = r_device)
            recons_w2 = recons_linear.forward(ident, force_cuda = True)
            recons_w2.sub_(quant_w)
            if recons_linear.has_bias: recons_w2.sub_(recons_dict["bias"])
            recons_w2.abs_()
            diff2 = torch.max(recons_w2)
        else:
            diff2 = 0
    except torch.cuda.OutOfMemoryError as e:
        print(f" !! Warning, not enough VRAM for second sanity check of {source.key}")
        diff2 = 0

    quant_w.sub_(recons_w)
    quant_w.abs_()
    diff1 = torch.max(quant_w)
    quant_w = None

    if diff1 > 0.05 or diff2 > 0.075:
        print(" ## Quantization error (2)")
        os._exit(1)
    elif diff1 > 0.01 or diff2 > 0.01:
        print(f" !! Warning, difference of ({diff1:.6f}, {diff2:.6f}) between unpacked and dequantized matrices")

    # Free reconstructed linear layer

    recons_linear.unload()

    # Apply reconstructed matrix to source layer

    source.linear.weight.data = recons_w.T.to(lq.apply_dev())


def _quant_group(job, tasks, devices, kind, stager):
    """
    Quantize a set of independent (linear, quantizer, qparams) tasks across ``devices``.

    Each task is one whole tensor. Nothing is split: see schedule.py for why. With a single device this is a plain
    serial loop and the output is identical to the pre-multi-GPU code.
    """

    def run_one(task, device):
        source, lq, qparams, key = task
        quant_linear(job, source, lq, qparams, device = device, stager = stager, stager_key = key)

    def size_fn(task):
        return task[0].linear.weight.numel()

    quantize_tensor_set(tasks, devices, kind, run_one, size_fn)


def quant_attn(job, module, hidden_states, target_states, quantizers, attn_params, strat, devices = None, stager = None):
    """
    Quantize the attention projections of one module.

    Phase 1 (serial, primary device): every Hessian inversion. ``prepare_hessian()`` mutates ``H`` in place and
    q/k/v share one ``hessian_inv`` via ``reuse_h``, so this half must not run concurrently. quant-plan.md §6.4 / R11.

    Phase 2 (parallel, one device per tensor): the GPTQ recursion and packing, which are independent per tensor.
    """

    devices = devices or [0]
    primary = devices[0]

    # --- Phase 1: Hessians, serial, on the primary device ---

    quantizers["q_proj"].hessian_device = primary
    quantizers["o_proj"].hessian_device = primary

    quantizers["q_proj"].prepare_hessian()
    # v_proj is None for attention_k_eq_v layers (Gemma-4 full attention): V is K,
    # so there is no separate matrix to quantize. PLAN.md §2.5.
    has_v = "v_proj" in quantizers
    quantizers["k_proj"].reuse_h_hessian(quantizers["q_proj"])
    if has_v:
        quantizers["v_proj"].reuse_h_hessian(quantizers["q_proj"])
    quantizers["o_proj"].prepare_hessian()

    # --- Phase 2: weights + recursion + packing, parallel ---

    tasks = [
        (module.q_proj, quantizers["q_proj"], strat["q_proj"], 0),
        (module.k_proj, quantizers["k_proj"], strat["k_proj"], 1),
    ]
    if has_v:
        tasks.append((module.v_proj, quantizers["v_proj"], strat["v_proj"], 2))
    tasks.append((module.o_proj, quantizers["o_proj"], strat["o_proj"], 3))

    _quant_group(job, tasks, devices, "attn", stager)

    for k in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if k in quantizers: del quantizers[k]

    gc.collect()
    torch.cuda.empty_cache()


def quant_gdn(job, module, hidden_states, target_states, quantizers, attn_params, strat, devices = None, stager = None):
    """
    Quantize the gated-delta-net projections (Qwen3.5, newplan.md P6). q/k/v/z share
    the post-norm Hessian; out_proj gets its own from the gated-norm output. in_proj_b/a
    are fp16 and never quantized.
    """

    devices = devices or [0]
    primary = devices[0]

    quantizers["in_proj_q"].hessian_device = primary
    quantizers["out_proj"].hessian_device = primary

    quantizers["in_proj_q"].prepare_hessian()
    quantizers["in_proj_k"].reuse_h_hessian(quantizers["in_proj_q"])
    quantizers["in_proj_v"].reuse_h_hessian(quantizers["in_proj_q"])
    quantizers["in_proj_z"].reuse_h_hessian(quantizers["in_proj_q"])
    quantizers["out_proj"].prepare_hessian()

    tasks = [
        (module.in_proj_q, quantizers["in_proj_q"], strat["in_proj_q"], 0),
        (module.in_proj_k, quantizers["in_proj_k"], strat["in_proj_k"], 1),
        (module.in_proj_v, quantizers["in_proj_v"], strat["in_proj_v"], 2),
        (module.in_proj_z, quantizers["in_proj_z"], strat["in_proj_z"], 3),
        (module.out_proj, quantizers["out_proj"], strat["out_proj"], 4),
    ]

    _quant_group(job, tasks, devices, "gdn", stager)

    for k in ["in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z", "out_proj"]:
        if k in quantizers: del quantizers[k]

    gc.collect()
    torch.cuda.empty_cache()


def quant_mlp(job, module, hidden_states, target_states, quantizers, attn_params, strat, reuse_h_up_proj = None,
              devices = None, stager = None):

    devices = devices or [0]
    primary = devices[0]
    has_mlp = module.model.config.arch.lm.mlp_gate

    # --- Phase 1: Hessians, serial ---

    quantizers["up_proj"].hessian_device = primary
    quantizers["down_proj"].hessian_device = primary

    if reuse_h_up_proj is not None:
        quantizers["up_proj"].reuse_h_hessian(quantizers[reuse_h_up_proj])
        del quantizers[reuse_h_up_proj]
    else:
        quantizers["up_proj"].prepare_hessian()

    if has_mlp:
        quantizers["gate_proj"].reuse_h_hessian(quantizers["up_proj"])

    quantizers["down_proj"].prepare_hessian()

    # --- Phase 2: weights + recursion + packing, parallel ---

    tasks = []
    if has_mlp:
        tasks.append((module.gate_proj, quantizers["gate_proj"], strat["gate_proj"], 0))
    tasks.append((module.up_proj, quantizers["up_proj"], strat["up_proj"], 1 if has_mlp else 0))
    tasks.append((module.down_proj, quantizers["down_proj"], strat["down_proj"], 2 if has_mlp else 1))

    _quant_group(job, tasks, devices, "mlp", stager)

    for k in ["gate_proj", "up_proj", "down_proj"]:
        if k in quantizers: del quantizers[k]

    gc.collect()
    torch.cuda.empty_cache()


def quant_moe_mlp(job, module, hidden_states, target_states, quantizers, attn_params, strat, devices = None,
                  stager = None):

    devices = devices or [0]
    primary = devices[0]
    num_experts = module.model.config.num_experts

    # --- Phase 1: Hessians, serial ---

    for i in range(num_experts):
        quantizers[f"w1.{i}"].hessian_device = primary
        quantizers[f"w2.{i}"].hessian_device = primary
        quantizers[f"w3.{i}"].hessian_device = primary

    quantizers["w1.0"].prepare_hessian()
    for i in range(num_experts):
        if i > 0: quantizers[f"w1.{i}"].reuse_h_hessian(quantizers["w1.0"])
        quantizers[f"w2.{i}"].prepare_hessian()
        quantizers[f"w3.{i}"].reuse_h_hessian(quantizers["w1.0"])

    # --- Phase 2: weights + recursion + packing, parallel ---

    tasks = []
    key = 0
    for i in range(num_experts):
        tasks.append((module.w1[i], quantizers[f"w1.{i}"], strat["w1"], key)); key += 1
        tasks.append((module.w3[i], quantizers[f"w3.{i}"], strat["w3"], key)); key += 1
        tasks.append((module.w2[i], quantizers[f"w2.{i}"], strat["w2"], key)); key += 1

    _quant_group(job, tasks, devices, "moe", stager)

    for i in range(num_experts):
        for k in [f"w1.{i}", f"w2.{i}", f"w3.{i}"]:
            if k in quantizers: del quantizers[k]


def quant_lm_head(job, module, hidden_states, quantizers, attn_params, rtn = False, device = 0, stager = None):

    # head_bits == 16 means "do not quantize the head": leave the fp16 lm_head in place.
    # qparams_headoptions[16] is None, so calling .get_dict() on it would raise AttributeError.
    # PLAN.md §2.7 step 2.
    if job["head_bits"] == 16:
        return

    qp = qparams_headoptions[job["head_bits"]]
    q = quantizers["lm_head"]

    q.hessian_device = device
    q.prepare_hessian(no_h_inv = rtn)
    quant_linear(job, module, q, qp.get_dict(), drop = True, rtn = rtn, device = device, stager = stager)


def quant_parallel_decoder(job, module, hidden_states, target_states, quantizers, attn_params, strat_attn, strat_mlp,
                           devices = None, stager = None):

    print(f" -- Sublayer: {module.key}.self_attn")
    quant_attn(job, module.attn, hidden_states, target_states, quantizers, attn_params, strat_attn,
               devices = devices, stager = stager)
    print(f" -- Sublayer: {module.key}.mlp")
    quant_mlp(job, module.mlp, hidden_states, target_states, quantizers, attn_params, strat_mlp, "q_proj",
              devices = devices, stager = stager)


# def testc(module, states, target_states, norm, layers):
#
#     rows = len(states)
#     cols = states[0].shape[1]
#     dim = module.model.config.hidden_size
#
#     a_batch = torch.empty((rows * cols, dim), dtype = torch.float, device = "cuda:0")
#     b_batch = torch.empty((rows * cols, dim), dtype = torch.float, device = "cuda:0")
#
#     r = 0
#     for state, target_state in zip(states, target_states):
#         a = norm.forward(state.to("cuda:0"))
#         b = norm.forward(target_state.to("cuda:0"))
#         a_batch[r:r+cols] = a.view(-1, dim)
#         b_batch[r:r+cols] = b.view(-1, dim)
#         r += cols
#
#     # diff = F.mse_loss(b_batch, a_batch)
#     m_a = torch.mean(a_batch.abs(), dim = 0)
#     m_b = torch.mean(b_batch.abs(), dim = 0)
#     m_ab = m_b / m_a
#     # a_batch *= m_ab
#     # diff = F.mse_loss(b_batch, a_batch)
#     norm.weight.data *= m_ab
#
#     # s = torch.linalg.lstsq(a_batch, b_batch)
#     # s = s.solution
#     #
#     # for linear in layers:
#     #     m = torch.matmul(s, linear.linear.weight.data.T.float())
#     #     linear.linear.weight.data = nn.Parameter(m.T.half())
#
#     xx = 0


@torch.inference_mode()
def quant(job, save_fn, model):

    last_snapshot_time = time.time()
    snapshot_interval_s = 180

    temp_filename = os.path.join(job["out_dir"], "hidden_states_temp.safetensors")
    states_filename = os.path.join(job["out_dir"], "hidden_states.safetensors")
    strategy = job["strategy"]

    # EXL2.5: every quantizer gets the job's exl25 options at creation, before calibration.
    def mk_q(layer):
        lq = AdaptiveGPTQ(layer)
        _configure_exl25(lq, job)
        return lq

    # Multi-GPU setup. quant-plan.md §6.
    #
    # ``devices`` is the ordered list of CUDA indices this job may use; the first entry is the primary and
    # owns every Hessian inversion. With the default (no -d flag deflates to [0]) this is a single-element
    # list and every code path below degenerates to the historical serial behaviour.

    devices = resolve_devices(job.get("devices"))
    if not devices:
        devices = [0]
    primary = devices[0]
    if len(devices) > 1:
        print(f" -- Parallel quantization across devices: {devices}")

    # Pinned weight staging (M5). Single-GPU win: it replaces the synchronous pageable
    # .cpu().T.clone().float().contiguous() with a pinned D2H copy on a side stream.

    stager = WeightStager(primary) if job.get("pinned_staging", True) else None

    # Quantize

    if not "q_last_module_idx" in job:
        job["q_last_module_idx"] = 0

    hidden_states = []
    # hidden_i_states = []
    with safe_open(states_filename, framework = "pt", device = "cpu") as f:
        for k in sorted(f.keys()):
            if k.startswith("row"):
                hidden_states.append(f.get_tensor(k))

    index = job["q_last_module_idx"]
    while True:

        print_stage(job, "Quantizing", index, len(model.modules))

        index += 1
        if index >= len(model.modules): break

        # Prepare module

        module = model.modules[index]

        rtn = False
        if module.key == "lm_head" and module.numel() > 1e9:  # every part of the buffalo
            model.free_device_context()
            gc.collect()
            torch.cuda.empty_cache()
            rtn = True

        module.load()

        # Device this module's forward passes run on. The model's device map is unchanged by multi-GPU
        # quantization: the module still lives where the loader put it, and only the quantizer work is spread.
        # quant-plan.md §6.4.
        if isinstance(module.device_idx, int) and module.device_idx >= 0:
            module_dev = torch.device(module.device_idx)
        else:
            module_dev = torch.device(primary)

        print(f" -- Layer: {module.key} ({module.name})")

        # Create quantizers

        quantizers = {}

        if isinstance(module, ExLlamaV2Attention):
            mode = "self_attn"
            # if index > 1: testc(module, hidden_states, hidden_i_states, module.input_layernorm, [module.q_proj, module.k_proj, module.v_proj])
            quantizers["q_proj"] = mk_q(module.q_proj.linear)
            quantizers["k_proj"] = mk_q(module.k_proj.linear)
            # v_proj is None for attention_k_eq_v layers (Gemma-4 full attention).
            # PLAN.md §2.5 / §2.7.
            if module.v_proj is not None:
                quantizers["v_proj"] = mk_q(module.v_proj.linear)
            quantizers["o_proj"] = mk_q(module.o_proj.linear)

        elif isinstance(module, ExLlamaV2GatedDeltaNet):
            # Qwen3.5 gated delta net (newplan.md P6). in_proj_b/a stay fp16.
            mode = "gdn"
            quantizers["in_proj_q"] = mk_q(module.in_proj_q.linear)
            quantizers["in_proj_k"] = mk_q(module.in_proj_k.linear)
            quantizers["in_proj_v"] = mk_q(module.in_proj_v.linear)
            quantizers["in_proj_z"] = mk_q(module.in_proj_z.linear)
            quantizers["out_proj"] = mk_q(module.out_proj.linear)

        elif isinstance(module, ExLlamaV2MLP):
            mode = "mlp"
            has_mlp = model.config.arch.lm.mlp_gate
            # testc(module, hidden_states, hidden_i_states, module.post_attention_layernorm, [module.gate_proj, module.up_proj])
            if has_mlp: quantizers["gate_proj"] = mk_q(module.gate_proj.linear)
            quantizers["up_proj"] = mk_q(module.up_proj.linear)
            quantizers["down_proj"] = mk_q(module.down_proj.linear)

        elif isinstance(module, ExLlamaV2Gemma4MoEBlock):
            mode = "block_sparse_moe"
            for i in range(model.config.num_experts):
                quantizers[f"w1.{i}"] = mk_q(module.w1[i].linear)
                quantizers[f"w3.{i}"] = mk_q(module.w3[i].linear)
                quantizers[f"w2.{i}"] = mk_q(module.w2[i].linear)

        elif isinstance(module, ExLlamaV2MoEMLP):
            mode = "block_sparse_moe"
            for i in range(model.config.num_experts):
                quantizers[f"w1.{i}"] = mk_q(module.w1[i].linear)
                quantizers[f"w3.{i}"] = mk_q(module.w3[i].linear)
                quantizers[f"w2.{i}"] = mk_q(module.w2[i].linear)

        elif isinstance(module, ExLlamaV2Linear):
            mode = "linear"
            assert module.key == model.config.arch.lm_prefix + "lm_head"
            quantizers["lm_head"] = mk_q(module.linear)
            # EXL2.5: Hadamard requires an fp16 head (-hb 16). With a quantized head, skip
            # the rotation on the head tensor. Decided HERE, before calibration, so the
            # head's Hessian is collected in the same space its weights are quantized in.
            if quantizers["lm_head"].exl25_had and job["head_bits"] != 16:
                print(" !! EXL2.5 hadamard requires -hb 16 (fp16 head); skipping hadamard on the quantized lm_head")
                quantizers["lm_head"].exl25_had = 0

        elif isinstance(module, ExLlamaV2RMSNorm) or isinstance(module, ExLlamaV2LayerNorm):
            mode = "norm"

        elif isinstance(module, ExLlamaV2LayerScalar):
            # 1-element pass-through tensor, never quantized. PLAN.md §2.7.
            mode = "layer_scalar"

        elif isinstance(module, ExLlamaV2PosEmbedding):
            mode = "pos_emb"

        elif isinstance(module, ExLlamaV2HyperConnection):
            mode = "hc"

        elif isinstance(module, ExLlamaV2PLELayer):
            mode = "ple"

        elif isinstance(module, ExLlamaV2ParallelDecoder):
            mode = "parallel_decoder"
            quantizers["q_proj"] = mk_q(module.attn.q_proj.linear)
            quantizers["k_proj"] = mk_q(module.attn.k_proj.linear)
            quantizers["v_proj"] = mk_q(module.attn.v_proj.linear)
            quantizers["o_proj"] = mk_q(module.attn.o_proj.linear)
            has_gate = module.model.config.arch.lm.mlp_gate
            if has_gate: quantizers["gate_proj"] = mk_q(module.mlp.gate_proj.linear)
            quantizers["up_proj"] = mk_q(module.mlp.up_proj.linear)
            quantizers["down_proj"] = mk_q(module.mlp.down_proj.linear)

        # Reference forward pass

        cache = None
        attn_params = ExLlamaV2Attention.Params(1, hidden_states[0].shape[1], 0, None, None) \
            if mode in ["self_attn", "parallel_decoder"] else None

        # layer_scalar is a 1-element pass-through tensor: no calibration, no forward pass, no
        # Hessian. Leave the hidden states untouched and let compile.py copy the tensor through.
        if mode == "layer_scalar":
            module.unload()
            torch.cuda.empty_cache()
            continue

        target_states = []

        if mode == "block_sparse_moe":
            uncalibrated_experts = [0 for _ in range(model.config.num_experts)]

        for i in range(len(hidden_states)):

            x = hidden_states[i].to(module_dev)
            if mode == "ple":
                outputs = { "hidden_states": x }
            else:
                outputs = module.forward(x, cache, attn_params, intermediates = True)

            # Hessians

            if mode == "self_attn":
                quantizers["q_proj"].add_batch(outputs["post_norm"])  # Reuse H for K and V
                quantizers["o_proj"].add_batch(outputs["attn_output"])

            if mode == "gdn":
                quantizers["in_proj_q"].add_batch(outputs["post_norm"])  # Reuse H for k/v/z
                quantizers["out_proj"].add_batch(outputs["pre_out"])

            if mode == "mlp":
                quantizers["up_proj"].add_batch(outputs["post_norm"])  # Reuse H for gate_proj
                quantizers["down_proj"].add_batch(outputs["pre_down"])

            if mode == "block_sparse_moe":
                for j in range(model.config.num_experts):
                    if f"pre_down.{j}" in outputs:
                        if j == 0: quantizers[f"w1.{j}"].add_batch(outputs["post_norm"])
                        quantizers[f"w2.{j}"].add_batch(outputs[f"pre_down.{j}"])
                        if outputs[f"pre_down.{j}"].shape[0] < outputs["post_norm"].shape[0] / 10:
                            uncalibrated_experts[j] += 1
                    else:
                        uncalibrated_experts[j] += 1

            if mode == "parallel_decoder":
                quantizers["q_proj"].add_batch(outputs["post_norm"])  # Reuse H for K, V, up_proj and gate_proj
                quantizers["o_proj"].add_batch(outputs["attn_output"])
                quantizers["down_proj"].add_batch(outputs["pre_down"])

            if mode == "linear":
                quantizers["lm_head"].add_batch(x)

            if mode != "linear":
                target_states.append(outputs["hidden_states"].to("cpu"))

            outputs = None

        # For MoE layers, warn if any expert received less than 20% of a calibration batch

        if mode == "block_sparse_moe":
            for j in range(model.config.num_experts):
                ue = uncalibrated_experts[j]
                if ue > len(hidden_states) * 0.20:
                    print(f" !! Warning: w2.{j} has less than 10% calibration for {ue}/{len(hidden_states)} rows")

        # fp16 pass-through modules (hyper-connections, PLE): nothing to quantize;
        # compile.py copies the raw weights. Propagate the reference forward states.
        if mode in ("hc", "ple"):
            hidden_states = target_states
            module.unload()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Conversion

        if mode == "self_attn":
            strat = strategy[module.key + "." + mode]
            quant_attn(job, module, hidden_states, target_states, quantizers, attn_params, strat,
                       devices = devices, stager = stager)

        if mode == "gdn":
            strat = strategy[module.key + "." + mode]
            quant_gdn(job, module, hidden_states, target_states, quantizers, attn_params, strat,
                      devices = devices, stager = stager)

        if mode == "mlp":
            strat = strategy[module.key + "." + mode]
            quant_mlp(job, module, hidden_states, target_states, quantizers, attn_params, strat,
                      devices = devices, stager = stager)

        if mode == "block_sparse_moe":
            strat = strategy[module.key + "." + mode]
            quant_moe_mlp(job, module, hidden_states, target_states, quantizers, attn_params, strat,
                          devices = devices, stager = stager)

        if mode == "linear":
            model.drop_device_context()
            gc.collect()  # shruge
            torch.cuda.empty_cache()
            quant_lm_head(job, module, hidden_states, quantizers, attn_params, rtn,
                          device = primary, stager = stager)

        if mode == "parallel_decoder":
            strat_attn = strategy[module.key + ".self_attn"]
            strat_mlp = strategy[module.key + ".mlp"]
            quant_parallel_decoder(job, module, hidden_states, target_states, quantizers, attn_params, strat_attn, strat_mlp,
                                   devices = devices, stager = stager)

        torch.cuda.synchronize()
        quantizers.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # Post-quantization forward pass

        if mode == "linear":
            with safe_open(job["cal_filename"], framework = "pt", device = "cpu") as f:
                cal_ids = f.get_tensor("input_ids")
            module.linear.weight.data = module.linear.weight.data.to(module_dev)

        # Accumulate on the device the forward pass runs on, not on whatever `.cuda()` resolves to in this thread.
        rfn_sum = torch.tensor(0.0, device = module_dev)
        rfn_count = 0
        logprob_sum = 0.0
        logprob_count = 0

        q_states = []
        for i in range(len(hidden_states)):

            if mode != "linear":

                x = hidden_states[i].to(module_dev)
                output = module.forward(x, cache, attn_params)
                x = None
                q_states.append(output.to("cpu"))

                output = output[0].float()
                output_ref = target_states[i].to(module_dev)
                output_ref = output_ref[0].float()

                rfn_sum += torch.linalg.norm(output - output_ref, 'fro') / torch.linalg.norm(output_ref, 'fro')
                rfn_count += 1

                output_ref = None
                output = None

            elif i < job["measurement_rows"]:

                x = hidden_states[i].to(module_dev)
                output = module.forward(x, cache, attn_params)
                if module.padding > 0: output = output[:, :, :-module.padding]

                if model.config.final_logit_softcapping:
                    output = output.contiguous()
                    ext_c.softcap_(output, model.config.final_logit_softcapping)

                logits = output[:, :-1, :]
                logits = logits.float() + 1e-10
                target_ids = cal_ids[i:i+1, 1:].to(module_dev)

                log_probs = F.log_softmax(logits, dim = -1)
                token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                logprob_sum += token_log_probs.sum().item()
                logprob_count += target_ids.numel()

                output = None
                logits = None
                token_log_probs = None

        if mode != "linear":

            err = rfn_sum.item() / rfn_count
            print(f" -- Module quantized, rfn_error: {err:1.6f}")

        else:

            mean_log_prob = logprob_sum / logprob_count
            perplexity = math.exp(-mean_log_prob)

            print(f" -- Module quantized, calibration perplexity (quant): {perplexity:.4f}")

        # Unload module

        module.unload()
        torch.cuda.empty_cache()

        # Advance

        if mode != "linear":
            # hidden_i_states = hidden_states
            # hidden_states = target_states
            # hidden_states = [(x + y) / 2 for x, y in zip(target_states, q_states)]
            hidden_states = q_states
            q_states = None

        # Checkpoint

        time_since_snapshot = time.time() - last_snapshot_time
        if time_since_snapshot > snapshot_interval_s or index == len(model.modules) - 1:

            print(" -- Saving checkpoint...")

            if mode != "linear":
                save_dict = {f"row.{idx:05}": h for idx, h in enumerate(hidden_states)}
                # save_dict.update( {f"i_row.{idx:05}": h for idx, h in enumerate(hidden_i_states)} )
                save_file(save_dict, temp_filename)
                save_dict = None

            job["invalid"] = True
            save_fn()

            if mode != "linear":
                os.replace(temp_filename, states_filename)

            job["q_last_module_idx"] = index

            del job["invalid"]
            save_fn()

            last_snapshot_time = time.time()

    print_stage(job, "Quantizing", len(model.modules), len(model.modules))

    # Release the pinned staging arena. It is process-global and can be hundreds of MB on a large model.

    if stager is not None: stager.drain()
    free_stage_bufs()
