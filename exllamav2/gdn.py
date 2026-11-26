from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.compat import safe_move_tensor
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.tensor_p import (
    BROADCAST_RS,
    BROADCAST_GDN_K,
    BROADCAST_GDN_V,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2
    from exllamav2.lora import ExLlamaV2Lora


# ---------------------------------------------------------------------------
# Gated delta rule, pure-torch reference math (fp32 state).
#
# Ported from the HF Qwen3.5 / Qwen3-Next reference (torch_chunk_gated_delta_rule
# / torch_recurrent_gated_delta_rule) and cross-checked against exllamav3's
# gated_delta_net_fn. NO Triton: prefill runs the chunked form in torch, decode
# runs the per-token recurrence. A CUDA C++ extension kernel is the follow-up
# perf lever (newplan.md P6); correctness first.
#
# State layout: (B, nv, dk, dv) fp32. Decay g is a per-value-head scalar; the
# delta rule is independent per head, which is what makes the module shard
# cleanly across TP ranks by value-head groups.
# ---------------------------------------------------------------------------

def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def _recurrent_scan(q, k, v, g, beta, state):
    """
    Per-token gated delta recurrence, fp32. q/k/v: (B, T, H, D) (q/k already
    l2-normalised, q pre-scaled), g/beta: (B, T, H). state: (B, H, dk, dv),
    advanced IN PLACE. Returns output (B, T, H, dv).
    """
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    scale = dk ** -0.5
    q = q * scale
    out = torch.empty((B, T, H, dv), dtype=torch.float32, device=q.device)
    for t in range(T):
        q_t = q[:, t, :, :].unsqueeze(-1)          # (B, H, dk, 1)
        k_t = k[:, t, :, :].unsqueeze(-1)          # (B, H, dk, 1)
        v_t = v[:, t, :, :]                        # (B, H, dv)
        g_t = g[:, t, :].exp().unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
        beta_t = beta[:, t, :].unsqueeze(-1)       # (B, H, 1)
        state.mul_(g_t)
        kv_mem = (state * k_t).sum(dim=-2)         # (B, H, dv)
        delta = (v_t - kv_mem) * beta_t
        state.add_(k_t * delta.unsqueeze(-2))
        out[:, t] = (state * q_t).sum(dim=-2)
    return out


def _chunk_scan(q, k, v, g, beta, state, chunk_size: int = 64):
    """
    Chunked gated delta rule for prefill, fp32. Direct port of HF
    torch_chunk_gated_delta_rule (UT transform + sequential chunk scan), which
    is mathematically identical to the per-token recurrence. q/k: (B, T, H, dk)
    l2-normalised (unscaled); v: (B, T, H, dv); g/beta: (B, T, H).
    state: (B, H, dk, dv) advanced IN PLACE. Returns output (B, T, H, dv).
    """
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    q = q * (dk ** -0.5)
    q, k, v, beta, g = [x.transpose(1, 2).contiguous() for x in (q, k, v, beta, g)]

    C = chunk_size
    pad = (C - T % C) % C
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    Tp = T + pad
    nc = Tp // C

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = [x.reshape(B, H, nc, C, x.shape[-1]) for x in (q, k, k_beta, v_beta)]
    g = g.reshape(B, H, nc, C)

    upper = torch.ones(C, C, dtype=torch.bool, device=q.device).triu(1)
    cum = g.cumsum(-1)
    pairwise = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()

    ut_system = (k_beta @ k.transpose(-1, -2)) * pairwise
    intra_chunk_attn = (q @ k.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)

    # UT transform: solve the unit-lower-triangular system (I - tril(ut, -1)).
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)

    q = q * cum.exp().unsqueeze(-1)
    k = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]

    out = torch.zeros_like(new_values)
    for i in range(nc):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = q[:, :, i] @ state + intra_chunk_attn[:, :, i] @ v_new
        state.mul_(chunk_decay[:, :, i]).add_(k[:, :, i].transpose(-1, -2) @ v_new)

    out = out.reshape(B, H, Tp, dv)[:, :, :T]
    return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class ExLlamaV2GatedDeltaNet(ExLlamaV2Module):
    """
    Qwen3.5 gated-delta-net linear-attention block (newplan.md P6).

    Replaces self_attn on "linear_attention" layers. No K/V cache: the block
    carries a depthwise short-conv state (last K-1 raw inputs per channel) and
    a fp32 recurrent state per value head, both owned by the hybrid cache.

    Projections: in_proj_qkv is stored fused in the checkpoint ([q | k | v]);
    it is loaded as three ExLlamaV2Linears via the f_key slice mechanism so
    each part shards independently under TP (q/k by key heads, v by value
    heads). in_proj_b/a (beta / decay gate logits) stay fp16: tiny and
    sensitive. out_proj is row-parallel under TP (value-head input split).
    """

    name: str = "GatedDeltaNet"

    layer_idx: int
    hidden_size: int
    nk: int
    nv: int
    dk: int
    dv: int
    group: int
    kd: int
    vd: int
    conv_dim: int
    conv_kernel: int

    pre_layernorm: ExLlamaV2RMSNorm
    in_proj_q: ExLlamaV2Linear
    in_proj_k: ExLlamaV2Linear
    in_proj_v: ExLlamaV2Linear
    in_proj_z: ExLlamaV2Linear
    in_proj_b: ExLlamaV2Linear
    in_proj_a: ExLlamaV2Linear
    out_proj: ExLlamaV2Linear

    conv1d_weight: torch.Tensor | None     # (conv_dim, K) fp16
    A_log: torch.Tensor | None             # (nv,) fp32
    dt_bias: torch.Tensor | None           # (nv,) fp32
    norm_weight: torch.Tensor | None       # (dv,) fp16, gated RMSNorm (plain w)

    is_tp: bool
    tp_dq_size: list[int] | None

    # TP-local geometry (rank-local / per-device shard)
    tp_nk: int
    tp_nv: int
    tp_k_a: int
    tp_v_a: int

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        layer_idx: int,
        archparams = None,
        has_norm: bool = True,
        has_residual: bool = True,
    ):
        super().__init__(model, key, archparams)
        cfg = model.config
        ap = self.archparams
        km = ap.keys

        # Qwen4-Exp (Flash-Next) wraps the block in hyper-connections: the grouped
        # hc_norm replaces the block's own pre-norm and the residual add happens in
        # the HC inject module instead. P6 (Qwen3.5) keeps both defaults.
        self.has_norm = has_norm
        self.has_residual = has_residual
        # Gated-norm activation: Qwen3.5 "swish"/silu; Qwen4-Exp output_gate_type
        # "sigmoid" applies to the GDN gated norm (the attn output gate is always
        # sigmoid). HF Qwen4ExpTextGatedDeltaNet.
        self.norm_act = getattr(cfg, "output_gate_type", "silu")

        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.nk = cfg.linear_num_key_heads
        self.nv = cfg.linear_num_value_heads
        self.dk = cfg.linear_key_head_dim
        self.dv = cfg.linear_value_head_dim
        self.group = self.nv // self.nk
        self.kd = self.nk * self.dk
        self.vd = self.nv * self.dv
        self.conv_dim = 2 * self.kd + self.vd
        self.conv_kernel = cfg.linear_conv_kernel_dim

        self.is_tp = False
        self.tp_dq_size = None
        self.tp_nk = self.nk
        self.tp_nv = self.nv
        self.tp_k_a = 0
        self.tp_v_a = 0

        self.pre_layernorm = ExLlamaV2RMSNorm(model, key + km["norm_1"], archparams = ap) \
            if has_norm else None

        fused = key + ".linear_attn.in_proj_qkv"
        self.in_proj_q = ExLlamaV2Linear(model, fused, self.hidden_size, self.kd, False,
                                         f_key = fused, f_beg = 0, f_end = self.kd,
                                         archparams = ap)
        self.in_proj_k = ExLlamaV2Linear(model, fused + "_k", self.hidden_size, self.kd, False,
                                         f_key = fused, f_beg = self.kd, f_end = 2 * self.kd,
                                         archparams = ap)
        self.in_proj_v = ExLlamaV2Linear(model, fused + "_v", self.hidden_size, self.vd, False,
                                         f_key = fused, f_beg = 2 * self.kd, f_end = self.conv_dim,
                                         archparams = ap)
        self.in_proj_z = ExLlamaV2Linear(model, key + ".linear_attn.in_proj_z",
                                         self.hidden_size, self.vd, False, archparams = ap)
        # Gate logits: fp16, never quantized (48 outputs each; sensitivity).
        # pad32=False: the gate width is exactly nv (no kernel padding wanted).
        self.in_proj_b = ExLlamaV2Linear(model, key + ".linear_attn.in_proj_b",
                                         self.hidden_size, self.nv, False, pad32 = False, archparams = ap)
        self.in_proj_a = ExLlamaV2Linear(model, key + ".linear_attn.in_proj_a",
                                         self.hidden_size, self.nv, False, pad32 = False, archparams = ap)
        self.out_proj = ExLlamaV2Linear(model, key + ".linear_attn.out_proj",
                                        self.vd, self.hidden_size, False, archparams = ap)

        self.conv1d_weight = None
        self.A_log = None
        self.dt_bias = None
        self.norm_weight = None

        self.submodules = [
            self.in_proj_q, self.in_proj_k, self.in_proj_v,
            self.in_proj_z, self.in_proj_b, self.in_proj_a,
            self.out_proj,
        ]
        if self.pre_layernorm is not None:
            self.submodules.insert(0, self.pre_layernorm)

        # Raw (non-linear) weight footprint estimate for lazy-load accounting.
        self.assumed_footprint = 2 * (
            self.conv_dim * self.conv_kernel + 2 * self.nv + self.dv
        )

    # ------------------------------------------------------------------ load

    def numel(self) -> int:
        return sum(m.numel() for m in self.submodules) + \
            self.conv_dim * self.conv_kernel + 2 * self.nv + self.dv

    def _load_raw(self, cpu: bool = False):
        dev = "cpu" if cpu else self.device()
        w = self.load_weight(self.key + ".linear_attn.conv1d", cpu = cpu)
        assert isinstance(w, nn.Parameter), self.key + ".linear_attn.conv1d.weight expected"
        # checkpoint layout (conv_dim, 1, K) -> (conv_dim, K)
        self.conv1d_weight = w.data.view(self.conv_dim, self.conv_kernel).to(dev)
        # A_log / dt_bias are BARE keys in the checkpoint (no .weight suffix).
        bare = self.load_multi(self.key + ".linear_attn", ["A_log", "dt_bias"], cpu = cpu)
        self.A_log = bare["A_log"].float().to(dev)
        self.dt_bias = bare["dt_bias"].float().to(dev)
        w = self.load_weight(self.key + ".linear_attn.norm", cpu = cpu)
        self.norm_weight = w.data.half().to(dev)

    @torch.inference_mode()
    def load(self, device_context: bool = True):
        if self.pre_layernorm is not None: self.pre_layernorm.load()
        self.in_proj_q.load(device_context = device_context)
        self.in_proj_k.load(device_context = device_context)
        self.in_proj_v.load(device_context = device_context)
        self.in_proj_z.load(device_context = device_context)
        self.in_proj_b.load(device_context = device_context)
        self.in_proj_a.load(device_context = device_context)
        self.out_proj.load(device_context = device_context)
        self._load_raw()

    def unload(self):
        for m in self.submodules:
            m.unload()
        self.conv1d_weight = None
        self.A_log = None
        self.dt_bias = None
        self.norm_weight = None

    def weight_footprint(self) -> int:
        fp = sum(m.weight_footprint() for m in self.submodules)
        fp += 2 * (self.conv_dim * self.conv_kernel + 2 * self.nv + self.dv)
        return fp

    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)
        for m in self.submodules:
            m.set_device_idx(idx)
        if self.conv1d_weight is not None:
            self.conv1d_weight = safe_move_tensor(self.conv1d_weight, self.device())
            self.A_log = safe_move_tensor(self.A_log, self.device())
            self.dt_bias = safe_move_tensor(self.dt_bias, self.device())
            self.norm_weight = safe_move_tensor(self.norm_weight, self.device())

    def is_quant(self) -> bool:
        return self.in_proj_q.is_quant()

    # ------------------------------------------------------------- scratch

    def temp_dq_size(self) -> int:
        return max(m.temp_dq_size() for m in (
            self.in_proj_q, self.in_proj_k, self.in_proj_v,
            self.in_proj_z, self.out_proj))

    def scratch_space_fixed(self) -> int:
        return self.temp_dq_size()

    def scratch_space(self) -> int:
        return self.temp_dq_size()

    def scratch_space_tp(self):
        cfg = self.model.config
        ctx = self.model.tp_context
        devs = ctx.num_devices
        scratch = [0] * devs

        def amax(res):
            for i, s in enumerate(res):
                scratch[i] = max(scratch[i], s)

        amax(self.in_proj_q.scratch_space_tp(BROADCAST_GDN_K, self.dk))
        amax(self.in_proj_k.scratch_space_tp(BROADCAST_GDN_K, self.dk))
        amax(self.in_proj_v.scratch_space_tp(BROADCAST_GDN_V, self.dv))
        amax(self.in_proj_z.scratch_space_tp(BROADCAST_GDN_V, self.dv))
        amax(self.out_proj.scratch_space_tp(BROADCAST_GDN_V, self.dv))
        self.tp_dq_size = [s for s in scratch]
        return scratch

    # ------------------------------------------------------------------ TP

    def tp_split(self):
        """Single-process TP: per-device shard lists (mode 1)."""
        cfg = self.model.config
        ctx = self.model.tp_context
        ctx.set_geometry(self.layer_idx)

        if self.pre_layernorm is not None: self.pre_layernorm.tp_split(BROADCAST_RS)
        self.in_proj_q.tp_split(BROADCAST_GDN_K, dim = self.dk)
        self.in_proj_k.tp_split(BROADCAST_GDN_K, dim = self.dk)
        self.in_proj_v.tp_split(BROADCAST_GDN_V, dim = self.dv)
        self.in_proj_z.tp_split(BROADCAST_GDN_V, dim = self.dv)
        # b/a: fp16, replicated on the primary (tiny).
        self.in_proj_b.tp_replicate(ctx.device)
        self.in_proj_a.tp_replicate(ctx.device)
        # out_proj row-parallel over value heads; partials summed on primary.
        self.out_proj.tp_split_row(BROADCAST_GDN_V, BROADCAST_RS, dim = self.dv)
        self.out_proj.out_features_tp = []

        self.is_tp = True
        self.set_device_idx(None)

    def tp_load_rank(self, rank: int):
        """MP rank-local: load only this rank's shard (mode 2)."""
        cfg = self.model.config
        ctx = self.model.tp_context
        self.device_idx = rank

        if self.pre_layernorm is not None: self.pre_layernorm.tp_load_rank(rank, BROADCAST_RS)

        self.in_proj_q.tp_load_rank(rank, BROADCAST_GDN_K, dim = self.dk)
        self.in_proj_k.tp_load_rank(rank, BROADCAST_GDN_K, dim = self.dk)
        self.in_proj_v.tp_load_rank(rank, BROADCAST_GDN_V, dim = self.dv)
        self.in_proj_z.tp_load_rank(rank, BROADCAST_GDN_V, dim = self.dv)
        self.in_proj_b.tp_load_rank_replicated(rank)
        self.in_proj_a.tp_load_rank_replicated(rank)
        self.out_proj.tp_load_rank_row(rank, BROADCAST_GDN_V, BROADCAST_RS, dim = self.dv)

        # Local geometry from the value-head split.
        split_v = ctx.get_split(BROADCAST_GDN_V)
        target = [(d, a, b) for d, a, b in split_v if d == rank]
        if target:
            _, a, b = target[0]
            self.tp_v_a = a
            self.tp_nv = b - a
        else:
            self.tp_v_a = 0
            self.tp_nv = 0
        self.tp_nk = self.tp_nv // self.group
        self.tp_k_a = self.tp_v_a // self.group

        self._load_raw()
        self._slice_raw_tp()

        self.is_tp = True

    def _slice_raw_tp(self):
        """Slice conv / A_log / dt_bias / norm to this rank's head shard."""
        if self.tp_nv == self.nv:
            return
        ka, kb = self.tp_k_a, self.tp_k_a + self.tp_nk
        va, vb = self.tp_v_a, self.tp_v_a + self.tp_nv
        q_lo, q_hi = ka * self.dk, kb * self.dk
        k_lo, k_hi = self.kd + ka * self.dk, self.kd + kb * self.dk
        v_lo, v_hi = 2 * self.kd + va * self.dv, 2 * self.kd + vb * self.dv
        idx = torch.cat([
            torch.arange(q_lo, q_hi),
            torch.arange(k_lo, k_hi),
            torch.arange(v_lo, v_hi),
        ])
        self.conv1d_weight = self.conv1d_weight[idx].contiguous()
        self.A_log = self.A_log[va:vb].contiguous()
        self.dt_bias = self.dt_bias[va:vb].contiguous()
        # norm weight is per-head-dim (dv), shared across heads: keep full.

    # -------------------------------------------------------------- forward

    def _gates(self, b: torch.Tensor, a: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        return beta, g

    def _conv_prefill(self, qkv: torch.Tensor, conv_state: torch.Tensor | None):
        """
        Causal depthwise conv over (B, T, conv_dim) with state carry.
        conv_state: (B, conv_dim, K-1) holding the last K-1 RAW inputs, or None
        (zero padding). Returns (convolved (B, T, conv_dim), new_state).
        """
        B, T, C = qkv.shape
        K = self.conv_kernel
        x = qkv.transpose(1, 2)                       # (B, C, T)
        if conv_state is not None:
            x = torch.cat([conv_state, x], dim=-1)    # (B, C, K-1+T)
        else:
            x = F.pad(x, (K - 1, 0))
        w = self.conv1d_weight.view(C, 1, K)
        y = F.conv1d(x, w, groups = C)                # (B, C, T)
        new_state = x[..., -(K - 1):].contiguous() if T >= 1 else x[..., -(K-1):]
        return y.transpose(1, 2), new_state

    def _conv_decode(self, qkv: torch.Tensor, conv_state: torch.Tensor):
        """Single-token conv step. qkv: (B, 1, C); state (B, C, K-1) updated in place."""
        B, _, C = qkv.shape
        K = self.conv_kernel
        x = qkv[:, 0, :]                              # (B, C)
        # The state holds the last K-1 RAW inputs; the conv window is state + x.
        window = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)   # (B, C, K)
        w = self.conv1d_weight                        # (C, K)
        y = (window * w.unsqueeze(0)).sum(-1)         # (B, C)
        # Shift state left + append new input. MUST copy from the fresh `window`
        # tensor: an in-place conv_state[..., :-1].copy_(conv_state[..., 1:])
        # aliases src/dst and the parallel CUDA copy kernel races (dst[i+1] is
        # src[i+1]), silently corrupting the conv state on every decode step.
        conv_state.copy_(window[..., 1:])
        return y.unsqueeze(1)

    def _gated_norm(self, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Per-head gated RMSNorm (plain w, silu gate), matching HF
        Qwen3_5RMSNormGated / exl3 GatedRMSNorm. o/z: (B, T, nv, dv) or flat
        (rows, nv*dv). Returns same shape/dtype as o.
        """
        dtype = o.dtype
        shape = o.shape
        of = o.reshape(-1, self.dv).float()
        zf = z.reshape(-1, self.dv).float()
        var = of.pow(2).mean(-1, keepdim=True)
        eps = self.pre_layernorm.variance_epsilon if self.pre_layernorm is not None \
            else self.model.config.norm_eps
        on = of * torch.rsqrt(var + eps)
        on = on.to(dtype) * self.norm_weight
        gate = zf.sigmoid() if self.norm_act == "sigmoid" else F.silu(zf)
        out = on * gate
        return out.to(dtype).view(shape)

    def _scan(self, q, k, v, g, beta, state, chunked: bool):
        """q/k/v (B, T, H, D) fp16 -> fp32 scan; state advanced in place."""
        q = _l2norm(q.float())
        k = _l2norm(k.float())
        v = v.float()
        g = g.float()
        beta = beta.float()
        if chunked and q.shape[1] > 1:
            return _chunk_scan(q, k, v, g, beta, state)
        return _recurrent_scan(q, k, v, g, beta, state)

    def _local_state(self, cache, past_len: int):
        """
        Fetch (conv_state, ssm_state) for this forward, handling reset and the
        one-token rewind backup protocol (token healing / MTP rejection).

        Returns (conv_state, ssm_state, backup_conv, backup_ssm) -- the backup
        pair is where the PRE-step state must be stashed before advancing.
        """
        li = self.layer_idx
        conv = cache.gdn_conv_states[li]
        ssm = cache.gdn_ssm_states[li]
        bconv = cache.gdn_conv_backup[li]
        bssm = cache.gdn_ssm_backup[li]

        if past_len == 0:
            conv.zero_()
            ssm.zero_()
            cache.gdn_backup_pos[li] = -1
        elif cache.gdn_backup_pos.get(li, -1) == past_len:
            # Rewound to the snapshot (e.g. token healing popped the last token):
            # restore the pre-step state before re-advancing.
            conv.copy_(bconv)
            ssm.copy_(bssm)
        return conv, ssm, bconv, bssm

    def _stash_backup(self, cache, conv, ssm, bconv, bssm, past_len: int):
        bconv.copy_(conv)
        bssm.copy_(ssm)
        cache.gdn_backup_pos[self.layer_idx] = past_len

    def rewind_state(self, cache, pos: int):
        """
        Restore this layer's recurrent state to the prefix ending at `pos` using
        the staging recorded by the last staged (MTP verify) forward. Replays the
        delta scan over the accepted tokens' staged inputs from the base snapshot.
        """
        stage = cache.gdn_stage.get(self.layer_idx)
        if stage is None:
            return
        n = pos - cache.gdn_stage_start
        assert 0 <= n <= cache.gdn_stage_len, "gdn rewind outside staged window"
        conv = cache.gdn_conv_states[self.layer_idx]
        ssm = cache.gdn_ssm_states[self.layer_idx]
        K = self.conv_kernel
        conv.copy_(stage["hist"][:, :, n:n + K - 1])
        ssm.copy_(stage["base_ssm"])
        if n > 0:
            co = stage["qkv"][:, :n]
            Bn = co.shape[0]
            qc, kc, vc = co.split([self.kd, self.kd, self.vd], dim=-1)
            qh = _l2norm(qc.view(Bn, n, self.nk, self.dk).float()).repeat_interleave(self.group, dim=2)
            kh = _l2norm(kc.view(Bn, n, self.nk, self.dk).float()).repeat_interleave(self.group, dim=2)
            vh = vc.view(Bn, n, self.nv, self.dv).float()
            _recurrent_scan(qh, kh, vh, stage["g"][:, :n], stage["beta"][:, :n], ssm)
        cache.gdn_backup_pos[self.layer_idx] = -1

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict:

        if self.is_tp:
            return self.forward_tp_old(
                hidden_states, cache, attn_params, past_len, intermediates, loras, **kwargs)

        B, T, _ = hidden_states.shape
        residual = hidden_states
        x = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states
        staging = False

        q = self.in_proj_q.forward(x, loras = loras)
        k = self.in_proj_k.forward(x, loras = loras)
        v = self.in_proj_v.forward(x, loras = loras)
        z = self.in_proj_z.forward(x, loras = loras)
        b = self.in_proj_b.forward(x, loras = loras)
        a = self.in_proj_a.forward(x, loras = loras)

        qkv = torch.cat([q, k, v], dim=-1)

        if cache is None:
            # Converter measurement: stateless, zero initial state.
            conv_out, _ = self._conv_prefill(qkv, None)
            state = torch.zeros((B, self.nv, self.dk, self.dv),
                                dtype=torch.float32, device=hidden_states.device)
            past_len = 0
        else:
            if past_len is None:
                past_len = cache.current_seq_len
            conv, ssm, bconv, bssm = self._local_state(cache, past_len)
            staging = getattr(cache, "gdn_staging", False) and T > 1
            split_scan = False
            if staging:
                # MTP verify: record base state + conv history + scan inputs so the
                # accepted prefix can be restored afterwards (gdn_rewind_to).
                stage = cache.gdn_stage.setdefault(self.layer_idx, {})
                stage["base_conv"] = conv.clone()
                stage["base_ssm"] = ssm.clone()
                xh = torch.cat([conv, qkv.transpose(1, 2)], dim=-1)
                stage["hist"] = xh.clone()
                C, K = self.conv_dim, self.conv_kernel
                conv_out = F.conv1d(xh, self.conv1d_weight.view(C, 1, K), groups=C).transpose(1, 2)
                conv.copy_(xh[..., -(K - 1):])
                cache.gdn_stage_start = past_len
                cache.gdn_stage_len = T
            elif T == 1:
                self._stash_backup(cache, conv, ssm, bconv, bssm, past_len)
                conv_out = self._conv_decode(qkv, conv)
            else:
                # Snapshot before the LAST token so a one-token rewind can restore.
                # The conv backup (new_conv) is the state after T-1 tokens. The SSM
                # backup must ALSO be the state after T-1 tokens -- but the scan has
                # not run yet, so `ssm` here is still the pre-scan state. The split
                # scan below scans the prefix first, then fixes bssm to that state.
                conv_out, new_conv = self._conv_prefill(qkv[:, :-1], conv)
                self._stash_backup(cache, new_conv, ssm, bconv, bssm, past_len + T - 1)
                last, new_conv = self._conv_prefill(qkv[:, -1:], new_conv)
                conv_out = torch.cat([conv_out, last], dim=1)
                conv.copy_(new_conv)
                split_scan = True
            state = ssm

        conv_out = F.silu(conv_out)
        qc, kc, vc = conv_out.split([self.kd, self.kd, self.vd], dim=-1)
        qh = qc.view(B, T, self.nk, self.dk)
        kh = kc.view(B, T, self.nk, self.dk)
        vh = vc.view(B, T, self.nv, self.dv)
        # GQA: each key head serves `group` consecutive value heads.
        qh = qh.repeat_interleave(self.group, dim=2)
        kh = kh.repeat_interleave(self.group, dim=2)

        beta, g = self._gates(b, a)
        if staging:
            stage["qkv"] = conv_out.detach().clone()
            stage["beta"] = beta.detach().clone()
            stage["g"] = g.detach().clone()
        if split_scan:
            # Scan the prefix (chunked) to reach the state after T-1 tokens, fix the
            # SSM backup to that state (the early stash captured the pre-scan state),
            # then scan the last token recurrently. Output equals a single chunked
            # scan over all T tokens (prefix property), but the one-token-rewind
            # backup now holds the correct recurrent state.
            o = self._scan(qh[:, :-1], kh[:, :-1], vh[:, :-1], g[:, :-1], beta[:, :-1],
                           state, chunked=True)
            bssm.copy_(state)
            o_last = self._scan(qh[:, -1:], kh[:, -1:], vh[:, -1:], g[:, -1:], beta[:, -1:],
                                state, chunked=False)
            o = torch.cat([o, o_last], dim=1)
        else:
            o = self._scan(qh, kh, vh, g, beta, state, chunked=(cache is None or T > 1))
        o = self._gated_norm(o.view(B, T, self.nv, self.dv), z.view(B, T, self.nv, self.dv))
        o = o.view(B, T, self.vd).to(hidden_states.dtype)
        out = self.out_proj.forward(o, loras = loras)
        if self.has_residual:
            out = out + residual

        if intermediates:
            return {
                "hidden_states": out,
                "post_norm": x,
                "pre_out": o,
            }
        return out

    # -------------------------------------------------------------- TP fwd

    def forward_tp_old(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict:
        """
        TP forward, both modes (mirrors attn/mlp.forward_tp_old):

        - single-process (ctx.rank is None): residual broadcast to every device,
          per-device shards, out_proj partials summed on the primary.
        - MP rank-local (ctx.seqpar): residual is this rank's RS shard (prefill)
          or full width (decode_repl); out_proj partial allreduced /
          reduce-scattered back into the residual stream.

        The delta scan itself is purely local: each rank advances its own
        value-head shard of the recurrent state, no cross-rank state traffic.
        """
        cfg = self.model.config
        ctx = self.model.tp_context
        ctx.set_geometry(self.layer_idx)

        B, T, _ = hidden_states.shape
        rows = B * T
        x = hidden_states.view(rows, self.hidden_size)
        rank_local = ctx.rank is not None

        if rank_local and ctx.seqpar:
            residual = x
            if ctx.decode_repl and T <= 32:
                post_norm = self.pre_layernorm.forward(x) if self.has_norm else x
            else:
                full = ctx.gather_residual(x)
                post_norm = self.pre_layernorm.forward(full) if self.has_norm else full
        else:
            x = ctx.broadcast(0, x, BROADCAST_RS)
            # Rank-local broadcast returns a one-element list; the residual add
            # wants the tensor (single-process keeps the per-device list).
            residual = x[0] if isinstance(x, list) and rank_local else x
            if not self.has_norm:
                post_norm = x
            else:
                post_norm = self.pre_layernorm.forward(x) if not isinstance(x, list) \
                    else self.pre_layernorm.forward_tp(x, output_split = True)

        # Projections. In rank-local mode these are scalar single-device layers;
        # forward_tp degenerates to a local GEMM. In single-process mode they
        # return per-device shard lists.
        q = self.in_proj_q.forward_tp(post_norm, output_split = True, dim = self.dk, loras = loras)
        k = self.in_proj_k.forward_tp(post_norm, output_split = True, dim = self.dk, loras = loras)
        v = self.in_proj_v.forward_tp(post_norm, output_split = True, dim = self.dv, loras = loras)
        z = self.in_proj_z.forward_tp(post_norm, output_split = True, dim = self.dv, loras = loras)
        b = self.in_proj_b.forward(post_norm if not isinstance(post_norm, list) else post_norm[0], loras = loras)
        a = self.in_proj_a.forward(post_norm if not isinstance(post_norm, list) else post_norm[0], loras = loras)

        split_v = ctx.get_split(BROADCAST_GDN_V)
        if not isinstance(q, list):
            q, k, v, z = [q], [k], [v], [z]
            split_v = [(ctx.rank if rank_local else ctx.device, self.tp_v_a, self.tp_v_a + self.tp_nv)]
            b_list = [b]
            a_list = [a]
        else:
            b_list = [b] * len(q)
            a_list = [a] * len(q)

        if past_len is None:
            past_len = 0 if cache is None else cache.current_seq_len

        partials = []
        for idx, (dev, va, vb) in enumerate(split_v):
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)
            nv_l = vb - va
            nk_l = nv_l // self.group
            ka = va // self.group

            qkv = torch.cat([q[idx], k[idx], v[idx]], dim=-1)   # (rows, local_conv_dim)
            qkv = qkv.view(B, T, -1)

            # Per-device state. Rank-local: this rank's cache holds the shard.
            # Single-process: each device's base cache holds full-size state;
            # slice this device's head range below.
            if cache is None:
                conv_state = None
                ssm = torch.zeros((B, nv_l, self.dk, self.dv),
                                  dtype=torch.float32, device=qkv.device)
                bconv = bssm = None
                pl = 0
            else:
                dev_cache = cache
                if not rank_local and hasattr(cache, "caches"):
                    dev_cache = cache.caches[dev]
                conv_full = dev_cache.gdn_conv_states[self.layer_idx]
                ssm_full = dev_cache.gdn_ssm_states[self.layer_idx]
                if rank_local:
                    conv_state = conv_full
                    ssm = ssm_full
                else:
                    # Full-size per-device state: view this device's slice.
                    q_lo, q_hi = ka * self.dk, (ka + nk_l) * self.dk
                    k_lo, k_hi = self.kd + ka * self.dk, self.kd + (ka + nk_l) * self.dk
                    v_lo, v_hi = 2 * self.kd + va * self.dv, 2 * self.kd + vb * self.dv
                    cidx = torch.cat([
                        torch.arange(q_lo, q_hi),
                        torch.arange(k_lo, k_hi),
                        torch.arange(v_lo, v_hi),
                    ]).to(conv_full.device)
                    conv_state = conv_full[:, cidx, :]
                    ssm = ssm_full[:, va:vb]
                bconv = dev_cache.gdn_conv_backup[self.layer_idx]
                bssm = dev_cache.gdn_ssm_backup[self.layer_idx]
                if rank_local:
                    bconv_s, bssm_s = bconv, bssm
                else:
                    bconv_s = bconv[:, cidx, :]
                    bssm_s = bssm[:, va:vb]
                pl = past_len
                if pl == 0:
                    conv_state.zero_()
                    ssm.zero_()
                    dev_cache.gdn_backup_pos[self.layer_idx] = -1
                elif dev_cache.gdn_backup_pos.get(self.layer_idx, -1) == pl:
                    conv_state.copy_(bconv_s)
                    ssm.copy_(bssm_s)

            # Conv (local channels)
            cw = self.conv1d_weight
            if not rank_local:
                cw = self._raw_conv_slice(dev, ka, nk_l, va, nv_l)
            tp_split = False
            if cache is None:
                conv_out, _ = self._conv_prefill_with(qkv, None, cw)
            elif T == 1:
                if bconv_s is not None:
                    bconv_s.copy_(conv_state)
                    bssm_s.copy_(ssm)
                    dev_cache.gdn_backup_pos[self.layer_idx] = pl
                conv_out = self._conv_decode_with(qkv, conv_state, cw)
            else:
                conv_out, new_conv = self._conv_prefill_with(qkv[:, :-1], conv_state, cw)
                if bconv_s is not None:
                    bconv_s.copy_(new_conv)
                    bssm_s.copy_(ssm)
                    dev_cache.gdn_backup_pos[self.layer_idx] = pl + T - 1
                last, new_conv = self._conv_prefill_with(qkv[:, -1:], new_conv, cw)
                conv_out = torch.cat([conv_out, last], dim=1)
                conv_state.copy_(new_conv)
                tp_split = True

            conv_out = F.silu(conv_out)
            kd_l = nk_l * self.dk
            vd_l = nv_l * self.dv
            qc, kc, vc = conv_out.split([kd_l, kd_l, vd_l], dim=-1)
            qh = qc.view(B, T, nk_l, self.dk).repeat_interleave(self.group, dim=2)
            kh = kc.view(B, T, nk_l, self.dk).repeat_interleave(self.group, dim=2)
            vh = vc.view(B, T, nv_l, self.dv)

            if rank_local:
                # b/a are replicated full-width (nv); A_log/dt_bias are sliced to
                # this rank's value heads by _slice_raw_tp -- match the widths.
                va_l = self.tp_v_a
                vb_l = self.tp_v_a + self.tp_nv
                beta, g = self._gates(b_list[idx][..., va_l:vb_l], a_list[idx][..., va_l:vb_l])
                A_log = self.A_log
                dt_bias = self.dt_bias
            else:
                A_log = self.A_log_full[va:vb]
                dt_bias = self.dt_bias_full[va:vb]
                beta = b_list[idx][..., va:vb].sigmoid()
                g = -A_log.exp() * F.softplus(a_list[idx][..., va:vb] + dt_bias)

            # b/a run on the flattened (rows, H) residual; the scan wants (B, T, nv).
            beta = beta.view(B, T, nv_l)
            g = g.view(B, T, nv_l)
            if tp_split:
                # Fix the SSM backup to the state after T-1 tokens (the early stash
                # captured the pre-scan state); see the non-TP forward for rationale.
                o = self._scan(qh[:, :-1], kh[:, :-1], vh[:, :-1], g[:, :-1], beta[:, :-1],
                               ssm, chunked=True)
                if bssm_s is not None:
                    bssm_s.copy_(ssm)
                o_last = self._scan(qh[:, -1:], kh[:, -1:], vh[:, -1:], g[:, -1:], beta[:, -1:],
                                    ssm, chunked=False)
                o = torch.cat([o, o_last], dim=1)
            else:
                o = self._scan(qh, kh, vh, g, beta, ssm, chunked=(cache is None or T > 1))
            nw = self.norm_weight if rank_local else self.norm_weight_full
            o = self._gated_norm_with(o.view(B, T, nv_l, self.dv),
                                      z[idx].view(B, T, nv_l, self.dv), nw)
            o = o.view(rows, vd_l).to(hidden_states.dtype)

            # out_proj row-parallel: local input shard -> full-hidden partial.
            if rank_local:
                partial = self.out_proj.forward(o, loras = loras)
            else:
                partial = self.out_proj.forward_tp_row([o], output_split = True)[0]
            partials.append(partial)

        # Reduce partials into the residual stream. out_proj is row-parallel:
        # each rank's partial covers only its value heads, so the partials must
        # be SUMMED across ranks in every mode (the rank-local loader leaves a
        # plain single-device GEMM with no built-in reduce).
        if rank_local:
            partial = partials[0]
            if ctx.seqpar and not (ctx.decode_repl and T <= 32):
                out_shard = ctx.reduce_scatter_hidden(partial)
                if self.has_residual: out_shard = out_shard + residual
                return out_shard.view(B, T, -1)
            out_full = ctx.all_reduce_hidden(partial, "gdn")
            if self.has_residual: out_full = out_full + residual
            return out_full.view(B, T, self.hidden_size)

        # Single-process: sum partials on the primary device.
        dev0 = ctx.device
        torch.cuda.set_stream(self.model.get_device_context(dev0).stream)
        acc = partials[0].to(dev0)
        for p in partials[1:]:
            acc = acc + p.to(dev0)
        res0 = residual[0] if isinstance(residual, list) else residual
        out = acc + res0 if self.has_residual else acc
        return out.view(B, T, self.hidden_size)

    # Raw-tensor helpers for single-process TP (full tensors kept on primary;
    # per-device slices cached). Rank-local mode slices at load time instead.

    def _raw_conv_slice(self, dev, ka, nk_l, va, nv_l):
        if not hasattr(self, "conv1d_weight_full"):
            self.conv1d_weight_full = self.conv1d_weight
            self.A_log_full = self.A_log
            self.dt_bias_full = self.dt_bias
            self.norm_weight_full = self.norm_weight
        cache = self.__dict__.setdefault("_raw_slice_cache", {})
        key = (dev, ka, nk_l, va, nv_l)
        ent = cache.get(key)
        if ent is None:
            q_lo, q_hi = ka * self.dk, (ka + nk_l) * self.dk
            k_lo, k_hi = self.kd + ka * self.dk, self.kd + (ka + nk_l) * self.dk
            v_lo, v_hi = 2 * self.kd + va * self.dv, 2 * self.kd + (va + nv_l) * self.dv
            idx = torch.cat([
                torch.arange(q_lo, q_hi),
                torch.arange(k_lo, k_hi),
                torch.arange(v_lo, v_hi),
            ])
            ent = self.conv1d_weight_full[idx].contiguous().to(dev)
            cache[key] = ent
        return ent

    def _conv_prefill_with(self, qkv, conv_state, w):
        B, T, C = qkv.shape
        K = self.conv_kernel
        x = qkv.transpose(1, 2)
        if conv_state is not None:
            x = torch.cat([conv_state, x], dim=-1)
        else:
            x = F.pad(x, (K - 1, 0))
        y = F.conv1d(x, w.view(C, 1, K), groups=C)
        new_state = x[..., -(K - 1):].contiguous()
        return y.transpose(1, 2), new_state

    def _conv_decode_with(self, qkv, conv_state, w):
        x = qkv[:, 0, :]
        # State holds the last K-1 raw inputs; the conv window is state + x.
        window = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)
        y = (window * w.unsqueeze(0)).sum(-1)
        # Non-aliasing shift (see _conv_decode): copy from the fresh window.
        conv_state.copy_(window[..., 1:])
        return y.unsqueeze(1)

    def _gated_norm_with(self, o, z, w):
        dtype = o.dtype
        shape = o.shape
        of = o.reshape(-1, self.dv).float()
        zf = z.reshape(-1, self.dv).float()
        var = of.pow(2).mean(-1, keepdim=True)
        eps = self.pre_layernorm.variance_epsilon if self.pre_layernorm is not None \
            else self.model.config.norm_eps
        on = of * torch.rsqrt(var + eps)
        on = on.to(dtype) * w
        gate = zf.sigmoid() if self.norm_act == "sigmoid" else F.silu(zf)
        out = on * gate
        return out.to(dtype).view(shape)
