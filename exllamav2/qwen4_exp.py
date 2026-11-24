from __future__ import annotations

"""
Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration) backbone modules.

Architecture (HF modeling_qwen4_exp.py + SGLang qwen4_exp.py cross-checked):
  * Hyper-connections (hc_count 4, lowrank 320): the residual stream is 4x hidden.
    Each block (attn/GDN and MoE) is wrapped by a mix module (grouped RMSNorm (1+w)
    -> silu(down/hc) -> sigmoid(up) per-dim stream weights -> weighted stream MEAN)
    and an inject module (hyper_input + block_out * 2*sigmoid(inject_logits/hc)).
    A global mixer after the last layer collapses the stream back to hidden for
    the (untied) lm_head; there is NO final model norm.
  * 36 gated-delta-net layers (P6 gdn.py, gated-norm activation = sigmoid here)
    + 12 full-attention layers with the Qwen3.5-style output gate, per-head q/k
    RMSNorm (1+w), partial NEOX rope (64 of 256) and a QSA INDEXER: an MQA
    (4 query heads / 1 key head, hd 128) scorer over 4-token key blocks selects
    the top-512 blocks (2048 tokens) plus the always-visible incomplete tail.
  * MoE on every layer: 512 routed experts (top-10, renormalised softmax) stored
    as fused 3D tensors, plus one shared expert gated by sigmoid.
  * PLE at layer index 1: hashed n-gram embeddings (the fork's engram machinery,
    ngram_embedding.py) -> key/value projections, grouped norms, dilated causal
    conv, gated addition into every hyper-stream.

All small tensors (hc weights, indexer, PLE projections, router, shared gate)
stay fp16; the big projections/experts quantize through the ordinary EXL2/EXL2.5
pipeline. NO Triton, NO trellis: the indexer/HC/PLE math is plain torch, the
heavy GEMMs use the existing extension kernels.
"""

import os
import math
import torch
import torch.nn.functional as F
from torch import nn

from exllamav2.module import ExLlamaV2Module
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.moe_mlp import ExLlamaV2MoEMLP
from exllamav2.ngram_embedding import ExLlamaV2NGramEmbedding
from exllamav2.compat import safe_move_tensor
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.tensor_p import BROADCAST_RS, BROADCAST_VC, BROADCAST_Q
from exllamav2.architecture import RopeStyle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2
    from exllamav2.lora import ExLlamaV2Lora


def _grouped_rmsnorm(x: torch.Tensor, weight: torch.Tensor, group: int, eps: float):
    """RMSNorm over the last `group` dims, zero-centered (1+w). x: (..., group*n)."""
    dtype = x.dtype
    shape = x.shape
    n = shape[-1] // group
    xf = x.reshape(-1, n, group).float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    w = (1.0 + weight.float()).view(1, n, group)
    return (xn * w).to(dtype).view(shape)


def _rmsnorm_1pw(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """Plain zero-centered (1+w) RMSNorm over the last dim, fp32 reduction."""
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    return (xn * (1.0 + weight.float())).to(dtype)


# ---------------------------------------------------------------------------
# Hyper-connections
# ---------------------------------------------------------------------------

class ExLlamaV2HyperConnection(ExLlamaV2Module):
    """
    mode "expand":    (B,T,H) -> (B,T,hc*H)          (after the embedding)
    mode "mix":       (B,T,hc*H) -> (B,T,H)          (before a block; stashes the
                                                      hyper input + inject weights)
    mode "inject":    (B,T,H) -> (B,T,hc*H)          (after a block; reads the stash)
    mode "mix_only":  (B,T,hc*H) -> (B,T,H)          (global mixer before the head)

    The mix/inject pair shares a slot on the model keyed by (layer_idx, block,
    device). Eager-only for now: the stash is a plain tensor reference, so CUDA
    graph capture of the hybrid stack is a follow-up (GDN scan is eager too).
    """

    name: str = "HyperConnection"

    def __init__(self, model: ExLlamaV2, key: str, mode: str,
                 layer_idx: int = -1, block: str = "", archparams = None):
        super().__init__(model, key, archparams)
        cfg = model.config
        self.mode = mode
        self.layer_idx = layer_idx
        self.block = block
        self.hc = cfg.hc_count
        self.hidden = cfg.hidden_size
        self.lowrank = cfg.hc_lowrank
        self.eps = cfg.norm_eps

        self.hc_norm_w = None        # (hc*H,)
        self.down_w = None           # (lowrank, hc*H)
        self.up_w = None             # (hc*H, lowrank)
        self.inject_w = None         # (hc, hc*H)  -- mix mode only

        if not hasattr(model, "_hc_slots"):
            model._hc_slots = {}

    def numel(self):
        n = self.hc * self.hidden
        if self.mode == "expand":
            return 0
        return n + self.lowrank * n + n * self.lowrank + \
            (self.hc * n if self.mode == "mix" else 0)

    @torch.inference_mode()
    def load(self, device_context: bool = True):
        if self.mode == "expand":
            return
        dev = self.device()
        w = self.load_weight(self.key + ".hc_norm")
        self.hc_norm_w = w.data.half().to(dev)
        w = self.load_weight(self.key + ".input_mix_weight_down")
        self.down_w = w.data.half().to(dev)
        w = self.load_weight(self.key + ".input_mix_weight_up")
        self.up_w = w.data.half().to(dev)
        if self.mode == "mix":
            w = self.load_weight(self.key + ".block_inject_weight")
            self.inject_w = w.data.half().to(dev)

    def unload(self):
        self.hc_norm_w = None
        self.down_w = None
        self.up_w = None
        self.inject_w = None

    def weight_footprint(self):
        return self.numel() * 2

    def scratch_space_fixed(self):
        return 0

    def scratch_space(self):
        return 0

    def scratch_space_tp(self):
        return [0] * self.model.tp_context.num_devices

    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)
        for t in (self.hc_norm_w, self.down_w, self.up_w, self.inject_w):
            if t is not None:
                safe_move_tensor(t, self.device())

    def is_quant(self):
        return False

    def tp_load_rank(self, rank: int):
        # Replicated on every rank: the hyper-stream is full-width (seqpar is
        # disabled for this architecture), so each rank runs the identical mix.
        self.device_idx = rank
        self.load()
        self.is_tp = True

    def tp_split(self):
        self.load()
        self.is_tp = True

    def _mix(self, x: torch.Tensor):
        # HF Qwen4ExpTextGatedResidual (generated modeling): grouped norm ->
        # silu(down/hc) -> sigmoid(up) gives PER-DIM stream weights; the mixed
        # input is their weighted MEAN over streams. Inject: 2*sigmoid(W@normed/hc).
        B, T, _ = x.shape
        rows = B * T
        normed = _grouped_rmsnorm(x.view(rows, -1), self.hc_norm_w, self.hidden, self.eps)
        flat = normed.view(rows, -1)
        m = F.silu(F.linear(flat, self.down_w).float() / self.hc)
        w = torch.sigmoid(F.linear(m.to(x.dtype), self.up_w).float())
        w = w.view(rows, self.hc, self.hidden).to(x.dtype)
        mixed = (w * normed.view(rows, self.hc, self.hidden)).mean(dim=1)
        inject = None
        if self.mode == "mix":
            inject = 2.0 * torch.sigmoid(F.linear(flat, self.inject_w).float() / self.hc).to(x.dtype)
        return mixed.view(B, T, self.hidden), inject

    def forward(self, hidden_states: torch.Tensor, cache = None, attn_params = None,
                past_len: int | None = None, intermediates: bool = False,
                loras = None, **kwargs):
        # Single-process TP: per-device list, weights replicated on the primary
        # and moved per device on demand (tiny). MP rank-local: plain tensor.
        if isinstance(hidden_states, list):
            return [self.forward(x, cache, attn_params, past_len, intermediates, loras, **kwargs)
                    for x in hidden_states]

        if self.mode == "expand":
            out = hidden_states.repeat(1, 1, self.hc)
            return {"hidden_states": out} if intermediates else out

        dev = hidden_states.device
        if self.hc_norm_w.device != dev:
            self.hc_norm_w = self.hc_norm_w.to(dev)
            self.down_w = self.down_w.to(dev)
            self.up_w = self.up_w.to(dev)
            if self.inject_w is not None:
                self.inject_w = self.inject_w.to(dev)

        if self.mode in ("mix", "mix_only"):
            mixed, inject = self._mix(hidden_states)
            if self.mode == "mix":
                slot_key = (self.layer_idx, self.block, dev.index if dev.index is not None else 0)
                self.model._hc_slots[slot_key] = (hidden_states, inject)
            return {"hidden_states": mixed} if intermediates else mixed

        # inject
        slot_key = (self.layer_idx, self.block, dev.index if dev.index is not None else 0)
        hyper_input, inject = self.model._hc_slots[slot_key]
        B, T, _ = hidden_states.shape
        y = hyper_input.view(B, T, self.hc, self.hidden) + \
            hidden_states.view(B, T, 1, self.hidden) * inject.view(B, T, self.hc, 1)
        out = y.view(B, T, self.hc * self.hidden)
        return {"hidden_states": out} if intermediates else out


# ---------------------------------------------------------------------------
# QSA indexer + attention
# ---------------------------------------------------------------------------

class ExLlamaV2QSAAttention(ExLlamaV2Attention):
    """
    Qwen4-Exp full attention: the P6 output-gated attention (no internal norm or
    residual -- the hyper-connections own those) plus the QSA indexer. When the
    whole visible prefix fits the token budget the selection is ALL tokens and
    the parent's dense path is exact; beyond that, keys/values are gathered per
    query row and attended with SDPA (fp32 softmax).
    """

    name: str = "QSA Attention"

    def __init__(self, model: ExLlamaV2, key: str, layer_idx: int, archparams = None):
        super().__init__(model, key, layer_idx, has_norm = False, has_residual = False,
                         archparams = archparams)
        cfg = model.config
        self.index_n_heads = cfg.indexer_n_heads
        self.index_kv_heads = cfg.indexer_kv_heads
        self.index_head_dim = cfg.indexer_head_dim
        self.index_budget = cfg.indexer_budget
        self.index_compress = cfg.indexer_compress_ratio
        self.index_block_topk = self.index_budget // self.index_compress

        self.index_qk_w = None       # ((nh+kv)*hd, H) fp16
        self.index_q_norm_w = None   # (hd,) fp16
        self.index_k_norm_w = None   # (hd,) fp16

    def numel(self):
        n = super().numel()
        n += (self.index_n_heads + self.index_kv_heads) * self.index_head_dim * self.hidden_size
        n += 2 * self.index_head_dim
        return n

    @torch.inference_mode()
    def load(self, device_context: bool = True):
        super().load(device_context = device_context)
        dev = self.device()
        w = self.load_weight(self.key + ".self_attn.indexer.index_qk_proj")
        self.index_qk_w = w.data.half().to(dev)
        bare = self.load_multi(self.key + ".self_attn.indexer",
                               ["q_layernorm.weight", "k_layernorm.weight"])
        self.index_q_norm_w = bare["q_layernorm.weight"].half().to(dev)
        self.index_k_norm_w = bare["k_layernorm.weight"].half().to(dev)

    def unload(self):
        super().unload()
        self.index_qk_w = None
        self.index_q_norm_w = None
        self.index_k_norm_w = None

    def weight_footprint(self):
        fp = super().weight_footprint()
        fp += 2 * ((self.index_n_heads + self.index_kv_heads) * self.index_head_dim *
                   self.hidden_size + 2 * self.index_head_dim)
        return fp

    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)
        for t in (self.index_qk_w, self.index_q_norm_w, self.index_k_norm_w):
            if t is not None:
                safe_move_tensor(t, self.device())

    def tp_load_rank(self, rank: int):
        super().tp_load_rank(rank)
        dev = self.device()
        w = self.load_weight(self.key + ".self_attn.indexer.index_qk_proj")
        self.index_qk_w = w.data.half().to(dev)
        bare = self.load_multi(self.key + ".self_attn.indexer",
                               ["q_layernorm.weight", "k_layernorm.weight"])
        self.index_q_norm_w = bare["q_layernorm.weight"].half().to(dev)
        self.index_k_norm_w = bare["k_layernorm.weight"].half().to(dev)

    # ------------------------------------------------------------- indexer

    def _indexer_keys(self, x: torch.Tensor, cache, past_len: int):
        """Project + store raw (un-roped) indexer keys; return (q_idx, all_keys).
        q_idx: (B, T, nh, hd) normed+roped; all_keys: (B, kv_len, hd) raw."""
        B, T, _ = x.shape
        cfg = self.model.config
        qk = F.linear(x, self.index_qk_w)
        q, raw_k = torch.split(qk, [self.index_n_heads * self.index_head_dim,
                                    self.index_kv_heads * self.index_head_dim], dim=-1)
        q = q.view(B, T, self.index_n_heads, self.index_head_dim)
        raw_k = raw_k.view(B, T, self.index_head_dim)
        q = _rmsnorm_1pw(q, self.index_q_norm_w, cfg.norm_eps)

        constants = self.model.get_device_context(self.device_idx, scratch = False)
        sin, cos = constants.sin[self.rope_index], constants.cos[self.rope_index]
        # Partial rope on the first rotary_dim dims of the indexer heads.
        ext_c.rope_(q, sin, cos, past_len, self.index_n_heads, self.index_head_dim,
                    none_tensor, self.archparams.rope_style == RopeStyle.NEOX)

        if cache is not None:
            ks = cache.idx_k_states[self.layer_idx]
            ks[:, past_len:past_len + T].copy_(raw_k)
            all_keys = ks[:, :past_len + T]
        else:
            all_keys = raw_k
        return q, all_keys

    def _select(self, q_idx: torch.Tensor, all_keys: torch.Tensor, past_len: int):
        """
        Vectorised block selection. q_idx (B,T,nh,hd) roped; all_keys (B,kv_len,hd)
        raw. Returns (sel_idx (B,T,S) int64, sel_valid (B,T,S) bool) with
        S = budget + compress - 1; invalid slots are clamped to 0 and masked.
        """
        B, T, nh, hd = q_idx.shape
        kv_len = all_keys.shape[1]
        C = self.index_compress
        dev = q_idx.device
        cfg = self.model.config

        nblocks = kv_len // C
        S = self.index_budget + C - 1
        sel_idx = torch.zeros((B, T, S), dtype=torch.long, device=dev)
        sel_valid = torch.zeros((B, T, S), dtype=torch.bool, device=dev)
        qpos = past_len + torch.arange(T, device=dev)      # (T,) absolute query positions
        k = 0
        if nblocks > 0:
            # Pooled block keys: mean of C raw keys -> (1+w) RMSNorm -> rope at block start.
            blocks = all_keys[:, :nblocks * C].view(B, nblocks, C, hd)
            pooled = blocks.float().mean(dim=2).to(all_keys.dtype)
            pooled = _rmsnorm_1pw(pooled, self.index_k_norm_w, cfg.norm_eps)
            constants = self.model.get_device_context(self.device_idx, scratch = False)
            sin, cos = constants.sin[self.rope_index], constants.cos[self.rope_index]
            starts = torch.arange(nblocks, device=dev) * C
            # Gather the rope table at the block-start positions, then rotate at 0.
            sin_s = sin[0, 0].index_select(0, starts)[None, None]
            cos_s = cos[0, 0].index_select(0, starts)[None, None]
            pooled_t = pooled.view(B, nblocks, 1, hd).contiguous()
            ext_c.rope_(pooled_t, sin_s, cos_s, 0, 1, hd, none_tensor,
                        self.archparams.rope_style == RopeStyle.NEOX)

            # Scores: (B, T, nblocks) = sum_heads relu(q . pooled) / sqrt(hd).
            # relu is per-head BEFORE the head sum (HF indexer), so contract hd
            # first, relu, then sum the head axis.
            per_head = torch.einsum("btqh,bkh->btkq", q_idx.float(), pooled_t.view(B, nblocks, hd).float())
            scores = F.relu(per_head).sum(dim=-1) / math.sqrt(hd)

            # A block is fully visible to query t iff its last token <= qpos[t].
            block_end = starts + C - 1                     # (nblocks,)
            visible = block_end.unsqueeze(0) <= qpos.unsqueeze(1)   # (T, nblocks)
            scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))

            k = min(self.index_block_topk, nblocks)
            top = scores.topk(k, dim=-1).indices           # (B, T, k)
            top_valid = scores.gather(-1, top) > float("-inf")

            tok = (top.unsqueeze(-1) * C + torch.arange(C, device=dev)).view(B, T, k * C)
            tok_valid = top_valid.unsqueeze(-1).expand(B, T, k, C).reshape(B, T, k * C)
            sel_idx[:, :, :k * C] = tok
            sel_valid[:, :, :k * C] = tok_valid

        # Incomplete tail: tokens [floor((qpos+1)/C)*C, qpos] per row (<= C-1 slots).
        lo = ((qpos + 1) // C) * C                         # (T,)
        for j in range(C - 1):
            tok = lo + j
            valid = tok <= qpos
            sel_idx[:, :, k * C + j] = tok.clamp_max(kv_len - 1).unsqueeze(0)
            sel_valid[:, :, k * C + j] = valid.unsqueeze(0)
        return sel_idx, sel_valid

    def _sparse_attention(self, q, k_cache, v_cache, sel_idx, sel_valid, scaling):
        """
        Gather-and-attend. q (B,T,nq,hd); k/v_cache (B,kv_len,nkv,hd);
        sel (B,T,S). Returns (B,T,nq,hd). Sub-chunks the query rows to bound the
        gathered K/V footprint.
        """
        B, T, nq, hd = q.shape
        nkv = k_cache.shape[2]
        nkg = nq // nkv
        S = sel_idx.shape[-1]
        out = torch.empty((B, T, nq, hd), dtype=q.dtype, device=q.device)
        # Bound the gathered K/V + score footprint (~256 MB) per sub-chunk.
        sub = max(1, min(T, 268435456 // max(1, S * nkv * hd * 2 * 2 + nq * S * 4)))
        for b in range(B):
            for s in range(0, T, sub):
                e = min(s + sub, T)
                t = e - s
                idx = sel_idx[b, s:e]                      # (t, S)
                valid = sel_valid[b, s:e]
                kg = k_cache[b].index_select(0, idx.view(-1)).view(t, S, nkv, hd)
                vg = v_cache[b].index_select(0, idx.view(-1)).view(t, S, nkv, hd)
                qf = q[b, s:e].float()                     # (t, nq, hd)
                kf = kg.float()
                vf = vg.float()
                if nkg > 1:
                    kf = kf.repeat_interleave(nkg, dim=2)  # (t, S, nq, hd)
                    vf = vf.repeat_interleave(nkg, dim=2)
                aw = torch.einsum("tqh,tsqh->tqs", qf, kf) * scaling
                aw = aw.masked_fill(~valid.unsqueeze(1), float("-inf"))
                aw = torch.softmax(aw, dim=-1)
                ao = torch.einsum("tqs,tsqh->tqh", aw, vf)
                out[b, s:e] = ao.to(q.dtype)
        return out

    # ------------------------------------------------------------- forward

    def _kv_len(self, cache, past_len, T):
        if cache is None:
            return T
        return past_len + T

    def forward(self, hidden_states, cache = None, attn_params = None,
                past_len: int | None = None, intermediates: bool = False,
                loras = None, **kwargs):
        if self.is_tp:
            return self.forward_tp_old(hidden_states, cache, attn_params, past_len,
                                       intermediates, loras, **kwargs)
        return self.forward_torch(hidden_states, cache, attn_params, past_len,
                                  intermediates, loras, **kwargs)

    def forward_torch(self, hidden_states, cache = None, attn_params = None,
                      past_len: int | None = None, intermediates: bool = False,
                      loras = None, **kwargs):
        B, T, _ = hidden_states.shape
        if past_len is None:
            past_len = 0 if cache is None else cache.current_seq_len
        kv_len = self._kv_len(cache, past_len, T)
        if kv_len <= self.index_budget + self.index_compress - 1:
            # Selection would keep every visible token: dense attention is exact.
            return super().forward_torch(hidden_states, cache, attn_params, past_len,
                                         intermediates, loras, **kwargs)
        return self._forward_sparse(hidden_states, cache, attn_params, past_len,
                                    intermediates, loras, **kwargs)

    def _forward_sparse(self, hidden_states, cache, attn_params, past_len,
                        intermediates, loras, **kwargs):
        cfg = self.model.config
        B, T, _ = hidden_states.shape
        x = hidden_states
        q = self.q_proj.forward(x, loras = loras)
        k = self.k_proj.forward(x, loras = loras)
        v = self.v_proj.forward(x, loras = loras)
        qg = q.view(B, T, self.num_attention_heads, 2 * self.head_dim)
        q = qg[..., :self.head_dim].contiguous()
        gate = qg[..., self.head_dim:].contiguous()
        k = k.view(B, T, self.num_key_value_heads, self.head_dim)
        v = v.view(B, T, self.num_key_value_heads, self.head_dim)
        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        constants = self.model.get_device_context(self.device_idx, scratch = False)
        sin, cos = constants.sin[self.rope_index], constants.cos[self.rope_index]
        sc = attn_params.get_alt_rope_embed(self.device_idx) if attn_params is not None else None
        if sc:
            sin, cos = sc
        offsets = attn_params.get_rope_offsets(self.device_idx) if attn_params is not None else None
        pl_rope = past_len if offsets is None else past_len + offsets
        ext_c.rope_(q, sin, cos, pl_rope, self.num_attention_heads, self.head_dim,
                    none_tensor, self.archparams.rope_style == RopeStyle.NEOX)
        ext_c.rope_(k, sin, cos, pl_rope, self.num_key_value_heads, self.head_dim,
                    none_tensor, self.archparams.rope_style == RopeStyle.NEOX)

        if cache is not None:
            cache.key_states[self.layer_idx][:, past_len:past_len + T].copy_(k)
            cache.value_states[self.layer_idx][:, past_len:past_len + T].copy_(v)
            k_all = cache.key_states[self.layer_idx][:, :past_len + T]
            v_all = cache.value_states[self.layer_idx][:, :past_len + T]
        else:
            k_all, v_all = k, v

        q_idx, all_keys = self._indexer_keys(x, cache, past_len)
        sel_idx, sel_valid = self._select(q_idx, all_keys, past_len)
        ao = self._sparse_attention(q, k_all, v_all, sel_idx, sel_valid, self.scaling)
        ao = ao * torch.sigmoid(gate.float()).to(ao.dtype)
        ao = ao.view(B, T, self.num_attention_heads * self.head_dim)
        out = self.o_proj.forward(ao, loras = loras)
        # Row-parallel reduce ONLY when o_proj is actually sharded. In replicated
        # mode (tp_kv_replicate) every rank holds the full o_proj and the local
        # GEMM is already the complete output -- summing would multiply it.
        if self.is_tp and not getattr(self, "_tp_replicated", False):
            out = self.model.tp_context.all_reduce_hidden(out, "qsa")
        if intermediates:
            return {"hidden_states": out, "post_norm": x, "attn_output": ao}
        return out

    # ------------------------------------------------------------- TP fwd

    def forward_tp_old(self, hidden_states, cache = None, attn_params = None,
                       past_len: int | None = None, intermediates: bool = False,
                       loras = None, **kwargs):
        ctx = self.model.tp_context
        # Replicated attention (the qwen4_exp default): the parent's replicated
        # branch gathers/shards the seqpar residual and calls self.forward_torch,
        # which dispatches dense vs sparse on the kv length.
        if getattr(self, "_tp_replicated", False):
            return super().forward_tp_old(hidden_states, cache, attn_params, past_len,
                                          intermediates, loras, **kwargs)
        B, T, _ = hidden_states.shape
        if past_len is None:
            past_len = 0 if cache is None else cache.current_seq_len
        kv_len = self._kv_len(cache, past_len, T)
        rank_local = ctx.rank is not None
        if kv_len <= self.index_budget + self.index_compress - 1:
            return super().forward_tp_old(hidden_states, cache, attn_params, past_len,
                                          intermediates, loras, **kwargs)
        if not rank_local:
            raise NotImplementedError(
                "QSA sparse attention supports MP rank-local TP only (single-process "
                "TP falls back to dense within the budget)")
        return self._forward_sparse_tp(hidden_states, cache, attn_params, past_len,
                                       intermediates, loras, **kwargs)

    def _forward_sparse_tp(self, hidden_states, cache, attn_params, past_len,
                           intermediates, loras, **kwargs):
        ctx = self.model.tp_context
        ctx.set_geometry(self.layer_idx)
        cfg = self.model.config
        B, T, _ = hidden_states.shape
        x = hidden_states
        # seqpar is disabled for qwen4_exp: x is the full mixed hidden on every rank.
        q = self.q_proj.forward_tp(x, output_split = True, dim = self.head_dim * self.q_gate_width, loras = loras)[0]
        k = self.k_proj.forward_tp(x, output_split = True, dim = self.head_dim, loras = loras)[0]
        v = self.v_proj.forward_tp(x, output_split = True, dim = self.head_dim, loras = loras)[0]

        nq_l = q.shape[-1] // (2 * self.head_dim)
        nkv_l = k.shape[-1] // self.head_dim
        qg = q.view(B, T, nq_l, 2 * self.head_dim)
        q = qg[..., :self.head_dim].contiguous()
        gate = qg[..., self.head_dim:].contiguous()
        k = k.view(B, T, nkv_l, self.head_dim)
        v = v.view(B, T, nkv_l, self.head_dim)

        # Head norms are per-head-dim: apply locally with the full weight.
        q = _head_norm_local(q, self.q_norm, cfg)
        k = _head_norm_local(k, self.k_norm, cfg)

        constants = self.model.get_device_context(self.device_idx, scratch = False)
        sin, cos = constants.sin[self.rope_index], constants.cos[self.rope_index]
        ext_c.rope_(q, sin, cos, past_len, nq_l, self.head_dim, none_tensor,
                    self.archparams.rope_style == RopeStyle.NEOX)
        ext_c.rope_(k, sin, cos, past_len, nkv_l, self.head_dim, none_tensor,
                    self.archparams.rope_style == RopeStyle.NEOX)

        cache.key_states[self.layer_idx][:, past_len:past_len + T].copy_(k)
        cache.value_states[self.layer_idx][:, past_len:past_len + T].copy_(v)
        k_all = cache.key_states[self.layer_idx][:, :past_len + T]
        v_all = cache.value_states[self.layer_idx][:, :past_len + T]

        # Indexer is replicated: identical selection on every rank.
        q_idx, all_keys = self._indexer_keys(x, cache, past_len)
        sel_idx, sel_valid = self._select(q_idx, all_keys, past_len)
        ao = self._sparse_attention(q, k_all, v_all, sel_idx, sel_valid, self.scaling)
        ao = ao * torch.sigmoid(gate.float()).to(ao.dtype)
        ao = ao.view(B * T, nq_l * self.head_dim)
        partial = self.o_proj.forward(ao, loras = loras)
        out = ctx.all_reduce_hidden(partial, "qsa")
        out = out.view(B, T, self.hidden_size)
        if intermediates:
            return {"hidden_states": out, "post_norm": x, "attn_output": ao}
        return out


def _head_norm_local(x: torch.Tensor, head_norm, cfg):
    """Per-head RMSNorm on a rank-local head shard, fp32 reduction. The head-norm
    weight already carries the (1+w) constant bias (baked at load time)."""
    if head_norm is None:
        return x
    dtype = x.dtype
    shape = x.shape
    hd = shape[-1]
    xf = x.reshape(-1, hd).float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + cfg.norm_eps)
    w = head_norm.weight.data.float().view(-1, hd)[0]
    return (xn * w).to(dtype).view(shape)


# ---------------------------------------------------------------------------
# MoE with shared expert (Qwen4-Exp)
# ---------------------------------------------------------------------------

class ExLlamaV2Qwen4MoE(ExLlamaV2MoEMLP):
    """
    Flash-Next MoE: parent's fused-3D expert slicing + offload + TP machinery,
    without internal norm/residual (hyper-connections own those), plus a shared
    expert gated by sigmoid (Qwen3NextSparseMoeBlock).
    """

    name: str = "Qwen4 MoE"

    def __init__(self, model: ExLlamaV2, key: str, layer_idx: int, archparams = None):
        super().__init__(model, key, layer_idx, archparams = archparams,
                         has_norm = False, has_residual = False,
                         fused_gate_up_key = key + ".mlp.experts.gate_up_proj",
                         fused_down_key = key + ".mlp.experts.down_proj")
        cfg = model.config
        self.shared_inter = cfg.shared_expert_intermediate_size
        h = cfg.hidden_size
        si = self.shared_inter
        self.shared_w1 = ExLlamaV2Linear(model, key + ".mlp.shared_expert.gate_proj", h, si, False, archparams = archparams)
        self.shared_w2 = ExLlamaV2Linear(model, key + ".mlp.shared_expert.down_proj", si, h, False, archparams = archparams)
        self.shared_w3 = ExLlamaV2Linear(model, key + ".mlp.shared_expert.up_proj", h, si, False, archparams = archparams)
        self.shared_gate = ExLlamaV2Linear(model, key + ".mlp.shared_expert_gate", h, 1, False, pad32 = False)
        self.submodules += [self.shared_w1, self.shared_w2, self.shared_w3, self.shared_gate]

    def numel(self):
        return super().numel() + sum(m.numel() for m in
            (self.shared_w1, self.shared_w2, self.shared_w3, self.shared_gate))

    @torch.inference_mode()
    def load(self):
        super().load()
        self.shared_w1.load()
        self.shared_w2.load()
        self.shared_w3.load()
        self.shared_gate.load()

    def unload(self):
        super().unload()
        self.shared_w1.unload()
        self.shared_w2.unload()
        self.shared_w3.unload()
        self.shared_gate.unload()

    def weight_footprint(self):
        return super().weight_footprint() + sum(m.weight_footprint() for m in
            (self.shared_w1, self.shared_w2, self.shared_w3, self.shared_gate))

    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)
        for m in (self.shared_w1, self.shared_w2, self.shared_w3, self.shared_gate):
            m.set_device_idx(idx)

    def temp_dq_size(self):
        return max(super().temp_dq_size(), self.shared_w1.temp_dq_size())

    def scratch_space(self):
        return max(super().scratch_space(), self.temp_dq_size())

    def scratch_space_fixed(self):
        return self.scratch_space()

    def tp_load_rank(self, rank: int):
        super().tp_load_rank(rank)
        if self._tp_shard:
            from exllamav2.tensor_p import BROADCAST_ID
            perm = self.shared_w2.tp_load_rank(rank, BROADCAST_RS, unmap = True)
            self.shared_w1.tp_load_rank(rank, BROADCAST_ID, output_map = perm)
            self.shared_w3.tp_load_rank(rank, BROADCAST_ID, output_map = perm)
            self.shared_gate.set_device_idx(rank)
            self.shared_gate.load()
        else:
            for m in (self.shared_w1, self.shared_w2, self.shared_w3, self.shared_gate):
                m.set_device_idx(rank)
                m.load()

    def _shared(self, x: torch.Tensor, loras = None):
        g = self.shared_w1.forward(x, loras = loras)
        u = self.shared_w3.forward(x, loras = loras)
        d = self.shared_w2.forward(F.silu(g) * u, loras = loras)
        gate = torch.sigmoid(self.shared_gate.forward(x, loras = loras).float()).to(x.dtype)
        return d * gate

    def _route(self, x: torch.Tensor, loras = None):
        logits = self.gate.forward(x, loras = loras)
        rw = F.softmax(logits, dim=-1, dtype=torch.float)
        rw, sel = torch.topk(rw, self.num_experts_per_token, dim=-1)
        rw = rw / rw.sum(dim=-1, keepdim=True)
        return rw.to(x.dtype), sel

    def forward_torch(self, hidden_states, cache = None, attn_params = None,
                      past_len = None, intermediates = False, loras = None, **kwargs):
        B, T, H = hidden_states.shape
        x = hidden_states.view(-1, H)
        rw, sel = self._route(x, loras)
        out = torch.zeros_like(x)
        expert_mask = F.one_hot(sel, num_classes = self.num_experts).permute(2, 1, 0)
        first_off = self.num_experts - self.offload_experts
        result = {"post_norm": x} if intermediates else None
        for e in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[e])
            if top_x.shape[0] == 0:
                continue
            if self.offload_experts and e >= first_off:
                self.w1[e]._stage_offload()
                self.w3[e]._stage_offload()
                self.w2[e]._stage_offload()
            cur = x[top_x]
            g = self.w1[e].forward(cur, loras = loras)
            u = self.w3[e].forward(cur, loras = loras)
            act = F.silu(g) * u
            if intermediates:
                result[f"pre_down.{e}"] = act
            d = self.w2[e].forward(act, loras = loras)
            d = d * rw[top_x, idx, None]
            out.index_add_(0, top_x, d.to(out.dtype))
        out = out + self._shared(x, loras)
        out = out.view(B, T, H)
        if intermediates:
            result["hidden_states"] = out
            return result
        return out

    def forward(self, hidden_states, cache = None, attn_params = None,
                past_len = None, intermediates = False, loras = None, **kwargs):
        if self.is_tp:
            return self.forward_tp_old(hidden_states, cache, attn_params, past_len,
                                       intermediates, loras, **kwargs)
        return self.forward_torch(hidden_states, cache, attn_params, past_len,
                                  intermediates, loras, **kwargs)

    def forward_tp_old(self, hidden_states, cache = None, attn_params = None,
                       past_len = None, intermediates = False, loras = None, **kwargs):
        ctx = self.model.tp_context
        B, T, H = hidden_states.shape
        repl_chunk = ctx.decode_repl and T <= 32
        if ctx.seqpar and not repl_chunk:
            full = ctx.gather_residual(hidden_states)
        else:
            full = hidden_states

        if self._tp_shard:
            out = self._forward_tp_sharded_q4(full, ctx, loras)
            if ctx.seqpar and not repl_chunk:
                return ctx.shard_residual(out)
            return out

        # Replicated: identical full MoE on every rank (seqpar off for qwen4_exp).
        out = self.forward_torch(full, cache, attn_params, past_len, False, loras, **kwargs)
        if ctx.seqpar and not repl_chunk:
            out = ctx.shard_residual(out)
        return out

    def _forward_tp_sharded_q4(self, full, ctx, loras):
        from exllamav2.tensor_p import BROADCAST_ID
        B, T, H = full.shape
        x = full.view(-1, H)
        rw, sel = self._route(x, loras)
        dev, a, b = ctx.get_split(BROADCAST_RS)[0]
        hidden_local = b - a
        out = torch.zeros((x.shape[0], hidden_local), dtype=x.dtype, device=x.device)
        expert_mask = F.one_hot(sel, num_classes = self.num_experts).permute(2, 1, 0)
        first_off = self.num_experts - self.offload_experts
        for e in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[e])
            if top_x.shape[0] == 0:
                continue
            if self.offload_experts and e >= first_off:
                self.w1[e]._stage_offload()
                self.w3[e]._stage_offload()
                self.w2[e]._stage_offload()
            cur = x[top_x]
            g = self.w1[e].forward_tp([cur], output_split = True, skip_broadcast = True, loras = loras)[0]
            u = self.w3[e].forward_tp([cur], output_split = True, skip_broadcast = True, loras = loras)[0]
            d = self.w2[e].forward_tp([F.silu(g) * u], output_split = True, skip_broadcast = True, loras = loras)[0]
            d = d * rw[top_x, idx, None]
            out.index_add_(0, top_x, d.to(out.dtype))
        # Shared expert sharded the same way.
        g = self.shared_w1.forward_tp([x], output_split = True, skip_broadcast = True, loras = loras)[0]
        u = self.shared_w3.forward_tp([x], output_split = True, skip_broadcast = True, loras = loras)[0]
        d = self.shared_w2.forward_tp([F.silu(g) * u], output_split = True, skip_broadcast = True, loras = loras)[0]
        gate = torch.sigmoid(self.shared_gate.forward(x, loras = loras).float()).to(x.dtype)
        out = out + d * gate
        return ctx.allgather(1, [out], BROADCAST_RS, BROADCAST_RS)[0].view(B, T, H)


# ---------------------------------------------------------------------------
# PLE (per-layer n-gram embeddings)
# ---------------------------------------------------------------------------

class ExLlamaV2PLELayer(ExLlamaV2Module):
    """
    Qwen4-Exp PLE at layer index 1 (ple_layer_ids [2] = idx+1). HF
    Qwen4ExpTextPLELayer (generated modeling):

      emb   = ngram(ids)                                  (ple_dim)
      key   = norm_key(key_proj(emb))                     (hc*H, grouped by H)
      value = value_proj(emb)                             (H)
      query = norm_query(hidden)                          (hc*H, grouped by H)
      gate  = sum(key*query, per stream)/sqrt(H) -> signed-sqrt -> sigmoid
      gv    = gate * value (broadcast over streams)       (hc*H)
      out   = gv + silu(dilated_depthwise_conv(norm_conv(gv)))
      hidden += out

    The n-gram table is the fork's engram module (disk/ram/hybrid host storage,
    never VRAM). The dilated conv state lives on the hybrid cache; verify
    staging + rewind mirror the GDN protocol.
    """

    name: str = "PLE Layer"

    def __init__(self, model: ExLlamaV2, key: str, layer_idx: int, archparams = None):
        super().__init__(model, key, archparams)
        cfg = model.config
        self.layer_idx = layer_idx
        self.hidden = cfg.hidden_size
        self.hc = cfg.hc_count
        self.hc_hidden = cfg.hidden_size * cfg.hc_count
        self.ple_dim = cfg.ple_embed_dim
        self.ngram_size = cfg.ngram_size
        self.heads_per_ngram = cfg.heads_per_ngram
        self.num_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.head_dim = self.ple_dim // self.num_heads
        self.conv_kernel = cfg.ple_conv_kernel_size
        self.dilation = self.ngram_size
        self.state_len = (self.conv_kernel - 1) * self.dilation
        self.eps = cfg.norm_eps

        self.ngram = ExLlamaV2NGramEmbedding(
            model, key + ".ple_embedding.ngram_embedding",
            ngram_size = self.ngram_size,
            heads_per_ngram = self.heads_per_ngram,
            ple_embed_dim = self.ple_dim,
            eos_token_id = cfg.eos_token_id if isinstance(cfg.eos_token_id, int) else cfg.eos_token_id[0],
            head_dim = self.head_dim,
            archparams = archparams,
        )
        self.key_proj_w = None       # (hc*H, ple_dim)
        self.value_proj_w = None     # (H, ple_dim)
        self.norm_key_w = None       # (hc*H,)
        self.norm_query_w = None     # (hc*H,)
        self.norm_conv_w = None      # (hc*H,)
        self.conv1d_w = None         # (hc*H, 1, k)

        self.submodules = [self.ngram]

    def numel(self):
        return self.ngram.numel() + self.hc_hidden * self.ple_dim + \
            self.hidden * self.ple_dim + 3 * self.hc_hidden + \
            self.hc_hidden * self.conv_kernel

    @torch.inference_mode()
    def load(self, device_context: bool = True):
        self.ngram.load()
        self.load_raw()

    @torch.inference_mode()
    def load_raw(self):
        """Projections/norms/conv only -- skips the n-gram table (compile stage)."""
        dev = self.device()
        for attr, suffix in (
            ("key_proj_w", ".key_proj"),
            ("value_proj_w", ".value_proj"),
            ("norm_key_w", ".norm_key"),
            ("norm_query_w", ".norm_query"),
            ("norm_conv_w", ".norm_conv"),
            ("conv1d_w", ".conv1d"),
        ):
            w = self.load_weight(self.key + suffix)
            setattr(self, attr, w.data.half().to(dev))

    def unload(self):
        self.ngram.unload()
        self.key_proj_w = None
        self.value_proj_w = None
        self.norm_key_w = None
        self.norm_query_w = None
        self.norm_conv_w = None
        self.conv1d_w = None

    def weight_footprint(self):
        # Projections/norms/conv fp16 on device; the ngram table is host/disk.
        return self.ngram.weight_footprint() + 2 * (
            self.hc_hidden * self.ple_dim + self.hidden * self.ple_dim +
            3 * self.hc_hidden + self.hc_hidden * self.conv_kernel)

    def scratch_space_fixed(self):
        return 0

    def scratch_space(self):
        return 0

    def scratch_space_tp(self):
        return [0] * self.model.tp_context.num_devices

    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)
        self.ngram.set_device_idx(idx)
        for t in (self.key_proj_w, self.value_proj_w, self.norm_key_w,
                  self.norm_query_w, self.norm_conv_w, self.conv1d_w):
            if t is not None:
                safe_move_tensor(t, self.device())

    def is_quant(self):
        return False

    def tp_load_rank(self, rank: int):
        self.device_idx = rank
        self.ngram.set_device_idx(rank)
        self.load()
        self.is_tp = True

    def tp_split(self):
        self.ngram.set_device_idx(self.model.tp_context.device)
        self.load()
        self.is_tp = True

    # ------------------------------------------------------------- conv state

    def _stash_backup(self, cache, conv, gvn, ids, past_len, T):
        """Snapshot the state BEFORE the last token (GDN backup protocol): the
        conv state is a sliding window over the conv inputs."""
        hist_c = torch.cat([conv, gvn[:, :-1].transpose(1, 2)], dim=-1) if T > 1 else conv
        cache.ple_backup_conv.copy_(hist_c[..., -self.state_len:])
        prev = torch.cat([cache.ple_prev2, ids[:, :-1]], dim=1) if T > 1 else cache.ple_prev2
        cache.ple_backup_prev2 = prev[:, -2:].clone()
        cache.ple_backup_pos = past_len + T - 1

    def rewind_state(self, cache, pos: int):
        """Restore the conv state to the prefix ending at `pos` from the staged
        verify inputs (mirrors gdn.rewind_state)."""
        stage = getattr(cache, "ple_stage", None)
        if stage is None:
            return
        n = pos - stage["start"]
        assert 0 <= n <= stage["gvn"].shape[1], "ple rewind outside staged window"
        hist_c = torch.cat([stage["base_conv"], stage["gvn"][:, :n].transpose(1, 2)], dim=-1)
        cache.ple_conv_state.copy_(hist_c[..., -self.state_len:])
        prev = torch.cat([stage["base_prev2"], stage["ids"][:, :n]], dim=1)
        cache.ple_prev2 = prev[:, -2:].clone()
        cache.ple_backup_pos = -1

    # ------------------------------------------------------------- forward

    def _embeddings(self, cache, ids, dev, past_len):
        """N-gram embeddings with the committed 2-token context; handles the
        fresh-sequence reset and the one-token backup restore."""
        B, T = ids.shape
        if cache is None:
            prev2 = torch.full((B, 2), self.ngram.eos_token_id, dtype=torch.long)
            hist = torch.cat([prev2, ids], dim=1)
            return self.ngram.forward(hist).to(dev), prev2
        if past_len == 0:
            cache.ple_conv_state.zero_()
            cache.ple_prev2 = torch.full((B, 2), self.ngram.eos_token_id, dtype=torch.long)
            cache.ple_backup_pos = -1
        elif cache.ple_backup_pos == past_len:
            cache.ple_conv_state.copy_(cache.ple_backup_conv)
            cache.ple_prev2 = cache.ple_backup_prev2.clone()
        hist = torch.cat([cache.ple_prev2, ids], dim=1)
        return self.ngram.forward(hist).to(dev), cache.ple_prev2

    def forward(self, hidden_states: torch.Tensor, cache = None, attn_params = None,
                past_len: int | None = None, intermediates: bool = False,
                loras = None, **kwargs):
        if isinstance(hidden_states, list):
            return [self.forward(x, cache, attn_params, past_len, intermediates, loras, **kwargs)
                    for x in hidden_states]
        B, T, _ = hidden_states.shape
        dev = hidden_states.device
        if self.key_proj_w.device != dev:
            self.key_proj_w = self.key_proj_w.to(dev)
            self.value_proj_w = self.value_proj_w.to(dev)
            self.norm_key_w = self.norm_key_w.to(dev)
            self.norm_query_w = self.norm_query_w.to(dev)
            self.norm_conv_w = self.norm_conv_w.to(dev)
            self.conv1d_w = self.conv1d_w.to(dev)

        ids = getattr(cache, "_ple_ids", None) if cache is not None else None
        if ids is None:
            # Converter / no-cache reference: PLE needs the original ids; the
            # measure/quantize passes skip this module (identity), so this branch
            # only serves direct callers that stash ids on a dummy cache.
            raise RuntimeError("PLE requires cache._ple_ids (set by forward_chunk)")
        ids = ids.to("cpu", torch.long)
        emb, prev2 = self._embeddings(cache, ids, dev, past_len or 0)

        key = F.linear(emb.view(B * T, -1), self.key_proj_w)
        key = _grouped_rmsnorm(key, self.norm_key_w, self.hidden, self.eps).view(B, T, self.hc, self.hidden)
        value = F.linear(emb.view(B * T, -1), self.value_proj_w).view(B, T, 1, self.hidden)
        query = _grouped_rmsnorm(hidden_states.view(B * T, -1), self.norm_query_w,
                                 self.hidden, self.eps).view(B, T, self.hc, self.hidden)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden)
        gate = gate.float().abs().clamp_min(1e-6).sqrt() * gate.float().sign()
        gv = (torch.sigmoid(gate).to(hidden_states.dtype) * value).flatten(-2)   # (B,T,4H)
        gvn = _grouped_rmsnorm(gv.view(B * T, -1), self.norm_conv_w,
                               self.hidden, self.eps).view(B, T, self.hc_hidden)

        cw = self.conv1d_w.view(self.hc_hidden, 1, self.conv_kernel)
        if cache is not None:
            conv_state = cache.ple_conv_state
            staging = getattr(cache, "gdn_staging", False) and T > 1
            if staging:
                cache.ple_stage = {
                    "base_conv": conv_state.clone(),
                    "base_prev2": prev2.clone(),
                    "start": past_len,
                    "ids": ids.clone(),
                    "gvn": gvn.detach().clone(),
                }
            cs = torch.cat([conv_state, gvn.transpose(1, 2)], dim=-1)
            conv_out = F.conv1d(cs, cw, groups = self.hc_hidden,
                                dilation = self.dilation).transpose(1, 2)
            conv_state.copy_(cs[..., -self.state_len:])
            self._stash_backup(cache, conv_state, gvn, ids, past_len, T)
            cache.ple_prev2 = torch.cat([prev2, ids], dim=1)[:, -2:].clone()
        else:
            cpad = F.pad(gvn.transpose(1, 2), (self.state_len, 0))
            conv_out = F.conv1d(cpad, cw, groups = self.hc_hidden,
                                dilation = self.dilation).transpose(1, 2)

        out = hidden_states + gv + F.silu(conv_out.float()).to(hidden_states.dtype)
        if intermediates:
            return {"hidden_states": out}
        return out
