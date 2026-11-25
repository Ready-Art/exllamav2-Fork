from __future__ import annotations

"""
Gemma-4-26B-A4B MoE block (enable_moe_block).

Every decoder layer runs a dense MLP and a routed-expert branch IN PARALLEL on
the post-attention residual, then sums them (HF modeling_gemma4.py
Gemma4TextDecoderLayer):

    residual = x
    dense  = post_ffn_1( mlp( pre_ffn(residual) ) )
    logits = router_proj( rmsnorm_nw(residual) * router.scale * H**-0.5 )
    w, i   = topk(softmax(logits, fp32), top_k_experts)
    w      = w / w.sum(-1) * per_expert_scale[i]
    moe    = post_ffn_2( experts( pre_ffn_2(residual), i, w ) )   # gelu-tanh
    x      = post_ffn(dense + moe) + residual
    (the existing ExLlamaV2LayerScalar module applies layer_scalar afterwards)

Experts are fused 3D checkpoint tensors (gate_up_proj [E, 2I, H], down_proj
[E, H, I]) sliced per expert through ExLlamaV2Linear f_key, so the ordinary
EXL2/EXL2.5 quantizer, the pinned-RAM offload and the TP shard all apply.
NO Triton / NO trellis.
"""

import os
import torch
import torch.nn.functional as F
from torch import nn

from exllamav2.module import ExLlamaV2Module
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.tensor_p import BROADCAST_RS, BROADCAST_ID
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2


class ExLlamaV2Gemma4MoEBlock(ExLlamaV2Module):

    name: str = "Gemma4 MoE Block"

    def __init__(self, model: ExLlamaV2, key: str, layer_idx: int, archparams = None):
        super().__init__(model, key, archparams)
        cfg = model.config
        ap = self.archparams
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.intermediate_size = cfg.moe_intermediate_size

        # Dense branch: pre_ffn -> MLP -> post_ffn_1, no residual add (the block
        # owns the final residual). The MLP's default post norm key is swapped
        # for the branch-specific post_feedforward_layernorm_1.
        self.dense = ExLlamaV2MLP(model, key, layer_idx, has_residual = False)
        old_post = self.dense.post_layernorm
        self.dense.post_layernorm = ExLlamaV2RMSNorm(model, key + ".post_feedforward_layernorm_1")
        self.dense.submodules = [m for m in self.dense.submodules if m is not old_post]
        self.dense.submodules.append(self.dense.post_layernorm)

        self.pre_ffn_2 = ExLlamaV2RMSNorm(model, key + ".pre_feedforward_layernorm_2")
        self.post_ffn_2 = ExLlamaV2RMSNorm(model, key + ".post_feedforward_layernorm_2")
        self.post_ffn = ExLlamaV2RMSNorm(model, key + ".post_feedforward_layernorm")

        # Router: unweighted RMSNorm (no checkpoint tensor) x scale x H^-0.5.
        self.router_scale = None        # (H,) fp16
        self.per_expert_scale = None    # (E,) fp16
        self.router_proj = ExLlamaV2Linear(model, key + ".router.proj",
                                           self.hidden_size, self.num_experts, False, pad32 = False)

        # Routed experts, fused 3D slices.
        gu_key = key + ".experts.gate_up_proj"
        dn_key = key + ".experts.down_proj"
        I = self.intermediate_size
        H = self.hidden_size
        self.w1, self.w2, self.w3 = [], [], []
        for e in range(self.num_experts):
            self.w1.append(ExLlamaV2Linear(model, key + f".experts.{e}.gate_proj", H, I, False,
                                           f_key = gu_key, f_beg = e * 2 * I, f_end = e * 2 * I + I))
            self.w3.append(ExLlamaV2Linear(model, key + f".experts.{e}.up_proj", H, I, False,
                                           f_key = gu_key, f_beg = e * 2 * I + I, f_end = (e + 1) * 2 * I))
            self.w2.append(ExLlamaV2Linear(model, key + f".experts.{e}.down_proj", I, H, False,
                                           f_key = dn_key, f_beg = e * H, f_end = (e + 1) * H))

        self.submodules = [self.dense, self.pre_ffn_2, self.post_ffn_2, self.post_ffn,
                           self.router_proj] + self.w1 + self.w2 + self.w3
        # Expose the dense branch's linears/norms in modules_dict (one-level
        # registration in model.py) so the converter's optimize pass can find
        # them by key. numel/footprint are computed explicitly, no double count.
        self.submodules += self.dense.submodules

        self.is_tp = False
        self._tp_shard = False
        self.offload_experts = int(os.environ.get("EXL2_MOE_OFFLOAD", "0"))
        if self.offload_experts:
            first_off = self.num_experts - self.offload_experts
            shared = [{}, {}, {}]
            for e in range(self.num_experts):
                is_off = e >= first_off
                for wi, wl in enumerate((self.w1, self.w2, self.w3)):
                    wl[e].offload = is_off
                    if is_off:
                        wl[e]._offload_shared = shared[wi]

    def numel(self):
        return self.dense.numel() + sum(l.numel() for l in self.w1 + self.w2 + self.w3) + \
            self.router_proj.numel() + self.hidden_size + self.num_experts

    @torch.inference_mode()
    def load(self):
        self.dense.load()
        self.pre_ffn_2.load()
        self.post_ffn_2.load()
        self.post_ffn.load()
        self.router_proj.load()
        dev = self.device()
        bare = self.load_multi(self.key + ".router", ["scale", "per_expert_scale"])
        self.router_scale = bare["scale"].half().to(dev)
        self.per_expert_scale = bare["per_expert_scale"].half().to(dev)
        for e in range(self.num_experts):
            self.w1[e].load()
            self.w2[e].load()
            self.w3[e].load()

    def unload(self):
        self.dense.unload()
        self.pre_ffn_2.unload()
        self.post_ffn_2.unload()
        self.post_ffn.unload()
        self.router_proj.unload()
        self.router_scale = None
        self.per_expert_scale = None
        for e in range(self.num_experts):
            self.w1[e].unload()
            self.w2[e].unload()
            self.w3[e].unload()

    def weight_footprint(self):
        fp = self.dense.weight_footprint() + self.router_proj.weight_footprint()
        fp += 2 * (self.hidden_size + self.num_experts)
        off = self.offload_experts
        n_gpu = self.num_experts - off
        fp += n_gpu * (self.w1[0].weight_footprint() + self.w2[0].weight_footprint() +
                       self.w3[0].weight_footprint())
        return fp

    def scratch_space_fixed(self):
        return self.scratch_space()

    def scratch_space(self):
        s = self.dense.scratch_space()
        if self.w1:
            s = max(s, self.w1[0].temp_dq_size(), self.w2[0].temp_dq_size(), self.w3[0].temp_dq_size())
        return s

    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)
        for m in self.submodules:
            m.set_device_idx(idx)
        if self.router_scale is not None:
            self.router_scale = safe_move_tensor(self.router_scale, self.device())
            self.per_expert_scale = safe_move_tensor(self.per_expert_scale, self.device())

    def is_quant(self):
        return len(self.w1) > 0 and self.w1[0].is_quant()

    def temp_dq_size(self):
        return max(self.w1[0].temp_dq_size(), self.w2[0].temp_dq_size(), self.w3[0].temp_dq_size())

    # ------------------------------------------------------------------ TP

    def scratch_space_tp(self):
        devs = self.model.tp_context.num_devices
        s = self.scratch_space()
        self.tp_dq_size = [s] * devs
        return [s] * devs

    def tp_load_rank(self, rank: int):
        self.device_idx = rank
        self._tp_shard = os.environ.get("EXL2_MOE_TP_SHARD", "0") != "0"
        self.dense.set_device_idx(rank)
        self.dense.load()
        for n in (self.pre_ffn_2, self.post_ffn_2, self.post_ffn):
            n.set_device_idx(rank)
            n.load()
        self.router_proj.set_device_idx(rank)
        self.router_proj.load()
        dev = self.device()
        bare = self.load_multi(self.key + ".router", ["scale", "per_expert_scale"])
        self.router_scale = bare["scale"].half().to(dev)
        self.per_expert_scale = bare["per_expert_scale"].half().to(dev)

        off = self.offload_experts
        first_off = self.num_experts - off
        if self._tp_shard:
            for e in range(self.num_experts):
                is_off = bool(off and e >= first_off)
                for wl in (self.w1, self.w2, self.w3):
                    wl[e].offload = is_off
                if is_off:
                    shared = [{}, {}, {}]
                    for wi, wl in enumerate((self.w1, self.w2, self.w3)):
                        wl[e]._offload_shared = shared[wi]
                perm = self.w2[e].tp_load_rank(rank, BROADCAST_RS, unmap = True)
                self.w1[e].tp_load_rank(rank, BROADCAST_ID, output_map = perm)
                self.w3[e].tp_load_rank(rank, BROADCAST_ID, output_map = perm)
        else:
            shared = [{}, {}, {}] if off else None
            for e in range(self.num_experts):
                is_off = bool(off and e >= first_off)
                for wi, wl in enumerate((self.w1, self.w2, self.w3)):
                    wl[e].offload = is_off
                    wl[e].set_device_idx(rank)
                    if is_off:
                        wl[e]._offload_shared = shared[wi]
            for e in range(self.num_experts):
                self.w1[e].load()
                self.w2[e].load()
                self.w3[e].load()
        self.is_tp = True

    # ------------------------------------------------------------------ fwd

    def _route(self, residual: torch.Tensor):
        x = residual.view(-1, self.hidden_size)
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        hn = xf * torch.rsqrt(var + self.model.config.norm_eps)
        hn = hn * self.router_scale.float() * (self.hidden_size ** -0.5)
        logits = self.router_proj.forward(hn.to(self.router_scale.dtype))
        probs = F.softmax(logits.float(), dim=-1)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True)
        topw = topw * self.per_expert_scale[topi].float()
        return topw.to(x.dtype), topi

    def _experts(self, h2: torch.Tensor, topi: torch.Tensor, topw: torch.Tensor,
                 shard_ctx = None, result = None, fp32_act = False):
        x = h2.view(-1, self.hidden_size)
        if shard_ctx is not None:
            dev, a, b = shard_ctx.get_split(BROADCAST_RS)[0]
            out = torch.zeros((x.shape[0], b - a), dtype=x.dtype, device=x.device)
        else:
            out = torch.zeros_like(x)
        expert_mask = F.one_hot(topi, num_classes=self.num_experts).permute(2, 1, 0)
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
            if shard_ctx is not None:
                g = self.w1[e].forward_tp([cur], output_split=True, skip_broadcast=True)[0]
                u = self.w3[e].forward_tp([cur], output_split=True, skip_broadcast=True)[0]
                act = F.gelu(g, approximate="tanh") * u
                d = self.w2[e].forward_tp([act], output_split=True, skip_broadcast=True)[0]
            elif fp32_act:
                # Measurement (intermediates) path: the gelu(gate)*up product and the
                # down projection overflow fp16 (max 65504) on real checkpoints even
                # though the bf16 source and the normed residual are finite. Compute the
                # expert in fp32 so pre_down carries the TRUE activation for the Hessian
                # (no inf/clamp distortion); the down output is cast back to fp16 for the
                # residual stream. Runtime/TP keep the fp16 path below unchanged.
                cur32 = cur.float()
                g = F.linear(cur32, self.w1[e].linear.weight.float())
                u = F.linear(cur32, self.w3[e].linear.weight.float())
                act = F.gelu(g, approximate="tanh") * u
                d = F.linear(act, self.w2[e].linear.weight.float())
            else:
                g = self.w1[e].forward(cur)
                u = self.w3[e].forward(cur)
                act = F.gelu(g, approximate="tanh") * u
                d = self.w2[e].forward(act)
            if result is not None:
                result[f"pre_down.{e}"] = act
            d = d * topw[top_x, idx, None]
            out.index_add_(0, top_x, d.to(out.dtype))
        return out

    def forward_torch(self, hidden_states, cache=None, attn_params=None, past_len=None,
                      intermediates=False, loras=None, **kwargs):
        B, T, H = hidden_states.shape
        residual = hidden_states
        dense_out = self.dense.forward(hidden_states, cache, attn_params, past_len, False, loras, **kwargs)
        topw, topi = self._route(residual)
        h2 = self.pre_ffn_2.forward(residual)
        result = {"post_norm": h2} if intermediates else None
        moe_out = self._experts(h2, topi, topw, result = result, fp32_act = intermediates)
        moe_out = self.post_ffn_2.forward(moe_out.view(B, T, H))
        out = self.post_ffn.forward(dense_out + moe_out)
        out = out + residual
        if intermediates:
            result["hidden_states"] = out
            return result
        return out

    def forward(self, hidden_states, cache=None, attn_params=None, past_len=None,
                intermediates=False, loras=None, **kwargs):
        if self.is_tp:
            return self.forward_tp_old(hidden_states, cache, attn_params, past_len,
                                       intermediates, loras, **kwargs)
        return self.forward_torch(hidden_states, cache, attn_params, past_len,
                                  intermediates, loras, **kwargs)

    def forward_tp_old(self, hidden_states, cache=None, attn_params=None, past_len=None,
                       intermediates=False, loras=None, **kwargs):
        ctx = self.model.tp_context
        B, T, H = hidden_states.shape
        repl_chunk = ctx.decode_repl and T <= 32
        if ctx.seqpar and not repl_chunk:
            full = ctx.gather_residual(hidden_states)
        else:
            full = hidden_states

        residual = full
        dense_out = self.dense.forward(full, cache, attn_params, past_len, False, loras, **kwargs)
        topw, topi = self._route(residual)
        h2 = self.pre_ffn_2.forward(residual)
        if self._tp_shard:
            moe_out = self._experts(h2, topi, topw, shard_ctx=ctx)
            moe_out = ctx.allgather(1, [moe_out], BROADCAST_RS, BROADCAST_RS)[0].view(B, T, -1)
        else:
            # full is the gathered (full-width) residual under seqpar; H from the
            # incoming shard would be wrong here.
            moe_out = self._experts(h2, topi, topw).view(B, T, -1)
        moe_out = self.post_ffn_2.forward(moe_out)
        out = self.post_ffn.forward(dense_out + moe_out)
        out = out + residual
        if intermediates:
            return {"hidden_states": out, "post_norm": h2}
        if ctx.seqpar and not repl_chunk:
            out = ctx.shard_residual(out)
        return out
