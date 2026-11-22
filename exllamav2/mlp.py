from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.tensor_p import BROADCAST_ID, BROADCAST_RS
# from line_profiler import profile

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2


class ExLlamaV2MLP(ExLlamaV2Module):

    name: str = "MLP"

    layer_idx: int
    pre_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm | None
    post_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm | None
    gate_proj: ExLlamaV2Linear | None
    up_proj: ExLlamaV2Linear | None
    down_proj: ExLlamaV2Linear | None

    q_handle: int | None

    temp_lora_size: int

    has_norm: bool
    has_residual: bool

    is_tp: bool
    tp_dq_size: list[int] | None
    merge: int | None


    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        layer_idx: int,
        has_norm: bool = True,
        has_residual: bool = True,
        archparams = None,
        in_features: int | None = None,
        out_features: int | None = None,
        interm_features: int | None = None,
        merge: int | None = None,
        pad32: bool = True,
    ):
        super().__init__(model, key, archparams)
        cfg = self.model.config
        ap = self.archparams
        km = self.archparams.keys

        if ap.is_vision:
            self.hidden_size = cfg.vision_hidden_size
            self.intermediate_size = cfg.vision_intermediate_size
        else:
            self.hidden_size = cfg.hidden_size
            self.intermediate_size = cfg.intermediate_size

        if in_features is None: in_features = self.hidden_size
        if out_features is None: out_features = self.hidden_size
        if interm_features is None: interm_features = self.intermediate_size
        self.in_features = in_features
        self.out_features = out_features
        self.interm_features = interm_features
        self.merge = merge

        self.is_tp = False
        self.tp_dq_size = None

        self.layer_idx = layer_idx
        self.has_norm = has_norm
        self.has_residual = has_residual

        self.q_handle = None
        self.temp_lora_size = 0

        f_a = 0
        f_b = interm_features
        f_c = f_b + interm_features
        f_key = (key + ".mlp." + km["fused_mlp_12"]) if km["fused_mlp_12"] else None

        if self.has_norm and (km["norm_2"] or km["norm_2_post"]):
            if ap.norm == "layernorm":
                self.pre_layernorm = ExLlamaV2LayerNorm(model, key + km["norm_2"])
                self.post_layernorm = ExLlamaV2LayerNorm(model, key + km["norm_2_post"]) if km["norm_2_post"] else None
            elif ap.norm == "rmsnorm":
                self.pre_layernorm = ExLlamaV2RMSNorm(model, key + km["norm_2"])
                self.post_layernorm = ExLlamaV2RMSNorm(model, key + km["norm_2_post"]) if km["norm_2_post"] else None
        else:
            self.pre_layernorm = None
            self.post_layernorm = None

        self.up_proj = ExLlamaV2Linear(model, key + km["mlp_up"], in_features, interm_features, ap.mlp_bias, f_key = f_key, f_beg = f_b, f_end = f_c, pad32 = pad32)
        self.down_proj = ExLlamaV2Linear(model, key + km["mlp_down"], interm_features, out_features, ap.mlp_bias, prescale = cfg.scale_depth, pad32 = pad32)

        self.submodules = [self.up_proj,
                           self.down_proj]
        if self.pre_layernorm:
            self.submodules += [self.pre_layernorm]
        if self.post_layernorm:
            self.submodules += [self.post_layernorm]

        if ap.mlp_gate:
            self.gate_proj = ExLlamaV2Linear(model, key + km["mlp_gate"], in_features, interm_features, ap.mlp_bias, f_key = f_key, f_beg = f_a, f_end = f_b, pad32 = pad32)
            self.submodules += [self.gate_proj]
        else:
            self.gate_proj = None

        if merge and ap.mlp_patch_merger:
            self.patch_merger_proj = ExLlamaV2Linear(model, key + km["patch_merger"], in_features * merge**2, in_features, ap.mlp_bias)
            self.submodules += [self.patch_merger_proj]
        else:
            self.patch_merger_proj = None

    def numel(self) -> int:

        numel = self.up_proj.numel() + \
                self.down_proj.numel()

        if self.archparams.mlp_gate:
            numel += self.gate_proj.numel()

        if self.pre_layernorm is not None:
            numel += self.pre_layernorm.numel()
        if self.post_layernorm is not None:
            numel += self.pre_layernorm.numel()

        return numel


    @torch.inference_mode
    def load(
        self,
        device_context: bool = True
    ):
        cfg = self.model.config
        km = self.archparams.keys

        if self.pre_layernorm is not None:
            self.pre_layernorm.load()
        if self.post_layernorm is not None:
            self.post_layernorm.load()

        if cfg.checkpoint_fused_mlp:
            w12 = self.load_weight(self.key + km["fused_mlp_12"])
            w1 = nn.Parameter(w12[:self.interm_features, :].contiguous())
            w2 = nn.Parameter(w12[self.interm_features:, :].contiguous())
            w3 = self.load_weight(self.key + km["fused_mlp_3"])
            self.down_proj.load(w3, device_context = device_context)
            self.gate_proj.load(w1, device_context = device_context)
            self.up_proj.load(w2, device_context = device_context)
        else:
            down_map = self.down_proj.load(device_context = device_context, unmap = True)
            if self.gate_proj is not None: self.gate_proj.load(device_context = device_context, output_map = down_map)
            self.up_proj.load(device_context = device_context, output_map = down_map)

        if self.patch_merger_proj is not None:
            self.patch_merger_proj.load()

        if self.up_proj.is_quant():
            assert self.gate_proj is None or self.gate_proj.is_quant()
            assert self.up_proj.is_quant(), "Partially quantized MLP layer"

            if device_context:
                device_context = self.model.get_device_context(self.device_idx)
                device_context.begin_scratch_alloc()
                temp_state = device_context.get_scratch_slice(self.temp_state_size())
                temp_a = device_context.get_scratch_slice(self.temp_a_size())
                temp_b = device_context.get_scratch_slice(self.temp_b_size())
                temp_dq = device_context.get_scratch_slice(self.temp_dq_size())
            else:
                temp_state = none_tensor
                temp_a = none_tensor
                temp_b = none_tensor
                temp_dq = none_tensor

            if self.has_norm:
                norm_weight = self.pre_layernorm.weight if self.pre_layernorm.weight is not None else none_tensor
                norm_bias = self.pre_layernorm.bias if self.pre_layernorm.bias is not None else none_tensor
                is_rms = isinstance(self.pre_layernorm, ExLlamaV2RMSNorm)
                eps = self.pre_layernorm.variance_epsilon
            else:
                norm_weight = none_tensor
                norm_bias = none_tensor
                is_rms = False
                eps = 0

            if self.post_layernorm is not None:
                post_norm_weight = self.post_layernorm.weight if self.post_layernorm.weight is not None else none_tensor
                post_norm_bias = self.post_layernorm.bias if self.post_layernorm.bias is not None else none_tensor
            else:
                post_norm_weight = none_tensor
                post_norm_bias = none_tensor

            self.q_handle = ext_c.make_q_mlp(
                norm_weight,
                norm_bias,
                is_rms,
                eps,
                0 if self.gate_proj is None else self.gate_proj.q_handle,
                self.up_proj.q_handle,
                self.down_proj.q_handle,
                temp_state,
                temp_a,
                temp_b,
                temp_dq,
                cfg.max_input_len * cfg.max_batch_size,
                self.archparams.mlp_act_func == "gelu",
                self.has_residual,
                post_norm_weight,
                post_norm_bias,
                self.archparams.residual_stream_fp32,
                not cfg.no_graphs
            )


    def unload(self):

        if self.q_handle is not None:
            ext_c.free_q_mlp(self.q_handle)
            self.q_handle = None

        if self.pre_layernorm is not None: self.pre_layernorm.unload()
        if self.post_layernorm is not None: self.post_layernorm.unload()
        if self.gate_proj is not None: self.gate_proj.unload()
        self.up_proj.unload()
        self.down_proj.unload()


    def weight_footprint(self) -> int:

        if self.model.config.checkpoint_fused_mlp:
            fp = 2 * self.in_features * self.interm_features * 2 + \
                self.interm_features * self.out_features * 2
        else:
            fp = self.up_proj.weight_footprint() + \
                 self.down_proj.weight_footprint()
            if self.gate_proj is not None:
                fp += self.gate_proj.weight_footprint()

        if self.pre_layernorm is not None:
            fp += self.pre_layernorm.weight_footprint()
        if self.post_layernorm is not None:
            fp += self.post_layernorm.weight_footprint()

        return fp


    def scratch_space_fixed(self) -> int:

        return self.temp_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_dq_size()


    def scratch_space(self) -> int:

        return (
            self.temp_state_size() +
            self.temp_a_size() +
            self.temp_b_size() +
            self.temp_dq_size()
        )


    def temp_state_size(self) -> int:

        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * self.out_features * 2 + 128


    def temp_a_size(self) -> int:

        temp = max(self.in_features, self.interm_features, self.out_features)
        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * temp * 2 + 128


    def temp_b_size(self) -> int:

        temp = max(self.in_features, self.interm_features, self.out_features)
        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * temp * 2 + 128


    def temp_dq_size(self) -> int:

        return max(0 if self.gate_proj is None else self.gate_proj.temp_dq_size(),
                   self.up_proj.temp_dq_size(),
                   self.down_proj.temp_dq_size())


    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)

        if self.pre_layernorm is not None:
            self.pre_layernorm.set_device_idx(idx)
        if self.post_layernorm is not None:
            self.post_layernorm.set_device_idx(idx)
        if self.gate_proj is not None: self.gate_proj.set_device_idx(idx)
        self.up_proj.set_device_idx(idx)
        self.down_proj.set_device_idx(idx)
        if self.patch_merger_proj is not None:
            self.patch_merger_proj.set_device_idx(idx)


    # @profile
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        if self.is_tp:
            # BUG 34c: the fused TP MLP kernel (ext_c.tp_mlp_forward_) has no post-FF norm
            # parameter at all -- see ext_qmlp.cpp, which accepts pre_layernorm only. Gemma-4
            # folds a post-FF norm into the MLP (post_layernorm), so on the fused path that
            # norm was silently skipped and the residual stream came out un-normalised.
            # forward_tp_old applies it explicitly (gather -> norm full-width -> residual add),
            # which is the same order as the non-TP path, and was verified to reproduce the
            # non-TP fused kernel to cos 1.0000093. Models without a post-FF norm keep the
            # fused kernel.
            if self.post_layernorm is not None:
                return self.forward_tp_old(
                    hidden_states,
                    cache,
                    attn_params,
                    past_len,
                    intermediates,
                    loras,
                    **kwargs
                )
            return self.forward_tp(
                hidden_states,
                cache,
                attn_params,
                past_len,
                intermediates,
                loras,
                **kwargs
            )

        cfg = self.model.config

        if self.q_handle is None or intermediates:
            return self.forward_torch(hidden_states, cache, attn_params, past_len, intermediates, loras = loras, **kwargs)

        if loras is None or self.temp_lora_size == 0:
            pass_loras = []
            pass_lora_temp = none_tensor
        else:
            pass_loras = [id(x) for x in loras]
            pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        # print (hidden_states.storage().data_ptr())
        ext_c.q_mlp_forward_(self.q_handle,
                             hidden_states,
                             pass_loras,
                             pass_lora_temp)

        if self.archparams.clamp_hidden_states:
            hidden_states.clamp_(-65504, 65504)

        return hidden_states


    # @profile
    def forward_tp(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        cfg = self.model.config
        ctx = self.model.tp_context

        batch_size, q_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        ext_c.tp_mlp_forward_(
            self.model.tp_context.ext_tp_context,
            hidden_states,
            self.temp_bc0,
            self.temp_bc1,
            self.temp_bc2,
            self.temp_gate,
            self.temp_up,
            self.temp_down,
            self.pre_layernorm.weight if self.pre_layernorm is not None else [],
            self.pre_layernorm.variance_epsilon if self.pre_layernorm is not None else 0.0,
            self.gate_proj.q_handle if self.gate_proj is not None else [],
            self.up_proj.q_handle,
            self.down_proj.q_handle,
            self.archparams.mlp_act_func == "gelu"
        )

        return ctx.get_pinned(0, batch_size, q_len, self.hidden_size)


    def _post_norm_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ``post_layernorm`` to a full-width tensor.

        The norm is TP-split (``tp_split(BROADCAST_RS)``), so its ``forward()`` only
        accepts per-device slices. Rebuild the full weight on the tensor's own device
        (cached) and apply the RMSNorm math directly. BUG 34c.
        """
        norm = self.post_layernorm
        if norm is None:
            return x
        if not getattr(norm, "is_tp", False):
            return norm.forward(x, output_fp32 = self.archparams.residual_stream_fp32)

        w = getattr(self, "_post_norm_w_full", None)
        if w is None or w.device != x.device:
            # tp_split(BROADCAST_RS) gives every device a FULL-width copy, so the list
            # holds N identical tensors. Take one, do not concat.
            parts = norm.weight if isinstance(norm.weight, list) else [norm.weight]
            w = parts[0].to(x.device)
            self._post_norm_w_full = w

        # BUG 38: fused kernel instead of the pure-torch fp32 round trip. See the
        # matching note in attn.py::_post_norm_full.
        x = x.contiguous()
        out = torch.empty_like(x)
        ext_c.rms_norm(x, w, out, norm.variance_epsilon)
        return out

    def forward_tp_old(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        cfg = self.model.config
        ctx = self.model.tp_context

        batch_size, q_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if ctx.seqpar:
            # Sequence-parallel: incoming residual is this rank's hidden shard. Allgather
            # to full width for the input norm + column-parallel gate/up; keep the shard
            # for the residual add. down_proj is ALREADY column-parallel on the RS split,
            # so its output shard is the next residual shard with no extra collective.
            # decode_repl (q_len <= 32): residual stays FULL-width -- no input gather,
            # full-width post-norm below, no tiny stat all_reduces (see attn.py).
            residual = hidden_states
            if ctx.decode_repl and q_len <= 32:
                post_norm = [self.pre_layernorm.forward(hidden_states)] if self.pre_layernorm else [hidden_states]
            else:
                full = ctx.gather_residual(hidden_states)
                post_norm = [self.pre_layernorm.forward(full)] if self.pre_layernorm else [full]
        else:
            hidden_states = ctx.broadcast(0, hidden_states, BROADCAST_ID)

            residual = hidden_states

            post_norm = self.pre_layernorm.forward_tp(hidden_states, output_split = True) \
                if self.pre_layernorm else hidden_states

        gate = self.gate_proj.forward_tp(post_norm, output_split = True)
        up = self.up_proj.forward_tp(post_norm, output_split = True)

        outputs = []
        for idx, hs in enumerate(post_norm):
            dev = hs.device.index
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)

            if self.archparams.mlp_act_func == "silu":
                output = F.silu(gate[idx])
            elif self.archparams.mlp_act_func == "gelu":
                output = F.gelu(gate[idx], approximate = "tanh")
            output *= up[idx]
            # output.clamp_(min = -65504.0, max = 65504.0)
            outputs.append(output)

        if ctx.seqpar and ctx.decode_repl and q_len <= 32 \
                and getattr(self, "_down_row_repl", False) and not loras:
            # Stage 3 row-parallel down (EXL2_TP_ROW_REPL=1): outputs[0] is this rank's
            # gate/up shard remapped into down's row-shard input order (output_map=down_perm
            # baked at load); the quantized row GEMM runs locally -> full-hidden partial ->
            # one allreduce. Eliminates the intermediate AG + the RS-shard AG (2 -> 1).
            partial = self.down_proj.forward(outputs[0], loras = loras)
            out_full = ctx.all_reduce_hidden(partial, "down")
            if self.post_layernorm is not None:
                out_full = self.post_layernorm.forward(out_full)
            if self.has_residual:
                out_full = out_full + residual
            return out_full.view(batch_size, q_len, out_full.shape[-1])

        if ctx.seqpar and ctx.decode_repl and q_len <= 32:
            # decode_repl: gather the down RS output shard to FULL width, post-norm
            # full-width (no stat all_reduce), add the full residual. Mirrors attn.py.
            outputs = ctx.allgather(1, outputs, BROADCAST_ID, BROADCAST_ID)
            down = self.down_proj.forward_tp(outputs, output_split = True)
            # Fused AG+post-norm+residual (EXL2_TP_FUSE_AGNR=1), mirrors attn.py.
            _fused = None
            if os.environ.get("EXL2_TP_FUSE_AGNR", "0") == "1" \
                    and self.post_layernorm is not None and self.has_residual:
                _fused = ctx.ag_norm_res(down[0], 3, self.post_layernorm.weight,
                                         residual, self.post_layernorm.variance_epsilon)
            if _fused is not None:
                out_full = _fused
            else:
                out_full = ctx.allgather(1, [down[0]], BROADCAST_RS, BROADCAST_RS)
                if self.post_layernorm is not None:
                    out_full = self.post_layernorm.forward(out_full)
                if self.has_residual:
                    out_full = out_full + residual
            return out_full.view(batch_size, q_len, out_full.shape[-1])

        if ctx.seqpar and (getattr(self, "_down_rs_row", False)
                           or getattr(self, "_down_row_repl", False)):
            # Row-parallel down_proj (DOWN_RS / ROW_REPL): this rank's activation shard IS its
            # input slice (output_map remap), so the quantized row GEMM runs locally (no
            # intermediate allgather). Full-hidden partial reduce-scattered to the residual
            # shard. Serves prefill (q_len>32); decode_repl ROW_REPL uses the allreduce branch.
            partial = self.down_proj.forward(outputs[0], loras = loras)
            out_shard = ctx.reduce_scatter_hidden(partial)
            if self.post_layernorm is not None:
                out_shard = self.post_layernorm.forward_tp_sharded(out_shard)
            if self.has_residual:
                out_shard = out_shard + residual
            return out_shard.view(batch_size, q_len, out_shard.shape[-1])

        outputs = self.model.tp_context.allgather(1, outputs, BROADCAST_ID, BROADCAST_ID)

        down = self.down_proj.forward_tp(outputs, output_split = True)

        if ctx.seqpar:
            # Sequence-parallel: down's RS-split output shard is the residual shard.
            # Post-FF norm on the shard (tiny cross-rank RMS stat all_reduce), then a
            # shard-local residual add. No gather, no reduce_scatter.
            out_shard = down[0]
            if self.post_layernorm is not None:
                out_shard = self.post_layernorm.forward_tp_sharded(out_shard)
            if self.has_residual:
                out_shard = out_shard + residual
            return out_shard.view(batch_size, q_len, out_shard.shape[-1])

        # BUG 34c: the post-FF norm was missing here too. forward_tp_old went straight
        # from down_proj to the residual add, while the non-TP path applies
        # post_layernorm to the down_proj output first (mlp.py:503). Dropping it leaves
        # the residual stream un-normalised, exactly as in the attention module.
        #
        # The norm needs the full hidden width for its RMS reduction, so gather first.
        # exllamav3 does the equivalent with a separate full-width `mlp_post_norm` block
        # (transformer.py:224).
        if self.post_layernorm is not None:
            dev0 = self.model.tp_context.device
            context = self.model.get_device_context(dev0)
            torch.cuda.set_stream(context.stream)
            down_full = self.model.tp_context.gather(0, down, BROADCAST_RS)
            down_full = down_full.to(dev0, non_blocking = True)
            if os.environ.get("EXL2_MLP_DUMP") and self.key.endswith("layers.0"):
                torch.save({
                    "down_pre_norm": down_full.detach().float().cpu(),
                    "res": (residual[0] if isinstance(residual, (list, tuple)) else residual).detach().float().cpu(),
                    "w_shape": tuple(self._post_norm_w_full.shape),
                    "x_shape": tuple(down_full.shape),
                }, os.environ["EXL2_MLP_DUMP"] + "_tp_pre.pt")
            down_full = self._post_norm_full(down_full)
            if self.has_residual:
                res = residual[0] if isinstance(residual, (list, tuple)) else residual
                if res.device != down_full.device:
                    res = res.to(down_full.device, non_blocking = True)
                down_full = down_full + res
            if os.environ.get("EXL2_MLP_DUMP") and self.key.endswith("layers.0"):
                torch.save({
                    "out": down_full.detach().float().cpu(),
                }, os.environ["EXL2_MLP_DUMP"] + "_tp.pt")
            return down_full.view(batch_size, q_len, down_full.shape[-1])

        if self.has_residual:
            self.model.tp_context.add_residual(down, residual, BROADCAST_RS)

        down = self.model.tp_context.gather(0, down, BROADCAST_RS)
        down = down.view(batch_size, q_len, down.shape[-1])
        return down


    def forward_torch(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        cfg = self.model.config

        residual = hidden_states
        post_norm = self.pre_layernorm.forward(hidden_states) \
            if self.pre_layernorm else hidden_states

        if self.merge:
            if self.archparams.mlp_patch_merger:
                bsz = hidden_states.shape[0]
                assert bsz == 1
                (h, w), d = kwargs["patch_size"], hidden_states.shape[-1]
                image_grid = post_norm.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
                grid = F.unfold(image_grid, kernel_size = int(self.merge ** 0.5), stride = int(self.merge ** 0.5))
                grid = grid.view(bsz, d * self.merge, -1).transpose(1, 2)
                post_norm = self.patch_merger_proj.forward(grid)
            else:
                bd = post_norm.shape[:-2]
                l, d = post_norm.shape[-2:]
                post_norm = post_norm.view(*bd, l // self.merge, d * self.merge)

        if self.gate_proj is not None:
            gate = self.gate_proj.forward(post_norm, loras = loras)
            if self.archparams.mlp_act_func == "silu":
                y = F.silu(gate)
            elif self.archparams.mlp_act_func == "gelu":
                y = F.gelu(gate, approximate = "tanh")
            up = self.up_proj.forward(post_norm, loras = loras)
            y *= up
            y.clamp_(min = -65504.0, max = 65504.0)
        else:
            up = self.up_proj.forward(post_norm, loras = loras)
            if self.archparams.mlp_act_func == "silu":
                y = F.silu(up)
            elif self.archparams.mlp_act_func == "gelu":
                y = F.gelu(up, approximate = "tanh")
            elif self.archparams.mlp_act_func == "quickgelu":
                y = up * torch.sigmoid(1.702 * up)

        down = self.down_proj.forward(y, loras = loras)
        if self.post_layernorm:
            down = self.post_layernorm.forward(down, output_fp32 = self.archparams.residual_stream_fp32)
        hidden_states = down + residual if self.has_residual else down
        if os.environ.get("EXL2_MLP_DUMP") and self.key.endswith("layers.0"):
            torch.save({
                "post_norm": post_norm.detach().float().cpu(),
                "down": down.detach().float().cpu(),
                "residual": residual.detach().float().cpu(),
                "out": hidden_states.detach().float().cpu(),
                "post_ln_is_none": self.post_layernorm is None,
            }, os.environ["EXL2_MLP_DUMP"] + "_nontp.pt")

        if self.archparams.residual_stream_fp32:
            hidden_states = hidden_states.float()
        elif self.archparams.clamp_hidden_states:
            hidden_states = hidden_states.clamp(-65504, 65504)

        if intermediates:
            return {"post_norm": post_norm,
                    "pre_down": y,
                    "hidden_states": hidden_states}
        else:
            return hidden_states


    def update_loras(self):

        if self.q_handle is None: return

        if self.gate_proj is None:
            gate_proj_lora_a = {}
            gate_proj_lora_b = {}
        else:
            gate_proj_lora_a = { id(k): v for k, v in self.gate_proj.lora_a_tensors.items() }
            gate_proj_lora_b = { id(k): v for k, v in self.gate_proj.lora_b_tensors.items() }

        up_proj_lora_a = { id(k): v for k, v in self.up_proj.lora_a_tensors.items() }
        up_proj_lora_b = { id(k): v for k, v in self.up_proj.lora_b_tensors.items() }
        down_proj_lora_a = { id(k): v for k, v in self.down_proj.lora_a_tensors.items() }
        down_proj_lora_b = { id(k): v for k, v in self.down_proj.lora_b_tensors.items() }

        temp_lora_size = ext_c.q_mlp_set_loras(self.q_handle,
                                               gate_proj_lora_a,
                                               gate_proj_lora_b,
                                               up_proj_lora_a,
                                               up_proj_lora_b,
                                               down_proj_lora_a,
                                               down_proj_lora_b)

        self.temp_lora_size = temp_lora_size * self.model.config.max_batch_size * self.model.config.max_input_len


    def is_quant(self):
        return self.q_handle is not None


    def rank_reduce(self, k):

        if self.gate_proj is not None: self.gate_proj.rank_reduce(k)
        self.up_proj.rank_reduce(k)
        self.down_proj.rank_reduce(k)


    def tp_split(self):

        cfg = self.model.config
        ctx = self.model.tp_context

        if self.pre_layernorm is not None:
            self.pre_layernorm.tp_split(BROADCAST_RS)
        if self.post_layernorm is not None:
            self.post_layernorm.tp_split(BROADCAST_RS)
        if self.gate_proj is not None:
            self.gate_proj.tp_split(BROADCAST_ID)
        if self.up_proj is not None:
            self.up_proj.tp_split(BROADCAST_ID)
        if self.down_proj is not None:
            self.down_proj.tp_split(BROADCAST_RS)

        maxrows = cfg.max_batch_size * cfg.max_input_len
        dtype = torch.half

        ctx.begin_scratch_alloc_tp()
        ctx.reserve_scratch(self.tp_dq_size)
        self.temp_bc0 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_RS)
        self.temp_bc1 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_RS)
        self.temp_bc2 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_ID)
        self.temp_gate = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_ID)
        self.temp_up = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_ID)
        self.temp_down = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_RS)

        self.is_tp = True


    def tp_load_rank(self, rank: int):
        """
        Load only this MLP block's slice for device ``rank``.

        Mirrors :meth:`tp_split`. ``down_proj`` is loaded with ``unmap = True``, exactly as
        ``ExLlamaV2MLP.load()`` does, and the permutation it returns is applied to
        ``gate_proj`` / ``up_proj`` through ``tensor_remap`` -- see BUG 52. Getting this wrong
        permutes the down_proj input twice and puts its GEMM 142% off peak.
        """
        cfg = self.model.config

        if self.pre_layernorm is not None:
            # SP: the input norm runs full-width on the allgathered residual (full weight).
            self.pre_layernorm.tp_load_rank(rank, BROADCAST_RS)
        if self.post_layernorm is not None:
            # FULL weight on every rank (~10 KB): the seqpar shard flow slices it in
            # forward_tp_sharded; the decode_repl flow runs the norm full-width.
            self.post_layernorm.tp_load_rank(rank, BROADCAST_RS)

        # down_proj first: it yields the permutation that gate/up must be remapped with.
        #
        # Seqpar: ROW-parallel on the ID split. gate/up's output shard (rows, inter/world)
        # is exactly this rank's down_proj input slice -- the unmap remap bakes down's
        # q_perm into gate/up's outputs, so the row GEMM needs no permutation (BUG 113
        # does not apply here). forward_tp_old then reduce-scatters the full-hidden
        # partial to the residual shard instead of allgathering the full intermediate
        # (4x the bytes). OPT-IN (EXL2_TP_DOWN_RS=1): default OFF while the row-shard
        # path is under suspicion for the long-context repetition salad (debug5.txt /
        # bugfix.md -- both runs had it ON by default; the confirmed-good f0a27e8 state
        # used column-parallel + allgather). Default OFF restores that verified layout;
        # =1 re-enables the reduce-scatter win for A/B once correctness is re-confirmed.
        self._down_rs_row = False
        # Stage 3 (EXL2_TP_ROW_REPL): row-parallel down via an fp16 LOGICAL-order shard
        # (mirrors the o_proj O_RS path). matmul(gate/up shard in NATURAL column order,
        # down_fp16_shard logical rows [a,b)) -> full-hidden partial -> allreduce (decode_repl)
        # / reduce_scatter (prefill). This REPLACES the quantized row-handle + output_map
        # q_perm remap, which mis-ordered the down input and broke parity from token 0 (the
        # "coherent but never thinks" bug). gate/up load natural (no remap); the down fp16
        # shard is built below. 2 collectives -> 1 per block, same as the o_proj row path.
        self._down_row_repl = False
        self._down_fp16_row_repl = None
        # ROW_REPL down row-parallel defaults OFF: the quantized row-handle + output_map
        # remap path breaks greedy parity (the o_proj fp16 row path is verified EXACT, down
        # is not), and an fp16 logical down shard is VRAM-prohibitive (~3.4 GB/rank). So
        # ROW_REPL fuses the o_proj block only (down stays column-parallel). EXL2_TP_ROW_REPL_DOWN=1
        # re-enables the experimental down row path for debugging.
        if self.model.tp_context.seqpar and os.environ.get("EXL2_TP_ROW_REPL", "0") == "1" \
                and os.environ.get("EXL2_TP_ROW_REPL_DOWN", "0") != "0":
            down_perm = self.down_proj.tp_load_rank_row(rank, BROADCAST_ID, BROADCAST_RS)
            self._down_row_repl = True
        elif self.model.tp_context.seqpar and os.environ.get("EXL2_TP_DOWN_RS", "0") != "0":
            down_perm = self.down_proj.tp_load_rank_row(rank, BROADCAST_ID, BROADCAST_RS)
            self._down_rs_row = True
        else:
            down_perm = self.down_proj.tp_load_rank(rank, BROADCAST_RS, unmap = True)

        if self.gate_proj is not None:
            self.gate_proj.tp_load_rank(rank, BROADCAST_ID, output_map = down_perm)

        self.up_proj.tp_load_rank(rank, BROADCAST_ID, output_map = down_perm)

        maxrows = cfg.max_batch_size * cfg.max_input_len
        dtype = torch.half

        ctx = self.model.tp_context
        ctx.begin_scratch_alloc_tp()
        ctx.reserve_scratch(self.tp_dq_size)
        self.temp_bc0 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_RS)
        self.temp_bc1 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_RS)
        self.temp_bc2 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_ID)
        self.temp_gate = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_ID)
        self.temp_up = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_ID)
        self.temp_down = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_RS)

        self.is_tp = True
        # BUG 72: forward_chunk() reads module.device_idx, which only set_device_idx()
        # creates. Mirror attn.tp_load_rank, which ends the same way.
        self.set_device_idx(None)

    def scratch_space_tp(self):

        cfg = self.model.config
        ctx = self.model.tp_context
        devs = ctx.num_devices
        scratch = [0] * devs

        def add(res: list[int]):
            for i, s in enumerate(res):
                scratch[i] += s

        def amax(res: list[int]):
            for i, s in enumerate(res):
                scratch[i] = max(scratch[i], s)

        amax(self.gate_proj.scratch_space_tp(BROADCAST_ID, 1))
        amax(self.up_proj.scratch_space_tp(BROADCAST_ID, 1))
        amax(self.down_proj.scratch_space_tp(BROADCAST_RS, 1))
        self.tp_dq_size = [s for s in scratch]

        maxrows = cfg.max_batch_size * cfg.max_input_len

        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_RS))
        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_RS))
        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_ID))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_ID))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_ID))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_RS))

        return scratch
