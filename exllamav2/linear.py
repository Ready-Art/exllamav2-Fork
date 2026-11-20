from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.module import ExLlamaV2Module
from exllamav2.compat import safe_move_tensor
from exllamav2.tensor_p import BROADCAST_VC
from exllamav2.util import unpack_4bit, pack_4bit
import gc
from exllamav2.experimental.fpx import fpxify

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exllamav2.lora import ExLlamaV2Lora
    from exllamav2.model import ExLlamaV2


class ExLlamaV2Linear(ExLlamaV2Module):

    name: str = "Linear"

    in_features: int
    in_features_tp: list[int] | None
    out_features: int
    out_features_tp: list[int] | None
    has_bias: bool
    prescale: float

    linear: nn.Linear | None
    q_handle: int | list[int | None] | None
    q_tensors: dict | list[dict | None] | None
    q4_weight: torch.Tensor | None
    q4_scales: torch.Tensor | None
    fp16_bias: torch.Tensor | None

    temp_dq: torch.Tensor | list[torch.Tensor]
    padding: int
    max_out_len: int

    lora_a_tensors: dict
    lora_b_tensors: dict

    f_key: str | None
    f_beg: int | None
    f_end: int | None

    is_tp: bool
    broadcast_type: int | None
    broadcast_type_out: int | None
    is_sub_module: bool

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        in_features: int,
        out_features: int,
        has_bias: bool,
        pad32: bool = True,
        max_out_len: int | None = None,
        prescale: float = 1,
        f_key: str = None,
        f_beg: int = None,
        f_end: int = None,
        is_sub_module: bool = True,
        altpack_qkv: bool = False,
        normalize_unq: bool = False,
        archparams = None,
        maybe_transpose: bool = False
    ):
        super().__init__(model, key, archparams)

        self.is_sub_module = is_sub_module

        self.is_tp = False
        self.broadcast_type = None
        self.broadcast_type_out = None

        if pad32:
            self.padding = -out_features % 32
        else:
            self.padding = 0

        self.in_features = in_features
        self.out_features = out_features + self.padding
        self.has_bias = has_bias
        self.temp_dq = None
        self.footprint = -1
        self.max_out_len = max_out_len
        self.prescale = prescale
        self.prev_prescale = None

        self.linear = None
        self.q_handle = None
        self.q_tensors = None
        self.q4_weight = None
        self.q4_scales = None
        self.fp16_bias = None

        self.lora_a_tensors = {}
        self.lora_b_tensors = {}

        self.f_key = f_key
        self.f_beg = f_beg
        self.f_end = f_end
        self.altpack_qkv = altpack_qkv

        self.assumed_footprint = in_features * (out_features + self.padding) * 2 + 128
        self.normalize_unq = normalize_unq

        self.out_features_tp = None

        self.maybe_transpose = maybe_transpose

    @torch.inference_mode
    def load(
        self,
        w: dict | nn.Parameter | tuple | None = None,
        device_context: bool = True,
        unmap: bool = False,
        output_map: torch.Tensor | None = None
    ):
        cfg = self.model.config

        if self.f_key: w = self.load_weight_fused(self.f_key, self.f_beg, self.f_end, self.in_features, self.out_features, self.altpack_qkv)
        if w is None: w = self.load_weight(cpu = output_map is not None)

        # Load quantized linear layer from dictionary

        if isinstance(w, dict):
            assert not cfg.load_in_q4, "Can't load quantized layer in Q4 mode"
            if self.has_bias:
                assert "bias" in w, self.key + " has no bias but bias expected"
            else:
                assert "bias" not in w, self.key + " has bias but bias is not expected"
            if device_context:
                device_context = self.model.get_device_context(self.device_idx)
                device_context.begin_scratch_alloc()
                self.temp_dq = device_context.get_scratch_slice(self.temp_dq_size())
            else:
                self.temp_dq = none_tensor
            self.q_tensors = w

            if unmap and "q_perm" in w:
                perm = w["q_perm"].cpu()
                del w["q_perm"]
                del w["q_invperm"]
                # w["q_perm"] = torch.arange(0, w["q_perm"].shape[-1], dtype = w["q_perm"].dtype, device = w["q_perm"].device)
                # w["q_invperm"] = w["q_perm"].clone()
            else:
                perm = None

            if output_map is not None:
                ext_c.tensor_remap(w["q_weight"], output_map)
                ext_c.tensor_remap_4bit(w["q_scale"], output_map)
                for k in w.keys():
                    w[k] = safe_move_tensor(w[k], self.device())

            self.q_handle = ext.make_q_matrix(w,
                                              self.temp_dq,
                                              prescale = self.prescale,
                                              max_dq_rows = cfg.max_dq_size // self.out_features,
                                              offset_qzeros = cfg.checkpoint_offset_qzeros)
            self.prev_prescale = self.prescale
            self.prescale = 1

            if unmap:
                return perm

        # Load FP16 linear layer without bias, optionally quantize to Q4

        elif isinstance(w, nn.Parameter):
            if self.maybe_transpose and \
                w.data.shape[0] == self.in_features and w.data.shape[1] == self.out_features and \
                self.in_features != self.out_features:
                w = nn.Parameter(w.data.T)
            assert not self.has_bias, self.key + " has no bias tensor but bias is expected"
            # w = nn.Parameter(fpxify(w.data, 2, 3), requires_grad = False)
            if self.normalize_unq:
                w = self.normalize(w)
            if self.padding > 0: w = nn.Parameter(F.pad(w.data, (0, 0, 0, self.padding)).contiguous())
            if not self.model.config.load_in_q4 or not ".layers." in self.key:
                self.linear = nn.Linear(self.in_features, self.out_features, self.has_bias, device = "meta", dtype = torch.float16)
                self.linear.weight = w
            else:
                self.q4_weight = torch.empty((self.out_features * self.in_features // 2,), device = self.device(), dtype = torch.uint8)
                self.q4_scales = torch.empty((self.out_features * self.in_features // 32,), device = self.device(), dtype = torch.half)
                ext_c.matrix_fp16_to_q4(w.contiguous(), self.q4_weight, self.q4_scales)

        # Load FP16 linear layer with bias, optionally quantize to Q4

        elif isinstance(w, tuple):
            assert self.has_bias, self.key + " has bias tensor but bias is not expected"
            if self.normalize_unq:
                w = self.normalize(w[0]), w[1]
            ww = w[0]
            # ww = nn.Parameter(fpxify(ww.data, 2, 3), requires_grad = False)
            wb = w[1]
            if self.padding > 0:
                ww = nn.Parameter(F.pad(ww.data, (0, 0, 0, self.padding)).contiguous())
                wb = nn.Parameter(F.pad(wb.data, (0, 0, 0, self.padding)).contiguous())
            if not self.model.config.load_in_q4 or not ".layers." in self.key:
                self.linear = nn.Linear(self.in_features, self.out_features, self.has_bias, device = "meta", dtype = torch.float16)
                self.linear.weight = ww
                self.linear.bias = wb
            else:
                self.q4_weight = torch.empty((self.out_features * self.in_features // 2,), device = self.device(), dtype = torch.uint8)
                self.q4_scales = torch.empty((self.out_features * self.in_features // 32,), device = self.device(), dtype = torch.half)
                ext_c.matrix_fp16_to_q4(ww.contiguous(), self.q4_weight, self.q4_scales)
                self.fp16_bias = wb


    def normalize(self, w: torch.Tensor):
        return nn.functional.normalize(w)


    def matrix_shape(self):

        return self.in_features, self.out_features


    def numel(self) -> int:

        return self.in_features * self.out_features


    def unload(self):

        if self.linear is not None:
            del self.linear
            self.linear = None

        if self.q_handle is not None:
            for h in (self.q_handle if isinstance(self.q_handle, list) else [self.q_handle]):
                ext_c.free_q_matrix(h)
            self.q_handle = None

        if self.q_tensors is not None:
            for t in (self.q_tensors if isinstance(self.q_tensors, list) else [self.q_tensors]):
                for k, v in t.items():
                    del v
            self.q_tensors = None

        if self.q4_weight is not None:
            del self.q4_weight
            self.q4_weight = None

        if self.q4_scales is not None:
            del self.q4_scales
            self.q4_scales = None

        if self.fp16_bias is not None:
            del self.fp16_bias
            self.fp16_bias = None

        self.temp_dq = None
        if self.prev_prescale is not None:
            self.prescale = self.prev_prescale
            self.prev_prescale = None


    def get_weight(self) -> torch.Tensor:

        return self.linear.weight.data


    def scratch_space_fixed(self) -> int:

        return self.temp_dq_size() + \
               (self.temp_fwd_size() if self.is_sub_module else 0)


    def scratch_space(self) -> int:

        return self.temp_dq_size() + \
               (self.temp_fwd_size() if self.is_sub_module else 0)


    def temp_dq_size(self, in_features = None, out_features = None) -> int:

        dq = (self.in_features if in_features is None else in_features) * \
             (self.out_features if out_features is None else out_features)
        dq = 2 * min(dq, self.model.config.max_dq_size)
        return dq


    def scratch_space_tp(self, broadcast_type: int = BROADCAST_VC, dim = 1):

        ctx = self.model.tp_context
        split = ctx.get_split(broadcast_type)
        self.out_features_tp = [0] * ctx.num_devices
        for dev, a, b in split:
            self.out_features_tp[dev] = (b - a) * dim

        scratch = [self.in_features * of for of in self.out_features_tp]
        scratch = [2 * min(s, self.model.config.max_dq_size) for s in scratch]
        return scratch


    def temp_fwd_size(self) -> int:

        max_len = self.model.config.max_input_len if self.max_out_len is None else \
            min(self.max_out_len, self.model.config.max_input_len)
        return self.out_features * max_len * self.model.config.max_batch_size * 4 + 128


    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        force_recons: bool = False,
        force_cuda: bool = False,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        if self.is_tp:
            if self.out_features_tp:
                return self.forward_tp(
                    hidden_states,
                    cache,
                    attn_params,
                    past_len,
                    intermediates,
                    loras,
                    force_recons,
                    force_cuda,
                    **kwargs
                )
            elif self.in_features_tp:
                return self.forward_tp_row(
                    hidden_states,
                    cache,
                    attn_params,
                    past_len,
                    intermediates,
                    loras,
                    force_recons,
                    force_cuda,
                    **kwargs
                )
            else:
                assert False, "Unitialized TP linear layer"

        # Linear forward

        if self.key == 'lm_head' and loras is not None and loras[0].lm_head is not None:
            hidden_states_out = loras[0].lm_head(hidden_states)

            if intermediates:
                return {"hidden_states": hidden_states_out}
            else:
                return hidden_states_out

        if self.q_handle is not None and not force_recons:

            output_shape = hidden_states.shape[:-1] + (self.out_features,)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            output = torch.empty((hidden_states.shape[0], self.out_features), dtype = torch.half, device = self.device())
            ext_c.gemm_half_q_half(hidden_states, self.q_handle, output, force_cuda)

            hidden_states_out = output.view(output_shape)

        else:

            matrix = self.get_weight_tensor_dq()
            hidden_states_out = torch.matmul(hidden_states, matrix)
            if self.has_bias:
                bias = self.get_bias_tensor()
                hidden_states_out += bias

            if self.prescale != 1:
                hidden_states_out.mul_(self.prescale)

            if getattr(self, "_tp_vocab_parallel", False):
                # Vocab-parallel head: allgather the logits shards into the full vocab.
                # ~0.5 MB per token at fp16 -- negligible next to the per-layer collectives.
                # Symmetric on every rank and graph-capturable (same _gather_nccl contract
                # as the Q/ID/residual gathers the captured decode already uses).
                _ctx = self.model.tp_context
                _shape = hidden_states_out.shape
                _full = _ctx._gather_nccl(hidden_states_out.view(-1, _shape[-1]), BROADCAST_VC, 1, tag = 3)
                hidden_states_out = _full.view(_shape[:-1] + (_full.shape[-1],))

        # Evaluate LoRAs

        if loras is not None:
            for lora in loras:
                lora_a = self.lora_a_tensors.get(lora)
                lora_b = self.lora_b_tensors.get(lora)
                if lora_a is not None:
                    assert lora_b is not None
                    temp = torch.matmul(hidden_states, lora_a)
                    hidden_states_out += torch.matmul(temp, lora_b)

        if intermediates:
            return {"hidden_states": hidden_states_out}
        else:
            return hidden_states_out


    def forward_tp(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        force_recons: bool = False,
        force_cuda: bool = False,
        output_split: bool = False,
        dim: int = 1,
        skip_broadcast: bool = False,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        ctx = self.model.tp_context
        split = ctx.get_split(self.broadcast_type)

        if isinstance(hidden_states, torch.Tensor) and not skip_broadcast:
            output_shape = hidden_states.shape[:-1] + (self.out_features,)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            # TPContext.broadcast() takes (buffer, input_tensor, broadcast_type, dim); the
            # two-argument form here was a stale signature that raised TypeError on the very
            # first TP linear, i.e. TP was broken for every architecture, not just Gemma-4.
            # Stage M bug 26.
            hidden_states = ctx.broadcast(0, hidden_states, self.broadcast_type)
        elif isinstance(hidden_states, torch.Tensor):
            # skip_broadcast: the caller guarantees this tensor is already full-width and
            # replicated on every device (forward_tp_old passes a post-norm derived from a
            # tensor that was itself just broadcast). Just flatten it; no collective.
            output_shape = hidden_states.shape[:-1] + (self.out_features,)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        else:
            output_shape = hidden_states[0].shape[:-1] + (self.out_features,)
            hidden_states = [hs.view(-1, hs.shape[-1]) for hs in hidden_states]

        # BUG 41: reuse the output buffers instead of allocating 4 tensors per
        # projection per token. See tp_reuse_buffers in tensor_p.py.
        from exllamav2.tensor_p import tp_reuse_buffers
        _cache = self.__dict__.get("_tp_out_cache")
        if _cache is None:
            _cache = {}
            self._tp_out_cache = _cache
        outputs = tp_reuse_buffers(
            _cache,
            [(hs.device, hs.shape[0], (b - a) * dim, hs.dtype)
             for hs, (dev, a, b) in zip(hidden_states, split)]
        )

        # BUG 66: C++ takes Sequence[int]; a rank-local worker has a scalar handle.
        _handles = self.q_handle if isinstance(self.q_handle, (list, tuple)) else [self.q_handle]
        ext_c.gemm_half_q_half_tp(
            hidden_states,
            _handles,
            outputs,
            force_cuda,
            ctx.ext_tp_context,
            -1
        )

        if output_split:
            return outputs

        output = ctx.gather(0, outputs, self.broadcast_type)
        hidden_states_out = output.view(output_shape)
        return hidden_states_out


    def forward_tp_row(
        self,
        hidden_states: list[torch.Tensor],
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        force_recons: bool = False,
        force_cuda: bool = False,
        output_split: bool = False,
        dim: int = 1,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        ctx = self.model.tp_context
        split = ctx.get_split(self.broadcast_type)

        assert isinstance(hidden_states, list)
        assert output_split

        rows = hidden_states[0].shape[0]
        dtype = hidden_states[0].dtype
        outputs = ctx.get_temp_tensors_bc(rows, dtype, self.broadcast_type_out)

        ext_c.gemm_half_q_half_tp(
            hidden_states,
            self.q_handle,
            outputs,
            force_cuda,
            ctx.ext_tp_context,
            -1
        )

        return outputs


    def get_weight_tensor_dq(self) -> torch.Tensor:

        if self.linear is not None:
            return self.linear.weight.data.T

        elif self.q_handle is not None:
            tensor = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.device())
            ext_c.reconstruct(self.q_handle, tensor)
            return tensor
            # ext_c.reconstruct(self.q_handle, self.temp_dq)
            # return self.temp_dq

        elif self.q4_weight is not None:
            tensor = torch.empty((self.out_features, self.in_features), dtype = torch.half, device = self.device())
            ext_c.matrix_q4_to_fp16(self.q4_weight, self.q4_scales, tensor)
            return tensor.T

        else:
            raise ValueError(f"Layer {self.key} has no data")


    def get_bias_tensor(self) -> torch.Tensor:

        if self.linear is not None:
            return self.linear.bias.data

        elif self.q_handle is not None:
            return self.q_tensors["bias"]

        elif self.fp16_bias is not None:
            return self.fp16_bias

        else:
            raise ValueError(f"Layer {self.key} has no data")


    def is_quant(self) -> bool:

        return self.q_handle is not None


    def rank_reduce(self, k: float):

        assert not self.is_quant(), "Can't rank-reduce quantized layer"

        weight = self.linear.weight.data.float()
        max_rank = min(weight.shape[0], weight.shape[1])
        desired_rank = int(max_rank * k)
        results = torch.svd_lowrank(weight, q = desired_rank, niter = 10)
        weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

        self.linear.weight = nn.Parameter(weight_approx.half())


    def tp_split(self, broadcast_type: int, dim = None):
        if self.q_handle is None:
            # Unquantized linear (e.g. Gemma-4's hb16 lm_head). Nothing to shard: keep the whole
            # tensor on the primary device and run the ordinary single-device GEMM.
            #
            # `is_tp` must stay False. forward() dispatches on `is_tp` and then asserts that one
            # of the split lists is populated, so setting it True here with both lists empty sent
            # lm_head into "Unitialized TP linear layer" on the first decode step (prefill never
            # touches lm_head). Same reasoning as tp_replicate(). Stage L bug 23.
            self.broadcast_type = broadcast_type
            self.out_features_tp = []
            self.in_features_tp = []
            self.is_tp = False
            # Pin to the primary device: load_tp() calls set_device_idx(None) after tp_split(),
            # and without this the layer would lose its device and fall back to CPU.
            # NOTE: this branch runs before `ctx = self.model.tp_context` further down, so the
            # context has to be resolved here rather than reused.
            _dev = self.model.tp_context.device
            self.device_idx = _dev
            self._tp_replicated_dev = _dev
            return

        assert all(x in self.q_tensors for x in [
            "q_scale",
            "q_scale_max",
            "q_group_map",
            "q_groups",
            "q_weight"
        ]), "Can only split fully loaded EXL2 tensor."

        cfg = self.model.config
        ctx = self.model.tp_context
        self.broadcast_type = broadcast_type
        split = ctx.get_split(broadcast_type)

        if dim:
            split = [(d, a * dim, b * dim) for (d, a, b) in split]

        self.out_features_tp = [0] * ctx.num_devices
        for dev, a, b in split:
            self.out_features_tp[dev] = b - a

        new_q_handle = []
        new_q_tensors = []
        new_temp_dq = []

        for idx, a, b in split:
            s = b - a
            if s == 0: continue

            w = {
                "q_scale": safe_move_tensor(self.q_tensors["q_scale"][:, a // 8:b // 8], idx).contiguous(),
                "q_scale_max": safe_move_tensor(self.q_tensors["q_scale_max"], idx),
                "q_group_map": safe_move_tensor(self.q_tensors["q_group_map"], idx),
                "q_groups": safe_move_tensor(self.q_tensors["q_groups"], idx),
                "q_weight": safe_move_tensor(self.q_tensors["q_weight"][:, a:b], idx).contiguous()
            }

            if "q_perm" in self.q_tensors:
                w.update({
                    "q_perm": safe_move_tensor(self.q_tensors["q_perm"], idx),
                    "q_invperm": safe_move_tensor(self.q_tensors["q_invperm"], idx),
                })

            if "bias" in self.q_tensors:
                w["bias"] = safe_move_tensor(self.q_tensors["bias"][a:b], idx).contiguous()

            new_q_tensors.append(w)

            device_context = self.model.get_device_context(idx)
            device_context.begin_scratch_alloc()
            new_temp_dq.append(device_context.get_scratch_slice(self.temp_dq_size(out_features = s)))
            max_dq_rows = cfg.max_dq_size // s

            new_q_handle.append(
                ext_c.make_q_matrix_split(
                    w["q_weight"],
                    w.get("q_perm", none_tensor),
                    w.get("q_invperm", none_tensor),
                    w["q_scale"],
                    w["q_scale_max"],
                    w["q_groups"],
                    w["q_group_map"],
                    none_tensor,
                    none_tensor,
                    none_tensor,
                    w.get("bias", none_tensor),
                    new_temp_dq[-1],
                    max_dq_rows
                )
            )

        torch.cuda.synchronize()

        ext_c.free_q_matrix(self.q_handle)
        self.q_handle = new_q_handle
        self.q_tensors = new_q_tensors
        self.temp_dq = new_temp_dq
        self.is_tp = True


    def tp_load_rank(self, rank: int, broadcast_type: int, dim = None,
                     unmap: bool = False, output_map = None):
        """
        Load only the shard owned by device ``rank``.

        Unlike tp_split(), which materialises shards for every device and frees the
        originals, this builds exactly one shard and leaves it on ``rank``. That is what
        lets a per-GPU worker process load its own slice of a model that does not fit on a
        single card.

        Only the out_features (column-parallel) case is handled here. The in_features
        (row-parallel) case is delegated to tp_split_row(), which already slices the group
        map correctly; rank-local loading for it still loads the full tensor to host first.

        After this call the module behaves as a normal single-device layer: ``q_handle`` is
        a scalar and ``out_features_tp`` / ``in_features_tp`` are empty, so ``forward()``
        takes the ordinary path.

        ``unmap`` mirrors ``load(unmap=True)``, which ``ExLlamaV2MLP.load()`` uses for
        ``down_proj``: the permutation is dropped from this layer and applied to
        ``gate_proj``/``up_proj`` instead, via ``tensor_remap``. The permutation is returned
        so a rank-local caller can do the same. See BUG 52.

        ``output_map`` mirrors ``load(output_map=...)``: the permutation to apply to the RAW
        tensor before it is shuffled and sliced. It permutes dim 1 (out_features), which is the
        same dimension the column split slices, so it has to be applied first -- exactly as
        ``load()`` does, where the remap precedes ``make_q_matrix``'s shuffle.
        """
        assert self.q_handle is None, "tp_load_rank must run before the layer is loaded"

        cfg = self.model.config
        ctx = self.model.tp_context
        # BUG 65: forward_tp() reads self.broadcast_type; tp_split() sets it and
        # tp_load_rank() did not, so get_split(None) raised TypeError on the first GEMM.
        self.broadcast_type = broadcast_type
        split = ctx.get_split(broadcast_type)
        if dim:
            split = [(d, a * dim, b * dim) for (d, a, b) in split]

        target = [(d, a, b) for (d, a, b) in split if d == rank]
        assert len(target) == 0 or len(target) == 1
        if not target:
            # This rank owns no columns of this layer. Leave it empty; the caller skips it.
            self.out_features_tp = []
            self.in_features_tp = []
            self.is_tp = False
            self.device_idx = rank
            self._tp_empty_rank = True
            return

        _, a, b = target[0]

        # Load the full tensor to HOST, slice, then move only the shard.
        prev_device = getattr(self, "device_idx", None)
        self.device_idx = -1
        try:
            w = self.load_weight(cpu = True)
        finally:
            if prev_device is not None:
                self.device_idx = prev_device

        if not isinstance(w, dict):
            raise NotImplementedError(
                "tp_load_rank: only quantized (EXL2) tensors are supported, got "
                + type(w).__name__ + " for " + self.key
            )

        s = b - a

        # Apply the permutation to the RAW tensor, before the column slice and before the
        # shuffle. This is the same order load() uses: remap -> make_q_matrix (shuffle) -> split.
        # The permutation indexes dim 1 (out_features), which is the dimension being sliced, so
        # doing it after the slice would index past the end of the shard.
        if output_map is not None:
            _p = output_map.to(torch.int32)
            ext_c.tensor_remap(w["q_weight"], _p)
            ext_c.tensor_remap_4bit(w["q_scale"], _p)

        # q_group_map is not stored in the checkpoint; ext.make_q_matrix() derives it from
        # q_groups and the packed row count. Do the same here so the shard is self-contained.
        if "q_group_map" not in w:
            from exllamav2.ext import make_group_map
            w["q_group_map"] = make_group_map(w["q_groups"], w["q_weight"].shape[0])

        shard = {
            "q_weight":    w["q_weight"][:, a:b].contiguous(),
            "q_scale":     w["q_scale"][:, a // 8:b // 8].contiguous(),
            "q_scale_max": w["q_scale_max"].contiguous(),
            "q_groups":    w["q_groups"].contiguous(),
            "q_group_map": w["q_group_map"].contiguous(),
        }
        if "bias" in w:
            shard["bias"] = w["bias"][a:b].contiguous()
        # q_perm / q_invperm index the INPUT dimension, so a column split keeps them as-is.
        #
        # BUG 52: unless the caller asked for unmap, in which case they must be DROPPED. That is
        # what load(unmap=True) does, and ExLlamaV2MLP.load() uses it for down_proj: the
        # permutation is instead applied to gate_proj/up_proj through tensor_remap, so down_proj's
        # own GEMM has to run with q_perm = NULL. Passing it here permuted the input twice and
        # put the down_proj GEMM 142% off peak.
        perm_out = None
        if "q_perm" in w:
            if unmap:
                perm_out = w["q_perm"].clone()
            else:
                shard["q_perm"] = w["q_perm"].contiguous()
                shard["q_invperm"] = w["q_invperm"].contiguous()

        # Free the host copy before moving the shard.
        del w

        dev = self.model.get_device_context(rank)
        dev.begin_scratch_alloc()
        self.temp_dq = [dev.get_scratch_slice(self.temp_dq_size(out_features = s))]

        # The Python wrapper ext.make_q_matrix() folds prescale into q_scale_max
        # (w["q_scale_max"] *= prescale / 256) and load() then resets self.prescale to 1.
        # We call the C++ binding ext_c.make_q_matrix directly, which does NOT do that, so
        # apply it here. Verified: without this the shard's q_scale_max is exactly 256x the
        # reference (prescale is 1 for these layers).
        # Match ext.make_q_matrix's in-place fp16 multiply exactly: torch promotes a half
        # tensor times a python float to fp32 for the multiply and rounds back to half, which
        # is the same sequence `w["q_scale_max"] *= prescale / 256` performs. Doing it as an
        # explicit fp32 round-trip keeps the shard bit-identical to the reference path.
        shard["q_scale_max"] = (
            shard["q_scale_max"].float() * (self.prescale / 256.0)
        ).half()

        # ext.make_q_matrix() casts these to int16 before handing them to the kernel;
        # load_weight() returns them as int32. Do the same or make_q_matrix rejects them.
        if "q_perm" in shard:    shard["q_perm"] = shard["q_perm"].short()
        if "q_invperm" in shard: shard["q_invperm"] = shard["q_invperm"].short()

        shard = {k: safe_move_tensor(v, rank) for k, v in shard.items()}
        self.q_tensors = shard

        # Use make_q_matrix, NOT make_q_matrix_split.
        #
        # make_q_matrix passes no_map = false, so QMatrix's constructor runs shuffle_kernel,
        # which reorders the packed rows into the quantiser's bit-width grouping IN PLACE.
        # make_q_matrix_split passes no_map = true and does not shuffle.
        #
        # The existing path is: load() builds a QMatrix on the FULL tensor (shuffle once, in
        # place), then tp_split() column-slices that already-shuffled tensor and builds a
        # make_q_matrix_split (no further shuffle). Net: shuffled exactly once.
        #
        # We slice the RAW tensor, so we must shuffle exactly once here -- which is what
        # make_q_matrix does. Using make_q_matrix_split left q_weight unshuffled and the GEMM
        # output was off by ~5.3.
        #
        # make_q_matrix also folds prescale into q_scale_max, so load()'s prescale bookkeeping
        # is mirrored below.
        self.q_handle = ext_c.make_q_matrix(
            shard["q_weight"],
            shard.get("q_perm", none_tensor),
            shard.get("q_invperm", none_tensor),
            shard["q_scale"],
            shard["q_scale_max"],
            shard["q_groups"],
            shard["q_group_map"],
            none_tensor, none_tensor, none_tensor,
            shard.get("bias", none_tensor),
            self.temp_dq[0],
            cfg.max_dq_size // s
        )
        self.prev_prescale = self.prescale
        self.prescale = 1

        # Behave as a plain single-device layer from here on.
        self.out_features_tp = []
        self.in_features_tp = []
        self.is_tp = False
        self.device_idx = rank
        self._tp_replicated_dev = rank
        self._tp_local_out_features = s
        self._tp_unmapped_perm = perm_out
        return perm_out

    def tp_load_rank_row(self, rank: int, broadcast_type_in: int, broadcast_type_out: int,
                         dim = None):
        """
        Load only the shard owned by device ``rank`` for an in_features (row-parallel) layer.

        This is the row-parallel counterpart of :meth:`tp_load_rank`. ``down_proj`` and
        ``o_proj`` split their INPUT features across devices and all-reduce the partial sums,
        so the shard is a range of packed rows rather than a range of columns.

        Why this cannot simply be "load the full tensor, call tp_split_row()": that is exactly
        what it does, but the full tensor has to be shuffled first, and the shuffle has to
        happen on the host. See ``exllamav2/qshuffle.py`` for the whole story. The short
        version:

          * ``tp_split_row`` slices packed rows ``[row_a, row_b)`` out of a tensor that
            ``load()`` has **already shuffled in place** (``make_q_matrix`` ->
            ``QMatrix`` ctor -> ``shuffle_kernel``), and then builds the shard with
            ``make_q_matrix_split``, which passes ``no_map = true`` and does **not** shuffle.
            Net: shuffle once, then slice rows.

          * A rank-local loader reads *raw* rows from disk. It must therefore apply the
            shuffle itself, before slicing -- and it must apply it to the **whole** tensor,
            because the shuffle blocks are not aligned to group boundaries (a 5-bit group holds
            ``qrows * 32 / 5`` packed rows, not a multiple of the 5-row shuffle block). So
            "shuffle the shard" is a different permutation from "shuffle everything, then take
            the shard".

        ``qshuffle.shuffle_q_weight_host`` reproduces ``shuffle_kernel`` exactly on the host
        (verified byte-identical against the GPU kernel on 5-bit, 4-bit, 8-bit and mixed
        layers), so the shard this produces is bit-identical to the one ``tp_split_row``
        produces.

        After this call the module behaves as a plain single-device layer: ``q_handle`` is a
        scalar, ``in_features_tp`` / ``out_features_tp`` are empty, ``is_tp`` is False. The
        caller is responsible for column-selecting its input slice through ``q_perm`` before
        the GEMM -- see ``ExLlamaV2Attention._o_proj_row_parallel``.

        :param rank:
            Device index to load for.

        :param broadcast_type_in:
            Split to use for the input features (BROADCAST_RS for down_proj, BROADCAST_Q for
            o_proj).

        :param broadcast_type_out:
            Split the output features follow. Recorded for the caller's benefit; a row-parallel
            layer's output is full-width on every device.

        :param dim:
            Multiplier applied to the split boundaries (``head_dim`` for o_proj, None for
            down_proj).
        """
        assert self.q_handle is None, "tp_load_rank_row must run before the layer is loaded"

        cfg = self.model.config
        ctx = self.model.tp_context
        split = ctx.get_split(broadcast_type_in)
        if dim:
            split = [(d, a * dim, b * dim) for (d, a, b) in split]

        target = [(d, a, b) for (d, a, b) in split if d == rank]
        assert len(target) == 0 or len(target) == 1
        if not target:
            # This rank owns no input columns of this layer.
            self.out_features_tp = []
            self.in_features_tp = []
            self.is_tp = False
            self.device_idx = rank
            self._tp_empty_rank = True
            return

        _, a, b = target[0]
        s = b - a

        # Load the full tensor to HOST.
        prev_device = getattr(self, "device_idx", None)
        self.device_idx = -1
        try:
            w = self.load_weight(cpu = True)
        finally:
            if prev_device is not None:
                self.device_idx = prev_device

        if not isinstance(w, dict):
            raise NotImplementedError(
                "tp_load_rank_row: only quantized (EXL2) tensors are supported, got "
                + type(w).__name__ + " for " + self.key
            )

        if "q_group_map" not in w:
            from exllamav2.ext import make_group_map
            w["q_group_map"] = make_group_map(w["q_groups"], w["q_weight"].shape[0])

        # --- reproduce load()'s in-place GPU shuffle on the host, on the FULL tensor ---
        #
        # load() does: make_q_matrix(full) -> QMatrix ctor -> shuffle_kernel, in place. The
        # reference path then slices rows out of that shuffled tensor. We have the raw tensor,
        # so we shuffle it here. BUG 49's mirror image: the shuffle must happen exactly once,
        # and it must happen before the row slice.
        from exllamav2.qshuffle import shuffle_q_weight_host
        shuffle_q_weight_host(w["q_weight"], w["q_groups"], self.in_features)

        # --- now slice the packed rows exactly as tp_split_row() does ---
        q_scale = w["q_scale"]
        q_scale_max = w["q_scale_max"]
        q_group_map = w["q_group_map"]
        q_groups = w["q_groups"]
        q_weight = w["q_weight"]

        groups = q_groups.shape[0] // 2
        group_a = q_group_map[a * 2].item()
        group_b = q_group_map[(b - 1) * 2].item() + 1
        row_a = q_groups[group_a * 2 + 1].item()
        row_b = q_weight.shape[0] if group_b == groups else q_groups[group_b * 2 + 1].item()

        tq_scale = q_scale[group_a:group_b, :].contiguous()
        tq_scale_max = q_scale_max[group_a:group_b].contiguous()
        tq_weight = q_weight[row_a:row_b, :].contiguous()

        # Clone BEFORE rebasing: these are views into the source tensors and the in-place
        # `-=` below would corrupt them. (Same aliasing bug that was fixed in tp_split_row.)
        tq_group_map = q_group_map[a * 2:b * 2].clone()
        tq_groups = q_groups[group_a * 2:group_b * 2].clone()

        tq_group_map[::2] -= group_a
        tq_groups[1::2] -= row_a

        shard = {
            "q_weight":    tq_weight,
            "q_scale":     tq_scale,
            "q_scale_max": tq_scale_max,
            "q_groups":    tq_groups,
            "q_group_map": tq_group_map,
        }
        if "bias" in w:
            shard["bias"] = w["bias"].contiguous()

        # q_perm / q_invperm index the INPUT dimension. tp_split_row drops them from the
        # shard (the permutation is already baked into which rows were selected) and the caller
        # re-applies it to the gathered input. Do the same, but hand the full permutation
        # back so the caller can build its column index.
        if "q_perm" in w:
            self._tp_row_q_perm = w["q_perm"].to(torch.int64).clone()

        del w

        dev = self.model.get_device_context(rank)
        dev.begin_scratch_alloc()
        self.temp_dq = [dev.get_scratch_slice(self.temp_dq_size(in_features = s))]

        # Mirror ext.make_q_matrix()'s prescale fold (BUG 50).
        shard["q_scale_max"] = (
            shard["q_scale_max"].float() * (self.prescale / 256.0)
        ).half()

        shard = {k: safe_move_tensor(v, rank) for k, v in shard.items()}
        self.q_tensors = shard

        # make_q_matrix_split, NOT make_q_matrix: the shuffle has already been applied above,
        # and make_q_matrix would apply it a second time. This is the exact inverse of the
        # out_features case (BUG 49).
        self.q_handle = ext_c.make_q_matrix_split(
            shard["q_weight"],
            none_tensor,
            none_tensor,
            shard["q_scale"],
            shard["q_scale_max"],
            shard["q_groups"],
            shard["q_group_map"],
            none_tensor,
            none_tensor,
            none_tensor,
            shard.get("bias", none_tensor),
            self.temp_dq[0],
            cfg.max_dq_size
        )
        self.prev_prescale = self.prescale
        self.prescale = 1

        # Plain single-device state.
        self.out_features_tp = []
        self.in_features_tp = []
        self.is_tp = False
        self.device_idx = rank
        self._tp_replicated_dev = rank
        self._tp_local_in_features = s
        self.broadcast_type = broadcast_type_in
        self.broadcast_type_out = broadcast_type_out
        # The caller needs the full permutation to build its column index: tp_load_rank_row drops
        # q_perm from the shard (the permutation is baked into which rows were selected), so the
        # forward path re-applies it to the gathered input.
        return self.__dict__.get("_tp_row_q_perm", None)

    def set_device_idx(self, idx: int | None):
        # A replicated TP layer keeps its own device even though load_tp() calls
        # set_device_idx(None) on every module to mean "sharded".
        pinned = getattr(self, "_tp_replicated_dev", None)
        if pinned is not None and idx is None:
            self.device_idx = pinned
            return
        self.device_idx = idx

    def tp_load_rank_vocab(self, rank: int):
        """
        Vocab-parallel lm_head: load ONLY this rank's vocab rows (unquantized fp16).

        The replicated hb16 head is ~2.7 GB on the master alone -- the single largest
        rank-0-only tensor, and the reason the master OOMs while workers have gigabytes
        free (debug2.txt / debug5.txt era). Sharding the vocab dim puts ~0.67 GB on every
        rank instead; forward() allgathers the logits shards (0.5 MB per token at fp16 --
        negligible next to the per-layer collectives). The allgather lives in forward(),
        so BOTH the eager forward_chunk and the captured graph step get it for free, and
        every rank's graph captures the same symmetric collective.

        Mirrors load()'s unquantized branch (transpose check / normalize / padding) on the
        full CPU tensor, then slices rows [a, b) of the VC split. Padding rows stay at the
        full-vocab tail (last rank); forward_chunk masks them on the gathered logits.
        lm_head LoRAs are incompatible with a vocab shard (they emit full-vocab deltas).
        """
        assert self.q_handle is None and self.linear is None, \
            "tp_load_rank_vocab must run before the layer is loaded"
        cfg = self.model.config
        ctx = self.model.tp_context
        split = ctx.get_split(BROADCAST_VC)
        assert len(split) == 1, f"rank {rank} owns no vocab slice of {self.key}"
        _, a, b = split[0]

        # device_idx does not exist yet on an unloaded module (it is created by the
        # load paths); load_weight(cpu = True) reads to host regardless.
        self.device_idx = -1
        w = self.load_weight(cpu = True)

        if isinstance(w, dict):
            raise NotImplementedError(
                self.key + ": EXL2_TP_VOCAB_HEAD requires an unquantized (fp16) lm_head; "
                "set EXL2_TP_VOCAB_HEAD=0 for quantized heads")
        has_bias = isinstance(w, tuple)
        if has_bias:
            ww, wb = w
        else:
            ww, wb = w, None
        if self.maybe_transpose and \
            ww.data.shape[0] == self.in_features and ww.data.shape[1] == self.out_features and \
            self.in_features != self.out_features:
            ww = nn.Parameter(ww.data.T)
        if self.normalize_unq:
            ww = self.normalize(ww)
        if self.padding > 0:
            ww = nn.Parameter(F.pad(ww.data, (0, 0, 0, self.padding)).contiguous())
            if has_bias:
                wb = nn.Parameter(F.pad(wb.data, (0, 0, 0, self.padding)).contiguous())

        self.device_idx = rank
        self.linear = nn.Linear(self.in_features, b - a, has_bias, device = "meta", dtype = torch.float16)
        self.linear.weight = nn.Parameter(safe_move_tensor(ww.data[a:b].contiguous(), rank), requires_grad = False)
        if has_bias:
            self.linear.bias = nn.Parameter(safe_move_tensor(wb.data[a:b].contiguous(), rank), requires_grad = False)
        del w, ww, wb

        self._tp_vocab_parallel = True
        self.broadcast_type = None
        self.broadcast_type_out = None
        self.out_features_tp = []
        self.in_features_tp = []
        self.is_tp = False
        self._tp_replicated_dev = rank

    def tp_load_rank_replicated(self, rank: int):
        """
        Load this layer in full, pinned to one device, for a rank-local worker.

        Used for ``lm_head``, which is replicated rather than sharded: logits are produced on
        the master only, and the head is not split because a vocab-column split would need a
        gather of the logits afterwards for no benefit.

        Unlike :meth:`tp_replicate`, this does not assume the layer has already been loaded --
        it runs ``load()`` itself, so a worker can call it on a fresh module.

        :param rank:
            Device index this worker owns.
        """
        assert self.q_handle is None and self.linear is None, \
            "tp_load_rank_replicated must run before the layer is loaded"

        self.device_idx = rank
        self.load()
        self.broadcast_type = None
        self.broadcast_type_out = None
        self.out_features_tp = []
        self.in_features_tp = []
        # Deliberately NOT is_tp: forward() must take the ordinary single-device path.
        self.is_tp = False
        self._tp_replicated_dev = rank

    def tp_replicate(self, device_idx: int):
        """
        Keep the whole tensor on one device instead of sharding it.

        Used for a row-parallel projection whose rows are permuted by ``q_perm`` (EXL2 stores
        the output channels in a quantizer-chosen order, so slicing rows by head index would
        need the permutation undone first). Replicating is correct and simple; the layer is small
        relative to the block. ``forward()`` then uses the ordinary single-device path.
        """
        assert self.q_handle is not None, "Can only replicate a quantized tensor."

        for k in ("q_scale", "q_scale_max", "q_group_map", "q_groups", "q_weight",
                  "q_perm", "q_invperm", "bias"):
            if k in self.q_tensors:
                self.q_tensors[k] = safe_move_tensor(self.q_tensors[k], device_idx)

        # BUG 46: allocate a real dequantisation buffer for this device.
        #
        # `load_tp()` loads every module with `device_context = False`, so `self.temp_dq` is
        # `none_tensor` -- a 1-element META tensor -- at the time this runs. The old code
        # passed `self.temp_dq[0]`, i.e. `none_tensor[0]`: still a meta tensor, so
        # `make_q_matrix_split` stored `temp_dq = NULL` in the QMatrix.
        #
        # That is harmless while the GEMM runs its q_gemm kernel, which never touches
        # temp_dq. But `gemm_half_q_half_cuda` switches to the reconstruct+cuBLAS path when
        # `size_m > MAX_Q_GEMM_ROWS` (32) -- i.e. for any prompt longer than 32 tokens --
        # and `reconstruct_kernel` then writes the dequantised weights through that NULL
        # pointer. Result: `cudaErrorIllegalAddress` on every prefill longer than 32 tokens,
        # while short prompts worked. This is why TP prefill was never exercised: every
        # verification prompt in the project is under 32 tokens.
        device_context = self.model.get_device_context(device_idx)
        device_context.begin_scratch_alloc()
        self.temp_dq = [device_context.get_scratch_slice(self.temp_dq_size())]

        # A scalar handle, not a list: the ordinary single-device forward path passes
        # self.q_handle straight to gemm_half_q_half(), which takes an int.
        self.q_handle = ext_c.make_q_matrix_split(
            self.q_tensors["q_weight"],
            self.q_tensors.get("q_perm", none_tensor),
            self.q_tensors.get("q_invperm", none_tensor),
            self.q_tensors["q_scale"],
            self.q_tensors["q_scale_max"],
            self.q_tensors["q_groups"],
            self.q_tensors["q_group_map"],
            none_tensor,
            none_tensor,
            none_tensor,
            self.q_tensors.get("bias", none_tensor),
            self.temp_dq[0],
            self.model.config.max_dq_size // self.out_features
        )

        self.out_features_tp = []
        self.in_features_tp = []
        # Deliberately NOT is_tp: forward() asserts on an is_tp layer with no split, and the
        # whole point here is to run the ordinary single-device GEMM on the primary.
        self.is_tp = False
        # Pin the layer to that device so forward() resolves self.device() correctly. The flag
        # survives the module.set_device_idx(None) that load_tp() issues after tp_split().
        self.device_idx = device_idx
        self._tp_replicated_dev = device_idx

    def tp_split_row(self, broadcast_type_in: int, broadcast_type_out: int, dim = None):
        assert self.q_handle is not None, \
            "Can only split quantized tensor."
        assert all(x in self.q_tensors for x in [
            "q_scale",
            "q_scale_max",
            "q_group_map",
            "q_groups",
            "q_weight"
        ]), "Can only split fully loaded EXL2 tensor."
        # q_perm/q_invperm may be present. They permute the INPUT (k) dimension -- q_perm
        # has numel == in_features and the GEMM applies it as a_ptr[b_q_perm[offset_k + t]]
        # -- so stored weight rows [a, b) multiply logical input columns q_perm[a:b].
        # The shard below drops the permutation (it is already baked into which rows were
        # selected) and the caller is responsible for column-selecting its input slice
        # through q_perm before the GEMM. See ExLlamaV2Attention._o_proj_row_parallel.

        cfg = self.model.config
        ctx = self.model.tp_context
        self.broadcast_type = broadcast_type_in
        self.broadcast_type_out = broadcast_type_out
        split = ctx.get_split(broadcast_type_in)

        if dim:
            split = [(d, a * dim, b * dim) for (d, a, b) in split]

        self.in_features_tp = [0] * ctx.num_devices
        for dev, a, b in split:
            self.in_features_tp[dev] = b - a

        new_q_handle = []
        new_q_tensors = []
        new_temp_dq = []

        q_scale = self.q_tensors["q_scale"]
        q_scale_max = self.q_tensors["q_scale_max"]
        q_group_map = self.q_tensors["q_group_map"]
        q_groups = self.q_tensors["q_groups"]
        q_weight = self.q_tensors["q_weight"]

        assert q_group_map[b * 2 - 1].item() == 1

        for idx, a, b in split:
            s = b - a
            if s == 0: continue

            groups = q_groups.shape[0] // 2
            group_a = q_group_map[a * 2].item()
            group_b = q_group_map[(b - 1) * 2].item() + 1
            row_a = q_groups[group_a * 2 + 1].item()
            row_b = q_weight.shape[0] if group_b == groups else q_groups[group_b * 2 + 1].item()

            tq_scale = q_scale[group_a:group_b, :]
            tq_scale_max = q_scale_max[group_a:group_b]
            tq_weight = q_weight[row_a:row_b, :]

            # Clone BEFORE rebasing: these are views into the source tensors, and the
            # in-place `-=` below would otherwise corrupt them for every later device in
            # the loop (the source is re-sliced on each iteration).
            tq_group_map = q_group_map[a * 2:b * 2].clone()
            tq_groups = q_groups[group_a * 2:group_b * 2].clone()

            tq_group_map[::2] -= group_a
            tq_groups[1::2] -= row_a

            w = {
                "q_scale": safe_move_tensor(tq_scale, idx).clone().contiguous(),
                "q_scale_max": safe_move_tensor(tq_scale_max, idx).clone().contiguous(),
                "q_group_map": safe_move_tensor(tq_group_map, idx).clone().contiguous(),
                "q_groups": safe_move_tensor(tq_groups, idx).clone().contiguous(),
                "q_weight": safe_move_tensor(tq_weight, idx).clone().contiguous()
            }

            if "bias" in self.q_tensors:
                w["bias"] = safe_move_tensor(self.q_tensors["bias"], idx).contiguous()

            new_q_tensors.append(w)

            device_context = self.model.get_device_context(idx)
            device_context.begin_scratch_alloc()
            new_temp_dq.append(device_context.get_scratch_slice(self.temp_dq_size(in_features = s)))
            max_dq_rows = cfg.max_dq_size

            new_q_handle.append(
                ext_c.make_q_matrix_split(
                    w["q_weight"],
                    none_tensor,
                    none_tensor,
                    w["q_scale"],
                    w["q_scale_max"],
                    w["q_groups"],
                    w["q_group_map"],
                    none_tensor,
                    none_tensor,
                    none_tensor,
                    w.get("bias", none_tensor),
                    new_temp_dq[-1],
                    max_dq_rows
                )
            )

        ext_c.free_q_matrix(self.q_handle)
        self.q_handle = new_q_handle
        self.q_tensors = new_q_tensors
        self.temp_dq = new_temp_dq
        self.is_tp = True


