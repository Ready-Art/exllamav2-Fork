from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.ext import exllamav2_ext as ext_c
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2RMSNorm(ExLlamaV2Module):

    name: str = "RMSNorm"

    weight: nn.Parameter | None | list[nn.Parameter | None]
    bias: nn.Parameter | None | list[nn.Parameter | None]
    variance_epsilon: float

    is_tp: bool
    broadcast_type: int | None

    def __init__(
        self,
        model,
        key,
        archparams = None
    ):
        super().__init__(model, key, archparams)

        self.is_tp = False
        self.broadcast_type = None
        # Set by tp_replicate(); None means "not a replicated TP norm". Stage L bug 25.
        self._tp_replicated_dev = None

        self.weight = None
        self.bias = None
        self.variance_epsilon = 1e-6


    @torch.inference_mode
    def load(self, device_context = True):

        w = self.load_weight()

        if isinstance(w, tuple):
            self.weight = w[0]
            self.bias = w[1]
        else:
            self.weight = w
            self.bias = None

        assert isinstance(self.weight, nn.Parameter)
        assert self.bias is None, "RMSNorm does not support bias"
        # or isinstance(self.bias, nn.Parameter)

        self.variance_epsilon = self.model.config.norm_eps

        # Gemma adds 1 to the norm tensor for some reason
        if self.archparams.norm_constant_bias != 0:
            self.weight += self.archparams.norm_constant_bias


    def unload(self):

        if self.weight is not None:
            del self.weight
            self.weight = None

        if self.bias is not None:
            del self.bias
            self.bias = None


    def numel(self):

        return 0
        # return self.weight.numel()


    def get_weight(self) -> torch.Tensor:

        # Make sure to return the original weight tensor for Gemma
        if self.archparams.norm_constant_bias != 0:
            return self.weight.data - self.archparams.norm_constant_bias

        return self.weight.data


    def weight_footprint(self) -> int:

        hidden_size = self.model.config.hidden_size
        return hidden_size * 2


    def scratch_space_fixed(self) -> int:

        return 0


    def scratch_space(self) -> int:

        return 0


    def scratch_space_tp(self) -> list[int]:

        return [0] * self.model.tp_context.num_devices


    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        output_fp32 = False,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        if self.is_tp:
            return self.forward_tp(
                hidden_states,
                cache,
                attn_params,
                past_len,
                intermediates,
                loras,
                output_fp32,
                **kwargs
            )

        # BUG 64: in a rank-local worker, TPContext.broadcast() returns a one-element list
        # (the tensor is already full-width and replicated), but this norm is not is_tp, so it
        # takes the plain single-tensor path below. Unwrap it.
        if isinstance(hidden_states, list):
            if len(hidden_states) == 1:
                hidden_states = hidden_states[0]
            else:
                return self.forward_tp(
                    hidden_states, cache, attn_params, past_len, intermediates,
                    loras, output_fp32, **kwargs
                )

        # A replicated TP norm can be handed a pinned *host* tensor: the preceding TP MLP
        # returns ctx.get_pinned(...) so that the next module can broadcast it. Every other
        # module in a TP block broadcasts (and therefore moves) its input itself, but this
        # norm is deliberately not TP (see tp_replicate), so nothing has moved the tensor to
        # the GPU yet. ext_c.rms_norm() takes its CUDA guard from the *input*, so it raises
        # "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu". Move it here.
        # Stage L bug 25.
        if self._tp_replicated_dev is not None:
            dev = torch.device(self._tp_replicated_dev)
            if hidden_states.device != dev:
                hidden_states = safe_move_tensor(hidden_states, self._tp_replicated_dev)

        output_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if not output_fp32:
            norm = torch.empty_like(hidden_states, dtype = torch.half)
        else:
            norm = torch.empty_like(hidden_states, dtype = torch.float)

        ext_c.rms_norm(hidden_states, self.weight, norm, self.variance_epsilon)
        hidden_states = norm.view(output_shape)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


    def forward_tp(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        output_fp32 = False,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        # BUG 79: a rank-local worker keeps this norm in single-device state with the FULL
        # width weight, so the sharded rms_norm_tp path does not apply. Callers such as
        # mlp.forward_tp_old invoke forward_tp unconditionally, so fall back here.
        if not getattr(self, "is_tp", False):
            out = self.forward(
                hidden_states, cache, attn_params, past_len,
                intermediates, loras, output_fp32, **kwargs
            )
            if kwargs.get("output_split"):
                return [out]
            return out

        if isinstance(hidden_states, torch.Tensor):
            output_shape = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            hidden_states = self.model.tp_context.broadcast(0, hidden_states, self.broadcast_type)
        else:
            output_shape = hidden_states[0].shape
            hidden_states = [hs.view(-1, hs.shape[-1]) for hs in hidden_states]

        # BUG 41: reuse the output buffers instead of allocating 4 tensors per
        # norm per token. See tp_reuse_buffers in tensor_p.py.
        from exllamav2.tensor_p import tp_reuse_buffers
        _cache = self.__dict__.get("_tp_out_cache")
        if _cache is None:
            _cache = {}
            self._tp_out_cache = _cache
        outputs = tp_reuse_buffers(
            _cache,
            [(hs.device, hs.shape[0], hs.shape[1], hs.dtype) for hs in hidden_states]
        )
        # BUG 73: C++ takes Sequence[Tensor] for the weights. A rank-local worker has a
        # single-device Parameter, not a per-device list.
        _weights = self.weight if isinstance(self.weight, (list, tuple)) else [self.weight]
        ext_c.rms_norm_tp(
            hidden_states,
            _weights,
            outputs,
            self.variance_epsilon,
            self.model.tp_context.ext_tp_context
        )

        outputs = [x.view(output_shape) for x in outputs]
        return outputs


    def forward_torch(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        output_fp32 = False,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        # hidden_states.clamp_(-65504.0, 65504.0)

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if not output_fp32:
            hidden_states = hidden_states.to(self.weight.dtype)

        hidden_states *= self.weight

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


    def tp_load_rank(self, rank: int, broadcast_type: int):
        """
        Load this norm in full, pinned to one device, for a rank-local worker.

        RMSNorm is not sharded: ``tp_split()`` gives every device a full-width copy (it appends
        ``self.weight.data`` unchanged for each entry in the split), because the RMS reduction
        needs the whole hidden width. So the rank-local path is simply a full load.

        What it must NOT do is leave the module in TP state. ``tp_split()`` turns ``weight``
        into a per-device list and sets ``is_tp = True``; a worker owns one device and
        ``forward()`` would then take the TP branch and hand a one-element list to
        ``ext_c.rms_norm_tp``, which expects a list parallel to the split.

        :param rank:
            Device index this worker owns.

        :param broadcast_type:
            Recorded for parity with ``tp_split``; unused by the forward path.
        """
        # device_idx must be set BEFORE load(): load_weight() -> load_multi() reads
        # self.device(), which reads self.device_idx.
        self.device_idx = rank
        self.load()
        self.broadcast_type = broadcast_type
        self.is_tp = False
        self._tp_replicated_dev = rank
        if self.weight is not None:
            self.weight.data = safe_move_tensor(self.weight.data, rank)
        if self.bias is not None:
            self.bias.data = safe_move_tensor(self.bias.data, rank)

    def tp_replicate(self, device_idx: int):
        """
        Keep the whole norm on one device instead of splitting it.

        Used for the final norm that feeds a replicated `lm_head`. `tp_split(BROADCAST_VC)` gives
        every device a full-width copy and makes `forward()` return a *list*, which the following
        replicated linear cannot consume. Replicating keeps a single tensor on the head's device.
        Stage L bug 24.
        """
        if isinstance(self.weight, list):
            self.weight = self.weight[0]
        if isinstance(self.bias, list):
            self.bias = self.bias[0]
        # safe_move_tensor returns a plain tensor; assign into .data so the nn.Parameter
        # keeps its identity and the module's attribute really points at GPU memory.
        if self.weight is not None:
            self.weight.data = safe_move_tensor(self.weight.data, device_idx)
        if self.bias is not None:
            self.bias.data = safe_move_tensor(self.bias.data, device_idx)
        self.broadcast_type = None
        # Deliberately NOT is_tp: forward() must take the ordinary single-device path and return
        # one tensor.
        self.is_tp = False
        self.device_idx = device_idx
        self._tp_replicated_dev = device_idx

    def tp_split(self, broadcast_type: int):

        cfg = self.model.config
        self.broadcast_type = broadcast_type
        split = self.model.tp_context.get_split(broadcast_type)
        maxdev = max(dev for dev, _, _ in split)

        new_weight = []
        new_bias = []

        for idx, a, b in split:
            s = b - a
            if s == 0: continue

            if self.weight is not None:
                if self.weight.device.index == idx:
                    new_weight.append(self.weight.data)
                else:
                    new_weight.append(safe_move_tensor(self.weight, idx))

            if self.bias is not None:
                if self.bias.device.index == idx:
                    new_bias.append(self.bias.data)
                else:
                    new_bias.append(safe_move_tensor(self.bias, idx))

        self.weight = new_weight
        self.bias = new_bias
        self.is_tp = True
