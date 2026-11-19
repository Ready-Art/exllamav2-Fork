from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2HeadNorm(ExLlamaV2Module):

    name: str = "LayerNorm"

    layernorm: nn.LayerNorm | None
    weight: nn.Parameter | None
    bias: nn.Parameter | None
    variance_epsilon: float

    head_dim: int
    num_heads: int

    # Unweighted RMSNorm: normalize only, no learnable weight. Used by Gemma-4's `v_norm`.
    # Mirrors exllamav3/modules/rmsnorm.py:23 `unweighted`.
    unweighted: bool


    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        num_heads: int,
        head_dim: int,
        archparams = None,
        unweighted: bool = False
    ):
        super().__init__(model, key, archparams)

        self.layernorm = None
        self.weight = None
        self.bias = None
        self.variance_epsilon = self.model.config.norm_eps

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.unweighted = unweighted


    @torch.inference_mode
    def load(self):

        # Unweighted: there is no tensor in the checkpoint at all. Skip loading entirely.
        if self.unweighted:
            self.weight = None
            self.bias = None
            self.layernorm = None
            return

        w = self.load_weight()

        # NOTE: compare against the per-layer head_dim, NOT config.head_dim. Gemma-4 mixes
        # head_dim 256 (sliding layers) and 512 (global layers) in one model, so using
        # config.head_dim here would silently skip the repeat for the 512-dim layers and then
        # fail the shape assert below. PLAN.md §2.4 / §8 trap 2.
        if isinstance(w, tuple):
            weight = w[0]
            bias = w[1]
            if len(weight.shape) == 1 and weight.shape[0] == self.head_dim:
                weight = nn.Parameter(weight.repeat(self.num_heads, 1))
                bias = nn.Parameter(bias.repeat(self.num_heads, 1))
        else:
            weight = w
            bias = None
            if len(weight.shape) == 1 and weight.shape[0] == self.head_dim:
                weight = nn.Parameter(weight.repeat(self.num_heads, 1))

        assert isinstance(weight, nn.Parameter)
        assert bias is None or isinstance(bias, nn.Parameter)

        self.layernorm = nn.LayerNorm(
            self.model.config.hidden_size,
            elementwise_affine = True,
            bias = bias is not None
        )

        self.layernorm.weight = weight
        self.weight = weight

        if bias is not None:
            self.layernorm.bias = bias
            self.bias = bias

        assert self.weight.shape == (self.num_heads, self.head_dim), "Head norm tensor shape mismatch"

        # NOTE: use `headnorm_constant_bias`, NOT `norm_constant_bias`. Gemma-2/3 apply `(1 + w)` to
        # every norm, but Gemma-4's Gemma4RMSNorm is a plain `x * w` with no bias. The main norms and
        # the per-head norms therefore need independent flags. PLAN.md §2.4 / §8 trap 19.
        if self.archparams.headnorm_constant_bias != 0:
            self.weight += self.archparams.headnorm_constant_bias


    def unload(self):

        self.layernorm = None
        self.weight = None
        self.bias = None


    def numel(self):

        return 0
        # return self.layernorm.weight.data.numel()


    def get_weight(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:

        # Unweighted norm has no tensor to save; the loader must not look for one.
        if self.unweighted:
            return None

        w = self.weight.data
        if self.archparams.headnorm_constant_bias != 0:
            return w - self.archparams.headnorm_constant_bias

        if self.bias is not None: return w, self.bias
        return w


    def weight_footprint(self) -> int:

        # Unweighted norm occupies no weight memory.
        if self.unweighted:
            return 0

        hidden_size = self.model.config.hidden_size
        return hidden_size * 2


    def scratch_space_fixed(self) -> int:

        return 0


    def scratch_space(self) -> int:

        return 0


    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        # Unweighted: pass none_tensor for w so the kernel does the RMS reduction without a
        # multiply. Requires the C++ weightless path in ext_norm.cpp / head_norm.cu.
        weight = none_tensor if self.unweighted else self.weight.data

        ext_c.head_norm(hidden_states,
                        weight,
                        self.bias.data if self.bias is not None else none_tensor,
                        hidden_states,
                        self.variance_epsilon,
                        self.archparams.headnorm == "rmsnorm")

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


    def forward_torch(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        if self.archparams.headnorm == "rmsnorm":
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim = True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            if not self.unweighted:
                hidden_states = self.weight.to(torch.float32) * hidden_states
            hidden_states = hidden_states.to(input_dtype)
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            mean = hidden_states.mean(-1, keepdim = True)
            variance = (hidden_states - mean).pow(2).mean(-1, keepdim = True)
            hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
            if not self.unweighted:
                hidden_states = self.weight.to(torch.float32) * hidden_states
            hidden_states = hidden_states.to(input_dtype)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states
