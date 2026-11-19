from __future__ import annotations

import os
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.headnorm import ExLlamaV2HeadNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.cache import ExLlamaV2CacheBase, ExLlamaV2Cache_SWA
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.architecture import RopeStyle
from exllamav2.tensor_p import BROADCAST_KV, BROADCAST_Q, BROADCAST_RS, BROADCAST_ID, BROADCAST_VC
from exllamav2 import attn_backend
from exllamav2.attn_backend import FlashInferPaged
import math
import torch.nn.functional as F
import inspect
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

# Detect available options for attention

has_flash_attn = False
has_flash_attn_with_paged = False
has_flash_attn_with_window = False
has_flash_attn_with_softcap = False
has_xformers = False
has_lower_right_sdpa = False

if 'EXLLAMA_NO_FLASH_ATTN' not in os.environ:

    try:
        import flash_attn
        flash_attn_ver = [int(t) for t in flash_attn.__version__.split(".") if t.isdigit()]
        is_ampere_or_newer_gpu = any(torch.cuda.get_device_properties(i).major >= 8 for i in range(torch.cuda.device_count()))

        if not is_ampere_or_newer_gpu:
            print(" ## Warning: Flash Attention is installed but unsupported GPUs were detected.")

        if [2, 2, 1] <= flash_attn_ver < [2, 5, 7]:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            has_flash_attn = True

        if [2, 5, 7] <= flash_attn_ver:
            from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
            # import flash_attn_2_cuda as flash_attn_cuda

            signature = list(inspect.signature(flash_attn_func).parameters)
            has_flash_attn_with_window = "window_size" in signature
            has_flash_attn_with_softcap = "softcap" in signature

            import flash_attn_2_cuda as flash_attn_cuda
            # ext_c.set_flash_attn_func()

            has_flash_attn = True
            has_flash_attn_with_paged = True

    except ModuleNotFoundError:
        pass
    except NameError:
        pass

if 'EXLLAMA_NO_XFORMERS' not in os.environ:

    try:
        import xformers.ops as xops
        # LowerTriangularFromBottomRightMask was added in xformers version 2.4
        from xformers.ops.fmha.attn_bias import LowerTriangularFromBottomRightMask
        has_xformers = True
    except ModuleNotFoundError:
        pass

if 'EXLLAMA_NO_SDPA' not in os.environ:
    try:
        from torch.nn.attention.bias import causal_lower_right
        has_lower_right_sdpa = True
    except ImportError:
        pass

# FlashInfer detection lives in attn_backend so the capability policy is in one place.
# flash-attn is deliberately kept as a fallback for one release so this change stays bisectable
# (PLAN.md §3.5 step 2).
has_flashinfer = attn_backend.has_flashinfer


def assert_paged_attn():
    """
    Raise an exception if paged attention is not available.
    """
    global has_flash_attn_with_paged
    assert has_flash_attn_with_paged or has_flashinfer, \
        "Paged attention requires FlashInfer or Flash Attention 2.5.7 or later"


def use_flashinfer(cfg) -> bool:
    """Whether the FlashInfer backend should be preferred over flash-attn for this model."""
    return has_flashinfer and not getattr(cfg, "no_flashinfer", False)


class ExLlamaV2Attention(ExLlamaV2Module):

    name: str = "Attention"

    layer_idx: int
    pre_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm | None
    post_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm | None
    q_proj: ExLlamaV2Linear | None
    k_proj: ExLlamaV2Linear | None
    v_proj: ExLlamaV2Linear | None
    o_proj: ExLlamaV2Linear | None
    q_norm: ExLlamaV2HeadNorm | None
    k_norm: ExLlamaV2HeadNorm | None
    v_norm: ExLlamaV2HeadNorm | None

    q_handle: int | None

    attn_backend: attn_backend.FlashInferPaged | None
    use_k_as_v: bool
    is_full_attention: bool

    temp_state: torch.tensor
    temp_q: torch.tensor
    temp_k: torch.tensor
    temp_v: torch.tensor
    temp_o: torch.tensor
    temp_dq: torch.tensor
    # temp_kv: torch.tensor

    temp_lora_size: int

    has_norm: bool
    has_residual: bool
    scaling: float
    sliding_window: int

    is_tp: bool
    tp_dq_size: list[int] | None

    from exllamav2.attn_params import Params
    from exllamav2.attn_params import PagedParams

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        layer_idx: int,
        has_norm: bool = True,
        has_residual: bool = True,
        sliding_window: int = 0,
        archparams = None,
        rope_index: int = 0
    ):
        super().__init__(model, key, archparams)

        cfg = self.model.config
        ap = self.archparams
        km = self.archparams.keys

        self.is_tp = False
        self.tp_dq_size = None

        self.layer_idx = layer_idx
        self.has_norm = has_norm
        self.has_residual = has_residual
        self.rope_index = rope_index

        self.q_handle = None
        self.temp_lora_size = 0

        if ap.is_vision:
            self.num_attention_heads = cfg.vision_num_attention_heads
            self.num_key_value_heads = cfg.vision_num_key_value_heads
            self.num_key_value_groups = cfg.vision_num_key_value_groups
            self.head_dim = cfg.vision_head_dim
            self.hidden_size = cfg.vision_hidden_size
            self.use_k_as_v = False
            self.is_full_attention = False
            self.has_v_norm = False
        else:
            # Per-layer attention geometry (PLAN.md §2.2). For every architecture except
            # Gemma-4 this returns the uniform config values, so nothing else changes.
            (
                self.head_dim,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.is_full_attention,
            ) = cfg.get_layer_attn_geometry(layer_idx)
            self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
            self.hidden_size = cfg.hidden_size
            # attention_k_eq_v: full-attention layers have no v_proj, V is K (PLAN.md §2.5)
            self.use_k_as_v = bool(getattr(cfg, "attention_k_eq_v", False)) and self.is_full_attention
            # Unweighted RMSNorm applied to V (PLAN.md §2.4)
            self.has_v_norm = bool(getattr(cfg, "v_norm", False))

        hidden_size = self.hidden_size

        if self.has_norm and (km["norm_1"] or km["norm_1_post"]):
            if ap.norm == "layernorm":
                self.pre_layernorm = ExLlamaV2LayerNorm(model, key + km["norm_1"], archparams)
                self.post_layernorm = ExLlamaV2LayerNorm(model, key + km["norm_1_post"], archparams) if km["norm_1_post"] else None
            elif ap.norm == "rmsnorm":
                self.pre_layernorm = ExLlamaV2RMSNorm(model, key + km["norm_1"], archparams)
                self.post_layernorm = ExLlamaV2RMSNorm(model, key + km["norm_1_post"], archparams) if km["norm_1_post"] else None
        else:
            self.pre_layernorm = None
            self.post_layernorm = None
            self.has_norm = False

        f_a = 0
        f_b = self.num_attention_heads * self.head_dim
        f_c = f_b + self.num_key_value_heads * self.head_dim
        f_d = f_c + self.num_key_value_heads * self.head_dim
        f_key = (key + km["fused_qkv"]) if km["fused_qkv"] else None

        self.q_proj = ExLlamaV2Linear(model, key + km["attn_q"], hidden_size, self.num_attention_heads * self.head_dim, ap.attention_bias_qkv, f_key = f_key, f_beg = f_a, f_end = f_b, altpack_qkv = ap.fused_qkv_altpack)
        self.k_proj = ExLlamaV2Linear(model, key + km["attn_k"], hidden_size, self.num_key_value_heads * self.head_dim, ap.attention_bias_qkv, f_key = f_key, f_beg = f_b, f_end = f_c, altpack_qkv = ap.fused_qkv_altpack)
        if self.use_k_as_v:
            # No v_proj tensor exists in the checkpoint for these layers (PLAN.md §2.5)
            self.v_proj = None
        else:
            self.v_proj = ExLlamaV2Linear(model, key + km["attn_v"], hidden_size, self.num_key_value_heads * self.head_dim, ap.attention_bias_qkv, f_key = f_key, f_beg = f_c, f_end = f_d, altpack_qkv = ap.fused_qkv_altpack)
        self.o_proj = ExLlamaV2Linear(model, key + km["attn_o"], self.num_attention_heads * self.head_dim, hidden_size, ap.attention_bias_o, prescale = cfg.scale_depth)

        if cfg.use_qk_norm and not ap.is_vision:
            self.q_norm = ExLlamaV2HeadNorm(model, key + ".self_attn.q_norm", self.num_attention_heads, self.head_dim)
            self.k_norm = ExLlamaV2HeadNorm(model, key + ".self_attn.k_norm", self.num_key_value_heads, self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        # Unweighted RMSNorm on V (Gemma-4). No weight tensor; normalizes only. PLAN.md §2.4.
        if self.has_v_norm:
            self.v_norm = ExLlamaV2HeadNorm(
                model, key + ".self_attn.v_norm", self.num_key_value_heads, self.head_dim,
                unweighted = True
            )
        else:
            self.v_norm = None

        self.submodules = [
            self.q_proj,
            self.k_proj,
            self.o_proj
        ]
        if self.v_proj is not None:
            self.submodules.insert(2, self.v_proj)
        if self.pre_layernorm:
            self.submodules += [self.pre_layernorm]
        if self.post_layernorm:
            self.submodules += [self.post_layernorm]
        if self.q_norm is not None:
            self.submodules += [self.q_norm, self.k_norm]
        if self.v_norm is not None:
            self.submodules += [self.v_norm]

        # Attention logit scale.
        #
        # Gemma-4 is the exception: its Gemma4TextAttention hardcodes `self.scaling = 1.0`
        # (transformers/models/gemma4/modeling_gemma4.py:1170) because q_norm/k_norm already
        # normalise every head vector to unit RMS. Applying the usual 1/sqrt(head_dim) on top
        # attenuates each logit by a further 16x (head_dim 256) or 22.6x (head_dim 512), which
        # flattens the softmax towards uniform and destroys the model. See PLAN.md §2.7.
        if getattr(self.archparams, "unit_attn_scale", False):
            self.scaling = 1.0
        elif cfg.attention_multiplier:
            self.scaling = cfg.attention_multiplier
        elif cfg.query_pre_attn_scalar:
            self.scaling = cfg.query_pre_attn_scalar ** (-0.5)
        else:
            self.scaling = 1 / math.sqrt(self.head_dim)

        self.sliding_window = sliding_window

        # FlashInfer backend instance (PLAN.md §3.5). The capability decision is per layer because
        # Gemma-4 mixes head_dim 256 and 512 in one model.
        self.attn_backend = attn_backend.FlashInferPaged()


    def numel(self) -> int:

        numel = self.q_proj.numel() + \
                self.k_proj.numel() + \
                self.o_proj.numel()
        if self.v_proj is not None: numel += self.v_proj.numel()

        if self.pre_layernorm is not None: numel += self.pre_layernorm.numel()
        if self.post_layernorm is not None: numel += self.post_layernorm.numel()
        if self.q_norm is not None: numel += self.q_norm.numel()
        if self.k_norm is not None: numel += self.k_norm.numel()
        if self.v_norm is not None: numel += self.v_norm.numel()

        return numel


    @torch.inference_mode
    def load(self, device_context: bool = True):

        cfg = self.model.config

        if self.pre_layernorm is not None: self.pre_layernorm.load()
        if self.post_layernorm is not None: self.post_layernorm.load()
        self.q_proj.load(device_context = device_context)
        self.k_proj.load(device_context = device_context)
        if self.v_proj is not None: self.v_proj.load(device_context = device_context)
        self.o_proj.load(device_context = device_context)
        if self.q_norm is not None: self.q_norm.load()
        if self.k_norm is not None: self.k_norm.load()
        if self.v_norm is not None: self.v_norm.load()

        # The fused quantized attention kernel (make_q_attn) requires a non-null q_v_proj
        # (ext_qattn.cpp:33). k-eq-v layers have no v_proj, so the fused path must be disabled
        # for them; forward() then falls through to forward_torch. PLAN.md §2.5 step 4.
        if self.use_k_as_v and self.q_proj.is_quant():
            self.q_handle = None
            return

        # BUG 31: make_q_attn has no v_norm argument. The kernel applies q_norm and k_norm
        # (q_attn.cu:285-289) but nothing to V, so any layer that needs an unweighted RMSNorm on
        # V must avoid the fused path. Gemma-4 needs v_norm on all 60 layers; the 50 sliding ones
        # are quantized and would otherwise silently drop it. The non-fused branch of
        # forward_paged applies v_norm correctly (attn.py:846), so decline the handle here.
        if self.v_norm is not None and self.q_proj.is_quant():
            self.q_handle = None
            return

        if self.q_proj.is_quant():

            assert self.k_proj.is_quant() and self.o_proj.is_quant() and \
                (self.v_proj is None or self.v_proj.is_quant()), "Partially quantized attention layer"

            if device_context:
                device_context = self.model.get_device_context(self.device_idx)
                device_context.begin_scratch_alloc()
                self.temp_state = device_context.get_scratch_slice(self.temp_state_size())
                # self.temp_q = device_context.get_scratch_slice(self.temp_q_size())
                # self.temp_k = device_context.get_scratch_slice(self.temp_k_size())
                # self.temp_v = device_context.get_scratch_slice(self.temp_v_size())
                self.temp_dq = device_context.get_scratch_slice(self.temp_dq_size())
                # self.temp_kv = device_context.get_scratch_slice(self.temp_kv_size()) if self.num_attention_heads != self.num_key_value_heads else None
            else:
                self.temp_state = none_tensor
                self.temp_dq = none_tensor

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

            if self.q_norm is None:
                q_norm = none_tensor
            else:
                q_norm = self.q_norm.weight

            if self.k_norm is None:
                k_norm = none_tensor
            else:
                k_norm = self.k_norm.weight

            self.q_handle = ext_c.make_q_attn(
                norm_weight,
                norm_bias,
                is_rms,
                self.archparams.headnorm == "rmsnorm",
                eps,
                self.q_proj.q_handle,
                self.k_proj.q_handle,
                self.v_proj.q_handle,
                self.o_proj.q_handle,
                self.temp_state,
                # self.temp_q,
                # self.temp_k,
                # self.temp_v,
                self.temp_dq,
                cfg.max_input_len * cfg.max_batch_size,
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                cfg.max_seq_len,
                self.has_residual,
                self.archparams.rope_style.value,
                # Per-layer sincos width. For proportional rope this is the full head_dim (512),
                # not head_dim * partial_rotary_factor (128). PLAN.md §2.3 / §8 trap 1.
                cfg.get_layer_sincos_size(self.layer_idx),
                q_norm,
                k_norm,
                post_norm_weight,
                post_norm_bias,
                self.archparams.residual_stream_fp32,
                not cfg.no_graphs
            )


    def unload(self):
        if self.q_handle is not None:
            ext_c.free_q_attn(self.q_handle)
            self.q_handle = None

        if self.pre_layernorm is not None: self.pre_layernorm.unload()
        if self.post_layernorm is not None: self.post_layernorm.unload()
        if self.q_proj is not None: self.q_proj.unload()
        if self.k_proj is not None: self.k_proj.unload()
        if self.v_proj is not None: self.v_proj.unload()
        self.o_proj.unload()

        self.temp_state = None
        self.temp_dq = None

        if self.q_norm is not None: self.q_norm.unload()
        if self.k_norm is not None: self.k_norm.unload()
        if self.v_norm is not None: self.v_norm.unload()


    def weight_footprint(self):

        fp = self.q_proj.weight_footprint() + \
             self.k_proj.weight_footprint() + \
             self.o_proj.weight_footprint()
        if self.v_proj is not None:
            fp += self.v_proj.weight_footprint()
        if self.pre_layernorm is not None:
            fp += self.pre_layernorm.weight_footprint()
        if self.post_layernorm is not None:
            fp += self.post_layernorm.weight_footprint()
        if self.q_norm is not None:
            fp += self.q_norm.weight_footprint()
        if self.k_norm is not None:
            fp += self.k_norm.weight_footprint()
        if self.v_norm is not None:
            fp += self.v_norm.weight_footprint()

        return fp


    def scratch_space_fixed(self):

        return self.temp_state_size() + \
               self.temp_dq_size()


    def scratch_space(self):

        return self.temp_state_size() + \
               self.temp_q_size() + \
               self.temp_k_size() + \
               self.temp_v_size() + \
               self.temp_dq_size() + \
               self.temp_kv_size()
               # self.temp_attn_size() +  # Accounted for separately in model.set_device_map()


    def temp_state_size(self):

        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * max(self.num_attention_heads * self.head_dim, self.hidden_size) * 2 + 128


    def temp_q_size(self):

        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * self.num_attention_heads * self.head_dim * 2 + 128


    def temp_k_size(self):

        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * self.num_key_value_heads * self.head_dim * 2 + 128


    def temp_v_size(self):

        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * self.num_key_value_heads * self.head_dim * 2 + 128


    def temp_dq_size(self):

        return max(self.q_proj.temp_dq_size(),
                   self.k_proj.temp_dq_size(),
                   self.v_proj.temp_dq_size() if self.v_proj is not None else 0,
                   self.o_proj.temp_dq_size())


    def temp_kv_size(self):

        cfg = self.model.config
        if self.num_key_value_heads == self.num_attention_heads: return 0
        return 2 * cfg.max_seq_len * cfg.max_batch_size * self.num_attention_heads * self.head_dim * 2 + 128


    def temp_attn_size(self):
        global has_flash_attn
        global has_xformers

        cfg = self.model.config
        att_max = min(cfg.max_attention_size, cfg.max_seq_len ** 2)

        # FlashInfer and flash-attn both avoid materialising the full attention matrix, so
        # either one justifies shrinking this scratch allocation. Omitting the FlashInfer check
        # would over-allocate on a FlashInfer-only install.
        if (has_flash_attn and not cfg.no_flash_attn) or \
           (has_xformers and not cfg.no_xformers) or \
           use_flashinfer(cfg):
            #in sm>=80 devices, xformers uses the same memory as flash_attn
            #todo: due to the different implementions. in sm<80 devices, xformers uses less memory than it in sm>=80. There may still be room for optimization.
            eff = cfg.max_attention_size ** 0.5 / 190  # based on supposed memory savings listed in flash-attn repo + some fudging
            att_max //= eff

        return 2 * att_max * self.num_attention_heads * 2 + 128


    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)

        if self.pre_layernorm is not None: self.pre_layernorm.set_device_idx(idx)
        if self.post_layernorm is not None: self.post_layernorm.set_device_idx(idx)
        self.q_proj.set_device_idx(idx)
        self.k_proj.set_device_idx(idx)
        if self.v_proj is not None: self.v_proj.set_device_idx(idx)
        self.o_proj.set_device_idx(idx)
        if self.q_norm is not None: self.q_norm.set_device_idx(idx)
        if self.k_norm is not None: self.k_norm.set_device_idx(idx)
        if self.v_norm is not None: self.v_norm.set_device_idx(idx)


    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

        if n_rep == 1: return hidden_states

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        hidden_states = hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        return hidden_states


    # @profile
    def forward_paged(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.PagedParams | None = None,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:

        if self.is_tp:
            return self.forward_paged_tp(
                hidden_states,
                cache,
                attn_params,
                loras,
                **kwargs,
            )

        is_q = self.q_handle is not None
        cfg = self.model.config
        constants = self.model.get_device_context(self.device_idx, scratch = is_q)
        page_size = attn_params.page_size
        batch_size, q_len, _ = hidden_states.shape
        cache_seqlens = attn_params.get_cache_seqlens(self.device_idx)
        block_table = attn_params.get_block_index(self.device_idx)

        # SWA ring-buffer path (workstream ③). PLAN.md §4.3 Phase 3C.
        #
        # For a sliding layer the cache is a contiguous ring of `kv_state_size` tokens, not a paged
        # buffer, so attention runs over a plain slice. The shift must happen before the write and must
        # be a whole number of pages.
        if isinstance(cache, ExLlamaV2Cache_SWA) and cache.is_swa_layer[self.layer_idx]:
            return self._forward_paged_swa(
                hidden_states, cache, attn_params, loras = loras, **kwargs
            )

        # TODO: We only need keys/values when preprocess_only == True, so we could skip q projection and attention
        #   on the last layer. Would need custom kernel to update paged cache if not calling flash_attn_with_kvcache
        # skip_attn = kwargs.get("kv_only")

        # TODO: Potentially we could emulate paged cache when in Q4 mode, since that requires copying the active part
        #   of the current cache layer anyway. Test if block diagonal masking works with lower-right aligned mask.

        if cache.q_block > 1:
            k_cache_f, v_cache_f = cache.get_kv_state(self.layer_idx, batch_size, 0, attn_params.max_cache_seqlen, page_size, cache_seqlens, block_table)
        else:
            k_cache_f, v_cache_f = cache.get_kv_state(self.layer_idx, batch_size, 0, 0, page_size, cache_seqlens, block_table)

        k_cache = k_cache_f.view(k_cache_f.shape[1] // page_size, page_size, k_cache_f.shape[2], k_cache_f.shape[3])
        v_cache = v_cache_f.view(v_cache_f.shape[1] // page_size, page_size, v_cache_f.shape[2], v_cache_f.shape[3])

        sc = attn_params.get_alt_rope_embed(self.device_idx)
        if not sc:
            sin, cos = constants.sin[self.rope_index], constants.cos[self.rope_index]
        else:
            sin, cos = sc

        cache_seqlens_rope = cache_seqlens
        offsets = attn_params.get_rope_offsets(self.device_idx)
        if offsets is not None:
            cache_seqlens_rope = cache_seqlens_rope + offsets

        if is_q:
            q = torch.empty((batch_size, q_len, self.num_attention_heads, self.head_dim), device = hidden_states.device, dtype = torch.half)
            if attn_params.is_sequential:
                assert batch_size == 1
                k = k_cache_f[:, attn_params.first_index : attn_params.first_index + q_len, :, :]
                v = v_cache_f[:, attn_params.first_index : attn_params.first_index + q_len, :, :]
            else:
                k = torch.empty((batch_size, q_len, self.num_key_value_heads, self.head_dim), device = hidden_states.device, dtype = torch.half)
                v = torch.empty((batch_size, q_len, self.num_key_value_heads, self.head_dim), device = hidden_states.device, dtype = torch.half)

            if loras is None or self.temp_lora_size == 0:
                pass_loras = []
                pass_lora_temp = none_tensor
            else:
                pass_loras = [id(x) for x in loras]
                pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

            ext_c.q_attn_forward_1(
                self.q_handle,
                hidden_states,
                batch_size,
                q_len,
                0,
                cache_seqlens_rope,
                q,
                k,
                v,
                sin,
                cos,
                pass_loras,
                pass_lora_temp
            )
        else:
            residual = hidden_states
            hidden_states = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states
            q = self.q_proj.forward(hidden_states, loras = loras)
            k = self.k_proj.forward(hidden_states, loras = loras)
            # attention_k_eq_v: V is K, there is no v_proj (PLAN.md §2.5)
            v = k if self.use_k_as_v else self.v_proj.forward(hidden_states, loras = loras)
            q = q.view(batch_size, q_len, self.num_attention_heads, self.head_dim)
            k = k.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
            v = v.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
            # v_norm applies to V even when V is a copy of K. Ordering matters and is taken from
            # exllamav3/modules/attn.py:732 `finish_qkv`: v_norm runs on the *raw projected* V,
            # i.e. after the k-eq-v assignment but BEFORE q_norm/k_norm and BEFORE rope. V is
            # never roped in Gemma-4. PLAN.md §2.4, §2.5 step 5.
            #
            # BUG 30: on a k-eq-v layer `v` is the *same tensor object* as `k`, and
            # ExLlamaV2HeadNorm.forward normalises IN PLACE (dst == src). Without a copy,
            # v_norm would rewrite K and then k_norm would be applied on top of the v_normed
            # values. HF is immune because Gemma4RMSNorm returns a new tensor, so its
            # `value_states` keeps pointing at the raw k_proj output. Clone first.
            if self.v_norm is not None:
                if self.use_k_as_v:
                    v = v.clone()
                v = self.v_norm.forward(v)
            if self.q_norm is not None:
                q = self.q_norm.forward(q)
                k = self.k_norm.forward(k)
            if self.archparams.rope_style != RopeStyle.NONE:
                for t, heads in [(q, self.num_attention_heads), (k, self.num_key_value_heads)]:
                    ext_c.rope_(
                        t,
                        sin,
                        cos,
                        0,
                        heads,
                        self.head_dim,
                        cache_seqlens_rope,
                        self.archparams.rope_style == RopeStyle.NEOX
                    )
            if attn_params.is_sequential:
                k_ = k_cache_f[:, attn_params.first_index : attn_params.first_index + q_len, :, :]
                v_ = v_cache_f[:, attn_params.first_index : attn_params.first_index + q_len, :, :]
                k_.copy_(k)
                v_.copy_(v)

        if attn_params.is_sequential:
            k = None
            v = None
            cache_seqlens_a = attn_params.get_cache_seqlens_after(self.device_idx)
        else:
            cache_seqlens_a = cache_seqlens

        if cache.q_block == 1:
            cache.get_kv_state(self.layer_idx, batch_size, 0, attn_params.max_cache_seqlen, page_size, cache_seqlens, block_table)

        # Sliding window / softcap are passed to the backend directly. FlashInfer supports both
        # natively (unlike flash-attn, where they are gated behind version checks).
        window_left = self.sliding_window if self.sliding_window else -1
        softcap = cfg.attn_logit_softcapping or None

        # Backend selection. FlashInfer >= 0.7.4 serves head_dim 512 (Gemma-4 global
        # layers) on sm86 via the paged decode/prefill wrappers, so those layers take
        # the FlashInfer path too; torch_paged_attention remains the reference fallback.
        fi_ok = (
            use_flashinfer(cfg)
            and FlashInferPaged.supports(
                self.head_dim,
                self.num_attention_heads,
                self.num_key_value_heads,
                page_size,
                q.dtype if q is not None else torch.half,
            )
        )

        if fi_ok:
            attn_output = self.attn_backend.plan_and_run(
                q = q,
                k_cache = k_cache,
                v_cache = v_cache,
                cache_seqlens = cache_seqlens_a,
                block_index = block_table,
                page_size = page_size,
                sm_scale = self.scaling,
                window_left = window_left,
                logits_soft_cap = softcap,
            )
        elif has_flash_attn_with_paged and not cfg.no_flash_attn and self.head_dim <= 256:
            # BUG 37: flash-attn caps at head_dim 256; Gemma-4's global layers use 512.
            flash_kwargs = {}
            if self.sliding_window and has_flash_attn_with_window:
                flash_kwargs["window_size"] = (self.sliding_window, self.sliding_window)
            if softcap and has_flash_attn_with_softcap:
                flash_kwargs["softcap"] = softcap

            attn_output = flash_attn_with_kvcache(
                q = q,
                k = k,
                v = v,
                k_cache = k_cache,
                v_cache = v_cache,
                cache_seqlens = cache_seqlens_a,
                block_table = block_table,
                causal = True,
                softmax_scale = self.scaling,
                **flash_kwargs
            )
        else:
            # Pure-torch fallback: the reference implementation. Used for head_dim 512 on
            # Gemma-4, and whenever neither FlashInfer nor flash-attn can serve the shape.
            attn_output = attn_backend.torch_paged_attention(
                q = q,
                k_cache = k_cache,
                v_cache = v_cache,
                cache_seqlens = cache_seqlens_a,
                block_index = block_table,
                page_size = page_size,
                sm_scale = self.scaling,
                window_left = window_left,
                logits_soft_cap = softcap,
            )

        attn_output = attn_output.view((batch_size, q_len, self.num_attention_heads * self.head_dim))

        cache.store_kv_state(self.layer_idx, batch_size, 0, q_len, page_size, cache_seqlens, block_table)

        # Output projection

        if is_q:
            ext_c.q_attn_forward_2(
                self.q_handle,
                hidden_states,
                attn_output,
                batch_size,
                q_len,
                pass_loras,
                pass_lora_temp
            )
        else:
            hidden_states = self.o_proj.forward(attn_output, loras = loras)
            if self.post_layernorm:
                hidden_states = self.post_layernorm.forward(hidden_states, output_fp32 = self.archparams.residual_stream_fp32)
            if self.has_residual:
                hidden_states += residual

        return hidden_states


    def _forward_paged_swa(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2Cache_SWA,
        attn_params: ExLlamaV2Attention.PagedParams,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sliding-window attention over a contiguous ring buffer.

        Structure follows ``_torch_bighead_fallback`` in
        ``exllamav3/modules/attention_fn/torch.py:88``: work out where in the ring this chunk
        lands, shift the ring if it would overflow, write the new K/V, attend over
        ``[0, cache_pos + q_len)``, then advance.

        PLAN.md §4.3 Phase 3C.
        """

        cfg = self.model.config
        constants = self.model.get_device_context(self.device_idx, scratch = False)
        batch_size, q_len, _ = hidden_states.shape
        page_size = cache.page_size
        ring_size = cache.kv_state_size
        window = cache.sliding_window

        cache_seqlens = attn_params.get_cache_seqlens(self.device_idx)
        window_beg = attn_params.get_window_beg(self.device_idx)

        assert window_beg is not None, \
            "SWA layers require attn_params.window_beg (per-sequence ring position)"
        assert window_beg.numel() == batch_size, \
            f"window_beg has {window_beg.numel()} entries for batch {batch_size}"

        # Absolute position of the first token in this chunk, per row.
        #
        # The ring shift itself is performed once per forward pass in the generator, before the
        # layers run, so that every sliding layer shifts by the same amount. Doing it here would
        # shift once per layer and desynchronise them. PLAN.md §4.3 Phase 3C / Phase 3E.
        chunk_start = cache_seqlens - q_len
        cache_pos = chunk_start - window_beg          # [batch], position within the ring

        assert bool((cache_pos >= 0).all()), \
            "Negative ring position; window_beg bookkeeping is inconsistent"
        assert bool((cache_pos + q_len <= ring_size).all()), \
            f"Chunk overflows the ring (ring_size {ring_size}); " \
            f"the generator should have shifted it before the forward pass"

        # Projections

        residual = hidden_states
        hidden_states = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states
        q = self.q_proj.forward(hidden_states, loras = loras)
        k = self.k_proj.forward(hidden_states, loras = loras)
        v = k if self.use_k_as_v else self.v_proj.forward(hidden_states, loras = loras)
        q = q.view(batch_size, q_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)

        # v_norm on the raw projected V, before q/k norm and before rope.
        if self.v_norm is not None:
            # BUG 90a (defensive): `v` aliases `k` on a k-eq-v layer and head_norm writes
            # in place. SWA layers are never k-eq-v today, so this cannot fire, but the
            # guard keeps this path consistent with forward_paged / forward_torch.
            if self.use_k_as_v:
                v = v.clone()
            v = self.v_norm.forward(v)

        if self.q_norm is not None:
            q = self.q_norm.forward(q)
        if self.k_norm is not None:
            k = self.k_norm.forward(k)

        # RoPE

        sc = attn_params.get_alt_rope_embed(self.device_idx)
        if not sc:
            sin, cos = constants.sin[self.rope_index], constants.cos[self.rope_index]
        else:
            sin, cos = sc

        cache_seqlens_rope = cache_seqlens
        offsets = attn_params.get_rope_offsets(self.device_idx)
        if offsets is not None:
            cache_seqlens_rope = cache_seqlens_rope + offsets

        if self.archparams.rope_style != RopeStyle.NONE:
            for t, heads in [(q, self.num_attention_heads), (k, self.num_key_value_heads)]:
                ext_c.rope_(
                    t, sin, cos, 0, heads, self.head_dim,
                    cache_seqlens_rope,
                    self.archparams.rope_style == RopeStyle.NEOX
                )

        # Write the new K/V into the ring at cache_pos. Rows in a batch may have
        # different cache_pos, so this is a per-row copy.

        k_ring = cache.key_states[self.layer_idx]
        v_ring = cache.value_states[self.layer_idx]

        for b in range(batch_size):
            p = int(cache_pos[b].item())
            k_ring[b, p:p + q_len].copy_(k[b])
            v_ring[b, p:p + q_len].copy_(v[b])

        # Attention over [0, cache_pos + q_len) of the ring, with a sliding window.
        # The ring is contiguous, so build a per-row paged view: one page per
        # page_size tokens, which is exactly the layout FlashInfer wants.

        num_pages = ring_size // page_size
        block_index = torch.arange(num_pages, device = q.device, dtype = torch.int32)
        block_index = block_index.unsqueeze(0).expand(batch_size, num_pages).contiguous()

        k_cache = k_ring.view(num_pages, page_size, self.num_key_value_heads, self.head_dim)
        v_cache = v_ring.view(num_pages, page_size, self.num_key_value_heads, self.head_dim)

        # Valid length is cache_pos + q_len, which is <= ring_size.
        valid_len = (cache_pos + q_len).to(torch.int32)

        window_left = self.sliding_window if self.sliding_window else -1
        softcap = cfg.attn_logit_softcapping or None

        fi_ok = (
            use_flashinfer(cfg)
            and FlashInferPaged.supports(
                self.head_dim,
                self.num_attention_heads,
                self.num_key_value_heads,
                page_size,
                q.dtype,
            )
        )

        if fi_ok:
            attn_output = self.attn_backend.plan_and_run(
                q = q,
                k_cache = k_cache,
                v_cache = v_cache,
                cache_seqlens = valid_len,
                block_index = block_index,
                page_size = page_size,
                sm_scale = self.scaling,
                window_left = window_left,
                logits_soft_cap = softcap,
            )
        else:
            attn_output = attn_backend.torch_paged_attention(
                q = q,
                k_cache = k_cache,
                v_cache = v_cache,
                cache_seqlens = valid_len,
                block_index = block_index,
                page_size = page_size,
                sm_scale = self.scaling,
                window_left = window_left,
                logits_soft_cap = softcap,
            )

        attn_output = attn_output.reshape(
            batch_size, q_len, self.num_attention_heads * self.head_dim
        )

        # Output projection

        hidden_states = self.o_proj.forward(attn_output, loras = loras)
        if self.post_layernorm:
            hidden_states = self.post_layernorm.forward(
                hidden_states, output_fp32 = self.archparams.residual_stream_fp32
            )
        if self.has_residual:
            hidden_states = hidden_states + residual

        return hidden_states


    # @profile
    def forward_paged_tp(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.PagedParams | None = None,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:

        cfg = self.model.config
        ctx = self.model.tp_context

        # Select the split set and C++ context for this layer's attention geometry.
        ctx.set_geometry(self.layer_idx)

        attn_params.prep_tp(self.model)
        page_size = attn_params.page_size

        batch_size, q_len, _ = hidden_states.shape
        rows = batch_size * q_len
        hidden_states = hidden_states.view(-1, self.hidden_size)
        dtype = hidden_states.dtype

        k_cache_f, v_cache_f = cache.get_kv_state(
            self.layer_idx,
            batch_size,
            0,
            attn_params.max_cache_seqlen,
            page_size,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp
        )

        k_cache = [x.view(x.shape[1] // page_size, page_size, x.shape[2], x.shape[3]) for x in k_cache_f]
        v_cache = [x.view(x.shape[1] // page_size, page_size, x.shape[2], x.shape[3]) for x in v_cache_f]

        sin, cos = ctx.get_sin_cos()

        ext_c.tp_attn_forward_paged_(
            self.model.tp_context.ext_tp_context,
            hidden_states,
            self.temp_bc0,
            self.temp_bc1,
            self.temp_bc2,
            self.temp_q,
            self.temp_k,
            self.temp_v,
            self.temp_o,
            k_cache,
            v_cache,
            self.pre_layernorm.weight if self.pre_layernorm is not None else [],
            self.pre_layernorm.variance_epsilon if self.pre_layernorm is not None else 0.0,
            self.q_proj.q_handle,
            self.k_proj.q_handle,
            self.v_proj.q_handle,
            self.o_proj.q_handle,
            self.head_dim,
            int(self.archparams.rope_style),
            batch_size,
            q_len,
            sin,
            cos,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp,
            self.scaling
        )

        cache.store_kv_state(
            self.layer_idx,
            batch_size,
            0,
            q_len,
            page_size,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp
        )

        return ctx.get_pinned(0, batch_size, q_len, self.hidden_size)


    @staticmethod
    def _head_norm_slice(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        """
        RMSNorm over the last dim of a (batch, seq, heads, head_dim) tensor.

        ``w`` is the per-device slice of the head-norm weight, shape (heads, head_dim), or
        None for an unweighted norm (Gemma-4 ``v_norm``).

        BUG 38: this used to be pure torch -- upcast to fp32, pow, mean, rsqrt, multiply,
        cast back -- which is ~5 kernels and a full fp32 round trip per device per layer. At
        720 calls/token that alone was 56 ms of a 161 ms decode step (36%). The fused
        ``head_norm`` CUDA kernel already does exactly this math (it is what the non-TP
        ``ExLlamaV2HeadNorm.forward`` uses) and accepts a NULL weight for the unweighted
        case, so route through it instead.

        Numerics are unchanged: ``config.norm_eps`` is 1e-6 for Gemma-4, which is the
        constant the torch version hardcoded.
        """
        # The kernel writes in place, so it needs a contiguous fp16 view. The callers pass
        # views of the flat (rows, heads*head_dim) projection output, which are contiguous.
        if x.dtype != torch.half:
            x = x.to(torch.half)
        if not x.is_contiguous():
            x = x.contiguous()
        ext_c.head_norm(
            x,
            w if w is not None else none_tensor,
            none_tensor,
            x,
            1e-6,
            True
        )
        return x

    def _post_norm_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ``post_layernorm`` to a full-width tensor.

        The norm is TP-split (``tp_split(BROADCAST_KV)``), so its ``forward()`` only
        accepts per-device slices; handing it a full-width tensor faults in the CUDA
        kernel. Rebuild the full weight on the tensor's own device (cached) and apply
        the RMSNorm math directly. BUG 34.
        """
        norm = self.post_layernorm
        if norm is None:
            return x
        if not getattr(norm, "is_tp", False):
            return norm.forward(x, output_fp32 = self.archparams.residual_stream_fp32)

        w = getattr(self, "_post_norm_w_full", None)
        if w is None or w.device != x.device:
            # tp_split(BROADCAST_KV) gives every device a FULL-width copy of the norm
            # weight, so the list holds N identical tensors. Take one, do not concat.
            parts = norm.weight if isinstance(norm.weight, list) else [norm.weight]
            w = parts[0].to(x.device)
            self._post_norm_w_full = w

        # BUG 38: fused kernel instead of the pure-torch fp32 round trip.
        #
        # The TP kernels downstream (tp_mlp_forward_) are fp16-only, and handing them a
        # fp32 tensor produces inf on the first layer, so the output dtype must match the
        # input. rms_norm_cuda supports fp32 in / fp16 out directly, which is exactly the
        # "compute in fp32, return input dtype" behaviour the torch version had.
        x = x.contiguous()
        out = torch.empty_like(x)
        ext_c.rms_norm(x, w, out, norm.variance_epsilon)
        return out

    def _norm_weight_slice(self, norm, a: int, b: int, device: int) -> torch.Tensor:
        """
        Per-device slice of a head-norm weight, cached on that device.

        The full weight lives on the primary device. Slicing it there and multiplying
        into a tensor on another device raises a cross-device error, so the slice is
        copied to the owning device once and reused for every later forward pass.
        BUG 34b.
        """
        cache = getattr(self, "_norm_slice_cache", None)
        if cache is None:
            cache = {}
            self._norm_slice_cache = cache
        key = (id(norm), a, b, device)
        w = cache.get(key)
        if w is None:
            w = norm.weight[a:b].to(device, non_blocking = True)
            cache[key] = w
        return w

    # @profile
    def forward_paged_tp_old(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.PagedParams | None = None,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:

        assert self.q_handle is not None
        cfg = self.model.config
        split = self.model.tp_context.get_split(BROADCAST_KV)
        batch_size, q_len, _ = hidden_states.shape
        attn_params.prep_tp(self.model)
        page_size = attn_params.page_size
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        k_cache_f, v_cache_f = cache.get_kv_state(
            self.layer_idx,
            batch_size,
            0,
            attn_params.max_cache_seqlen,
            page_size,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp
        )

        k_cache = [x.view(x.shape[1] // page_size, page_size, x.shape[2], x.shape[3]) for x in k_cache_f]
        v_cache = [x.view(x.shape[1] // page_size, page_size, x.shape[2], x.shape[3]) for x in v_cache_f]

        hidden_states = self.model.tp_context.broadcast(0, hidden_states, BROADCAST_KV, dim = self.head_dim)

        residual = hidden_states

        post_norm = self.pre_layernorm.forward_tp(hidden_states, output_split = True) if self.has_norm else hidden_states
        # skip_broadcast: post_norm is already full-width and replicated on every device
        # (it is derived from a tensor that forward_tp_old broadcast at the top of this
        # function), so re-broadcasting it inside each projection is 3 pure-barrier
        # collectives per layer -- 180 per token, each a full host round trip with 4 event
        # records and 12 event waits, moving no data.
        q = self.q_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim, skip_broadcast = True)
        k = self.k_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim, skip_broadcast = True)
        # BUG 90a (TP, defensive): same aliasing hazard as forward_tp_old -- v must be an
        # independent list of clones so the in-place v_norm / k_norm / rope_ cannot clobber
        # each other on a k-eq-v layer.
        v = [x.clone() for x in k] if self.use_k_as_v else self.v_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim, skip_broadcast = True)
        q = [q_.view(batch_size, q_len, q_.shape[1] // self.head_dim, self.head_dim) for q_ in q]
        k = [k_.view(batch_size, q_len, k_.shape[1] // self.head_dim, self.head_dim) for k_ in k]
        v = [v_.view(batch_size, q_len, v_.shape[1] // self.head_dim, self.head_dim) for v_ in v]

        # Per-head norms (Gemma-4). BUG 32: these were commented out, so Gemma-4 TP ran with
        # un-normalised Q/K/V on every layer.
        #
        # Each device holds a slice of the heads, so the full-size norm weight
        # (num_heads, head_dim) must be sliced to the same range. Q is split by
        # `dim = head_dim` over `num_attention_heads`, i.e. device idx holds heads
        # [a * nkg, b * nkg); K/V hold [a, b).
        #
        # Ordering follows the non-TP path and exllamav3 `finish_qkv`: v_norm on the raw
        # projected V first, then q_norm/k_norm, then rope. V is never roped.
        if self.v_norm is not None or self.q_norm is not None:
            for idx, (dev, a, b) in enumerate(split):
                context = self.model.get_device_context(dev)
                torch.cuda.set_stream(context.stream)
                if self.v_norm is not None:
                    # k-eq-v layers alias v to k; clone so v_norm cannot corrupt K.
                    if self.use_k_as_v:
                        v[idx] = v[idx].clone()
                    v[idx] = self.v_norm.forward_torch(v[idx])
                if self.q_norm is not None:
                    nkg = self.num_key_value_groups
                    qw = self.q_norm.weight[a * nkg : b * nkg]
                    kw = self.k_norm.weight[a : b]
                    q[idx] = self._head_norm_slice(q[idx], qw)
                    k[idx] = self._head_norm_slice(k[idx], kw)
        if self.archparams.rope_style != RopeStyle.NONE:
            for idx, (dev, a, b) in enumerate(split):
                context = self.model.get_device_context(dev)
                torch.cuda.set_stream(context.stream)
                for t, heads in [(q[idx], self.num_key_value_groups), (k[idx], 1)]:
                    ext_c.rope_(
                        t,
                        context.sin[self.rope_index],
                        context.cos[self.rope_index],
                        0,
                        (b - a) * heads,
                        self.head_dim,
                        attn_params.cache_seqlens_tp[idx],
                        self.archparams.rope_style == RopeStyle.NEOX
                    )
        if attn_params.is_sequential:
            k_ = [x[:, attn_params.first_index: attn_params.first_index + q_len, :, :] for x in k_cache_f]
            v_ = [x[:, attn_params.first_index: attn_params.first_index + q_len, :, :] for x in v_cache_f]
            for (dev, a, b), x_, x, y_, y in zip(split, k_, k, v_, v):
                context = self.model.get_device_context(dev)
                torch.cuda.set_stream(context.stream)
                x_.copy_(x)
                y_.copy_(y)
            k = None
            v = None
            cache_seqlens_a = attn_params.cache_seqlens_after_tp
        else:
            cache_seqlens_a = attn_params.cache_seqlens_tp

        # if cache.q_block == 1:
        #     cache.get_kv_state(
        #         self.layer_idx,
        #         batch_size,
        #         0,
        #         attn_params.max_cache_seqlen,
        #         page_size,
        #         attn_params.cache_seqlens_tp,
        #         attn_params.block_index_tp
        #     )

        flash_kwargs = {}
        if self.sliding_window:
            # assert has_flash_attn_with_window, \
            #     "Installed version of flash-attn does not support sliding window"
            if has_flash_attn_with_window:
                flash_kwargs["window_size"] = (self.sliding_window, self.sliding_window)
        if cfg.attn_logit_softcapping:
            # assert has_flash_attn_with_softcap, \
            #     "Installed version of flash-attn does not support softcapping"
            if has_flash_attn_with_softcap:
                flash_kwargs["softcap"] = cfg.attn_logit_softcapping

        attn_outputs = []
        for idx in range(len(split)):
            dev, a, b = split[idx]
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)

            # BUG 37: flash-attn caps at head_dim 256. Gemma-4's global layers use 512 and
            # must take the torch path, exactly as they did before flash-attn was installed.
            if self.head_dim > 256:
                attn_output = attn_backend.torch_paged_attention(
                    q = q[idx],
                    k_cache = k_cache[idx],
                    v_cache = v_cache[idx],
                    cache_seqlens = cache_seqlens_a[idx],
                    block_table = attn_params.block_index_tp[idx],
                    page_size = page_size,
                    sm_scale = self.scaling,
                    window_left = self.sliding_window if self.sliding_window else -1,
                    logits_soft_cap = cfg.attn_logit_softcapping or None,
                )
            else:
                attn_output = flash_attn_with_kvcache(
                    q = q[idx],
                    k = k[idx] if k is not None else None,
                    v = v[idx] if v is not None else None,
                    k_cache = k_cache[idx],
                    v_cache = v_cache[idx],
                    cache_seqlens = cache_seqlens_a[idx],
                    block_table = attn_params.block_index_tp[idx],
                    causal = True,
                    softmax_scale = self.scaling,
                    **flash_kwargs
            )
            attn_output = attn_output.view(batch_size * q_len, (b - a) * self.head_dim * self.num_key_value_groups)
            attn_outputs.append(attn_output)

        cache.store_kv_state(
            self.layer_idx,
            batch_size,
            0,
            q_len,
            page_size,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp
        )

        # Output projection

        attn_outputs = self.model.tp_context.allgather(1, attn_outputs, BROADCAST_Q, BROADCAST_Q, dim = self.head_dim)

        hidden_states = self.o_proj.forward_tp(attn_outputs, loras = loras, dim = self.head_dim, output_split = True)

        if self.has_residual:
            self.model.tp_context.add_residual(hidden_states, residual, BROADCAST_Q, dim = self.head_dim)

        hidden_states = self.model.tp_context.gather(0, hidden_states, BROADCAST_Q, dim = self.head_dim)

        # if self.post_layernorm:  # TODO: ...
        #     hidden_states = self.post_layernorm.forward(hidden_states)

        hidden_states = hidden_states.view(batch_size, q_len, hidden_states.shape[-1])
        return hidden_states


    def _attn_flashinfer_decode(self, q, k_cache, v_cache, kv_len, dev):
        """PERF: single-token decode for head_dim > 256 layers via FlashInfer.

        Gemma-4's full-attention layers use head_dim 512, which the flash-attn paged
        kernel (head_dim <= 256) refuses. FlashInfer >= 0.7.4 serves hd512 on sm86:
        ``flashinfer.single_decode_with_kv_cache`` handles the contiguous per-device
        TP buffer directly (no plan step, no paging; kv_len is read from the tensor
        shape), ~5x faster than the torch GEMV at 16k, gap widening with context.
        Verified cos=1.0 vs SDPA at 16k/64k/128k. Eager path only: a captured FI
        kernel bakes kv_len, so the TP CUDA-graph path uses the paged decode wrapper
        with a per-step re-plan (see _fi_tp_decode_plan) or the torch fallback.

        :param q: ``(1, 1, H, D)`` local query heads for this device.
        :param k_cache / v_cache: ``(1, S, HKV, D)`` per-device KV buffer (NHD).
        :param kv_len: number of valid prefix tokens (past_len + q_len).
        :param dev: device index the layer runs on.
        :return: ``(1, 1, H * D)`` attention output, matching ``_attn_torch``.
        """
        import flashinfer
        H = q.shape[2]
        D = q.shape[3]
        q2 = q.view(H, D)
        k2 = k_cache[0, :kv_len].contiguous()   # (kv_len, HKV, D)  NHD
        v2 = v_cache[0, :kv_len].contiguous()
        # Launch on this device's TP stream, exactly like the rest of the layer (the
        # per-device loop does get_device_context(dev) + set_stream(context.stream) and
        # does not restore, so we match that convention).
        context = self.model.get_device_context(dev)
        torch.cuda.set_device(dev)
        torch.cuda.set_stream(context.stream)
        out = flashinfer.single_decode_with_kv_cache(
            q2,
            k2,
            v2,
            kv_layout = "NHD",
            pos_encoding_mode = "NONE",
            sm_scale = self.scaling,
        )
        return out.view(1, 1, H * D)

    def _attn_torch(self, batch_size, q_len, q_states, k_states, v_states, attn_params, cfg, causal = True, mask_override = None):

        num_attn_heads = q_states.shape[2]
        head_dim = q_states.shape[3]

        q_states = q_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        # BUG 85: decode-shaped GEMV fast path.
        #
        # At q_len == 1 this is not attention, it is a GEMV: 8 query heads against
        # kv_len keys. SDPA's only usable backend for head_dim 512 is the mem-efficient
        # one (`fmha_cutlassF_f16_aligned_32x128_gmem_sm80`), which is a long-sequence
        # kernel and whose cost scales LINEARLY with kv_len even though only one query row
        # is involved. Two explicit matmuls are flat in kv_len because the work is
        # bandwidth-bound on reading K and V once.
        #
        # Measured, 1x 3090, half, B=1 H=8 HKV=1 D=512 (the real Gemma-4 shape):
        #
        #   kv_len   sdpa(auto)   gemv(bcast)   speedup
        #     1046     0.1126 ms    0.1048 ms     1.08x
        #     2048     0.2016 ms    0.1113 ms     1.81x
        #     4096     0.4017 ms    0.1114 ms     3.61x
        #
        # and the GEMV matches SDPA to cos=1.000000, maxabs=1e-4.
        #
        # This matters because graph mode hands these layers the FULL cache length
        # (4096) rather than the used prefix, so they sit at the worst point of that
        # curve: 40 calls/token x 0.4017 ms = 16.1 ms/token, which is the 17.70
        # ms/token the profiler attributed to this kernel.
        #
        # Guarded to the cases where it is exactly equivalent:
        #   - q_len == 1 (a GEMV; for q_len > 1 the kernel is the right tool)
        #   - no logit softcapping (the matmul path below handles that separately)
        #   - no block-diagonal mask (that is a prefill-only construct)
        # The sliding window and the graph-mode dynamic mask are both applied to the
        # score matrix, which is (B, H, 1, kv_len) here, so they broadcast unchanged.
        if (q_len == 1
                and not cfg.attn_logit_softcapping
                and self.layer_idx not in attn_params.block_diag_layers
                # matmul broadcasts the KV head dim only when it is 1 or already equal to
                # the query head count. A general GQA ratio (e.g. 32 query : 8 kv) would
                # not broadcast, so leave those on SDPA.
                and (k_states.shape[1] == 1 or k_states.shape[1] == q_states.shape[1])):

            # (B, H, 1, D) @ (B, HKV, D, S) -> (B, H, 1, S). matmul broadcasts the
            # KV head dim, so no repeat_kv / expand is needed and no copy is made.
            attn_weights = torch.matmul(q_states, k_states.transpose(-1, -2))
            attn_weights *= self.scaling

            if mask_override is not None:
                # CUDA-graph mode: the valid-prefix bound is dynamic, so the caller
                # built the additive mask on-device from `pos`. BUG 43.
                attn_weights = attn_weights + mask_override
            elif not causal:
                pass
            elif not attn_params.is_causal():
                attn_weights = attn_weights + attn_params.get_attn_mask(attn_weights.device)
            # else: q_len == 1 and causal means every cached position is visible, so no
            # mask is needed at all.

            if self.sliding_window and attn_weights.shape[-1] >= self.sliding_window:
                attn_weights = attn_weights[..., -self.sliding_window:]
                v_states = v_states[:, :, -self.sliding_window:, :]

            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)
            attn_output = torch.matmul(attn_weights, v_states)

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape((batch_size, q_len, num_attn_heads * head_dim))
            return attn_output

        # SDPA

        if has_lower_right_sdpa and not cfg.no_sdpa and not cfg.attn_logit_softcapping:

            # Do NOT expand K/V here. SDPA reads the KV heads directly with enable_gqa=True,
            # which avoids materialising num_key_value_groups copies of K and V on every
            # layer of every token. For Gemma-4's head_dim=512 layers that temporary is
            # 32 x 1028 x 512 x 2 bytes x 2 = 67 MB per layer, per token.
            #
            # repeat_kv is still applied in the matmul fallback below, which needs the
            # explicit broadcast.
            pass

            # CUDA-graph mode expresses the sliding window in the mask instead (it is
            # position-dependent, so it cannot be a static slice). Slicing here would also
            # mismatch the mask width. BUG 43.
            if mask_override is None and self.sliding_window and k_states.shape[2] >= self.sliding_window:
                k_states = k_states[:, :, -self.sliding_window:, :]
                v_states = v_states[:, :, -self.sliding_window:, :]

            # NOTE: this used to read `if self.layer_idx in attn_params.block_diag_layers or causal:`.
            # That `or causal` sent EVERY causal call down the block-diag branch. For a normal model
            # `block_diag_layers` is empty, so `get_block_diag_mask()` returns None and SDPA then ran
            # with NO mask at all -- i.e. fully bidirectional attention. That is the TP corruption:
            # the non-TP path uses the fused C++ kernel (which masks internally) and was unaffected,
            # while the TP path falls through to `_attn_torch` because its KV cache is contiguous
            # rather than paged. Measured on Llama2-7B: TP attention output matched non-causal SDPA
            # (cos 1.0000) instead of causal (cos 0.668). Stage M bug 28.
            if self.layer_idx in attn_params.block_diag_layers:
                attn_mask_lr = attn_params.get_block_diag_mask(q_states.device)
            elif mask_override is not None:
                # CUDA-graph mode: the valid-prefix bound is dynamic, so the mask was built
                # on-device from `pos` by the caller. BUG 43.
                attn_mask_lr = mask_override
            elif not causal:
                attn_mask_lr = None
            elif attn_params.is_causal():
                attn_mask_lr = causal_lower_right(q_len, k_states.shape[2])
            else:
                attn_mask_lr = attn_params.get_attn_mask(q_states.device)
            attn_output = F.scaled_dot_product_attention(
                q_states,
                k_states,
                v_states,
                attn_mask_lr,
                scale = self.scaling,
                enable_gqa = (k_states.shape[1] != q_states.shape[1])
            )

        # Matmul attn

        else:

            k_states = self.repeat_kv(k_states, self.num_key_value_groups)
            k_states = k_states.transpose(-1, -2)

            attn_weights = torch.matmul(q_states, k_states)

            attn_weights *= self.scaling
            if self.layer_idx in attn_params.block_diag_layers:
                attn_mask = attn_params.get_block_diag_mask(attn_weights.device)
            elif mask_override is not None:
                attn_mask = mask_override
            elif causal:
                attn_mask = attn_params.get_attn_mask(attn_weights.device)
            else:
                attn_mask = None

            if cfg.attn_logit_softcapping:
                ext_c.softcap_(attn_weights, cfg.attn_logit_softcapping)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            if self.sliding_window and k_states.shape[-1] >= self.sliding_window:
                attn_weights = attn_weights[:, :, :, -self.sliding_window:]
                v_states = v_states[:, :, -self.sliding_window:, :]

            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

            v_states = self.repeat_kv(v_states, self.num_key_value_groups)
            attn_output = torch.matmul(attn_weights, v_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape((batch_size, q_len, num_attn_heads * head_dim))
        return attn_output


    def _attn_flash(self, batch_size, q_len, q_states, k_states, v_states, attn_params, cfg, causal = True):
        """
        Non-paged attention (no KV cache). Prefers FlashInfer, falls back to flash-attn.

        The function name is kept so ``forward`` needs no change (PLAN.md §3.5 step 5).
        """

        window_left = self.sliding_window if self.sliding_window else -1
        softcap = cfg.attn_logit_softcapping or None

        fi_ok = (
            use_flashinfer(cfg)
            and FlashInferPaged.supports(
                self.head_dim,
                self.num_attention_heads,
                self.num_key_value_heads,
                128,  # page size is irrelevant for the non-paged path; use the fast-path value
                q_states.dtype,
            )
        )

        if fi_ok:
            if self.layer_idx in attn_params.block_diag_layers:
                cu_seqlens = attn_params.get_cu_seqlens(self.device_idx)
                attn_output = attn_backend.varlen_prefill(
                    q_states.flatten(start_dim = 0, end_dim = 1),
                    k_states.flatten(start_dim = 0, end_dim = 1),
                    v_states.flatten(start_dim = 0, end_dim = 1),
                    cu_seqlens,
                    sm_scale = self.scaling,
                    causal = causal,
                )
            else:
                attn_output = attn_backend.single_prefill(
                    q_states,
                    k_states,
                    v_states,
                    sm_scale = self.scaling,
                    causal = causal,
                    window_left = window_left,
                    logits_soft_cap = softcap,
                )
            # Derive the head count from the tensor, not from the model: in TP mode this rank
            # holds only its local slice of the query heads.
            _n_heads = attn_output.shape[-2] if attn_output.dim() == 4 else \
                (attn_output.shape[1] if attn_output.dim() == 3 else self.num_attention_heads)
            attn_output = attn_output.reshape((batch_size, q_len, _n_heads * self.head_dim))
            return attn_output

        # flash-attn fallback
        #
        # `_attn_flash` is selected whenever EITHER flash-attn or FlashInfer is available.
        # FlashInfer >= 0.7.4 serves head_dim 512 on sm86, so the fi_ok branch above covers
        # Gemma-4's global layers; this fallback only runs for shapes FlashInfer cannot
        # serve (or on the impossible FlashInfer-less install). `flash_attn_func` is only
        # bound when the import at the top of this file succeeded, so referencing it here
        # raised NameError instead of degrading. Fall back to the pure-torch reference
        # path, which has the same signature.
        if not has_flash_attn:
            return self._attn_torch(
                batch_size, q_len, q_states, k_states, v_states, attn_params, cfg, causal
            )

        # BUG 37: flash-attn caps at head_dim 256. head_dim 512 layers are served by
        # FlashInfer above; if flash-attn is the only backend, use the torch reference.
        if self.head_dim > 256:
            return self._attn_torch(
                batch_size, q_len, q_states, k_states, v_states, attn_params, cfg, causal
            )

        flash_kwargs = {}
        if self.sliding_window and has_flash_attn_with_window:
            flash_kwargs["window_size"] = (self.sliding_window, self.sliding_window)
        if softcap and has_flash_attn_with_softcap:
            flash_kwargs["softcap"] = softcap

        if self.layer_idx in attn_params.block_diag_layers:
            q_states = q_states.flatten(start_dim = 0, end_dim = 1)
            k_states = k_states.flatten(start_dim = 0, end_dim = 1)
            v_states = v_states.flatten(start_dim = 0, end_dim = 1)
            max_seqlen = attn_params.get_cu_seqlens_max()
            cu_seqlens = attn_params.get_cu_seqlens(self.device_idx)
            attn_output = flash_attn_varlen_func(
                q_states,
                k_states,
                v_states,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen
            )
        else:
            attn_output = flash_attn_func(
                q_states,
                k_states,
                v_states,
                causal = causal,
                softmax_scale = self.scaling,
                **flash_kwargs
            )
        attn_output = attn_output.reshape((batch_size, q_len, self.num_attention_heads * self.head_dim))
        return attn_output


    def _attn_xformers(self, batch_size, q_len, q_states, k_states, v_states, attn_params, cfg, causal = True):

        # assert not self.sliding_window, \
        #     "Sliding window not currently supported for xformers"

        # assert not cfg.attn_logit_softcapping, \
        #     "Softcap not yet supported for xformers"

        # xformers memory_efficient_attention, could be beneficial if your device's architecture is less than <sm_80
        # xformer does not expand the kv automatically, we need to do it manually. The efficiency between
        # xformers.memory_efficient_attention and flash_attn in >sm_80 are almost the same. But the martix operation
        # make this implemention much slower.

        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        k_states = self.repeat_kv(k_states, self.num_key_value_groups)
        v_states = self.repeat_kv(v_states, self.num_key_value_groups)

        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        attn_output = xops.memory_efficient_attention(
            q_states,
            k_states,
            v_states,
            attn_bias = LowerTriangularFromBottomRightMask() if causal else None,
            scale = self.scaling
        )
        attn_output = attn_output.reshape((batch_size, q_len, self.num_attention_heads * self.head_dim))

        return attn_output


    # @profile
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.Params | None = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        cfg = self.model.config
        global has_flash_attn
        global has_xformers
        use_flash_attn = has_flash_attn and not cfg.no_flash_attn

        if isinstance(attn_params, ExLlamaV2Attention.PagedParams):
            return self.forward_paged(
                hidden_states,
                cache,
                attn_params,
                loras = loras,
                **kwargs
            )

        if self.is_tp:
            # Gemma-4 (per-layer geometry) must avoid the fused kernel: it assumes one global
            # split and a v_proj q_handle, and has no sliding-window support.
            _per_layer_geo = bool(getattr(cfg.arch.lm, "tp_per_layer_geometry", False))
            if cache is not None and use_flash_attn and not _per_layer_geo:
                return self.forward_tp(
                    hidden_states,
                    cache,
                    attn_params,
                    past_len,
                    intermediates,
                    loras,
                    **kwargs,
                )
            else:
                # TODO: Can't use the optimized forward function because it writes directly to a fixed output
                #   tensor, and flash-attn currently has a bug that prevents that from working when q_len == 1
                return self.forward_tp_old(
                    hidden_states,
                    cache,
                    attn_params,
                    past_len,
                    intermediates,
                    loras,
                    **kwargs,
                )

        if self.q_handle is None or intermediates:
            return self.forward_torch(
                hidden_states,
                cache,
                attn_params,
                past_len,
                intermediates,
                loras = loras,
                **kwargs
            )

        constants = self.model.get_device_context(self.device_idx)

        batch_size, q_len, _ = hidden_states.shape
        direct = (batch_size == 1 and cache is not None and isinstance(cache, ExLlamaV2CacheBase))

        q_shape = hidden_states.shape[:-1] + (self.q_proj.out_features,)
        k_shape = hidden_states.shape[:-1] + (self.k_proj.out_features,)
        v_shape = hidden_states.shape[:-1] + (self.k_proj.out_features if self.use_k_as_v else self.v_proj.out_features,)
        q_states = torch.empty(q_shape, device = hidden_states.device, dtype = torch.half)

        # If conditions are right we can write the K/V projections directly into the cache

        if direct:
            batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
            k_states = batch_keys[:batch_size, past_len : past_len + q_len, :]
            v_states = batch_values[:batch_size, past_len : past_len + q_len, :]
        else:
            k_states = torch.empty(k_shape, device = hidden_states.device, dtype = torch.half)
            v_states = torch.empty(v_shape, device = hidden_states.device, dtype = torch.half)

        # RMS norm, Q/K/V projections, position embeddings

        if loras is None or self.temp_lora_size == 0:
            pass_loras = []
            pass_lora_temp = none_tensor
        else:
            pass_loras = [id(x) for x in loras]
            pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        if attn_params.position_offsets is not None:
            pass_past_len_1 = past_len
            pass_past_len_2 = attn_params.get_position_offsets(hidden_states.device)
            if pass_past_len_1 == 0:
                offsets = attn_params.get_rope_offsets(self.device_idx)
                if offsets is not None:
                    pass_past_len_2 = pass_past_len_2 + offsets
        else:
            pass_past_len_1 = past_len
            pass_past_len_2 = none_tensor
            if attn_params.rope_offsets is not None:
                offset = attn_params.rope_offsets.cpu().item()
                pass_past_len_1 += offset

        sc = attn_params.get_alt_rope_embed(self.device_idx)
        if not sc:
            sin, cos = constants.sin[self.rope_index], constants.cos[self.rope_index]
        else:
            sin, cos = sc

        ext_c.q_attn_forward_1(
            self.q_handle,
            hidden_states,
            batch_size,
            q_len,
            pass_past_len_1,
            pass_past_len_2,
            q_states,
            k_states,
            v_states,
            sin,
            cos,
            pass_loras,
            pass_lora_temp
        )

        # Select attention function
        #
        # `_attn_flash` now serves FlashInfer as well as flash-attn, so it must be
        # selected when EITHER backend is available. Gating it on `has_flash_attn` alone
        # would silently fall back to the torch path on a FlashInfer-only install.
        use_fast_attn = (has_flash_attn and not cfg.no_flash_attn) or use_flashinfer(cfg)

        if use_fast_attn and attn_params.is_causal():
            attn_func = self._attn_flash
        elif (has_xformers and not cfg.no_xformers) and attn_params.is_causal():
            attn_func = self._attn_xformers
        else:
            attn_func = self._attn_torch

        # Straight attention without cache

        if cache is None:

            q_states = q_states.view(batch_size, q_len, self.num_attention_heads, self.head_dim)
            k_states = k_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
            v_states = v_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)

            attn_output = attn_func(batch_size, q_len, q_states, k_states, v_states, attn_params, cfg)

        # Regular cache (FP16, FP8, Q4)

        elif isinstance(cache, ExLlamaV2CacheBase):

            q_states = q_states.view(batch_size, q_len, self.num_attention_heads, self.head_dim)
            k_states = k_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
            v_states = v_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)

            if not direct:
                batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
                batch_keys[:batch_size, past_len:past_len + q_len, :].copy_(k_states)
                batch_values[:batch_size, past_len:past_len + q_len, :].copy_(v_states)

            k_states = batch_keys[:batch_size, :past_len + q_len, :]
            v_states = batch_values[:batch_size, :past_len + q_len, :]

            cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

            attn_output = attn_func(batch_size, q_len, q_states, k_states, v_states, attn_params, cfg)

        # Output projection

        ext_c.q_attn_forward_2(
            self.q_handle,
            hidden_states,
            attn_output,
            batch_size,
            q_len,
            pass_loras,
            pass_lora_temp
        )

        if self.archparams.clamp_hidden_states:
            hidden_states.clamp_(-65504, 65504)

        return hidden_states

    def forward_tp(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.Params | None = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:

        cfg = self.model.config
        ctx = self.model.tp_context

        assert not cache or cache.q_block != 1, \
            "Models with odd key/value dims not supported in TP mode with quantized cache"

        # Select the split set and C++ context for this layer's attention geometry.
        # Gemma-4 mixes head_dim 256/16-kv-head and 512/4-kv-head layers in one model, so
        # the TP split is per layer, not per model. No-op for single-geometry models.
        ctx.set_geometry(self.layer_idx)

        attn_params.prep_tp(self.model)

        batch_size, q_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        past_len = 0 if cache is None else cache.current_seq_len

        k_cache, v_cache = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len) if cache else ([], [])

        sin, cos = ctx.get_sin_cos()

        ext_c.tp_attn_forward_(
            self.model.tp_context.ext_tp_context,
            hidden_states,
            self.temp_bc0,
            self.temp_bc1,
            self.temp_bc2,
            self.temp_q,
            self.temp_k,
            self.temp_v,
            self.temp_o,
            k_cache,
            v_cache,
            self.pre_layernorm.weight if self.pre_layernorm is not None else [],
            self.pre_layernorm.variance_epsilon if self.pre_layernorm is not None else 0.0,
            self.q_proj.q_handle,
            self.k_proj.q_handle,
            self.v_proj.q_handle,
            self.o_proj.q_handle,
            self.head_dim,
            int(self.archparams.rope_style),
            batch_size,
            q_len,
            sin,
            cos,
            attn_params.past_len_tp,
            self.scaling
        )

        if cache is not None:
            cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

        return ctx.get_pinned(0, batch_size, q_len, self.hidden_size)


    def forward_tp_old(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.Params | None = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
   ):
        cfg = self.model.config

        # Select the split set and C++ context for this layer's attention geometry (Gemma-4
        # mixes 16-kv-head sliding layers with 4-kv-head full layers). Must happen before any
        # get_split() call, or the residual/output splits use the previous layer's geometry.
        self.model.tp_context.set_geometry(self.layer_idx)

        split = self.model.tp_context.get_split(BROADCAST_KV)
        batch_size, q_len, _ = hidden_states.shape
        attn_params.prep_tp(self.model)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        past_len = 0 if cache is None else cache.current_seq_len

        # NOTE: ExLlamaV2Attention is a container, not a linear layer, so it has no
        # `q_handle` of its own — the projections do. The original assert here was
        # vestigial and fired on every layer once TP actually reached this path.
        # `_attn_flash` serves FlashInfer too, so the SDPA fallback assertion only
        # applies when neither backend is available.
        use_flash_attn = (has_flash_attn and not cfg.no_flash_attn) or use_flashinfer(cfg)
        if not use_flash_attn:
            assert has_lower_right_sdpa and attn_params.is_causal() and not cfg.no_sdpa and not cfg.attn_logit_softcapping, \
                "TP attention without flash-attn or FlashInfer must use Torch SDPA with lower-right " \
                "attention mask (use PyTorch 2.4.0+) and does not support logit softcapping."

        hidden_states = self.model.tp_context.broadcast(0, hidden_states, BROADCAST_KV, dim = self.head_dim)

        residual = hidden_states

        post_norm = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states
        q = self.q_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim)
        k = self.k_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim)
        # BUG 90a (TP): on a k-eq-v layer V is K, but the two must be INDEPENDENT
        # storages from this point on: the per-head-norm block applies v_norm to v and
        # k_norm to k, both in place, and rope_ rotates k in place. If `v` were the same
        # list as `k` (the old `v = k`), the norm block's `v[idx] = v[idx].clone()` would
        # only replace the shared slot and the subsequent `k[idx] = _head_norm_slice(...)`
        # would read that same slot and clobber the v_norm result -- cached V came out
        # byte-identical to cached K (cos 1.0, V rms = the k_norm constant instead of 1.0,
        # on all 10 full layers x 4 devices). Clone each device's raw k output into a new
        # list so v and k diverge cleanly.
        v = [x.clone() for x in k] if self.use_k_as_v else self.v_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim)

        if os.environ.get("EXL2_TP_DUMP"):
            _d = os.environ["EXL2_TP_DUMP"]
            if self.layer_idx == 0:
                torch.save({
                    "post_norm": (post_norm[0] if isinstance(post_norm, list) else post_norm).detach().float().cpu(),
                    "q_flat": q[0].detach().float().cpu(),
                    "k_flat": k[0].detach().float().cpu(),
                    "v_flat": v[0].detach().float().cpu(),
                }, _d + "_pre_rope.pt")
                print("[DUMP] pre_rope saved", flush=True)

        # Per-head norms (Gemma-4). BUG 34: forward_tp_old never applied v_norm/q_norm/k_norm.
        #
        # BUG 32 added this block to forward_paged_tp_old, but Gemma-4 does not run that
        # function: ExLlamaV2StreamingGenerator uses the non-paged forward, which routes to
        # forward_tp_old. The fix therefore landed on a path Gemma-4 never executes, and TP
        # kept running with un-normalised Q/K/V on all 60 layers.
        #
        # q/k/v are still flat (rows, heads*head_dim) here because rope_ runs BEFORE the 4-D
        # view in this function, so each device's slice is viewed as (rows, local_heads,
        # head_dim) for the norm and flattened back afterwards. Ordering matches the non-TP
        # path and exllamav3 `finish_qkv`: v_norm on the raw projected V first, then
        # q_norm/k_norm, then rope. V is never roped.
        if self.v_norm is not None or self.q_norm is not None:
            for idx, (dev, a, b) in enumerate(split):
                context = self.model.get_device_context(dev)
                torch.cuda.set_stream(context.stream)
                rows = q[idx].shape[0]
                if self.v_norm is not None:
                    # k-eq-v layers alias v to k; clone so v_norm cannot corrupt K.
                    if self.use_k_as_v:
                        v[idx] = v[idx].clone()
                    nkv = v[idx].shape[1] // self.head_dim
                    vf = v[idx].view(rows, nkv, self.head_dim)
                    v[idx] = self._head_norm_slice(vf, None).view(rows, nkv * self.head_dim)
                if self.q_norm is not None:
                    nkg = self.num_key_value_groups
                    nq = q[idx].shape[1] // self.head_dim
                    nk = k[idx].shape[1] // self.head_dim
                    qw = self._norm_weight_slice(self.q_norm, a * nkg, b * nkg, dev)
                    kw = self._norm_weight_slice(self.k_norm, a, b, dev)
                    qf = q[idx].view(rows, nq, self.head_dim)
                    kf = k[idx].view(rows, nk, self.head_dim)
                    q[idx] = self._head_norm_slice(qf, qw).view(rows, nq * self.head_dim)
                    k[idx] = self._head_norm_slice(kf, kw).view(rows, nk * self.head_dim)

        # RoPE is applied to the flat (batch, seq, heads*head_dim) layout, BEFORE the 4-D view.
        # rope_ derives rows_per_batch = numel/head_dim/batch and writes num_heads rows per
        # batch, so it must see the un-folded tensor. Applying it after the view operated on a
        # non-contiguous copy and ran off the end of the buffer.
        if self.archparams.rope_style != RopeStyle.NONE:
            for idx, (dev, a, b) in enumerate(split):
                context = self.model.get_device_context(dev)
                torch.cuda.set_stream(context.stream)
                for t, heads in [(q[idx], self.num_key_value_groups), (k[idx], 1)]:
                    # rope_ expects (batch, seq, heads*head_dim). q/k are 2-D (rows, features)
                    # here, so it would read batch_size = rows and write rows_per_batch * num_heads
                    # rows per batch, running off the end of the buffer.
                    t = t.view(batch_size, q_len, -1)
                    # CUDA-graph mode: take the position from device memory (past_len = -1
                    # makes rope_cuda read past_lens[blockIdx.z]) so the captured graph
                    # replays correctly at every step instead of freezing the capture-time
                    # position. BUG 43.
                    if self.model.tp_context.graph_mode:
                        _rope_past = -1
                        # pos is per-device: rope_ runs on this layer's device and can only read
                        # that device's memory. A single dev0-resident pos made devices 1-3 rotate
                        # by garbage, which corrupted K before the KV write and showed up as a
                        # graph-vs-eager divergence at the second step.
                        _rope_off = self.model.tp_context.pos_tp.get(
                            dev, self.model.tp_context.pos
                        )
                    else:
                        _rope_past = past_len
                        _rope_off = attn_params.position_offsets_tp[idx] \
                            if attn_params.position_offsets is not None else none_tensor
                    ext_c.rope_(
                        t,
                        context.sin[self.rope_index],
                        context.cos[self.rope_index],
                        _rope_past,
                        (b - a) * heads,
                        self.head_dim,
                        _rope_off,
                        self.archparams.rope_style == RopeStyle.NEOX
                    )


        q = [q_.view(batch_size, q_len, q_.shape[1] // self.head_dim, self.head_dim) for q_ in q]
        k = [k_.view(batch_size, q_len, k_.shape[1] // self.head_dim, self.head_dim) for k_ in k]
        # BUG 90a (TP): reshape v normally for EVERY layer, including k-eq-v ones.
        #
        # The per-head-norm block above has ALREADY detached v from k on a k-eq-v layer
        # (`if self.use_k_as_v: v[idx] = v[idx].clone()`) and applied v_norm in place, so by
        # here v is an independent flat (rows, nkv*head_dim) tensor -- exactly like the
        # non-k-eq-v case. The old `v = k` discarded that v_normed clone and substituted the
        # roped K, so cached V came out byte-identical to cached K (measured cos(K, V) = 1.0
        # and V per-head rms = 0.0615, the k_norm constant, instead of 1.0, on all 10 full
        # layers x 4 devices). HF is immune because its v_norm returns a fresh tensor rather
        # than writing in place.
        v = [v_.view(batch_size, q_len, v_.shape[1] // self.head_dim, self.head_dim) for v_ in v]

        if cache:
            k_cache, v_cache = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
        else:
            k_cache, v_cache = None, None

        attn_outputs = []
        for idx in range(len(split)):
            dev, a, b = split[idx]
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)
            if os.environ.get("EXL2_TP_DEBUG"):
                print(f"[TP] layer {self.layer_idx} hd={self.head_dim} kvh={self.num_key_value_heads} "
                      f"nkg={self.num_key_value_groups} rope_idx={self.rope_index} "
                      f"split=({a},{b}) q={tuple(q[idx].shape)} k={tuple(k[idx].shape)} v={tuple(v[idx].shape)} "
                      f"kcache={None if k_cache is None else tuple(k_cache[idx].shape)} "
                      f"past={past_len} qlen={q_len}", flush=True)

            if k_cache is not None:
                # `use_flash_attn` is also True when only FlashInfer is installed, but
                # `flash_attn_with_kvcache` is imported from the `flash_attn` package and does
                # not exist there. The TP KV cache is a contiguous per-device buffer rather than
                # a paged one, so FlashInfer's paged API cannot serve it either: fall through to
                # the torch path unless real flash-attn paged support is present. (Stage L bug 22)
                # BUG 37: flash_attn_with_kvcache() supports head_dim <= 256 only. Gemma-4's
                # global layers use head_dim 512 and are served by FlashInfer (below) instead;
                # before FlashInfer hd512 support was verified they fell through to the torch
                # path, and installing flash-attn made has_flash_attn_with_paged True so this
                # branch raised "FlashAttention forward only supports head dimension at most
                # 256". Keep the head_dim guard.
                if has_flash_attn_with_paged and not cfg.no_flash_attn and self.head_dim <= 256 \
                        and not self.model.tp_context.graph_mode:
                    # FA2 is kept only for the EAGER TP path. Graphed decode (incl. SWA hd256)
                    # is served by FlashInfer's use_cuda_graph wrapper below, phasing FA2 out
                    # of the captured path entirely.
                    # CUDA-graph mode: cache_seqlens must come from a tensor whose address is
                    # stable across steps, because the captured graph holds that address. The
                    # per-device tensors on attn_params are rebuilt every forward() call, so the
                    # graph would keep reading the capture-time copy and every replay would
                    # attend to a frozen sequence length. TPContext owns a stable set instead.
                    # BUG 94: sliding-window layers keep their K/V in a small ring
                    # (kv_state_size deep), not a full-length cache. flash_attn_with_kvcache
                    # writes the new K/V at the cache_seqlens offset, so the offset and the
                    # attention length must be expressed relative to the ring start
                    # (window_beg), not the absolute sequence position: an absolute
                    # cache_seqlens runs past the end of the ring as soon as the sequence
                    # grows beyond it (the 16k prefill crash). window_size restores the
                    # sliding-window semantics the non-TP path already applies. RoPE above
                    # stays absolute -- the ring shift only drops tokens older than the
                    # window, which the attention no longer reads.
                    _swa_win = self.sliding_window if self.sliding_window else 0
                    if _swa_win:
                        if self.model.tp_context.graph_mode:
                            _rel = self.model.tp_context.ring_pos_tp.get(
                                dev, self.model.tp_context.ring_pos
                            )
                        else:
                            _rel = torch.full(
                                (batch_size,), past_len - cache.window_beg,
                                dtype = torch.int32, device = q[idx].device
                            )
                        _cache_seqlens = _rel
                    else:
                        if self.model.tp_context.graph_mode:
                            _cs = self.model.tp_context.get_past_len_tp(
                                self.model.tp_context.geometry_key_active
                            )
                            _cache_seqlens = _cs[idx] if _cs is not None else attn_params.past_len_tp[idx]
                        else:
                            _cache_seqlens = attn_params.past_len_tp[idx]
                    attn_output = flash_attn_with_kvcache(
                        q = q[idx],
                        k = k[idx],
                        v = v[idx],
                        k_cache = k_cache[idx],
                        v_cache = v_cache[idx],
                        causal = True,
                        softmax_scale = self.scaling,
                        cache_seqlens = _cache_seqlens,
                        window_size = (_swa_win, _swa_win) if _swa_win else (-1, -1),
                        # BUG 103: FA2 auto split-KV (num_splits=0) picks a schedule and
                        # reduction-buffer sizes that change with cache_seqlens, so the
                        # kernel reallocates buffers a captured graph holds addresses of ->
                        # stale reads -> token garbage once the sequence grows past the SWA
                        # ring (~kv_state_size). num_splits=1 forces single-pass decode with
                        # stable buffer addresses, making the graphed SWA decode graph-safe.
                        num_splits = 1,
                    )
                else:
                    # FlashInfer >= 0.7.4 serves head_dim > 256 (Gemma-4's hd512 global
                    # layers) on sm86: single_decode eagerly, paged decode wrapper in
                    # graph mode (re-planned per step outside the graph), and prefill
                    # eagerly. The torch path remains the fallback for everything else.
                    # FlashInfer serves hd512 full layers AND hd256 SWA layers (window_left
                    # set in the plan), so the graphed decode no longer needs FA2.
                    _fi_hd = (
                        use_flashinfer(cfg)
                        and not cfg.attn_logit_softcapping
                    )
                    if self.model.tp_context.graph_mode:
                        # CUDA-graph mode: the write offset comes from device memory, so the
                        # captured graph is position-agnostic. A host slice here would bake
                        # past_len into the copy and the graph would replay it forever.
                        # BUG 43.
                        # BUG 99: sliding layers store K/V in a ring; the write offset must
                        # be ring-relative (ring_pos, maintained by set_ring_pos), not the
                        # absolute position, which runs past the ring once window_beg > 0.
                        _write_pos = (
                            self.model.tp_context.ring_pos_tp.get(
                                dev, self.model.tp_context.ring_pos
                            )
                            if self.sliding_window else
                            self.model.tp_context.pos_tp.get(
                                dev, self.model.tp_context.pos
                            )
                        )
                        ext_c.kv_write(
                            k[idx],
                            v[idx],
                            k_cache[idx],
                            v_cache[idx],
                            _write_pos,
                            batch_size,
                            q_len,
                            k[idx].shape[2],
                            self.head_dim
                        )
                        _fi_key = (
                            dev, q[idx].shape[2], k_cache[idx].shape[2],
                            self.head_dim, self.scaling,
                            self.sliding_window if self.sliding_window else -1
                        )
                        if (_fi_hd and q_len == 1 and batch_size == 1
                                and _fi_key in self.model.tp_context.fi_decode):
                            # Graph-safe hd512 decode: the paged wrapper was planned for
                            # this step's kv_len OUTSIDE the graph (model._tp_graph_step);
                            # run() reads the refreshed plan and is captured here.
                            attn_output = self.model.tp_context.fi_decode_run(
                                _fi_key, q[idx], k_cache[idx], v_cache[idx]
                            )
                        elif (q_len > 1 and batch_size == 1 and _fi_hd
                                and all((_fi_key + (i,)) in self.model.tp_context.fi_decode
                                        for i in range(q_len))):
                            # BUG 109: graph-safe MTP verify window via per-row FI DECODE.
                            # Each query row i is a decode query attending to its causal
                            # prefix kv_len = pos + i + 1 (planned per row outside the graph,
                            # model._fi_tp_graph_plan). The decode wrapper is graph-safe
                            # (BUG 103: use_cuda_graph=True freezes its buffers) and reads
                            # ONLY the valid prefix -- not the full 160k allocation. This
                            # replaces both the full-depth matmul (slow, BUG 108) and the FI
                            # paged-prefill wrapper (re-plan realloc -> segfault). Serves
                            # hd512 full AND hd256 SWA (window_left in the plan). The W rows'
                            # KV is already written above, so row i sees draft[0..i] too.
                            _rows = []
                            for i in range(q_len):
                                _rows.append(self.model.tp_context.fi_decode_run(
                                    _fi_key + (i,), q[idx][:, i:i + 1],
                                    k_cache[idx], v_cache[idx]
                                ))
                            attn_output = torch.cat(_rows, dim = 1)   # (1, q_len, nq*hd)
                        else:
                            # Torch fallback with a dynamic device mask: shapes FlashInfer
                            # cannot serve (no plan, softcap, odd depth). Attention must
                            # only see the valid prefix; pure tensor ops, no .item()
                            # (illegal in capture).
                            cache_b = k_cache[idx].shape[1]
                            _dev = k_cache[idx].device
                            _ki = torch.arange(cache_b, device = _dev).view(1, cache_b)
                            # pos is per-device: the kernel runs on this layer's device and
                            # can only read that device's memory. `_dev` is a torch.device
                            # here, so normalise it to an int index before the dict lookup.
                            _dev_idx = _dev.index if isinstance(_dev, torch.device) else int(_dev)
                            # BUG 99: ring-relative position for sliding layers so the mask
                            # bounds line up with ring slots; the absolute position mislabels
                            # every slot once the ring has shifted.
                            _pos = (
                                self.model.tp_context.ring_pos_tp.get(
                                    _dev_idx, self.model.tp_context.ring_pos
                                )
                                if self.sliding_window else
                                self.model.tp_context.pos_tp.get(
                                    _dev_idx, self.model.tp_context.pos
                                )
                            )
                            # Change98: per-query-row causal bound. Query row i sits at
                            # absolute position pos + i and may only see keys <= pos + i.
                            # The old single bound pos + (q_len - 1) was correct for
                            # q_len == 1 but would let every row of a multi-token (MTP
                            # verify) window attend to the whole window, including future
                            # keys -- non-causal. For q_len == 1 this reduces to the
                            # original (1, 1, 1, cache_b) mask exactly.
                            _lim = _pos + torch.arange(q_len, device = _dev).view(q_len, 1)
                            # Both bounds are needed. The upper bound is the causal limit;
                            # the lower bound is the sliding window. _attn_torch would
                            # otherwise slice the *last* sliding_window entries of the
                            # whole cache, which are unwritten zeros early in the sequence,
                            # so the window has to be expressed here instead, relative to
                            # the dynamic position.
                            if self.sliding_window:
                                _lo = _lim - (self.sliding_window - 1)
                                _ok = (_ki <= _lim) & (_ki >= _lo)
                            else:
                                _ok = _ki <= _lim
                            _graph_mask = torch.where(
                                _ok, 0.0, -65504.0
                            ).to(torch.half).view(1, 1, q_len, cache_b)
                            attn_output = self._attn_torch(
                                batch_size,
                                q_len,
                                q[idx],
                                k_cache[idx][:batch_size, :cache_b, :, :],
                                v_cache[idx][:batch_size, :cache_b, :, :],
                                attn_params,
                                cfg,
                                mask_override = _graph_mask
                            )
                    else:
                        # BUG 99: sliding layers keep K/V in a ring (kv_state_size deep),
                        # not a full-length buffer. Write at the ring-relative offset
                        # (past_len - window_beg), mirroring the flash branch's _rel: the
                        # absolute offset runs past the ring on the second prefill chunk
                        # (empty-slice copy crash) and corrupts KV from then on. The chunk
                        # loop's shift protocol guarantees the write fits in [0, ring);
                        # a negative rel only drops leading rows of the chunk.
                        _swa_win = self.sliding_window if self.sliding_window else 0
                        if _swa_win:
                            _rel = attn_params.past_len - getattr(cache, "window_beg", 0)
                            _drop = max(0, -_rel)
                            cache_a = max(0, _rel)
                        else:
                            _drop = 0
                            cache_a = attn_params.past_len
                        cache_b = cache_a + q_len - _drop
                        k_cache[idx][:batch_size, cache_a:cache_b, :, :].copy_(k[idx][:, _drop:, :, :])
                        v_cache[idx][:batch_size, cache_a:cache_b, :, :].copy_(v[idx][:, _drop:, :, :])
                        if _swa_win:
                            # Windowed causal mask over the written ring prefix, in ring
                            # coordinates (same geometry as the graph-mode mask above).
                            # mask_override makes _attn_torch skip its static last-window
                            # slice, which would misalign for multi-token chunks.
                            _ql = q_len - _drop
                            _dev = k_cache[idx].device
                            _qi = torch.arange(_ql, device = _dev).view(_ql, 1) + cache_a
                            _kj = torch.arange(cache_b, device = _dev).view(1, cache_b)
                            _ok = (_kj <= _qi) & (_kj > _qi - _swa_win)
                            _swa_mask = torch.where(
                                _ok, 0.0, -65504.0
                            ).to(torch.half).view(1, 1, _ql, cache_b)
                            attn_output = self._attn_torch(
                                batch_size,
                                _ql,
                                q[idx][:, _drop:, :, :],
                                k_cache[idx][:batch_size, :cache_b, :, :],
                                v_cache[idx][:batch_size, :cache_b, :, :],
                                attn_params,
                                cfg,
                                mask_override = _swa_mask
                            )
                        elif _fi_hd and q_len == 1 and batch_size == 1:
                            # PERF: hd512 single-token decode via FlashInfer single_decode
                            # (~5x faster than the torch GEMV at 16k, gap widens with ctx).
                            attn_output = self._attn_flashinfer_decode(
                                q[idx], k_cache[idx], v_cache[idx], cache_b, dev
                            )
                        elif _fi_hd:
                            # hd512 prefill / MTP verify window via FlashInfer prefill
                            # (ragged hd512 verified eager, cos 1.0). Eager only: graphed
                            # multi-token hd512 keeps the torch mask path above.
                            attn_output = self._attn_flash(
                                batch_size,
                                q_len,
                                q[idx],
                                k_cache[idx][:batch_size, :cache_b, :, :],
                                v_cache[idx][:batch_size, :cache_b, :, :],
                                attn_params,
                                cfg,
                                causal = True
                            )
                        else:
                            attn_output = self._attn_torch(
                                batch_size,
                                q_len,
                                q[idx],
                                k_cache[idx][:batch_size, :cache_b, :, :],
                                v_cache[idx][:batch_size, :cache_b, :, :],
                                attn_params,
                                cfg
                            )
            else:
                if use_flash_attn:
                    # _attn_flash dispatches to FlashInfer (or flash-attn when available) and
                    # applies the sliding window. Calling flash_attn_func directly here was
                    # dead code on FlashInfer-only installs.
                    attn_output = self._attn_flash(
                        batch_size,
                        q_len,
                        q[idx],
                        k[idx],
                        v[idx],
                        attn_params,
                        cfg,
                        causal = True
                    )
                else:
                    attn_output = self._attn_torch(
                        batch_size,
                        q_len,
                        q[idx],
                        k[idx],
                        v[idx],
                        attn_params,
                        cfg
                    )

            attn_output = attn_output.view(batch_size * q_len, (b - a) * self.head_dim * self.num_key_value_groups)
            attn_outputs.append(attn_output)


        if os.environ.get("EXL2_TP_DUMP") and self.layer_idx == 0:
            torch.save({
                "attn_out": attn_outputs[0].detach().float().cpu(),
                "q4": q[0].detach().float().cpu(),
                "k4": k[0].detach().float().cpu(),
                "v4": v[0].detach().float().cpu(),
            }, os.environ["EXL2_TP_DUMP"] + "_post_attn.pt")
            print("[DUMP] post_attn saved", flush=True)

        # Output projection

        if getattr(self, "o_proj_row_parallel", False):

            # Row-parallel o_proj: each device runs a partial GEMM over its own slice of
            # the INPUT features, then the partial sums are all-reduced. See tp_split() for
            # why q_perm does not block this.
            #
            # allgather (not gather) because q_perm scatters every device's input slice
            # across the whole width, so each device needs the full attention output.
            gathered = self.model.tp_context.allgather(
                1, attn_outputs, BROADCAST_Q, BROADCAST_Q, dim = self.head_dim
            )
            hidden_states = self._o_proj_row_parallel(gathered, loras = loras)

            # BUG 34: the post-attention norm was commented out in both TP paths, so
            # the residual stream was never normalised. Applied here, before the
            # residual add, exactly as the non-TP path does.
            if self.post_layernorm is not None:
                hidden_states = self._post_norm_full(hidden_states)

            if self.has_residual:
                # `residual` is the per-device broadcast list; take the primary's slice.
                res = residual[0] if isinstance(residual, (list, tuple)) else residual
                if res.device != hidden_states.device:
                    res = res.to(hidden_states.device, non_blocking = True)
                hidden_states.add_(res)

        else:

            attn_outputs = self.model.tp_context.allgather(
                1, attn_outputs, BROADCAST_Q, BROADCAST_Q, dim = self.head_dim
            )
            hidden_states = self.o_proj.forward_tp(
                attn_outputs, loras = loras, dim = self.head_dim, output_split = True
            )

            if self.post_layernorm is not None:
                # BUG 34: gather the split o_proj output to the primary, apply the
                # post-attention norm full-width there, then add the residual. The
                # non-TP path does norm -> residual add in this order.
                dev0 = self.model.tp_context.device
                context = self.model.get_device_context(dev0)
                torch.cuda.set_stream(context.stream)
                hs_full = self.model.tp_context.gather(
                    0, hidden_states, BROADCAST_Q, dim = self.head_dim
                )
                hs_full = hs_full.to(dev0, non_blocking = True)
                hs_full = self._post_norm_full(hs_full)
                if self.has_residual:
                    res = residual[0] if isinstance(residual, (list, tuple)) else residual
                    if res.device != hs_full.device:
                        res = res.to(hs_full.device, non_blocking = True)
                    hs_full = hs_full + res
                hidden_states = hs_full
            else:
                if self.has_residual:
                    self.model.tp_context.add_residual(
                        hidden_states, residual, BROADCAST_Q, dim = self.head_dim
                    )

                hidden_states = self.model.tp_context.gather(
                    0, hidden_states, BROADCAST_Q, dim = self.head_dim
                )

        hidden_states = hidden_states.view(batch_size, q_len, hidden_states.shape[-1])
        if os.environ.get("EXL2_L0_POST") and self.layer_idx == 0:
            torch.save({"final": hidden_states.detach().float().cpu()},
                       os.environ["EXL2_L0_POST"] + "_tp_final.pt")
        if os.environ.get("EXL2_TP_DUMP") and self.layer_idx == 0:
            torch.save({"out": hidden_states.detach().float().cpu()},
                       os.environ["EXL2_TP_DUMP"] + "_out.pt")
            print("[DUMP] out saved", flush=True)
        return hidden_states


    def forward_torch(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.Params | None = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict:
    
        global has_flash_attn
        global has_xformers

        cfg = self.model.config
        num_attention_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        head_dim = self.head_dim

        batch_size, q_len, _ = hidden_states.size()

        past_len = 0 if cache is None else cache.current_seq_len

        # Project q, k, v

        residual = hidden_states
        post_norm = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states

        query_states = self.q_proj.forward(post_norm, loras = loras)
        key_states = self.k_proj.forward(post_norm, loras = loras)
        # attention_k_eq_v: V is K, there is no v_proj (PLAN.md §2.5)
        value_states = key_states if self.use_k_as_v else self.v_proj.forward(post_norm, loras = loras)

        # Shape for attention

        query_states = query_states.view(batch_size, q_len, num_attention_heads, head_dim)
        key_states = key_states.view(batch_size, q_len, num_key_value_heads, head_dim)
        value_states = value_states.view(batch_size, q_len, num_key_value_heads, head_dim)

        # Unweighted RMSNorm on V, on the raw projected V (after k-eq-v, before q/k norm and
        # before rope). Mirrors exllamav3/modules/attn.py:732 `finish_qkv`.
        # PLAN.md §2.4, §2.5 step 5.
        if self.v_norm is not None:
            # BUG 90a: on a k-eq-v layer `value_states` is the *same tensor object* as
            # `key_states`, and the `.view()` above keeps sharing that storage.
            # ExLlamaV2HeadNorm.forward normalises IN PLACE (ext_c.head_norm is called with
            # dst == src), so without a copy v_norm writes into the shared buffer and then
            # k_norm and rope_ overwrite it again. The cached V ends up byte-identical to the
            # cached K -- measured cos(K, V) = 1.000000 and V per-head rms = 0.0615 (the
            # k_norm constant) instead of 1.0 -- on all 10 full-attention layers. HF is
            # immune because Gemma4RMSNorm returns a new tensor. Same hazard as BUG 30,
            # already guarded in forward_paged / forward_tp_old / forward_paged_tp_old.
            if self.use_k_as_v:
                value_states = value_states.clone()
            value_states = self.v_norm.forward(value_states)

        # Apply Q/K norms
        #
        # Guard q_norm and k_norm independently: they are separate modules and an
        # architecture could in principle define only one of them. Nesting k_norm under
        # `if self.q_norm is not None` would silently skip the K norm in that case.
        if self.q_norm is not None:
            query_states = self.q_norm.forward(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm.forward(key_states)
        # Apply position embeddings

        constants = self.model.get_device_context(self.device_idx, scratch = False)

        if attn_params.position_offsets is not None:
            position_offsets = attn_params.get_position_offsets(hidden_states.device)
        else:
            position_offsets = none_tensor

        if self.archparams.rope_style != RopeStyle.NONE:
            alt_cs = kwargs.get("alt_rope_embedding")
            cos, sin = alt_cs if alt_cs else (constants.cos[self.rope_index], constants.sin[self.rope_index])

            sc = attn_params.get_alt_rope_embed(self.device_idx)
            if sc: sin, cos = sc

            past_len_rope = past_len
            if attn_params.position_offsets is None:
                if attn_params.rope_offsets is not None:
                    offset = attn_params.rope_offsets.cpu().item()
                    past_len_rope += offset
            else:
                offsets = attn_params.get_rope_offsets(self.device_idx)
                if offsets is not None:
                    position_offsets = position_offsets + offsets

            ext_c.rope_(query_states, sin, cos, past_len_rope, num_attention_heads, head_dim, position_offsets, self.archparams.rope_style == RopeStyle.NEOX)
            ext_c.rope_(key_states, sin, cos, past_len_rope, num_key_value_heads, head_dim, position_offsets, self.archparams.rope_style == RopeStyle.NEOX)


        # Add keys and values to cache

        if cache is not None:

            batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)

            # BUG 94: sliding layers keep K/V in a ring (kv_state_size deep). The
            # model.forward chunk loop shifts the ring and advances window_beg before
            # each chunk, so writes/reads here must be ring-relative, not absolute.
            # Full-attention layers keep the ordinary full-length buffer.
            _is_swa = getattr(cache, "is_swa_layer", None)
            if _is_swa is not None and _is_swa[self.layer_idx]:
                _rel = past_len - getattr(cache, "window_beg", 0)
            else:
                _rel = past_len

            new_keys = batch_keys.narrow(1, _rel, q_len).narrow(0, 0, batch_size)
            new_values = batch_values.narrow(1, _rel, q_len).narrow(0, 0, batch_size)
            new_keys.copy_(key_states)
            new_values.copy_(value_states)

            # Key/value tensors with past (valid ring prefix for sliding layers; the
            # attention functions apply the sliding window on top of it)
            key_states = batch_keys.narrow(1, 0, _rel + q_len).narrow(0, 0, batch_size)
            value_states = batch_values.narrow(1, 0, _rel + q_len).narrow(0, 0, batch_size)

        use_flash_attn = has_flash_attn and not cfg.no_flash_attn
        use_xformers = has_xformers and not cfg.no_xformers
        # `_attn_flash` serves FlashInfer too, so it is usable without flash-attn.
        use_fast_attn = use_flash_attn or use_flashinfer(cfg)

        # Select attention function

        if not (use_fast_attn or use_xformers) or not attn_params.is_causal():
            attn_func = self._attn_torch
        elif use_fast_attn:
            attn_func = self._attn_flash
        else:
            attn_func = self._attn_xformers

        if os.environ.get("EXL2_TP_DUMP") and self.layer_idx == 0:
            torch.save({
                "post_norm": post_norm.detach().float().cpu(),
                "q4": query_states.detach().float().cpu(),
                "k4": key_states.detach().float().cpu(),
                "v4": value_states.detach().float().cpu(),
            }, os.environ["EXL2_TP_DUMP"] + "_single_pre_attn.pt")
            print("[DUMP] single pre_attn saved", flush=True)

        # Attention

        attn_output = attn_func(
            batch_size,
            q_len,
            query_states,
            key_states,
            value_states,
            attn_params,
            cfg,
            not attn_params.non_causal_attn
        )


        if os.environ.get("EXL2_TP_DUMP") and self.layer_idx == 0:
            torch.save({"attn_out": attn_output.detach().float().cpu()},
                       os.environ["EXL2_TP_DUMP"] + "_single_attn_out.pt")
            print("[DUMP] single attn_out saved", flush=True)

        # Output projection

        attn_proj = self.o_proj.forward(attn_output, loras = loras)


        # Post layernorm

        if self.post_layernorm:
            attn_proj = self.post_layernorm.forward(attn_proj, output_fp32 = self.archparams.residual_stream_fp32)

        # Add residual connection

        hidden_states = (attn_proj + residual) if self.has_residual else attn_proj

        if self.archparams.residual_stream_fp32:
            hidden_states = hidden_states.float()
        elif self.archparams.clamp_hidden_states:
            hidden_states.clamp_(-65504, 65504)

        if intermediates:
            return {"post_norm": post_norm,
                    "attn_output": attn_output,
                    "hidden_states": hidden_states}
        else:
            return hidden_states


    def _o_proj_perm_slice(self, dev: int, lo: int, hi: int) -> torch.Tensor:
        """
        Cached int64 column index for device ``dev``'s input slice [lo, hi) of o_proj.

        ``q_perm`` maps a stored weight row to the logical input column it multiplies, so
        the slice this device owns is ``q_perm[lo:hi]``. Cached per device because the
        permutation is fixed at load time and index_select() needs a stable tensor.
        """
        cache = self.__dict__.get("_o_proj_perm_cache")
        if cache is None:
            cache = {}
            self._o_proj_perm_cache = cache
        key = (dev, lo, hi)
        p = cache.get(key)
        if p is None:
            p = self._o_proj_q_perm[lo:hi].to(dev, non_blocking = True)
            cache[key] = p
        return p

    def _o_proj_row_parallel(self, gathered, loras = None) -> torch.Tensor:
        """
        Row-parallel o_proj forward: per-device partial GEMM + all-reduce.

        ``gathered`` is the full-width attention output replicated on every device (the
        result of allgather). Device i owns input columns ``q_perm[lo:hi)``, produces a
        partial sum over hidden_size, and the partials are summed into one full-width
        tensor on the primary device.

        All buffers are cached: this runs inside a CUDA-graph capture region during decode,
        where torch.empty() is not allowed and every address must be stable across replays.
        """
        ctx = self.model.tp_context
        split = ctx.get_split(BROADCAST_Q)
        dev0 = ctx.device

        rows = gathered[0].shape[0]
        dtype = gathered[0].dtype
        out_features = self.o_proj.out_features

        cache = self.__dict__.get("_o_proj_row_bufs")
        if cache is None:
            cache = {}
            self._o_proj_row_bufs = cache

        # BUG 70: q_handle is a per-device list in the single-process path but a scalar in a
        # rank-local worker (the layer is in single-device state). Normalise once.
        _q_handles = self.o_proj.q_handle \
            if isinstance(self.o_proj.q_handle, (list, tuple)) else [self.o_proj.q_handle]

        partials = []
        for idx, (dev, a, b) in enumerate(split):
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)

            lo, hi = a * self.head_dim, b * self.head_dim
            perm = self._o_proj_perm_slice(dev, lo, hi)

            # Column-select this device's input slice. Cached: index_select allocates.
            xkey = ("x", dev, rows, lo, hi)
            x = cache.get(xkey)
            if x is None or x.shape[0] != rows:
                x = torch.empty((rows, hi - lo), device = dev, dtype = dtype)
                cache[xkey] = x
            torch.index_select(gathered[idx], 1, perm, out = x)

            okey = ("o", dev, rows)
            buf = cache.get(okey)
            if buf is None or buf.shape[0] != rows:
                buf = torch.empty((rows, out_features), device = dev, dtype = dtype)
                cache[okey] = buf

            ext_c.gemm_half_q_half_tp(
                [x],
                [_q_handles[idx]],
                [buf],
                False,
                ctx.ext_tp_context,
                dev
            )
            partials.append(buf)

        # BUG 75: in a rank-local worker there is exactly one device context, so the
        # host-staged all-reduce below would dereference None. The reduction is still
        # required -- each rank holds only its own partial sum over its slice of the input
        # features -- so do it with NCCL.
        if self.model.tp_context.rank is not None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(partials[0], op = dist.ReduceOp.SUM)
            return partials[0]

        # All-reduce the partial sums onto the primary device.
        torch.cuda.set_stream(self.model.get_device_context(dev0).stream)

        akey = ("acc", dev0, rows)
        acc = cache.get(akey)
        if acc is None or acc.shape[0] != rows:
            acc = torch.empty((rows, out_features), device = dev0, dtype = dtype)
            cache[akey] = acc
        acc.copy_(partials[0])

        for idx in range(1, len(split)):
            dev = split[idx][0]
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)
            skey = ("s", dev, rows)
            stage = cache.get(skey)
            if stage is None or stage.shape[0] != rows:
                stage = torch.empty((rows, out_features), device = dev0, dtype = dtype)
                cache[skey] = stage
            stage.copy_(partials[idx])
            torch.cuda.set_stream(self.model.get_device_context(dev0).stream)
            acc.add_(stage)

        return acc

    def update_loras(self):

        if self.q_handle is None: return

        cfg = self.model.config

        q_proj_lora_a = { id(k): v for k, v in self.q_proj.lora_a_tensors.items() }
        q_proj_lora_b = { id(k): v for k, v in self.q_proj.lora_b_tensors.items() }
        k_proj_lora_a = { id(k): v for k, v in self.k_proj.lora_a_tensors.items() }
        k_proj_lora_b = { id(k): v for k, v in self.k_proj.lora_b_tensors.items() }
        v_proj_lora_a = { id(k): v for k, v in self.v_proj.lora_a_tensors.items() } if self.v_proj is not None else {}
        v_proj_lora_b = { id(k): v for k, v in self.v_proj.lora_b_tensors.items() } if self.v_proj is not None else {}
        o_proj_lora_a = { id(k): v for k, v in self.o_proj.lora_a_tensors.items() }
        o_proj_lora_b = { id(k): v for k, v in self.o_proj.lora_b_tensors.items() }

        temp_lora_size = ext_c.q_attn_set_loras(
            self.q_handle,
            q_proj_lora_a,
            q_proj_lora_b,
            k_proj_lora_a,
            k_proj_lora_b,
            v_proj_lora_a,
            v_proj_lora_b,
            o_proj_lora_a,
            o_proj_lora_b
        )

        self.temp_lora_size = temp_lora_size * cfg.max_batch_size * cfg.max_input_len


    def is_quant(self):
        return self.q_handle is not None


    def tp_split(self):

        cfg = self.model.config
        ctx = self.model.tp_context

        # Select THIS layer's geometry before any get_split() call. tp_split() runs at load
        # time, when geometry_key_active is still unset, so without this every layer would be
        # split with the first geometry's ratios.
        ctx.set_geometry(self.layer_idx)

        if self.pre_layernorm is not None:
            self.pre_layernorm.tp_split(BROADCAST_KV)
        if self.post_layernorm is not None:
            self.post_layernorm.tp_split(BROADCAST_KV)

        self.q_proj.tp_split(BROADCAST_Q, dim = self.head_dim)
        self.k_proj.tp_split(BROADCAST_KV, dim = self.head_dim)
        if self.v_proj is not None:
            self.v_proj.tp_split(BROADCAST_KV, dim = self.head_dim)

        # o_proj is column-parallel only when its out_features equals num_heads * head_dim.
        # Gemma-4 projects 8192 (or 16384) back down to hidden_size 5376, so it must be
        # row-parallel instead: split in_features by the Q split and sum the partial outputs
        # across devices. PLAN.md §2.5.
        self.o_proj_row_parallel = (
            self.o_proj.out_features != self.num_attention_heads * self.head_dim
        )
        if self.o_proj_row_parallel:
            # Row-parallel (Megatron) o_proj: split the INPUT features across devices and
            # all-reduce the partial sums, instead of replicating the whole projection on the
            # primary device.
            #
            # The old code called tp_replicate(ctx.device) on the theory that q_perm permutes
            # the OUTPUT channels, which would make a head-aligned row split impossible. That
            # is wrong: q_perm has numel == in_features (8192 for the head_dim-256 layers,
            # 16384 for the head_dim-512 ones) and the GEMM applies it as
            # `a0 = a_ptr[b_q_perm[offset_k + t]]`, i.e. it permutes the INPUT dimension.
            # An input-dim split is therefore clean -- it just has to re-apply the permutation
            # to the gathered attention output, which `_o_proj_row_parallel` does.
            #
            # Verified numerically: gathering the input by q_perm and running the shard with
            # q_perm = NULL reproduces the logical matmul to 3.4e-3 (quantisation noise; the
            # unsharded handle gives 4.1e-3), whereas skipping the gather gives 1.36.
            #
            # Replicating was a hard serialization point in all 60 layers: every layer
            # gathered all attention output to device 0, ran the full 5376x5376 GEMM there
            # alone, then broadcast back. Device 0 measured 19.43 ms/token against 12.88 on
            # the others (1.67x max/mean).
            #
            # Keep the full permutation: tp_split_row() drops q_perm from each shard, so the
            # forward path needs the original to build each device's column index.
            self._o_proj_q_perm = self.o_proj.q_tensors["q_perm"].to(torch.int64).clone()
            self.o_proj.tp_split_row(BROADCAST_Q, BROADCAST_Q, dim = self.head_dim)
            # scratch_space_tp() populated out_features_tp before this ran. forward()
            # dispatches on out_features_tp FIRST, so leaving it set would route the layer
            # into the column-parallel forward_tp() and ignore the row split entirely.
            self.o_proj.out_features_tp = []
        else:
            self.o_proj.tp_split(BROADCAST_Q, dim = self.head_dim)

        maxrows = cfg.max_batch_size * cfg.max_input_len
        dtype = torch.half

        ctx.begin_scratch_alloc_tp()
        ctx.reserve_scratch(self.tp_dq_size)
        self.temp_bc0 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_bc1 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_bc2 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_q = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_k = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_KV, dim = self.head_dim)
        self.temp_v = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_KV, dim = self.head_dim)
        self.temp_o = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)

        self.is_tp = True
        self.set_device_idx(None)


    def tp_load_rank(self, rank: int):
        """
        Load only this attention block's slice for device ``rank``.

        Mirrors :meth:`tp_split` step for step, but each projection is built through its rank-local
        loader instead of being loaded in full and then split, and only this rank's scratch buffers
        are allocated.

        The geometry has to be selected first, exactly as in tp_split: get_split() returns the split
        for the *active* geometry, and Gemma-4 mixes head_dim 256 (16 kv heads) with head_dim 512
        (4 kv heads). Selecting the wrong one cuts k_proj into the wrong number of slices and the
        GEMM reads past the end of the weight buffer (Stage L bug 21).
        """
        cfg = self.model.config
        ctx = self.model.tp_context

        ctx.set_geometry(self.layer_idx)

        if self.pre_layernorm is not None:
            self.pre_layernorm.tp_load_rank(rank, BROADCAST_KV)
        if self.post_layernorm is not None:
            self.post_layernorm.tp_load_rank(rank, BROADCAST_KV)

        # BUG 67: the per-head norms are loaded by module.load(), which the rank-local path
        # replaces. Without this q_norm/k_norm/v_norm stay None and forward_tp_old dies in
        # _norm_weight_slice. ExLlamaV2HeadNorm.load() takes no device argument.
        for _n in (self.q_norm, self.k_norm, self.v_norm):
            if _n is not None:
                # BUG 68: device_idx must exist before load(); load_multi() reads device().
                _n.set_device_idx(rank)
                _n.load()
                if _n.weight is not None:
                    _n.weight.data = _n.weight.data.to(rank, non_blocking = False)
                if _n.bias is not None:
                    _n.bias.data = _n.bias.data.to(rank, non_blocking = False)

        self.q_proj.tp_load_rank(rank, BROADCAST_Q, dim = self.head_dim)
        self.k_proj.tp_load_rank(rank, BROADCAST_KV, dim = self.head_dim)
        if self.v_proj is not None:
            self.v_proj.tp_load_rank(rank, BROADCAST_KV, dim = self.head_dim)

        # o_proj is column-parallel only when its out_features equals num_heads * head_dim.
        # Gemma-4 projects 8192 (or 16384) back down to hidden_size 5376, so it is row-parallel
        # instead. Same test as tp_split.
        self.o_proj_row_parallel = (
            self.o_proj.out_features != self.num_attention_heads * self.head_dim
        )
        if self.o_proj_row_parallel:
            # Keep the full permutation: tp_load_rank_row drops q_perm from the shard, so the forward
            # path needs the original to build each device's column index.
            perm = self.o_proj.tp_load_rank_row(
                rank, BROADCAST_Q, BROADCAST_Q, dim = self.head_dim
            )
            self._o_proj_q_perm = perm.to(torch.int64)
            self.o_proj.out_features_tp = []
        else:
            self.o_proj.tp_load_rank(rank, BROADCAST_Q, dim = self.head_dim)

        # Only this rank's scratch buffers. scratch_space_tp() has already populated the
        # per-device size lists during load_tp_rank, so the sizes are known.
        maxrows = cfg.max_batch_size * cfg.max_input_len
        dtype = torch.half

        ctx.begin_scratch_alloc_tp()
        ctx.reserve_scratch(self.tp_dq_size)
        self.temp_bc0 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_bc1 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_bc2 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_q = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_k = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_KV, dim = self.head_dim)
        self.temp_v = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_KV, dim = self.head_dim)
        self.temp_o = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)

        self.is_tp = True
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

        amax(self.q_proj.scratch_space_tp(BROADCAST_Q, self.head_dim))
        amax(self.k_proj.scratch_space_tp(BROADCAST_KV, self.head_dim))
        if self.v_proj is not None:
            amax(self.v_proj.scratch_space_tp(BROADCAST_KV, self.head_dim))
        amax(self.o_proj.scratch_space_tp(BROADCAST_Q, self.head_dim))
        self.tp_dq_size = [s for s in scratch]

        maxrows = cfg.max_batch_size * cfg.max_input_len

        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))
        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))
        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_KV, dim = self.head_dim))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_KV, dim = self.head_dim))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))

        return scratch