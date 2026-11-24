from __future__ import annotations

def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

from exllamav2.architecture import RopeStyle
from exllamav2 import rope

import torch
import math


global_streams = {}


def set_device_streams():
    global global_streams
    for(k, v) in global_streams.items():
        with torch.cuda.device(torch.device(k)):
            torch.cuda.set_device(torch.device(k))
            torch.cuda.set_stream(v)


def get_device_stream(index: int):
    global global_streams
    return global_streams.get(index)


class ExLlamaV2DeviceContext:

    model: ExLlamaV2
    device_idx: int
    ready: bool

    scratch_bytes: int
    scratch_idx: int

    sin: list[torch.Tensor] | None
    cos: list[torch.Tensor] | None
    local_sin: list[torch.Tensor] | None
    local_cos: list[torch.Tensor] | None

    scratch: torch.Tensor | None

    stream: torch.cuda.Stream

    def __init__(
        self,
        model: ExLlamaV2,
        device_idx: int,
        scratch_bytes: int,
        archparams = None
    ):
        self.model = model
        self.device_idx = device_idx
        self.ready = False
        self.scratch = None
        self.scratch_bytes = scratch_bytes
        self.scratch_idx = 0
        self.archparams = archparams or model.config.arch.lm

        # Create streams (only one per device)

        if device_idx not in global_streams:
            s = torch.cuda.Stream(torch.device(device_idx), -100)
            global_streams[device_idx] = s

        self.stream = global_streams[device_idx]

        xx = 0


    def prepare(self, scratch):

        self.prepare_sincos()

        if scratch:
            self.scratch = torch.empty((self.scratch_bytes // 2,), dtype = torch.half, device = _torch_device(self.device_idx))

        self.ready = True


    def drop(self):

        self.scratch = None
        self.sin = None
        self.cos = None
        self.ready = False


    def free(self):

        self.drop()
        self.scratch_bytes = 1


    def begin_scratch_alloc(self):

        self.scratch_idx = 0


    def get_scratch_slice(self, size_bytes):

        if self.scratch is None: self.prepare(True)

        size_bytes = ((size_bytes + 63) // 64) * 64
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, self.scratch_idx, size_half)
        self.scratch_idx += size_half
        return scratch_slice


    def prepare_sincos(self):
        """
        Build the sin/cos tables for each rope variant.

        Table index 0 is the base table (theta = ``rotary_embedding_base``); index 1 is the alt
        table (theta = ``rotary_embedding_base_alt``) when present. ``model.py`` selects index 1
        for sliding layers via ``rope_index``.

        For Gemma-4 the base table serves the full-attention layers and must therefore be built
        with ``global_head_dim`` (512) and *proportional* rope, while the alt table serves the
        sliding layers with ``head_dim`` (256) and default rope. PLAN.md §2.2 step 4, §2.3.
        """

        device = _torch_device(self.device_idx)
        cfg = self.model.config

        # Per-table geometry. `table_head_dim` is the *full* head dim the table must span, and
        # `rope_type` selects the frequency computation.
        #
        # TRAP (PLAN.md §8 trap 1): the sin/cos tensor width must be the full head_dim (512), NOT
        # head_dim * partial_rotary_factor (128). The CUDA rope kernel derives `half_dim` from
        # sincos_size and reads `column + half_dim`, so a 128-wide table makes it rotate the wrong
        # span and silently corrupts the 512-dim layers.

        # Hybrid linear-attention backbones (Qwen3.5) use layer_types for linear/full
        # routing, not a second attention geometry: one default-NEOX table spanning
        # head_dim * partial_rotary_factor (64 of 256) serves every full layer. The rope
        # kernel pairs (i, i + sincos_size/2), so a 64-wide table rotates exactly the
        # first 64 dims, matching HF Qwen3_5TextRotaryEmbedding. newplan.md P6.
        _hybrid = getattr(cfg, "hybrid_linear_attn", False)
        has_global = getattr(cfg, "layer_types", None) is not None and not _hybrid

        thetas = [cfg.rotary_embedding_base]
        scales = [cfg.scale_pos_emb]
        if has_global:
            _base_head_dim = cfg.global_head_dim
        elif _hybrid:
            _base_head_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
        else:
            _base_head_dim = cfg.head_dim
        table_head_dims = [_base_head_dim]
        table_rope_types = [cfg.rope_type_global if has_global else None]
        table_rope_props = [cfg.rope_partial_rotary_factor_global if has_global else None]

        if cfg.rotary_embedding_base_alt:
            thetas.append(cfg.rotary_embedding_base_alt)
            scales.append(cfg.scale_pos_emb_alt)
            table_head_dims.append(cfg.head_dim)
            table_rope_types.append(None)
            table_rope_props.append(None)

        self.sin = []
        self.cos = []

        for theta, lscale, t_head_dim, t_rope_type, t_rope_prop in zip(
            thetas, scales, table_head_dims, table_rope_types, table_rope_props
        ):

            if self.archparams.rope_style == RopeStyle.NONE:
                sin = torch.zeros((1,), device = device, dtype = torch.half)
                cos = sin
                self.sin.append(sin)
                self.cos.append(cos)
                break

            # RoPE params

            inv_freq, scaling_factor = rope.get_rope_params(
                device,
                cfg,
                theta,
                head_dim = t_head_dim,
                rope_type = t_rope_type,
                rope_proportion = t_rope_prop,
            )

            # Common

            scale = lscale or 1.0
            t = torch.arange(cfg.max_seq_len, device = device, dtype = torch.float32)
            if cfg.pos_id_index != 0:
                t += cfg.pos_id_index
            if scale != 1.0: t /= scale

            freqs = torch.einsum("i,j->ij", t, inv_freq)
            if self.archparams.rope_style == RopeStyle.NEOX:
                emb = torch.cat((freqs, freqs), dim=-1)
            elif self.archparams.rope_style == RopeStyle.GPTJ:
                emb = torch.repeat_interleave(freqs, 2, dim=-1)
            else:
                raise ValueError()

            # For proportional rope the zero-filled tail yields sin=0, cos=1, i.e. identity.
            # Assert the width matches the full head dim so a future refactor cannot silently
            # shrink the table.
            assert emb.shape[-1] == t_head_dim, \
                f"sincos table width {emb.shape[-1]} != head_dim {t_head_dim}"

            sin = emb.sin()[None, None, :, :]
            cos = emb.cos()[None, None, :, :]
            if scaling_factor != 1.0:
                sin *= scaling_factor
                cos *= scaling_factor
            sin = sin.half()
            cos = cos.half()

            self.sin.append(sin)
            self.cos.append(cos)