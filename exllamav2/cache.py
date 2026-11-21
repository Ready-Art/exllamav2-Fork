from __future__ import annotations

import torch
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.tensor_p import BROADCAST_KV

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2
    from exllamav2 import ExLlamaV2Tokenizer

class ExLlamaV2CacheBase:

    model = None
    max_seq_len: int
    batch_size: int

    current_seq_len: int

    key_states: list[torch.Tensor | None]
    value_states: list[torch.Tensor | None]
    key_scales: list[torch.Tensor | None]
    value_scales: list[torch.Tensor | None]
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int

    dtype: torch.dtype
    weights_per_element_k: int
    weights_per_element_v: int
    has_scales: bool
    fixed_device: int | None


    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        weights_per_element_k: int,
        weights_per_element_v: int,
        has_scales: bool,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):

        self.model = model
        self.max_seq_len = max_seq_len if max_seq_len != -1 else self.model.config.max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.weights_per_element_k = weights_per_element_k
        self.weights_per_element_v = weights_per_element_v
        self.has_scales = has_scales

        self.key_states = []
        self.value_states = []
        self.key_scales = []
        self.value_scales = []

        self.num_key_value_heads = num_key_value_heads or self.model.config.num_key_value_heads
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.head_dim = self.model.config.head_dim

        self.current_seq_len = 0

        # Per-layer attention geometry.
        #
        # exl2 previously built one `shape_basic` from `config.head_dim` and reused it for every
        # layer. Gemma-4 mixes head_dim 256 (sliding) and 512 (global) in one model, so the shapes
        # must be per layer or the 512-dim layers index out of bounds / read garbage.
        # PLAN.md §2.2 / §8 trap 3.
        #
        # `shape_basic` is kept as the *uniform* shape for backwards compatibility with callers that
        # assume it; `layer_shapes` is the authoritative per-layer list.
        self.layer_head_dims: list[int] = []
        self.layer_num_kv_heads: list[int] = []
        self.layer_shapes_basic: list[tuple] = []
        self.layer_shapes_wk: list[tuple] = []
        self.layer_shapes_wv: list[tuple] = []
        self.layer_shapes_s: list[tuple] = []

        for i in range(self.num_hidden_layers):
            if num_key_value_heads is not None:
                # Explicit override (e.g. TP sharding): uniform across layers.
                hd = self.head_dim
                kvh = num_key_value_heads
            else:
                hd, _, kvh, _ = self.model.config.get_layer_attn_geometry(i)
            self.layer_head_dims.append(hd)
            self.layer_num_kv_heads.append(kvh)
            self.layer_shapes_basic.append((self.batch_size, self.max_seq_len, kvh, hd))
            self.layer_shapes_wk.append((self.batch_size, self.max_seq_len, kvh, hd // self.weights_per_element_k))
            self.layer_shapes_wv.append((self.batch_size, self.max_seq_len, kvh, hd // self.weights_per_element_v))
            self.layer_shapes_s.append((self.batch_size, self.max_seq_len, kvh, hd // 32))

        # Uniform shapes, retained for callers that still assume a single geometry. These describe
        # the *first* layer and are only correct for uniform-geometry models.
        self.shape_basic = self.layer_shapes_basic[0]
        self.shape_wk = self.layer_shapes_wk[0]
        self.shape_wv = self.layer_shapes_wv[0]
        self.shape_s = self.layer_shapes_s[0]

        self.q_block = 0
        self.fixed_device = fixed_device


    def create_state_tensors(
        self,
        copy_from: ExLlamaV2CacheBase | None,
        lazy = False
    ):
        assert copy_from is None or lazy == False, "Cannot use lazy cache initialization while copying"

        if copy_from:
            self.current_seq_len = copy_from.current_seq_len

        if not lazy:

            for i in range(self.num_hidden_layers):

                if copy_from is None:
                    device = self.model.cache_map.get(i, self.fixed_device)
                    p_key_states = torch.zeros(self.layer_shapes_wk[i], dtype = self.dtype, device = device).contiguous()
                    p_value_states = torch.zeros(self.layer_shapes_wv[i], dtype = self.dtype, device = device).contiguous()
                    if self.has_scales:
                        p_key_scales = torch.zeros(self.layer_shapes_s[i], dtype = torch.float16, device = device).contiguous()
                        p_value_scales = torch.zeros(self.layer_shapes_s[i], dtype = torch.float16, device = device).contiguous()
                else:
                    p_key_states = copy_from.key_states[i].clone()
                    p_value_states = copy_from.value_states[i].clone()
                    if self.has_scales:
                        p_key_scales = copy_from.key_scales[i].clone()
                        p_value_scales = copy_from.value_scales[i].clone()

                self.key_states.append(p_key_states)
                self.value_states.append(p_value_states)
                if self.has_scales:
                    self.key_scales.append(p_key_scales)
                    self.value_scales.append(p_value_scales)

        else:

            for i in range(self.num_hidden_layers):

                self.key_states.append(None)
                self.value_states.append(None)
                if self.has_scales:
                    self.key_scales.append(None)
                    self.value_scales.append(None)


    def update_cache_tensors(self):

        for k, v in self.model.cache_map.items():

            self.touch_device(v)

            if self.key_states[k] is not None:

                if str(self.key_states[k].device) == v: continue
                self.key_states[k] = None
                self.value_states[k] = None

            p_key_states = torch.zeros(self.layer_shapes_wk[k], dtype = self.dtype, device = v).contiguous()
            p_value_states = torch.zeros(self.layer_shapes_wv[k], dtype = self.dtype, device = v).contiguous()
            self.key_states[k] = p_key_states
            self.value_states[k] = p_value_states
            if self.has_scales:
                p_key_scales = torch.zeros(self.layer_shapes_s[k], dtype = torch.float16, device = v).contiguous()
                p_value_scales = torch.zeros(self.layer_shapes_s[k], dtype = torch.float16, device = v).contiguous()
                self.key_scales[k] = p_key_scales
                self.value_scales[k] = p_value_scales


    def roll_left(self):

        # NOTE: this is the *non-paged* full-cache roll. It must not be used for the SWA ring, which
        # needs a page-granular shift and its own position bookkeeping. PLAN.md §8 trap 15.
        for i in range(self.model.config.num_hidden_layers):

            self.key_states[i] = torch.roll(self.key_states[i], shifts = -1, dims = 2)
            self.value_states[i] = torch.roll(self.value_states[i], shifts = -1, dims = 2)

        self.current_seq_len -= 1


    def get_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None
    ) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError


    def store_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None
    ):
        raise NotImplementedError


    @torch.inference_mode
    def copy_states(
        self,
        target: ExLlamaV2CacheBase,
        from_column: int,
        from_columns: int,
        to_column: int,
        to_columns: int,
        from_row: int,
        from_rows: int,
        to_row: int,
        to_rows: int
    ):
        assert from_rows == 1
        assert from_columns == to_columns
        assert to_column + to_columns <= target.max_seq_len
        assert from_column + from_columns <= self.max_seq_len

        num_hidden_layers = self.model.config.num_hidden_layers

        tensors = [
            (self.key_states, target.key_states),
            (self.value_states, target.value_states),
        ]
        if self.has_scales:
            tensors += [
                (self.key_scales, target.key_scales),
                (self.value_scales, target.value_scales),
            ]

        for i in range(num_hidden_layers):
            for (src, dst) in tensors:
                src_view = src[i].narrow(0, from_row, from_rows).narrow(1, from_column, from_columns)
                dst_view = dst[i].narrow(0, to_row, to_rows).narrow(1, to_column, to_columns)
                if to_rows > 1:
                    src_view = src_view.expand_as(dst_view)
                dst_view.copy_(src_view, non_blocking = True)


    def touch_device(self, device):
        pass


    def all_tensors(self):
        raise NotImplementedError()


    def reset(self):
        self.current_seq_len = 0


class ExLlamaV2Cache(ExLlamaV2CacheBase):
    """
    FP16 cache
    """

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):
        super().__init__(
            model,
            batch_size,
            max_seq_len,
            torch.half,
            1, 1,
            False,
            num_key_value_heads,
            fixed_device
        )

        self.create_state_tensors(copy_from, lazy)


    def get_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None
    ) -> (torch.Tensor, torch.Tensor):

        return self.key_states[layer_idx], self.value_states[layer_idx]


    def store_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None
    ):
        pass


    def footprint(self):

        fp = []
        for layer in self.key_states + self.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 2
        return fp


    def clone(self):

        new = ExLlamaV2Cache(self.model, self.batch_size, self.max_seq_len, self)
        return new


    def all_tensors(self):
        return self.key_states + self.value_states


class ExLlamaV2Cache_SWA(ExLlamaV2Cache):
    """
    FP16 cache with per-layer ring buffers for sliding-window attention layers.

    Without this, Gemma-4-31B allocates a full ``max_seq_len`` KV buffer for all 60 layers, including
    the 50 sliding layers that can only ever attend to the last 1024 tokens. At 256k context that is
    ~236 GB; with the ring it is ~22.5 GB. This is not an optimization, it is the difference between
    the model running and not running. PLAN.md §4.0.

    Ring size per sliding layer:

        kv_state_size = ceil((sliding_window + overp) / page_size) * page_size

    With page_size 128, sliding_window 1024, overp 256: ``ceil(1280/128)*128 = 1280`` tokens.

    Full-attention layers keep the ordinary full-length buffer. ``key_states`` / ``value_states`` are
    therefore a list of differently-shaped tensors, which is why the per-layer shapes in
    ``ExLlamaV2CacheBase`` matter.

    The ring is a *separate allocation* from the paged cache: it is per-layer same-size, whereas the
    page table is one table covering all layers. Do not try to fit it into ``allocated_pages``.
    PLAN.md §8 trap 16.

    Speculative-decoding rollback (``rewind`` / ``stash`` / ``unstash``) is deliberately NOT
    implemented: speculative decoding is hard-blocked (§5), so §4.4 rules it out of scope.
    """

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache | None = None,
        lazy: bool = False,
        page_size: int = 128,
        fixed_device: int | None = None
    ):
        cfg = model.config

        self.page_size = page_size
        # Full-attention KV depth must be a whole number of pages: the graphed FI verify
        # wrappers guard depth % page == 0 and silently fall back to torch attention on
        # unaligned depths. The ring depth is already page-rounded below; round the full
        # depth here too (<= page-1 extra rows, negligible VRAM).
        if max_seq_len == -1:
            max_seq_len = cfg.max_seq_len
        max_seq_len = ((max_seq_len + page_size - 1) // page_size) * page_size
        self.sliding_window = cfg.sliding_window

        overp = cfg.sliding_window_overp
        if overp is None:
            # The ring must hold the sliding window PLUS a full prefill chunk: the chunk
            # loop shifts the ring so the new chunk fits, and after the shift the retained
            # prefix is ring - chunk. With overp < chunk that prefix is shorter than the
            # window, silently truncating SWA attention for the first (window - prefix)
            # rows of every chunk (chunk_size == ring was the worst case: NO prefix at
            # all). Default overp = max_input_len keeps the full window for every row at
            # the configured chunk size; model.forward additionally caps the chunk at
            # ring - window + 1 so an explicit small overp can never truncate.
            overp = max(2 * page_size, cfg.max_input_len)
        self.sliding_window_overp = overp

        # Ring size, rounded up to a whole number of pages. This is required: the shift must move a
        # whole number of pages or the block table silently points at the wrong slots.
        # PLAN.md §8 trap 13 / trap 17.
        raw = self.sliding_window + overp
        self.kv_state_size = ((raw + page_size - 1) // page_size) * page_size
        assert self.kv_state_size % page_size == 0, \
            f"kv_state_size {self.kv_state_size} must be a multiple of page_size {page_size}"

        # Allocate lazily: the base class would otherwise create a full max_seq_len buffer for
        # every layer, and only afterwards would the sliding layers' buffers be replaced by rings.
        # At 64k that transient peak is ~59 GB. `_allocate_rings()` below creates the rings, and
        # `create_state_tensors()` is called afterwards for the full-attention layers only.
        super().__init__(
            model,
            batch_size,
            max_seq_len,
            copy_from,
            True,
            None,
            fixed_device,
        )

        # Replace the sliding layers' full-length buffers with rings.
        self.is_swa_layer: list[bool] = []
        for i in range(self.num_hidden_layers):
            _, _, _, is_full = cfg.get_layer_attn_geometry(i)
            self.is_swa_layer.append(not is_full and self.sliding_window > 0)

        # BUG 94: absolute position of the ring's first slot. The streaming/TP path
        # (model.forward chunk loop + forward_tp_old) shifts the ring by this amount
        # so a chunk that would overflow it is shifted first, mirroring the paged
        # generator's window_beg protocol. Zero at construction and on reset.
        self.window_beg = 0

        if not lazy:
            self._allocate_rings()
            # Now allocate the full-length buffers for the non-sliding layers only.
            self._allocate_full_attention()

    def _allocate_rings(self):

        for i in range(self.num_hidden_layers):
            if not self.is_swa_layer[i]:
                continue

            device = self.model.cache_map.get(i, self.fixed_device)
            kvh = self.layer_num_kv_heads[i]
            hd = self.layer_head_dims[i]

            self.key_states[i] = torch.zeros(
                (self.batch_size, self.kv_state_size, kvh, hd),
                dtype = self.dtype, device = device
            ).contiguous()
            self.value_states[i] = torch.zeros(
                (self.batch_size, self.kv_state_size, kvh, hd),
                dtype = self.dtype, device = device
            ).contiguous()

    def update_cache_tensors(self):
        """
        Lazy/autosplit allocation: the base implementation allocates full-depth buffers
        for every layer, which defeats the ring (and crashes shift_ring, whose copy
        shapes assume ring-depth buffers). Allocate rings for sliding layers and
        full-length buffers for the rest, mirroring _allocate_rings/_allocate_full_attention.
        """

        for k, v in self.model.cache_map.items():

            self.touch_device(v)

            if self.key_states[k] is not None:
                if str(self.key_states[k].device) == v:
                    continue
                self.key_states[k] = None
                self.value_states[k] = None

            if self.is_swa_layer[k]:
                kvh = self.layer_num_kv_heads[k]
                hd = self.layer_head_dims[k]
                self.key_states[k] = torch.zeros(
                    (self.batch_size, self.kv_state_size, kvh, hd),
                    dtype = self.dtype, device = v
                ).contiguous()
                self.value_states[k] = torch.zeros(
                    (self.batch_size, self.kv_state_size, kvh, hd),
                    dtype = self.dtype, device = v
                ).contiguous()
            else:
                self.key_states[k] = torch.zeros(
                    self.layer_shapes_wk[k], dtype = self.dtype, device = v
                ).contiguous()
                self.value_states[k] = torch.zeros(
                    self.layer_shapes_wv[k], dtype = self.dtype, device = v
                ).contiguous()

    def _allocate_full_attention(self):
        """Allocate full-length KV buffers for the non-sliding layers only."""
        for i in range(self.num_hidden_layers):
            if self.is_swa_layer[i]:
                continue
            device = self.model.cache_map.get(i, self.fixed_device)
            kvh = self.layer_num_kv_heads[i]
            hd = self.layer_head_dims[i]
            self.key_states[i] = torch.zeros(
                self.layer_shapes_wk[i], dtype = self.dtype, device = device
            ).contiguous()
            self.value_states[i] = torch.zeros(
                self.layer_shapes_wv[i], dtype = self.dtype, device = device
            ).contiguous()

    def reset(self):
        self.current_seq_len = 0
        self.window_beg = 0

    def ring_size(self, layer_idx: int) -> int:
        """Number of tokens the ring for this layer can hold (0 for full-attention layers)."""
        return self.kv_state_size if self.is_swa_layer[layer_idx] else 0

    @torch.inference_mode()
    def shift_ring(self, layer_idx: int, shift_tokens: int):
        """
        Shift the ring left by ``shift_tokens``, which MUST be a whole number of pages.

        Mirrors ``_decode_state_prep`` in ``exllamav3/modules/sliding_attn.py:929-941``.

        Shifting by a non-page-aligned token count makes the block table point at the wrong slots and
        degrades output gradually rather than failing loudly, so this asserts.
        PLAN.md §8 trap 13.
        """

        assert self.is_swa_layer[layer_idx], f"Layer {layer_idx} is not a sliding layer"
        assert shift_tokens > 0, "shift_tokens must be positive"
        assert shift_tokens % self.page_size == 0, \
            f"Ring shift must be a whole number of pages, got {shift_tokens} tokens " \
            f"with page_size {self.page_size}"
        assert shift_tokens < self.kv_state_size, \
            f"Cannot shift the ring by {shift_tokens} of {self.kv_state_size} tokens"

        k = self.key_states[layer_idx]
        v = self.value_states[layer_idx]

        n = self.kv_state_size
        k[:, :n - shift_tokens].copy_(k[:, shift_tokens:].clone())
        v[:, :n - shift_tokens].copy_(v[:, shift_tokens:].clone())

    @torch.inference_mode()
    def shift_ring_row(self, layer_idx: int, row: int, shift_tokens: int):
        """Multi-batch: shift ONE sequence's ring (batch row ``row``) left by a whole
        number of pages. Same contract as shift_ring, scoped to a single row so slots
        at different sequence lengths shift independently. Every TP rank computes the
        same per-row shift from the shared (seq_lens, window_begs) bookkeeping, so the
        rings stay in lockstep with no cross-rank communication (BUG 94 protocol)."""

        assert self.is_swa_layer[layer_idx], f"Layer {layer_idx} is not a sliding layer"
        assert shift_tokens > 0, "shift_tokens must be positive"
        assert shift_tokens % self.page_size == 0, \
            f"Ring shift must be a whole number of pages, got {shift_tokens} tokens " \
            f"with page_size {self.page_size}"
        assert shift_tokens < self.kv_state_size, \
            f"Cannot shift the ring by {shift_tokens} of {self.kv_state_size} tokens"

        k = self.key_states[layer_idx][row]
        v = self.value_states[layer_idx][row]

        n = self.kv_state_size
        k[:n - shift_tokens].copy_(k[shift_tokens:].clone())
        v[:n - shift_tokens].copy_(v[shift_tokens:].clone())

    @torch.inference_mode()
    def reset_rings(self):
        """Clear every ring. Called on sequence reset. PLAN.md §4.3 Phase 3E item 3."""
        for i in range(self.num_hidden_layers):
            if not self.is_swa_layer[i]:
                continue
            if self.key_states[i] is not None:
                self.key_states[i].zero_()
                self.value_states[i].zero_()

    def footprint(self):

        # Sum actual per-layer tensor sizes rather than assuming a uniform shape.
        # PLAN.md §4.3 Phase 3A item 3.
        fp = []
        for layer in self.key_states + self.value_states:
            if layer is None:
                continue
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 2
        return fp

    def clone(self):
        return ExLlamaV2Cache_SWA(
            self.model,
            self.batch_size,
            self.max_seq_len,
            self,
            page_size = self.page_size,
        )

    def roll_left(self):
        """
        Full-cache roll for the non-paged path. SWA layers are rolled by one token like the rest, but
        the paged path must NOT use this for the ring: the ring needs a page-granular shift and its own
        position bookkeeping. PLAN.md §8 trap 15.
        """
        for i in range(self.num_hidden_layers):
            if self.key_states[i] is None:
                continue
            self.key_states[i] = torch.roll(self.key_states[i], shifts = -1, dims = 1)
            self.value_states[i] = torch.roll(self.value_states[i], shifts = -1, dims = 1)
        self.current_seq_len -= 1


class ExLlamaV2Cache_8bit(ExLlamaV2CacheBase):
    """
    8-bit cache. Keys and values are compressed to FP8 (e5m2) format by truncation.
    """

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_8bit | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):

        super().__init__(
            model,
            batch_size,
            max_seq_len,
            torch.uint8,
            1, 1,
            False,
            num_key_value_heads,
            fixed_device
        )

        self.create_state_tensors(copy_from, lazy)

        # Create temp FP16 tensors for accessing FP8 layers

        self.temp_tensors = {}
        if not lazy:
            devs = self.model.get_cache_devices() if self.fixed_device is None else [self.fixed_device]
            for device in devs: self.touch_device(device)


    def touch_device(self, device):

        if device in self.temp_tensors: return
        k = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        v = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        self.temp_tensors[device] = (k, v)


    def get_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None
    ) -> (torch.Tensor, torch.Tensor):

        device = self.model.cache_map.get(layer_idx, self.fixed_device)

        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width > 0: ext_c.fp8_to_fp16(self.key_states[layer_idx], temp_key_state, batch_size, offset, width)
        if width > 0: ext_c.fp8_to_fp16(self.value_states[layer_idx], temp_value_state, batch_size, offset, width)
        return temp_key_state, temp_value_state


    def store_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None
    ):
        device = self.model.cache_map.get(layer_idx, self.fixed_device)

        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width > 0: ext_c.fp16_to_fp8(temp_key_state, self.key_states[layer_idx], batch_size, offset, width)
        if width > 0: ext_c.fp16_to_fp8(temp_value_state, self.value_states[layer_idx], batch_size, offset, width)


    def footprint(self) -> list[int]:

        fp = []
        for layer in self.key_states + self.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 1
        for temp_k, temp_v in self.temp_tensors.values():
            fp[temp_k.device.index] += temp_k.numel() * 2
            fp[temp_v.device.index] += temp_v.numel() * 2
        return fp


    def clone(self) -> ExLlamaV2Cache_8bit:
        new = ExLlamaV2Cache_8bit(self.model, self.batch_size, self.max_seq_len, self)
        return new


    def all_tensors(self):
        return self.key_states + self.value_states


class ExLlamaV2Cache_Q(ExLlamaV2CacheBase):
    """
    Q cache. Uses grouped RTN quantization for keys/values
    """

    wbits: int

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_Q4 | None = None,
        lazy: bool = False,
        weights_per_byte_k: int = -1,
        weights_per_byte_v: int = -1,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):
        super().__init__(
            model,
            batch_size,
            max_seq_len,
            torch.uint8,
            weights_per_byte_k,
            weights_per_byte_v,
            True,
            num_key_value_heads,
            fixed_device
        )
        cfg = self.model.config

        self.create_state_tensors(copy_from, lazy)

        # Models with odd key/value dims need to quantize/dequantize in multi-token blocks. Make sure the quant
        # blocksize aligns with a whole number of tokens

        if not num_key_value_heads:
            num_key_value_heads = cfg.num_key_value_heads

        Q_CACHE_BLOCKSIZE_Q = 512
        kv_dim = num_key_value_heads * cfg.head_dim
        self.q_block = 1
        while (kv_dim * self.q_block) % Q_CACHE_BLOCKSIZE_Q:
            self.q_block += 1
        self.max_seq_len = (self.max_seq_len + self.q_block - 1) // self.q_block * self.q_block

        # Create temp FP16 tensors for accessing Q4 layers

        self.temp_tensors = {}
        if not lazy:
            devs = self.model.get_cache_devices() if self.fixed_device is None else [self.fixed_device]
            for device in devs: self.touch_device(device)


    def touch_device(self, device):

        if device in self.temp_tensors: return
        k = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        v = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        self.temp_tensors[device] = (k, v)


    def get_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None
    ) -> (torch.Tensor, torch.Tensor):

        device = self.model.cache_map.get(layer_idx, self.fixed_device)

        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width == 0: return temp_key_state, temp_value_state

        if self.q_block > 1 and not page_size:
            a = offset
            b = offset + width
            a = a // self.q_block * self.q_block
            b = (b + self.q_block - 1) // self.q_block * self.q_block
            offset = a
            width = b - a

        ext_c.q_to_fp16_kv(
            self.key_states[layer_idx],
            temp_key_state,
            self.key_scales[layer_idx],
            self.value_states[layer_idx],
            temp_value_state,
            self.value_scales[layer_idx],
            batch_size,
            offset,
            width,
            page_size,
            cache_seqlens if cache_seqlens is not None else none_tensor,
            block_table if block_table is not None else none_tensor,
            # none_tensor,
            # none_tensor
            self.wbits
        )

        return temp_key_state, temp_value_state


    def store_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None
    ):
        if width == 0: return

        if self.q_block > 1 and not page_size:
            a = offset
            b = offset + width
            a = a // self.q_block * self.q_block
            b = (b + self.q_block - 1) // self.q_block * self.q_block
            offset = a
            width = b - a

        device = self.model.cache_map.get(layer_idx, self.fixed_device)
        temp_key_state, temp_value_state = self.temp_tensors[device]

        ext_c.fp16_to_q_kv(
            temp_key_state,
            self.key_states[layer_idx],
            self.key_scales[layer_idx],
            temp_value_state,
            self.value_states[layer_idx],
            self.value_scales[layer_idx],
            batch_size,
            offset,
            width,
            page_size,
            cache_seqlens if cache_seqlens is not None else none_tensor,
            block_table if block_table is not None else none_tensor,
            # none_tensor,
            # none_tensor
            self.wbits
        )


    def footprint(self) -> list[int]:

        fp = []
        for layer in self.key_states + self.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 1
        for layer in self.key_scales + self.value_scales:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 2
        for temp_k, temp_v in self.temp_tensors.values():
            fp[temp_k.device.index] += temp_k.numel() * 2
            fp[temp_v.device.index] += temp_v.numel() * 2
        return fp


    def clone(self) -> ExLlamaV2Cache_Q4:
        new = ExLlamaV2Cache_Q4(self.model, self.batch_size, self.max_seq_len, self)
        return new

    def all_tensors(self):
        return self.key_states + self.value_states + self.key_scales + self.value_scales


class ExLlamaV2Cache_Q4(ExLlamaV2Cache_Q):

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_Q4 | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):
        super().__init__(
            model,
            batch_size,
            max_seq_len,
            copy_from,
            lazy,
            2, 2,
            num_key_value_heads,
            fixed_device
        )
        self.wbits = 4


class ExLlamaV2Cache_Q6(ExLlamaV2Cache_Q):

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_Q6 | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):
        super().__init__(
            model,
            batch_size,
            max_seq_len,
            copy_from,
            lazy,
            1, 2,
            num_key_value_heads,
            fixed_device
        )
        self.wbits = 6


class ExLlamaV2Cache_Q8(ExLlamaV2Cache_Q):

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_Q8 | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):
        super().__init__(
            model,
            batch_size,
            max_seq_len,
            copy_from,
            lazy,
            1, 1,
            num_key_value_heads,
            fixed_device
        )
        self.wbits = 8


def tp_narrow_cache_to_rank(cache: ExLlamaV2CacheBase, tp_context, rank: int):
    """
    Narrow an already-constructed cache to one rank's slice of the KV split, in place.

    A cache built from ``model.config`` has the FULL per-layer KV head count, so on a worker it
    would allocate the whole model's KV rather than its share. This rewrites each layer's head
    count and shapes to the slice ``rank`` owns under that layer's own geometry, then rebuilds the
    state tensors.

    Per-layer geometry matters: Gemma-4's sliding layers have 16 kv heads and split 8/8 on two
    devices, while its full-attention layers have only 4 and split 2/2. A single split cannot
    serve both -- device 1 would get 0 heads on the 4-head layers and end up with a zero-width
    buffer.

    The state tensors are rebuilt lazily and then only the rings and full-attention buffers are
    allocated. Calling ``create_state_tensors(lazy = False)`` here would transiently allocate a full
    ``max_seq_len`` buffer for every layer, including the sliding ones, which OOMs at 64k.

    :param cache:
        Cache to narrow. Must have been built with ``lazy = True`` so nothing is allocated yet.

    :param tp_context:
        The model's TPContext, used for the per-geometry KV splits.

    :param rank:
        Device index this cache belongs to.
    """
    for li in range(cache.num_hidden_layers):
        kvh = cache.layer_num_kv_heads[li]
        if kvh <= 0:
            continue
        geo = tp_context.geometry_keys[li]
        sp = tp_context.geometry_splits[geo]["kv"][0]
        la = lb = 0
        for d, a, b in sp:
            if d == rank:
                la, lb = a, b
                break
        n = max(0, lb - la)
        cache.layer_num_kv_heads[li] = n
        hd = cache.layer_head_dims[li]
        cache.layer_shapes_basic[li] = (cache.batch_size, cache.max_seq_len, n, hd)
        cache.layer_shapes_wk[li] = (cache.batch_size, cache.max_seq_len, n,
                                     hd // cache.weights_per_element_k)
        cache.layer_shapes_wv[li] = (cache.batch_size, cache.max_seq_len, n,
                                     hd // cache.weights_per_element_v)
        cache.layer_shapes_s[li] = (cache.batch_size, cache.max_seq_len, n, hd // 32)

    cache.key_states = []
    cache.value_states = []
    cache.key_scales = []
    cache.value_scales = []
    cache.create_state_tensors(None, True)
    if hasattr(cache, "_allocate_rings"):
        cache._allocate_rings()
        cache._allocate_full_attention()
    return cache


class _RankCacheView(ExLlamaV2CacheBase):
    """
    Thin wrapper making a single-rank cache look like ExLlamaV2Cache_TP.

    The TP forward paths index the KV state per device (``k_cache[idx]``). A rank-local worker
    has exactly one device, so the underlying cache's bare (k, v) tensors are wrapped in a
    one-element list. BUG 69.

    BUG 96: must subclass ExLlamaV2CacheBase, otherwise model.forward takes the no-cache
    branch and skips the chunked streaming path (and BUG 94's SWA ring handling); prefills
    longer than config.max_input_len (2048) then trip the "Maximum input length exceeded"
    assert. The 1k MP bench only passed because 1028 <= 2048. Attribute access delegates to
    the wrapped cache via __getattr__; the base class adds no instance attributes, so nothing
    is shadowed.
    """

    def __init__(self, cache):
        self._cache = cache
        # Multi-batch bookkeeping (MP slot scheduler): per-row sequence length and SWA
        # ring origin. The scalar current_seq_len / window_beg fields remain for the
        # verified batch-1 path; batched forwards read these lists instead.
        cache.seq_lens = [0] * cache.batch_size
        cache.window_begs = [0] * cache.batch_size
        cache._row_views = {}

    def __getattr__(self, name):
        return getattr(self._cache, name)

    def get_kv_state(self, layer_idx, batch_size, offset, width, page_size = 0,
                     cache_seqlens = None, block_table = None):
        k, v = self._cache.get_kv_state(layer_idx, batch_size, offset, width,
                                        page_size, cache_seqlens, block_table)
        return [k], [v]

    def row_view(self, row: int):
        """Persistent row-scoped facade (stable identity for graph-cache checks)."""
        v = self._row_views.get(row)
        if v is None:
            v = self._row_views[row] = _RowCacheView(self, row)
        return v


class _RowCacheView(ExLlamaV2CacheBase):
    """
    One batch row of a rank cache, presented as a batch-1 cache.

    The verified scalar paths (eager prefill chunk loop, streaming generator, ring
    snapshot/restore) read ``current_seq_len`` / ``window_beg`` and slice KV with
    ``[:batch_size]``. Narrowing every state tensor to row ``r`` and mapping the
    bookkeeping to ``seq_lens[r]`` / ``window_begs[r]`` lets all of that code run
    unchanged on a single slot of a multi-batch cache, with zero cross-slot writes:
    every view shares storage with the parent but only ever touches row r.
    """

    def __init__(self, rank_cache, row: int):
        self._rc = rank_cache
        self._row = row

    def __getattr__(self, name):
        return getattr(self._rc, name)

    @property
    def batch_size(self):
        return 1

    @property
    def current_seq_len(self):
        return self._rc.seq_lens[self._row]

    @current_seq_len.setter
    def current_seq_len(self, value):
        self._rc.seq_lens[self._row] = int(value)

    @property
    def window_beg(self):
        return self._rc.window_begs[self._row]

    @window_beg.setter
    def window_beg(self, value):
        self._rc.window_begs[self._row] = int(value)

    @property
    def key_states(self):
        return [t[self._row:self._row + 1] if t is not None else None
                for t in self._rc.key_states]

    @property
    def value_states(self):
        return [t[self._row:self._row + 1] if t is not None else None
                for t in self._rc.value_states]

    def get_kv_state(self, layer_idx, batch_size, offset, width, page_size = 0,
                     cache_seqlens = None, block_table = None):
        ks, vs = self._rc.get_kv_state(layer_idx, batch_size, offset, width,
                                       page_size, cache_seqlens, block_table)
        r = self._row
        return [k[r:r + 1] for k in ks], [v[r:r + 1] for v in vs]

    def shift_ring(self, layer_idx, shift_tokens):
        self._rc.shift_ring_row(layer_idx, self._row, shift_tokens)

    def reset(self):
        self._rc.seq_lens[self._row] = 0
        self._rc.window_begs[self._row] = 0


def tp_rank_cache(
    model,
    rank: int,
    base: type = None,
    batch_size: int = 1,
    max_seq_len: int = -1
):
    """
    Build the KV cache for a single rank, for a rank-local worker process.

    This is the per-device slice of what ``ExLlamaV2Cache_TP`` would build, without constructing
    the other devices' caches.

    :param model:
        Model whose ``tp_context`` describes the split. The model does not have to be fully loaded --
        only ``tp_context`` and ``config`` are read.

    :param rank:
        Device index this worker owns.

    :param base:
        Cache class to use. Defaults to ``ExLlamaV2Cache_SWA`` when the config has a sliding
        window, otherwise ``ExLlamaV2Cache``.

    :param batch_size:
        Batch size.

    :param max_seq_len:
        Cache length in tokens.

    :return:
        A cache holding only this rank's KV slice, pinned to ``rank``.
    """
    assert model.tp_context is not None, \
        "tp_rank_cache requires a model loaded with a TP context"

    cfg = model.config
    if base is None:
        base = ExLlamaV2Cache_SWA if getattr(cfg, "sliding_window", 0) else ExLlamaV2Cache

    per_layer_geo = bool(getattr(cfg.arch.lm, "tp_per_layer_geometry", False))

    kwargs = dict(
        model = model,
        batch_size = batch_size,
        max_seq_len = max_seq_len,
        copy_from = None,
        lazy = True,
        fixed_device = rank,
    )
    if not per_layer_geo:
        # Uniform geometry: pass the narrowed head count directly.
        sp = model.tp_context.get_split(BROADCAST_KV)
        n = 0
        for d, a, b in sp:
            if d == rank:
                n = b - a
                break
        kwargs["num_key_value_heads"] = n

    cache = base(**kwargs)
    tp_narrow_cache_to_rank(cache, model.tp_context, rank)
    return _RankCacheView(cache)

class ExLlamaV2Cache_TP(ExLlamaV2CacheBase):

    caches: list[ExLlamaV2CacheBase]

    max_seq_len: int
    batch_size: int

    current_seq_len: int

    def __init__(
        self,
        model: ExLlamaV2,
        base: type = ExLlamaV2Cache,
        batch_size: int = 1,
        max_seq_len: int = -1
    ):
        super().__init__(model, batch_size, max_seq_len, torch.half, 1, 1, False, None)

        assert model.tp_context is not None, \
            "Cannot create TP cache unless model is loaded with load_tp()"

        # `num_key_value_heads` is only passed when the model has a single attention geometry.
        # For a per-layer-geometry model (Gemma-4: 16 kv heads on sliding layers, 4 on full
        # layers) the per-device head count differs per layer, so the cache must derive it from
        # get_layer_attn_geometry() and then narrow it to this device's slice.
        per_layer_geo = bool(getattr(model.config.arch.lm, "tp_per_layer_geometry", False))

        self.caches = []
        for idx, a, b in model.tp_context.get_split(BROADCAST_KV):
            if b - a <= 0:
                self.caches.append(None)
                continue
            # ExLlamaV2Cache_SWA takes no num_key_value_heads (it derives per-layer geometry
            # itself); the plain caches do. Pass it only where it is accepted.
            _kwargs = dict(
                model = model,
                batch_size = batch_size,
                max_seq_len = max_seq_len,
                copy_from = None,
                lazy = False,
                fixed_device = idx,
            )
            if not per_layer_geo:
                _kwargs["num_key_value_heads"] = b - a
            cache = base(**_kwargs)
            if per_layer_geo:
                # Narrow each layer's KV head count to this device's slice, using THAT
                # layer's own geometry split. A single split cannot serve both geometries:
                # the sliding layers split 16 kv heads 8/8 while the full-attention layers
                # have only 4 kv heads, so device 1 would get 0 heads on those layers and
                # end up with a zero-width KV buffer.
                for li in range(cache.num_hidden_layers):
                    kvh = cache.layer_num_kv_heads[li]
                    if kvh <= 0:
                        continue
                    _geo = model.tp_context.geometry_keys[li]
                    _sp = model.tp_context.geometry_splits[_geo]["kv"][0]
                    la = lb = 0
                    for _d, _a, _b in _sp:
                        if _d == idx:
                            la, lb = _a, _b
                            break
                    n = max(0, lb - la)
                    cache.layer_num_kv_heads[li] = n
                    hd = cache.layer_head_dims[li]
                    cache.layer_shapes_basic[li] = (cache.batch_size, cache.max_seq_len, n, hd)
                    cache.layer_shapes_wk[li] = (cache.batch_size, cache.max_seq_len, n, hd // cache.weights_per_element_k)
                    cache.layer_shapes_wv[li] = (cache.batch_size, cache.max_seq_len, n, hd // cache.weights_per_element_v)
                    cache.layer_shapes_s[li] = (cache.batch_size, cache.max_seq_len, n, hd // 32)
                # Rebuild the state tensors with the narrowed shapes. Allocate lazily and
                # then build rings + full-attention buffers only: calling create_state_tensors
                # with lazy=False here would transiently allocate a full max_seq_len buffer for
                # every layer, including the sliding ones, which OOMs at 64k.
                cache.key_states = []
                cache.value_states = []
                cache.key_scales = []
                cache.value_scales = []
                cache.create_state_tensors(None, True)
                if hasattr(cache, "_allocate_rings"):
                    cache._allocate_rings()
                    cache._allocate_full_attention()
            self.caches.append(cache)

        # for idx, cache in enumerate(self.caches):
        #     if cache is None: continue
        #     cache.fixed_device = idx
        #     cache.create_state_tensors(copy_from = None, lazy = False)


        # BUG 94: SWA ring bookkeeping (see ExLlamaV2Cache_SWA.window_beg).
        self.window_beg = 0

    def shift_ring(self, layer_idx: int, shift_tokens: int):
        """BUG 94: shift every per-device ring slice (see ExLlamaV2Cache_SWA.shift_ring)."""
        for cache in self.caches:
            if cache is not None:
                cache.shift_ring(layer_idx, shift_tokens)

    def roll_left(self):
        for cache in self.caches:
            cache.roll_left()


    def get_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | list[torch.Tensor] | None = None,
        block_table: torch.Tensor | list[torch.Tensor] | None = None
    ) -> (torch.Tensor, torch.Tensor):

        kc, vc = [], []
        for idx, cache in enumerate(self.caches):
            k, v = cache.get_kv_state(
                layer_idx,
                batch_size,
                offset,
                width,
                page_size,
                cache_seqlens[idx] if cache_seqlens else None,
                block_table[idx] if block_table else None
            )
            kc.append(k)
            vc.append(v)
        return kc, vc


    def store_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | list[torch.Tensor] | None = None,
        block_table: torch.Tensor | list[torch.Tensor] | None = None
    ):
        for idx, cache in enumerate(self.caches):
            cache.store_kv_state(
                layer_idx,
                batch_size,
                offset,
                width,
                page_size,
                cache_seqlens[idx] if cache_seqlens else None,
                block_table[idx] if block_table else None
            )


    @torch.inference_mode
    def copy_states(
        self,
        target: ExLlamaV2Cache_TP,
        from_column: int,
        from_columns: int,
        to_column: int,
        to_columns: int,
        from_row: int,
        from_rows: int,
        to_row: int,
        to_rows: int
    ):
        for cache, tcache in zip(self.caches, target.caches):
            cache.copy_states(
                tcache,
                from_column,
                from_columns,
                to_column,
                to_columns,
                from_row,
                from_rows,
                to_row,
                to_rows
            )


    def touch_device(self, device):
        raise NotImplementedError()


    def all_tensors(self):
        tensors = []
        for cache in self.caches:
            tensors += cache.all_tensors()
        return tensors


    def reset(self):
        self.current_seq_len = 0
        self.window_beg = 0
