from __future__ import annotations
import os, sys

min_version = (3, 8)
if sys.version_info < min_version:
    print("")
    print(f" ## Warning: this project requires Python {min_version[0]}.{min_version[1]} or higher.")
    print("")

# Set CUDA context to lazy loading since we won't need 95% of the modules in Torch
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# # Set cudaMallocAsync allocator by default as it appears slightly more memory efficient, unless Torch is already
# # imported in which case changing the allocator would cause it to crash
# if not "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
#     try:
#         x = torch.__version__
#     except NameError:
#         os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import torch
if not (torch.version.cuda or torch.version.hip):
    print("")
    print(f" ## Warning: The installed version of PyTorch is {torch.__version__} and does not support CUDA or ROCm.")
    print("")

# PyTorch, especially v2.3.1, gets confused when working with small CPU tensors and likes to use way too many worker
# threads for small operations, adding considerable overhead. Limit it to a single thread to avoid that (globally
# because that seems to be the only way)
torch.set_num_threads(1)

import math
from exllamav2.config import ExLlamaV2Config
from exllamav2.cache import ExLlamaV2CacheBase, ExLlamaV2Cache_SWA
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.attn import ExLlamaV2Attention, has_flash_attn, has_xformers
from exllamav2.layer_scalar import ExLlamaV2LayerScalar
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.gdn import ExLlamaV2GatedDeltaNet
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.moe_mlp import ExLlamaV2MoEMLP
from exllamav2.parallel_decoder import ExLlamaV2ParallelDecoder
from exllamav2.embedding import ExLlamaV2Embedding
from exllamav2.pos_embedding import ExLlamaV2PosEmbedding
from exllamav2.compat import safe_move_tensor
from exllamav2.stloader import cleanup_stfiles
from exllamav2.device import ExLlamaV2DeviceContext, set_device_streams
from exllamav2.tensor_p import TPContext, BROADCAST_VC
from exllamav2.tp_graph import TPGraphManager, TPGraphRunner
from exllamav2.architecture import RopeStyle
import gc
import threading
from typing import Callable
# from exllamav2.util import list_live_tensors, print_vram_usage, set_snapshot, diff_snapshot, print_vram_usage_peak
from exllamav2.util import get_basic_progress, timed
# from line_profiler import profile
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor


# BUG 117: a TP whole-step graph is only valid within this many tokens of the kv length it
# was captured at. Beyond it the FlashInfer graph-mode attention launch (baked at capture)
# writes out of bounds, so the step recaptures at the current length. Large enough that
# normal generation (a few hundred tokens per job) never recaptures, small enough to stay
# well inside the kernel's valid range; jumps across it happen at job boundaries / resets.
# 2026-11-24 (USER-CONFIRMED FIX): long-prompt (EXL2.5 + hd512 + INT8) output degenerated
# in the window LEADING UP TO the 2048 recapture (~round 650) -> the FI graph-mode kernel's
# valid range is narrower than 2048 in that regime. Tightening the default to 256 so the
# graph recaptures before it drifts out of range eliminated the degeneration (user-verified
# on a 70k prompt). Env-tunable (EXL2_TP_GRAPH_KV_ENVELOPE) to trade recapture frequency vs
# safety without a rebuild; graphs stay enabled (recapture, not disable).
# 2026-11-26: default 512 (user-directed; kept -- do NOT narrow as a "fix"). Degeneration
# is NOT the envelope width: 256 ALSO degenerates after the recent changes, so recapture
# frequency is not the root cause. The forward fix is the always-on FI guard (tensor_p.py
# _fi_guard, kv>20000 gate removed): validate + repair the fixed FI plan buffers and sync
# the TP streams on EVERY plan/replay, catching the metadata spray / ordering race that
# writes wrong KV within the envelope window. Env-tunable (EXL2_TP_GRAPH_KV_ENVELOPE).
TP_GRAPH_KV_ENVELOPE = int(os.environ.get("EXL2_TP_GRAPH_KV_ENVELOPE", "512"))


def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"


class ExLlamaV2:

    @staticmethod
    def _swa_ring_params(cache):
        """BUG 94: (ring_size, page_size, is_swa_layer) for an SWA TP/rank cache, else None.

        The streaming/TP path must keep the sliding-window layers' K/V inside their small
        ring buffers. The ring parameters live on the per-device SWA cache; the TP wrapper
        and the rank-local view expose them via the first non-None member.
        """
        if cache is None:
            return None
        _is_swa = getattr(cache, "is_swa_layer", None)
        _ring = getattr(cache, "kv_state_size", None)
        _page = getattr(cache, "page_size", 128)
        if _is_swa is None:
            for _c in getattr(cache, "caches", ()):
                if _c is None:
                    continue
                _is_swa = _c.is_swa_layer
                _ring = _c.kv_state_size
                _page = _c.page_size
                break
        if _is_swa is None or not any(_is_swa) or _ring is None:
            return None
        return _ring, _page, _is_swa

    config: ExLlamaV2Config
    modules: list[ExLlamaV2Module]
    modules_dict: dict[str: ExLlamaV2Module]
    device_context: list[ExLlamaV2DeviceContext | None]
    cache_map: dict[int: str]
    last_kv_layer_idx: int
    head_layer_idx: int
    loaded: bool

    tp_context: TPContext | None

    def __init__(
        self,
        config: ExLlamaV2Config,
        lazy_load = False,
        archparams = None
    ):

        self.config = config
        self.modules = []
        self.modules_dict = {}
        self.device_context = []
        self.cache_map = {}
        self.loaded = False
        self.tp_context = None

        # CUDA-graph decode state (BUG 43). Created in load_tp_gen; None for non-TP models,
        # which keep using the existing fused-kernel graph path in q_attn.cu / q_mlp.cu.
        # Change98: the static input/logits/state buffers live on each TPGraphRunner, not here.
        self.tp_graph = None
        # BUG 107: set if the MTP verify-window graph capture fails; later verifies stay
        # eager WITHOUT disabling the decode/bonus graphs (see _tp_graph_step).
        self._tp_graph_verify_failed = False

        # DFlash: module indices whose INPUT residual (= the target decoder
        # layer's output, the HF hidden_states[layer_id + 1] convention) is
        # captured when a forward requests return_row_aux. Set by
        # TPProcessGroup.enable_mtp on every rank, derived from the module keys
        # (Gemma-4 has attn + mlp + layer_scalar modules per layer, so the tap
        # for layer L is the index right after its layer_scalar).
        # None = no DFlash taps configured.
        self._dflash_aux_taps = None
        # Grow-only pinned-host staging for the per-chunk aux D2H (DFLASH5.txt:
        # GPU-resident accumulation OOMed the master on a 71k prefill).
        self._dflash_aux_staging = None

        # Build model

        cfg = self.config
        if archparams is None: archparams = cfg.arch.lm
        self.archparams = archparams

        # `embed_key` / `norm_key` are relative to lm_prefix. Gemma-4 nests them directly
        # under `model.language_model.`, so the old hard-coded "model.embed_tokens" produced
        # `model.language_model.model.embed_tokens`, which does not exist. PLAN.md §2.1.
        emb = ExLlamaV2Embedding(self, cfg.arch.lm_prefix + getattr(archparams, "embed_key", "model.embed_tokens"))
        self.modules += [emb]

        # Qwen4-Exp: widen the embedding into the hc_count-stream hyper-residual.
        if getattr(cfg.arch.lm, "hc_residual", False):
            from exllamav2.qwen4_exp import ExLlamaV2HyperConnection
            self.modules += [ExLlamaV2HyperConnection(self, "hc_expand", "expand")]

        if archparams.keys["learned_pos_emb"]:
            pos_emb = ExLlamaV2PosEmbedding(self, archparams.keys["learned_pos_emb"])
            self.modules += [pos_emb]

        for layer_idx in range(cfg.num_hidden_layers):

            # `layers_root` is relative to lm_prefix. Gemma-4 uses "layers".
            # PLAN.md §2.1.
            layers_root = getattr(archparams, "layers_root", "model.layers")
            layer_key = cfg.arch.lm_prefix + f"{layers_root}.{layer_idx}"
            rope_index = 0

            if cfg.arch.lm.alternating_swa:
                if cfg.sliding_window_pattern > 1:
                    swa = cfg.sliding_window if (layer_idx + 1) % cfg.sliding_window_pattern != 0 else 0
                    if swa and cfg.rotary_embedding_base_alt:
                        rope_index = 1
                else:
                    swa = cfg.sliding_window if not bool(layer_idx % 2) else 0
            elif cfg.arch.lm.swa:
                swa = cfg.sliding_window
            else:
                swa = 0

            if cfg.arch.lm.parallel_decoder_blocks:
                pd = ExLlamaV2ParallelDecoder(self, layer_key, layer_idx, sliding_window = swa, rope_index = rope_index)
                self.modules += [pd]
            elif getattr(cfg.arch.lm, "hc_residual", False):
                # Qwen3.8-Flash-Next (P7): hyper-connection wrapped blocks. The
                # residual stream is hc_count x hidden; mix collapses it for the
                # block, inject scatters the block output back into the streams.
                from exllamav2.qwen4_exp import (
                    ExLlamaV2HyperConnection, ExLlamaV2QSAAttention,
                    ExLlamaV2Qwen4MoE, ExLlamaV2PLELayer,
                )
                ple_ids = getattr(cfg, "ple_layer_ids", None) or []
                if layer_idx + 1 in ple_ids:
                    self.modules += [ExLlamaV2PLELayer(self, layer_key + ".ple", layer_idx)]
                if cfg.layer_types[layer_idx] == "linear_attention":
                    block = ExLlamaV2GatedDeltaNet(self, layer_key, layer_idx,
                                                   has_norm = False, has_residual = False)
                    hc_key = layer_key + ".attn_hyper_connection"
                else:
                    block = ExLlamaV2QSAAttention(self, layer_key, layer_idx)
                    hc_key = layer_key + ".attn_hyper_connection"
                self.modules += [
                    ExLlamaV2HyperConnection(self, hc_key, "mix", layer_idx, "attn"),
                    block,
                    ExLlamaV2HyperConnection(self, hc_key, "inject", layer_idx, "attn"),
                    ExLlamaV2HyperConnection(self, layer_key + ".mlp_hyper_connection", "mix", layer_idx, "mlp"),
                    ExLlamaV2Qwen4MoE(self, layer_key, layer_idx),
                    ExLlamaV2HyperConnection(self, layer_key + ".mlp_hyper_connection", "inject", layer_idx, "mlp"),
                ]
            elif getattr(cfg, "hybrid_linear_attn", False) and \
                    cfg.layer_types[layer_idx] == "linear_attention":
                # Qwen3.5 gated-delta-net block replaces self_attn on linear layers.
                # newplan.md P6.
                gdn = ExLlamaV2GatedDeltaNet(self, layer_key, layer_idx)
                mlp = ExLlamaV2MLP(self, layer_key, layer_idx)
                self.modules += [gdn, mlp]
            else:
                attn = ExLlamaV2Attention(self, layer_key, layer_idx, sliding_window = swa, rope_index = rope_index)
                if getattr(cfg.arch.lm, "gemma4_moe_block", False):
                    # Gemma-4-26B-A4B: parallel dense MLP + routed experts per layer.
                    from exllamav2.gemma4_moe import ExLlamaV2Gemma4MoEBlock
                    mlp = ExLlamaV2Gemma4MoEBlock(self, layer_key, layer_idx)
                elif cfg.arch.lm.is_moe: mlp = ExLlamaV2MoEMLP(self, layer_key, layer_idx)
                else: mlp = ExLlamaV2MLP(self, layer_key, layer_idx)
                self.modules += [attn, mlp]

                # Per-layer scalar applied to the block output, after both residual adds.
                # PLAN.md §2.6. Gemma-4 only.
                if getattr(cfg.arch.lm, "layer_scalar", False):
                    ls = ExLlamaV2LayerScalar(self, layer_key + ".layer_scalar")
                    self.modules += [ls]

        _norm_key_rel = getattr(archparams, "norm_key", "model.norm")
        if _norm_key_rel is None:
            # Qwen4-Exp has no final model norm: the global hyper-connection mixer
            # (grouped RMSNorm + weighted stream sum) produces the head input.
            from exllamav2.qwen4_exp import ExLlamaV2HyperConnection
            self.modules += [ExLlamaV2HyperConnection(
                self, cfg.arch.lm_prefix + "hyper_connection_mixer", "mix_only")]
        else:
            _norm_key = cfg.arch.lm_prefix + _norm_key_rel
            if cfg.arch.lm.norm == "layernorm":
                norm = ExLlamaV2LayerNorm(self, _norm_key)
            elif cfg.arch.lm.norm == "rmsnorm":
                norm = ExLlamaV2RMSNorm(self, _norm_key)
            else:
                raise ValueError("unknown norm type")
            self.modules += [norm]

        self.head_layer_idx = len(self.modules)
        head = ExLlamaV2Linear(
            self,
            cfg.arch.lm_prefix + "lm_head",
            cfg.hidden_size,
            cfg.vocab_size,
            False,
            max_out_len = cfg.max_output_len,
            prescale = cfg.logit_scale,
            is_sub_module = False,
            normalize_unq = bool(cfg.norm_head)
        )
        if archparams.keys["lm_head"] != "lm_head":
            head.alt_key = cfg.arch.lm_prefix + archparams.keys["lm_head"]
        self.modules += [head]

        # Compile dictionary of modules

        for module in self.modules:
            if len(module.submodules) > 0:
                for m in module.submodules: self.modules_dict[m.key] = m
            else:
                self.modules_dict[module.key] = module

        # Find last layer that affects k/v cache

        layer_idx = len(self.modules)
        while True:
            layer_idx -= 1
            if isinstance(self.modules[layer_idx], ExLlamaV2Attention) or \
               isinstance(self.modules[layer_idx], ExLlamaV2ParallelDecoder):
                break

        self.last_kv_layer_idx = layer_idx


    def set_device_map(
        self,
        allocation: list[float],
        embed_cpu: bool = True
    ) -> list[float]:

        cfg = self.config
        self.cache_map = {}

        # Constant shared between layers

        if self.archparams.rope_style != RopeStyle.NONE:
            # Two tables may exist (base + alt). For Gemma-4 the base table spans global_head_dim
            # (512) while the alt table spans head_dim (256), so size for the larger of the two.
            # Using cfg.head_dim alone would under-allocate by 2x and corrupt the 512-dim layers.
            # PLAN.md §2.3 / §8 trap 1.
            max_sincos_dim = max(cfg.head_dim, getattr(cfg, "global_head_dim", cfg.head_dim))
            sincos_size = max_sincos_dim * cfg.max_seq_len * 2
            n_tables = 2 if cfg.rotary_embedding_base_alt else 1
            constant_size = sincos_size * 2 * n_tables
        else:
            constant_size = 0

        # Max size of hidden state

        state_size = cfg.hidden_size * cfg.max_input_len * cfg.max_batch_size * 2
        mask_size = cfg.max_input_len ** 2 * cfg.max_batch_size * 2

        # Bytes remaining per device

        allocation_bytes =  [a * 1024**3 - (constant_size + state_size + mask_size) for a in allocation]

        # Scratch space required per device

        reserve_bytes = [0 for a in allocation]
        reserve_bytes_attn = [0 for a in allocation]
        fixed_bytes = [0 for a in allocation]

        current_idx = 0
        for idx, module in enumerate(self.modules):

            # Special case for token embeddings on CPU

            if isinstance(module, ExLlamaV2Embedding) and embed_cpu:

                module.set_device_idx(-1)
                continue

            # Special case for attention

            attn_bytes_current = 0
            if isinstance(module, ExLlamaV2Attention): attn_bytes_current = module.temp_attn_size()

            # Advance current_idx until module fits in allocation

            footprint = module.weight_footprint()   # Footprint, in bytes
            scratch = module.scratch_space()        # Scratch space required by module

            while True:
                assert current_idx < len(allocation_bytes), "Insufficient space in device allocation"
                dev_scratch = max(scratch, reserve_bytes[current_idx])
                dev_scratch_attn = max(attn_bytes_current, reserve_bytes_attn[current_idx])
                if footprint + dev_scratch + dev_scratch_attn <= allocation_bytes[current_idx]: break
                current_idx += 1

            # Size for fixed tensors

            scratch_fixed = module.scratch_space_fixed()
            fixed_bytes[current_idx] = max(scratch_fixed, fixed_bytes[current_idx])

            # Subtract module size from allocation

            reserve_bytes[current_idx] = dev_scratch
            reserve_bytes_attn[current_idx] = dev_scratch_attn
            allocation_bytes[current_idx] -= footprint

            module.set_device_idx(current_idx)

        # Prepare to prepare device tensors

        self.device_context = []
        for idx, scratch_bytes in enumerate(fixed_bytes):
            if scratch_bytes > 0:
                self.device_context.append(ExLlamaV2DeviceContext(self, idx, scratch_bytes))
            else:
                self.device_context.append(None)

        # Create map for cache

        self.set_cache_map()

        # Return unused space, in GB

        return [(ab - rb - rba) / 1024**3 for (ab, rb, rba) in zip(allocation_bytes, reserve_bytes, reserve_bytes_attn)]


    def load(
        self,
        gpu_split: list[float] | None = None,
        lazy: bool = False,
        stats: bool = False,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None,
        progress: bool = False
    ):
        """
        Load model, regular manual split mode.

        :param gpu_split:
            List of VRAM allocations for weights and fixed buffers per GPU. Does not account for the size of the cache
            which must be allocated with reference to the model subsequently and whose split across GPUs will depend
            on which devices end up receiving which attention layers.

            If None, only the first GPU is used.

        :param lazy:
            Only set the device map according to the split, but don't actually load any of the modules. Modules can
            subsequently be loaded and unloaded one by one for layer-streaming mode.

        :param stats:
            Legacy, unused

        :param callback:
            Callable function that triggers after each layer has loaded, for progress update etc.

        :param callback_gen:
            Same as callback, but for use by async functions

        :param progress:
            If True, create a rich progress bar in the console while loading. Cannot be used with callbacks
        """

        from exllamav2.version import __version_string__
        print(f" -- {__version_string__}")

        if progress:
            progressbar = get_basic_progress()
            progressbar.start()
            task_id = progressbar.add_task("Loading: " + self.config.model_dir, total = len(self.modules))
            module = 0
            def callback_pb(a, b):
                progressbar.update(task_id, advance = 1)
            assert callback is None, \
                "Cannot use callback function and console progress bar at the same time."
            callback = callback_pb
        f = self.load_gen(gpu_split, lazy, stats, callback, callback_gen)
        for item in f:
            pass
        if progress:
            progressbar.stop()


    def load_gen(
        self,
        gpu_split: list[float] | None = None,
        lazy: bool = False,
        stats: bool = False,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None
    ):
        with torch.inference_mode():
            set_device_streams()

            stats_ = self.set_device_map(gpu_split or [99999])

            # Load module weights

            if not lazy:

                for idx, module in enumerate(self.modules):

                    if callback is not None: callback(idx, len(self.modules))
                    if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                    module.load()

                if callback is not None: callback(len(self.modules), len(self.modules))
                if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            # Cache map

            self.set_cache_map()

            self.loaded = True
            cleanup_stfiles()


    def load_tp(
        self,
        gpu_split: list[float] | None = None,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None,
        progress: bool = False,
        expect_cache_tokens: int = 0,
        expect_cache_base: type = None
    ):
        """
        Load model, tensor-parallel mode.

        :param gpu_split:
            List of VRAM allocations per GPU. The loader attempts to balance tensor splits to stay within these
            allocations, accounting for an uneven distribution of attention heads and the expected size of the cache.

            If None, the loader attempts to use all available GPUs and creates a split based on the currently available
            VRAM according to nvidia-smi etc.

        :param callback:
            Callable function that triggers after each layer has loaded, for progress update etc.

        :param callback_gen:
            Same as callback, but for use by async functions

        :param progress:
            If True, create a rich progress bar in the console while loading. Cannot be used with callbacks

        :param expect_cache_tokens:
            Expected size of the cache, in tokens (i.e. max_seq_len * max_batch_size, or just the cache size for use
            with the dynamic generator) to inform the automatic tensor split. If not provided, the configured
            max_seq_len for the model is assumed.

        :param expect_cache_base:
            Cache type to expect, e.g. ExLlamaV2Cache_Q6. Also informs the tensor split. If not provided, FP16 cache
            is assumed.
        """
        if progress:
            progressbar = get_basic_progress()
            progressbar.start()
            task_id = progressbar.add_task("Loading: " + self.config.model_dir, total = len(self.modules))
            module = 0
            def callback_pb(a, b):
                progressbar.update(task_id, advance = 1)
            assert callback is None, \
                "Cannot use callback function and console progress bar at the same time."
            callback = callback_pb
        f = self.load_tp_gen(gpu_split, callback, callback_gen, expect_cache_tokens, expect_cache_base)
        for item in f:
            pass
        if progress:
            progressbar.stop()


    def load_tp_gen(
        self,
        gpu_split: list[float] | None = None,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None,
        expect_cache_tokens: int = 0,
        expect_cache_base: type = None
    ):
        # The fused single-device q_attn/q_mlp graph path is still disabled for TP: those
        # kernels assume one device. TP decode now has its own whole-step graph instead
        # (tp_graph.py), which is what actually removes the per-op launch latency.
        self.config.no_graphs = True
        self.tp_context = TPContext(self, gpu_split, expect_cache_tokens, expect_cache_base)
        self.tp_graph = TPGraphManager(self)

        # Create device tensors

        devs = self.tp_context.num_devices
        scratch = [0] * devs
        for module in self.modules[1:]:
            ms = module.scratch_space_tp()
            for i, m in enumerate(ms):
                scratch[i] = max(scratch[i], m)

        self.device_context = [None] * devs
        for idx in range(devs):
            if scratch[idx] > 0:
                self.device_context[idx] = ExLlamaV2DeviceContext(self, idx, scratch[idx])

        # Load module weights

        with torch.inference_mode():
            set_device_streams()

            for idx in range(len(self.modules)):
                module = self.modules[idx]

                if callback is not None: callback(idx, len(self.modules))
                if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                if isinstance(module, ExLlamaV2Embedding):
                    module.set_device_idx(-1)
                else:
                    module.set_device_idx(self.tp_context.device)

                module.load(device_context = False)

                if isinstance(module, ExLlamaV2Embedding):
                    module.tp_split()
                if isinstance(module, ExLlamaV2Attention):
                    module.tp_split()
                if isinstance(module, ExLlamaV2GatedDeltaNet):
                    module.tp_split()
                if isinstance(module, ExLlamaV2MLP):
                    module.tp_split()
                if isinstance(module, ExLlamaV2RMSNorm):
                    # The norm that feeds the head must stay on the head's device and return a
                    # single tensor: the head is replicated (bug 23), so a TP-split norm would
                    # hand it a per-device list. Stage L bug 24.
                    if idx == self.head_layer_idx - 1:
                        module.tp_replicate(self.tp_context.device)
                    else:
                        module.tp_split(BROADCAST_VC)
                elif isinstance(module, ExLlamaV2Linear):
                    module.tp_split(BROADCAST_VC)

                module.set_device_idx(None)

            torch.cuda.empty_cache()
            gc.collect()

            if callback is not None: callback(len(self.modules), len(self.modules))
            if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            self.loaded = True
            cleanup_stfiles()

            self.tp_context.finalize()


    def load_tp_rank(
        self,
        rank: int,
        gpu_split: list[float] | None = None,
        expect_cache_tokens: int = 0,
        expect_cache_base: type = None,
        load_head: bool | None = None,
        callback: Callable[[int, int], None] | None = None,
        max_batch: int = 1
    ):
        """
        Load only the shard of the model owned by device ``rank``.

        This is the worker-side counterpart of :meth:`load_tp`. A multiprocess TP backend runs one
        process per GPU, and each process must build its own slice without ever materialising the
        whole model -- that is the entire point, since the model does not fit on one card.

        Every rank builds the same ``TPContext`` from the same ``gpu_split``, because the split
        definition has to agree across ranks for the collectives to line up. What differs is the device
        context: only ``rank``'s scratch space is allocated, and only ``rank``'s shards are loaded.

        ``load_head`` controls whether ``lm_head`` is loaded. It is replicated, not sharded, and on
        Gemma-4 it is 2.8 GB (262144 x 5376 fp16), so only the master should load it -- the
        workers never run it.

        :param rank:
            Device index this process owns.

        :param gpu_split:
            Per-GPU VRAM allocations in GB. Must be identical on every rank.

        :param expect_cache_tokens:
            Expected cache size in tokens, used to inform the split. Must be identical on every rank.

        :param expect_cache_base:
            Cache type to expect, used to inform the split. Must be identical on every rank.

        :param load_head:
            Whether to load ``lm_head``. Defaults to True only for rank 0.

        :param callback:
            Called as ``callback(module_idx, num_modules)`` after each module loads.
        """
        from exllamav2.embedding import ExLlamaV2Embedding
        from exllamav2.attn import ExLlamaV2Attention
        from exllamav2.mlp import ExLlamaV2MLP
        from exllamav2.rmsnorm import ExLlamaV2RMSNorm
        from exllamav2.layernorm import ExLlamaV2LayerNorm
        from exllamav2.layer_scalar import ExLlamaV2LayerScalar
        from exllamav2.tensor_p import BROADCAST_VC, BROADCAST_Q, BROADCAST_RS

        if load_head is None:
            load_head = (rank == 0)

        # BUG 77: forward_chunk must know whether this process owns lm_head.
        # Vocab-parallel head (default ON; EXL2_TP_VOCAB_HEAD=0 restores the replicated
        # master-only head): every rank owns a vocab shard of the unquantized head and
        # runs it, so every rank "has" the head. Frees ~2 GB on the master (the hb16
        # head was rank-0-only) -- the fix for the master OOM at 160k cache + drafter
        # + MTP shadow. Logits shards are allgathered inside head.forward (~0.5 MB/token).
        self._tp_vocab_head = os.environ.get("EXL2_TP_VOCAB_HEAD", "1") != "0"
        # Vocab-sharded embedding table (EXL2_TP_VOCAB_EMBED=0 restores the 2.8 GB
        # replicated table): same VC split as the head, SUM-reduced masked lookups.
        self._tp_vocab_embed = os.environ.get("EXL2_TP_VOCAB_EMBED", "1") != "0"
        self._tp_has_head = bool(load_head) or self._tp_vocab_head

        # The fused single-device graph path assumes one device.
        self.config.no_graphs = True

        self.tp_context = TPContext(self, gpu_split, expect_cache_tokens, expect_cache_base,
                                    max_batch = max_batch)
        # BUG 58: enter multiprocess mode BEFORE any module loads. get_split() filters to
        # this rank from here on, and the loaders size their shards from it -- setting it
        # afterwards would make every module load the full split.
        self.tp_context.set_rank(rank)
        # Sequence-parallel (hidden-sharded residual) mode. Must be decided BEFORE the
        # scratch sizing and module loads below: it changes o_proj's parallel layout and
        # the post-norm weight sharding. MP rank-local path only; requires the per-layer
        # geometry TP scheme (Gemma-4's forward_tp_old), which is the only validated one.
        # Default ON (user directive 2026-11-20 after results.md A/B): the hidden-sharded
        # residual cuts prefill/long-context collective volume; EXL2_TP_SEQPAR=0 restores
        # the replicated-residual path (slightly faster pure short-ctx decode).
        self.tp_context.seqpar = os.environ.get("EXL2_TP_SEQPAR", "1") != "0"
        # Qwen4-Exp: the hyper-connection mix reads all hc_count streams, so the
        # residual must stay full-width on every rank (no hidden-sharded seqpar).
        if bool(getattr(self.config.arch.lm, "tp_no_seqpar", False)):
            self.tp_context.seqpar = False
        # Decode/verify-sized forwards keep a FULL-width replicated residual (no tiny
        # post-norm stat all_reduces); prefill chunks keep the seqpar shard flow.
        self.tp_context.decode_repl = self.tp_context.seqpar and \
            os.environ.get("EXL2_TP_DECODE_REPLICATED", "1") != "0"
        # PCIe P2P small-collective fast path (probed + agreed across ranks; any
        # failure falls back to NCCL). EXL2_TP_P2P=0 forces NCCL.
        from exllamav2.p2p_comm import P2PComm
        # Arena scales with max_batch: every (B, q_len) graph shape owns a permanent
        # slot region (~12 slots/graph, stable across recaptures). B=2 keeps the
        # proven 128; B=4 needs headroom for the 8 decode+verify shapes + eager ops.
        self.tp_context.p2p = P2PComm(rank, self.tp_context.num_devices,
                                      n_slots = 64 if max_batch <= 1 else 64 + 32 * max_batch)
        if self.tp_context.seqpar:
            assert bool(getattr(self.config.arch.lm, "tp_per_layer_geometry", False)), \
                "EXL2_TP_SEQPAR requires the per-layer-geometry TP path (Gemma-4)"
            print(f"[TP] rank {rank}: sequence-parallel residual mode (default ON; "
                  f"EXL2_TP_SEQPAR=0 disables)", flush=True)
        # BUG 80: keep the CUDA-graph path. Graphs are worth 2.55x on decode in the
        # single-process path and are the largest single factor in the MP path too. Capture is
        # per-process and only touches the devices this process owns (TPContext.all_devices() is
        # rank-filtered), so it is valid in a worker; if it fails, TPGraphManager.disable()
        # falls back to the eager path, which is what this used to do unconditionally.
        self.tp_graph = TPGraphManager(self)

        # Replicated resident MoE uses the graph-safe fused kernel (TP graphs stay ON).
        # Offloaded OR sharded MoE runs the data-dependent torch routing loop (+ H2D
        # staging / per-expert collectives), which cannot be baked into a replayed graph
        # -> disable capture for those cases.
        if any(isinstance(m, ExLlamaV2MoEMLP) and (m.offload_experts > 0 or m._tp_shard)
               for m in self.modules):
            self.tp_graph.disable("MoE offload/sharding: data-dependent routing is not graph-capturable")
        # Hybrid backbones (GDN torch scan + hyper-connection stash + PLE n-gram
        # lookup) are not CUDA-graph-capturable yet; run eager. The GDN CUDA decode
        # kernel (Qwen27B-Plan.md Phase 1) is the follow-up that re-enables graphs.
        if getattr(self.config, "hybrid_linear_attn", False):
            self.tp_graph.disable("Hybrid GDN backbone: eager until the GDN CUDA kernel lands")
        if getattr(self.config.arch.lm, "gemma4_moe_block", False):
            self.tp_graph.disable("Gemma-4 MoE block: data-dependent torch routing is not graph-capturable")

        # Size the device context for THIS rank only.
        #
        # scratch_space_tp() returns one entry per device and also populates the
        # out_features_tp / in_features_tp lists that the TP forward paths dispatch on. It has to run
        # for every module so those lists are consistent, but only this rank's scratch is allocated.
        devs = self.tp_context.num_devices
        scratch = [0] * devs
        for _mi, module in enumerate(self.modules[1:], start = 1):
            # The unquantized vocab-sharded head needs no temp_dq scratch; the generic
            # VC-split sizing would otherwise reserve ~0.7 GB of dead pool on EVERY rank
            # (it dominated the device scratch max). The vocab loader resets the head's
            # out_features_tp lists itself, so skipping keeps dispatch consistent.
            if self._tp_vocab_head and _mi == self.head_layer_idx:
                continue
            ms = module.scratch_space_tp()
            for i, m in enumerate(ms):
                scratch[i] = max(scratch[i], m)

        self.device_context = [None] * devs
        if scratch[rank] > 0:
            self.device_context[rank] = ExLlamaV2DeviceContext(self, rank, scratch[rank])

        with torch.inference_mode():
            set_device_streams()

            for idx, module in enumerate(self.modules):
                if callback is not None:
                    callback(idx, len(self.modules))

                if isinstance(module, ExLlamaV2Embedding):
                    if self._tp_vocab_embed:
                        module.tp_load_rank_sharded(rank)
                    else:
                        module.tp_load_rank(rank)

                elif isinstance(module, ExLlamaV2Attention):
                    module.tp_load_rank(rank)

                elif isinstance(module, ExLlamaV2GatedDeltaNet):
                    module.tp_load_rank(rank)

                elif isinstance(module, ExLlamaV2MLP):
                    module.tp_load_rank(rank)

                elif isinstance(module, ExLlamaV2MoEMLP):
                    module.tp_load_rank(rank)

                elif isinstance(module, (ExLlamaV2RMSNorm, ExLlamaV2LayerNorm)):
                    # The norm feeding the head must stay full-width on the head's device; every
                    # other norm is replicated full-width on every device anyway.
                    if idx == self.head_layer_idx - 1:
                        module.tp_load_rank(rank, BROADCAST_VC)
                    else:
                        module.tp_load_rank(rank, BROADCAST_RS)

                elif isinstance(module, ExLlamaV2LayerScalar):
                    module.tp_load_rank(rank)

                elif type(module).__name__ in ("ExLlamaV2HyperConnection", "ExLlamaV2PLELayer",
                                               "ExLlamaV2Gemma4MoEBlock"):
                    # Qwen4-Exp / Gemma-4 MoE block: rank-local loaders.
                    module.tp_load_rank(rank)

                elif isinstance(module, ExLlamaV2Linear):
                    if idx == self.head_layer_idx:
                        if self._tp_vocab_head:
                            module.tp_load_rank_vocab(rank)
                        elif load_head:
                            module.tp_load_rank_replicated(rank)
                        else:
                            module.set_device_idx(rank)
                            module.out_features_tp = []
                            module.in_features_tp = []
                            module.is_tp = False
                    else:
                        module.tp_load_rank(rank, BROADCAST_VC)

                else:
                    raise NotImplementedError(
                        "load_tp_rank: unsupported module type "
                        + type(module).__name__ + " at index " + str(idx)
                    )

            torch.cuda.empty_cache()
            gc.collect()

            if callback is not None:
                callback(len(self.modules), len(self.modules))

            self.loaded = True
            cleanup_stfiles()

            # BUG 57: build streams / ext_tp_contexts / pos / past_len_tp. Without this the
            # first TP GEMM dereferences a NULL ctx->streams[dev].
            self.tp_context.finalize()

    def load_autosplit(
        self,
        cache: ExLlamaV2CacheBase,
        reserve_vram: int | list[int] | None = None,
        last_id_only: bool = False,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None,
        progress: bool = False
    ):
        """
        Load model, auto-split mode. This mode loads the model and builds the cache in parallel, using available
        devices in turn and moving on to the next device whenever the previous one is full.

        :param cache:
            Cache constructed with lazy = True. Actual tensor allocation for the cache will happen while loading the
            model.

        :param reserve_vram:
            Number of bytes to reserve on each device, either for all devices (as an int) or per-device (as a list).

        :param last_id_only:
            If True, model will be loaded in a mode that does can only output one set of logits (i.e. one token
            position) per forward pass. This conserves memory if the model is only to be used for generating text and
            not e.g. perplexity measurement.

        :param callback:
            Callable function that triggers after each layer has loaded, for progress update etc.

        :param callback_gen:
            Same as callback, but for use by async functions

        :param progress:
            If True, create a rich progress bar in the console while loading. Cannot be used with callbacks
        """
        if progress:
            progressbar = get_basic_progress()
            progressbar.start()
            task_id = progressbar.add_task("Loading: " + self.config.model_dir, total = len(self.modules))
            module = 0
            def callback_pb(a, b):
                progressbar.update(task_id, advance = 1)
            assert callback is None, \
                "Cannot use callback function and console progress bar at the same time."
            callback = callback_pb
        f = self.load_autosplit_gen(cache, reserve_vram, last_id_only, callback, callback_gen)
        for item in f:
            pass
        if progress:
            progressbar.stop()

    def load_autosplit_gen(
        self,
        cache: ExLlamaV2CacheBase,
        reserve_vram: int | list[int] | None = None,
        last_id_only: bool = False,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None
    ):

        # Limit model's max_input_len to max_seq_len if necessary
        self.config.max_input_len = min(self.config.max_input_len, self.config.max_seq_len)

        minimum_reserve_vram = 256 * 1024**2
        last_touched_device = -1
        current_device = 0
        num_devices = torch.torch.cuda.device_count()
        loras = None  # TODO: Autosplit load with LoRAs

        with torch.inference_mode():
            set_device_streams()

            self.device_context = []

            # Reserved space

            if reserve_vram is None:
                reserve_vram = [192 * 1024**2] + [64 * 1024**2] * (num_devices - 1)
            elif isinstance(reserve_vram, int):
                reserve_vram = [reserve_vram] * num_devices

            reserved_vram_tensors = []
            minimum_reserve_tensor = None

            # Largest hidden state to ever forward through model.
            # SWA ring caches cannot hold a full max_input_len chunk (the per-module
            # test forward below bypasses model.forward's ring-aware chunk loop), so
            # cap the test sequence at the ring depth; it still measures per-module
            # VRAM representative of real decode/prefill chunks.
            _test_len = self.config.max_input_len
            _swa_layers = getattr(cache, "is_swa_layer", None)
            _ring = getattr(cache, "kv_state_size", None)
            if _ring is not None and _swa_layers is not None and any(_swa_layers):
                _test_len = min(_test_len, _ring)

            hidden_state = torch.zeros((1, _test_len), dtype = torch.long)
            batch_size, seq_len = hidden_state.shape
            past_len = 0
            attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, past_len, None, None)

            # Size of fixed scratch space

            scratch_fixed = max(module.scratch_space_fixed() for module in self.modules)

            # Load modules and create cache tensors sequentially

            self.cache_map = {}
            for idx, module in enumerate(self.modules):

                if callback is not None: callback(idx, len(self.modules))
                if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                # Embedding layer on CPU

                if idx == 0:

                    module.set_device_idx(-1)
                    module.load()
                    hidden_state = module.forward(hidden_state)
                    continue

                while True:

                    # If we've reached a new device, allocate fixed tensors

                    if current_device > last_touched_device:

                        self.device_context.append(ExLlamaV2DeviceContext(self, current_device, scratch_fixed))
                        # if attn_mask is not None:
                        #     reserved_vram_tensors.append(attn_mask)
                        #     attn_mask = safe_move_tensor(attn_mask, _torch_device(current_device))
                        # else:
                        #     attn_mask = self.build_attn_mask(batch_size, seq_len, past_len, None, _torch_device(current_device))

                        b = reserve_vram[current_device]
                        reserved_vram_tensors.append(torch.empty((b,), dtype = torch.int8, device = _torch_device(current_device)))
                        minimum_reserve_tensor = torch.empty((minimum_reserve_vram,), dtype = torch.int8, device = _torch_device(current_device))

                        last_touched_device = current_device

                    # Attempt to load module and forward state

                    module.set_device_idx(current_device)

                    hidden_state_backup = hidden_state
                    # if hidden_state.device == "cpu":
                    #     hidden_state_backup = hidden_state.clone()
                    # else:
                    #     hidden_state_backup = safe_move_tensor(hidden_state, "cpu").clone()

                    try:
                        if isinstance(module, ExLlamaV2Attention) or \
                           isinstance(module, ExLlamaV2ParallelDecoder):
                            self.cache_map[module.layer_idx] = module.device()
                            cache.update_cache_tensors()

                        module.load()

                        if idx == self.head_layer_idx:
                            if last_id_only:
                                hidden_state = hidden_state.narrow(-2, -1, 1)
                            elif self.config.max_output_len is not None:
                                hidden_state = hidden_state.narrow(-2, -self.config.max_output_len, self.config.max_output_len)

                        hidden_state = safe_move_tensor(hidden_state, _torch_device(current_device))
                        hidden_state = module.forward(hidden_state, cache = cache, attn_params = attn_params, past_len = past_len, loras = loras)
                        fail = False

                    except Exception as e:

                        test = 0
                        if e.__class__.__name__ == "OutOfMemoryError" or \
                            "CUDA out of memory" in str(e) or \
                            "HIP out of memory" in str(e):
                            fail = True  # Exception object will hold references to tensors so we can't free them here
                        else:
                            raise

                    # If we failed, roll back and advance to next device

                    if fail:

                        module.unload()
                        hidden_state = None

                        if minimum_reserve_tensor is not None: del minimum_reserve_tensor
                        minimum_reserve_tensor = None

                        gc.collect()
                        torch.cuda.empty_cache()
                        hidden_state = hidden_state_backup.clone()

                        current_device += 1
                        if current_device >= num_devices:
                            raise RuntimeError("Insufficient VRAM for model and cache")

                        continue

                    break

            if callback is not None: callback(len(self.modules), len(self.modules))
            if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            hidden_state = None
            attn_params = None
            reserved_vram_tensors = None

            gc.collect()
            torch.cuda.empty_cache()
            self.loaded = True
            cleanup_stfiles()

        if 'yield' in locals():
            yield


    def unload(self):
        """
        Unloads the model and frees all unmanaged resources.
        """

        for module in self.modules:
            module.unload()

        self.modules = []
        self.modules_dict = {}
        self.device_context = []

        if self.tp_context:
            self.tp_context.unload()
            self.tp_context = None


    def set_cache_map(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention) or \
               isinstance(module, ExLlamaV2ParallelDecoder) or \
               isinstance(module, ExLlamaV2GatedDeltaNet):
                # GDN layers carry recurrent cache state (hybrid backbone) and
                # need a cache_map entry just like attention layers.
                self.cache_map[module.layer_idx] = module.device()


    def get_cache_devices(self) -> list[str]:

        return list(set(self.cache_map.values()))


    def create_device_context(self, scratch_bytes):

        for idx, b in enumerate(scratch_bytes):

            tensors = ExLlamaV2DeviceContext(self, idx, b)
            self.device_context.append(tensors)


    def drop_device_context(self):

        for dc in self.device_context:
            dc.drop()


    def free_device_context(self):

        for dc in self.device_context:
            dc.free()


    def get_device_context(self, device_idx, scratch = True):

        if device_idx == -1:
            return None

        context = self.device_context[device_idx]
        if not context.ready: context.prepare(scratch)
        return context


    def get_modules(self) -> list[ExLlamaV2Module]:

        return [module for module in self.modules]  #?


    def update_loras(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention): module.update_loras()
            if isinstance(module, ExLlamaV2MLP): module.update_loras()
            if isinstance(module, ExLlamaV2MoEMLP): module.update_loras()


    def is_quant(self) -> bool:

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention):
                if module.is_quant(): return True

        return False


    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: ExLlamaV2CacheBase | list[ExLlamaV2CacheBase] | None = None,
        input_mask: torch.Tensor | None = None,
        preprocess_only: bool = False,
        last_id_only: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        return_last_state: bool = False,
        return_row_states: bool = False,
        return_row_aux: bool = False,
        position_offsets: torch.Tensor | None = None,
        abort_event: threading.Event | None = None,
        cpu_logits: bool = False,
        **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """
        Runs a forward pass through the model. If a cache is used, also appends keys/values to the cache
        and advances it.

        :param input_ids:
            LongTensor of input token IDs, shape (batch_size, q_len)

        :param cache:
            Optional ExLlamaV2Cache. If not provided, q_len must be less than config.max_input_len

        :param input_mask:
            Additive attention bias, shape (batch_size, past_len + q_len, q_len)

        :param preprocess_only:
            Only forward up to the last layer that affects the K/V cache. Does not return logits. Used
            to prefill the cache.

        :param last_id_only:
            Process the entire input sequence but only pass the last token through the head layer and
            only return logits for the last token.

        :param loras:
            List of ExLlamaV2Lora objects to apply during the forward pass

        :param return_last_state:
            Also return the hidden state right before the head layer

        :param position_offsets:
            Tensor of position offsets, shape (batch_size, 1). Offset is applied to position IDs during
            RoPE application.

        :param abort_event:
            Optional event that, if set, will abort the forward pass. Function will return None if aborted.

        :param cpu_logits:
            If True, logits are collected and returned in system RAM. This is somewhat slower but can prevent
            out-of-memory errors when computing logits for all positions in a long sequence, such as during a
            perplexity test.

        :return:
            FP16 logits tensor, shape (batch_size, q_len, vocab_size)
            (optional) state tensor, shape (batch_size, q_len, hidden_size)

        :indexed_embeddings:
            Tensor of embeddings, shape (batch_size, q_len, hidden_size), indexed by input token IDs >=
            ExLlamaV2.EMBEDDING_INDEX
        """

        bsz, q_len = input_ids.shape
        remaining_q_len = q_len

        # Attn and MLP layers have preallocated buffers for temp states, sized by the model config. Effective max input
        # length depends on the current batch size

        effective_max_input_len = self.config.max_input_len * self.config.max_batch_size // bsz

        # Without a cache we can't process the sequence in chunks, so forward the whole thing and assume the input length
        # is less than config.max_input_len

        if cache is None or not isinstance(cache, ExLlamaV2CacheBase):

            assert q_len <= effective_max_input_len, "Maximum input length exceeded in model.forward"

            result = self.forward_chunk(
                input_ids = input_ids,
                cache = cache,
                input_mask = input_mask,
                preprocess_only = preprocess_only,
                last_id_only = last_id_only,
                loras = loras,
                return_last_state = return_last_state,
                return_row_states = return_row_states,
                return_row_aux = return_row_aux,
                position_offsets = position_offsets,
                abort_event = abort_event,
                **kwargs
            )

            if abort_event and abort_event.is_set(): return

            if return_row_aux:
                return result.get("logits"), result.get("last_state"), result.get("row_aux")
            if "row_states" in result:
                return result.get("logits"), result["row_states"]
            elif "last_state" in result:
                return result.get("logits"), result["last_state"]
            else:
                return result.get("logits")

        # Confirm that the input fits within the allocated cache space

        past_len = cache.current_seq_len
        assert past_len + q_len <= cache.max_seq_len, "Total sequence length exceeds cache size in model.forward"

        # INT8-primary prefill staging: the full-layer fp16 buffers exist ONLY
        # during prefill (the fp16 attention kernels read them); every chunk is
        # quantized into the INT8 primary as it lands, and the staging is freed
        # when the prefill ends. Row-view batch prefills are staged by the slot
        # scheduler around its chunk loop; a row-view forward that finds no
        # active staging (adapter solo path) owns its own begin/end here.
        _i8_own_staging = False
        if (getattr(cache, "int8_primary", False) and q_len > 1
                and not kwargs.get("graph_verify", False)):
            from exllamav2.cache import _RowCacheView
            if not isinstance(cache, _RowCacheView) or not cache.i8_staging_active():
                cache.begin_prefill_staging(past_len + q_len)
                _i8_own_staging = True

        # Split sequence

        result = None
        last_state = None
        row_states = None
        row_aux = None
        _prefill_trace = os.environ.get("EXL2_PREFILL_TRACE", "1") != "0"

        chunk_begin = 0
        while chunk_begin < q_len:

            # Abortable prefill (MP rank mode): a client disconnect must stop the chunk
            # loop, not just the token loop -- an 80k prompt otherwise keeps processing
            # for minutes after the request is gone. The master polls the shared abort
            # event; a 4-byte MAX all_reduce makes EVERY rank reach the same decision at
            # the SAME chunk boundary (a local-only check would desync the ranks and hang
            # the next collective). One tiny collective per chunk is negligible against
            # seconds of chunk work; decode (q_len == 1) and graphs are untouched.
            if self.tp_context is not None and self.tp_context.rank is not None and q_len > 1:
                import torch.distributed as _dist
                _ctx = self.tp_context
                if _ctx._abort_dev is None:
                    _ctx._abort_dev = torch.zeros(1, dtype = torch.int32, device = _ctx.device)
                _ctx._abort_dev.fill_(
                    1 if (_ctx.abort_flag is not None and _ctx.abort_flag.is_set()) else 0)
                _dist.all_reduce(_ctx._abort_dev, op = _dist.ReduceOp.MAX)
                if _ctx._abort_dev.item():
                    if _i8_own_staging:
                        cache.end_prefill_staging()
                    return None

            # Limit chunk_size to max_input_len

            chunk_size = min(remaining_q_len, effective_max_input_len)

            # BUG 94: sliding-window (SWA) layers keep their K/V in a small ring
            # (kv_state_size deep), not a full-length cache. Cap the prefill chunk at
            # the ring size -- a longer chunk can never fit -- and shift the ring
            # before the chunk if it would overflow, so the new K/V always lands in
            # [0, ring). The shift is a deterministic function of
            # (current_seq_len, window_beg, chunk_size), so every TP rank computes the
            # same shift and shifts its own ring slice with no cross-rank communication.
            # (current_seq_len, not the `past_len` local: in the flash branch that local
            # is not refreshed between chunks.)
            _swa_p = self._swa_ring_params(cache)
            if _swa_p is not None:
                _ring, _page, _swa = _swa_p
                if cache.current_seq_len == 0 and cache.window_beg != 0:
                    cache.window_beg = 0  # new sequence
                chunk_size = min(chunk_size, _ring)
                # Correctness cap: after the overflow shift the ring retains ring - chunk
                # prefix rows, and the chunk's first row needs window - 1 of them to see
                # its full sliding window. Capping at ring - window + 1 guarantees that
                # for EVERY row of every chunk (the default ring = window + max_input_len
                # makes this a no-op at the configured chunk size; an explicit smaller
                # sliding_window_overp shrinks the chunk instead of truncating attention).
                _swa_win = getattr(cache, "sliding_window", None)
                if _swa_win is None:
                    for _c in getattr(cache, "caches", ()):
                        if _c is not None:
                            _swa_win = getattr(_c, "sliding_window", 0)
                            break
                if _swa_win:
                    chunk_size = min(chunk_size, max(1, _ring - _swa_win + 1))
                # Decode-time fill cap: the graphed SWA attention wrappers (windowed
                # paged decode + per-row verify) are verified only up to the OLD ring
                # fill (window + 2 pages = 1280). The deeper ring exists for prefill
                # staging (window + chunk); letting decode fill grow into it exposed
                # unverified windowed-graph kv_lens and corrupted generation
                # (debug5.txt: repetition salad a few hundred tokens in, acc 0.5).
                # Small forwards (decode/verify, chunk <= 2 pages) keep the ring
                # trimmed to the verified envelope; prefill chunks use the full ring.
                # The shift stays a deterministic function of (cpos, chunk, cap), so
                # all ranks move in lockstep with no communication (BUG 94).
                _fill_cap = _ring
                if _swa_win and chunk_size <= 2 * _page:
                    _fill_cap = min(_ring, _swa_win + 2 * _page)
                _cpos = cache.current_seq_len - cache.window_beg
                _overflow = _cpos + chunk_size - _fill_cap
                if _overflow > 0:
                    _shift = ((_overflow + _page - 1) // _page) * _page
                    assert 0 < _shift <= _ring, \
                        f"BUG 94: bad SWA ring shift {_shift} (ring {_ring})"
                    if _shift < _ring:
                        for _li in range(len(_swa)):
                            if _swa[_li]:
                                cache.shift_ring(_li, _shift)
                    # _shift == _ring is a full rotation: the ring data is unchanged.
                    cache.window_beg += _shift

            # Limit chunk_size to keep size of attention operation <= max_attention_size

            if (has_flash_attn and not self.config.no_flash_attn) \
                or (has_xformers and not self.config.no_xformers):

                # Can't measure increase in VRAM usage with longer k_len, assume usage is constant
                # for given chunk_size
                pass

            else:

                past_len = cache.current_seq_len
                attn_size = (past_len + remaining_q_len) * remaining_q_len
                max_a = self.config.max_attention_size
                if attn_size > max_a:
                    cs = (math.sqrt(past_len ** 2 + 4 * max_a) - past_len) / 2
                    chunk_size = min(chunk_size, math.floor(cs))

            # Process chunk

            chunk_end = min(chunk_begin + chunk_size, q_len)

            # print(f"Forward chunk length: {chunk_end - chunk_begin}")

            _last_id_only = last_id_only
            _preprocess_only = preprocess_only or (chunk_end < q_len and last_id_only)

            # Long-prefill progress visibility (EXL2_PREFILL_TRACE=0 disables): an 80k
            # prompt is dozens of chunks over PCIe TP -- without this it looks hung.
            _trace = _prefill_trace and q_len > 64
            if _trace:
                import time as _time
                _chunk_t0 = _time.perf_counter()

            r = self.forward_chunk(
                input_ids = input_ids[:, chunk_begin : chunk_end],
                cache = cache,
                input_mask = input_mask,
                preprocess_only = _preprocess_only,
                last_id_only = _last_id_only,
                loras = loras,
                return_last_state = return_last_state and remaining_q_len <= chunk_size,
                return_row_states = return_row_states and remaining_q_len <= chunk_size,
                return_row_aux = return_row_aux,
                position_offsets = position_offsets,
                abort_event = abort_event,
                **kwargs
            )

            # In MP rank mode the collective flag at the top of the loop is the abort
            # authority -- a master-local return here would leave the workers parked in
            # the next chunk's collective.
            if abort_event and abort_event.is_set() \
                    and (self.tp_context is None or self.tp_context.rank is None):
                if _i8_own_staging:
                    cache.end_prefill_staging()
                return

            if _trace:
                # Cost attribution: coll = rank-mode NCCL collectives, attn = eager
                # prefill attention kernels (FI ragged / cuDNN / triton), gemm+other =
                # the remainder (GEMMs incl. reconstruct, norms, rope, KV writes, plans,
                # host). One sync here at the chunk boundary drains the event pairs.
                from exllamav2 import tensor_p as _tp_mod
                from exllamav2 import attn_backend as _ab_mod
                _coll_ms = _tp_mod.prefill_trace_coll_ms()
                _attn_ms = _ab_mod.prefill_trace_attn_ms()
                _chunk_ms = 1000 * (_time.perf_counter() - _chunk_t0)
                print(f"[prefill] {chunk_end}/{q_len} ({_chunk_ms:.0f} ms/chunk; "
                      f"coll {_coll_ms:.0f} attn {_attn_ms:.0f} "
                      f"gemm+other {max(0.0, _chunk_ms - _coll_ms - _attn_ms):.0f})", flush=True)

            if not _preprocess_only:
                if cpu_logits:
                    r["logits"] = r["logits"].cpu()
                result = r["logits"] if result is None else torch.cat((result, r["logits"]), dim = 1)

            # DFlash aux: stage THIS chunk to pinned host memory immediately.
            # GPU-resident accumulation grows linearly with prompt length (71590
            # tokens = 1.15 GB/rank plus cat transients) and OOMs the master
            # mid-prefill (DFLASH5.txt). Grow-only model-level staging (the
            # kv_pack pattern): one ~33 MB D2H per chunk, GPU aux footprint stays
            # at one chunk. Under seqpar this is the rank's hidden shard; the MP
            # parent joins the ranks after the pipe. Safe to hand the staging
            # view out: the pipe pickler copies at send time, and the in-process
            # master join / feed_context consume it before the next prefill.
            ca = r.get("row_aux")
            if ca is not None:
                stag = self._dflash_aux_staging
                if stag is None or stag.shape[1] < q_len or stag.shape[-1] != ca.shape[-1]:
                    stag = torch.empty((bsz, q_len, ca.shape[-1]),
                                       dtype = ca.dtype, pin_memory = True)
                    self._dflash_aux_staging = stag
                stag[:, chunk_begin:chunk_end].copy_(ca, non_blocking = True)
                row_aux = stag

            chunk_begin = chunk_end
            remaining_q_len -= chunk_size
            last_state = r.get("last_state")
            if r.get("row_states") is not None:
                row_states = r["row_states"]

        if _i8_own_staging:
            cache.end_prefill_staging()

        # BUG 117: checkpoint the FI plan buffers right after a long prefill so the
        # guard evidence distinguishes "corrupted during prefill" from "corrupted by
        # the first graphed decode/verify step at long context".
        if q_len > 64 and self.tp_context is not None and cache is not None:
            self.tp_context._fi_guard("post-prefill", cache.current_seq_len, 0)

        if return_row_aux:
            if row_aux is not None:
                # Every chunk's non_blocking D2H must have landed before the
                # staging buffer leaves for the pipe / parent join.
                torch.cuda.current_stream().synchronize()
            return result, last_state, row_aux
        if return_row_states:
            return result, row_states
        if last_state is None:
            return result
        else:
            return result, last_state


    def _fi_tp_graph_plan(self, cache, kv_len: int, q_len: int = 1):
        """
        Refresh the FlashInfer paged attention plans for a graphed step.

        Host-side work, called OUTSIDE the capture region: once before warmup/capture and
        once before every replay (the plan encodes the dynamic kv_len). All hd512 layers
        share the per-device geometry, so one wrapper per (device, geometry) serves every
        layer's captured run(). After planning, order dev0's replay launch after every
        device's plan work so no replayed run() reads a half-written plan.

        Single-pass verify (2026-11-20): the graphed MTP verify window (q_len > 1) is
        served by ONE FI paged-PREFILL wrapper per (device, geometry, q_len), reading the
        KV prefix ONCE for all rows instead of q_len per-row decode passes (BUG 109).
        fa2 causal=True aligns query row i to the last kv_len - q_len + i + 1 keys =
        base + i + 1 -- exactly the verify geometry. FI 0.7.4's prefill wrapper honors
        the same graph contract as the BUG 103 decode fix (use_cuda_graph=True + fixed
        qo_indptr/indptr/indices/last_page_len buffers; plan() refreshes metadata in
        place, backend pinned to fa2), verified against the installed source. The
        BUG 107 segfault was the pre-graph-buffer re-plan pattern; fi_prefill_plan uses
        the fixed-buffer pattern. EXL2_VERIFY_PREFILL=0 forces the per-row decode path,
        which remains the automatic fallback whenever the prefill plan cannot serve a
        shape (e.g. SWA base+q_len beyond the ring depth would misalign causal rows).
        q_len == 1 (decode/bonus) keeps the single FI decode wrapper per geometry.
        """
        return self._fi_tp_graph_plan_batch(
            cache, [kv_len - q_len], [getattr(cache, "window_beg", 0)], q_len, [0])

    def _fi_tp_graph_plan_batch(self, cache, seq_lens, window_begs, q_len, row_map):
        """
        Multi-batch generalization of _fi_tp_graph_plan: one plan per (geometry, wrapper
        kind, batch size) covering every active row. ``seq_lens[b]`` / ``window_begs[b]``
        are row b's absolute position / SWA ring origin; ``row_map[b]`` is its cache row.
        The scalar path delegates here with single-element lists (B == 1), which produces
        exactly the legacy wrapper shapes and plans.
        """
        from exllamav2.attn import use_flashinfer
        from exllamav2.tensor_p import BROADCAST_KV

        cfg = self.config
        ctx = self.tp_context
        if not use_flashinfer(cfg) or cfg.attn_logit_softcapping:
            return
        # Full-hd512 verify (q_len > 1) uses single-pass paged-PREFILL: it reads the KV
        # prefix ONCE for all rows. Per-row split-KV decode (EXL2_VERIFY_PREFILL=0) reads
        # the prefix q_len times -- MEASURED SLOWER at long ctx (79k: verify 31.6->43.0 ms,
        # 2026-11-21), the extra KV traffic outweighs the prefill kernel's small-q
        # inefficiency. Keep prefill the default; =0 remains for A/B only.
        verify_prefill = os.environ.get("EXL2_VERIFY_PREFILL", "1") != "0"
        planned = False
        seen = set()
        B = len(seq_lens)
        for m in self.modules:
            if not isinstance(m, ExLlamaV2Attention):
                continue
            # INT8-primary full layers never use FI wrappers (int8_attn_fast serves
            # decode AND verify, graph-safe without plans). Skip planning entirely:
            # no fp16 KV exists for the wrapper to page-view, and the wrapper VRAM
            # and per-replay plan work are pure waste in this mode.
            if (not m.sliding_window and getattr(cache, "int8_primary", False)
                    and cache.get_kv_i8(m.layer_idx)[0] is not None):
                continue
            # SWA layers keep K/V in a ring: row i's valid length is ring-relative
            # (seq_len - window_beg + i + 1), clamped to the ring depth. Full layers use
            # the absolute causal prefix seq_len + i + 1.
            # FlashInfer window_left is INCLUSIVE: a sliding window of SW keys needs
            # window_left = SW - 1 (matches the eager SWA path attn.py and the torch
            # mask). Using SW attended one extra stale key -> graphed verify logits
            # diverged from solo eager (xcheck maxlogitdiff ~87) -> biased temp>0
            # sampling -> degeneration.
            window_left = (m.sliding_window - 1) if m.sliding_window else -1
            if m.sliding_window:
                bases = [p - wb for p, wb in zip(seq_lens, window_begs)]
                ring_depth = getattr(cache, "kv_state_size", None)
            else:
                bases = list(seq_lens)
                ring_depth = None
            ctx.set_geometry(m.layer_idx)
            split = ctx.get_split(BROADCAST_KV)
            k_cache, _ = cache.get_kv_state(m.layer_idx, 1, 0, 1)
            for idx, (dev, a, b) in enumerate(split):
                nq = (b - a) * m.num_key_value_groups
                nkv = b - a
                base_key = (dev, nq, nkv, m.head_dim, m.scaling, window_left)
                # SWA layers stay on windowed DECODE wrappers (the path verified
                # 2026-11-19): one batched B*q_len-request fi_decode_rows wrapper at
                # every batch size (B == 1 included -- one run per layer instead of
                # q_len per-row runs). The windowed paged-PREFILL verify wrapper under
                # CUDA graphs was never validated (debug5.txt). Full hd512 layers keep
                # the single-pass prefill verify at every batch size.
                if q_len > 1 and verify_prefill and not m.sliding_window:
                    pf_key = base_key + (q_len, B)
                    if pf_key not in seen:
                        seen.add(pf_key)
                        if ctx.fi_prefill_plan(pf_key, [base + q_len for base in bases],
                                               k_cache[idx], row_map):
                            planned = True
                    if pf_key in ctx.fi_prefill:
                        continue   # single-pass plan ready; skip the per-row wrappers
                if q_len > 1 and m.sliding_window:
                    rows_key = base_key + (q_len, B)
                    if rows_key not in seen:
                        seen.add(rows_key)
                        kv_lens, rows = [], []
                        for bi, base in enumerate(bases):
                            for i in range(q_len):
                                kv_lens.append(min(base + i + 1, ring_depth))
                                rows.append(row_map[bi])
                        if ctx.fi_decode_rows_plan(rows_key, kv_lens, rows, k_cache[idx]):
                            planned = True
                    if rows_key in ctx.fi_decode_rows:
                        continue
                for i in range(q_len):
                    # Per-row decode wrappers serve B == 1 (decode + legacy SWA verify).
                    # At B > 1 multi-token steps are served by the rows wrapper above;
                    # if it could not plan, the attn torch-mask fallback takes over.
                    if B > 1 and q_len > 1:
                        break
                    row_kvs = [base + i + 1 for base in bases]
                    if ring_depth:
                        row_kvs = [min(l, ring_depth) for l in row_kvs]
                    if any(l < 1 for l in row_kvs):
                        continue
                    if q_len == 1:
                        key = base_key + (B,)
                    else:
                        key = base_key + (i, B)
                    if key in seen:
                        continue
                    seen.add(key)
                    if ctx.fi_decode_plan(key, row_kvs, k_cache[idx], row_map):
                        planned = True
        if planned:
            ctx.fi_decode_order_before_replay()

    def _tp_graph_invalidate_shape(self, batch_size: int, q_len: int):
        """BUG 117 envelope invalidation scoped to one (B, q_len) graph family.

        Releases every runner of the family (all state modes / last_id_only -- they
        share the family's FI wrappers) BEFORE dropping those wrappers, so no live
        graph references freed wrapper buffers (the BUG 117 ordering). Wrapper
        scoping mirrors _fi_tp_graph_plan_batch's keys:

          * q_len == 1: the 7-tuple fi_decode entries (..., B), shared by the decode
            and bonus-commit graphs of that batch size.
          * q_len > 1: fi_prefill / fi_decode_rows entries (..., q_len, B); at B == 1
            also the 8-tuple per-row fi_decode verify wrappers (..., row, 1). The
            7-tuple B=1 decode wrappers belong to the (1, 1) family and stay.

        Graphs stay enabled; the family recaptures on demand at the new kv, reusing
        its stable P2P arena epoch (TPGraphManager.p2p_epoch)."""
        ctx = self.tp_context
        for key in list(self.tp_graph.runners.keys()):
            if key[0] == batch_size and key[1] == q_len:
                self.tp_graph.runners.pop(key).release()
        for key in list(ctx.fi_prefill.keys()):
            if key[-2] == q_len and key[-1] == batch_size:
                del ctx.fi_prefill[key]
        for key in list(ctx.fi_decode_rows.keys()):
            if key[-2] == q_len and key[-1] == batch_size:
                del ctx.fi_decode_rows[key]
        for key in list(ctx.fi_decode.keys()):
            if q_len == 1:
                if len(key) == 7 and key[-1] == batch_size:
                    del ctx.fi_decode[key]
            elif batch_size == 1 and len(key) == 8 and key[-1] == 1:
                del ctx.fi_decode[key]

    def _tp_graph_step(
        self,
        input_ids: torch.Tensor,
        cache,
        attn_params,
        past_len: int,
        batch_size: int,
        seq_len: int,
        last_id_only: bool,
        return_last_state: bool = False,
        return_row_states: bool = False,
        return_row_aux: bool = False,
        row_map = None,
        window_begs = None
    ):
        """
        Run one graphed step -- decode, MTP bonus commit (return_last_state), or the MTP
        verify window (seq_len > 1, optionally with per-row states) -- capturing it on
        first use.

        ``past_len`` is an int for the single-sequence path, or a per-batch-row list for
        a multi-batch step (with ``window_begs`` the matching SWA ring origins and
        ``row_map`` the cache row each batch row writes). The graph is position-agnostic:
        every per-row value lives in device tensors refreshed before each replay, so one
        captured (batch, q_len) graph serves any active slot subset of that size.

        Returns the runner on success, or None to fall back to the eager path. Falling back is
        always safe: the eager path is unchanged and still correct.

        The step is captured with ``graph_mode`` set on the TP context, which makes the
        collectives device-resident and takes the sequence position from device memory. The
        graph is therefore position-agnostic and replays at every subsequent token.

        Change98: every static buffer (embedded input, logits, optional pre-final-norm state)
        is owned by the RUNNER, not the model. The decode (1,1), bonus (1,1,state) and verify
        (1,W) graphs coexist; a single shared input buffer would be reallocated whenever the
        shape changes and leave every previously captured graph reading freed memory -- the
        BUG 44 failure mode generalized to multiple runners.
        """
        ctx = self.tp_context

        # Graph state mode: 0 = logits only (decode / plain verify), 1 = last row's
        # pre-final-norm state (bonus-commit fallback), 2 = EVERY row's pre-final-norm
        # state (bonus-eliminated verify: the driver seeds the next round's drafter from
        # the last accepted draft's state, so no separate bonus forward is needed),
        # 3 = EVERY row's DFlash aux states (the 6 target-layer residual taps, full
        # width -- only reached for decode_repl steps, see forward_chunk's gate).
        state_mode = 3 if return_row_aux else \
            (2 if return_row_states else (1 if return_last_state else 0))
        aux_taps = list(self._dflash_aux_taps or []) if state_mode == 3 else []

        # The embedding module holds its weight on the CPU (embedding.py:55 keeps it there
        # deliberately), so the token lookup runs on the host and produces a host tensor. Host work
        # is not recorded by a CUDA graph, so the graph must NOT contain the embedding: if it did,
        # replay would re-emit the capture-time embedding and every step would be one token behind
        # (the lag-by-one symptom).
        #
        # Instead the embedding is run eagerly on every step, exactly as the eager path does, and its
        # result is copied into a static device buffer that the graph reads. The graph therefore
        # starts at module 1 and is driven purely by that buffer plus `pos`.
        self._tp_graph_embed = None
        embed_module = self.modules[0]
        first_module = 1

        # Static device buffer for the embedded input: allocated once per runner (per shape),
        # never reallocated -- the captured graph holds its address. BUG 44 was a shared buffer
        # being reallocated on a shape mismatch, making every replay re-read the capture-time
        # embedding; with multiple coexisting runners (decode / bonus / verify) the buffer must
        # be per-runner for the same reason.

        def run_embedding(ids: torch.Tensor) -> torch.Tensor:
            """Embedding lookup on the host, result moved to dev0. Mirrors the eager path."""
            with torch.inference_mode():
                h = embed_module.forward(ids, cache = cache, attn_params = attn_params,
                                        past_len = past_len, loras = None)
            if h.device.type == "cpu":
                h = h.to(ctx.device, non_blocking = False)
            return h

        # Device-side sequence position. RoPE and the KV write both read this, so it must be
        # updated *before* replay and must not be reallocated (the graph holds its address).
        if ctx.pos is None:
            ctx.pos = torch.zeros((1,), dtype = torch.int32, device = ctx.device)

        # BUG 47: pass the cache so a runner captured against a different KV cache is
        # rejected instead of replayed against freed buffers.
        runner = self.tp_graph.get(batch_size, seq_len, cache,
                                   state = state_mode, last_id_only = last_id_only)

        # BUG 117: a whole-step graph captured at short context corrupts memory when it is
        # replayed far beyond that length -- the FlashInfer graph-mode attention launch is
        # baked at capture and is not valid at arbitrarily longer kv (the bonus graph captured
        # at ~3k sprayed fp16 over the FI metadata buffers on its first 122k replay; a graph
        # captured AT long context replays fine). Recapture when this step's kv falls outside
        # the capture envelope. This fires on large context jumps (a new job after a long
        # prefill, or a cache reset), not per token: normal generation grows kv by a few
        # hundred tokens, well inside the envelope, so the amortized recapture cost is tiny.
        # Per-row positions: scalar legacy path vs multi-batch lists.
        _batched = isinstance(past_len, (list, tuple))
        _pos_list = list(past_len) if _batched else None
        _wb_list = list(window_begs) if _batched else None
        _rmap = list(row_map) if row_map is not None else list(range(batch_size))
        _max_past = max(_pos_list) if _batched else past_len

        if runner is not None:
            _kv = _max_past + seq_len
            _cap_kv = getattr(runner, "capture_kv", 0)
            if _kv > _cap_kv + TP_GRAPH_KV_ENVELOPE or _kv < _cap_kv - TP_GRAPH_KV_ENVELOPE:
                # Context-regime change for THIS shape: recapture only its (B, q_len)
                # family at the new length (graphs stay enabled). The old global clear
                # thrashed under multi-batch -- concurrent jobs at different context
                # lengths change the batch composition (and its max kv) at every job
                # boundary, rebuilding every shape each time. Other shapes keep their
                # own envelopes, wrappers and P2P arena regions; only the violating
                # shape's launch is capture-baked at the wrong kv.
                self._tp_graph_invalidate_shape(batch_size, seq_len)
                runner = None

        def stage_inputs(input_buf):
            # Write every dynamic input on dev0's TP stream, ordered before the graph launch.
            # flash_attn_with_kvcache takes cache_seqlens as a device tensor; fill the STABLE
            # per-device tensors owned by TPContext (the graph holds their addresses), not the
            # per-call copies on attn_params, which would freeze the capture-time length.
            _dev0_stream = ctx.streams_dev[ctx.device]
            _prev = torch.cuda.current_stream(ctx.device)
            torch.cuda.set_stream(_dev0_stream)
            try:
                _h0 = run_embedding(input_ids)
                input_buf.copy_(_h0)
                ctx.set_pos(past_len)
                if _batched:
                    ctx.set_ring_pos([p - wb for p, wb in zip(_pos_list, _wb_list)])
                else:
                    ctx.set_ring_pos(past_len - getattr(cache, "window_beg", 0))
                ctx.set_row_map(_rmap)
                for _pl in ctx.past_len_tp.values():
                    ctx.fill_tp(_pl, past_len)
            finally:
                torch.cuda.set_stream(_prev)

        if runner is None:
            # First step of this shape: allocate the runner's static buffers, then capture.
            _h0 = run_embedding(input_ids)
            input_buf = torch.empty_like(_h0, device = ctx.device)
            # The bonus graph also returns the pre-final-norm hidden state. The copy into this
            # static buffer is recorded INSIDE the graph, so every replay refreshes it.
            state_buf = None
            aux_buf = None
            if state_mode == 1:
                state_buf = torch.empty((batch_size, 1, self.config.hidden_size),
                                        dtype = _h0.dtype, device = ctx.device)
            elif state_mode == 2:
                state_buf = torch.empty((batch_size, seq_len, self.config.hidden_size),
                                        dtype = _h0.dtype, device = ctx.device)
            elif state_mode == 3:
                # DFlash aux: one full-width residual snapshot per tap, in
                # target_layer_ids order (the caller's feed_context layout).
                aux_buf = torch.empty((batch_size, seq_len,
                                       len(aux_taps) * self.config.hidden_size),
                                      dtype = _h0.dtype, device = ctx.device)
            out = {}

            def step_fn():
                # Start from the static embedded-input buffer; the embedding itself is host work
                # and therefore outside the graph (see the note above).
                x = input_buf
                # Sequence-parallel: the residual stream runs as this rank's hidden shard;
                # slice the replicated embedding locally (no collective). Decode-sized
                # steps keep the full width (decode_repl) -- no shard, no final gather.
                _repl_step = ctx.decode_repl and seq_len <= 32
                if ctx.seqpar and not _repl_step:
                    x = ctx.shard_residual(x)
                for idx, module in enumerate(self.modules):
                    if idx < first_module:
                        continue
                    # BUG 81: a rank-local worker without lm_head must stop before it, exactly as
                    # forward_chunk does (BUG 77). Otherwise capture aborts with
                    # "Layer model.language_model.lm_head has no data" and graphs get disabled.
                    if idx == self.head_layer_idx and not getattr(self, "_tp_has_head", True):
                        break
                    # Sequence-parallel: reassemble the full hidden width before the final
                    # norm / head / state extraction (the only full-width consumers left).
                    # decode_repl steps are already full-width -- gathering again would
                    # allgather the FULL tensor to world*width (garbage).
                    if ctx.seqpar and not _repl_step and idx == self.head_layer_idx - 1:
                        x = ctx.gather_residual(x)
                    # MTP state: capture BEFORE the final norm (module head_layer_idx - 1)
                    # runs -- the same point forward_chunk's eager extract_state_indices clones
                    # at. Mode 1 keeps the last row (bonus commit); mode 2 keeps every row
                    # (bonus-eliminated verify).
                    if state_mode and state_mode != 3 and idx == self.head_layer_idx - 1:
                        if state_mode == 1:
                            state_buf.copy_(x.narrow(-2, -1, 1))
                        else:
                            state_buf.copy_(x)
                    # DFlash aux taps: same clone point as the eager extraction
                    # (before module idx runs = after the tapped decoder layer's
                    # last module). The forward_chunk gate guarantees full width
                    # here (decode_repl).
                    if state_mode == 3 and idx in aux_taps:
                        j = aux_taps.index(idx)
                        H = self.config.hidden_size
                        aux_buf[:, :, j * H : (j + 1) * H].copy_(x)
                    if idx == self.head_layer_idx and last_id_only:
                        x = x.narrow(-2, -1, 1)
                    # No safe_move_tensor here: in graph mode the residual stream stays on the
                    # primary device for the whole step, and safe_move_tensor host-synchronises,
                    # which is illegal inside a capture region. BUG 43.
                    x = module.forward(
                        x,
                        cache = cache,
                        attn_params = attn_params,
                        past_len = past_len,
                        loras = None,
                    )
                # Keep a reference to the logits so they are not freed; the graph writes into
                # this buffer on every replay.
                out["logits"] = x

            # attn_params.prep_tp() must run BEFORE capture: it calls safe_move_tensor, which
            # does a torch.cuda.synchronize(), and a host sync is illegal inside a capture
            # region. Warming it here means the in-capture call finds its per-device tensors
            # already built for every geometry and returns immediately.
            ctx.graph_mode = False
            for _li in range(self.config.num_hidden_layers):
                ctx.set_geometry(_li)
                attn_params.prep_tp(self)
            ctx.set_geometry(0)

            # BUG 100: capture invalidation is often a transient allocator/GC race, so
            # retry once with a fresh runner before giving up. capture() force-ends the
            # capture and resets the TP counter protocol on failure, so the retry (and
            # the eager fallback if it also fails) starts from a clean protocol state.
            runner = None
            _capture_err = None
            for _attempt in range(2):
                runner = TPGraphRunner(step_fn, ctx.device,
                                       (batch_size, seq_len, state_mode, last_id_only),
                                       tp_context = ctx)
                # Let capture() restore the KV cache position after its warmup iterations. The
                # warmup runs the real step, which advances current_seq_len and writes KV rows.
                runner.cache = cache
                runner.input_buf = input_buf
                runner.state_buf = state_buf
                runner.aux_buf = aux_buf
                # capture() reuses the manager's stable per-shape graph pool (graph_pool(key))
                # so KV-envelope recaptures do not ratchet reserved VRAM. The runner has no
                # pool of its own; give it the manager reference (None -> fresh pool fallback).
                runner.tp_graph = self.tp_graph
                # Stage the real inputs so warmup and capture run at the caller's position with the
                # caller's tokens (the warmup KV writes are idempotent: same rows, same values).
                stage_inputs(input_buf)
                # hd>256 FlashInfer decode plans must exist before warmup/capture: run() is
                # captured, plan() is host work and cannot run inside the capture region.
                if _batched:
                    self._fi_tp_graph_plan_batch(cache, _pos_list, _wb_list, seq_len, _rmap)
                else:
                    self._fi_tp_graph_plan(cache, past_len + seq_len, q_len = seq_len)
                ctx.graph_mode = True
                # Per-graph P2P slot region: warmup + capture bake slots from the
                # shape's STABLE permanent epoch, so this graph never shares arena
                # slots with another persistent graph (the in-graph clobber bug) and
                # a recapture of the same shape reuses its region instead of leaking
                # a fresh one (arena exhaustion under multi-batch envelope churn).
                # Restored to eager (0) after.
                _p2p = getattr(ctx, "p2p", None)
                if _p2p is not None:
                    self.tp_graph.p2p_epoch(
                        self.tp_graph.key_for(batch_size, seq_len, state_mode, last_id_only),
                        _p2p)
                try:
                    runner.capture()
                    break
                except Exception as e:
                    _capture_err = e
                    if os.environ.get("EXL2_TP_GRAPH_DEBUG", "0") != "0":
                        import traceback
                        print("[TP graph] capture traceback:\n" + traceback.format_exc(), flush=True)
                    ctx.graph_mode = False
                    runner = None
                finally:
                    if _p2p is not None:
                        _p2p.end_graph()
            if runner is None:
                if seq_len > 1:
                    # BUG 107: a verify-window capture failure must NOT disable the proven
                    # decode/bonus graphs (that would regress the whole path to eager).
                    # Fall back to eager VERIFY only and skip future verify captures.
                    self._tp_graph_verify_failed = True
                    print(f"[TP graphs] verify capture failed ({type(_capture_err).__name__}: "
                          f"{_capture_err}); eager verify fallback, decode graphs kept", flush=True)
                    return None
                self.tp_graph.disable(f"capture failed: {type(_capture_err).__name__}: {_capture_err}")
                print(f"[TP graphs] disabled: {self.tp_graph.disabled_reason}", flush=True)
                return None
            ctx.graph_mode = False
            runner.logits = out["logits"]
            # BUG 117: remember the kv length this graph was captured at, so a later step far
            # outside this envelope recaptures instead of replaying an invalid baked launch.
            runner.capture_kv = _max_past + seq_len
            self.tp_graph.add(batch_size, seq_len, runner,
                              state = state_mode, last_id_only = last_id_only)

            # Capture only *records* the work; it does not execute it. Replay once so this first
            # decode step actually produces logits. Without this the caller reads the
            # still-uninitialised logits buffer and the first token comes back as garbage (0).
            ctx.graph_mode = True
            try:
                with torch.cuda.device(ctx.device):
                    prev_stream = torch.cuda.current_stream(ctx.device)
                    torch.cuda.set_stream(ctx.streams_dev[ctx.device])
                    try:
                        runner.replay()
                    finally:
                        torch.cuda.set_stream(prev_stream)
            finally:
                ctx.graph_mode = False
            return runner

        # Subsequent steps: update the dynamic inputs, then replay.
        #
        # The graph is launched on dev0's TP stream (that is the stream the capture was made on),
        # so every dynamic input must be written on that same stream and the stream must still be
        # current when replay() is called. Writing on a different stream, or restoring the previous
        # stream before replaying, leaves a race in which the graph reads the previous token's id
        # and position -- which presents exactly as the bug-39 repetition-loop signature.
        ctx.graph_mode = True
        try:
            dev0_stream = ctx.streams_dev[ctx.device]
            # set_stream() only affects the CURRENT device, so the device must be selected too.
            # Without this the copy_ below runs on whichever device the previous module left
            # current, and the graph (launched on dev0) reads the buffer before the copy lands.
            # That is the lag-by-one-token symptom.
            with torch.cuda.device(ctx.device):
                prev_stream = torch.cuda.current_stream(ctx.device)
                torch.cuda.set_stream(dev0_stream)
                try:
                    # Run the embedding eagerly (it is host work, so it is not in the graph) and
                    # stage the result in the static device buffer the graph reads.
                    _h0 = run_embedding(input_ids)
                    runner.input_buf.copy_(_h0)
                    ctx.set_pos(past_len)
                    if _batched:
                        ctx.set_ring_pos([p - wb for p, wb in zip(_pos_list, _wb_list)])
                    else:
                        ctx.set_ring_pos(past_len - getattr(cache, "window_beg", 0))
                    ctx.set_row_map(_rmap)
                    for _pl in ctx.past_len_tp.values():
                        ctx.fill_tp(_pl, past_len)
                    # Refresh the hd>256 FlashInfer decode plans for this step's kv_len
                    # (host work, outside the graph) and order the replay after them.
                    if _batched:
                        self._fi_tp_graph_plan_batch(cache, _pos_list, _wb_list, seq_len, _rmap)
                    else:
                        self._fi_tp_graph_plan(cache, past_len + seq_len, q_len = seq_len)
                    # Replay while dev0's stream is still current: CUDAGraph.replay() launches on
                    # the torch current stream, and the graph's nodes were captured on this one.
                    runner.replay()
                    # BUG 117: pinpoint whether a long-context graph replay is the one-shot
                    # writer that sprays the FI metadata buffers (guard syncs the device).
                    self.tp_context._fi_guard(f"post-replay{runner.key}", _max_past + seq_len, seq_len)
                finally:
                    torch.cuda.set_stream(prev_stream)
        finally:
            ctx.graph_mode = False
        return runner

    @torch.inference_mode()
    def forward_tp_batch(
        self,
        input_ids: torch.Tensor,
        cache,
        seq_lens: list,
        window_begs: list,
        row_map: list,
        last_id_only: bool = True,
        graph_verify: bool = False,
        return_last_state: bool = False,
        return_row_states: bool = False,
    ):
        """
        One multi-batch TP step over independent cache rows (the MP slot scheduler).

        ``input_ids`` is (B, q_len); batch row b belongs to cache row ``row_map[b]`` at
        position ``seq_lens[b]`` with SWA ring origin ``window_begs[b]``. Every rank runs
        the same per-row ring prep and the same graphed step, so the collectives stay in
        lockstep. The step does NOT advance the bookkeeping: the caller (MTP driver or
        streaming proxy) owns seq_lens, exactly as the scalar graph path leaves
        current_seq_len to its caller.

        Falls back to per-row scalar eager forwards (persistent row views, the verified
        batch-1 path) if the batched capture fails -- always correct, just slower.
        """
        B, q_len = input_ids.shape
        assert B == len(seq_lens) == len(window_begs) == len(row_map), \
            "forward_tp_batch: per-row bookkeeping must match the batch"

        # Per-row SWA ring prep: the decode-size fill cap and page-granular overflow
        # shift from model.forward's chunk loop, scoped to each row so slots at different
        # lengths shift independently. Deterministic in (seq_lens, window_begs, q_len), so
        # every rank computes the same shifts with no communication (BUG 94 protocol).
        _swa_p = self._swa_ring_params(cache)
        if _swa_p is not None:
            _ring, _page, _swa = _swa_p
            _swa_win = getattr(cache, "sliding_window", None)
            if not _swa_win:
                for _c in getattr(cache, "caches", ()):
                    if _c is not None:
                        _swa_win = getattr(_c, "sliding_window", 0)
                        break
            _fill_cap = _ring
            if _swa_win and q_len <= 2 * _page:
                _fill_cap = min(_ring, _swa_win + 2 * _page)
            for b in range(B):
                r = row_map[b]
                _cpos = seq_lens[b] - window_begs[b]
                _overflow = _cpos + q_len - _fill_cap
                if _overflow > 0:
                    _shift = ((_overflow + _page - 1) // _page) * _page
                    assert 0 < _shift <= _ring, \
                        f"forward_tp_batch: bad SWA ring shift {_shift} (ring {_ring})"
                    if _shift < _ring:
                        for _li in range(len(_swa)):
                            if _swa[_li]:
                                cache.shift_ring_row(_li, r, _shift)
                    window_begs[b] += _shift
                    # Persist into the cache row (not just the per-batch list) so the
                    # next round's gather/draft and the eager row-view fallback both
                    # see the shifted origin. Other slots' rows are untouched.
                    cache.window_begs[r] = window_begs[b]

        attn_params = ExLlamaV2Attention.Params(B, q_len, seq_lens[0], None, None)
        # Batch metadata for the EAGER warmup passes inside graph capture: the eager
        # rope/KV-write path must use per-row positions and the slot row map, or the
        # warmup would scribble dummy-position K/V into the wrong live cache rows.
        attn_params.batch_seq_lens = list(seq_lens)
        attn_params.batch_window_begs = list(window_begs)
        attn_params.batch_row_map = list(row_map)
        runner = self._tp_graph_step(
            input_ids, cache, attn_params, list(seq_lens), B, q_len, last_id_only,
            return_last_state = return_last_state,
            return_row_states = return_row_states,
            row_map = row_map, window_begs = window_begs,
        )
        if runner is not None:
            if return_row_states or return_last_state:
                return runner.logits, runner.state_buf
            return runner.logits

        # Eager fallback: run each row through the verified scalar path on its persistent
        # row view. use_graphs=False keeps row-view captures out of the graph manager (a
        # row view is a different cache identity than the batched graphs' full cache).
        logits_rows, state_rows = [], []
        for b in range(B):
            rv = cache.row_view(row_map[b])
            r = self.forward(
                input_ids[b:b + 1], cache = rv, last_id_only = last_id_only,
                return_last_state = return_last_state,
                return_row_states = return_row_states,
                use_graphs = False,
            )
            if return_row_states or return_last_state:
                lg, st = r
                state_rows.append(st)
            else:
                lg = r
            logits_rows.append(lg)
        logits = torch.cat(logits_rows, dim = 0)
        if return_row_states or return_last_state:
            return logits, torch.cat(state_rows, dim = 0)
        return logits

    def gdn_rewind_to(self, cache, pos: int):
        """
        Restore the hybrid backbone's gated-delta-net recurrent state to the prefix
        ending at `pos`, using the staging recorded by the last MTP verify forward.
        Fans out across per-device caches in single-process TP. newplan.md P6.
        """
        from exllamav2.qwen4_exp import ExLlamaV2PLELayer
        caches = cache.caches if hasattr(cache, "caches") else [cache]
        for c in caches:
            if c is None:
                continue
            if getattr(c, "gdn_stage", None):
                for m in self.modules:
                    if isinstance(m, ExLlamaV2GatedDeltaNet):
                        m.rewind_state(c, pos)
            if getattr(c, "ple_stage", None):
                for m in self.modules:
                    if isinstance(m, ExLlamaV2PLELayer):
                        m.rewind_state(c, pos)
                c.ple_stage = None
            c.gdn_staging = False
            c.gdn_stage = {}
            c.gdn_stage_start = -1
            c.gdn_stage_len = 0

    @torch.inference_mode()
    # @profile
    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        input_mask: torch.Tensor | None = None,
        preprocess_only: bool = False,
        last_id_only: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        return_last_state: bool = False,
        return_row_states: bool = False,
        return_row_aux: bool = False,
        position_offsets: torch.Tensor | None = None,
        abort_event: threading.Event | None = None,
        attn_params: ExLlamaV2Attention.Params | ExLlamaV2Attention.PagedParams | None = None,
        extract_state_indices: list[int] | None = None,
        **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len = input_ids.shape
        past_len = 0
        if cache is not None:
            past_len = cache.current_seq_len

        assert self.config.max_output_len is None or \
            preprocess_only or \
            last_id_only or \
            seq_len <= self.config.max_output_len, \
            "seq_len exceeds max_output_len"

        if self.config.arch.lm.mrope and kwargs.get("indexed_embeddings"):
            assert attn_params is not None and (
                attn_params.rope_offsets is not None or
                attn_params.get_alt_rope_embed("cpu") is not None
            ), "MRoPE model requires precomputed RoPE when using indexed embeddings."

        # Ensure streams are always set in the current thread

        set_device_streams()

        # Output

        r = {}
        extract_state_indices = set(extract_state_indices or [])
        if return_last_state or return_row_states:
            extract_state_indices.add(self.head_layer_idx - 1)
        # DFlash aux taps: the residual BEFORE each tap module runs (= the previous
        # decoder layer's output, the HF hidden_states[layer_id + 1] convention).
        # Under seqpar prefill these clones are this rank's hidden shard; the MP
        # parent joins the ranks along the hidden dim. decode_repl steps (verify)
        # are full width on every rank.
        if return_row_aux:
            assert self._dflash_aux_taps, "return_row_aux requires DFlash aux taps (enable_mtp)"
            extract_state_indices.update(self._dflash_aux_taps)
        if extract_state_indices:
            r["states"] = {}

        # assert cache is None or isinstance(cache, list) or batch_size <= cache.batch_size

        x = input_ids

        # Qwen4-Exp PLE: the n-gram lookup needs the original token ids (the module
        # loop only passes the residual). Stash them for the PLE layer.
        if cache is not None and getattr(self.config.arch.lm, "ple_ngram", False):
            cache._ple_ids = input_ids

        if not attn_params:
            attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, past_len, input_mask, position_offsets)
        else:
            if not isinstance(attn_params, ExLlamaV2Attention.PagedParams):
                past_len = attn_params.past_len
                cache.current_seq_len = past_len

        device = self.modules[0].device_idx

        # CUDA-graph decode path (BUG 43). Only the single-token decode step is captured:
        # prefill shapes vary with the prompt and are not the bottleneck. Everything the step
        # needs that used to come from the host (the sequence position) is read from
        # tp_context.pos instead, so one graph replays correctly at every step.
        graph_runner = None
        # Change98: the graph path also serves the two MTP forwards.
        #  * bonus commit: seq_len == 1 with return_last_state -- the graph copies the
        #    pre-final-norm state into a static buffer. Only the head-1 extraction is
        #    graphable; any other extract_state_indices falls back to eager.
        #  * verify window: a fixed W-token forward marked explicitly by the caller
        #    (graph_verify), so arbitrary prefill chunks are never captured.
        graph_verify = kwargs.pop("graph_verify", False)
        # use_graphs=False forces the eager path (row-view forwards in the multi-batch
        # fallback: a row view is a different cache identity than the batched graphs').
        use_graphs = kwargs.pop("use_graphs", True)
        graph_state = (return_last_state or return_row_states) \
            and extract_state_indices == {self.head_layer_idx - 1}
        # DFlash aux graphs need FULL-WIDTH taps: decode_repl steps (q_len <= 32)
        # run the residual replicated, so the captured clones are the complete
        # hidden state. Seqpar-sharded steps (prefill, or decode_repl disabled)
        # fall back to the eager extraction, whose per-rank shards the MP parent
        # joins. graph_state and graph_aux are mutually exclusive by construction.
        # return_last_state + aux (the heal re-commit) also stays eager: mode 3
        # captures only the aux taps, not the final-norm state the caller needs.
        graph_aux = return_row_aux and not return_last_state and (
            self.tp_context is None
            or not self.tp_context.seqpar
            or (self.tp_context.decode_repl and seq_len <= 32))
        if (
            use_graphs
            and self.tp_context is not None
            and cache is not None
            # Hybrid GDN/PLE caches run EAGER: the recurrent-state staging clones
            # (MTP verify) and per-layer state advances are not CUDA-graph-safe.
            and not hasattr(cache, "gdn_conv_states")
            and (seq_len == 1 or graph_verify)
            and not preprocess_only
            and (not extract_state_indices or graph_state or graph_aux)
            and not loras
            and not kwargs.get("indexed_embeddings")
            and self.tp_graph is not None
            and not (graph_verify and self._tp_graph_verify_failed)
            and self.tp_graph.can_capture(batch_size, seq_len, loras, allow_multi = graph_verify)
        ):
            graph_runner = self._tp_graph_step(
                input_ids, cache, attn_params, past_len, batch_size, seq_len, last_id_only,
                return_last_state = graph_state and return_last_state,
                return_row_states = graph_state and return_row_states,
                return_row_aux = graph_aux
            )
            if graph_runner is not None:
                # The graph wrote its logits into the runner's static buffer; hand it back.
                x = graph_runner.logits
                if graph_state:
                    r["states"][self.head_layer_idx - 1] = graph_runner.state_buf
                    if return_row_states:
                        r["row_states"] = graph_runner.state_buf
                    else:
                        r["last_state"] = graph_runner.state_buf
                if graph_aux:
                    r["row_aux"] = graph_runner.aux_buf

        if graph_runner is None:

            for idx, module in enumerate(self.modules):

                # Sequence-parallel: reassemble the full hidden width before the final
                # norm / head / state extraction (mirrors the graphed step_fn).
                # decode_repl steps are already full-width (see attn/mlp.forward_tp_old).
                _repl_chunk = self.tp_context is not None and self.tp_context.decode_repl \
                    and seq_len <= 32
                if self.tp_context is not None and self.tp_context.seqpar \
                        and not _repl_chunk and idx == self.head_layer_idx - 1:
                    x = self.tp_context.gather_residual(x)

                if idx == self.head_layer_idx and last_id_only:
                    x = x.narrow(-2, -1, 1)

                if idx in extract_state_indices:
                    r["states"][idx] = x.clone()
                    if idx == self.head_layer_idx - 1:
                        if return_row_states:
                            r["row_states"] = r["states"][idx]
                        else:
                            r["last_state"] = r["states"][idx]

                # Respect abort signal

                if abort_event and abort_event.is_set():
                    return None, None

                # Onward

                n_device = module.device_idx
                if n_device is not None and n_device != device and n_device >= 0:
                    x = safe_move_tensor(x, n_device, non_blocking = True)

                # BUG 77: a rank-local worker that does not own lm_head must not run it.
                # Only rank 0 has the head loaded; the others stop after the last layer.
                if idx == self.head_layer_idx and not getattr(self, "_tp_has_head", True):
                    break

                x = module.forward(x, cache = cache, attn_params = attn_params, past_len = past_len, loras = loras, **kwargs)

                # Sequence-parallel: the replicated embedding output enters the block
                # stack as this rank's hidden shard (pure local slice, no collective).
                if idx == 0 and self.tp_context is not None and self.tp_context.seqpar \
                        and not _repl_chunk:
                    x = self.tp_context.shard_residual(x)

                if os.environ.get("EXL2_TRACE_X") and not isinstance(x, torch.Tensor):
                    print(f"[TRACE] module {idx} {type(module).__name__} key={getattr(module, 'key', '?')} "
                          f"returned {type(x).__name__} "
                          f"len={len(x) if isinstance(x, (list, tuple)) else '-'} "
                          f"shapes={[tuple(t.shape) for t in x] if isinstance(x, (list, tuple)) else '-'}",
                          flush=True)

                if preprocess_only and idx == self.last_kv_layer_idx:
                    x = None
                    break

            # DFlash aux: concatenate the tap snapshots in target_layer_ids order.
            # Seqpar prefill: each clone is this rank's hidden shard, so the result
            # is (B, rows, n_features * hidden_local) -- the MP parent joins ranks.
            if return_row_aux:
                r["row_aux"] = torch.cat(
                    [r["states"][t] for t in self._dflash_aux_taps], dim = -1)

        # Advance cache

        if cache is not None:
            cache.current_seq_len += seq_len

        # Final sync if gathering logits
        #
        # Only when there is something to hand back to the host. This used to run on every TP
        # forward, including the CUDA-graph decode path, which drained all four devices to the
        # host on every token and made the GPUs unable to overlap:
        #
        #   decode 13.31 ms/token wall, dev0 GPU busy 29.59 ms/token  (busy/wall = 2.22)
        #   one thread pegged at 100% of wall, total CPU exactly 1.00 cores
        #   profiler: only 0.39 ms/token of real aten work -> the thread was spinning in this
        #   synchronise, not dispatching.
        #
        # When `x` is None (preprocess_only) or the step already produced its result through a
        # captured graph, nothing is read on the host here and the sync is pure stall.

        if self.tp_context and x is not None:
            self.tp_context.wait_streams(x.device.index)

        if x is not None and x.is_cuda:
            context = self.get_device_context(x.device.index)
            torch.cuda.set_stream(context.stream)

        # Apply logit scale

        # if x is not None and self.config.logit_scale != 1:
        #     x.mul_(self.config.logit_scale)

        if x is not None and self.config.final_logit_softcapping:
            ext_c.softcap_(x, self.config.final_logit_softcapping)

        # Set padding logits to -inf

        if x is not None:
            head_padding = self.modules[-1].padding
            if head_padding > 0:
                x[:, :, -head_padding:] = -65504.
            r["logits"] = x

        return r
