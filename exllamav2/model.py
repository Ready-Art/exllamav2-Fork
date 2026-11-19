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

        # Build model

        cfg = self.config
        if archparams is None: archparams = cfg.arch.lm
        self.archparams = archparams

        # `embed_key` / `norm_key` are relative to lm_prefix. Gemma-4 nests them directly
        # under `model.language_model.`, so the old hard-coded "model.embed_tokens" produced
        # `model.language_model.model.embed_tokens`, which does not exist. PLAN.md §2.1.
        emb = ExLlamaV2Embedding(self, cfg.arch.lm_prefix + getattr(archparams, "embed_key", "model.embed_tokens"))
        self.modules += [emb]

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
            else:
                attn = ExLlamaV2Attention(self, layer_key, layer_idx, sliding_window = swa, rope_index = rope_index)
                if cfg.arch.lm.is_moe: mlp = ExLlamaV2MoEMLP(self, layer_key, layer_idx)
                else: mlp = ExLlamaV2MLP(self, layer_key, layer_idx)
                self.modules += [attn, mlp]

                # Per-layer scalar applied to the block output, after both residual adds.
                # PLAN.md §2.6. Gemma-4 only.
                if getattr(cfg.arch.lm, "layer_scalar", False):
                    ls = ExLlamaV2LayerScalar(self, layer_key + ".layer_scalar")
                    self.modules += [ls]

        _norm_key = cfg.arch.lm_prefix + getattr(archparams, "norm_key", "model.norm")
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
        callback: Callable[[int, int], None] | None = None
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
        self._tp_has_head = bool(load_head)

        # The fused single-device graph path assumes one device.
        self.config.no_graphs = True

        self.tp_context = TPContext(self, gpu_split, expect_cache_tokens, expect_cache_base)
        # BUG 58: enter multiprocess mode BEFORE any module loads. get_split() filters to
        # this rank from here on, and the loaders size their shards from it -- setting it
        # afterwards would make every module load the full split.
        self.tp_context.set_rank(rank)
        # BUG 80: keep the CUDA-graph path. Graphs are worth 2.55x on decode in the
        # single-process path and are the largest single factor in the MP path too. Capture is
        # per-process and only touches the devices this process owns (TPContext.all_devices() is
        # rank-filtered), so it is valid in a worker; if it fails, TPGraphManager.disable()
        # falls back to the eager path, which is what this used to do unconditionally.
        self.tp_graph = TPGraphManager(self)

        # Size the device context for THIS rank only.
        #
        # scratch_space_tp() returns one entry per device and also populates the
        # out_features_tp / in_features_tp lists that the TP forward paths dispatch on. It has to run
        # for every module so those lists are consistent, but only this rank's scratch is allocated.
        devs = self.tp_context.num_devices
        scratch = [0] * devs
        for module in self.modules[1:]:
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
                    module.tp_load_rank(rank)

                elif isinstance(module, ExLlamaV2Attention):
                    module.tp_load_rank(rank)

                elif isinstance(module, ExLlamaV2MLP):
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

                elif isinstance(module, ExLlamaV2Linear):
                    if idx == self.head_layer_idx:
                        if load_head:
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
               isinstance(module, ExLlamaV2ParallelDecoder):
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
                position_offsets = position_offsets,
                abort_event = abort_event,
                **kwargs
            )

            if abort_event and abort_event.is_set(): return

            if "last_state" in result:
                return result.get("logits"), result["last_state"]
            else:
                return result.get("logits")

        # Confirm that the input fits within the allocated cache space

        past_len = cache.current_seq_len
        assert past_len + q_len <= cache.max_seq_len, "Total sequence length exceeds cache size in model.forward"

        # Split sequence

        result = None
        last_state = None

        chunk_begin = 0
        while chunk_begin < q_len:

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
                _cpos = cache.current_seq_len - cache.window_beg
                _overflow = _cpos + chunk_size - _ring
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

            r = self.forward_chunk(
                input_ids = input_ids[:, chunk_begin : chunk_end],
                cache = cache,
                input_mask = input_mask,
                preprocess_only = _preprocess_only,
                last_id_only = _last_id_only,
                loras = loras,
                return_last_state = return_last_state and remaining_q_len <= chunk_size,
                position_offsets = position_offsets,
                abort_event = abort_event,
                **kwargs
            )

            if abort_event and abort_event.is_set(): return

            if not _preprocess_only:
                if cpu_logits:
                    r["logits"] = r["logits"].cpu()
                result = r["logits"] if result is None else torch.cat((result, r["logits"]), dim = 1)

            chunk_begin = chunk_end
            remaining_q_len -= chunk_size
            last_state = r.get("last_state")

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

        BUG 109: the graphed MTP verify window (q_len > 1) is served by q_len FI DECODE
        wrappers -- one per query row i, planned to that row's causal prefix
        kv_len_i = past_len + i + 1 (ring-relative for SWA). The decode wrapper is
        graph-safe (BUG 103: use_cuda_graph=True freezes its buffers) and reads ONLY the
        valid prefix -- unlike a full-160k-depth matmul (slow, BUG 108) or the FI
        paged-prefill wrapper (re-plan reallocates captured buffers -> segfault). Each
        unique (device, geometry, row) wrapper is planned ONCE per replay; layers sharing
        a geometry reuse it (this also removes the old per-layer redundant re-plans).
        """
        from exllamav2.attn import use_flashinfer
        from exllamav2.tensor_p import BROADCAST_KV

        cfg = self.config
        ctx = self.tp_context
        if not use_flashinfer(cfg) or cfg.attn_logit_softcapping:
            return
        planned = False
        seen = set()
        past_len = kv_len - q_len
        for m in self.modules:
            if not isinstance(m, ExLlamaV2Attention):
                continue
            # SWA layers keep K/V in a ring: row i's valid length is ring-relative
            # (past_len - window_beg + i + 1), clamped to the ring depth. Full layers use
            # the absolute causal prefix past_len + i + 1.
            window_left = m.sliding_window if m.sliding_window else -1
            if m.sliding_window:
                base = past_len - getattr(cache, "window_beg", 0)
                ring_depth = getattr(cache, "kv_state_size", None)
            else:
                base = past_len
                ring_depth = None
            ctx.set_geometry(m.layer_idx)
            split = ctx.get_split(BROADCAST_KV)
            k_cache, _ = cache.get_kv_state(m.layer_idx, 1, 0, 1)
            for idx, (dev, a, b) in enumerate(split):
                nq = (b - a) * m.num_key_value_groups
                nkv = b - a
                for i in range(q_len):
                    row_kv = base + i + 1
                    if ring_depth:
                        row_kv = min(row_kv, ring_depth)
                    if row_kv < 1:
                        continue
                    # q_len == 1 keeps the legacy 6-tuple key (attn decode branch); the
                    # verify rows append the row index so each row gets its own plan.
                    if q_len == 1:
                        key = (dev, nq, nkv, m.head_dim, m.scaling, window_left)
                    else:
                        key = (dev, nq, nkv, m.head_dim, m.scaling, window_left, i)
                    if key in seen:
                        continue
                    seen.add(key)
                    if ctx.fi_decode_plan(key, row_kv, k_cache[idx]):
                        planned = True
        if planned:
            ctx.fi_decode_order_before_replay()

    def _tp_graph_step(
        self,
        input_ids: torch.Tensor,
        cache,
        attn_params,
        past_len: int,
        batch_size: int,
        seq_len: int,
        last_id_only: bool,
        return_last_state: bool = False
    ):
        """
        Run one graphed step -- decode, MTP bonus commit (return_last_state), or the MTP
        verify window (seq_len > 1) -- capturing it on first use.

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
                                   state = return_last_state, last_id_only = last_id_only)

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
                ctx.set_ring_pos(past_len - getattr(cache, "window_beg", 0))
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
            if return_last_state:
                state_buf = torch.empty((batch_size, 1, self.config.hidden_size),
                                        dtype = _h0.dtype, device = ctx.device)
            out = {}

            def step_fn():
                # Start from the static embedded-input buffer; the embedding itself is host work
                # and therefore outside the graph (see the note above).
                x = input_buf
                for idx, module in enumerate(self.modules):
                    if idx < first_module:
                        continue
                    # BUG 81: a rank-local worker without lm_head must stop before it, exactly as
                    # forward_chunk does (BUG 77). Otherwise capture aborts with
                    # "Layer model.language_model.lm_head has no data" and graphs get disabled.
                    if idx == self.head_layer_idx and not getattr(self, "_tp_has_head", True):
                        break
                    # MTP bonus state: capture BEFORE the final norm (module head_layer_idx - 1)
                    # runs -- the same point forward_chunk's eager extract_state_indices clones at.
                    if return_last_state and idx == self.head_layer_idx - 1:
                        state_buf.copy_(x)
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
                                       (batch_size, seq_len, return_last_state, last_id_only),
                                       tp_context = ctx)
                # Let capture() restore the KV cache position after its warmup iterations. The
                # warmup runs the real step, which advances current_seq_len and writes KV rows.
                runner.cache = cache
                runner.input_buf = input_buf
                runner.state_buf = state_buf
                # Stage the real inputs so warmup and capture run at the caller's position with the
                # caller's tokens (the warmup KV writes are idempotent: same rows, same values).
                stage_inputs(input_buf)
                # hd>256 FlashInfer decode plans must exist before warmup/capture: run() is
                # captured, plan() is host work and cannot run inside the capture region.
                self._fi_tp_graph_plan(cache, past_len + seq_len, q_len = seq_len)
                ctx.graph_mode = True
                try:
                    runner.capture()
                    break
                except Exception as e:
                    _capture_err = e
                    ctx.graph_mode = False
                    runner = None
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
            self.tp_graph.add(batch_size, seq_len, runner,
                              state = return_last_state, last_id_only = last_id_only)

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
                    ctx.set_ring_pos(past_len - getattr(cache, "window_beg", 0))
                    for _pl in ctx.past_len_tp.values():
                        ctx.fill_tp(_pl, past_len)
                    # Refresh the hd>256 FlashInfer decode plans for this step's kv_len
                    # (host work, outside the graph) and order the replay after them.
                    self._fi_tp_graph_plan(cache, past_len + seq_len, q_len = seq_len)
                    # Replay while dev0's stream is still current: CUDAGraph.replay() launches on
                    # the torch current stream, and the graph's nodes were captured on this one.
                    runner.replay()
                finally:
                    torch.cuda.set_stream(prev_stream)
        finally:
            ctx.graph_mode = False
        return runner

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
        if return_last_state:
            extract_state_indices.add(self.head_layer_idx - 1)
        if extract_state_indices:
            r["states"] = {}

        # assert cache is None or isinstance(cache, list) or batch_size <= cache.batch_size

        x = input_ids

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
        graph_state = return_last_state and extract_state_indices == {self.head_layer_idx - 1}
        if (
            self.tp_context is not None
            and cache is not None
            and (seq_len == 1 or graph_verify)
            and not preprocess_only
            and (not extract_state_indices or graph_state)
            and not loras
            and not kwargs.get("indexed_embeddings")
            and self.tp_graph is not None
            and not (graph_verify and self._tp_graph_verify_failed)
            and self.tp_graph.can_capture(batch_size, seq_len, loras, allow_multi = graph_verify)
        ):
            graph_runner = self._tp_graph_step(
                input_ids, cache, attn_params, past_len, batch_size, seq_len, last_id_only,
                return_last_state = graph_state
            )
            if graph_runner is not None:
                # The graph wrote its logits into the runner's static buffer; hand it back.
                x = graph_runner.logits
                if graph_state:
                    r["states"][self.head_layer_idx - 1] = graph_runner.state_buf
                    r["last_state"] = graph_runner.state_buf

        if graph_runner is None:

            for idx, module in enumerate(self.modules):

                if idx == self.head_layer_idx and last_id_only:
                    x = x.narrow(-2, -1, 1)

                if idx in extract_state_indices:
                    r["states"][idx] = x.clone()
                    if idx == self.head_layer_idx - 1:
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

                if os.environ.get("EXL2_TRACE_X") and not isinstance(x, torch.Tensor):
                    print(f"[TRACE] module {idx} {type(module).__name__} key={getattr(module, 'key', '?')} "
                          f"returned {type(x).__name__} "
                          f"len={len(x) if isinstance(x, (list, tuple)) else '-'} "
                          f"shapes={[tuple(t.shape) for t in x] if isinstance(x, (list, tuple)) else '-'}",
                          flush=True)

                if preprocess_only and idx == self.last_kv_layer_idx:
                    x = None
                    break

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
