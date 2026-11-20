from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

EMBEDDING_INDEX: int = 1000000000

class ExLlamaV2Embedding(ExLlamaV2Module):

    name: str = "Embedding"

    embedding: nn.Embedding | None
    native_vocab_size: int | None

    is_tp: bool

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        archparams = None
    ):
        super().__init__(model, key, archparams)

        self.is_tp = False

        self.native_vocab_size = None
        self.embedding = None


    def tp_split(self):

        self.is_tp = True


    @torch.inference_mode
    def load(self, device_context: bool = True):

        vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        pad_token_id = self.model.config.pad_token_id

        w = self.load_weight()
        assert isinstance(w, nn.Parameter)
        # TODO: Figure out why pinning this tensor allocates GPU memory??
        # w.pin_memory()
        self.native_vocab_size = w.shape[0]

        self.embedding = nn.Embedding(vocab_size, hidden_size, pad_token_id, device = "meta")
        if self.model.config.scale_emb != 1:
            w *= self.model.config.scale_emb
        self.embedding.weight = w


    def tp_load_rank_sharded(self, rank: int):
        """
        Vocab-sharded embedding: load ONLY rows [a, b) of the VC split (~0.67 GB instead
        of the 2.8 GB replicated table). forward() masks the local lookup and SUM-reduces
        the partials -- each token id lives on exactly one rank, so the sum reconstructs
        the full embedding row. Mirror of the vocab-parallel lm_head; together they
        remove ~4 GB of replicated vocab tables per rank (master OOM fix, debug10.txt).
        """
        from exllamav2.tensor_p import BROADCAST_VC
        cfg = self.model.config
        ctx = self.model.tp_context
        split = ctx.get_split(BROADCAST_VC)
        assert len(split) == 1, f"rank {rank} owns no vocab slice of {self.key}"
        _, a, b = split[0]

        self.device_idx = -1
        w = self.load_weight(cpu = True)
        assert isinstance(w, nn.Parameter)
        self.native_vocab_size = w.shape[0]
        if cfg.scale_emb != 1:
            w = nn.Parameter(w.data * cfg.scale_emb)

        self.device_idx = rank
        # padding_idx=None: nn.Embedding would zero LOCAL row 0 on every rank, but only
        # the rank owning the global pad id may zero it. Replicate the semantics explicitly.
        self.embedding = nn.Embedding(b - a, cfg.hidden_size, None, device = "meta")
        self.embedding.weight = nn.Parameter(
            safe_move_tensor(w.data[a:b].contiguous(), rank), requires_grad = False)
        if cfg.pad_token_id is not None and a <= cfg.pad_token_id < b:
            self.embedding.weight.data[cfg.pad_token_id - a].zero_()
        del w
        self._tp_vocab_shard = (a, b)
        self.is_tp = False

    def tp_load_rank(self, rank: int):
        """
        Load the embedding table in full, pinned to one device, for a rank-local worker.

        The table is replicated, not sharded: ``tp_split()`` only sets ``is_tp = True`` and
        leaves the whole weight in place, because every rank needs to look up arbitrary token
        ids. So the rank-local path is a full load.

        ``is_tp`` must stay False here. In the single-threaded TP path it means "copy the
        result into the pinned staging buffer so the next module can broadcast it", which a
        worker process does not need -- its result is already on its own device.

        :param rank:
            Device index this worker owns.
        """
        # device_idx must be set BEFORE load(): load_weight() -> load_multi() reads
        # self.device(), which reads self.device_idx.
        self.device_idx = rank
        self.load()
        self.is_tp = False
        if self.embedding is not None:
            self.embedding.weight.data = safe_move_tensor(self.embedding.weight.data, rank)

    def unload(self):

        del self.embedding
        self.embedding = None


    def get_weight(self) -> torch.Tensor:

        if self.model.config.scale_emb != 1:
            return self.embedding.weight.data / self.model.config.scale_emb
        else:
            return self.embedding.weight.data


    def weight_footprint(self) -> int:

        vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        # kv_size = self.model.config.num_key_value_heads * self.model.config.head_dim

        return vocab_size * hidden_size * 2


    def scratch_space_fixed(self) -> int:

        return 0


    def scratch_space_tp(self) -> list[int]:

        return [0] * self.model.tp_context.num_devices


    def scratch_space(self) -> int:

        return 0


    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params: ExLlamaV2Attention.Params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        cfg = self.model.config

        # If input IDs contain negative values, assume they are padding tokens from a model with no pad_token_id
        # defined or noise values for quantizing

        input_ids = hidden_states
        hidden_states = hidden_states.clamp(min = 0)

        if getattr(self, "_tp_vocab_shard", None) is not None:
            if kwargs.get("indexed_embeddings"):
                raise NotImplementedError(
                    "vocab-sharded embedding does not support indexed_embeddings")
            if loras is not None and loras[0].embed_tokens is not None:
                raise NotImplementedError(
                    "vocab-sharded embedding does not support embed_tokens LoRAs")

        # Apply indexed embeddings

        indexed_embeddings = kwargs.get("indexed_embeddings")
        if indexed_embeddings:

            # Split prompt

            input_ids = hidden_states
            standard_mask = input_ids < EMBEDDING_INDEX
            indexed_masks = [
                (input_ids >= e.first_index) & (input_ids < (e.first_index + e.length))
                for e in indexed_embeddings
            ]

        if indexed_embeddings and any(im.any() for im in indexed_masks):

            # Create combined tensor on the target device

            batch_size, seq_len = input_ids.shape
            hidden_size = cfg.hidden_size
            combined_embeddings = torch.empty(
                batch_size, seq_len, hidden_size,
                device = indexed_embeddings[0].embeddings.device,
                dtype = indexed_embeddings[0].embeddings.dtype
            )

            # Extract standard embeddings, copy to target device and insert in-place

            attn_params.rope_mask = standard_mask
            if standard_mask.any():
                for i in range(batch_size):
                    standard_mask_ = standard_mask[i]
                    input_ids_ = input_ids[i]
                    standard_ids_ = input_ids_[standard_mask_]
                    if loras is not None and loras[0].embed_tokens is not None:
                        standard_embeddings_ = loras[0].embed_tokens(standard_ids_)
                    else:
                        standard_embeddings_ = self.embedding(standard_ids_)
                    standard_embeddings_ = safe_move_tensor(standard_embeddings_, indexed_embeddings[0].embeddings.device)
                    combined_embeddings[i][standard_mask_] = standard_embeddings_

            # Normalization

            if self.archparams.residual_stream_fp32:
                combined_embeddings = combined_embeddings.float()
            if self.archparams.normalize_embeddings:
                combined_embeddings *= cfg.hidden_size ** 0.5

            # Extract indexed embeddings and insert in-place

            for im, ie in zip(indexed_masks, indexed_embeddings):
                if im.any():
                    for i in range(batch_size):
                        indexed_ids_ = input_ids[i][im[i]] - ie.first_index
                        combined_embeddings[i][im[i]] = ie.embeddings[indexed_ids_].to(combined_embeddings.dtype)

            hidden_states = combined_embeddings

        # Call embedding module if no indexed embeddings

        else:
            if loras is not None and loras[0].embed_tokens is not None:
                hidden_states = loras[0].embed_tokens(hidden_states)
            elif getattr(self, "_tp_vocab_shard", None) is not None:
                # Vocab-sharded lookup: out-of-range ids clamp to local row 0 and are
                # zeroed, then the partials are SUM-reduced -- each id belongs to exactly
                # one rank, so the sum is the exact full embedding row (no precision
                # loss: one nonzero addend). ~1.5*rows*hidden*2 bytes per rank (33 MB per
                # 2048-token chunk, 16 KB per decode token). Symmetric on every rank and
                # graph-capturable, like the other rank-mode collectives.
                a, b = self._tp_vocab_shard
                ids = hidden_states
                local = (ids - a).clamp(0, b - a - 1)
                part = self.embedding(local)
                part.mul_(((ids >= a) & (ids < b)).unsqueeze(-1).to(part.dtype))
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    from exllamav2.tensor_p import _CollTimer, _PREFILL_TRACE
                    _ctx = self.model.tp_context
                    with _CollTimer(_PREFILL_TRACE and ids.numel() > 32 and not _ctx.graph_mode):
                        _p2p = getattr(_ctx, "p2p", None)
                        if _p2p is None or not _p2p.allreduce(part):
                            dist.all_reduce(part, op = dist.ReduceOp.SUM)
                hidden_states = part
            else:
                hidden_states = self.embedding(hidden_states)

            if self.archparams.residual_stream_fp32:
                hidden_states = hidden_states.float()
            if self.archparams.normalize_embeddings:
                hidden_states *= cfg.hidden_size ** 0.5

        # Rows with negative tokens during quantization are noise tokens

        if kwargs.get("negative_ids_noise"):

            n = 0
            mean = torch.tensor([0.0], dtype = torch.float, device = hidden_states.device)
            M2 = torch.tensor([0.0], dtype = torch.float, device = hidden_states.device)

            for i in range(input_ids.shape[0]):
                if input_ids[i][0] < 0:
                    continue

                er = hidden_states[i].float()
                n += er.numel()
                delta = er - mean
                mean += delta.sum() / n
                delta2 = er - mean
                M2 += (delta * delta2).sum()
                del er
                del delta
                del delta2

            if n > 1:
                std = torch.sqrt(M2 / (n - 1))

            for i in range(input_ids.shape[0]):
                if input_ids[i][0] >= 0:
                    continue

                er = hidden_states[i]
                noise = torch.randn(er.size(), dtype = torch.float, device = hidden_states.device) * std + mean
                er.copy_(noise.half())
                del er
                del noise

        # Move to pinned temp buffer for TP

        if self.is_tp:
            ctx = self.model.tp_context
            hidden_states = ctx.copy_pinned(0, hidden_states)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states
