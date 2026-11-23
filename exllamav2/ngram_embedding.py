from __future__ import annotations

"""
EXL2.5 engram support: hashed n-gram embedding tables (DeepSeek-V4.1 / Qwen3-Next style
``ple_embedding``). Each token position maps to (ngram_size-1)*heads_per_ngram table rows
via XOR-mixed shifted-token hashes mod per-head prime vocab sizes; the gathered rows are
decoded, bias-added and concatenated into one feature vector.

The table is enormous (tens of billions of parameters), so it is NEVER resident in VRAM:
rows live in host RAM (``stream_from_disk=False``) or are pread from the safetensors file
on demand (default). Only the unique rows a forward pass touches are staged to the GPU and
decoded by ext_c.ngram_dequant_rows.

EXL2.5 row format (scalar uniform-bit, NO trellis -- sm86-friendly):
    <key>.q_rows            int32  (num_rows, head_dim*bits/32)  symmetric packed rows
    <key>.q_row_scales      fp16   (num_rows,)                   per-row scale
    <key>.head_offsets      int64  (num_heads,)                  global row offsets
    <key>.head_vocab_sizes  int64  (num_heads,)                  per-head prime moduli
    <key>.layer_multipliers int64  (ngram_size,)                 hash mix multipliers
    <key>.head_bias         fp16   (num_heads, head_dim)         per-head dequant bias

Produced by exllamav2/conversion/convert_ngram.py. Hashing/segmentation logic mirrors the
reference implementation (eos-bounded n-grams), kept in torch on the CPU where the token
ids live.
"""

import os
import json
import torch
from exllamav2.module import ExLlamaV2Module
from exllamav2.ext import exllamav2_ext as ext_c
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2


class ExLlamaV2NGramEmbedding(ExLlamaV2Module):

    name: str = "NGram Embedding"

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        ngram_size: int,
        heads_per_ngram: int,
        ple_embed_dim: int,
        eos_token_id: int,
        bits: int = 8,
        head_dim: int = 160,
        stream_from_disk: bool = True,
        archparams = None
    ):
        super().__init__(model, key, archparams)
        assert bits in (4, 8)
        self.ngram_size = ngram_size
        self.context_len = ngram_size - 1
        self.heads_per_ngram = heads_per_ngram
        self.num_heads = (ngram_size - 1) * heads_per_ngram
        self.ple_embed_dim = ple_embed_dim
        self.head_dim = head_dim
        assert ple_embed_dim == self.num_heads * head_dim
        self.eos_token_id = eos_token_id
        self.bits = bits
        # Storage mode: "disk" (pread per lookup, default), "ram" (full table in host
        # RAM), "hybrid" (disk + bounded LRU row cache in host RAM). Env overrides the
        # constructor default so tabby/API users can choose without code changes.
        mode = os.environ.get("EXL2_NGRAM_MODE", "disk" if stream_from_disk else "ram").lower()
        if mode not in ("disk", "ram", "hybrid"):
            mode = "disk"
        self.ngram_mode = mode
        self.stream_from_disk = mode != "ram"
        self._hybrid_cap = 0
        self._hybrid_cache = None
        self._hybrid_bytes = 0
        if mode == "hybrid":
            try:
                mb = int(os.environ.get("EXL2_NGRAM_HYBRID_MB", "1024"))
            except ValueError:
                mb = 1024
            self._hybrid_cap = max(1, mb) * 1024 * 1024
            from collections import OrderedDict
            self._hybrid_cache = OrderedDict()

        self.head_offsets = None        # int64 CPU
        self.head_vocab_sizes = None    # int64 CPU
        self.layer_multipliers = None   # int64 CPU
        self.head_bias = None           # fp16 device
        self.row_scales = None          # fp16 CPU (RAM mode) or None (disk)
        self.table_ram = None           # int32 CPU (RAM mode)
        self._disk = None               # (fd, data_base, row_bytes, num_rows)
        self._stage = None              # grow-only pinned staging
        self._stage_rows = 0

    def numel(self) -> int:
        return self._num_rows() * self.head_dim

    def _num_rows(self) -> int:
        if self.table_ram is not None: return self.table_ram.shape[0]
        if self._disk is not None: return self._disk[3]
        return int(self.head_offsets[-1].item() + self.head_vocab_sizes[-1].item())

    def weight_footprint(self) -> int:
        # Host/disk resident; VRAM footprint is the per-forward staging only.
        return self.head_dim * 2 * 1024 + self.num_heads * self.head_dim * 2

    def scratch_space_fixed(self) -> int:
        return 0

    def scratch_space(self) -> int:
        return 0

    @torch.inference_mode
    def load(self):
        w = self.load_multi(self.key, [
            "q_rows", "q_row_scales", "head_offsets", "head_vocab_sizes",
            "layer_multipliers", "head_bias"
        ], cpu = True)
        assert "head_offsets" in w, f"{self.key}: missing engram hash params (run convert_ngram.py)"
        self.head_offsets = w["head_offsets"].long().contiguous()
        self.head_vocab_sizes = w["head_vocab_sizes"].long().contiguous()
        self.layer_multipliers = w["layer_multipliers"].long().contiguous()
        assert self.head_offsets.shape[0] == self.num_heads
        self.head_bias = safe_move_tensor(w["head_bias"].half().contiguous(), self.device())

        if self.stream_from_disk:
            path = self.model.config.tensor_file_map[self.key + ".q_rows"]
            self._open_disk(path)
            self.row_scales = w["q_row_scales"].half().contiguous()  # small: keep in RAM
        else:
            self.table_ram = w["q_rows"].contiguous()
            self.row_scales = w["q_row_scales"].half().contiguous()
            if "q_rows" in w: del w["q_rows"]

    def _open_disk(self, path: str):
        with open(path, "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(hlen))
        info = header[self.key + ".q_rows"]
        assert info["dtype"] == "I32"
        num_rows, words = info["shape"]
        row_bytes = words * 4
        data_base = 8 + hlen + info["data_offsets"][0]
        fd = os.open(path, os.O_RDONLY)
        self._disk = (fd, data_base, row_bytes, num_rows)

    def unload(self):
        if self._disk is not None:
            os.close(self._disk[0])
            self._disk = None
        self.table_ram = None
        self.row_scales = None
        self.head_bias = None
        self._stage = None
        self._stage_rows = 0
        if self._hybrid_cache is not None:
            self._hybrid_cache.clear()
        self._hybrid_bytes = 0

    def set_device_idx(self, idx: int | None):
        self.device_idx = idx

    # ---- hashing (CPU torch; mirrors the reference implementation) ----

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0: return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim = 1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1),
                                  previous_eos_inclusive[:, :-1]], dim = 1)
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim = 1, index = gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def compute_ngram_ids(self, token_history: torch.Tensor, out_len: int) -> torch.Tensor:
        th = token_history.long()
        shifted = [self._shift_right_ignore_eos(th, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position])
            sizes = self.head_vocab_sizes[lo:hi]
            offsets = self.head_offsets[lo:hi]
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1))
                          + offsets.view(1, 1, -1))
        return torch.cat(blocks, dim = -1)[:, -out_len:]

    # ---- row gather + decode ----

    def _pread_rows(self, uids: torch.Tensor) -> torch.Tensor:
        """Fetch rows from the safetensors file, coalescing contiguous runs."""
        fd, data_base, row_bytes, num_rows = self._disk
        assert int(uids.max().item()) < num_rows
        out = torch.empty(uids.shape[0], row_bytes, dtype = torch.uint8)
        order = torch.argsort(uids)
        su = uids[order]
        i = 0
        n = su.shape[0]
        while i < n:
            j = i + 1
            while j < n and su[j].item() == su[j - 1].item() + 1: j += 1
            a = su[i].item()
            raw = os.pread(fd, (j - i) * row_bytes, data_base + a * row_bytes)
            buf = torch.frombuffer(bytearray(raw), dtype = torch.uint8).view(j - i, row_bytes)
            out.index_copy_(0, order[i:j], buf)
            i = j
        return out

    def _gather_packed(self, uids: torch.Tensor) -> torch.Tensor:
        if self.table_ram is not None:
            return self.table_ram.index_select(0, uids)
        fd, data_base, row_bytes, num_rows = self._disk
        if self._hybrid_cache is None:
            return self._pread_rows(uids).view(torch.int32)
        # Hybrid: bounded LRU row cache in host RAM over the disk backend. Hot rows
        # (repeated n-grams across a conversation) serve from RAM; cold rows pread.
        cache = self._hybrid_cache
        out = torch.empty(uids.shape[0], row_bytes, dtype = torch.uint8)
        miss_idx = []
        for i in range(uids.shape[0]):
            u = int(uids[i].item())
            ent = cache.get(u)
            if ent is not None:
                cache.move_to_end(u)
                out[i] = torch.frombuffer(ent, dtype = torch.uint8)
            else:
                miss_idx.append(i)
        if miss_idx:
            mu = uids[miss_idx]
            fetched = self._pread_rows(mu)
            for k, i in enumerate(miss_idx):
                u = int(mu[k].item())
                ent = bytearray(fetched[k].numpy().tobytes())
                out[i] = fetched[k]
                if u in cache:
                    cache.move_to_end(u)
                else:
                    cache[u] = ent
                    self._hybrid_bytes += len(ent)
            while self._hybrid_bytes > self._hybrid_cap and cache:
                _, old = cache.popitem(last = False)
                self._hybrid_bytes -= len(old)
        return out.view(torch.int32)

    def _stage_h2d(self, packed: torch.Tensor, scales: torch.Tensor, heads: torch.Tensor):
        u = packed.shape[0]
        words = packed.shape[1]
        need = u * (words * 4 + 2 + 4) + 4
        if self._stage is None or self._stage.numel() < need:
            self._stage = torch.empty(need, dtype = torch.uint8, pin_memory = True)
            self._stage_rows = u
        off = 0
        p = self._stage[off:off + u * words * 4].view(torch.int32).view(u, words); off += u * words * 4
        s = self._stage[off:off + u * 2].view(torch.float16); off += u * 2
        off = (off + 3) & ~3  # int32 view requires 4-byte alignment
        h = self._stage[off:off + u * 4].view(torch.int32)
        p.copy_(packed)
        s.copy_(scales)
        h.copy_(heads)
        dev = self.device()
        return (p.to(dev, non_blocking = True),
                s.to(dev, non_blocking = True),
                h.to(dev, non_blocking = True))

    def forward(self, token_history: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        token_history: (bsz, context_len + seq_len) token ids INCLUDING the ngram_size-1
        tokens preceding the sequence (eos-padded at the start of a new sequence).
        Returns (bsz, seq_len, ple_embed_dim) fp16 on the module's device.
        """
        ids = token_history.to("cpu", torch.long).contiguous()
        bsz = ids.shape[0]
        out_len = ids.shape[1] - self.context_len
        assert out_len > 0
        ngram_ids = self.compute_ngram_ids(ids, out_len)
        flat = ngram_ids.reshape(-1)
        uids, inverse = torch.unique(flat, return_inverse = True)
        heads = (torch.searchsorted(self.head_offsets, uids, right = True) - 1) \
            .clamp(0, self.num_heads - 1).to(torch.int32)
        packed = self._gather_packed(uids)
        scales = self.row_scales.index_select(0, uids)
        p, s, h = self._stage_h2d(packed, scales, heads)
        rows = torch.empty(uids.shape[0], self.head_dim, dtype = torch.half, device = self.device())
        ext_c.ngram_dequant_rows(p, s, h, self.head_bias, rows, self.bits)
        out = rows.index_select(0, inverse.to(self.device()))
        return out.view(bsz, out_len, self.ple_embed_dim)
