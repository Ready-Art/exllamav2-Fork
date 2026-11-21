"""
DFlash drafter (z-lab/gemma-4-31B-it-DFlash) for the Gemma-4-31B target.

DFlash ("Block Diffusion for Flash Speculative Decoding", arXiv 2602.06036) is a
lightweight block-diffusion draft model. Unlike the Gemma-4 MTP assistant
(mtp_drafter.py), which cross-attends to the target's K/V one token at a time,
DFlash drafts a WHOLE BLOCK in one parallel forward:

  * The target's residual streams after layers ``target_layer_ids``
    ([1, 12, 23, 35, 46, 57] for Gemma-4-31B) are concatenated and projected by
    ``fc`` + ``hidden_norm`` into a compact context feature per token.
  * Each draft layer turns that feature into context K/V with its OWN shared
    k_proj/v_proj (the same weights the block tokens use), k_norm + RoPE them,
    and stores them in a persistent per-layer draft KV cache. Context K/V is
    therefore derived from the target's hidden states -- the draft model never
    re-processes the prompt itself.
  * A draft block is [anchor, mask x (block_size-1)]: the anchor is the last
    decided (bonus) token, the mask positions (mask_token_id = 4) are denoised
    in parallel. Queries attend to the context K/V plus the block's own K/V:
    the sliding layers (0-3) use causal + sliding-window attention, the full
    layer (4) is NON-CAUSAL (bidirectional within the block, full context).
    Block isolation across rounds comes from the cache length bookkeeping:
    the block's noise K/V sits beyond the committed context length and is
    overwritten by the next round's context feed.
  * Logits come from the TARGET's lm_head (borrowed, full fp16 copy on the
    drafter device) with the target's final_logit_softcapping (30). Embeddings
    are the target's table scaled by sqrt(hidden_size) (Gemma-4 convention,
    matching vLLM's Gemma-4 DFlash integration).

Round protocol (maps onto MTPMPDriver's bonus-eliminated mode B, W = 15):

    prefill:  target forwards the prompt, returning per-row aux states
              (the 6-layer concat) -> feed_context(aux, 0) builds the draft
              KV context [0, n_prompt).
    round:    anchor = pending bonus at position pos (KV not yet committed).
              draft_window(anchor, pos) -> W tokens at pos+1 .. pos+W.
              verify [anchor, drafts] (W+1 rows) on the target -> accept the
              longest prefix, bonus' = row m_acc. The verify's aux rows
              0..m_acc (bonus + accepted drafts) are fed back as context at
              [pos, pos+m_acc+1), overwriting the block's noise K/V.

All draft attention uses FlashInfer paged prefill (fa2 backend, sm86-verified
hd128): one wrapper for the SWA geometry (causal=True, window_left=W-1) and one
for the full geometry (causal=False), re-planned per round at the round's kv_len
(the fork's standard host-side plan contract). Norms/RoPE/softcap use the fork's
fused ext kernels (BUG 38 pattern -- no pure-torch norm chains in the hot loop).

Reference semantics: z-lab/dflash model.py (transformers) + vLLM PR #41703
(qwen3_dflash.py / dflash.py, the Gemma-4 integration).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from exllamav2.ext import exllamav2_ext as ext_c, none_tensor


# Known geometry of z-lab/gemma-4-31B-it-DFlash (read from its config.json at
# load; these defaults document the shape).
_DFLASH_HIDDEN = 5376
_DFLASH_INTERMEDIATE = 10752
_DFLASH_LAYERS = 5
_DFLASH_LAYER_TYPES = ["sliding_attention"] * 4 + ["full_attention"]
_DFLASH_HEAD_DIM = 128
_DFLASH_NUM_Q_HEADS = 64
_DFLASH_NUM_KV_HEADS = 8
_DFLASH_SLIDING_WINDOW = 2048
_DFLASH_BLOCK_SIZE = 16
_DFLASH_MASK_TOKEN_ID = 4
_DFLASH_TARGET_LAYER_IDS = [1, 12, 23, 35, 46, 57]
_DFLASH_VOCAB = 262144
_DFLASH_ROPE_THETA = 1000000.0
_DFLASH_SOFTCAP = 30.0

_PAGE = 128          # FlashInfer page size (fork-wide convention)
_CTX_CHUNK = 2048    # context-feed GEMM chunk (bounds the fc transient VRAM)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused weighted RMSNorm (fp32 reduce), same math as the target's norms."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    y = torch.empty_like(x2)
    ext_c.rms_norm(x2, weight, y, eps)
    return y.view(shape)


def _head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm, in-place. ``x`` (..., heads, hd), ``weight`` (heads, hd)
    (the loader repeats the checkpoint's shared (hd,) weight, headnorm.py style)."""
    if x.dtype != torch.half:
        x = x.to(torch.half)
    if not x.is_contiguous():
        x = x.contiguous()
    ext_c.head_norm(x, weight, none_tensor, x, eps, True)
    return x


@dataclass
class _DFlashLayer:
    layer_idx: int
    is_full: bool
    input_ln: torch.Tensor
    q_proj: torch.Tensor        # (nq*hd, hidden)
    k_proj: torch.Tensor        # (nkv*hd, hidden)
    v_proj: torch.Tensor        # (nkv*hd, hidden)
    o_proj: torch.Tensor        # (hidden, nq*hd)
    q_norm: torch.Tensor        # (nq, hd) repeated
    k_norm: torch.Tensor        # (nkv, hd) repeated
    post_attn_ln: torch.Tensor
    gate: torch.Tensor          # (inter, hidden)
    up: torch.Tensor
    down: torch.Tensor          # (hidden, inter)


class DFlashDrafter:
    """
    DFlash block-diffusion drafter. Construct with the MP target (TPProcessGroup),
    the draft checkpoint directory and a device. ``enable_mtp`` on the process
    group then supplies the full lm_head copy, the aux-tap wiring and the draft
    KV allocation (alloc_kv). Per round: draft_window(); after each verify:
    feed_verify().
    """

    is_dflash = True

    def __init__(
        self,
        target,
        draft_dir: str,
        device: int = 0,
        num_draft_tokens: int | None = None,
    ):
        self.target = target
        self.tcfg = target.config
        self.device = device
        self.dev = torch.device("cuda", device)

        self._read_config(draft_dir)

        # Default draft window: block_size - 1 (the anchor occupies one slot).
        # Trained BS16 generalizes DOWN (paper ablation); smaller windows are
        # allowed, larger ones are not.
        if num_draft_tokens is None:
            num_draft_tokens = self.block_size - 1
        assert 1 <= num_draft_tokens <= self.block_size - 1, \
            f"draft_num_tokens must be in [1, {self.block_size - 1}] for DFlash"
        self.num_draft_tokens = int(num_draft_tokens)

        # Target-side sources. The embedding table is kept as a FULL CPU copy
        # (the target's may be vocab-sharded); sqrt(hidden) is folded in once.
        self.t_embed = target.modules[0]
        self._t_embed_full = None
        if getattr(self.t_embed, "_tp_vocab_shard", None) is not None:
            _prev = self.t_embed.device_idx
            self.t_embed.device_idx = -1
            try:
                _w = self.t_embed.load_weight(cpu = True)
            finally:
                self.t_embed.device_idx = _prev
            _w = _w.data.clone()
        else:
            _w = self.t_embed.embedding.weight.data.clone()
        if getattr(self.t_embed.archparams, "normalize_embeddings", False):
            _w *= self.hidden_size ** 0.5
        self._t_embed_full = _w.half()

        self._load_weights(draft_dir)

        # Full lm_head copy + padding, supplied by enable_mtp (vocab shards are
        # gathered from the workers there). None until then.
        self.head = None
        self.head_padding = 0

        # Draft KV cache (per-layer K/V, position-indexed, page-aligned depth).
        # Allocated by alloc_kv at enable time with a clean allocator.
        self.k_cache = None      # list of (depth, nkv, hd) fp16
        self.v_cache = None
        self.depth = 0
        self.ctx_len = 0         # committed context length (== target cache pos)
        self._last_aux = None    # (1, 6*hidden) CPU aux row of token ctx_len-1

        # RoPE tables (own theta/hd, NEOX), built in alloc_kv.
        self._sin = None
        self._cos = None
        self._rope_maxpos = 0

        # FlashInfer prefill wrappers (eager; one per layer geometry).
        self._wrap_swa = None
        self._wrap_full = None

    # ------------------------------------------------------------------ config

    def _read_config(self, draft_dir: str):
        path = os.path.join(draft_dir, "config.json")
        assert os.path.exists(path), f"DFlash draft config not found: {path}"
        with open(path, encoding = "utf-8") as f:
            cfg = json.load(f)
        dc = cfg.get("dflash_config", {})

        self.hidden_size = cfg.get("hidden_size", _DFLASH_HIDDEN)
        self.intermediate_size = cfg.get("intermediate_size", _DFLASH_INTERMEDIATE)
        self.num_hidden_layers = cfg.get("num_hidden_layers", _DFLASH_LAYERS)
        self.layer_types = cfg.get("layer_types", _DFLASH_LAYER_TYPES)
        self.head_dim = cfg.get("head_dim", _DFLASH_HEAD_DIM)
        self.num_q_heads = cfg.get("num_attention_heads", _DFLASH_NUM_Q_HEADS)
        self.num_kv_heads = cfg.get("num_key_value_heads", _DFLASH_NUM_KV_HEADS)
        self.sliding_window = cfg.get("sliding_window", _DFLASH_SLIDING_WINDOW)
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)
        self.vocab_size = cfg.get("vocab_size", _DFLASH_VOCAB)
        self.block_size = int(cfg.get("block_size", _DFLASH_BLOCK_SIZE))
        self.mask_token_id = int(dc.get("mask_token_id", _DFLASH_MASK_TOKEN_ID))
        self.target_layer_ids = list(dc.get("target_layer_ids", _DFLASH_TARGET_LAYER_IDS))
        self.num_target_layers = cfg.get("num_target_layers", self.tcfg.num_hidden_layers)
        self.rope_theta = float(cfg.get("rope_theta", _DFLASH_ROPE_THETA))
        self.softcap = cfg.get("final_logit_softcapping", _DFLASH_SOFTCAP)
        self.sm_scale = self.head_dim ** -0.5

        assert self.hidden_size == self.tcfg.hidden_size, \
            "DFlash draft hidden size must match the target's"
        assert self.num_target_layers == self.tcfg.num_hidden_layers, \
            "DFlash draft num_target_layers must match the target's layer count"
        assert len(self.target_layer_ids) * self.hidden_size > 0
        assert all(0 <= i < self.num_target_layers for i in self.target_layer_ids)
        assert self.vocab_size == self.tcfg.vocab_size, \
            "DFlash draft vocab must match the target's (tied head)"

    # ------------------------------------------------------------------ weights

    def _load_weights(self, draft_dir: str):
        from safetensors import safe_open
        st_path = os.path.join(draft_dir, "model.safetensors")
        if not os.path.exists(st_path):
            cands = sorted(f for f in os.listdir(draft_dir) if f.endswith(".safetensors"))
            assert cands, f"no safetensors in {draft_dir}"
            st_path = os.path.join(draft_dir, cands[0])

        W = {}
        with safe_open(st_path, framework = "pt") as f:
            for k in f.keys():
                W[k] = f.get_tensor(k)

        dev = self.dev

        def g(key, dtype = torch.half):
            assert key in W, f"missing DFlash weight {key}"
            return W[key].to(dev, dtype = dtype).contiguous()

        nft = len(self.target_layer_ids)
        self.fc = g("fc.weight")                      # (hidden, nft*hidden)
        assert self.fc.shape == (self.hidden_size, nft * self.hidden_size), \
            f"fc shape {tuple(self.fc.shape)} != {(self.hidden_size, nft * self.hidden_size)}"
        self.hidden_norm = g("hidden_norm.weight")    # (hidden,)
        self.final_norm = g("norm.weight")            # (hidden,)

        self.layers = []
        kv_rows = []
        for i in range(self.num_hidden_layers):
            prefix = f"layers.{i}"
            full = self.layer_types[i] == "full_attention"
            k_proj = g(f"{prefix}.self_attn.k_proj.weight")
            v_proj = g(f"{prefix}.self_attn.v_proj.weight")
            # Fused context-KV projection: one GEMM for all layers (vLLM's
            # precompute_and_store_context_kv pattern). Row order:
            # [layer0 K | layer0 V | layer1 K | ...].
            kv_rows.append(k_proj)
            kv_rows.append(v_proj)
            self.layers.append(_DFlashLayer(
                layer_idx = i,
                is_full = full,
                input_ln = g(f"{prefix}.input_layernorm.weight"),
                q_proj = g(f"{prefix}.self_attn.q_proj.weight"),
                k_proj = k_proj,
                v_proj = v_proj,
                o_proj = g(f"{prefix}.self_attn.o_proj.weight"),
                q_norm = g(f"{prefix}.self_attn.q_norm.weight").repeat(self.num_q_heads, 1),
                k_norm = g(f"{prefix}.self_attn.k_norm.weight").repeat(self.num_kv_heads, 1),
                post_attn_ln = g(f"{prefix}.post_attention_layernorm.weight"),
                gate = g(f"{prefix}.mlp.gate_proj.weight"),
                up = g(f"{prefix}.mlp.up_proj.weight"),
                down = g(f"{prefix}.mlp.down_proj.weight"),
            ))
        self.fused_kv = torch.cat(kv_rows, dim = 0)   # (L*2*nkv*hd, hidden)

    def set_head(self, weight: torch.Tensor, padding: int):
        """Full target lm_head (vocab, hidden) fp16 + tail padding row count.
        Supplied by TPProcessGroup.enable_mtp (gathered vocab shards)."""
        self.head = weight.to(self.dev, dtype = torch.half).contiguous()
        self.head_padding = int(padding)

    # ------------------------------------------------------------------ KV cache / rope

    def alloc_kv(self, max_seq_len: int):
        """Allocate the position-indexed draft KV cache (all layers, full depth).

        Like vLLM's DFlash KV spec, the sliding layers keep FULL-depth context
        (the window is applied at attention time, not by eviction): a context
        token's K/V is written once and read by every later round. Depth covers
        max_seq_len plus the block headroom, page-aligned for FlashInfer.

        Normal tensors (inference_mode(False)): enable_mtp runs under the
        loader's InferenceMode but the job thread mutates these buffers in
        place outside it (the _alloc_mtp_kv contract).
        """
        assert self.k_cache is None, "alloc_kv called twice"
        depth = ((max_seq_len + self.block_size + _PAGE - 1) // _PAGE) * _PAGE
        self.depth = depth
        nkv, hd = self.num_kv_heads, self.head_dim
        with torch.inference_mode(False):
            self.k_cache = [torch.zeros((depth, nkv, hd), dtype = torch.half, device = self.dev)
                            for _ in range(self.num_hidden_layers)]
            self.v_cache = [torch.zeros((depth, nkv, hd), dtype = torch.half, device = self.dev)
                            for _ in range(self.num_hidden_layers)]
            # RoPE tables: own theta (1e6) and head_dim (128), NEOX style,
            # (1, 1, maxpos, hd) fp16 -- the ext_c.rope_ table contract.
            maxpos = depth + _PAGE
            inv_freq = 1.0 / (self.rope_theta ** (
                torch.arange(0, hd, 2, device = self.dev).float() / hd))
            t = torch.arange(maxpos, device = self.dev, dtype = torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)
            self._sin = emb.sin()[None, None].half()
            self._cos = emb.cos()[None, None].half()
            self._rope_maxpos = maxpos
        self.ctx_len = 0
        self._last_aux = None
        mb = depth * nkv * hd * 2 * 2 * self.num_hidden_layers / 2**20
        print(f"[dflash] draft KV cache: {self.num_hidden_layers} layers x "
              f"{depth} tokens x {nkv} heads x {hd} dim fp16 = {mb:.0f} MiB "
              f"on cuda:{self.device}", flush = True)

    def _rope(self, x: torch.Tensor, pos: int, num_heads: int):
        """In-place NEOX RoPE. x (1, rows, heads, hd); token i gets position pos+i
        (the kernel's sincos_row = past_len + row/num_heads)."""
        rows = x.shape[1]
        assert pos + rows <= self._rope_maxpos, "DFlash RoPE position overflow"
        ext_c.rope_(x.view(1, rows * num_heads, self.head_dim),
                    self._sin, self._cos, pos, num_heads, self.head_dim,
                    none_tensor, True)

    # ------------------------------------------------------------------ context feed

    def reset(self):
        """New lineage: drop all context (buffers kept; length bookkeeping only)."""
        self.ctx_len = 0
        self._last_aux = None

    def trim(self, n: int):
        """Rewind the context length to at most n (distributed rewind / cache trim).
        K/V beyond the length is never read; the next feed overwrites it."""
        if n < self.ctx_len:
            self._last_aux = None
        self.ctx_len = min(self.ctx_len, n)

    def feed_context(self, aux: torch.Tensor, p0: int):
        """Append (or overwrite) context K/V for tokens [p0, p0+rows).

        ``aux`` is (rows, n_features*hidden) -- the concatenated target residual
        states at target_layer_ids, in order. May be CPU (prefill pipe join) or
        device (verify rows). Processed in chunks to bound the fc GEMM transient.
        Overwrites are allowed (p0 <= ctx_len): the heal re-commit and the mode-A
        anchor re-feed rewrite already-committed positions.
        """
        assert self.k_cache is not None, "alloc_kv before feed_context"
        assert aux.shape[-1] == len(self.target_layer_ids) * self.hidden_size
        rows = aux.shape[0]
        if rows == 0:
            return
        assert p0 <= self.ctx_len, \
            f"DFlash context hole: feed at {p0} with ctx_len {self.ctx_len}"
        assert p0 + rows <= self.depth, "DFlash context exceeds draft KV depth"
        nkv, hd = self.num_kv_heads, self.head_dim
        L = self.num_hidden_layers
        for s in range(0, rows, _CTX_CHUNK):
            e = min(s + _CTX_CHUNK, rows)
            a = aux[s:e].to(self.dev, dtype = torch.half, non_blocking = True)
            fused = _rmsnorm(a @ self.fc.t(), self.hidden_norm, self.rms_norm_eps)
            kv = (fused @ self.fused_kv.t()).view(e - s, L, 2, nkv, hd)
            for li in range(L):
                k = kv[:, li, 0].contiguous()
                _head_rmsnorm(k, self.layers[li].k_norm, self.rms_norm_eps)
                self._rope(k.view(1, e - s, nkv, hd), p0 + s, nkv)
                self.k_cache[li][p0 + s : p0 + e].copy_(k)
                self.v_cache[li][p0 + s : p0 + e].copy_(kv[:, li, 1])
        self.ctx_len = p0 + rows
        # CPU invariant: the mode-A re-feed cats this row with the verify aux,
        # which always arrives on CPU from the pipe boundary (dflash3.txt:
        # cuda:0 vs cpu cat mismatch). 64 KB -- device residency buys nothing.
        self._last_aux = aux[-1:].to("cpu", dtype = torch.half).clone()

    def feed_verify(self, vaux: torch.Tensor, m_acc: int, pos: int, mode_b: bool):
        """Post-acceptance context feed from the verify window's aux rows.

        Mode B: verify rows 0..m_acc are [bonus@pos, accepted drafts@pos+1..]
        -> feed at pos. Mode A: the verify covers only the drafts (at pos..);
        the anchor at pos-1 had its context K/V overwritten by the block's
        noise K/V during drafting, so re-feed its stored aux row first.

        ``vaux`` is (rows, n_features*hidden) -- the verify window's aux states,
        consumed (copied off the graph's static buffer) by the driver.
        """
        if mode_b:
            self.feed_context(vaux[: m_acc + 1], pos)
        else:
            assert self._last_aux is not None, \
                "DFlash mode-A feed needs the anchor aux row (heal re-commit)"
            self.feed_context(torch.cat([self._last_aux, vaux[:m_acc]], dim = 0),
                              pos - 1)

    # ------------------------------------------------------------------ FlashInfer

    def _ensure_wrappers(self):
        if self._wrap_swa is not None:
            return
        import flashinfer
        from exllamav2 import attn_backend
        ws = attn_backend.FlashInferPaged._get_workspace(self.dev)
        # Eager wrappers: plan() per round at the round's kv_len (host-side),
        # run() once per layer. fa2 backend pinned (the fork's sm86-verified
        # family; auto resolves to cuDNN under graphs, irrelevant here).
        self._wrap_swa = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend = "fa2")
        self._wrap_full = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend = "fa2")

    def _plan(self, q_len: int, kv_len: int):
        """One single-request plan per geometry: q_len block queries over the
        kv_len prefix (context + block). Causal right-alignment puts query i at
        absolute position kv_len - q_len + i == pos + i, matching the RoPE'd
        keys. SWA: causal + window_left = W-1 (inclusive, == the reference's
        `query_pos - key_pos < sliding_window`). Full: non-causal, no window
        (bidirectional block + full context)."""
        pages = (kv_len + _PAGE - 1) // _PAGE
        indptr = torch.tensor([0, pages], dtype = torch.int32, device = self.dev)
        indices = torch.arange(pages, dtype = torch.int32, device = self.dev)
        lpl = torch.tensor([kv_len - (pages - 1) * _PAGE], dtype = torch.int32,
                           device = self.dev)
        qo_indptr = torch.tensor([0, q_len], dtype = torch.int32, device = self.dev)
        for wrap, causal, window in (
            (self._wrap_swa, True, self.sliding_window - 1),
            (self._wrap_full, False, -1),
        ):
            wrap.plan(
                qo_indptr = qo_indptr,
                paged_kv_indptr = indptr,
                paged_kv_indices = indices,
                paged_kv_last_page_len = lpl,
                num_qo_heads = self.num_q_heads,
                num_kv_heads = self.num_kv_heads,
                head_dim_qk = self.head_dim,
                page_size = _PAGE,
                causal = causal,
                pos_encoding_mode = "NONE",
                window_left = window,
                logits_soft_cap = None,
                sm_scale = self.sm_scale,
                q_data_type = torch.half,
                kv_data_type = torch.half,
            )

    # ------------------------------------------------------------------ draft block

    def gather_target(self, cache):
        """Interface compatibility with the MTP driver: DFlash context arrives
        via feed_context/feed_verify, not a K/V gather. No-op."""
        pass

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Target embedding rows (scale folded into the CPU table at load)."""
        tid = ids.to("cpu", non_blocking = False).long()
        e = self._t_embed_full[tid]
        return e.to(self.dev, non_blocking = True)

    def draft_window(self, last_token: torch.Tensor, last_state, pos: int,
                     window: int | None = None, return_logits: bool = False):
        """One parallel draft block. Returns ((1, window) CPU ids, None).

        ``last_token`` is the anchor (mode B: the pending bonus at ``pos``;
        mode A: the last committed token at ``pos-1``). ``last_state`` is
        ignored (DFlash conditions on context K/V, not a hidden-state seed).
        ``pos`` is the ANCHOR's absolute position; the mask queries sit at
        pos+1 .. pos+window and their argmax logits are the drafted tokens.
        """
        assert self.k_cache is not None and self.head is not None, \
            "DFlash drafter not enabled (alloc_kv/set_head)"
        assert pos in (self.ctx_len, self.ctx_len - 1), \
            f"DFlash anchor position {pos} vs ctx_len {self.ctx_len}"
        window = self.num_draft_tokens if window is None else int(window)
        assert 1 <= window <= self.block_size - 1
        q_len = window + 1
        kv_len = pos + q_len
        assert kv_len <= self.depth, "DFlash block exceeds draft KV depth"

        # NORMAL tensors: the first call runs inside the loader's InferenceMode
        # (load-time warmup), but these wrappers/buffers persist and are mutated
        # in place by every later plan() from the job thread OUTSIDE it --
        # inference tensors would raise there (dflash.txt: _kv_lens_buffer.copy_).
        # Same guard as the MTP drafter's batched path (_b_ensure_wrappers/_b_plan).
        with torch.inference_mode(False):
            self._ensure_wrappers()
            self._plan(q_len, kv_len)

        # Block input: [anchor, mask x window], target embeddings (scaled).
        ids = torch.full((1, q_len), self.mask_token_id, dtype = torch.long)
        ids[0, 0] = int(last_token.item())
        x = self._embed(ids).view(1, q_len, self.hidden_size)

        nq, nkv, hd = self.num_q_heads, self.num_kv_heads, self.head_dim
        for li, layer in enumerate(self.layers):
            y = _rmsnorm(x, layer.input_ln, self.rms_norm_eps)
            q = (y @ layer.q_proj.t()).view(1, q_len, nq, hd)
            _head_rmsnorm(q, layer.q_norm, self.rms_norm_eps)
            self._rope(q, pos, nq)
            k = (y @ layer.k_proj.t()).view(1, q_len, nkv, hd)
            _head_rmsnorm(k, layer.k_norm, self.rms_norm_eps)
            self._rope(k, pos, nkv)
            v = (y @ layer.v_proj.t()).view(1, q_len, nkv, hd)
            # Block (noise) K/V into the position-indexed cache; the attention
            # below reads context [0, pos) + block [pos, pos+q_len). The next
            # round's context feed overwrites the accepted prefix.
            self.k_cache[li][pos : pos + q_len].copy_(k.view(q_len, nkv, hd))
            self.v_cache[li][pos : pos + q_len].copy_(v.view(q_len, nkv, hd))
            wrap = self._wrap_full if layer.is_full else self._wrap_swa
            k_pages = self.k_cache[li].view(-1, _PAGE, nkv, hd)
            v_pages = self.v_cache[li].view(-1, _PAGE, nkv, hd)
            out = wrap.run(q.view(q_len, nq, hd), (k_pages, v_pages))
            a = out.view(1, q_len, nq * hd) @ layer.o_proj.t()
            x = x + a
            y = _rmsnorm(x, layer.post_attn_ln, self.rms_norm_eps)
            x = x + (F.silu(y @ layer.gate.t()) * (y @ layer.up.t())) @ layer.down.t()

        x = _rmsnorm(x, self.final_norm, self.rms_norm_eps)
        logits = x @ self.head.t()                    # (1, q_len, vocab)
        if self.softcap:
            ext_c.softcap_(logits, float(self.softcap))
        if self.head_padding > 0:
            logits[:, :, -self.head_padding:] = -65504.
        # Mask positions only: row 0 is the anchor (already decided).
        draft = logits[0, 1:, :].argmax(dim = -1).view(1, window)
        return draft.cpu(), None

    # ------------------------------------------------------------------ helpers

    def footprint(self) -> int:
        n = self.fc.numel() + self.hidden_norm.numel() + self.final_norm.numel()
        n += self.fused_kv.numel()
        if self.head is not None:
            n += self.head.numel()
        for L in self.layers:
            for v in vars(L).values():
                if torch.is_tensor(v):
                    n += v.numel()
        if self.k_cache is not None:
            n += sum(t.numel() for t in self.k_cache) + sum(t.numel() for t in self.v_cache)
        return n * 2
