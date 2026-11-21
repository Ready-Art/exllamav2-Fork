"""
Gemma-4 MTP drafter (google/gemma-4-31B-it-assistant).

A small 4-layer *cross-attention* draft model that rides on a Gemma-4-31B target.
It has no keys/values of its own: each of its layers projects only a query and
attends, with that query, over the TARGET's K/V cache -- the last sliding layer
(target layer 58) for its sliding layers and the last full layer (target layer 59)
for its full layer. It therefore writes no K/V, has no cache, and needs no prefill.

Per draft step it consumes:
  * the target's embedding of the token (5376-dim), and
  * a backbone-width state: the target's last decoder output *after* the target's
    final norm on the first step of a round, and the drafter's own post-projected
    output on subsequent steps.

Those are concatenated and projected down to the drafter width (1024) by
``pre_projection``. After the 4 blocks the drafter's ``post_projection`` produces
the next step's state, and its own tied head (1024-dim ``embed_tokens``) produces
logits over the shared 262144 vocab.

The drafter is a single-device model. The target may run tensor-parallel (KV
sharded across devices), so the two source layers' K/V are gathered onto the
drafter device once per verification round and reused for every draft step.

All drafter attention uses FlashInfer (``flashinfer.single_decode_with_kv_cache``),
per the task requirement -- NOT flash-attn. That single API covers both the
head_dim 512 full layer (which flash-attn's paged kernel refuses) and the
head_dim 256 sliding layers, and its ``window_left`` restores the sliding window.

Ported from the exllamav3-fork reference (``exllamav3/architecture/gemma4_assistant.py``
+ the generator's ``iterate_draftmodel_mtp_gen`` / ``iterate_gen`` MTP path),
adapted to the EXL2 fork's module/config/cache conventions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from exllamav2.compat import safe_move_tensor
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

# The drafter's own geometry (google/gemma-4-31B-it-assistant). These are read from
# the checkpoint's config.json at load time, but the defaults here document the shape.
_DRAFTER_HIDDEN = 1024
_DRAFTER_INTERMEDIATE = 8192
_DRAFTER_LAYERS = 4
_DRAFTER_LAYER_TYPES = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
_DRAFTER_HEAD_DIM = 256
_DRAFTER_GLOBAL_HEAD_DIM = 512
_DRAFTER_NUM_Q_HEADS = 32
_DRAFTER_NUM_KV_HEADS = 16
_DRAFTER_NUM_GLOBAL_KV_HEADS = 4
_DRAFTER_SLIDING_WINDOW = 1024
_DRAFTER_VOCAB = 262144


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Weighted RMSNorm over the last dim (fp32 reduce, back to input dtype).

    BUG 38 pattern (same fix the target got): the pure-torch fp32 chain was ~8 kernel
    dispatches per norm; at ~17 norms per draft step that was ~130 dispatches/step of
    host time, the bulk of the drafter's per-step cost. The fused ext kernel does the
    identical math (fp32 reduce, single rounding) in one launch.
    """
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    y = torch.empty_like(x2)
    ext_c.rms_norm(x2, weight, y, eps)
    return y.view(shape)


def _head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm. ``x`` is (..., heads, head_dim), ``weight`` is (heads, head_dim)
    (the loader repeats the checkpoint's shared (head_dim,) weight, as headnorm.py does)."""
    if x.dtype != torch.half:
        x = x.to(torch.half)
    if not x.is_contiguous():
        x = x.contiguous()
    ext_c.head_norm(x, weight, none_tensor, x, eps, True)
    return x


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate = "tanh")


@dataclass
class _DrafterLayer:
    layer_idx: int
    is_full: bool
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    rope_index: int          # 0 = base (proportional, 512), 1 = alt (default, 256)
    sliding_window: int      # 0 for full
    input_ln: torch.Tensor
    q_proj: torch.Tensor     # (num_q*hd, hidden)
    q_norm: torch.Tensor     # (num_q_heads, head_dim), repeated from the shared (head_dim,)
    o_proj: torch.Tensor     # (hidden, num_q*hd)
    post_attn_ln: torch.Tensor
    pre_ff_ln: torch.Tensor
    gate: torch.Tensor       # (inter, hidden)
    up: torch.Tensor         # (inter, hidden)
    down: torch.Tensor       # (hidden, inter)
    post_ff_ln: torch.Tensor
    layer_scalar: torch.Tensor


class Gemma4AssistantDrafter:

    """
    The Gemma-4 MTP draft model. Construct with the loaded target model, the draft
    checkpoint directory, and a device. Call :meth:`gather_target` once per
    verification round (after the target has advanced its cache), then
    :meth:`draft_window` to produce a speculative window.
    """

    def __init__(
        self,
        target,
        draft_dir: str,
        device: int = 0,
        num_draft_tokens: int = 6,
    ):
        self.target = target
        self.tcfg = target.config
        self.device = device
        self.dev = torch.device("cuda", device)
        self.num_draft_tokens = num_draft_tokens

        # Drafter geometry from its own config.json (fall back to the known defaults).
        self._read_config(draft_dir)

        # Target-side sources the drafter taps.
        #   * embedding  : target.modules[0] (CPU table, is_tp in TP mode)
        #   * final norm : the RMSNorm just before the target's lm_head
        #   * kv layers  : last full (59) and last sliding (58) target attention layers
        self.t_embed = target.modules[0]
        # The drafter indexes the target table with ARBITRARY token ids directly
        # (_target_embed / _stage_emb bypass the shard-aware forward). With the
        # vocab-sharded embedding (EXL2_TP_VOCAB_EMBED) the rank-local table holds
        # only this rank's rows -> out-of-bounds CUDA assert. Keep a full CPU copy
        # for the drafter: host RAM is plentiful, the per-token H2D is ~10 KB, and
        # the lookup stays off the collective path (the drafter runs on the master
        # alone, so it cannot participate in the embedding all_reduce).
        self._t_embed_full = None
        if getattr(self.t_embed, "_tp_vocab_shard", None) is not None:
            _prev = self.t_embed.device_idx
            self.t_embed.device_idx = -1
            try:
                _w = self.t_embed.load_weight(cpu = True)
            finally:
                self.t_embed.device_idx = _prev
            _w = _w.data.clone()
            if target.config.scale_emb != 1:
                _w *= target.config.scale_emb
            self._t_embed_full = _w
        self.t_final_norm = target.modules[target.head_layer_idx - 1]
        assert type(self.t_final_norm).__name__ == "ExLlamaV2RMSNorm", \
            f"drafting requires an RMSNorm final norm, got {type(self.t_final_norm).__name__}"

        n = self.tcfg.num_hidden_layers - getattr(self.tcfg, "num_kv_shared_layers", 0)
        types = list(self.tcfg.layer_types[:n])
        self.t_last_full = max(i for i, t in enumerate(types) if t == "full_attention")
        self.t_last_swa = max(i for i, t in enumerate(types) if t == "sliding_attention")

        # Drafter weights (fp16 on self.device).
        self._load_weights(draft_dir)

        # Per-round gathered target K/V (None until gather_target runs).
        self.kv_full = None   # (1, L, num_global_kv_heads, global_head_dim)
        self.kv_swa = None    # (1, L, num_kv_heads, head_dim)
        self.kv_full_len = 0
        self.kv_swa_len = 0
        self.kv_swa_window_beg = 0

        # RoPE tables come from the target's device context so the drafter's queries
        # rotate exactly as the target's keys did (proportional for full, default for
        # sliding). Cached per device.
        self._rope = None

        # Change98: CUDA-graph draft window (MP backend). The eager drafter is
        # host-launch-bound (~23 ms for 6 steps x 4 layers of small torch ops); one
        # per-step graph replayed `window` times cuts it to ~7 ms. The graph reads the
        # FULL-DEPTH gather buffers (stable addresses) with device-side valid-length
        # masks, so it is position- and length-agnostic across rounds. The embedding
        # lookup stays host-side (CPU table) and is staged into a static buffer between
        # replays; the argmax id is read back per step to feed the next embedding.
        self._draft_graph = None
        self._graph_failed = False
        self._g_emb = None          # (1,1,backbone) static embedded-token input
        self._g_state = None        # (1,1,backbone) state in/out (aliased next_state)
        self._g_id = None           # (1,1) int64 static argmax output
        self._g_pos = None          # (1,) int32 device frontier position (RoPE)
        self._g_kvfull_len = None   # (1,) int32 device valid full-layer length
        self._g_kvswa_len = None    # (1,) int32 device valid SWA prefix length
        self._kf_base = None        # full-depth (max_seq_len, kvh, hd) gather buffers
        self._vf_base = None
        self._ks_base = None        # ring-depth (ring, kvh, hd)
        self._vs_base = None

        # Multi-batch drafting (MP slot scheduler): the gathered shadows are 4-D
        # (max_batch, depth, kvh, hd); one eager FlashInfer batch-decode wrapper per
        # source geometry serves all active rows. The drafter writes no K/V, so the
        # per-row kv lengths are FIXED within a round: plan once, run per draft step.
        self._b_st = None           # target._mtp_kv (4-D shadows)
        self._b_rows = None         # cache rows of the active batch
        self._b_full_lens = None
        self._b_swa_lens = None
        self._b_wrap_full = None
        self._b_wrap_swa = None
        self._b_pos = None          # (max_batch,) int32 device frontier positions

    # ------------------------------------------------------------------ config

    def _read_config(self, draft_dir: str):
        path = os.path.join(draft_dir, "config.json")
        cfg = {}
        tc = {}
        if os.path.exists(path):
            with open(path, encoding = "utf-8") as f:
                cfg = json.load(f)
            tc = cfg.get("text_config", cfg)
        # backbone_hidden_size is the *target* width (5376), read from the top-level config.
        self.backbone_hidden_size = cfg.get("backbone_hidden_size", self.tcfg.hidden_size)
        self.hidden_size = tc.get("hidden_size", _DRAFTER_HIDDEN)
        self.intermediate_size = tc.get("intermediate_size", _DRAFTER_INTERMEDIATE)
        self.num_hidden_layers = tc.get("num_hidden_layers", _DRAFTER_LAYERS)
        self.layer_types = tc.get("layer_types", _DRAFTER_LAYER_TYPES)
        self.head_dim = tc.get("head_dim", _DRAFTER_HEAD_DIM)
        self.global_head_dim = tc.get("global_head_dim", tc.get("head_dim", _DRAFTER_GLOBAL_HEAD_DIM))
        self.num_q_heads = tc.get("num_attention_heads", _DRAFTER_NUM_Q_HEADS)
        self.num_kv_heads = tc.get("num_key_value_heads", _DRAFTER_NUM_KV_HEADS)
        self.num_global_kv_heads = tc.get("num_global_key_value_heads", tc.get("num_key_value_heads", _DRAFTER_NUM_GLOBAL_KV_HEADS))
        self.sliding_window = tc.get("sliding_window", _DRAFTER_SLIDING_WINDOW)
        self.rms_norm_eps = tc.get("rms_norm_eps", 1e-6)
        self.vocab_size = tc.get("vocab_size", _DRAFTER_VOCAB)
        # num_assistant_tokens from generation_config.json (default draft length).
        self.num_assistant_tokens = 6
        gpath = os.path.join(draft_dir, "generation_config.json")
        if os.path.exists(gpath):
            try:
                with open(gpath, encoding = "utf-8") as f:
                    self.num_assistant_tokens = int(json.load(f).get("num_assistant_tokens", 6) or 6)
            except (OSError, ValueError):
                pass

    # ------------------------------------------------------------------ weights

    def _load_weights(self, draft_dir: str):
        from safetensors import safe_open
        st_path = os.path.join(draft_dir, "model.safetensors")
        if not os.path.exists(st_path):
            # Multi-file checkpoints: pick the first *.safetensors.
            cands = sorted(f for f in os.listdir(draft_dir) if f.endswith(".safetensors"))
            assert cands, f"no safetensors in {draft_dir}"
            st_path = os.path.join(draft_dir, cands[0])

        W = {}
        with safe_open(st_path, framework = "pt") as f:
            for k in f.keys():
                W[k] = f.get_tensor(k)

        dev = self.dev
        h = self.hidden_size

        def g(key, dtype = torch.half):
            assert key in W, f"missing draft weight {key}"
            return W[key].to(dev, dtype = dtype).contiguous()

        self.pre_projection = g("pre_projection.weight")      # (h, 2*backbone)
        self.post_projection = g("post_projection.weight")    # (backbone, h)
        self.head = g("model.embed_tokens.weight")            # (vocab, h)  tied
        self.final_norm = g("model.norm.weight")              # (h,)

        self.layers = []
        for i in range(self.num_hidden_layers):
            full = self.layer_types[i] == "full_attention"
            hd = self.global_head_dim if full else self.head_dim
            nkv = self.num_global_kv_heads if full else self.num_kv_heads
            prefix = f"model.layers.{i}"
            self.layers.append(_DrafterLayer(
                layer_idx = i,
                is_full = full,
                head_dim = hd,
                num_q_heads = self.num_q_heads,
                num_kv_heads = nkv,
                rope_index = 0 if full else 1,
                sliding_window = 0 if full else self.sliding_window,
                input_ln = g(f"{prefix}.input_layernorm.weight"),
                q_proj = g(f"{prefix}.self_attn.q_proj.weight"),
                # head_norm kernel indexes w per head (w + hd*head); repeat the shared
                # (hd,) checkpoint weight to (nq, hd), exactly as ExLlamaV2HeadNorm.load.
                q_norm = g(f"{prefix}.self_attn.q_norm.weight").repeat(self.num_q_heads, 1),
                o_proj = g(f"{prefix}.self_attn.o_proj.weight"),
                post_attn_ln = g(f"{prefix}.post_attention_layernorm.weight"),
                pre_ff_ln = g(f"{prefix}.pre_feedforward_layernorm.weight"),
                gate = g(f"{prefix}.mlp.gate_proj.weight"),
                up = g(f"{prefix}.mlp.up_proj.weight"),
                down = g(f"{prefix}.mlp.down_proj.weight"),
                post_ff_ln = g(f"{prefix}.post_feedforward_layernorm.weight"),
                layer_scalar = g(f"{prefix}.layer_scalar").view(1),
            ))

    # ------------------------------------------------------------------ rope

    def _get_rope(self):
        if self._rope is None:
            ctx = self.target.get_device_context(self.device, scratch = False)
            # rope_index 0 = base (proportional, full 512), 1 = alt (default, sliding 256)
            self._rope = (
                (ctx.sin[0], ctx.cos[0]),
                (ctx.sin[1], ctx.cos[1]),
            )
        return self._rope

    # ------------------------------------------------------------------ target gather

    def gather_target(self, cache):
        """
        Gather the target's last full (59) and last sliding (58) K/V onto the drafter
        device, once per round. ``cache`` is the (TP) target cache.

        MP backend: when the target is a ``TPProcessGroup`` (its K/V lives in
        separate worker processes, one per GPU) the gather goes through
        ``target.gather_kv_mtp()`` -- incremental pipe pulls joined on the parent --
        instead of the in-process cross-device copies below.
        """
        if cache is None:
            return
        if hasattr(self.target, "gather_kv_mtp"):
            k_full, v_full, k_swa, v_swa, seq_len, window_beg = self.target.gather_kv_mtp()
            self.kv_full = (k_full, v_full)
            self.kv_full_len = seq_len
            self.kv_swa = (k_swa, v_swa)
            # Change98: the gathered SWA tensor is 3-D (S, kvh, hd) -- the valid length is
            # shape[0]. shape[1] is the kv-head COUNT (16), which made the drafter's three
            # sliding layers attend to only the oldest 16 ring rows instead of the last
            # `sliding_window` keys, depressing MP acceptance. The non-MP branch below
            # already uses the correct length.
            self.kv_swa_len = k_swa.shape[0]
            self.kv_swa_window_beg = window_beg
            # Change98: the draft graph reads the FULL-DEPTH buffers (fixed shape, stable
            # address) with device-side length masks, not the per-round prefix slices.
            st = getattr(self.target, "_mtp_kv", None)
            if st is not None:
                self._kf_base = st["kf"]
                self._vf_base = st["vf"]
                self._ks_base = st["ks"]
                self._vs_base = st["vs"]
            return
        # TP cache: per-device slices. Non-TP: a single cache.
        if hasattr(cache, "caches"):
            per_dev = [(i, c) for i, c in enumerate(cache.caches) if c is not None]
            seq_len = cache.current_seq_len
            window_beg = getattr(cache, "window_beg", 0)
        else:
            per_dev = [(self.device, cache)]
            seq_len = cache.current_seq_len
            window_beg = 0

        # Full layer (59): each device holds a slice of the global kv heads (1 head on a
        # 4-way split); joining them along the kv-head axis (dim 1 of the sliced (S, nkv, hd)
        # tensor) reconstructs the full (S, num_global_kv_heads, global_head_dim) cache.
        k_full = torch.cat([c.key_states[self.t_last_full][0, :seq_len].to(self.dev, non_blocking = True) for _, c in per_dev], dim = 1)
        v_full = torch.cat([c.value_states[self.t_last_full][0, :seq_len].to(self.dev, non_blocking = True) for _, c in per_dev], dim = 1)
        self.kv_full = (k_full, v_full)
        self.kv_full_len = seq_len

        # Sliding layer (58): ring of depth kv_state_size; valid prefix is
        # [0, seq_len - window_beg). Same kv-head join (16 heads total across devices).
        swa = per_dev[0][1]
        ring = swa.kv_state_size
        swa_len = min(seq_len - window_beg, ring)
        k_swa = torch.cat([c.key_states[self.t_last_swa][0, :swa_len].to(self.dev, non_blocking = True) for _, c in per_dev], dim = 1)
        v_swa = torch.cat([c.value_states[self.t_last_swa][0, :swa_len].to(self.dev, non_blocking = True) for _, c in per_dev], dim = 1)
        self.kv_swa = (k_swa, v_swa)
        self.kv_swa_len = swa_len
        self.kv_swa_window_beg = window_beg

    # ------------------------------------------------------------------ embedding / state

    def _target_embed(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        Target embedding of a (1,1) token id, moved to the drafter device (fp16).

        Does a direct row lookup on the target's (CPU-resident in TP mode) embedding table
        rather than calling ``t_embed.forward``: the latter, in TP mode, writes into the
        target's pinned staging buffer (``ctx.copy_pinned``), which we don't want to touch
        between rounds. The result is scaled exactly as the target's own forward scales it
        (``normalize_embeddings`` -> *sqrt(hidden)), so the drafter sees the same values the
        target's blocks saw.
        """
        tid = token_id.to("cpu", non_blocking = False).long()
        # Full CPU copy when the target table is vocab-sharded (see __init__).
        w = self._t_embed_full if self._t_embed_full is not None \
            else self.t_embed.embedding.weight                 # (vocab, backbone)
        e = w[tid]                                             # (1,1,backbone)
        ap = self.t_embed.archparams
        if getattr(ap, "normalize_embeddings", False):
            e = e * (self.backbone_hidden_size ** 0.5)
        if e.device != self.dev:
            e = e.to(self.dev, non_blocking = True)
        return e.half()

    def _target_final_norm(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the target's final RMSNorm to a (1,1,backbone) pre-norm state."""
        w = self.t_final_norm.weight
        if w.device != self.dev:
            w = w.to(self.dev, non_blocking = True)
        eps = self.t_final_norm.variance_epsilon
        return _rmsnorm(state, w, eps)

    # ------------------------------------------------------------------ attention

    def _cross_attn(self, layer: _DrafterLayer, x_norm: torch.Tensor, pos: int):
        """
        Cross-attention of one drafter layer: project the query, norm + RoPE it, and
        attend (FlashInfer) over the gathered target K/V for this layer's source.
        ``x_norm`` is (1, 1, hidden). Returns (1, 1, hidden) (the o_proj output).
        """
        import flashinfer
        hd = layer.head_dim
        nq = layer.num_q_heads
        q = x_norm @ layer.q_proj.t()                    # (1,1,nq*hd)
        q = q.view(1, 1, nq, hd)
        q = _head_rmsnorm(q, layer.q_norm, self.rms_norm_eps)
        # RoPE the query at absolute position pos (NEOX), matching the target's keys.
        sin, cos = self._get_rope()[layer.rope_index]
        ext_c.rope_(q, sin, cos, pos, nq, hd, none_tensor, True)

        # Select the source K/V and its valid length.
        if layer.is_full:
            k, v = self.kv_full
            kv_len = self.kv_full_len
            window_left = -1
        else:
            k, v = self.kv_swa
            kv_len = self.kv_swa_len
            # FlashInfer's window_left is INCLUSIVE (last window_left+1 keys), matching
            # flash-attn's window_size=(W, W) on a causal decode only when passed as W-1.
            window_left = self.sliding_window - 1

        # FlashInfer single-request decode: q (H, D), k/v (S, HKV, D) NHD, no RoPE
        # (pos_encoding_mode NONE -- the query is already rotated, the keys were
        # rotated by the target). GQA is handled internally.
        # gather_target() already stored k/v as (S, HKV, D); take the valid prefix only.
        q2 = q[0, 0].contiguous()                        # (nq, hd)
        k2 = k[:kv_len].contiguous()                     # (kv_len, nkv, hd)
        v2 = v[:kv_len].contiguous()
        out = flashinfer.single_decode_with_kv_cache(
            q2, k2, v2,
            kv_layout = "NHD",
            pos_encoding_mode = "NONE",
            window_left = window_left,
            sm_scale = 1.0,
        )                                                # (nq, hd)
        out = out.view(1, 1, nq * hd)
        return out @ layer.o_proj.t()                    # (1,1,hidden)

    def _cross_attn_graph(self, layer: _DrafterLayer, x_norm: torch.Tensor) -> torch.Tensor:
        """
        CUDA-graph-compatible cross-attention (Change98). Same math as _cross_attn but
        capturable: FlashInfer's single_decode bakes kv_len (the prefix slice shape), so
        the graph path instead attends over the FULL-DEPTH gather buffer with an additive
        mask built on-device from the dynamic valid length -- exactly the trick the target
        TP graph uses (BUG 43). RoPE reads the frontier position from a device tensor
        (past_len = -1). sm_scale stays 1.0 to match the FI call above.
        """
        hd = layer.head_dim
        nq = layer.num_q_heads
        q = x_norm @ layer.q_proj.t()                    # (1,1,nq*hd)
        q = q.view(1, 1, nq, hd)
        q = _head_rmsnorm(q, layer.q_norm, self.rms_norm_eps)
        sin, cos = self._get_rope()[layer.rope_index]
        ext_c.rope_(q, sin, cos, -1, nq, hd, self._g_pos, True)

        if layer.is_full:
            k, v = self._kf_base, self._vf_base          # (Df, nkv, hd)
            kv_len = self._g_kvfull_len
            window = 0
        else:
            k, v = self._ks_base, self._vs_base          # (Ds, nkv, hd)
            kv_len = self._g_kvswa_len
            window = self.sliding_window

        nkv = k.shape[1]
        g = nq // nkv                                    # query heads per kv head
        D = k.shape[0]
        # Change98: attention math in fp32 to match FlashInfer (which accumulates scores and
        # softmaxes in fp32). An fp16 score matmul rounds scores BEFORE softmax; at long
        # context (many near-tied keys) that shifted the attention output enough to change
        # drafts and drop acceptance (32k filler acc 3.18 -> 2.45). fp32 scores restore it.
        qg = q.view(nkv, g, hd).float()                  # GQA grouping: head h -> kv h//g
        scores = torch.matmul(qg, k.permute(1, 2, 0).float())   # (nkv, g, D) fp32
        ki = torch.arange(D, device = k.device).view(1, 1, D)
        kl = kv_len.view(1, 1, 1)
        ok = ki < kl
        if window:
            ok = ok & (ki >= kl - window)
        # SELECT masked scores, don't add a mask: the matmul reads the full-depth gather
        # buffer whose unwritten tail rows can be NaN/inf garbage; an additive mask leaves
        # NaN+(-65504)=NaN, poisoning the softmax row. torch.where replaces masked positions
        # outright. (Buffers are also zero-initialized now; this is belt-and-suspenders.)
        scores = torch.where(ok, scores, torch.full_like(scores, -65504.0))
        probs = torch.softmax(scores, dim = -1)          # fp32
        out = torch.matmul(probs, v.permute(1, 0, 2).float())   # (nkv, g, hd) fp32
        out = out.reshape(1, 1, nq * hd).to(torch.half)
        return out @ layer.o_proj.t()                    # (1,1,hidden)

    # ------------------------------------------------------------------ block

    def _block(self, layer: _DrafterLayer, x: torch.Tensor, pos: int,
               graph: bool = False) -> torch.Tensor:
        """
        One drafter transformer block (Gemma-4 post-norm convention, matching the
        exllamav3 reference and the EXL2 target):

            x += post_attn_ln( cross_attn( input_ln(x) ) )
            x += post_ff_ln(  mlp( pre_ff_ln(x) ) )
            x *= layer_scalar
        """
        h = self.hidden_size
        # Attention branch
        y = _rmsnorm(x, layer.input_ln, self.rms_norm_eps)
        a = self._cross_attn_graph(layer, y) if graph else self._cross_attn(layer, y, pos)
        a = _rmsnorm(a, layer.post_attn_ln, self.rms_norm_eps)
        x = x + a
        # MLP branch
        y = _rmsnorm(x, layer.pre_ff_ln, self.rms_norm_eps)
        gate = y @ layer.gate.t()
        up = y @ layer.up.t()
        y = _gelu_tanh(gate) * up
        y = y @ layer.down.t()
        y = _rmsnorm(y, layer.post_ff_ln, self.rms_norm_eps)
        x = x + y
        # Layer scalar
        x = x * layer.layer_scalar
        return x

    # ------------------------------------------------------------------ one step

    def _step(self, token_id: torch.Tensor, state: torch.Tensor, pos: int, first: bool):
        """
        One draft step. Returns (logits (1,1,vocab), next_state (1,1,backbone)).
        ``state`` is the backbone-width input state (pre-final-norm on the first
        step, already post-projected afterwards).
        """
        if first:
            state = self._target_final_norm(state)
        emb = self._target_embed(token_id)               # (1,1,backbone)
        y = torch.cat((emb, state), dim = -1)            # (1,1,2*backbone)
        x = y @ self.pre_projection.t()                  # (1,1,hidden)
        for layer in self.layers:
            x = self._block(layer, x, pos)
        x = _rmsnorm(x, self.final_norm, self.rms_norm_eps)
        next_state = x @ self.post_projection.t()        # (1,1,backbone)
        logits = x @ self.head.t()                       # (1,1,vocab)
        return logits, next_state

    # ------------------------------------------------------------------ draft window

    def draft_window(self, last_token: torch.Tensor, last_state: torch.Tensor, pos: int,
                     window: int | None = None, return_logits: bool = False):
        """
        Produce a speculative window of ``window`` token ids (greedy) starting from
        ``last_token`` / ``last_state``.

        The drafter reads the target's K/V, which ends at position ``pos - 1`` (the target
        has ``pos`` tokens in cache). Its query therefore sits at the frontier position
        ``pos`` for EVERY draft step -- it does not advance, because the drafter writes no
        K/V of its own. This matches the exllamav3 reference (``params["positions"] =
        kv_seqlens``, held constant across the draft loop).

        :param last_token: (1,1) id of the last accepted target token (the drafter's
            first input).
        :param last_state: (1,1,backbone) target pre-final-norm hidden state of that
            token (the round's first state input).
        :param pos: the target's current sequence length (number of cached tokens).
        :param window: number of draft tokens to produce (default num_draft_tokens).
        :return: (1, window) tensor of drafted token ids.
        """
        assert self.kv_full is not None and self.kv_swa is not None, \
            "call gather_target(cache) before draft_window"
        window = self.num_draft_tokens if window is None else window
        # Change98: graph path (MP backend only). Default OFF -- eager FlashInfer drafting
        # is the best verified state; EXL2_MTP_DRAFT_GRAPH=1 opts into the experiment.
        # Finding: the masked-GEMV graph (the only graph-safe dynamic-length attention here;
        # FI paged bakes scheduling, ragged crashes on replay) cannot beat eager FI. fp16
        # scores are fast (draft ~13.6 ms) but round before softmax and drop long-ctx
        # acceptance (32k 3.18->2.45). fp32 scores restore acceptance (cos 1.0 vs FI) but
        # the full-depth fp32 attention is bandwidth-bound (draft ~18 ms @4k, ~34 ms @32k),
        # erasing the launch-overhead saving: net neutral @4k/counting, a LOSS @32k vs eager.
        # Eager FI reads only kv_len in fp16 with optimized kernels. Re-enable only if a
        # graph-safe dynamic-length hd512 FI kernel becomes available.
        # BUG 105: the graph draft path only produces argmax ids (no per-step logits), so
        # when the caller needs logits for probabilistic acceptance (return_logits=True) we
        # must use the eager path, which returns (ids, logits). Otherwise the sampling caller
        # would unpack a single ids tensor and crash.
        if (self._kf_base is not None and not self._graph_failed and not return_logits
                and os.environ.get("EXL2_MTP_DRAFT_GRAPH", "0") == "1"):
            self._ensure_draft_graph()
            if self._draft_graph is not None:
                return self._draft_window_graph(last_token, last_state, pos, window), None
        ids = []
        logits_list = []
        tok = last_token.to(self.dev, non_blocking = True)
        state = last_state.to(self.dev, non_blocking = True).half()
        if state.dim() == 1:                      # tolerate a flattened (H,) state
            state = state.view(1, 1, -1)
        for k in range(window):
            logits, state = self._step(tok, state, pos, first = (k == 0))
            nxt = torch.argmax(logits, dim = -1)         # (1,1)
            ids.append(nxt)
            logits_list.append(logits.float())
            tok = nxt
        ids_t = torch.cat(ids, dim = -1)                 # (1, window)
        # BUG 105: ALWAYS return a 2-tuple (ids, logits_or_None) so callers never unpack a
        # bare tensor (which iterates the window dim and crashes with a window-size-dependent
        # "not enough values to unpack"). logits is None unless return_logits=True.
        if return_logits:
            return ids_t, torch.cat(logits_list, dim = 1)
        return ids_t, None

    # ------------------------------------------------------------------ batched drafting

    def gather_target_batch(self, rows):
        """
        Multi-batch counterpart of gather_target: the target's joined K/V shadows are
        already per-row (TPProcessGroup._mtp_kv, 4-D), so this only records which rows
        are active and their valid lengths. Call after gather_kv_mtp_batch(rows).
        """
        st = getattr(self.target, "_mtp_kv", None)
        assert st is not None and st["kf"].dim() == 4, \
            "batched drafting requires the multi-batch MTP shadows (max_batch > 1)"
        self._b_st = st
        self._b_rows = list(rows)
        cache = self.target.cache
        self._b_full_lens = [cache.seq_lens[r] for r in rows]
        ring = st["ks"].shape[1]
        self._b_swa_lens = [min(cache.seq_lens[r] - cache.window_begs[r], ring)
                            for r in rows]

    def _b_ensure_wrappers(self):
        import flashinfer
        from exllamav2 import attn_backend
        if self._b_wrap_full is not None:
            return
        ws = attn_backend.FlashInferPaged._get_workspace(self.dev)
        # Eager wrappers (no CUDA graph): plan() once per round refreshes the scheduling
        # for the current per-row lengths; run() is one batched kernel per draft step.
        self._b_wrap_full = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", backend = "fa2")
        self._b_wrap_swa = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", backend = "fa2")

    def _b_plan(self):
        st = self._b_st
        page = 128
        for wrap, tensor, lens, nkv, hd, window in (
            (self._b_wrap_full, st["kf"], self._b_full_lens,
             self.num_global_kv_heads, self.global_head_dim, -1),
            (self._b_wrap_swa, st["ks"], self._b_swa_lens,
             self.num_kv_heads, self.head_dim, self.sliding_window - 1),
        ):
            depth = tensor.shape[1]
            npages = depth // page
            indptr = [0]
            indices = []
            lpl = []
            for r, l in zip(self._b_rows, lens):
                l = max(1, l)
                pv = (l + page - 1) // page
                base = r * npages
                indices.extend(range(base, base + pv))
                indptr.append(indptr[-1] + pv)
                lpl.append(l - (pv - 1) * page)
            wrap.plan(
                indptr = torch.tensor(indptr, dtype = torch.int32, device = self.dev),
                indices = torch.tensor(indices, dtype = torch.int32, device = self.dev),
                last_page_len = torch.tensor(lpl, dtype = torch.int32, device = self.dev),
                num_qo_heads = self.num_q_heads,
                num_kv_heads = nkv,
                head_dim = hd,
                page_size = page,
                pos_encoding_mode = "NONE",
                window_left = window,
                logits_soft_cap = None,
                sm_scale = 1.0,
                q_data_type = torch.half,
                kv_data_type = tensor.dtype,
            )

    def _cross_attn_batch(self, layer: _DrafterLayer, x_norm: torch.Tensor,
                          pos_t: torch.Tensor) -> torch.Tensor:
        """Batched cross-attention: (B,1,hidden) queries over the per-row target K/V.
        Same math as _cross_attn (RoPE at the per-row frontier, sm_scale 1.0, inclusive
        window_left = W-1), one FlashInfer batch-decode run for all rows."""
        hd = layer.head_dim
        nq = layer.num_q_heads
        B = x_norm.shape[0]
        q = x_norm @ layer.q_proj.t()                    # (B,1,nq*hd)
        q = q.view(B, 1, nq, hd)
        q = _head_rmsnorm(q, layer.q_norm, self.rms_norm_eps)
        sin, cos = self._get_rope()[layer.rope_index]
        ext_c.rope_(q, sin, cos, -1, nq, hd, pos_t, True)
        st = self._b_st
        page = 128
        if layer.is_full:
            k, v = st["kf"], st["vf"]
            wrap = self._b_wrap_full
            nkv = self.num_global_kv_heads
        else:
            k, v = st["ks"], st["vs"]
            wrap = self._b_wrap_swa
            nkv = self.num_kv_heads
        k_pages = k.view(-1, page, nkv, hd)
        v_pages = v.view(-1, page, nkv, hd)
        out = wrap.run(q.view(B, nq, hd), (k_pages, v_pages))   # (B, nq, hd)
        return out.view(B, 1, nq * hd) @ layer.o_proj.t()

    def _block_batch(self, layer: _DrafterLayer, x: torch.Tensor,
                     pos_t: torch.Tensor) -> torch.Tensor:
        y = _rmsnorm(x, layer.input_ln, self.rms_norm_eps)
        a = self._cross_attn_batch(layer, y, pos_t)
        a = _rmsnorm(a, layer.post_attn_ln, self.rms_norm_eps)
        x = x + a
        y = _rmsnorm(x, layer.pre_ff_ln, self.rms_norm_eps)
        gate = y @ layer.gate.t()
        up = y @ layer.up.t()
        y = _gelu_tanh(gate) * up
        y = y @ layer.down.t()
        y = _rmsnorm(y, layer.post_ff_ln, self.rms_norm_eps)
        x = x + y
        return x * layer.layer_scalar

    def _step_batch(self, tok: torch.Tensor, state: torch.Tensor,
                    pos_t: torch.Tensor, first: bool):
        """One batched draft step. tok (B,1) device ids, state (B,1,backbone)."""
        if first:
            w = self.t_final_norm.weight
            if w.device != self.dev:
                w = w.to(self.dev, non_blocking = True)
            state = _rmsnorm(state, w, self.t_final_norm.variance_epsilon)
        tid = tok.to("cpu", non_blocking = False).long()
        w = self._t_embed_full if self._t_embed_full is not None \
            else self.t_embed.embedding.weight
        emb = w[tid]                                     # (B,1,backbone) CPU
        ap = self.t_embed.archparams
        if getattr(ap, "normalize_embeddings", False):
            emb = emb * (self.backbone_hidden_size ** 0.5)
        emb = emb.to(self.dev, non_blocking = True).half()
        y = torch.cat((emb, state), dim = -1)
        x = y @ self.pre_projection.t()
        for layer in self.layers:
            x = self._block_batch(layer, x, pos_t)
        x = _rmsnorm(x, self.final_norm, self.rms_norm_eps)
        next_state = x @ self.post_projection.t()
        logits = x @ self.head.t()
        return logits, next_state

    def draft_window_batch(self, base_tok: torch.Tensor, base_state: torch.Tensor,
                           pos: list, window: int | None = None) -> torch.Tensor:
        """
        Batched greedy draft window: (B, window) CPU ids for the active rows.

        ``base_tok`` (B,1) is each row's last decided token (the pending bonus),
        ``base_state`` (B,1,backbone) the target pre-final-norm state of each row's last
        COMMITTED token (the mode-B seed, one position stale exactly as in the scalar
        path), and ``pos[b]`` the row's frontier query position. Call
        gather_target_batch(rows) first; the per-row K/V lengths are fixed for the round.
        """
        assert self._b_st is not None, "call gather_target_batch(rows) first"
        window = self.num_draft_tokens if window is None else window
        B = len(self._b_rows)
        # NORMAL tensors: the first call may run inside the loader's InferenceMode
        # (load-time batch warmup), but these wrappers/buffers persist and are mutated
        # in place by every later plan()/fill from the scheduler thread OUTSIDE it.
        # Inference tensors would raise on those later in-place updates.
        with torch.inference_mode(False):
            self._b_ensure_wrappers()
            self._b_plan()
            if self._b_pos is None or self._b_pos.numel() < B:
                self._b_pos = torch.zeros(max(B, getattr(self.target, "max_batch", B)),
                                          dtype = torch.int32, device = self.dev)
            self._b_pos[:B].copy_(torch.tensor(pos, dtype = torch.int32))
        tok = base_tok.to(self.dev, non_blocking = True).long()
        state = base_state.to(self.dev, non_blocking = True).half()
        if state.dim() == 2:
            state = state.view(B, 1, -1)
        ids = []
        for k in range(window):
            logits, state = self._step_batch(tok, state, self._b_pos[:B], first = (k == 0))
            nxt = torch.argmax(logits, dim = -1)         # (B,1)
            ids.append(nxt)
            tok = nxt
        return torch.cat(ids, dim = -1).cpu()            # (B, window)

    # ------------------------------------------------------------------ draft graph

    def _ensure_draft_graph(self):
        """Capture one draft step (all 4 blocks + head + argmax) into a CUDA graph.

        Called lazily after the first gather (the graph reads the full-depth gather
        buffers, whose addresses are stable for the run). The step is position- and
        length-agnostic: RoPE reads ``_g_pos`` and attention masks read ``_g_kv*_len``
        from device memory, refreshed per round. ``_g_state`` is aliased: the step reads
        it as input and overwrites it with the post-projected next state, so consecutive
        replays chain without a copy. The embedding stays host-side (CPU table) and is
        staged into ``_g_emb`` between replays.
        """
        if self._draft_graph is not None or self._graph_failed:
            return
        if self._kf_base is None:
            return
        try:
            dev = self.dev
            H = self.backbone_hidden_size
            self._g_emb = torch.empty((1, 1, H), dtype = torch.half, device = dev)
            self._g_state = torch.empty((1, 1, H), dtype = torch.half, device = dev)
            self._g_id = torch.empty((1, 1), dtype = torch.long, device = dev)
            self._g_pos = torch.zeros((1,), dtype = torch.int32, device = dev)
            self._g_kvfull_len = torch.zeros((1,), dtype = torch.int32, device = dev)
            self._g_kvswa_len = torch.zeros((1,), dtype = torch.int32, device = dev)
            # Valid (non-degenerate) inputs for warmup/capture: zero emb/state is safe
            # (RMSNorm eps guards), lengths >= 1 avoid an all-masked softmax.
            self._g_emb.zero_()
            self._g_state.zero_()
            self._g_pos.fill_(1)
            self._g_kvfull_len.fill_(max(1, self.kv_full_len))
            self._g_kvswa_len.fill_(max(1, self.kv_swa_len))

            def step_fn():
                y = torch.cat((self._g_emb, self._g_state), dim = -1)
                x = y @ self.pre_projection.t()
                for layer in self.layers:
                    x = self._block(layer, x, 0, graph = True)
                x = _rmsnorm(x, self.final_norm, self.rms_norm_eps)
                self._g_state.copy_(x @ self.post_projection.t())
                logits = x @ self.head.t()
                self._g_id.copy_(logits.argmax(dim = -1))

            with torch.cuda.device(dev):
                s = torch.cuda.Stream(dev)
                s.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s):
                    for _ in range(3):
                        step_fn()
                torch.cuda.current_stream(dev).wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool = torch.cuda.graph_pool_handle()):
                    step_fn()
            self._draft_graph = g
        except Exception as e:
            self._graph_failed = True
            self._draft_graph = None
            print(f"[mtp-drafter] graph capture failed ({type(e).__name__}: {e}); "
                  f"eager fallback", flush = True)

    def _stage_emb(self, tok_id: int):
        """Host embedding lookup (CPU table) staged into the graph's static input.

        Mirrors _target_embed exactly, including the normalize_embeddings sqrt(hidden)
        scale; copy_ does the H2D move and fp16 cast in one op on the current stream.
        """
        w = self._t_embed_full if self._t_embed_full is not None \
            else self.t_embed.embedding.weight
        e = w[tok_id]                                    # (backbone,) on CPU
        ap = self.t_embed.archparams
        if getattr(ap, "normalize_embeddings", False):
            e = e * (self.backbone_hidden_size ** 0.5)
        self._g_emb[0, 0].copy_(e)

    def _draft_window_graph(self, last_token, last_state, pos, window):
        self._g_pos.fill_(pos)
        self._g_kvfull_len.fill_(self.kv_full_len)
        self._g_kvswa_len.fill_(self.kv_swa_len)
        state = last_state.to(self.dev, non_blocking = True).half()
        if state.dim() == 1:
            state = state.view(1, 1, -1)
        # First step consumes the target's FINAL-NORMED state (eager, outside the graph);
        # later steps consume the aliased post-projected state the graph itself writes.
        self._g_state.copy_(self._target_final_norm(state))
        tok = int(last_token.item())
        ids = []
        for _ in range(window):
            self._stage_emb(tok)
            self._draft_graph.replay()
            tok = int(self._g_id.item())
            ids.append(tok)
        return torch.tensor([ids], dtype = torch.long)   # (1, window) on CPU

    # ------------------------------------------------------------------ helpers

    def footprint(self) -> int:
        n = self.pre_projection.numel() + self.post_projection.numel() + self.head.numel() + self.final_norm.numel()
        for L in self.layers:
            n += (L.input_ln.numel() + L.q_proj.numel() + L.q_norm.numel() + L.o_proj.numel()
                  + L.post_attn_ln.numel() + L.pre_ff_ln.numel() + L.gate.numel() + L.up.numel()
                  + L.down.numel() + L.post_ff_ln.numel() + L.layer_scalar.numel())
        return n * 2
