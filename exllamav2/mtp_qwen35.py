from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from exllamav2.stloader import STFile
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2


# ---------------------------------------------------------------------------
# Built-in MTP drafter for Qwen3.5 / Qwen3.8 (newplan.md P6, EXPERIMENTAL).
#
# The checkpoint ships its own one-layer MTP decoder under `mtp.*` (no separate
# assistant model). Structure (HF config + exl3/SGLang/vLLM cross-checked):
#
#   mtp.pre_fc_norm_hidden / pre_fc_norm_embedding : RMSNorm (1 + w)
#   mtp.fc                                         : 2H -> H linear
#   mtp.layers.0                                   : full-attention decoder layer
#                                                    (output-gated q, q/k norm,
#                                                     partial NEOX rope, MLP)
#   mtp.norm                                       : final RMSNorm (1 + w)
#
# The drafter borrows the target's embed_tokens and lm_head. Input at each step
# is cat(norm_emb(tok), norm_hidden(h)) -> fc, where h is the target's
# POST-final-norm hidden state (first step) or the drafter's own post-mtp.norm
# output (later steps, EAGLE-style). The MTP layer keeps its OWN K/V cache,
# built by a draft-extend pass over the prompt (prefill) and advanced over
# committed tokens (commit).
#
# Everything here is fp16 torch (the mtp.* tensors are copied through the
# converter unquantized). Correctness first; quantizing the drafter and graphing
# it are follow-ups.
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5 RMSNorm: zero-centered (1 + w), fp32 reduction."""
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    return (xn * (1.0 + weight.float())).to(dtype)


class Qwen35MTPDrafter:

    def __init__(
        self,
        model: ExLlamaV2,
        device: int | str = 0,
        num_draft_tokens: int = 3,
        max_seq_len: int | None = None,
    ):
        self.model = model
        self.cfg = model.config
        self.device = torch.device(device if isinstance(device, str) else f"cuda:{device}")
        self.num_draft_tokens = num_draft_tokens
        self.hidden_size = self.cfg.hidden_size
        self.nq = self.cfg.num_attention_heads
        self.nkv = self.cfg.num_key_value_heads
        self.hd = self.cfg.head_dim
        self.nkg = self.nq // self.nkv
        self.eps = self.cfg.rms_norm_eps
        self.rope_dim = int(self.hd * self.cfg.partial_rotary_factor)
        self.max_seq_len = max_seq_len or self.cfg.max_seq_len

        self.weights = {}
        self._load_weights()

        # Own K/V cache for the single MTP attention layer.
        self.k_cache = torch.zeros((1, self.max_seq_len, self.nkv, self.hd), dtype=torch.half, device=self.device)
        self.v_cache = torch.zeros((1, self.max_seq_len, self.nkv, self.hd), dtype=torch.half, device=self.device)
        self.kv_len = 0

        # RoPE tables: same geometry as the target's full-attention layers.
        self._build_rope()

    # ------------------------------------------------------------------ load

    def _load_weights(self):
        cfg = self.cfg
        needed = [
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.fc.weight",
            "mtp.norm.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
        ]
        files = {}
        for k in needed:
            fn = cfg.tensor_file_map.get(k)
            assert fn is not None, f"Built-in MTP requires {k} in the checkpoint"
            files.setdefault(fn, []).append(k)
        for fn, keys in files.items():
            st = STFile.open(fn, keymap=cfg.arch.keymap)
            for k in keys:
                self.weights[k] = st.get_tensor(k, device=self.device).half()

    def _build_rope(self):
        from exllamav2 import rope
        inv_freq, _ = rope.get_rope_params_default(self.device, self.cfg, self.cfg.rotary_embedding_base, self.rope_dim)
        t = torch.arange(self.max_seq_len, device=self.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.sin = emb.sin()[None, None, :, :].half()
        self.cos = emb.cos()[None, None, :, :].half()

    def _w(self, key):
        return self.weights[key]

    # ------------------------------------------------------------- embeddings

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        emb_mod = self.model.modules[0]
        w = emb_mod.get_weight() if hasattr(emb_mod, "get_weight") else emb_mod.weight
        if isinstance(w, tuple):
            w = w[0]
        return F.embedding(ids.to(w.device), w).to(self.device)

    def _head_logits(self, h: torch.Tensor) -> torch.Tensor:
        head = self.model.modules[self.model.head_layer_idx]
        if head.q_handle is not None:
            # Quantized head: run through the module on its device.
            hh = safe_move_tensor(h, head.device())
            return head.forward(hh).to(self.device)
        w = head.linear.weight if head.linear is not None else head.get_weight()
        return F.linear(h.to(w.device), w).to(self.device)

    # -------------------------------------------------------------- MTP layer

    def _mtp_layer(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        One full-attention decoder layer over (B, T, H), writing/reading own KV at
        contiguous absolute positions [start_pos, start_pos+T). B=1 (the drafter is
        single-sequence). Returns post-layer hidden (B, T, H).
        """
        B, T, _ = x.shape
        assert B == 1, "built-in MTP drafter is single-sequence (experimental)"
        w = self._w
        residual = x
        xn = _rms_norm(x, w("mtp.layers.0.input_layernorm.weight"), self.eps)

        q = F.linear(xn, w("mtp.layers.0.self_attn.q_proj.weight"))
        k = F.linear(xn, w("mtp.layers.0.self_attn.k_proj.weight"))
        v = F.linear(xn, w("mtp.layers.0.self_attn.v_proj.weight"))

        qg = q.view(B, T, self.nq, 2 * self.hd)
        q = qg[..., :self.hd].contiguous()
        gate = qg[..., self.hd:].contiguous()
        k = k.view(B, T, self.nkv, self.hd)
        v = v.view(B, T, self.nkv, self.hd)

        q = _rms_norm(q, w("mtp.layers.0.self_attn.q_norm.weight"), self.eps)
        k = _rms_norm(k, w("mtp.layers.0.self_attn.k_norm.weight"), self.eps)

        # Partial NEOX rope on the first rope_dim dims (kernel pairs i, i+rope_dim/2).
        ext_c.rope_(q, self.sin, self.cos, start_pos, self.nq, self.hd, none_tensor, True)
        ext_c.rope_(k, self.sin, self.cos, start_pos, self.nkv, self.hd, none_tensor, True)

        # Write own KV at the contiguous positions.
        self.k_cache[0, start_pos:start_pos + T] = k[0]
        self.v_cache[0, start_pos:start_pos + T] = v[0]
        kv_len = start_pos + T
        self.kv_len = max(self.kv_len, kv_len)

        # Causal attention over own KV prefix (lower-right mask).
        qf = q.transpose(1, 2).float()
        kf = self.k_cache[:B, :kv_len].transpose(1, 2).float()
        vf = self.v_cache[:B, :kv_len].transpose(1, 2).float()
        kf = kf.repeat_interleave(self.nkg, dim=1)
        vf = vf.repeat_interleave(self.nkg, dim=1)
        qpos = torch.arange(start_pos, start_pos + T, device=self.device)[:, None]
        kpos = torch.arange(kv_len, device=self.device)[None, :]
        mask = kpos <= qpos                          # (T, kv_len)
        attn = torch.matmul(qf, kf.transpose(-1, -2)) * (self.hd ** -0.5)
        attn = attn.masked_fill(~mask[None, None, :, :], float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        ao = torch.matmul(attn, vf).to(x.dtype)      # (B, nq, T, hd)
        ao = ao.transpose(1, 2).contiguous()         # (B, T, nq, hd)
        ao = ao * torch.sigmoid(gate)
        ao = ao.view(B, T, self.nq * self.hd)
        attn_proj = F.linear(ao, w("mtp.layers.0.self_attn.o_proj.weight"))
        x = attn_proj + residual

        residual = x
        xn = _rms_norm(x, w("mtp.layers.0.post_attention_layernorm.weight"), self.eps)
        g = F.linear(xn, w("mtp.layers.0.mlp.gate_proj.weight"))
        u = F.linear(xn, w("mtp.layers.0.mlp.up_proj.weight"))
        d = F.linear(F.silu(g) * u, w("mtp.layers.0.mlp.down_proj.weight"))
        return d + residual

    def _norm_out(self, h: torch.Tensor) -> torch.Tensor:
        return _rms_norm(h, self._w("mtp.norm.weight"), self.eps)

    def _step(self, tok: torch.Tensor, h: torch.Tensor, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One drafter step. tok (B, 1) ids, h (B, 1, H) post-norm hidden, absolute
        position. Returns (logits (B, 1, vocab), next post-norm hidden (B, 1, H)).
        """
        emb = self._embed(tok)
        en = _rms_norm(emb, self._w("mtp.pre_fc_norm_embedding.weight"), self.eps)
        hn = _rms_norm(h, self._w("mtp.pre_fc_norm_hidden.weight"), self.eps)
        x = F.linear(torch.cat([en, hn], dim=-1), self._w("mtp.fc.weight"))
        hl = self._mtp_layer(x, position)
        out = self._norm_out(hl)
        logits = self._head_logits(out)
        return logits, out

    # ---------------------------------------------------- streaming interface

    def reset(self):
        self.kv_len = 0

    def prefill(self, ids: torch.Tensor, target_states: torch.Tensor):
        """
        Draft-extend: build the MTP layer's KV over the given positions using the
        target's per-position post-final-norm states. ids (1, T) and target_states
        (1, T, H) cover the SAME positions [0, T); the caller forwards prompt[:, :-1]
        so the last prompt token's entry is created by the first draft step instead.
        """
        self.reset()
        L = ids.shape[1]
        if L == 0:
            return
        chunk = 1024
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            tok = ids[:, s:e].to(self.device)
            h = target_states[:, s:e].to(self.device)
            emb = self._embed(tok)
            en = _rms_norm(emb, self._w("mtp.pre_fc_norm_embedding.weight"), self.eps)
            hn = _rms_norm(h, self._w("mtp.pre_fc_norm_hidden.weight"), self.eps)
            x = F.linear(torch.cat([en, hn], dim=-1), self._w("mtp.fc.weight"))
            self._mtp_layer(x, s)

    def gather_target(self, cache):
        # Built-in MTP keeps its own KV; nothing to gather from the target.
        pass

    def draft_window(self, base_tok: torch.Tensor, base_state: torch.Tensor, pos: int, num_tokens: int):
        """
        Greedy autoregressive draft of `num_tokens` tokens starting at frontier
        position `pos` (the position the first draft token will occupy). base_tok
        (1, 1) is the last committed token; base_state (1, 1, H) its post-norm
        hidden. Own KV is advanced provisionally then rewound to `pos` so the
        verify/commit path can re-advance it over accepted tokens.
        """
        kv_save = self.kv_len
        tok = base_tok.to(self.device)
        h = base_state.to(self.device)
        drafts = []
        # The first draft token occupies `pos`; its MTP input is the base token at
        # pos-1. Each step processes its input at p and predicts the token at p+1.
        p = pos - 1
        for _ in range(num_tokens):
            logits, h = self._step(tok, h, p)
            tok = logits.argmax(dim=-1)
            drafts.append(tok)
            p += 1
        # Rewind own KV: the committed prefix ends at pos-1.
        self.kv_len = kv_save
        return torch.cat(drafts, dim=-1), None

    def commit(self, ids: torch.Tensor, target_states: torch.Tensor, pos: int):
        """
        Advance own KV over committed tokens (accepted drafts + bonus) using the
        target's post-norm states at those positions. ids (1, n), states (1, n, H),
        pos = absolute position of ids[0].
        """
        n = ids.shape[1]
        if n == 0:
            return
        tok = ids.to(self.device)
        h = target_states.to(self.device)
        emb = self._embed(tok)
        en = _rms_norm(emb, self._w("mtp.pre_fc_norm_embedding.weight"), self.eps)
        hn = _rms_norm(h, self._w("mtp.pre_fc_norm_hidden.weight"), self.eps)
        x = F.linear(torch.cat([en, hn], dim=-1), self._w("mtp.fc.weight"))
        self._mtp_layer(x, pos)

    def rewind(self, pos: int):
        """Truncate own KV to `pos` tokens (after a rejected verify tail)."""
        self.kv_len = min(self.kv_len, pos)
