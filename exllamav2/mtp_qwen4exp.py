from __future__ import annotations

"""
Built-in MTP drafter for Qwen3.8-Flash-Next (Qwen4Exp, P7).

The checkpoint ships a one-layer MTP decoder under `mtp.*` (SGLang
qwen4_exp_mtp.py + HF weight map cross-checked):

  fusion:  en  = rmsnorm_1pw(embed(tok), pre_fc_norm_embedding)      (H)
           hn  = rmsnorm_1pw(h, pre_fc_norm_hidden)                  (4H, FULL width)
           x_i = en + fc_hidden(hn_i)   for each hyper-stream i      (4H)
  layer:   attn hyper-connection -> QSA attention (output gate, q/k norm,
           partial NEOX rope, indexer) -> inject -> MoE hyper-connection ->
           512-expert MoE + sigmoid shared expert -> inject
  output:  mtp.hyper_connection_mixer (grouped norm + softmax mix) -> H -> lm_head
  feedback: the PRE-mixer 4H stream (EAGLE chain).

The target state piped by the MTP driver is the target's pre-mixer hyper-stream
(model.extract_state_indices taps the input of the global mixer module), so no
extra target norm is applied here.

Everything is fp16 torch (mtp.* copies through the converter unquantized).
NO Triton / NO trellis. Correctness first; quantizing/graphing the drafter is a
follow-up.
"""

import math
import torch
import torch.nn.functional as F

from exllamav2.stloader import STFile
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.compat import safe_move_tensor
from exllamav2.qwen4_exp import _rmsnorm_1pw, _grouped_rmsnorm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2


class Qwen4ExpMTPDrafter:

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
        self.hc = self.cfg.hc_count
        self.lowrank = self.cfg.hc_lowrank
        self.nq = self.cfg.num_attention_heads
        self.nkv = self.cfg.num_key_value_heads
        self.hd = self.cfg.head_dim
        self.nkg = self.nq // self.nkv
        self.eps = self.cfg.norm_eps
        self.rope_dim = int(self.hd * self.cfg.partial_rotary_factor)
        self.max_seq_len = max_seq_len or self.cfg.max_seq_len

        self.index_n_heads = self.cfg.indexer_n_heads
        self.index_head_dim = self.cfg.indexer_head_dim
        self.index_budget = self.cfg.indexer_budget
        self.index_compress = self.cfg.indexer_compress_ratio
        self.index_block_topk = self.index_budget // self.index_compress

        self.num_experts = self.cfg.num_experts
        self.top_k = self.cfg.num_experts_per_token
        self.moe_inter = self.cfg.intermediate_size

        self.weights = {}
        self._load_weights()

        # On MP the master's embedding/lm_head are VOCAB-SHARDED; the drafter
        # runs on the master alone (workers idle), so it needs full tables.
        # Load private full copies from the checkpoint when sharded.
        self._full_embed = None
        self._full_head = None
        emb_mod = self.model.modules[0]
        if getattr(emb_mod, "_tp_vocab_shard", None) is not None:
            key = self.cfg.arch.lm_prefix + getattr(self.cfg.arch.lm, "embed_key", "embed_tokens") + ".weight"
            fn = self.cfg.tensor_file_map.get(key)
            if fn is not None:
                st = STFile.open(fn, keymap=self.cfg.arch.keymap)
                self._full_embed = st.get_tensor(key, device=self.device).half()
        head = self.model.modules[self.model.head_layer_idx]
        if getattr(head, "_tp_vocab_parallel", False):
            key = self.cfg.arch.lm_prefix + "lm_head.weight"
            fn = self.cfg.tensor_file_map.get(key)
            if fn is not None:
                st = STFile.open(fn, keymap=self.cfg.arch.keymap)
                self._full_head = st.get_tensor(key, device=self.device).half()

        # Own KV for the single MTP attention layer + raw indexer keys.
        self.k_cache = torch.zeros((1, self.max_seq_len, self.nkv, self.hd), dtype=torch.half, device=self.device)
        self.v_cache = torch.zeros((1, self.max_seq_len, self.nkv, self.hd), dtype=torch.half, device=self.device)
        self.idx_cache = torch.zeros((1, self.max_seq_len, self.index_head_dim), dtype=torch.half, device=self.device)
        self.kv_len = 0

        self._build_rope()

    # ------------------------------------------------------------------ load

    def _load_weights(self):
        cfg = self.cfg
        L = "mtp.layers.0"
        needed = [
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.fc_embedding.weight",
            "mtp.fc_hidden.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            L + ".self_attn.q_proj.weight",
            L + ".self_attn.k_proj.weight",
            L + ".self_attn.v_proj.weight",
            L + ".self_attn.o_proj.weight",
            L + ".self_attn.q_norm.weight",
            L + ".self_attn.k_norm.weight",
            L + ".self_attn.indexer.index_qk_proj.weight",
            L + ".self_attn.indexer.q_layernorm.weight",
            L + ".self_attn.indexer.k_layernorm.weight",
            L + ".mlp.experts.gate_up_proj",
            L + ".mlp.experts.down_proj",
            L + ".mlp.gate.weight",
            L + ".mlp.shared_expert.gate_proj.weight",
            L + ".mlp.shared_expert.up_proj.weight",
            L + ".mlp.shared_expert.down_proj.weight",
            L + ".mlp.shared_expert_gate.weight",
        ]
        for blk in ("attn_hyper_connection", "mlp_hyper_connection"):
            for t in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
                needed.append(f"{L}.{blk}.{t}.weight")
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
        if self._full_embed is not None:
            return F.embedding(ids.to(self._full_embed.device), self._full_embed).to(self.device)
        emb_mod = self.model.modules[0]
        w = emb_mod.get_weight() if hasattr(emb_mod, "get_weight") else emb_mod.weight
        if isinstance(w, tuple):
            w = w[0]
        return F.embedding(ids.to(w.device), w).to(self.device)

    def _head_logits(self, h: torch.Tensor) -> torch.Tensor:
        if self._full_head is not None:
            return F.linear(h.to(self._full_head.device), self._full_head).to(self.device)
        head = self.model.modules[self.model.head_layer_idx]
        if head.q_handle is not None:
            hh = safe_move_tensor(h, head.device())
            return head.forward(hh).to(self.device)
        w = head.linear.weight if head.linear is not None else head.get_weight()
        return F.linear(h.to(w.device), w).to(self.device)

    # ------------------------------------------------------- hyper-connections

    def _hc_mix(self, x: torch.Tensor, prefix: str):
        B, T, _ = x.shape
        rows = B * T
        w = self._w
        normed = _grouped_rmsnorm(x.view(rows, -1), w(prefix + ".hc_norm.weight"), self.hidden_size, self.eps)
        flat = normed.view(rows, -1)
        m = F.silu(F.linear(flat, w(prefix + ".input_mix_weight_down.weight")).float() / self.hc)
        mw = torch.sigmoid(F.linear(m.to(x.dtype), w(prefix + ".input_mix_weight_up.weight")).float())
        mw = mw.view(rows, self.hc, self.hidden_size).to(x.dtype)
        mixed = (mw * normed.view(rows, self.hc, self.hidden_size)).mean(dim=1).view(B, T, self.hidden_size)
        inject = 2.0 * torch.sigmoid(
            F.linear(flat, w(prefix + ".block_inject_weight.weight")).float() / self.hc).to(x.dtype)
        return mixed, inject.view(B, T, self.hc)

    def _hc_inject(self, hyper: torch.Tensor, block_out: torch.Tensor, inject: torch.Tensor):
        B, T, _ = block_out.shape
        y = hyper.view(B, T, self.hc, self.hidden_size) + \
            block_out.view(B, T, 1, self.hidden_size) * inject.view(B, T, self.hc, 1)
        return y.view(B, T, self.hc * self.hidden_size)

    def _mixer(self, x: torch.Tensor):
        B, T, _ = x.shape
        rows = B * T
        w = self._w
        p = "mtp.hyper_connection_mixer"
        normed = _grouped_rmsnorm(x.view(rows, -1), w(p + ".hc_norm.weight"), self.hidden_size, self.eps)
        flat = normed.view(rows, -1)
        m = F.silu(F.linear(flat, w(p + ".input_mix_weight_down.weight")).float() / self.hc)
        mw = torch.sigmoid(F.linear(m.to(x.dtype), w(p + ".input_mix_weight_up.weight")).float())
        mw = mw.view(rows, self.hc, self.hidden_size).to(x.dtype)
        return (mw * normed.view(rows, self.hc, self.hidden_size)).mean(dim=1).view(B, T, self.hidden_size)

    # -------------------------------------------------------------- indexer

    def _indexer_select(self, x: torch.Tensor, start_pos: int):
        """Returns (sel_idx (T,S), sel_valid (T,S)) over the own-KV prefix."""
        B, T, _ = x.shape
        assert B == 1
        w = self._w
        L = "mtp.layers.0.self_attn.indexer"
        qk = F.linear(x, w(L + ".index_qk_proj.weight"))
        q, raw_k = torch.split(qk, [self.index_n_heads * self.index_head_dim, self.index_head_dim], dim=-1)
        q = q.view(B, T, self.index_n_heads, self.index_head_dim)
        raw_k = raw_k.view(B, T, self.index_head_dim)
        q = _rmsnorm_1pw(q, w(L + ".q_layernorm.weight"), self.eps)
        ext_c.rope_(q, self.sin, self.cos, start_pos, self.index_n_heads, self.index_head_dim, none_tensor, True)
        self.idx_cache[0, start_pos:start_pos + T] = raw_k[0]
        kv_len = start_pos + T
        all_keys = self.idx_cache[0, :kv_len]

        C = self.index_compress
        hd = self.index_head_dim
        nblocks = kv_len // C
        S = self.index_budget + C - 1
        dev = self.device
        sel_idx = torch.zeros((T, S), dtype=torch.long, device=dev)
        sel_valid = torch.zeros((T, S), dtype=torch.bool, device=dev)
        qpos = start_pos + torch.arange(T, device=dev)
        k = 0
        if nblocks > 0:
            blocks = all_keys[:nblocks * C].view(nblocks, C, hd)
            pooled = blocks.float().mean(dim=1).to(all_keys.dtype)
            pooled = _rmsnorm_1pw(pooled, w(L + ".k_layernorm.weight"), self.eps)
            starts = torch.arange(nblocks, device=dev) * C
            sin_s = self.sin[0, 0].index_select(0, starts)[None, None]
            cos_s = self.cos[0, 0].index_select(0, starts)[None, None]
            pooled_t = pooled.view(1, nblocks, 1, hd).contiguous()
            ext_c.rope_(pooled_t, sin_s, cos_s, 0, 1, hd, none_tensor, True)
            per_head = torch.einsum("tqh,kh->tkq", q[0].float(), pooled_t.view(nblocks, hd).float())
            scores = F.relu(per_head).sum(dim=-1) / math.sqrt(hd)
            visible = (starts + C - 1).unsqueeze(0) <= qpos.unsqueeze(1)
            scores = scores.masked_fill(~visible, float("-inf"))
            k = min(self.index_block_topk, nblocks)
            top = scores.topk(k, dim=-1).indices
            top_valid = scores.gather(-1, top) > float("-inf")
            tok = (top.unsqueeze(-1) * C + torch.arange(C, device=dev)).view(T, k * C)
            sel_idx[:, :k * C] = tok
            sel_valid[:, :k * C] = top_valid.unsqueeze(-1).expand(T, k, C).reshape(T, k * C)
        lo = ((qpos + 1) // C) * C
        for j in range(C - 1):
            tok = lo + j
            sel_idx[:, k * C + j] = tok.clamp_max(kv_len - 1)
            sel_valid[:, k * C + j] = tok <= qpos
        return sel_idx, sel_valid

    # -------------------------------------------------------------- MTP layer

    def _attention(self, x: torch.Tensor, start_pos: int):
        B, T, _ = x.shape
        w = self._w
        L = "mtp.layers.0.self_attn"
        q = F.linear(x, w(L + ".q_proj.weight"))
        k = F.linear(x, w(L + ".k_proj.weight"))
        v = F.linear(x, w(L + ".v_proj.weight"))
        qg = q.view(B, T, self.nq, 2 * self.hd)
        q = qg[..., :self.hd].contiguous()
        gate = qg[..., self.hd:].contiguous()
        k = k.view(B, T, self.nkv, self.hd)
        v = v.view(B, T, self.nkv, self.hd)
        q = _rmsnorm_1pw(q, w(L + ".q_norm.weight"), self.eps)
        k = _rmsnorm_1pw(k, w(L + ".k_norm.weight"), self.eps)
        ext_c.rope_(q, self.sin, self.cos, start_pos, self.nq, self.hd, none_tensor, True)
        ext_c.rope_(k, self.sin, self.cos, start_pos, self.nkv, self.hd, none_tensor, True)
        self.k_cache[0, start_pos:start_pos + T] = k[0]
        self.v_cache[0, start_pos:start_pos + T] = v[0]
        kv_len = start_pos + T
        self.kv_len = max(self.kv_len, kv_len)

        if kv_len <= self.index_budget + self.index_compress - 1:
            # Dense: every visible token would be selected.
            qf = q.transpose(1, 2).float()
            kf = self.k_cache[:B, :kv_len].transpose(1, 2).float().repeat_interleave(self.nkg, dim=1)
            vf = self.v_cache[:B, :kv_len].transpose(1, 2).float().repeat_interleave(self.nkg, dim=1)
            qpos = torch.arange(start_pos, start_pos + T, device=self.device)[:, None]
            kpos = torch.arange(kv_len, device=self.device)[None, :]
            mask = kpos <= qpos
            attn = torch.matmul(qf, kf.transpose(-1, -2)) * (self.hd ** -0.5)
            attn = attn.masked_fill(~mask[None, None, :, :], float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            ao = torch.matmul(attn, vf).to(x.dtype).transpose(1, 2).contiguous()
        else:
            sel_idx, sel_valid = self._indexer_select(x, start_pos)
            S = sel_idx.shape[-1]
            ao = torch.empty((B, T, self.nq, self.hd), dtype=x.dtype, device=self.device)
            sub = max(1, min(T, 268435456 // max(1, S * self.nkv * self.hd * 4 + self.nq * S * 4)))
            for s in range(0, T, sub):
                e = min(s + sub, T)
                t = e - s
                idx = sel_idx[s:e]
                valid = sel_valid[s:e]
                kg = self.k_cache[0].index_select(0, idx.view(-1)).view(t, S, self.nkv, self.hd)
                vg = self.v_cache[0].index_select(0, idx.view(-1)).view(t, S, self.nkv, self.hd)
                qf = q[0, s:e].float()
                kf = kg.float().repeat_interleave(self.nkg, dim=2)
                vf = vg.float().repeat_interleave(self.nkg, dim=2)
                aw = torch.einsum("tqh,tsqh->tqs", qf, kf) * (self.hd ** -0.5)
                aw = aw.masked_fill(~valid.unsqueeze(1), float("-inf"))
                aw = torch.softmax(aw, dim=-1)
                ao[0, s:e] = torch.einsum("tqs,tsqh->tqh", aw, vf).to(x.dtype)

        ao = ao * torch.sigmoid(gate.float()).to(x.dtype)
        ao = ao.view(B, T, self.nq * self.hd)
        return F.linear(ao, w(L + ".o_proj.weight"))

    def _moe(self, x: torch.Tensor):
        B, T, H = x.shape
        xf = x.view(-1, H)
        w = self._w
        L = "mtp.layers.0.mlp"
        logits = F.linear(xf, w(L + ".gate.weight"))
        probs = F.softmax(logits.float(), dim=-1)
        rw, sel = torch.topk(probs, self.top_k, dim=-1)
        rw = (rw / rw.sum(dim=-1, keepdim=True)).to(x.dtype)
        gu = w(L + ".experts.gate_up_proj")     # (E, 2I, H)
        dn = w(L + ".experts.down_proj")        # (E, H, I)
        out = torch.zeros_like(xf)
        expert_mask = F.one_hot(sel, num_classes=self.num_experts).permute(2, 1, 0)
        I = self.moe_inter
        for e in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[e])
            if top_x.shape[0] == 0:
                continue
            cur = xf[top_x]
            gue = gu[e]
            g = F.linear(cur, gue[:I])
            u = F.linear(cur, gue[I:])
            d = F.linear(F.silu(g) * u, dn[e])
            out.index_add_(0, top_x, (d * rw[top_x, idx, None]).to(out.dtype))
        # Shared expert, sigmoid-gated.
        g = F.linear(xf, w(L + ".shared_expert.gate_proj.weight"))
        u = F.linear(xf, w(L + ".shared_expert.up_proj.weight"))
        d = F.linear(F.silu(g) * u, w(L + ".shared_expert.down_proj.weight"))
        sg = torch.sigmoid(F.linear(xf, w(L + ".shared_expert_gate.weight")).float()).to(x.dtype)
        out = out + d * sg
        return out.view(B, T, H)

    def _mtp_layer(self, x: torch.Tensor, start_pos: int):
        """x: (B, T, 4H) fused hyper-stream. Returns the post-layer 4H stream."""
        L = "mtp.layers.0"
        mixed, inject = self._hc_mix(x, L + ".attn_hyper_connection")
        ao = self._attention(mixed, start_pos)
        x = self._hc_inject(x, ao, inject)
        mixed, inject = self._hc_mix(x, L + ".mlp_hyper_connection")
        mo = self._moe(mixed)
        x = self._hc_inject(x, mo, inject)
        return x

    def _fuse(self, tok: torch.Tensor, h: torch.Tensor):
        """tok (B,T), h (B,T,4H) target/feedback hyper-stream -> fused 4H input.
        The driver pipes fp32 states; the drafter weights are fp16."""
        emb = self._embed(tok)
        en = _rmsnorm_1pw(emb, self._w("mtp.pre_fc_norm_embedding.weight"), self.eps).half()
        hn = _rmsnorm_1pw(h.half(), self._w("mtp.pre_fc_norm_hidden.weight"), self.eps)
        B, T, _ = hn.shape
        streams = hn.view(B, T, self.hc, self.hidden_size)
        enc = F.linear(streams, self._w("mtp.fc_hidden.weight"))
        x = en.unsqueeze(-2) + enc
        return x.view(B, T, self.hc * self.hidden_size)

    def _step(self, tok: torch.Tensor, h: torch.Tensor, position: int):
        x = self._fuse(tok, h)
        hl = self._mtp_layer(x, position)
        out = self._mixer(hl)
        logits = self._head_logits(out)
        return logits, hl

    # ---------------------------------------------------- streaming interface

    def reset(self):
        self.kv_len = 0

    def prefill(self, ids: torch.Tensor, target_states: torch.Tensor):
        """Draft-extend over positions [0, T) with the target's per-position
        pre-mixer hyper-states. ids (1,T), target_states (1,T,4H)."""
        self.reset()
        L = ids.shape[1]
        if L == 0:
            return
        chunk = 512
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            tok = ids[:, s:e].to(self.device)
            h = target_states[:, s:e].to(self.device)
            x = self._fuse(tok, h)
            self._mtp_layer(x, s)

    def gather_target(self, cache):
        pass

    def draft_window(self, base_tok: torch.Tensor, base_state: torch.Tensor, pos: int, num_tokens: int):
        kv_save = self.kv_len
        tok = base_tok.to(self.device)
        h = base_state.to(self.device)
        drafts = []
        p = pos - 1
        for _ in range(num_tokens):
            logits, h = self._step(tok, h, p)
            tok = logits.argmax(dim=-1)
            drafts.append(tok)
            p += 1
        self.kv_len = kv_save
        return torch.cat(drafts, dim=-1), None

    def commit(self, ids: torch.Tensor, target_states: torch.Tensor, pos: int):
        n = ids.shape[1]
        if n == 0:
            return
        tok = ids.to(self.device)
        h = target_states.to(self.device)
        x = self._fuse(tok, h)
        self._mtp_layer(x, pos)

    def rewind(self, pos: int):
        self.kv_len = min(self.kv_len, pos)
