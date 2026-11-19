"""MTP driver for the multiprocess (MP) TP backend.

Mirrors the validated single-process round protocol from
``exllamav2/generator/streaming.py::_gen_single_token_mtp`` / ``_mtp_commit_base``,
adapted to the MP topology:

  * the target is a ``TPProcessGroup`` -- one worker process per GPU, KV sharded;
  * the drafter (``Gemma4AssistantDrafter``) lives in the PARENT on device 0 and
    reads the target's K/V through ``pg.gather_kv_mtp()`` (pipe round-trips);
  * every forward is a collective: the parent fans out to the workers and runs its
    own slice, so the driver only ever calls ``pg.forward`` / ``pg.forward_state``
    (never a bare model.forward, which would deadlock waiting for the other ranks);
  * rewinds are distributed (``pg.mtp_set_seq_len``) so all ranks' caches -- and
    their SWA ring bookkeeping, a deterministic function of ``current_seq_len``
    (BUG 94) -- stay in lockstep.

Sampling support (BUG 104): acceptance still compares the drafter's argmax window
against the target's argmax (the draft is greedy), but every emitted token is SAMPLED
from the target's verified logits with the caller's settings. When a sampled token
deviates from the cached draft/bonus token, the driver rewinds the distributed cache
to that position and re-commits the sampled token (the same exact-resync scheme as the
SP path's BUG 100 fix), so the K/V cache never desyncs from the emitted text and the
output distribution is exactly target-autoregressive under any sampler. Greedy
(settings=None) reduces to pure argmax with zero resyncs, identical to plain greedy.

The first base (logits + pre-final-norm state) is taken for free from the prefill
``forward_state`` call, so steady-state rounds need no re-commit. A re-commit
fallback exists for the case where something moves ``current_seq_len`` out from
under us (token healing / banned-string rewinds).

Usage::

    pg  = TPProcessGroup(MD, [20]*4, MAXSEQ, MAXSEQ, "ExLlamaV2Cache_SWA"); pg.start()
    d   = Gemma4AssistantDrafter(pg, DRAFT_DIR, device=0, num_draft_tokens=6)
    pg.enable_mtp(d, 6)
    drv = MTPMPDriver(pg, d, window=6)
    logits, state = pg.forward_state(ids, last_id_only=True, cpu_logits=True)  # prefill
    drv.begin(ids, logits, state)
    for _ in range(max_new):
        tok = drv.step()
"""
from __future__ import annotations

import torch


class MTPMPDriver:
    """One-token-at-a-time MTP speculation over a ``TPProcessGroup`` target."""

    def __init__(self, pg, drafter, window: int = 6, verbose: bool = False,
                 heal: bool = True):
        self.pg = pg
        self.d = drafter
        self.window = int(window)
        self.verbose = verbose
        # Self-healing (Change95 §12): the prefill state can be corrupted (BUG 90a);
        # re-committing the boundary token -- and doing so again at every round via
        # the bonus commit -- escapes it. SP validation ran with token_healing=True.
        self.heal = heal
        self.tokens: list[int] = []
        self.state = None            # (base_tok (1,1) cpu, base_state (1,1,H) cpu, base_pos)
        self.base_logits = None      # (1,1,vocab) cpu fp32, predicts base_pos+1
        self.pending = None          # (1,R,vocab) cpu fp32 verified rows not yet consumed
        self.pending_tokens = []     # BUG 104: cache token id expected for each pending row
        self.pending_pos = 0         # BUG 104: sequence position of the first pending row
        self.resyncs = 0             # BUG 104: sampling deviations that forced rewind+recommit
        self.total_rounds = 0
        self.total_draft = 0
        self.total_acc = 0
        # per-phase wall clocks (seconds), for gen-cost accounting
        import time as _time
        self._time = _time
        self.t_gather = 0.0    # cross-process KV gather
        self.t_draft = 0.0     # drafter window (GPU, device 0)
        self.t_verify = 0.0    # W-token target verify forward
        self.t_bonus = 0.0     # bonus-commit forward_state (+ rewind sync)
        self.t_rewind = 0.0    # distributed set-seqlen round-trips
        self.t_accept = 0.0    # probabilistic acceptance (dense probs + residual sampling)

    # ------------------------------------------------------------------ begin
    def begin(self, prompt_ids: torch.Tensor, prefill_logits, prefill_state):
        """Seed the base from a prefill ``forward_state`` result.

        ``pg.forward_state(prompt_ids, ...)`` caches the whole prompt and returns the
        last prompt token's logits (predicting position ``n_prompt``) and its
        pre-final-norm state -- exactly the first round's base, at no extra forward.
        """
        self.tokens = prompt_ids[0].tolist()
        n = len(self.tokens)
        # prefill_state is (1, n_prompt, H): the state is captured before the
        # last_id_only narrowing, so take the last prompt token's row -- the same
        # (1, 1, H) the SP path gets by forwarding a single base token.
        state = prefill_state[:, -1:, :]
        self.state = (
            prompt_ids[:, -1:].clone().cpu(),
            state.float().cpu(),
            n - 1,
        )
        self.base_logits = prefill_logits[:, -1:, :].float().cpu()
        self.pending = None
        self.pending_tokens = []
        self.pending_pos = 0
        if self.heal:
            # Self-healing entry: re-commit the last prompt token (idempotent KV
            # rewrite + fresh pre-final-norm state), mirroring SP token_healing.
            self._commit_base()
        if self.verbose:
            print(f"[mtp-mp] begin: n_prompt={n} base_pos={n-1} heal={self.heal} "
                  f"cur={self.pg.cache.current_seq_len}", flush=True)

    # ------------------------------------------------------------------ base
    def _commit_base(self):
        """Re-establish a consistent base (fallback for a moved-out-from-under-us cache).

        Re-commits the token at ``base_pos = current_seq_len - 1``: an idempotent KV
        rewrite that also refreshes the pre-final-norm state and next-position logits.
        """
        base_pos = self.pg.cache.current_seq_len - 1
        if base_pos >= len(self.tokens):
            raise RuntimeError(
                f"[mtp-mp] commit_base: base_pos {base_pos} beyond tracked tokens "
                f"{len(self.tokens)}")
        base_tok = torch.tensor([[self.tokens[base_pos]]], dtype=torch.long)
        self.pg.mtp_set_seq_len(base_pos)
        logits, state = self.pg.forward_state(base_tok, last_id_only=True, cpu_logits=True)
        self.state = (base_tok, state.float().cpu(), self.pg.cache.current_seq_len - 1)
        self.base_logits = logits.float().cpu()
        if self.verbose:
            print(f"[mtp-mp] commit_base pos={self.state[2]} tok={self.tokens[base_pos]}",
                  flush=True)

    # ------------------------------------------------------------------ step
    def step(self, settings=None, tokenizer=None, sequence_ids=None,
             random_value=None, prefix_token=None, ban_tokens=None) -> int:
        """Emit one token from the current round's accepted sequence.

        BUG 105: with ``settings`` (sampling) the round uses probabilistic speculative
        acceptance (see _round), so the emitted tokens are already decided and their KV
        is already in the cache -- step() just hands them out in order. Without
        ``settings``: greedy argmax acceptance. Either way consuming a token never
        advances current_seq_len (the round's bonus commit already moved the cache).
        """
        if not self.pending_tokens:
            self._round(settings, tokenizer, sequence_ids, ban_tokens)
        tok = self.pending_tokens.pop(0)
        self.tokens.append(tok)
        return tok

    @staticmethod
    def _dense_probs(logits_row, settings, tokenizer, sequence_ids, ban_tokens):
        """Full processed target distribution (vocab,) via the sampler's own pipeline.

        BUG 105 perf: ext_c.sample_probs runs the exact softmax/top_k/top_p/top_a/min_p/
        tfs/typical/post-temp pipeline and scatters the result densely in O(V). The old
        approach (sample_basic with return_top_tokens=vocab) hit an O(V^2) extraction loop
        and ran ~0.5 tok/s. Repetition/frequency/presence penalties and bans are applied
        first, exactly as sample() does. Exotic modifiers handled outside sample_basic
        (mirostat/xtc/dry/bias/cfg/temperature_last/smoothing) fall back to the exact
        full-distribution sampler path.
        """
        from exllamav2.ext import exllamav2_ext as ext_c
        vocab = logits_row.shape[-1]
        exotic = (settings.mirostat or settings.xtc_probability > 0.0
                  or settings.dry_multiplier > 0.0 or settings.token_bias is not None
                  or settings.cfg_scale is not None or settings.temperature_last
                  or settings.smoothing_factor > 0.0)
        if exotic:
            from exllamav2.generator.sampler import ExLlamaV2Sampler
            _, kt, kp, _, _ = ExLlamaV2Sampler.sample(
                logits_row, settings, sequence_ids, 0.5, tokenizer,
                None, vocab, blocked_tokens=ban_tokens,
            )
            p = torch.zeros(vocab, dtype=torch.float32)
            p.scatter_(0, kt[0, 0].long(), kp[0, 0].float())
            return p

        lg = logits_row.clone().float()
        if (settings.token_repetition_penalty != 1.0
                or settings.token_frequency_penalty != 0.0
                or settings.token_presence_penalty != 0.0):
            ext_c.apply_rep_penalty(
                sequence_ids, settings.token_repetition_penalty,
                settings.token_repetition_range, settings.token_repetition_decay,
                settings.token_frequency_penalty, settings.token_presence_penalty, lg)
        if ban_tokens:
            lg[:, :, ban_tokens] = -1e30
        vs = tokenizer.get_vocab_size()
        if vs < vocab:
            lg[:, :, vs:] = float("-inf")
        out = torch.zeros((1, vocab), dtype=torch.float32)
        meta_filter = torch.empty(0, dtype=torch.bool, device="meta")
        ext_c.sample_probs(
            lg, settings.temperature, settings.top_k, settings.top_p, settings.top_a,
            settings.min_p, settings.tfs, settings.typical, out, meta_filter,
            1.0, settings.min_temp, settings.max_temp, settings.temp_exponent,
            settings.smoothing_factor)
        return out[0]

    @staticmethod
    def _sample_from(p, rv):
        """Sample a token id from a dense probability vector with uniform rv in [0,1)."""
        cdf = torch.cumsum(p, dim=0)
        return int(torch.searchsorted(cdf, torch.tensor(rv, dtype=torch.float32)).item())

    def _round(self, settings=None, tokenizer=None, sequence_ids=None, ban_tokens=None):
        import random as _random
        # ---- begin a round ----
        if self.state is None or self.pg.cache.current_seq_len != self.state[2] + 1:
            self._commit_base()
        base_tok, base_state, base_pos = self.state
        pos = self.pg.cache.current_seq_len            # == base_pos + 1
        if sequence_ids is None:
            sequence_ids = torch.tensor([self.tokens], dtype=torch.long)
        _t = self._time.perf_counter()

        # 1. gather target K/V for the drafter (once per round)
        self.d.gather_target(self.pg)
        self.t_gather += self._time.perf_counter() - _t
        _t = self._time.perf_counter()

        # 2. draft a greedy window (with per-step logits when sampling, for q)
        if settings is None:
            draft, _ = self.d.draft_window(base_tok, base_state, pos, self.window)   # (1,W) cpu
            draft_logits = None
        else:
            draft, draft_logits = self.d.draft_window(
                base_tok, base_state, pos, self.window, return_logits=True)
            draft_logits = draft_logits.float().cpu()
        self.total_draft += draft.shape[-1]
        self.t_draft += self._time.perf_counter() - _t
        _t = self._time.perf_counter()

        # 3. verify: one target forward over the draft tokens, writing their KV.
        #    vlogits[i] (processing draft[i] at position pos+i) predicts pos+i+1.
        #    NOTE: input_ids cross the worker pipes -- they MUST be CPU tensors
        #    (CUDA-IPC-over-pipe is not available in this environment; the baseline
        #    path already sends CPU ids and _worker_forward moves them per-rank).
        #    graph_verify: the window is a fixed shape (1, W), so every rank captures
        #    it once and replays it per round (Change98) instead of running eager.
        # BUG 105 perf: verify EAGER (graph_verify=False). The graphed verify path feeds
        # _attn_torch the FULL cache depth (160k) with a dynamic mask, and SDPA's hd512
        # kernel scales linearly with kv_len -> ~250 ms/verify. The eager path narrows K/V
        # to the valid prefix and uses FlashInfer prefill for hd512 -> ~tens of ms.
        vlogits = self.pg.forward(draft.cpu(), pos, draft.shape[-1], 1, False,
                                  graph_verify = False)
        self.t_verify += self._time.perf_counter() - _t
        vlogits = vlogits.float().cpu()

        # 4. acceptance.
        draft_flat = draft[0].tolist()
        W = draft.shape[-1]
        if settings is None:
            # Greedy: longest draft prefix the target's argmax agrees with; bonus = argmax.
            m_acc = 0
            v_argmax = vlogits[0].argmax(dim=-1).tolist()
            bonus_pred = int(self.base_logits[0, 0].argmax())
            for k in range(W):
                pred = bonus_pred if k == 0 else v_argmax[k - 1]
                if pred == draft_flat[k]:
                    m_acc = k + 1
                else:
                    break
            bonus_row = self.base_logits if m_acc == 0 else vlogits[:, m_acc - 1:m_acc, :]
            bonus_id = int(bonus_row[0, 0].argmax())
        else:
            # BUG 105: probabilistic speculative acceptance (exact target distribution).
            # The draft is greedy, so q_k is a point mass on draft_flat[k]: accept draft[k]
            # with probability p_target_k(draft[k]); on rejection resample from the residual
            # normalize(max(0, p_target_k - q_k)) and end the round. Unlike exact-resync this
            # costs NO extra forward on rejection -- the round truncates and the bonus commit
            # (below) seeds the next base -- so temp-1 throughput stays high (vLLM/llama.cpp
            # style). p_target_k is the sampler's own processed distribution.
            _t = self._time.perf_counter()
            m_acc = 0
            bonus_id = None
            # Extend the penalty history with each accepted draft so p_target_k matches
            # plain autoregressive sampling (which sees all prior emitted tokens).
            seq = sequence_ids
            for k in range(W):
                # Slice (not integer) index to keep the (1, 1, vocab) shape the sampler
                # requires; vlogits[:, k-1, :] would drop a dim -> (1, vocab) -> crash.
                row = self.base_logits if k == 0 else vlogits[:, k - 1:k, :]
                p = self._dense_probs(row, settings, tokenizer, seq, ban_tokens)
                dk = draft_flat[k]
                if _random.random() < float(p[dk]):
                    m_acc = k + 1
                    seq = torch.cat([seq, torch.tensor([[dk]], dtype=torch.long)], dim=-1)
                    continue
                resid = p.clone()
                resid[dk] = 0.0
                s = float(resid.sum())
                resid = resid / s if s > 1e-12 else p
                bonus_id = self._sample_from(resid, _random.random())
                break
            if bonus_id is None:
                pW = self._dense_probs(vlogits[:, W - 1:W, :], settings, tokenizer,
                                       seq, ban_tokens)
                bonus_id = self._sample_from(pW, _random.random())
            self.t_accept += self._time.perf_counter() - _t
        self.total_acc += m_acc
        self.total_rounds += 1
        if self.verbose:
            print(f"[mtp-mp] round pos={pos} acc={m_acc}/{W}", flush=True)
        elif self.total_rounds % 50 == 0:
            st = self.stats()
            print(f"[mtp-mp] {st['rounds']} rounds acc/round={st['acc_per_round']} "
                  f"emit/round={st['emitted_per_round']} ms/round={st['ms/round']}", flush=True)

        # 5. rewind past the unaccepted tail; commit the bonus token to seed the next base.
        _t = self._time.perf_counter()
        self.pg.mtp_set_seq_len(pos + m_acc)
        self.t_rewind += self._time.perf_counter() - _t
        _t = self._time.perf_counter()
        self.pending_tokens = draft_flat[:m_acc] + [bonus_id]
        self.pending_pos = pos
        base_tok = torch.tensor([[bonus_id]], dtype=torch.long)
        logits, state = self.pg.forward_state(base_tok, last_id_only=True, cpu_logits=True)
        self.t_bonus += self._time.perf_counter() - _t
        self.state = (base_tok, state.float().cpu(), self.pg.cache.current_seq_len - 1)
        self.base_logits = logits.float().cpu()
        if self.verbose:
            print(f"[mtp-mp] bonus={bonus_id} new_base_pos={self.state[2]}", flush=True)

    # ------------------------------------------------------------------ stats
    def stats(self):
        r = max(self.total_rounds, 1)
        s = {
            "rounds": self.total_rounds,
            "drafted": self.total_draft,
            "accepted": self.total_acc,
            "acc_per_round": round(self.total_acc / r, 3),
            "emitted_per_round": round((self.total_acc + self.total_rounds) / r, 3),
        }
        if self.total_rounds:
            s.update({
                "ms/round": {
                    "gather": round(self.t_gather / r * 1000, 2),
                    "draft": round(self.t_draft / r * 1000, 2),
                    "verify": round(self.t_verify / r * 1000, 2),
                    "rewind": round(self.t_rewind / r * 1000, 2),
                    "bonus": round(self.t_bonus / r * 1000, 2),
                    "accept": round(self.t_accept / r * 1000, 2),
                },
                "overhead_ms/token": round(
                    (self.t_gather + self.t_draft + self.t_verify + self.t_rewind
                     + self.t_bonus + self.t_accept) / max(self.total_acc + self.total_rounds, 1) * 1000, 2),
            })
        return s
