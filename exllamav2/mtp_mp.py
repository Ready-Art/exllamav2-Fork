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

Bonus-forward elimination (default ON, ``EXL2_MTP_BONUS_ELIM=0`` restores the old
protocol): the separate 1-token bonus ``forward_state`` (~21 ms/round, a full second
target forward) is gone. Instead the bonus token becomes ROW 0 of the NEXT round's
verify window (W+1 rows), so that one forward writes its KV, produces its logits
(acceptance row 0) and its pre-final-norm state (returned per-row via
``return_row_states``). The drafter seeds from ``(bonus_id, last-committed-token
state)`` at query position pos+1 -- the state is one position stale (the EAGLE-style
tradeoff: acceptance may shift, but the emitted distribution stays EXACTLY
target-autoregressive because acceptance is still probabilistic against the target's
verified logits). The first round after begin()/re-commit is mode B too: step()
decides the first bonus from the base logits, queues it for emission, and the next
step()'s verify commits it as row 0 (the exl3 reference always drafts from an
uncommitted bonus). The legacy W-row mode A survives only for DFlash's anchor
geometry, EXL2_MTP_BONUS_ELIM=0, and the moved-cache re-commit fallback.

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

import os
import threading

import torch


class MTPMPDriver:
    """One-token-at-a-time MTP speculation over a ``TPProcessGroup`` target."""

    def __init__(self, pg, drafter, window: int = 6, verbose: bool = False,
                 heal: bool = True):
        self.pg = pg
        self.d = drafter
        self.window = int(window)
        self.verbose = verbose
        # DFlash drafter: same round protocol (bonus-eliminated mode B), but the
        # drafter conditions on target aux states (context K/V) instead of a
        # hidden-state seed + gathered target K/V. The branch points below are
        # guarded by this flag; the MTP path is unchanged when it is False.
        self._dflash = getattr(drafter, "is_dflash", False)
        # Self-healing (Change95 §12): the prefill state can be corrupted (BUG 90a);
        # re-committing the boundary token -- and doing so again at every round via
        # the bonus commit -- escapes it. SP validation ran with token_healing=True.
        # DFlash REQUIRES heal: the mode-A anchor re-feed needs the fresh aux row
        # the heal re-commit produces (and the prefill aux can be corrupted too).
        self.heal = True if self._dflash else heal
        self.tokens: list[int] = []
        self.state = None            # (base_tok (1,1) cpu, base_state (1,1,H) cpu, base_pos)
        self.base_logits = None      # (1,1,vocab) cpu fp32, predicts base_pos+1
        self.pending = None          # (1,R,vocab) cpu fp32 verified rows not yet consumed
        self.pending_tokens = []     # BUG 104: cache token id expected for each pending row
        self.pending_pos = 0         # BUG 104: sequence position of the first pending row
        self.resyncs = 0             # BUG 104: sampling deviations that forced rewind+recommit
        # Bonus-forward elimination: the decided bonus token whose KV is NOT yet in the
        # cache (it becomes row 0 of the next verify). last_state is the pre-final-norm
        # state of the last COMMITTED token (cache.current_seq_len - 1) -- the drafter's
        # (stale) state seed while a bonus is pending. expected_pos is the cache position
        # the next round must start at; any deviation triggers the re-commit fallback.
        # DFlash always runs bonus-eliminated (the anchor IS the pending bonus;
        # the legacy bonus-forward protocol has no aux-state counterpart).
        self._bonus_elim = True if self._dflash \
            else os.environ.get("EXL2_MTP_BONUS_ELIM", "1") != "0"
        self.pending_bonus = None    # int token id decided but not yet KV-committed
        self.last_state = None       # (1,1,H) cpu, state of token at expected_pos-1
        self.expected_pos = 0
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
    def begin(self, prompt_ids: torch.Tensor, prefill_logits, prefill_state,
              aux=None, aux_p0: int = 0):
        """Seed the base from a prefill ``forward_state`` result.

        ``pg.forward_state(prompt_ids, ...)`` caches the whole prompt and returns the
        last prompt token's logits (predicting position ``n_prompt``) and its
        pre-final-norm state -- exactly the first round's base, at no extra forward.

        DFlash: ``aux`` is the prefill's (rows, n_features*hidden) target aux states
        for the tokens prefilled THIS job (the whole prompt on a fresh cache, only
        the new suffix on a prompt-cache hit/restore); ``aux_p0`` is their first
        absolute position. They become the draft KV context; the heal re-commit
        then refreshes the boundary token's row.
        """
        if self._dflash:
            assert aux is not None, "DFlash begin() requires the prefill aux states"
            self.d.feed_context(aux, aux_p0)
        # Sparse draft head: count the prompt's tokens and rebuild the subset at this
        # job boundary (never mid-round -- captured draft graphs stay valid).
        if hasattr(self.d, "observe_tokens"):
            self.d.observe_tokens(prompt_ids)
            self.d.refresh_sparse_head()
        # Built-in MTP (own draft KV): seed the drafter from the prefill row states.
        if hasattr(self.d, "prefill_target"):
            self.d.prefill_target(prompt_ids, prefill_state.float().cpu())
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
        self.pending_bonus = None
        self.last_state = self.state[1]
        self.expected_pos = self.pg.cache.current_seq_len
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
        if self._dflash:
            # The re-commit also refreshes the boundary token's draft context K/V
            # (mtp_set_seq_len trimmed the draft length to base_pos; this feed
            # overwrites that row and restores ctx_len = base_pos + 1).
            logits, state, aux = self.pg.forward_state(
                base_tok, last_id_only=True, cpu_logits=True, return_row_aux=True)
            self.d.feed_context(aux, base_pos)
        else:
            logits, state = self.pg.forward_state(base_tok, last_id_only=True, cpu_logits=True)
        if hasattr(self.d, "commit_bonus"):
            self.d.commit_bonus(base_tok, state.float().cpu(),
                                self.pg.cache.current_seq_len - 1)
        self.state = (base_tok, state.float().cpu(), self.pg.cache.current_seq_len - 1)
        self.base_logits = logits.float().cpu()
        # Re-commit restores an ALIGNED base: no pending bonus, fresh state/logits.
        self.pending_bonus = None
        self.last_state = self.state[1]
        self.expected_pos = self.pg.cache.current_seq_len
        if self.verbose:
            print(f"[mtp-mp] commit_base pos={self.state[2]} tok={self.tokens[base_pos]}",
                  flush=True)

    # ------------------------------------------------------------------ step
    def step(self, settings=None, tokenizer=None, sequence_ids=None,
             random_value=None, prefix_token=None, ban_tokens=None) -> int:
        """Emit one token from the current round's accepted sequence.

        BUG 105: with ``settings`` (sampling) the round uses probabilistic speculative
        acceptance (see _round), so the emitted tokens are already decided -- step() just
        hands them out in order. Without ``settings``: greedy argmax acceptance. Consuming
        a token never advances current_seq_len: the accepted drafts' KV is already in the
        cache, and the round's bonus token is KV-committed by the NEXT round's verify
        (row 0) under bonus-forward elimination (or by the legacy bonus forward).
        """
        if not self.pending_tokens:
            # MTP (bonus-elim): decide the first bonus from the base logits so EVERY
            # round is mode B -- the exl3 reference always drafts from an uncommitted
            # bonus with the query at its position. (DFlash keeps mode A: its block
            # anchor must be a context-committed token.) Same distribution as the old
            # mode-A first round: the bonus is drawn from the target's base logits.
            if (not self._dflash and self._bonus_elim
                    and self.pending_bonus is None and self.base_logits is not None):
                if settings is None:
                    self.pending_bonus = int(self.base_logits[0, 0].argmax())
                else:
                    import random as _random
                    if sequence_ids is None:
                        sequence_ids = torch.tensor([self.tokens], dtype=torch.long)
                    p = self._dense_probs(self.base_logits, settings, tokenizer,
                                          sequence_ids, ban_tokens)
                    self.pending_bonus = self._sample_from(p, _random.random())
                # This bonus has NOT been emitted yet -- steady-state bonuses are the
                # last token of the PREVIOUS round's queue, but the first one comes
                # straight from the base logits. Queue it for emission BEFORE the
                # round runs (the round executes on the next step()): verify row 0
                # commits its KV either way, so dropping it from the stream loses the
                # reply's first token (the thought-channel opener) and desyncs the
                # emitted text / token history from the cache.
                self.pending_tokens = [self.pending_bonus]
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
        # The cache must sit exactly where the previous round left it: committed tokens
        # only. A pending bonus (if any) is NOT in the cache yet -- the next verify writes
        # it as row 0. Any external move (token healing / banned-string rewind) falls back
        # to the aligned re-commit protocol.
        if self.state is None or self.pg.cache.current_seq_len != self.expected_pos:
            self.pending_bonus = None
            self._commit_base()
        pos = self.pg.cache.current_seq_len
        mode_b = self._bonus_elim and self.pending_bonus is not None
        if sequence_ids is None:
            sequence_ids = torch.tensor([self.tokens], dtype=torch.long)
        _t = self._time.perf_counter()

        # 1. gather target K/V for the drafter (once per round)
        self.d.gather_target(self.pg)
        self.t_gather += self._time.perf_counter() - _t
        _t = self._time.perf_counter()

        # 2. draft a greedy window.
        # QUERY POSITION RULE (exl3 reference, generator.py: positions = kv_seqlens):
        # RoPE attention scores depend on RELATIVE query-key distances. exl3 inputs the
        # UNCOMMITTED bonus at pos over KV [0, pos-1] with the query at pos -> distances
        # [1, pos]. Mode B matches that exactly (query at pos). The old pos+1 shifted
        # every distance by +1: mild local damage, catastrophic long-context acceptance
        # (the 82k collapse -- long-range key matching needs exact relative positions).
        # Mode A inputs the last COMMITTED token at pos-1 (its key IS in the KV):
        #   MTP legacy: query at pos keeps distances [1, pos] (the +1 compensates the
        #     committed input key, exactly like the SP streaming path).
        #   DFlash: the anchor self-attends by design -> query at pos-1 (distance 0).
        # MTP (bonus-elim) no longer enters mode A: step() decides the first bonus from
        # the base logits so every round is mode B, like exl3.
        if mode_b:
            base_tok = torch.tensor([[self.pending_bonus]], dtype=torch.long)
            dpos = pos
            draft, _ = self.d.draft_window(base_tok, self.last_state, dpos, self.window)
            draft = draft.cpu()                                    # cat with CPU bonus; pipe needs CPU
            verify_ids = torch.cat([base_tok, draft], dim=-1)      # (1, W+1)
        else:
            base_tok, base_state, base_pos = self.state
            dpos = pos - 1 if self._dflash else pos
            draft, _ = self.d.draft_window(base_tok, base_state, dpos, self.window)
            draft = draft.cpu()
            verify_ids = draft                                     # (1, W)
        self.total_draft += draft.shape[-1]
        self.t_draft += self._time.perf_counter() - _t
        _t = self._time.perf_counter()

        # 3. verify: ONE target forward over the window, writing its KV.
        #    Mode A: vlogits[i] (draft[i] at pos+i) predicts pos+i+1; draft[0]'s predictor
        #            is base_logits.
        #    Mode B: row 0 is the pending bonus at pos -> vlogits[0] predicts pos+1 (the
        #            bonus forward's logits, for free); vlogits[k] (k>=1) predicts the
        #            token after draft[k-1], i.e. draft[k]'s position. Acceptance row for
        #            draft[k] is therefore vlogits[k] directly, and the bonus row is
        #            vlogits[m_acc] -- always in range (W+1 rows).
        #    return_row_states: per-row pre-final-norm states (graph state mode 2), so the
        #    next round's drafter can seed from the last accepted draft's state.
        #    NOTE: input_ids cross the worker pipes -- they MUST be CPU tensors.
        if self._dflash:
            # DFlash verify: the per-row AUX states (6-layer residual taps) replace
            # the final-norm states -- they become the next round's draft context.
            # Already CPU (the pipe boundary copies off the graph's static buffer).
            vlogits, vaux = self.pg.forward(verify_ids.cpu(), pos, verify_ids.shape[-1],
                                            1, False, graph_verify = True,
                                            return_row_aux = True)
            vstates = None
        elif self._bonus_elim:
            vlogits, vstates = self.pg.forward(verify_ids.cpu(), pos, verify_ids.shape[-1],
                                               1, False, graph_verify = True,
                                               return_row_states = True)
            # Static graph buffer: consume (copy off device) before the next replay.
            vstates = vstates.cpu()
        else:
            vlogits = self.pg.forward(verify_ids.cpu(), pos, verify_ids.shape[-1], 1, False,
                                      graph_verify = True)
            vstates = None
        self.t_verify += self._time.perf_counter() - _t
        vlogits = vlogits.float().cpu()

        # 4. acceptance.
        draft_flat = draft[0].tolist()
        W = draft.shape[-1]
        if settings is None:
            # Greedy: longest draft prefix the target's argmax agrees with; bonus = argmax.
            m_acc = 0
            v_argmax = vlogits[0].argmax(dim=-1).tolist()
            if mode_b:
                for k in range(W):
                    if v_argmax[k] == draft_flat[k]:
                        m_acc = k + 1
                    else:
                        break
                bonus_row = vlogits[:, m_acc:m_acc + 1, :]
            else:
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
            # normalize(max(0, p_target_k - q_k)) and end the round. NO extra forward on
            # rejection. The emitted distribution is exactly target-autoregressive even
            # though mode B's drafts come from a stale state -- draft quality only affects
            # acceptance RATE, never correctness.
            _t = self._time.perf_counter()
            m_acc = 0
            bonus_id = None
            # Extend the penalty history with each accepted draft so p_target_k matches
            # plain autoregressive sampling (which sees all prior emitted tokens).
            seq = sequence_ids
            for k in range(W):
                # Slice (not integer) index to keep the (1, 1, vocab) shape the sampler
                # requires; vlogits[:, k-1, :] would drop a dim -> (1, vocab) -> crash.
                if mode_b:
                    row = vlogits[:, k:k + 1, :]
                else:
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
                last_row = vlogits[:, W:W + 1, :] if mode_b else vlogits[:, W - 1:W, :]
                pW = self._dense_probs(last_row, settings, tokenizer, seq, ban_tokens)
                bonus_id = self._sample_from(pW, _random.random())
            self.t_accept += self._time.perf_counter() - _t
        self.total_acc += m_acc
        self.total_rounds += 1
        if self.verbose:
            print(f"[mtp-mp] round pos={pos} acc={m_acc}/{W} mode={'B' if mode_b else 'A'}",
                  flush=True)
        elif self.total_rounds % 50 == 0:
            st = self.stats()
            print(f"[mtp-mp] {st['rounds']} rounds acc/round={st['acc_per_round']} "
                  f"emit/round={st['emitted_per_round']} ms/round={st['ms/round']}", flush=True)

        # 5. rewind past the unaccepted tail; seed the next round.
        # Mode B: row 0 is the bonus at pos, so accepted draft d_k sits at pos+1+k and
        # m_acc accepted drafts commit through pos+m_acc -> rewind to pos+m_acc+1.
        # Mode A: drafts start at pos -> rewind to pos+m_acc (original protocol).
        _t = self._time.perf_counter()
        rewind_to = pos + m_acc + 1 if mode_b else pos + m_acc
        self.pg.mtp_set_seq_len(rewind_to)
        # Built-in MTP: advance the drafter's own KV over the accepted rows using
        # the target's verified per-row states (mode B row 0 is the pending bonus).
        if hasattr(self.d, "commit_accepted") and vstates is not None:
            n_acc_rows = m_acc + 1 if mode_b else m_acc
            if n_acc_rows > 0:
                self.d.commit_accepted(verify_ids[:, :n_acc_rows].cpu(),
                                       vstates[:, :n_acc_rows].float().cpu(), pos)
        self.t_rewind += self._time.perf_counter() - _t
        _t = self._time.perf_counter()
        self.pending_tokens = draft_flat[:m_acc] + [bonus_id]
        # Sparse draft head: count the round's committed tokens (accepted drafts + bonus).
        if hasattr(self.d, "observe_tokens"):
            self.d.observe_tokens(torch.tensor([self.pending_tokens], dtype=torch.long))
        # Mode B's first pending token (draft[0]) sits at pos+1 -- the bonus at pos was
        # already emitted at the end of the previous round.
        self.pending_pos = pos + 1 if mode_b else pos
        if self._bonus_elim:
            # NO bonus forward. The bonus token's KV/logits/state come from row 0 of the
            # NEXT verify. Track the state of the last COMMITTED token (rewind_to-1):
            #   mode B: row m_acc holds it for EVERY m_acc -- m_acc>0 -> d_{m_acc-1} at
            #           pos+m_acc (row i = d_{i-1} at pos+i); m_acc=0 -> the bonus itself
            #           (row 0 at pos).
            #   mode A: m_acc>0 -> row m_acc-1 (d_{m_acc-1} at pos+m_acc-1); m_acc=0 ->
            #           the aligned base state (token at pos-1, not in this window).
            if self._dflash:
                # Feed the accepted rows' aux states into the draft KV context:
                # mode B = [bonus, accepted drafts] at pos; mode A = the anchor's
                # stored row + accepted drafts at pos-1. Overwrites the block's
                # noise K/V; ctx_len lands exactly at rewind_to. No state seed.
                self.d.feed_verify(vaux, m_acc, pos, mode_b)
                self.last_state = None
                committed_tok = draft_flat[m_acc - 1] if m_acc > 0 else \
                    (int(self.pending_bonus) if mode_b else int(self.tokens[pos - 1]))
            elif mode_b:
                self.last_state = vstates[:, m_acc:m_acc + 1, :].float().cpu()
                committed_tok = draft_flat[m_acc - 1] if m_acc > 0 else int(self.pending_bonus)
            else:
                if m_acc > 0:
                    self.last_state = vstates[:, m_acc - 1:m_acc, :].float().cpu()
                else:
                    self.last_state = self.state[1]
                committed_tok = draft_flat[m_acc - 1] if m_acc > 0 else int(self.tokens[pos - 1])
            self.pending_bonus = bonus_id
            self.expected_pos = rewind_to
            self.state = (torch.tensor([[committed_tok]], dtype=torch.long),
                          self.last_state, self.expected_pos - 1)
        else:
            # Legacy protocol: commit the bonus with a dedicated 1-token forward_state.
            base_tok = torch.tensor([[bonus_id]], dtype=torch.long)
            logits, state = self.pg.forward_state(base_tok, last_id_only=True, cpu_logits=True)
            if hasattr(self.d, "commit_bonus"):
                self.d.commit_bonus(base_tok, state.float().cpu(),
                                    self.pg.cache.current_seq_len - 1)
            self.state = (base_tok, state.float().cpu(), self.pg.cache.current_seq_len - 1)
            self.base_logits = logits.float().cpu()
            self.pending_bonus = None
            self.last_state = self.state[1]
            self.expected_pos = self.pg.cache.current_seq_len
        self.t_bonus += self._time.perf_counter() - _t
        if self.verbose:
            print(f"[mtp-mp] bonus={bonus_id} committed_pos={self.expected_pos}", flush=True)

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


_accept_pool = None


def _get_accept_pool():
    """Thread pool for per-row acceptance precompute. The sampler ext kernels
    (apply_rep_penalty / sample_probs) wrap their compute in Py_BEGIN_ALLOW_THREADS
    and torch CPU ops release the GIL during execution, so independent rows truly
    parallelize -- same kernels, same math, same random-draw ORDER (draws happen in
    the serial per-row walk, not in the workers), hence the emitted distribution is
    unchanged. At B=4 the serial accept was ~9 ms/round; threaded it is ~1 ms."""
    global _accept_pool
    if _accept_pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _accept_pool = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 4), thread_name_prefix="mtp-accept")
    return _accept_pool


_accept_tls = threading.local()


def _accept_precompute(vl, settings, tokenizer, seq, ban, d, W):
    """Dense target probs + CDFs for one row's whole verify window, for the
    penalty-free fast path (probs at row k then do not depend on which earlier
    drafts were accepted, so all W+1 rows can be computed up front -- and in
    parallel across slots). Residual CDFs remove the draft's point mass exactly
    as the serial walk's normalize(max(0, p - q_k)) does.

    PERF: every vocab-sized buffer is THREAD-LOCAL and reused, and every torch op
    writes via out=. Fresh 1-4 MB allocations per op (stack/cumsum/clone/where)
    page-fault-serialize the worker threads through the kernel mmap path --
    measured ~15 ms/round at B=4 with allocations vs ~3 ms with reuse. The
    non-exotic sampler pipeline is inlined here (eligibility guarantees no
    penalties/exotic modifiers) to keep the ext call's output buffer reusable;
    the serial fallback keeps using MTPMPDriver._dense_probs verbatim."""
    # torch defaults to one intra-op thread per core (32 here); 4 concurrent workers
    # would oversubscribe 128 OMP threads and spin-contend. 4 threads/worker keeps the
    # vocab-sized cumsum/div/where glue fast while total threads (4 workers x 4) stay
    # within the core count. Process-global setting; the scheduler's own CPU torch ops
    # are tiny and unaffected.
    torch.set_num_threads(4)
    from exllamav2.ext import exllamav2_ext as ext_c
    V = vl.shape[-1]
    buf = getattr(_accept_tls, "buf", None)
    if buf is None or buf["P"].shape[0] != W + 1 or buf["P"].shape[1] != V:
        buf = _accept_tls.buf = {
            "lg": torch.empty(V, dtype=torch.float32),
            "out": torch.empty(V, dtype=torch.float32),
            "P": torch.empty(W + 1, V, dtype=torch.float32),
            "C": torch.empty(W + 1, V, dtype=torch.float32),
            "R": torch.empty(W + 1, V, dtype=torch.float32),
            "RC": torch.empty(W + 1, V, dtype=torch.float32),
            "idx": torch.arange(W, dtype=torch.long),
            "meta": torch.empty(0, dtype=torch.bool, device="meta"),
        }
    lg, out, P, C, R, RC = (buf["lg"], buf["out"], buf["P"], buf["C"],
                            buf["R"], buf["RC"])
    vs = tokenizer.get_vocab_size()
    for k in range(W + 1):
        lg.copy_(vl[k])
        if ban:
            lg[ban] = -1e30
        if vs < V:
            lg[vs:] = float("-inf")
        ext_c.sample_probs(
            lg.view(1, 1, V), settings.temperature, settings.top_k, settings.top_p,
            settings.top_a, settings.min_p, settings.tfs, settings.typical,
            out.view(1, V), buf["meta"],
            1.0, settings.min_temp, settings.max_temp, settings.temp_exponent,
            settings.smoothing_factor)
        P[k].copy_(out)
    torch.cumsum(P, dim=-1, out=C)
    R.copy_(P)
    # Only the W draft rows carry a point mass to remove; row W is the bonus row
    # (sampled from C[W], its residual is never used).
    R[buf["idx"], torch.tensor(d, dtype=torch.long)] = 0.0
    s = R.sum(dim=-1, keepdim=True)
    torch.div(R, s, out=R)
    torch.where(s > 1e-12, R, P, out=R)
    torch.cumsum(R, dim=-1, out=RC)
    return P, C, RC


class MTPMPBatchDriver:
    """
    Multi-batch MTP driver for the MP slot scheduler: one independent mode-B state
    machine per cache row, advanced by shared batched rounds.

    Every round is the scalar driver's mode B, generalized to B rows: the verify window
    is [pending_bonus, draft_0 .. draft_{W-1}] (B, W+1) at each row's own position; row
    0's logits predict draft_0, row k's predict draft_k, row m_acc's give the next bonus.
    Acceptance, rewind and the bonus-elimination invariant are per-row; the KV rewrite,
    the drafter and the acceptance bookkeeping never mix rows. The first round after
    begin_row uses the bonus sampled from the prefill (heal) logits, so there is no
    separate mode-A shape -- one (B, W+1) graph per batch size serves every round.

    The target KV bookkeeping (cache.seq_lens / window_begs) is owned by this driver via
    the distributed row setters; the batched forward itself never advances it.
    """

    def __init__(self, pg, drafter, window: int = 6, heal: bool = True):
        self.pg = pg
        self.drafter = drafter
        self.W = window
        self.heal = heal
        self.rows = {}
        self.mtp_rounds = 0
        self.mtp_acc = 0
        self.mtp_drafts = 0
        self.mtp_emitted = 0
        import time as _time
        self._time = _time
        self.t_round = 0.0
        self.t_gather = 0.0
        self.t_draft = 0.0
        self.t_verify = 0.0
        self.t_accept = 0.0
        self.t_pre = 0.0
        self.t_rewind = 0.0
        # Draft-quality instrument: fraction of draft tokens equal to the target's
        # argmax on the same verify rows (separates draft degradation from accept math).
        self.draft_agree_sum = 0
        self.draft_agree_n = 0
        # Per-batch-index and per-draft-position split (localizes multi-row bugs).
        self.agree_bi = {}
        self.agree_k = [0] * 16
        self.agree_kn = [0] * 16

    # -- per-row lifecycle -------------------------------------------------

    def begin_row(self, r: int, prompt_ids: torch.Tensor, logits, state,
                  settings=None, tokenizer=None, seq_ids=None, ban_tokens=None):
        """Arm row r after its prefill. ``logits``/``state`` are the prefill's last-row
        outputs (None forces the heal recommit). The first bonus is decided here and
        queued for emission; its KV is committed by the first verify round (mode B)."""
        # Sparse draft head: count the prompt's tokens and rebuild the subset at this
        # job boundary (never mid-round). A changed subset invalidates the captured
        # batch draft graphs; they recapture lazily on the next round. Other active
        # rows keep decoding -- their pending drafts are just tokens, acceptance is
        # always against the target's full-head logits.
        if hasattr(self.drafter, "observe_tokens"):
            self.drafter.observe_tokens(prompt_ids)
            self.drafter.refresh_sparse_head()
        n_prompt = prompt_ids.shape[-1]
        tokens = prompt_ids[0].tolist()
        if logits is not None:
            logits = logits[:, -1:, :].float().cpu()
        if state is not None:
            # Normalize to CPU float: rounds mix rows at different lifecycle stages
            # (first-round GPU states vs later-round CPU copies) into one batch tensor.
            state = state[:, -1:, :].float().cpu()
        last_tok = int(tokens[-1])
        self.pg.mtp_set_seq_len_row(r, n_prompt)
        if self.heal or logits is None or state is None:
            # Heal: recommit the last prompt token through the decode path (idempotent
            # KV rewrite) for a fresh, decode-aligned state + logits (BUG 90a/93).
            self.pg.mtp_set_seq_len_row(r, n_prompt - 1)
            logits, state = self.pg.forward_state_row(
                r, prompt_ids[:, -1:], last_id_only=True)
            # Normalize the heal outputs too: rounds cat states across rows at
            # different lifecycle stages, so every stored state must be CPU float.
            logits = logits.float().cpu()
            state = state.float().cpu()
        bonus = self._first_bonus(logits, settings, tokenizer, seq_ids, ban_tokens)
        self.rows[r] = {
            "tokens": tokens,
            "pending": [bonus],
            "pending_bonus": bonus,
            "last_state": state,
            "expected_pos": n_prompt,
            "base_logits": logits,
            "base_tok": last_tok,
        }

    def finish_row(self, r: int):
        self.rows.pop(r, None)

    def _first_bonus(self, logits, settings, tokenizer, seq_ids, ban_tokens=None):
        import random as _random
        # settings is None -> greedy, exactly the scalar driver's contract. Any real
        # settings object goes through the full sampler pipeline (_dense_probs handles
        # temp/top_k/exotic modifiers/bans), so low-temperature + bias/DRY/etc. requests
        # keep their exact semantics.
        if settings is None:
            return int(logits.argmax(dim=-1).item())
        if seq_ids is None:
            seq_ids = torch.zeros((1, 0), dtype=torch.long)
        probs = MTPMPDriver._dense_probs(logits, settings, tokenizer, seq_ids, ban_tokens)
        return MTPMPDriver._sample_from(probs, _random.random())

    def _recommit_row(self, r, p, row, settings, tokenizer, seq_ids):
        """The row's cache moved outside the driver (external trim): rebuild its base
        with one solo decode-shaped forward, then re-arm a pending bonus (mode B)."""
        assert 1 <= p <= len(row["tokens"]), "batch MTP recommit: invalid position"
        base = row["tokens"][p - 1]
        self.pg.mtp_set_seq_len_row(r, p - 1)
        bl, st = self.pg.forward_state_row(
            r, torch.tensor([[base]], dtype=torch.long), last_id_only=True)
        bl = bl.float().cpu()
        st = st.float().cpu()
        bonus = self._first_bonus(bl, settings, tokenizer, seq_ids, None)
        row["tokens"] = row["tokens"][:p]
        row["last_state"] = st
        row["base_logits"] = bl
        row["base_tok"] = base
        row["expected_pos"] = p
        row["pending"] = [bonus]
        row["pending_bonus"] = bonus

    # -- round orchestration -----------------------------------------------

    def step_batch(self, active, settings_map, tokenizer, seq_ids_map, ban_map=None):
        """One scheduler step: run a shared verify round for the rows whose pending
        queue ran dry, then DRAIN every active row's decided tokens. Returns
        {row: [tokens]} (a burst of m_acc+1 per refreshed row).

        Draining fully (instead of popping one token per step) keeps every row's
        queue empty at round time, so EVERY round runs at the full active batch
        size. The old one-token pop desynchronized rows under staggered admission
        and variable acceptance: queues drifted apart, rounds ran at B~1 in
        rotation, and the batched verify almost never engaged (measured 83.7 t/s
        aggregate at 4 streams vs ~68 solo). Bursts of 3-4 chunks per round are
        imperceptible to streaming clients, and the penalty history is NOW aligned
        at round time (seq_ids caught up through the full drain -- strictly better
        than the old pending-queue lag). Stop/max-token truncation discards the
        burst tail exactly as the old mid-round stops did (overshoot <= W)."""
        need = [r for r in active if not self.rows[r]["pending"]]
        if need:
            self._round(need, settings_map, tokenizer, seq_ids_map, ban_map)
        out = {}
        for r in active:
            row = self.rows[r]
            if row["pending"]:
                ts = row["pending"]
                row["pending"] = []
                row["tokens"].extend(ts)
                out[r] = ts
        return out

    def _round(self, rows, settings_map, tokenizer, seq_ids_map, ban_map):
        pg = self.pg
        cache = pg.cache
        W = self.W
        _t0 = self._time.perf_counter()

        for r in rows:
            row = self.rows[r]
            if cache.seq_lens[r] != row["expected_pos"]:
                self._recommit_row(r, cache.seq_lens[r], row,
                                   settings_map.get(r), tokenizer, seq_ids_map.get(r))

        pos = [cache.seq_lens[r] for r in rows]

        # Batched gather + draft. Mode-B seed per row: (pending bonus, state of the last
        # COMMITTED token), query at the bonus's OWN position pos over KV ending at
        # pos-1 (exl3: positions = kv_seqlens) -- the scalar driver's geometry.
        _t = self._time.perf_counter()
        pg.gather_kv_mtp_batch(rows)
        self.drafter.gather_target_batch(rows)
        self.t_gather += self._time.perf_counter() - _t
        base_tok = torch.tensor([[self.rows[r]["pending_bonus"]] for r in rows],
                                dtype=torch.long)
        base_state = torch.cat([self.rows[r]["last_state"] for r in rows], dim=0)
        _t = self._time.perf_counter()
        draft = self.drafter.draft_window_batch(
            base_tok, base_state, list(pos), W)                 # (B, W) CPU
        self.t_draft += self._time.perf_counter() - _t

        # Batched verify: [bonus, drafts] per row. Row 0 commits the bonus KV; rows
        # 1..W write the drafts. vlogits[k] predicts the token after verify row k.
        vids = torch.cat([base_tok, draft], dim=-1)             # (B, W+1)
        _t = self._time.perf_counter()
        vlogits, vstates = pg.forward_batch(
            vids, pos, rows, last_id_only=False, graph_verify=True,
            return_row_states=True)
        vlogits = vlogits.float().cpu()
        self.t_verify += self._time.perf_counter() - _t
        _va = vlogits.argmax(dim=-1)
        _eq = (draft == _va[:, :W])
        self.draft_agree_sum += int(_eq.sum())
        self.draft_agree_n += draft.numel()
        # XCHECK: run the SCALAR draft path on row 0's shadow (3-D views of the same
        # gathered K/V, scalar RoPE position) and compare against the batch draft and
        # the target argmax. scalar==batch!=target -> verify logits wrong at B>1;
        # scalar==target!=batch -> batch draft compute wrong; all differ -> gather.
        if (os.environ.get("EXL2_BATCH_DRAFT_XCHECK", "0") == "1"
                and self.mtp_rounds % 25 == 0
                and not (pg._mtp_kv or {}).get("i8p")):
            d_ = self.drafter
            st_ = pg._mtp_kv
            r0 = rows[0]
            ring_ = st_["ks"].shape[1]
            sv = (d_.kv_full, d_.kv_full_len, d_.kv_swa, d_.kv_swa_len,
                  d_.kv_swa_window_beg)
            d_.kv_full = (st_["kf"][r0], st_["vf"][r0])
            d_.kv_full_len = cache.seq_lens[r0]
            d_.kv_swa = (st_["ks"][r0], st_["vs"][r0])
            d_.kv_swa_len = min(cache.seq_lens[r0] - cache.window_begs[r0], ring_)
            d_.kv_swa_window_beg = cache.window_begs[r0]
            try:
                ids_s, _ = d_.draft_window(base_tok[0:1], base_state[0:1], pos[0], W)
                print(f"[xcheck] row {r0} scalar={ids_s[0].tolist()} "
                      f"batch={draft[0].tolist()} target={_va[0, :W].tolist()}",
                      flush=True)
            finally:
                (d_.kv_full, d_.kv_full_len, d_.kv_swa, d_.kv_swa_len,
                 d_.kv_swa_window_beg) = sv
            # Solo EAGER verify of row 0 on the same vids/position (the verified B=1
            # path) vs the batched graphed verify: argmax agreement + max logit diff
            # separates fp16 near-tie flips (small diff) from a plumbing bug (large).
            # The KV rewrite is idempotent; restore the row position afterwards.
            vl_solo = pg.forward_row(r0, vids[0:1], last_id_only=False)
            pg.mtp_set_seq_len_row(r0, pos[0])
            vl_solo = vl_solo.float().cpu()[0]
            _vb = vlogits[0]
            _diff = (vl_solo - _vb).abs().max().item()
            _am = int((vl_solo.argmax(dim=-1) == _vb.argmax(dim=-1)).sum())
            print(f"[xcheck-verify] row {r0} solo-vs-batch argmax {_am}/{W + 1} "
                  f"maxlogitdiff={_diff:.4f} solo={vl_solo.argmax(dim=-1).tolist()} "
                  f"batch={_vb.argmax(dim=-1).tolist()}", flush=True)
            # Force a FULL re-pull of row 0's shadow and diff against the incremental
            # gather's content: nonzero diff = the incremental multi-row gather left
            # stale/wrong K/V (the draft reads the shadow; the target reads the cache).
            # Skipped under INT8-primary shadows (no full-depth fp16 buffer to diff).
            if not st_.get("i8p"):
                _seq0 = cache.seq_lens[r0]
                _wb0 = cache.window_begs[r0]
                _swa0 = min(_seq0 - _wb0, st_["ks"].shape[1])
                _kf_b = st_["kf"][r0, :_seq0].clone()
                _vf_b = st_["vf"][r0, :_seq0].clone()
                _ks_b = st_["ks"][r0, :_swa0].clone()
                _vs_b = st_["vs"][r0, :_swa0].clone()
                st_["fls"][r0] = 0
                st_["sls"][r0] = 0
                st_["wbs"][r0] = _wb0
                pg.gather_kv_mtp_batch([r0])
                _dk = (st_["kf"][r0, :_seq0] - _kf_b).abs().max().item()
                _dv = (st_["vf"][r0, :_seq0] - _vf_b).abs().max().item()
                _dks = (st_["ks"][r0, :_swa0] - _ks_b).abs().max().item()
                _dvs = (st_["vs"][r0, :_swa0] - _vs_b).abs().max().item()
                print(f"[xcheck-gather] row {r0} re-pull diffs K={_dk:.5f} V={_dv:.5f} "
                      f"Ks={_dks:.5f} Vs={_dvs:.5f} (seq={_seq0} swa={_swa0} wb={_wb0})",
                      flush=True)
        for _bi in range(len(rows)):
            _s, _n = self.agree_bi.get(_bi, (0, 0))
            self.agree_bi[_bi] = (_s + int(_eq[_bi].sum()), _n + W)
            for _k in range(W):
                self.agree_k[_k] += int(_eq[_bi, _k])
                self.agree_kn[_k] += 1

        new_lens = list(cache.seq_lens)
        _t = self._time.perf_counter()
        # Parallel acceptance precompute: penalty-free sampling rows have accept probs
        # that do not depend on the acceptance outcome (no rep-penalty history growth),
        # so each row's whole (W+1, V) probs + CDFs + residual CDFs can be computed up
        # front -- in worker threads (the ext sampler releases the GIL). Rows with
        # penalties/exotic modifiers keep the exact serial walk below. Random draws
        # stay in the serial per-row walk, in the original order.
        pre = {}
        for bi, r in enumerate(rows):
            st_ = settings_map.get(r)
            if st_ is None:
                continue
            if (st_.token_repetition_penalty != 1.0
                    or st_.token_frequency_penalty != 0.0
                    or st_.token_presence_penalty != 0.0
                    or st_.mirostat or st_.xtc_probability > 0.0
                    or st_.dry_multiplier > 0.0 or st_.token_bias is not None
                    or st_.cfg_scale is not None or st_.temperature_last
                    or st_.smoothing_factor > 0.0):
                continue
            seq_ = seq_ids_map.get(r)
            if seq_ is None:
                seq_ = torch.zeros((1, 0), dtype=torch.long)
            ban_ = ban_map.get(r) if ban_map else None
            pre[bi] = (vlogits[bi], st_, seq_, ban_, draft[bi].tolist())
        if len(pre) > 1:
            _pool = _get_accept_pool()
            _futs = {bi: _pool.submit(_accept_precompute, vl_, st_, tokenizer, seq_, ban_, d_, W)
                     for bi, (vl_, st_, seq_, ban_, d_) in pre.items()}
            _tp = self._time.perf_counter()
            pre = {bi: f.result() for bi, f in _futs.items()}
            self.t_pre += self._time.perf_counter() - _tp
        elif len(pre) == 1:
            bi_, (vl_, st_, seq_, ban_, d_) = next(iter(pre.items()))
            _tp = self._time.perf_counter()
            pre = {bi_: _accept_precompute(vl_, st_, tokenizer, seq_, ban_, d_, W)}
            self.t_pre += self._time.perf_counter() - _tp
        for bi, r in enumerate(rows):
            row = self.rows[r]
            p = pos[bi]
            settings = settings_map.get(r)
            seq_ids = seq_ids_map.get(r)
            ban = ban_map.get(r) if ban_map else None
            d = draft[bi].tolist()
            vl = vlogits[bi]                                    # (W+1, V)

            if settings is None:
                va = vl.argmax(dim=-1)
                m_acc = 0
                while m_acc < W and int(va[m_acc]) == d[m_acc]:
                    m_acc += 1
                bonus_id = int(va[m_acc])
                # (1,1,H) -- EVERY stored last_state must be 3-dim: begin_row seeds
                # (1,1,H), and rounds cat states across rows at different lifecycle
                # stages. vstates[bi, k:k+1] would drop a dim ((1,H)); homogeneous
                # batches tolerated it (the drafter reshapes all-2-dim input), but a
                # freshly admitted row mixed with rounded rows crashed the cat.
                last_state = vstates[bi:bi + 1, m_acc:m_acc + 1].float().cpu()
            elif bi in pre:
                # Fast path: precomputed probs/CDFs (parallel across rows). Same math
                # and draw order as the serial walk: u_k ~ U[0,1) per draft, residual
                # draw on the first rejection, one bonus draw on full acceptance.
                import random as _random
                res = pre[bi]
                P, C, RC = res.result() if hasattr(res, "result") else res
                vocab = vl.shape[-1]
                m_acc = 0
                bonus_id = None
                for k in range(W):
                    if _random.random() < float(P[k, d[k]]):
                        m_acc = k + 1
                        continue
                    bonus_id = min(int(torch.searchsorted(
                        RC[k], torch.tensor(_random.random(), dtype=torch.float32)
                    ).item()), vocab - 1)
                    last_state = vstates[bi:bi + 1, k:k + 1].float().cpu()
                    break
                if bonus_id is None:
                    bonus_id = min(int(torch.searchsorted(
                        C[W], torch.tensor(_random.random(), dtype=torch.float32)
                    ).item()), vocab - 1)
                    last_state = vstates[bi:bi + 1, W:W + 1].float().cpu()
            else:
                # BUG 105/106: probabilistic acceptance against the TARGET distribution,
                # per row -- the scalar mode-B math verbatim. The draft is greedy (point
                # mass q_k on d[k]): accept with p_target_k(d[k]); on rejection resample
                # the residual normalize(max(0, p - q_k)). Emitted distribution is exactly
                # target-autoregressive; draft quality only affects the acceptance RATE.
                # Serial fallback for penalty/exotic rows: probs at row k depend on the
                # accepted drafts appended to the penalty history.
                import random as _random
                m_acc = 0
                bonus_id = None
                last_state = None
                seq = seq_ids if seq_ids is not None else torch.zeros((1, 0), dtype=torch.long)
                for k in range(W):
                    # (1,1,V) slice shape the sampler pipeline requires. NB: locals here
                    # must NOT be named `row` (the per-slot state dict) or `p` (the row's
                    # cache position) -- shadowing either corrupts the post-loop updates.
                    lg_row = vl[k:k + 1].unsqueeze(0)
                    probs = MTPMPDriver._dense_probs(lg_row, settings, tokenizer, seq, ban)
                    dk = d[k]
                    if _random.random() < float(probs[dk]):
                        m_acc = k + 1
                        seq = torch.cat([seq, torch.tensor([[dk]], dtype=torch.long)], dim=-1)
                        continue
                    resid = probs.clone()
                    resid[dk] = 0.0
                    s = float(resid.sum())
                    resid = resid / s if s > 1e-12 else probs
                    bonus_id = MTPMPDriver._sample_from(resid, _random.random())
                    last_state = vstates[bi:bi + 1, k:k + 1].float().cpu()
                    break
                if bonus_id is None:
                    pW = MTPMPDriver._dense_probs(vl[W:W + 1].unsqueeze(0), settings,
                                                  tokenizer, seq, ban)
                    bonus_id = MTPMPDriver._sample_from(pW, _random.random())
                    last_state = vstates[bi:bi + 1, W:W + 1].float().cpu()

            rewind = p + m_acc + 1
            new_lens[r] = rewind
            row["pending"] = d[:m_acc] + [bonus_id]
            row["pending_bonus"] = bonus_id
            row["last_state"] = last_state
            row["expected_pos"] = rewind
            self.mtp_acc += m_acc
            self.mtp_drafts += W

        self.t_accept += self._time.perf_counter() - _t
        # One distributed rewind for every row (full layers: bookkeeping only; SWA: the
        # ring fill shrinks, slots beyond it are ignored until overwritten).
        _t = self._time.perf_counter()
        pg.mtp_set_seq_lens(new_lens)
        self.t_rewind += self._time.perf_counter() - _t
        self.mtp_rounds += 1
        self.mtp_emitted += sum(len(self.rows[r]["pending"]) for r in rows)
        self.t_round += self._time.perf_counter() - _t0
        # Sparse draft head: count every row's committed tokens (accepted drafts +
        # bonus) in ONE observation -- the scalar driver's per-round observe, batched.
        if hasattr(self.drafter, "observe_tokens"):
            self.drafter.observe_tokens(torch.tensor(
                [t for r in rows for t in self.rows[r]["pending"]], dtype=torch.long))
        if self.mtp_rounds % 50 == 0:
            r = self.mtp_rounds
            agree = (self.draft_agree_sum / self.draft_agree_n
                     if self.draft_agree_n else 0.0)
            bi_str = " ".join(
                f"bi{b}={s / n:.2f}" for b, (s, n) in sorted(self.agree_bi.items()) if n)
            k_str = " ".join(
                f"k{k}={self.agree_k[k] / self.agree_kn[k]:.2f}"
                for k in range(W) if self.agree_kn[k])
            print(f"[mtp-mp-batch] {r} rounds B={len(rows)} pre_n={len(pre)} "
                  f"acc/round={self.mtp_acc / r:.3f} "
                  f"emit/round={self.mtp_emitted / r:.3f} "
                  f"ms/round={self.t_round / r * 1000:.2f} "
                  f"(gather={self.t_gather / r * 1000:.2f} "
                  f"draft={self.t_draft / r * 1000:.2f} "
                  f"verify={self.t_verify / r * 1000:.2f} "
                  f"accept={self.t_accept / r * 1000:.2f} "
                  f"pre={self.t_pre / r * 1000:.2f} "
                  f"rewind={self.t_rewind / r * 1000:.2f}) "
                  f"draft_agree={agree:.3f} [{bi_str}] [{k_str}]", flush=True)

    def stats(self):
        return {
            "mtp_rounds": self.mtp_rounds,
            "mtp_draft_tokens": self.mtp_drafts,
            "mtp_accepted_tokens": self.mtp_acc,
            "mtp_acceptance_rate": (self.mtp_acc / self.mtp_drafts) if self.mtp_drafts else 0.0,
        }
