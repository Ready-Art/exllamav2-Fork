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

Greedy-only (temperature=0), matching the validation regime: acceptance compares
argmax predictions and every emitted token comes from the TARGET's verified logits,
so output is identical to plain autoregressive generation at temperature 0.

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
    def step(self) -> int:
        """Emit one token (greedy) from the target's verified logits.

        Bookkeeping: the target's cache already holds every verified row (the
        verify forward wrote the accepted drafts' KV; the round's bonus-commit
        forward_state wrote the bonus row's KV). So consuming a row never
        advances ``current_seq_len`` -- the round's bonus commit is the only
        thing that moves the cache, and it already did.
        """
        if self.pending is None:
            self._round()
        logits = self.pending[:, :1, :]
        tok = int(logits[0, 0].argmax())
        self.pending = self.pending[:, 1:, :]
        if self.pending.shape[1] == 0:
            self.pending = None
        self.tokens.append(tok)
        return tok

    def _round(self):
        # ---- begin a round ----
        if self.state is None or self.pg.cache.current_seq_len != self.state[2] + 1:
            self._commit_base()
        base_tok, base_state, base_pos = self.state
        pos = self.pg.cache.current_seq_len            # == base_pos + 1
        _t = self._time.perf_counter()

        # 1. gather target K/V for the drafter (once per round)
        self.d.gather_target(self.pg)
        self.t_gather += self._time.perf_counter() - _t
        _t = self._time.perf_counter()

        # 2. draft a greedy window
        draft = self.d.draft_window(base_tok, base_state, pos, self.window)   # (1,W) cpu
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
        vlogits = self.pg.forward(draft.cpu(), pos, draft.shape[-1], 1, False,
                                  graph_verify = True)
        self.t_verify += self._time.perf_counter() - _t
        vlogits = vlogits.float().cpu()

        # 4. accept the longest draft prefix the target agrees with.
        m_acc = 0
        draft_flat = draft[0].tolist()
        v_argmax = vlogits[0].argmax(dim=-1).tolist()
        bonus_pred = int(self.base_logits[0, 0].argmax())
        for k in range(draft.shape[-1]):
            pred = bonus_pred if k == 0 else v_argmax[k - 1]
            if pred == draft_flat[k]:
                m_acc = k + 1
            else:
                break
        self.total_acc += m_acc
        self.total_rounds += 1
        if self.verbose:
            print(f"[mtp-mp] round pos={pos} acc={m_acc}/{draft.shape[-1]}", flush=True)

        # Verified logits for the emitted tokens: base row + accepted drafts' rows.
        self.pending = torch.cat([self.base_logits, vlogits[:, :m_acc, :]], dim=1)

        # 5. rewind past the unaccepted tail; commit the bonus token (the target's
        #    own pick at position pos+m_acc) to obtain the next round's base.
        _t = self._time.perf_counter()
        self.pg.mtp_set_seq_len(pos + m_acc)
        self.t_rewind += self._time.perf_counter() - _t
        _t = self._time.perf_counter()
        bonus_id = int(self.pending[0, m_acc].argmax())
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
                },
                "overhead_ms/token": round(
                    (self.t_gather + self.t_draft + self.t_verify + self.t_rewind
                     + self.t_bonus) / max(self.total_acc + self.total_rounds, 1) * 1000, 2),
            })
        return s
