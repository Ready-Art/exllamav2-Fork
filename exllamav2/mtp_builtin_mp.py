from __future__ import annotations

"""
MP adapter for the built-in MTP drafters (Qwen3.5 / Qwen3.8-27B and
Qwen3.8-Flash-Next). Exposes the MTPMPDriver protocol on top of a drafter that
keeps its OWN draft KV (no target-KV gather):

  * gather_target(pg)      -> no-op (own KV)
  * prefill_target(ids, states) -> drafter.prefill  (driver begin() hook)
  * draft_window(...)      -> drafter.draft_window (provisional advance + rewind)
  * commit_accepted(ids, states, pos) -> drafter.commit (driver post-accept hook)
  * commit_bonus(tok, state, pos)     -> drafter.commit (legacy bonus forward)
  * trim(n)                -> distributed rewind tracking (mtp_set_seq_len hook)

The target-side GDN/PLE recurrent state is staged during the verify forward and
restored by the distributed rewind (tp_mp set-seqlen -> model.gdn_rewind_to),
so rejected verify rows never corrupt the hybrid backbone state.

Greedy drafting only (the built-in drafters are greedy chains); sampled MTP
acceptance still works because the driver resamples from the target's verified
logits -- draft quality only affects the acceptance rate.
"""

import torch


class BuiltinMTPMPAdapter:

    def __init__(self, drafter, pg):
        self.d = drafter
        self.pg = pg
        self.is_dflash = False
        self.is_builtin = True
        self.num_draft_tokens = drafter.num_draft_tokens
        self.backbone_hidden_size = drafter.hidden_size

    # -------------------------------------------------- driver protocol

    def observe_tokens(self, ids):
        # Greedy chain: no sparse-head subset to refresh.
        pass

    def refresh_sparse_head(self):
        pass

    def gather_target(self, pg):
        return 0

    def gather_target_batch(self, rows):
        return 0

    def prefill_target(self, ids, states):
        self.d.prefill(ids, states)

    def draft_window(self, base_tok, base_state, pos, num_tokens):
        return self.d.draft_window(base_tok, base_state, pos, num_tokens)

    def commit_accepted(self, ids, states, pos):
        self.d.commit(ids, states, pos)

    def commit_bonus(self, tok, state, pos):
        self.d.commit(tok, state, pos)

    def trim(self, n):
        # The built-in drafters keep ordinary attention KV (no recurrent state):
        # truncation to n is exact -- rows [0, n) stay valid. This also covers the
        # driver's token-healing rewind (n = committed - 1) and full resets.
        self.d.rewind(n)

    # -------------------------------------------------- batch-driver stubs
    # The multi-batch driver requires per-row draft KV shadowing the target's
    # batch rows; the built-in drafters are single-sequence. tabby keeps
    # max_batch=1 for built-in MTP on MP.

    def draft_window_batch(self, *a, **k):
        raise NotImplementedError("built-in MTP on MP is single-sequence (max_batch=1)")
