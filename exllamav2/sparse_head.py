"""
Dynamic sparse draft LM head (port of exllamav3's sparse_draft_head to the exl2 fork).

The MTP assistant and DFlash drafters both spend a large share of their draft latency
reading the FULL vocabulary head (MTP: 262144x1024 fp16 = 512 MiB per draft step;
DFlash: 262144x5376 = 2.7 GiB per block). Drafting only needs the argmax, and the
argmax almost always lives in a small high-frequency subset of the vocabulary. This
module keeps a frequency-selected subset of head rows on the drafter device; the
drafter argmaxes over the subset and maps back to full-vocab ids. VERIFICATION always
uses the target's full head, so the output distribution is unaffected except for
occasional draft misses (which speculative acceptance simply rejects).

Safety (exl3's self-gating pattern):
  * The subset is built from token frequencies observed in prompts + generations,
    persisted per model under ~/.cache/exllamav2-fork/token_stats/. Until enough
    tokens are observed, the full head is used.
  * Periodic probes compute BOTH argmaxes on the same hidden state and measure the
    miss rate; above a threshold the sparse head disables itself (full head resumes).
  * The subset is rebuilt only at job boundaries (driver.begin), never mid-round, so
    captured draft graphs stay valid; a rebuild invalidates the graph explicitly.

No Triton, no new kernels: the subset GEMM is the same torch matmul, narrower.
"""

from __future__ import annotations

import os
import tempfile

import torch


_STATS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "exllamav2-fork", "token_stats")
_MIN_OBSERVED = 4096      # tokens observed before the subset is trusted
_SAVE_EVERY = 8192        # observations between persistence writes
_PROBE_EVERY = 64         # draft windows between miss-rate probes
_PROBE_MIN = 16           # probes before the gate may disable
_MISS_RATE_MAX = 0.05     # disable above this probe miss rate


class SparseHead:
    """Frequency-subset draft head over a full (vocab, hidden) fp16 weight matrix.

    ``active`` is False until enough statistics exist (or after the miss gate trips);
    callers must fall back to the full head then. ``logits``/``map_ids`` operate on
    the subset; ``probe`` measures full-vs-sparse argmax agreement on one state.
    """

    def __init__(self, full_head: torch.Tensor, columns: int, stats_key: str,
                 vocab_size: int, device: torch.device, int8: bool = False):
        self.full = full_head
        self.columns = int(columns)
        self.device = device
        self.vocab_size = int(vocab_size)
        self.stats_key = stats_key
        # INT8 weight-only subset GEMV (opt-in, drafter EXL2_MTP_INT8_W): the subset
        # is re-quantized on every rebuild; the full head stays fp16 for the probe.
        self.int8 = bool(int8)
        self.subset8 = None
        self.subset_s = None
        # NORMAL tensor: constructed under the loader's InferenceMode, but observe()
        # does in-place += on the job thread OUTSIDE it (inference tensors forbid that).
        with torch.inference_mode(False):
            self.counts = torch.zeros(full_head.shape[0], dtype=torch.long)
        self._load()
        self.observed = int(self.counts.sum())
        self._since_save = 0
        self.subset = None          # (columns, hidden) fp16 device
        self.ids = None             # (columns,) int64 device, sorted ascending
        self._ids_cpu = None
        self.probes = 0
        self.misses = 0
        self.disabled = False
        self.window_count = 0
        self.rebuild()

    # ------------------------------------------------------------------ stats

    def _path(self) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.stats_key)
        return os.path.join(_STATS_DIR, safe + ".pt")

    def _load(self):
        try:
            t = torch.load(self._path(), map_location = "cpu", weights_only = True)
            if t.shape == self.counts.shape:
                # copy_ into the NORMAL buffer (legal under InferenceMode), instead of
                # replacing it with a possibly-inference loaded tensor.
                self.counts.copy_(t.long())
        except (OSError, RuntimeError, ValueError):
            pass

    def _save(self):
        try:
            os.makedirs(_STATS_DIR, exist_ok = True)
            fd, tmp = tempfile.mkstemp(dir = _STATS_DIR, suffix = ".tmp")
            os.close(fd)
            torch.save(self.counts, tmp)
            os.replace(tmp, self._path())
        except OSError:
            pass

    def observe(self, ids: torch.Tensor):
        """Count token ids (prompt or generated). CPU tensor, any shape."""
        flat = ids.detach().to("cpu").reshape(-1).long()
        flat = flat[(flat >= 0) & (flat < self.counts.shape[0])]
        if flat.numel() == 0:
            return
        self.counts += torch.bincount(flat, minlength = self.counts.shape[0])
        self.observed += int(flat.numel())
        self._since_save += int(flat.numel())
        if self._since_save >= _SAVE_EVERY:
            self._since_save = 0
            self._save()

    # ------------------------------------------------------------------ subset

    def rebuild(self) -> bool:
        """(Re)select the top-frequency columns. Returns True if the subset CHANGED
        (callers must invalidate captured draft graphs then)."""
        if self.columns <= 0 or self.observed < _MIN_OBSERVED:
            return False
        c = self.counts.clone()
        c[self.vocab_size:] = 0            # head padding rows are never real tokens
        c[0] = 0                           # pad id
        top = torch.topk(c, min(self.columns, c.shape[0])).indices.sort().values
        if self._ids_cpu is not None and torch.equal(top, self._ids_cpu):
            return False
        self._ids_cpu = top
        with torch.inference_mode(False):
            self.subset = self.full[top.to(self.full.device)].contiguous()
            self.ids = top.to(self.device)
            if self.int8:
                from exllamav2.mtp_drafter import Gemma4AssistantDrafter
                self.subset8, self.subset_s = Gemma4AssistantDrafter._quant_rows(self.subset)
        self.probes = 0
        self.misses = 0
        self.disabled = False
        return True

    @property
    def active(self) -> bool:
        return self.subset is not None and not self.disabled

    # ------------------------------------------------------------------ draft

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        if self.subset8 is not None and x.numel() // shp[-1] <= 8:
            from exllamav2.ext import exllamav2_ext as ext_c
            return ext_c.mtp_int8_gemm(
                x.reshape(-1, shp[-1]).contiguous(),
                self.subset8, self.subset_s).view(*shp[:-1], self.subset.shape[0])
        return x @ self.subset.t()

    def map_ids(self, subset_ids: torch.Tensor) -> torch.Tensor:
        return self.ids[subset_ids]

    def argmax(self, x: torch.Tensor) -> torch.Tensor:
        return self.map_ids(self.logits(x).argmax(dim = -1))

    # ------------------------------------------------------------------ gate

    def should_probe(self) -> bool:
        self.window_count += 1
        return self.active and self.window_count % _PROBE_EVERY == 0

    def probe(self, x: torch.Tensor):
        """Full-vs-sparse argmax on the same hidden state (the exl3 'probe' flag)."""
        if not self.active:
            return
        full_id = (x @ self.full.t()).argmax(dim = -1)
        sparse_id = self.argmax(x)
        self.probes += int(full_id.numel())
        self.misses += int((full_id != sparse_id).sum())
        if self.probes >= _PROBE_MIN and self.misses / self.probes > _MISS_RATE_MAX:
            self.disabled = True
            print(f"[sparse-head] miss rate {self.misses}/{self.probes} > "
                  f"{_MISS_RATE_MAX:.0%}; disabling sparse draft head (full head resumes)",
                  flush = True)


def stats_key_for(model_dir: str, vocab_size: int) -> str:
    return f"{os.path.basename(os.path.normpath(str(model_dir)))}_{vocab_size}"
