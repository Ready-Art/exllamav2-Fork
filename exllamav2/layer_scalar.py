from __future__ import annotations
import torch
from exllamav2.module import ExLlamaV2Module
from exllamav2.stloader import STFile

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2


class ExLlamaV2LayerScalar(ExLlamaV2Module):
    """
    Per-layer scalar multiplier applied to the transformer block output.

    Gemma-4 stores a 1-element ``layers.N.layer_scalar`` tensor per layer. exllamav3 applies it at
    ``modules/transformer.py:249``, *after* both residual adds:

        x = x + attn_out
        x = x + mlp_out
        x *= layer_scalar

    It therefore cannot be folded into the ``o_proj`` / ``down_proj`` prescale (which would scale
    each branch separately rather than their sum). This module exists to apply it in the right
    place. PLAN.md §2.6.

    The module holds no weights of its own beyond the 1-element scalar, so it is inserted into the
    module list immediately after each MLP and is a no-op for architectures without layer scalars.
    """

    name: str = "LayerScalar"

    scalar: torch.Tensor | None

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        archparams = None,
    ):
        super().__init__(model, key, archparams)
        self.scalar = None
        self.footprint = 0

    @torch.inference_mode
    def load(self, device_context: bool = True):

        w = self.load_weight()

        # load_weight only finds `<key>.weight` / `<key>.q_weight` / `<key>.qweight`. Gemma-4's
        # tensor is named exactly `layers.N.layer_scalar`, with no suffix at all, so it has to be
        # fetched by its full name.
        #
        # NOTE: `load_multi(self.key, ["layer_scalar"])` does NOT work here — load_multi always
        # appends `"." + k` to the key it is given, so it would look for
        # `layers.N.layer_scalar.layer_scalar`. Read the tensor straight out of the tensor map.
        if w is None:
            scalar = self._load_scalar_tensor()
            if scalar is None:
                raise ValueError(
                    f"Could not find layer scalar tensor at {self.key} "
                    f"(looked for the exact key, not {self.key}.weight)"
                )
            w = torch.nn.Parameter(scalar.half(), requires_grad = False)

        if isinstance(w, tuple):
            w = w[0]

        assert w.numel() == 1, \
            f"layer_scalar at {self.key} must have exactly 1 element, got {w.numel()}"

        self.scalar = w.data.reshape(1).to(self.device())

    def _load_scalar_tensor(self) -> torch.Tensor | None:
        """Fetch the 1-element tensor stored under exactly ``self.key``."""

        cfg = self.model.config
        filename = cfg.tensor_file_map.get(self.key)
        if filename is None:
            return None

        stfile = STFile.open(filename, keymap = cfg.arch.keymap)
        return stfile.get_tensor(self.key, device = "cpu")

    def unload(self):
        self.scalar = None

    def numel(self):
        return 1

    def get_weight(self) -> torch.Tensor | None:
        """Return the scalar for the conversion pipeline. Saved under `key`, not `key.weight`."""
        if self.scalar is None:
            return None
        return self.scalar.reshape(1).cpu()

    def weight_footprint(self) -> int:
        return 2

    def scratch_space_fixed(self) -> int:
        return 0

    def scratch_space(self) -> int:
        return 0

    def scratch_space_tp(self) -> list[int]:
        # The scalar multiply is elementwise and needs no scratch. load_tp() calls this on
        # every module in the list, so it must exist even when it allocates nothing.
        return [0] * self.model.tp_context.num_devices

    def tp_split(self):
        # Nothing to shard: the scalar is 1 element and is broadcast to every device by the
        # TP context. Present so load_tp()'s isinstance dispatch is uniform.
        self.is_tp = True
        self.set_device_idx(None)

    def tp_load_rank(self, rank: int):
        """
        Load the 1-element scalar for a rank-local worker.

        There is nothing to shard: the scalar is applied elementwise to the block output, so every
        rank needs the same value. ``tp_split()`` accordingly only sets ``is_tp = True`` and leaves
        the tensor alone.

        ``is_tp`` must stay False here. In the single-threaded path it means "the scalar is not
        pinned to a device and must be moved to the hidden-state device at forward time"; a worker
        owns one device and can pin it.

        :param rank:
            Device index this worker owns.
        """
        self.device_idx = rank
        self.load()
        self.is_tp = False
        if self.scalar is not None:
            self.scalar = self.scalar.to(self.device())

    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)
        # idx None means "sharded" in TP mode (see ExLlamaV2Module.set_device_idx), so
        # self.device() would be the invalid string "cuda:None". The scalar is 1 element and
        # is moved to the hidden-state device at forward time instead.
        if self.scalar is not None and idx is not None:
            self.scalar = self.scalar.to(self.device())

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        **kwargs
    ) -> torch.Tensor:

        if self.scalar is not None:
            # Move to the hidden-state device as well as dtype: in TP mode the scalar is not
            # pinned to a device (see set_device_idx), so it can still be on CPU here.
            hidden_states = hidden_states * self.scalar.to(
                device = hidden_states.device, dtype = hidden_states.dtype
            )

        # The conversion pipeline calls forward(intermediates = True) and reads
        # outputs["hidden_states"], so honour the flag like every other module.
        if intermediates:
            return {"hidden_states": hidden_states}
        return hidden_states
