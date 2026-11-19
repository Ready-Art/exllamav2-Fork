"""
hb16 vs hb8 logit parity test.

PLAN.md §2.7: "Add a test that converts a tiny model at -hb 16 and at -hb 8 and asserts
both load and produce identical logits on a fixed prompt (the head is the last layer, so a
broken head shows up immediately)."

Requires CUDA and a tiny HF checkpoint. Skipped otherwise.

Run with:
    pytest tests/test_hb16.py -v --hf-model /path/to/tiny-model --work-dir /tmp/hb16
"""

import os
import subprocess
import sys

import pytest
import torch


# NOTE: --hf-model / --work-dir are registered in tests/conftest.py. pytest ignores
# `pytest_addoption` defined inside a test module, so declaring them here had no effect.


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "hb16 parity requires CUDA",
)


def _convert(hf_model, out_dir, head_bits):
    """Run the converter at a given head bitrate."""
    from exllamav2.conversion.convert_exl2 import main as _  # noqa: F401

    cmd = [
        sys.executable, "-m", "exllamav2.conversion.convert_exl2",
        "-i", hf_model,
        "-o", out_dir,
        "-b", "4.0",
        "-hb", str(head_bits),
        "-c", os.environ.get("EXL2_CAL_DATASET", ""),
    ]
    subprocess.run(cmd, check = True)


def _logits(model_dir, ids):
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
    cfg = ExLlamaV2Config(model_dir)
    m = ExLlamaV2(cfg)
    m.load()
    cache = ExLlamaV2Cache(m, batch_size = 1, max_seq_len = ids.shape[-1] + 8)
    logits, _ = m.forward(ids, cache = cache, last_id_only = True)
    return logits


def test_hb16_accepted_by_converter():
    """`-hb 16` must be a supported option. PLAN.md §2.7."""
    from exllamav2.conversion.qparams import qparams_headoptions
    assert 16 in qparams_headoptions
    assert qparams_headoptions[16] is None, "16 must map to None (leave unquantized)"
    # hb8 must still work.
    assert 8 in qparams_headoptions


def test_quant_lm_head_early_return():
    """quant_lm_head must not touch qparams when head_bits == 16. PLAN.md §2.7 step 2."""
    from exllamav2.conversion.quantize import quant_lm_head

    # A job with head_bits 16 and a quantizers dict that would raise if touched.
    class Boom:
        def __getitem__(self, k):
            raise AssertionError("quantizers must not be touched for head_bits == 16")

    quant_lm_head({"head_bits": 16}, None, None, Boom(), None)


@pytest.mark.skipif(
    not os.environ.get("EXL2_CAL_DATASET"),
    reason = "EXL2_CAL_DATASET not set; conversion needs a calibration dataset",
)
def test_hb16_vs_hb8_logits(request):
    """
    Convert the same tiny model at -hb 16 and -hb 8 and assert identical logits on a
    fixed prompt. PLAN.md §2.7.
    """
    hf_model = request.config.getoption("--hf-model")
    if not hf_model:
        pytest.skip("--hf-model not given")

    work = request.config.getoption("--work-dir")
    d16 = os.path.join(work, "hb16")
    d8 = os.path.join(work, "hb8")

    _convert(hf_model, d16, 16)
    _convert(hf_model, d8, 8)

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype = torch.long)

    l16 = _logits(d16, ids)
    l8 = _logits(d8, ids)

    torch.testing.assert_close(l16, l8, atol = 1e-3, rtol = 1e-3)
