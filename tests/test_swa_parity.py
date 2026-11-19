"""
SWA KV compression parity tests.

PLAN.md §4.6 verification list. Requires a CUDA machine and a converted Gemma-4
checkpoint; skipped otherwise.

Run with:
    pytest tests/test_swa_parity.py -v --model-dir /path/to/gemma-4-31b-exl2

The core assertion is that the compressed (ring-buffer) path produces the same logits as
the uncompressed path. Sequences shorter than sliding_window must match *exactly*; longer
sequences exercise the page-aligned shift.
"""

import pytest
import torch


def _model_dir(request):
    return request.config.getoption("--model-dir")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "SWA parity requires CUDA and a converted checkpoint",
)


@pytest.fixture(scope = "module")
def model(request):
    md = _model_dir(request)
    if not md:
        pytest.skip("--model-dir not given")

    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
    cfg = ExLlamaV2Config(md)
    m = ExLlamaV2(cfg)
    m.load()
    tok = ExLlamaV2Tokenizer(cfg)
    return m, tok, cfg


def _logits(model, tokenizer, ids, cache_cls, max_seq_len):
    from exllamav2.generator import ExLlamaV2Sampler
    cache = cache_cls(model, batch_size = 1, max_seq_len = max_seq_len)
    logits, _ = model.forward(ids, cache = cache, last_id_only = True)
    return logits


def test_ring_size(model):
    """
    Ring must be 1280 tokens per sliding layer at page_size 128, sliding_window 1024,
    overp 256. Full layers keep max_seq_len. PLAN.md §4.6 / §3.6 item 6.
    """
    from exllamav2.cache import ExLlamaV2Cache_SWA
    from exllamav2.generator.dynamic import PAGED_PAGE_SIZE

    m, tok, cfg = model
    cache = ExLlamaV2Cache_SWA(m, batch_size = 1, max_seq_len = 4096,
                              page_size = PAGED_PAGE_SIZE)

    assert PAGED_PAGE_SIZE == 128
    assert cache.kv_state_size == 1280
    assert cache.kv_state_size % PAGED_PAGE_SIZE == 0

    for i in range(cfg.num_hidden_layers):
        if cache.is_swa_layer[i]:
            assert cache.key_states[i].shape[1] == 1280
        else:
            assert cache.key_states[i].shape[1] == 4096


def test_shift_is_page_aligned(model):
    """
    Every shift must be a whole number of pages. A non-aligned shift makes the block
    table point at the wrong slots and degrades output gradually rather than failing.
    PLAN.md §4.6 / §8 trap 13.
    """
    from exllamav2.cache import ExLlamaV2Cache_SWA
    from exllamav2.generator.dynamic import PAGED_PAGE_SIZE

    m, tok, cfg = model
    cache = ExLlamaV2Cache_SWA(m, batch_size = 1, max_seq_len = 4096,
                              page_size = PAGED_PAGE_SIZE)

    swa_layer = next(i for i in range(cfg.num_hidden_layers) if cache.is_swa_layer[i])

    # A page-aligned shift must succeed.
    cache.shift_ring(swa_layer, 128)

    # A non-aligned shift must be rejected.
    with pytest.raises(AssertionError):
        cache.shift_ring(swa_layer, 100)


@pytest.mark.parametrize("seq_len", [64, 512, 1024])
def test_short_sequence_exact(model, seq_len):
    """
    For sequences shorter than sliding_window the compressed path must match the
    uncompressed path exactly. PLAN.md §4.6.
    """
    from exllamav2.cache import ExLlamaV2Cache, ExLlamaV2Cache_SWA

    m, tok, cfg = model
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    ref = _logits(m, tok, ids, ExLlamaV2Cache, 2048)
    got = _logits(m, tok, ids, ExLlamaV2Cache_SWA, 2048)

    torch.testing.assert_close(got, ref, atol = 1e-3, rtol = 1e-3)


@pytest.mark.parametrize("mult", [2, 3, 10])
def test_long_sequence_parity(model, mult):
    """
    Sequences longer than sliding_window exercise the shift logic. Test at 2x, 3x and
    10x the window. PLAN.md §4.6.
    """
    from exllamav2.cache import ExLlamaV2Cache, ExLlamaV2Cache_SWA

    m, tok, cfg = model
    seq_len = cfg.sliding_window * mult
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    ref = _logits(m, tok, ids, ExLlamaV2Cache, seq_len + 128)
    got = _logits(m, tok, ids, ExLlamaV2Cache_SWA, seq_len + 128)

    torch.testing.assert_close(got, ref, atol = 1e-3, rtol = 1e-3)


def test_no_triton_import():
    """
    The SWA path must not import Triton: this plan targets Ampere with no Triton anywhere.
    PLAN.md §4.2 / §4.6.

    Asserted against exllamav2's own source rather than ``sys.modules``. The original version of
    this test checked ``"triton" not in sys.modules`` after importing exllamav2, which is not a
    statement about exllamav2 at all: FlashInfer imports Triton transitively, so the test failed
    purely because FlashInfer was installed. The property that actually matters is that no module
    under ``exllamav2/`` imports Triton, which is what this checks.
    """
    import pathlib
    import re

    import exllamav2

    root = pathlib.Path(exllamav2.__file__).parent
    pattern = re.compile(r"^\s*(import triton|from triton)", re.MULTILINE)

    offenders = []
    for py in root.rglob("*.py"):
        try:
            text = py.read_text(encoding = "utf8", errors = "ignore")
        except OSError:
            continue
        if pattern.search(text):
            offenders.append(str(py.relative_to(root)))

    assert not offenders, f"exllamav2 must not import triton, found in: {offenders}"
