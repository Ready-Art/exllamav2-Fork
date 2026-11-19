"""
FlashInfer vs torch parity tests.

PLAN.md §3.5 / Stage A step 2 and the §7 verification checklist.

These tests require a CUDA machine with torch and flashinfer installed; they are skipped
otherwise. Run with:

    pytest tests/test_flashinfer_parity.py -v

The reference implementation is ``attn_backend.torch_paged_attention``, which is a direct
transcription of the SDPA path in ``ExLlamaV2Attention._attn_torch``. Every FlashInfer result
must match it within atol=1e-3.
"""

import pytest
import torch

from exllamav2 import attn_backend


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "FlashInfer parity requires CUDA",
)


def _make_paged_inputs(batch, q_len, num_q_heads, num_kv_heads, head_dim,
                       seq_lens, page_size, device, seed = 0):
    """
    Build a random paged cache plus a query.

    The block table is deliberately non-contiguous (pages are permuted) so the page
    indirection is actually exercised, and ``seq_lens`` may be ragged so the
    indptr conversion is tested.
    """
    g = torch.Generator(device = device).manual_seed(seed)

    pages_per_seq = [(l + page_size - 1) // page_size for l in seq_lens]
    max_pages = max(pages_per_seq)
    num_pages = sum(pages_per_seq) + 5

    k_cache = torch.randn(
        (num_pages, page_size, num_kv_heads, head_dim),
        dtype = torch.half, device = device, generator = g,
    )
    v_cache = torch.randn(
        (num_pages, page_size, num_kv_heads, head_dim),
        dtype = torch.half, device = device, generator = g,
    )

    # Assign each sequence a permuted, disjoint set of pages.
    # NOTE: `device=` is required here. Passing a CUDA generator to `randperm` without it raises
    # "Expected a 'cpu' device type for generator but found 'cuda'" — `randperm` defaults to CPU
    # and then refuses the generator. This is a test-harness bug, not a product bug.
    perm = torch.randperm(num_pages, generator = g, device = device)
    block_index = torch.zeros((batch, max_pages), dtype = torch.int32, device = device)
    cursor = 0
    for b, npages in enumerate(pages_per_seq):
        block_index[b, :npages] = perm[cursor:cursor + npages]
        cursor += npages

    q = torch.randn(
        (batch, q_len, num_q_heads, head_dim),
        dtype = torch.half, device = device, generator = g,
    )

    cache_seqlens = torch.tensor(seq_lens, dtype = torch.int32, device = device)

    return q, k_cache, v_cache, cache_seqlens, block_index


@pytest.mark.skipif(not attn_backend.has_flashinfer, reason = "flashinfer not installed")
@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_prefill_parity(page_size, head_dim):
    """q_len > 1: the prefill path."""
    device = torch.device("cuda:0")
    q, k, v, csl, bi = _make_paged_inputs(
        batch = 1, q_len = 64, num_q_heads = 8, num_kv_heads = 2,
        head_dim = head_dim, seq_lens = [64], page_size = page_size, device = device,
    )
    ref, got = _run_both(q, k, v, csl, bi, page_size, head_dim ** -0.5)
    torch.testing.assert_close(got, ref, atol = 1e-3, rtol = 1e-3)


@pytest.mark.skipif(not attn_backend.has_flashinfer, reason = "flashinfer not installed")
@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_decode_parity(page_size, head_dim):
    """q_len == 1: the decode path, the hottest path in the model."""
    device = torch.device("cuda:0")
    q, k, v, csl, bi = _make_paged_inputs(
        batch = 4, q_len = 1, num_q_heads = 8, num_kv_heads = 2,
        head_dim = head_dim, seq_lens = [1, 1, 1, 1], page_size = page_size, device = device,
    )
    ref, got = _run_both(q, k, v, csl, bi, page_size, head_dim ** -0.5)
    torch.testing.assert_close(got, ref, atol = 1e-3, rtol = 1e-3)


@pytest.mark.skipif(not attn_backend.has_flashinfer, reason = "flashinfer not installed")
@pytest.mark.parametrize("page_size", [128, 256])
def test_ragged_batch_parity(page_size):
    """
    Batch > 1 with ragged cache_seqlens. This is where the indptr/indices conversion
    is most likely to be wrong. PLAN.md §7.
    """
    device = torch.device("cuda:0")
    q, k, v, csl, bi = _make_paged_inputs(
        batch = 3, q_len = 1, num_q_heads = 8, num_kv_heads = 2,
        head_dim = 128, seq_lens = [1, 200, 77], page_size = page_size, device = device,
    )
    ref, got = _run_both(q, k, v, csl, bi, page_size, 128 ** -0.5)
    torch.testing.assert_close(got, ref, atol = 1e-3, rtol = 1e-3)


@pytest.mark.skipif(not attn_backend.has_flashinfer, reason = "flashinfer not installed")
@pytest.mark.parametrize("window", [16, 64, 1024])
def test_sliding_window_parity(window):
    """Sliding window: FlashInfer's window_left vs the torch reference."""
    device = torch.device("cuda:0")
    q, k, v, csl, bi = _make_paged_inputs(
        batch = 1, q_len = 8, num_q_heads = 8, num_kv_heads = 2,
        head_dim = 128, seq_lens = [8], page_size = 128, device = device,
    )
    ref, got = _run_both(q, k, v, csl, bi, 128, 128 ** -0.5, window_left = window)
    torch.testing.assert_close(got, ref, atol = 1e-3, rtol = 1e-3)


@pytest.mark.skipif(not attn_backend.has_flashinfer, reason = "flashinfer not installed")
def test_softcap_parity():
    """Logit softcapping (Gemma-2/3/4). FlashInfer supports it natively."""
    device = torch.device("cuda:0")
    q, k, v, csl, bi = _make_paged_inputs(
        batch = 1, q_len = 32, num_q_heads = 8, num_kv_heads = 2,
        head_dim = 128, seq_lens = [32], page_size = 128, device = device,
    )
    ref, got = _run_both(q, k, v, csl, bi, 128, 128 ** -0.5, softcap = 30.0)
    torch.testing.assert_close(got, ref, atol = 1e-3, rtol = 1e-3)


def test_head_dim_512_is_rejected():
    """
    Gemma-4's global layers have head_dim 512. FlashInfer treats 512 as MLA, so the
    backend must refuse it and the caller must fall back to the torch path.
    PLAN.md §3.7 / §8 trap 5.
    """
    assert not attn_backend.FlashInferPaged.supports(
        head_dim = 512, num_q_heads = 32, num_kv_heads = 4, page_size = 128,
    )


def test_page_size_128_is_fast_path():
    """PAGED_PAGE_SIZE must be 128, not 256. PLAN.md §3.6."""
    from exllamav2.generator.dynamic import PAGED_PAGE_SIZE
    assert PAGED_PAGE_SIZE == 128
    assert PAGED_PAGE_SIZE in attn_backend.FLASHINFER_FAST_PAGE_SIZES


def test_indptr_dtype_is_int32():
    """
    indptr/indices must be int32; int64 causes indexing errors. PLAN.md §8 trap 9.
    """
    device = torch.device("cuda:0")
    _, _, _, csl, bi = _make_paged_inputs(
        batch = 2, q_len = 1, num_q_heads = 8, num_kv_heads = 2,
        head_dim = 128, seq_lens = [1, 200], page_size = 128, device = device,
    )
    indptr, indices, last_page_len = attn_backend.page_table_to_indptr(bi, csl, 128)
    assert indptr.dtype == torch.int32
    assert indices.dtype == torch.int32
    assert last_page_len.dtype == torch.int32
    assert indptr.numel() == 3
    assert int(indptr[-1]) == indices.numel()
    assert bool((last_page_len >= 1).all()) and bool((last_page_len <= 128).all())


def _run_both(q, k_cache, v_cache, cache_seqlens, block_index, page_size,
               sm_scale, window_left = -1, softcap = None):
    ref = attn_backend.torch_paged_attention(
        q, k_cache, v_cache, cache_seqlens, block_index, page_size,
        sm_scale, window_left = window_left, logits_soft_cap = softcap,
    )
    backend = attn_backend.FlashInferPaged()
    got = backend.plan_and_run(
        q = q, k_cache = k_cache, v_cache = v_cache,
        cache_seqlens = cache_seqlens, block_index = block_index,
        page_size = page_size, sm_scale = sm_scale,
        window_left = window_left, logits_soft_cap = softcap,
    )
    return ref, got
