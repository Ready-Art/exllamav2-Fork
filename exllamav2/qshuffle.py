"""
exllamav2/qshuffle.py

Host-side, bit-exact reimplementation of ``shuffle_kernel`` (cuda/q_matrix.cu).

WHY THIS EXISTS
---------------
Rank-local loading (``ExLlamaV2Linear.tp_load_rank``) reads only one rank's shard of a
tensor. For the **out_features** (column-parallel) case that is straightforward: the shard
is a column slice, and ``make_q_matrix`` shuffles it exactly once on the GPU, which is what
the reference path ends up doing too.

The **in_features** (row-parallel) case is different. ``tp_split_row`` selects a contiguous
range of *packed rows* ``[row_a, row_b)``, and it does so on a tensor that ``load()`` has
**already shuffled in place** (``make_q_matrix`` -> ``QMatrix`` ctor -> ``shuffle_kernel``).
The shard is then built with ``make_q_matrix_split``, which passes ``no_map = true`` and
does **not** shuffle again. Net: shuffle once, then slice rows.

A rank-local loader reads the *raw* rows from disk, so it must apply the shuffle itself
before slicing. It cannot simply let ``make_q_matrix`` do it, because the shuffle is applied
to a whole tensor and the shard is only a row range -- and the shuffle blocks are *not*
aligned to group boundaries (a 5-bit group holds ``qrows * 32 / 5`` packed rows, which is
not a multiple of the 5-row shuffle block), so "shuffle the shard" is not the same
permutation as "shuffle everything, then take the shard".

Hence this module: reproduce the permutation on the host, on the full tensor, then slice.

The permutation is a pure bit-level reordering of uint32 words within each column, so it is
exact -- no floating point is involved anywhere.

Verified against the GPU kernel by ``verify_rowshuffle.py``: byte-identical on every
bit-width group of real layers.
"""

from __future__ import annotations

import numpy as np
import torch


_M32 = np.uint32(0xFFFFFFFF)


def _shuffle_4bit_8(a, p):
    """One packed row. ``shuffle_4bit_8``."""
    qa = a[p].copy()
    qb = np.zeros_like(qa)
    for i in range(4):
        qa0 = qa & np.uint32(0x0F)
        qa1 = (qa & np.uint32(0xF0)) >> np.uint32(4)
        qa >>= np.uint32(8)
        qb |= qa1 << np.uint32(i * 4 + 16)
        qb |= qa0 << np.uint32(i * 4)
    a[p] = qb


def _shuffle_2bit_16(a, p):
    """One packed row. ``shuffle_2bit_16``."""
    qa = a[p].copy()
    qb = np.zeros_like(qa)
    for i in range(8):
        qa0 = qa & np.uint32(0x03)
        qa1 = (qa & np.uint32(0x0C)) >> np.uint32(2)
        qa >>= np.uint32(4)
        qb |= qa1 << np.uint32(i * 2 + 16)
        qb |= qa0 << np.uint32(i * 2)
    a[p] = qb


def _shuffle_6bit_16(a, p):
    """Three packed rows. ``shuffle_6bit_16``."""
    qa = a[p].copy()
    qb = a[p + 1].copy()
    qc = a[p + 2].copy()

    q00 = qa & np.uint32(0b111111)
    q01 = (qa >> np.uint32(6)) & np.uint32(0b111111)
    q02 = (qa >> np.uint32(12)) & np.uint32(0b111111)
    q03 = (qa >> np.uint32(18)) & np.uint32(0b111111)
    q04 = (qa >> np.uint32(24)) & np.uint32(0b111111)
    q05 = ((qa >> np.uint32(30)) & np.uint32(0b11)) | ((qb & np.uint32(0b1111)) << np.uint32(2))
    q06 = (qb >> np.uint32(4)) & np.uint32(0b111111)
    q07 = (qb >> np.uint32(10)) & np.uint32(0b111111)
    q08 = (qb >> np.uint32(16)) & np.uint32(0b111111)
    q09 = (qb >> np.uint32(22)) & np.uint32(0b111111)
    q0a = ((qb >> np.uint32(28)) & np.uint32(0b1111)) | ((qc & np.uint32(0b11)) << np.uint32(4))
    q0b = (qc >> np.uint32(2)) & np.uint32(0b111111)
    q0c = (qc >> np.uint32(8)) & np.uint32(0b111111)
    q0d = (qc >> np.uint32(14)) & np.uint32(0b111111)
    q0e = (qc >> np.uint32(20)) & np.uint32(0b111111)
    q0f = (qc >> np.uint32(26)) & np.uint32(0b111111)

    qa = q00 | (q01 << np.uint32(16)) | (q02 << np.uint32(6)) | (q03 << np.uint32(22))
    qb = q04 | (q05 << np.uint32(16)) | (q06 << np.uint32(6)) | (q07 << np.uint32(22))
    qc = q08 | (q09 << np.uint32(16)) | (q0a << np.uint32(6)) | (q0b << np.uint32(22))

    qa |= (q0c & np.uint32(0b001111)) << np.uint32(12)
    qc |= (q0c & np.uint32(0b110000)) << np.uint32(8)
    qa |= (q0d & np.uint32(0b001111)) << np.uint32(28)
    qc |= (q0d & np.uint32(0b110000)) << np.uint32(24)

    qb |= (q0e & np.uint32(0b001111)) << np.uint32(12)
    qc |= (q0e & np.uint32(0b110000)) << np.uint32(10)
    qb |= (q0f & np.uint32(0b001111)) << np.uint32(28)
    qc |= (q0f & np.uint32(0b110000)) << np.uint32(26)

    a[p] = qa
    a[p + 1] = qb
    a[p + 2] = qc


def _shuffle_5bit_32(a, p):
    """Five packed rows. ``shuffle_5bit_32``."""
    qa = a[p].copy()
    qb = a[p + 1].copy()
    qc = a[p + 2].copy()
    qd = a[p + 3].copy()
    qe = a[p + 4].copy()

    qf = qe >> np.uint32(22)
    qe = (qe << np.uint32(8)) & _M32
    qe |= qd >> np.uint32(24)
    qd = (qd << np.uint32(6)) & _M32
    qd |= qc >> np.uint32(26)
    qc = (qc << np.uint32(4)) & _M32
    qc |= qb >> np.uint32(28)
    qb = (qb << np.uint32(2)) & _M32
    qb |= qa >> np.uint32(30)

    za = np.zeros_like(qa)
    zb = np.zeros_like(qa)
    zc = np.zeros_like(qa)
    zd = np.zeros_like(qa)
    ze = np.zeros_like(qa)

    for i in range(3):
        t0 = qa & np.uint32(0x1F); t1 = (qa & np.uint32(0x3E0)) >> np.uint32(5); qa >>= np.uint32(10)
        za |= (t0 << np.uint32(i * 5)) | (t1 << np.uint32(i * 5 + 16))
        t0 = qb & np.uint32(0x1F); t1 = (qb & np.uint32(0x3E0)) >> np.uint32(5); qb >>= np.uint32(10)
        zb |= (t0 << np.uint32(i * 5)) | (t1 << np.uint32(i * 5 + 16))
        t0 = qc & np.uint32(0x1F); t1 = (qc & np.uint32(0x3E0)) >> np.uint32(5); qc >>= np.uint32(10)
        zc |= (t0 << np.uint32(i * 5)) | (t1 << np.uint32(i * 5 + 16))
        t0 = qd & np.uint32(0x1F); t1 = (qd & np.uint32(0x3E0)) >> np.uint32(5); qd >>= np.uint32(10)
        zd |= (t0 << np.uint32(i * 5)) | (t1 << np.uint32(i * 5 + 16))
        t0 = qe & np.uint32(0x1F); t1 = (qe & np.uint32(0x3E0)) >> np.uint32(5); qe >>= np.uint32(10)
        ze |= (t0 << np.uint32(i * 5)) | (t1 << np.uint32(i * 5 + 16))

    za |= ((qf & np.uint32(0x001)) >> np.uint32(0)) << np.uint32(15)
    zb |= ((qf & np.uint32(0x002)) >> np.uint32(1)) << np.uint32(15)
    zc |= ((qf & np.uint32(0x004)) >> np.uint32(2)) << np.uint32(15)
    zd |= ((qf & np.uint32(0x008)) >> np.uint32(3)) << np.uint32(15)
    ze |= ((qf & np.uint32(0x010)) >> np.uint32(4)) << np.uint32(15)
    za |= ((qf & np.uint32(0x020)) >> np.uint32(5)) << np.uint32(31)
    zb |= ((qf & np.uint32(0x040)) >> np.uint32(6)) << np.uint32(31)
    zc |= ((qf & np.uint32(0x080)) >> np.uint32(7)) << np.uint32(31)
    zd |= ((qf & np.uint32(0x100)) >> np.uint32(8)) << np.uint32(31)
    ze |= ((qf & np.uint32(0x200)) >> np.uint32(9)) << np.uint32(31)

    a[p] = za
    a[p + 1] = zb
    a[p + 2] = zc
    a[p + 3] = zd
    a[p + 4] = ze


def _shuffle_3bit_32(a, p):
    """Three packed rows. ``shuffle_3bit_32``."""
    qa = a[p].copy()
    qb = a[p + 1].copy()
    qc = a[p + 2].copy()

    qd = qc >> np.uint32(26)
    qc = (qc << np.uint32(4)) & _M32
    qc |= qb >> np.uint32(28)
    qb = (qb << np.uint32(2)) & _M32
    qb |= qa >> np.uint32(30)

    za = np.zeros_like(qa)
    zb = np.zeros_like(qa)
    zc = np.zeros_like(qa)

    for i in range(5):
        t0 = qa & np.uint32(0x07); t1 = (qa & np.uint32(0x38)) >> np.uint32(3); qa >>= np.uint32(6)
        za |= (t0 << np.uint32(i * 3)) | (t1 << np.uint32(i * 3 + 16))
        t0 = qb & np.uint32(0x07); t1 = (qb & np.uint32(0x38)) >> np.uint32(3); qb >>= np.uint32(6)
        zb |= (t0 << np.uint32(i * 3)) | (t1 << np.uint32(i * 3 + 16))
        t0 = qc & np.uint32(0x07); t1 = (qc & np.uint32(0x38)) >> np.uint32(3); qc >>= np.uint32(6)
        zc |= (t0 << np.uint32(i * 3)) | (t1 << np.uint32(i * 3 + 16))

    za |= ((qd & np.uint32(0x01)) >> np.uint32(0)) << np.uint32(15)
    zb |= ((qd & np.uint32(0x02)) >> np.uint32(1)) << np.uint32(15)
    zc |= ((qd & np.uint32(0x04)) >> np.uint32(2)) << np.uint32(15)
    za |= ((qd & np.uint32(0x08)) >> np.uint32(3)) << np.uint32(31)
    zb |= ((qd & np.uint32(0x10)) >> np.uint32(4)) << np.uint32(31)
    zc |= ((qd & np.uint32(0x20)) >> np.uint32(5)) << np.uint32(31)

    a[p] = za
    a[p + 1] = zb
    a[p + 2] = zc


def group_row_boundaries(q_groups, height: int):
    """
    Reproduce the row accounting at the top of ``QMatrix::QMatrix``.

    IMPORTANT -- ``height`` is the **value** count (``in_features``), not the packed-row count.

    The C++ loop keeps a single cursor ``row`` that it advances by ``rows``, where ``rows`` is
    ``qrows * 32 / bits`` -- a count of *packed* rows -- for every group except the last, and
    ``height - row`` for the last. That is only self-consistent if ``height`` is the value
    count: the last group's packed-row count is then derived as a remainder. Passing the packed
    row count instead makes the last group's extent wrong and the whole cascade shifts.

    :param q_groups:
        The ``q_groups`` tensor, a flat list of (bits, packed_row_start) pairs.

    :param height:
        ``in_features`` for this layer (the value count, i.e. ``q_groups``'s logical height).

    :return:
        (rows_8, rows_6, rows_5, rows_4, rows_3, rows_2) as *cumulative* boundaries,
        exactly as the C++ leaves them after the ``rows_6 += rows_8; ...`` cascade.
        The 8-bit region is ``[0, rows_8)``, the 6-bit region ``[rows_8, rows_6)``, etc.
    """
    gr = q_groups.tolist() if isinstance(q_groups, torch.Tensor) else list(q_groups)
    ngroups = len(gr) // 2

    r8 = r6 = r5 = r4 = r3 = r2 = 0
    row = 0
    for i in range(ngroups):
        bits = gr[i * 2]
        if i < ngroups - 1:
            qrows = gr[i * 2 + 3] - gr[i * 2 + 1]
            rows = qrows * 32 // bits
        else:
            rows = height - row
        if bits == 8: r8 += rows
        if bits == 6: r6 += rows
        if bits == 5: r5 += rows
        if bits == 4: r4 += rows
        if bits == 3: r3 += rows
        if bits == 2: r2 += rows
        row += rows

    r6 += r8
    r5 += r6
    r4 += r5
    r3 += r4
    r2 += r3
    return r8, r6, r5, r4, r3, r2


def shuffle_q_weight_host(
    q_weight: torch.Tensor,
    q_groups: torch.Tensor,
    in_features: int
) -> torch.Tensor:
    """
    Apply ``shuffle_kernel`` to a packed EXL2 weight tensor on the host, in place.

    ``q_weight`` must be ``(packed_rows, out_features)`` int32 and on the CPU. The result is
    byte-identical to what ``make_q_matrix`` leaves in ``q_weight`` after its ``QMatrix``
    constructor runs ``shuffle_kernel`` on the GPU.

    :param q_weight:
        Packed weight tensor, int32, CPU.

    :param q_groups:
        The ``q_groups`` tensor for this layer (bits / packed-row-start pairs).

    :param in_features:
        The layer's input width. This is the ``height`` the C++ accounting uses -- see
        ``group_row_boundaries``.

    :return:
        The same tensor object, shuffled in place.
    """
    assert q_weight.device.type == "cpu", "shuffle_q_weight_host needs a CPU tensor"
    assert q_weight.dtype == torch.int32, "shuffle_q_weight_host needs int32"

    packed_rows, width = q_weight.shape
    r8, r6, r5, r4, r3, r2 = group_row_boundaries(q_groups, in_features)

    if r8 == in_features:
        # All 8-bit: shuffle_8bit_4 is a no-op, so the whole tensor is untouched.
        return q_weight

    a = q_weight.numpy().view(np.uint32)

    # 8-bit region: no-op, but the row cursor still advances.
    k = 0
    p = 0
    while k < r8:
        p += 1
        k += 4

    while k < r6:
        _shuffle_6bit_16(a, p)
        p += 3
        k += 16

    while k < r5:
        _shuffle_5bit_32(a, p)
        p += 5
        k += 32

    while k < r4:
        _shuffle_4bit_8(a, p)
        p += 1
        k += 8

    while k < r3:
        _shuffle_3bit_32(a, p)
        p += 3
        k += 32

    while k < r2:
        _shuffle_2bit_16(a, p)
        p += 1
        k += 16

    return q_weight
