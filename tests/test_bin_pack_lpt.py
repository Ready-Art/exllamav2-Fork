"""
Regression test for Stage K bug 9: `bin_pack()` must sort largest-first (LPT) before packing.

Without the sort, Gemma-4 attention splits 88/44 on a homogeneous 2-GPU box because the caller's insertion order
is q, k, v, o with sizes 44, 22, 22, 44 — both big tensors land on device 0.

Run with:  python tests/test_bin_pack_lpt.py
No CUDA needed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2.conversion.schedule import bin_pack


def check(name, cond, detail = ""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        raise AssertionError(name)


def sizes(buckets, size_fn):
    return [sum(size_fn(it) for it in b) for b in buckets]


def test_attn_balances():
    """The exact case that produced the 88/44 split: q, k, v, o = 44, 22, 22, 44."""
    items = [("q", 44.0), ("k", 22.0), ("v", 22.0), ("o", 44.0)]
    b = bin_pack(items, [0, 1], [2.0, 2.0], lambda t: t[1])
    s = sizes(b, lambda t: t[1])
    check("attn splits 66/66", s == [66.0, 66.0], str(s))
    check("attn max/min ratio is 1.0", max(s) / min(s) == 1.0, f"{max(s) / min(s):.2f}")


def test_insertion_order_does_not_matter():
    """The same multiset in any order must produce the same bucket sizes."""
    base = [("a", 44.0), ("b", 22.0), ("c", 22.0), ("d", 44.0)]
    orders = [
        base,
        list(reversed(base)),
        [base[1], base[3], base[0], base[2]],
        [base[2], base[0], base[3], base[1]],
    ]
    results = []
    for o in orders:
        b = bin_pack(o, [0, 1], [2.0, 2.0], lambda t: t[1])
        results.append(sorted(sizes(b, lambda t: t[1])))
    check("all insertion orders give the same split",
          all(r == results[0] for r in results), str(results))


def test_equal_three_way():
    """Three equal MLP tensors on two devices: 2:1 is inherent, but the split must be the tightest possible."""
    items = [("gate", 74.3), ("up", 74.3), ("down", 74.3)]
    b = bin_pack(items, [0, 1], [2.0, 2.0], lambda t: t[1])
    s = sorted(sizes(b, lambda t: t[1]))
    check("mlp 3-on-2 gives 148.6/74.3", s == [74.3, 148.6], str(s))


def test_uneven_ratios():
    """A 1:3 ratio must put roughly 3x the work on the faster device."""
    items = [("a", 10.0), ("b", 10.0), ("c", 10.0), ("d", 10.0)]
    b = bin_pack(items, [0, 1], [1.0, 3.0], lambda t: t[1])
    s = sizes(b, lambda t: t[1])
    check("1:3 ratio gives 10/30", s == [10.0, 30.0], str(s))


def test_single_device():
    items = [("a", 5.0), ("b", 7.0)]
    b = bin_pack(items, [0], None, lambda t: t[1])
    check("single device takes everything", sizes(b, lambda t: t[1]) == [12.0], str(sizes(b, lambda t: t[1])))


def test_empty():
    check("no devices -> no buckets", bin_pack([("a", 1.0)], [], None, lambda t: t[1]) == [])
    b = bin_pack([], [0, 1], None, lambda t: t[1])
    check("no items -> empty buckets", b == [[], []], str(b))


if __name__ == "__main__":
    test_attn_balances()
    test_insertion_order_does_not_matter()
    test_equal_three_way()
    test_uneven_ratios()
    test_single_device()
    test_empty()
    print("\nall bin_pack LPT tests passed")
