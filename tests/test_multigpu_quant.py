"""
Tests for the multi-GPU quantization plumbing. quant-plan.md §6.

Run with:  python tests/test_multigpu_quant.py
Requires CUDA (the stager and the schedule tests allocate device tensors).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav2.conversion.devices import (
    arch_prior_speed, resolve_devices, AutoSplit, split_sizes, auto_split
)
from exllamav2.conversion.schedule import bin_pack, run_on_devices, quantize_tensor_set
from exllamav2.conversion.stager import WeightStager, free_stage_bufs
from exllamav2.conversion.calib import calibration_row_shards, merge_hessians


def check(name, cond, detail = ""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        raise AssertionError(name)


def test_arch_prior():
    n = torch.cuda.device_count()
    for d in range(n):
        p = arch_prior_speed(d)
        check(f"arch_prior_speed({d}) > 0", p > 0, f"{p}")
    # Ampere (8.6) must be 2.0, Ada (8.9) 4.0
    if n:
        cap = torch.cuda.get_device_capability(0)
        p = arch_prior_speed(0)
        if cap[0] == 8 and cap[1] == 6: check("sm_86 prior == 2.0", p == 2.0, f"{p}")
        if cap[0] == 8 and cap[1] == 9: check("sm_89 prior == 4.0", p == 4.0, f"{p}")
        if cap[0] >= 9: check("sm_90+ prior == 5.0", p == 5.0, f"{p}")


def test_resolve_devices():
    n = torch.cuda.device_count()
    check("resolve_devices(None) == all", resolve_devices(None) == list(range(n)))
    if n >= 2:
        check("resolve_devices([1,0])", resolve_devices([1, 0]) == [1, 0])
        check("resolve_devices('cuda:1')", resolve_devices("cuda:1") == [1])
        check("resolve_devices dedups", resolve_devices([0, 0, 1]) == [0, 1])
    try:
        resolve_devices([n + 5])
        check("resolve_devices out of range raises", False)
    except ValueError:
        check("resolve_devices out of range raises", True)


def test_split_sizes():
    check("split_sizes even", split_sizes(100, [0, 1]) == [50, 50])
    check("split_sizes ratio 1:3", split_sizes(100, [0, 1], [1.0, 3.0]) == [25, 75])
    check("split_sizes sums", sum(split_sizes(97, [0, 1, 2])) == 97)
    check("split_sizes sums ratio", sum(split_sizes(97, [0, 1, 2], [1, 2, 3])) == 97)
    # Too many devices for the work: negatives folded back, never negative output
    s = split_sizes(2, [0, 1, 2, 3])
    check("split_sizes no negatives", all(x >= 0 for x in s), str(s))
    check("split_sizes still sums", sum(s) == 2, str(s))


def test_auto_split():
    a = AutoSplit()
    check("no measurement -> prior", a.ratios("k", [0]) == [arch_prior_speed(0)])
    a.report("k", 0, 100.0, 1.0)
    check("one device measured", a.has_measured("k", [0]))
    check("measured ratio", abs(a.ratios("k", [0])[0] - 100.0) < 1e-6)
    a.report("k", 0, 100.0, 0.01)     # too short, must be discarded
    check("short measurement discarded", abs(a.ratios("k", [0])[0] - 100.0) < 1e-6)
    if torch.cuda.device_count() >= 2:
        check("partial measurement -> prior", a.ratios("k", [0, 1]) == [arch_prior_speed(0), arch_prior_speed(1)])
        a.report("k", 1, 50.0, 1.0)
        check("full measurement -> measured", a.ratios("k", [0, 1]) == [100.0, 50.0])


def test_bin_pack():
    items = [10, 20, 30, 40]
    b = bin_pack(items, [0, 1], None, lambda x: x)
    check("bin_pack covers all", sorted(sum(b, [])) == sorted(items), str(b))
    check("bin_pack balanced", abs(sum(b[0]) - sum(b[1])) <= 40, str(b))
    # Weighted: device 1 twice as fast should get roughly twice the work
    b = bin_pack(items, [0, 1], [1.0, 2.0], lambda x: x)
    check("bin_pack weighted covers all", sorted(sum(b, [])) == sorted(items), str(b))
    check("bin_pack weighted favours fast device", sum(b[1]) >= sum(b[0]), str(b))


def test_calibration_row_shards():
    s = calibration_row_shards(10, [0, 1], None)
    check("row shards cover all", sorted(x for r in s for x in r) == list(range(10)), str([list(r) for r in s]))
    check("row shards contiguous", [list(r) for r in s] == [list(range(0, 5)), list(range(5, 10))])
    s = calibration_row_shards(10, [0, 1], [1.0, 3.0])
    check("row shards weighted", [len(r) for r in s] == [3, 7], str([len(r) for r in s]))
    s = calibration_row_shards(10, [0, 1, 2], None)
    check("row shards sum", sum(len(r) for r in s) == 10)


def test_stager_identity():
    """
    The critical test for M5: staging must produce a bit-identical tensor to the old
    synchronous .cpu().T.clone().float().contiguous() path. If it does not, M5 is not a
    pure wall-clock change and every A/B against it is invalid.
    """
    if torch.cuda.device_count() == 0:
        print("SKIP  stager identity (no CUDA)")
        return

    dev = torch.device("cuda:0")
    for shape in [(256, 512), (1, 7), (4096, 4096), (33, 17)]:
        for dtype in [torch.half, torch.bfloat16, torch.float]:
            w = torch.randn(shape, device = dev, dtype = dtype)

            # Old path
            ref = w.cpu().T.clone().float().contiguous()

            # New path
            st = WeightStager(dev)
            got = st.get(0, w)

            check(f"stager identity {shape} {dtype}",
                  ref.shape == got.shape and ref.dtype == got.dtype and torch.equal(ref, got),
                  f"{ref.shape}/{got.shape} {ref.dtype}/{got.dtype} maxdiff={(ref - got).abs().max().item()}")

    # Prefetch one ahead, then consume in order: must still match
    ws = [torch.randn((128, 64), device = dev, dtype = torch.half) for _ in range(4)]
    st = WeightStager(dev)
    st.prefetch(0, ws[0])
    outs = []
    for t in range(4):
        if t + 1 < 4: st.prefetch(t + 1, ws[t + 1])
        outs.append(st.get(t, ws[t]))
    ok = all(torch.equal(outs[t], ws[t].cpu().T.clone().float().contiguous()) for t in range(4))
    check("stager prefetch identity", ok)
    free_stage_bufs()


def test_stager_host_input():
    if torch.cuda.device_count() == 0:
        print("SKIP  stager host input (no CUDA)")
        return
    w = torch.randn((64, 32), dtype = torch.half)          # already on host
    st = WeightStager(0)
    got = st.get(0, w)
    ref = w.cpu().T.clone().float().contiguous()
    check("stager host-input identity", torch.equal(got, ref))


def test_run_on_devices_error_propagation():
    """
    A worker that raises must abort the job. This is R12: a silently dead worker leaves
    linears unquantized and compiles a broken model.
    """
    if torch.cuda.device_count() < 2:
        print("SKIP  worker error propagation (needs 2 GPUs)")
        return

    def boom(dev, bucket):
        raise RuntimeError("deliberate worker failure")

    try:
        run_on_devices([[1], [2]], [0, 1], boom)
        check("worker exception re-raised", False)
    except RuntimeError as e:
        check("worker exception re-raised", "deliberate" in str(e))


def test_quantize_tensor_set_serial_matches():
    """
    With one device the scheduler must behave exactly like a serial loop, in order.
    """
    order = []
    quantize_tensor_set([1, 2, 3], [0], "test", lambda t, d: order.append((t, d)), lambda t: t)
    check("serial order preserved", order == [(1, 0), (2, 0), (3, 0)], str(order))


def test_quantize_tensor_set_parallel():
    if torch.cuda.device_count() < 2:
        print("SKIP  parallel dispatch (needs 2 GPUs)")
        return
    seen = []
    lock = __import__("threading").Lock()
    def run_one(t, d):
        with lock: seen.append((t, d))
        torch.cuda.synchronize(d)
    quantize_tensor_set([1, 2, 3, 4], [0, 1], "test", run_one, lambda t: t)
    check("parallel covers all tasks", sorted(t for t, d in seen) == [1, 2, 3, 4], str(sorted(seen)))
    check("parallel used both devices", len(set(d for t, d in seen)) == 2, str(sorted(set(d for t, d in seen))))


def test_merge_hessians():
    class Q:
        def __init__(self, h, nb):
            self.hessian = h
            self.num_batches = nb
    a = {"x": Q(torch.ones(4, 4), 2)}
    b = {"x": Q(torch.ones(4, 4) * 3, 5)}
    merge_hessians([a, b])
    check("merge sums hessian", torch.allclose(a["x"].hessian, torch.ones(4, 4) * 4))
    check("merge sums batches", a["x"].num_batches == 7)
    check("merge drops source", b["x"].hessian is None)


def test_shard_primary_uses_real_quantizers():
    """
    The primary shard must accumulate into the *real* quantizers, not clones.

    `shard_calibration_rows()` calls `row_fn(devices[0], None, primary_rows)` — it passes None for
    the module because it holds no reference to it. A `row_fn` that decides "is this the primary?"
    with `active is module` therefore sees `None is module` == False and hands the primary shard
    *cloned* quantizers. The primary's Hessian contributions are then merged into a throwaway dict and
    silently lost, which corrupts every tensor on the first device.

    This test drives the real `shard_calibration_rows` with a row_fn that mirrors measure.py's
    contract, and asserts the primary's Hessian actually lands in the caller's quantizer.
    """
    from exllamav2.conversion.calib import shard_calibration_rows

    if torch.cuda.device_count() < 2:
        print("SKIP  primary shard quantizers (needs 2 GPUs)")
        return

    class Q:
        def __init__(self):
            self.hessian = None
            self.num_batches = 0

    module = object()                       # stand-in for the loaded module
    quantizers = {"x": Q()}
    devices = [0, 1]
    rows_total = 4

    def row_fn(device_idx, active, rows):
        # Mirrors measure.py: None means "the primary, i.e. the already-loaded module".
        if active is None:
            active = module
        qz = quantizers if active is module else {"x": Q()}
        for _ in rows:
            qz["x"].num_batches += 1
            h = torch.ones(2, 2, device = f"cuda:{device_idx}")
            qz["x"].hessian = h if qz["x"].hessian is None else qz["x"].hessian + h
        return qz, [None] * len(rows), None

    # Drive it the way measure.py does, with a fake hidden_states of the right length.
    q_out, ts_out, _ = shard_calibration_rows([0] * rows_total, [object()], devices, None, row_fn)

    check("primary shard kept its own quantizers", q_out is quantizers)
    check("all rows accounted for", quantizers["x"].num_batches == rows_total,
          f"{quantizers['x'].num_batches} != {rows_total}")


if __name__ == "__main__":
    test_arch_prior()
    test_resolve_devices()
    test_split_sizes()
    test_auto_split()
    test_bin_pack()
    test_calibration_row_shards()
    test_stager_identity()
    test_stager_host_input()
    test_run_on_devices_error_propagation()
    test_quantize_tensor_set_serial_matches()
    test_quantize_tensor_set_parallel()
    test_merge_hessians()
    test_shard_primary_uses_real_quantizers()
    print("\nAll multi-GPU tests passed.")


def test_run_rows_returns_all_rows():
    """
    run_rows() must return one target state per row, not one per call.

    Regression for a bug introduced while fixing the dedented-body bug: the `return`
    ended up indented *inside* the row loop, so run_rows() returned after the first
    row. shard_calibration_rows() then saw a 1-element target_states list and raised
        IndexError: list index out of range
    at `ordered[r] = ts_primary[i]`.

    This is the second time a pure-indentation mistake in this function produced a
    runtime failure that py_compile accepted, so it gets an explicit arity test.
    """
    from exllamav2.conversion.calib import shard_calibration_rows

    if torch.cuda.device_count() < 2:
        print("SKIP  run_rows arity (needs 2 GPUs)")
        return

    class Q:
        def __init__(self):
            self.hessian = None
            self.num_batches = 0

    quantizers = {"x": Q()}
    module = object()
    devices = [0, 1]
    n_rows = 6

    def row_fn(device_idx, active, rows):
        if active is None:
            active = module
        qz = quantizers if active is module else {"x": Q()}
        ts = []
        for _ in rows:
            qz["x"].num_batches += 1
            h = torch.ones(2, 2, device = f"cuda:{device_idx}")
            qz["x"].hessian = h if qz["x"].hessian is None else qz["x"].hessian + h
            ts.append(torch.zeros(1))
        return qz, ts, None

    q_out, ts_out, _ = shard_calibration_rows([0] * n_rows, [object()], devices, None, row_fn)

    check("run_rows returns one state per row", len(ts_out) == n_rows,
          f"{len(ts_out)} != {n_rows}")
    check("all rows accounted for", quantizers["x"].num_batches == n_rows,
          f"{quantizers['x'].num_batches} != {n_rows}")


if __name__ == "__main__":
    test_run_rows_returns_all_rows()
