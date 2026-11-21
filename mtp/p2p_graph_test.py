"""Minimal definitive localizer for the in-graph P2P corruption (BUG 116).

Runs the P2P collectives EAGER vs inside a CUDA graph, on known patterns, and
reports the FIRST divergence (which collective, which replay, got vs expected).
No model load -- just a NCCL process group + P2PComm + graph capture. ~10 s.

Usage (from ~/exl2work, with the venv active):
    torchrun --nproc_per_node=4 exllamav2-Fork/mtp/p2p_graph_test.py

Interpretation:
  * EAGER fails           -> the P2P kernel/protocol itself is wrong (rare; probe passed).
  * EAGER ok, GRAPH fails -> graph-replay interaction; the printed site + values say
                             whether it is stale data, wrong-rank data, or zeros, which
                             pins the mechanism (counter desync / slot clobber / atomics).
  * All PASS              -> the kernel is graph-safe in isolation; the corruption is in
                             the model integration (slot/counter/state across the real
                             decode/verify/bonus graph sequence), not the kernel.

Only rank 0 prints the verdict; every rank checks its own outputs and reports failures.
"""
import os
import torch
import torch.distributed as dist


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    from exllamav2.p2p_comm import P2PComm
    p2p = P2PComm(rank, world)
    dist.barrier()
    if not p2p.enabled:
        log(rank, f"[p2p-test] P2P NOT enabled on rank {rank}; aborting test")
        dist.destroy_process_group()
        return
    log(rank, f"[p2p-test] P2P enabled, world={world}")

    n = 64
    fails = []

    def expect_ag(val_per_rank):
        return torch.cat([torch.full((n,), float(p + 1) * val_per_rank,
                                     dtype=torch.half, device=dev) for p in range(world)])

    # ---- 1) EAGER allgather / allreduce ----
    shard = torch.full((n,), float(rank + 1), dtype=torch.half, device=dev)
    out = p2p.allgather(shard, ("eager_ag",), (n * world,))
    torch.cuda.synchronize(dev)
    if out is None or not torch.equal(out, expect_ag(1.0)):
        fails.append(("EAGER_AG", -1, None if out is None else out[:8].tolist(),
                      expect_ag(1.0)[:8].tolist()))
    t = torch.ones(n, dtype=torch.float32, device=dev)
    p2p.allreduce(t)
    torch.cuda.synchronize(dev)
    if not torch.equal(t, torch.full((n,), float(world), dtype=torch.float32, device=dev)):
        fails.append(("EAGER_AR", -1, t[:4].tolist(), float(world)))
    log(rank, f"[p2p-test] eager: {'FAIL ' + str(fails) if fails else 'PASS'}")

    # ---- 1b) MULTI-ROW allgather (the MTP verify shape) ----
    # The 1-D patterns above cannot catch a rank-major vs row-major slot layout
    # bug: for rows == 1 the two layouts coincide. A (rows, width) shard must
    # gather to out[i, p*width:(p+1)*width] == shard_p[i] (NCCL dim=1 semantics).
    if not fails:
        R, W = 7, 64
        mr_shard = torch.empty((R, W), dtype=torch.half, device=dev)
        for i in range(R):
            mr_shard[i].fill_(float((rank + 1) * 100 + i))

        def mr_expect(mult):
            e = torch.empty((R, W * world), dtype=torch.half, device=dev)
            for p in range(world):
                for i in range(R):
                    e[i, p * W:(p + 1) * W].fill_(float((p + 1) * 100 + i) * mult)
            return e

        out = p2p.allgather(mr_shard, ("mr_eager",), (R, W * world))
        torch.cuda.synchronize(dev)
        if out is None or not torch.equal(out, mr_expect(1.0)):
            fails.append(("MULTIROW_EAGER", -1,
                          None if out is None else out[0, :8].tolist(),
                          mr_expect(1.0)[0, :8].tolist()))
        else:
            # Graphed multi-row: static shard, replays with changing values.
            def mr_step():
                p2p.barrier()
                return p2p._ag_raw(mr_shard, ("mr_graph",), (R, W * world))
            p2p.begin_graph()
            for _ in range(3):
                mr_step()
            torch.cuda.synchronize(dev)
            gg = torch.cuda.CUDAGraph()
            ss = torch.cuda.Stream(device=dev)
            ss.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(ss):
                mr_view = mr_step()
            torch.cuda.current_stream(dev).wait_stream(ss)
            torch.cuda.synchronize(dev)
            with torch.cuda.graph(gg):
                mr_view = mr_step()
            p2p.end_graph()
            for it in range(5):
                mult = float(it + 1)
                for i in range(R):
                    mr_shard[i].fill_(float((rank + 1) * 100 + i) * mult)
                gg.replay()
                torch.cuda.synchronize(dev)
                if mr_view is None or not torch.equal(mr_view, mr_expect(mult)):
                    fails.append(("MULTIROW_GRAPH", it,
                                  None if mr_view is None else mr_view[0, :8].tolist(),
                                  mr_expect(mult)[0, :8].tolist()))
                    break
        if fails:
            log(rank, f"[p2p-test] multi-row: FAIL {fails[-1]}")
        else:
            log(rank, "[p2p-test] multi-row: PASS (eager + 5 graph replays, 7x64 shards)")

    # ---- 2) GRAPHED allgather + allreduce, many replays, changing inputs ----
    # Static input tensors (graph-safe). Capture a small "step" of several collectives
    # with distinct keys (exercises slot allocation + alternation + counter evolution).
    g_shards = [torch.full((n,), float(rank + 1), dtype=torch.half, device=dev)
                for _ in range(3)]
    g_ar = torch.ones(n, dtype=torch.float32, device=dev)
    g_views = []

    def step():
        p2p.barrier()
        g_views.clear()
        for i, sh in enumerate(g_shards):
            g_views.append(p2p._ag_raw(sh, (f"graph_ag_{i}",), (n * world,)))
        p2p._ar_raw(g_ar)

    # warmup (eager) so allocator/JIT settle, like the real capture path
    p2p.begin_graph()
    for _ in range(3):
        step()
    torch.cuda.synchronize(dev)

    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream(dev).wait_stream(s)
    torch.cuda.synchronize(dev)
    try:
        with torch.cuda.graph(graph):
            step()
    except Exception as e:
        log(rank, f"[p2p-test] GRAPH CAPTURE FAILED: {type(e).__name__}: {e}")
        dist.destroy_process_group()
        return
    p2p.end_graph()

    REPLAYS = 20
    for it in range(REPLAYS):
        mult = float(it + 1)
        for sh in g_shards:
            sh.fill_(float(rank + 1) * mult)
        g_ar.fill_(mult)
        graph.replay()
        torch.cuda.synchronize(dev)
        for i, v in enumerate(g_views):
            exp = expect_ag(mult)
            if v is None or not torch.equal(v, exp):
                fails.append((f"GRAPH_AG_{i}", it,
                              None if v is None else v[:8].tolist(), exp[:8].tolist()))
        exp_ar = float(world) * mult
        if not torch.equal(g_ar, torch.full((n,), exp_ar, dtype=torch.float32, device=dev)):
            fails.append(("GRAPH_AR", it, g_ar[:4].tolist(), exp_ar))
        if fails:
            break

    if fails:
        site, it, got, exp = fails[0]
        log(rank, f"[p2p-test] GRAPH FAIL @ {site} replay {it}: got {got} expect {exp}")
    else:
        log(rank, f"[p2p-test] graph: PASS ({REPLAYS} replays, 3 AG + 1 AR each)")

    # ---- 3) MULTIPLE graphs of DIFFERENT sizes, interleaved replays ----
    # Mimics the real decode/verify/bonus sequence: several graphs with different
    # collective counts share ONE global round counter. A counter/slot desync across
    # graph boundaries shows up here even when a single graph is clean.
    if not fails:
        # Fresh slot state for this phase (the eager/single-graph phases consumed slots).
        p2p._slots.clear()
        p2p._alt.clear()
        p2p._next_slot = 0
        specs = [("decode", 3), ("verify", 4), ("bonus", 2)]  # (name, #AG per step)
        graphs = {}
        views = {}
        shards = {}
        for name, k in specs:
            sh = [torch.full((n,), float(rank + 1), dtype=torch.half, device=dev)
                  for _ in range(k)]
            shards[name] = sh
            vv = []

            def make_step(sh=sh, vv=vv, name=name):
                def step():
                    p2p.barrier()
                    vv.clear()
                    for i, s in enumerate(sh):
                        vv.append(p2p._ag_raw(s, (f"mg_{name}_{i}",), (n * world,)))
                return step

            st = make_step()
            p2p.begin_graph()
            for _ in range(3):
                st()
            torch.cuda.synchronize(dev)
            gg = torch.cuda.CUDAGraph()
            ss = torch.cuda.Stream(device=dev)
            ss.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(ss):
                st()
            torch.cuda.current_stream(dev).wait_stream(ss)
            torch.cuda.synchronize(dev)
            with torch.cuda.graph(gg):
                st()
            p2p.end_graph()
            graphs[name] = gg
            views[name] = vv

        if any(v is None for vv in views.values() for v in vv):
            log(rank, "[p2p-test] MULTI-GRAPH setup: None view (slot exhaustion) — test invalid")
            fails.append(("MULTI_SETUP", -1, "None view", "slot exhaustion"))

        seq = ["decode", "verify", "decode", "bonus", "verify", "decode"]
        for it in range(12):
            mult = float(it + 1)
            for name in seq:
                for s in shards[name]:
                    s.fill_(float(rank + 1) * mult)
                graphs[name].replay()
                torch.cuda.synchronize(dev)
                for i, v in enumerate(views[name]):
                    exp = expect_ag(mult)
                    if v is None or not torch.equal(v, exp):
                        fails.append((f"MULTI_{name}_AG{i}", it,
                                      None if v is None else v[:8].tolist(),
                                      exp[:8].tolist()))
                if fails:
                    break
            if fails:
                break
        if fails:
            site, it, got, exp = fails[0]
            log(rank, f"[p2p-test] MULTI-GRAPH FAIL @ {site} iter {it}: got {got} expect {exp}")
        else:
            log(rank, "[p2p-test] multi-graph: PASS (3 graphs, interleaved, 12 iters)")

    # ---- verdict ----
    local_fail = torch.tensor([1 if fails else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(local_fail, op=dist.ReduceOp.MAX)
    if local_fail.item():
        log(rank, "[p2p-test] VERDICT: FAIL (see first divergence above)")
    else:
        log(rank, "[p2p-test] VERDICT: ALL PASS — kernel is graph-safe in isolation; "
                  "corruption is in the model integration, not the kernel")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()