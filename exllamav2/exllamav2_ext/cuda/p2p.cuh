#ifndef P2P_CUH
#define P2P_CUH

#include <cuda_runtime.h>
#include <cstdint>

// Low-latency PCIe P2P collectives for the MP tensor-parallel decode path.
// NCCL small-message latency (~70 us/op over PXB) dominates base decode; these
// kernels do one-sided P2P writes into IPC-shared peer arenas + system-scope
// flag spins (~10-20 us). Graph-capturable: all state (peer pointers, flags,
// round counter) lives in the fixed arena; per-call tensor pointers are the
// same graph-pool-stable addresses the captured NCCL ops already rely on.
//
// Arena layout (bytes from base):
//   [0, nslots*slot_bytes)                  allgather output slots
//   [ar_off, ar_off + world*slot_bytes)     allreduce contribution scratch
//   [flags_off, flags_off + world*8)        u64 flags[peer] = data-ready round
//   [acks_off, acks_off + world*8)          u64 acks[peer]  = collective-completed round
//   [counter_off, +8)                       u64 round counter (device-incremented)
//   [ptrs_off, +world*8)                    i64 peer arena base pointers
//
// Two-phase rendezvous (BUG 116): phase 1 flags = "my round-r data is in your
// buffer"; phase 2 acks = "I have finished round r". No rank returns from round r
// until EVERY rank acks r, so a fast rank under graph replay can never lap a slow
// rank and overwrite a slot/scratch the slow rank has not consumed yet. (The
// single-phase protocol corrupted graphed decode: a replay launches ~240
// collectives at once, ranks skew, and one-sided writes clobbered pending data --
// eager execution never hit it because host dispatch paced the ranks.)

void p2p_allgather_cuda
(
    char* arena,
    const int64_t* peer_ptrs,
    const char* shard,
    int slot_off,
    int shard_bytes,
    int world,
    int rank,
    int flags_off,
    int acks_off,
    int counter_off,
    cudaStream_t stream
);

// dtype: 0 = half, 1 = float. In-place safe (input consumed before output written).
void p2p_allreduce_cuda
(
    char* arena,
    const int64_t* peer_ptrs,
    void* data,
    int ar_off,
    int ar_slot_elems,
    int elems,
    int world,
    int rank,
    int flags_off,
    int acks_off,
    int counter_off,
    int dtype,
    cudaStream_t stream
);

#endif