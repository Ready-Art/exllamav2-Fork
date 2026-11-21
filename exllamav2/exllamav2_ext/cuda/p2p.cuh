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
// Two-phase rendezvous (BUG 116), deferred-ack form: phase 1 flags = "my round-r
// data is in your buffer" (every op waits for all peers' flags); phase 2 acks =
// "I have finished round r" (every op WRITES its ack, but only eager ops spin for
// peer acks per-op). Graphed ops rely on p2p_barrier_kernel at the graph start
// (waits all peers' acks of the previous round before any write) plus the per-op
// flag waits + double-buffered slots + stream-ordered consumers for intra-replay
// safety. (The original single-phase protocol corrupted graphed decode: a replay
// launches ~240 collectives at once, ranks skew, and one-sided writes clobbered
// pending data -- eager execution never hit it because host dispatch paced the
// ranks. The allreduce keeps full per-op acks: its scratch is shared, not slotted.)

// row_bytes = one row of the caller's (rows, width) shard (shard_bytes for a
// 1-D shard). The output slot uses the NCCL all_gather(dim=1) layout: row i =
// [rank0_row_i | ... | rank(W-1)_row_i], rank r's row i at (i*world+r)*row_bytes.
void p2p_allgather_cuda
(
    char* arena,
    const int64_t* peer_ptrs,
    const char* shard,
    int slot_off,
    int shard_bytes,
    int row_bytes,
    int world,
    int rank,
    int flags_off,
    int acks_off,
    int counter_off,
    int wait_ack,
    cudaStream_t stream
);

// Graph-start barrier: wait for all peers' acks of the current round counter value
// before the replay's first write. Captured as the graph's first P2P node.
void p2p_barrier_cuda
(
    char* arena,
    const int64_t* peer_ptrs,
    int acks_off,
    int counter_off,
    int world,
    int rank,
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