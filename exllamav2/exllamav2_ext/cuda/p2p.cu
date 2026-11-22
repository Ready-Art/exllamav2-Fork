#include "p2p.cuh"
#include <cuda_fp16.h>

// One block, 256 threads. Small messages only (<= 1 MB): a single block keeps the
// flag protocol trivial and the launch overhead minimal. All ranks run the same
// collective sequence in lockstep on one stream; the device-side round counter
// makes the protocol graph-replay-safe (no host state, no ABA).

#define P2P_THREADS 256

__device__ __forceinline__ float p2p_to_float(half v) { return __half2float(v); }
__device__ __forceinline__ float p2p_to_float(float v) { return v; }
__device__ __forceinline__ half p2p_from_float(float v, half*) { return __float2half(v); }
__device__ __forceinline__ float p2p_from_float(float v, float*) { return v; }

__global__ void p2p_allgather_kernel
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
    int wait_ack
)
{
    __shared__ unsigned long long round;
    if (threadIdx.x == 0)
        round = atomicAdd((unsigned long long*)(arena + counter_off), 1ULL) + 1ULL;
    __syncthreads();
    unsigned long long r = round;

    // One-sided writes: my shard into every peer's output slot (self included),
    // in the NCCL all_gather(dim=1) layout: slot row i is the interleaving
    // [rank0_row_i | rank1_row_i | ... | rank(W-1)_row_i], so rank r's row i
    // lands at (i*world + r)*row_bytes. A rank-major block layout (rank r's
    // whole shard at r*shard_bytes) only coincides with this for rows == 1:
    // decode was coherent, but the MTP verify window (rows = W+1) reshaped the
    // slot to (rows, world*width) and read every row scrambled -> gibberish.
    int rows = shard_bytes / row_bytes;
    int n16 = row_bytes / 16;
    for (int p = 0; p < world; p++)
    {
        char* peer = (char*)peer_ptrs[p] + slot_off;
        for (int i = 0; i < rows; i++)
        {
            char* dst = peer + (i * world + rank) * row_bytes;
            const char* src = shard + i * row_bytes;
            for (int j = threadIdx.x; j < n16; j += blockDim.x)
                ((int4*)dst)[j] = ((const int4*)src)[j];
            for (int t = n16 * 16 + threadIdx.x; t < row_bytes; t += blockDim.x)
                dst[t] = src[t];
        }
    }
    // Fence every thread's writes, THEN barrier, THEN signal: thread 0's flag
    // must not outrun the other threads' system-scope visibility.
    __threadfence_system();
    __syncthreads();

    // Warp-parallel rendezvous (stage 2): one lane per peer issues the flag/ack PCIe
    // atomics CONCURRENTLY instead of a serial thread-0 chain (world flag writes +
    // world spins + world ack writes were ~3*world serial system-scope round trips).
    // Semantics unchanged: every lane publishes its flag then spins its peer's flag;
    // the block barrier proves ALL peers' data present before any ack is written
    // (the deferred-ack contract below relies on ack == "all round-r data received").
    if (threadIdx.x < 32)
    {
        int p = threadIdx.x;
        if (p < world)
        {
            unsigned long long* f =
                (unsigned long long*)((char*)peer_ptrs[p] + flags_off) + rank;
            atomicExch_system(f, r);
            unsigned long long* myf = (unsigned long long*)(arena + flags_off);
            while (atomicAdd_system(&myf[p], 0ULL) < r) {}
        }
    }
    __syncthreads();
    // BUG 116 rendezvous, deferred-ack form: every op still acks round r (the
    // graph-start barrier waits on these), but only EAGER ops (wait_ack=1) spin
    // for peer acks here. Inside a graph the per-op ack spin is redundant: slots
    // are double-buffered per key (reuse distance 2), each op's flag wait already
    // proves every peer STARTED the previous op, and a rank's consumers of an
    // older slot are stream-ordered before its next op kernel -- so a write to a
    // reused slot can never outrun a peer's read of it. The cross-replay hazard
    // (first op of replay r+1 aliasing a late slot of replay r) is covered by the
    // p2p_barrier_kernel at the graph start, which waits for ALL peers' acks of
    // the previous round before any write.
    if (threadIdx.x < 32)
    {
        int p = threadIdx.x;
        if (p < world)
        {
            unsigned long long* a =
                (unsigned long long*)((char*)peer_ptrs[p] + acks_off) + rank;
            atomicExch_system(a, r);
            if (wait_ack)
            {
                unsigned long long* mya = (unsigned long long*)(arena + acks_off);
                while (atomicAdd_system(&mya[p], 0ULL) < r) {}
            }
        }
    }
    __syncthreads();
}

// Graph-start barrier: wait until every peer has acked the round counter's current
// value (the last collective of the previous replay / eager op) BEFORE this replay's
// first one-sided write. Reads the device counter at replay time, so one captured
// node serves every replay. See the deferred-ack comment in p2p_allgather_kernel.
__global__ void p2p_barrier_kernel
(
    char* arena,
    const int64_t* peer_ptrs,
    int acks_off,
    int counter_off,
    int world,
    int rank
)
{
    if (threadIdx.x == 0)
    {
        unsigned long long c =
            atomicAdd((unsigned long long*)(arena + counter_off), 0ULL);
        unsigned long long* mya = (unsigned long long*)(arena + acks_off);
        for (int p = 0; p < world; p++)
            while (atomicAdd_system(&mya[p], 0ULL) < c) {}
    }
    __syncthreads();
}

// Fused allgather + RMSNorm + residual add (decode_repl block tail). The wire protocol
// is identical to p2p_allgather_kernel (same arena slots/flags/acks/counter); after the
// flag wait proves every peer's shard is present, the block normalizes the gathered
// full row and adds the residual, writing `out`. One rendezvous + one kernel replaces
// AG + norm + add (3 launches). fp16 data, fp32 norm reduction; weight has Gemma's +1
// bias baked in (matches ext_c.rms_norm). 256 threads, one block.
__global__ void p2p_ag_norm_res_kernel
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
    const half* __restrict__ weight,
    const half* __restrict__ residual,
    half* __restrict__ out,
    int rows,
    int hidden,
    float eps
)
{
    __shared__ unsigned long long round;
    __shared__ float red[P2P_THREADS];
    if (threadIdx.x == 0)
        round = atomicAdd((unsigned long long*)(arena + counter_off), 1ULL) + 1ULL;
    __syncthreads();
    unsigned long long r = round;

    // One-sided writes (NCCL all_gather(dim=1) row-interleaved layout), as allgather.
    int nrows = shard_bytes / row_bytes;
    int n16 = row_bytes / 16;
    for (int p = 0; p < world; p++)
    {
        char* peer = (char*)peer_ptrs[p] + slot_off;
        for (int i = 0; i < nrows; i++)
        {
            char* dst = peer + (i * world + rank) * row_bytes;
            const char* src = shard + i * row_bytes;
            for (int j = threadIdx.x; j < n16; j += blockDim.x)
                ((int4*)dst)[j] = ((const int4*)src)[j];
            for (int t = n16 * 16 + threadIdx.x; t < row_bytes; t += blockDim.x)
                dst[t] = src[t];
        }
    }
    __threadfence_system();
    __syncthreads();

    // Warp-parallel rendezvous (stage 2), identical contract to p2p_allgather_kernel.
    if (threadIdx.x < 32)
    {
        int p = threadIdx.x;
        if (p < world)
        {
            unsigned long long* f =
                (unsigned long long*)((char*)peer_ptrs[p] + flags_off) + rank;
            atomicExch_system(f, r);
            unsigned long long* myf = (unsigned long long*)(arena + flags_off);
            while (atomicAdd_system(&myf[p], 0ULL) < r) {}
        }
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        int p = threadIdx.x;
        if (p < world)
        {
            unsigned long long* a =
                (unsigned long long*)((char*)peer_ptrs[p] + acks_off) + rank;
            atomicExch_system(a, r);
            if (wait_ack)
            {
                unsigned long long* mya = (unsigned long long*)(arena + acks_off);
                while (atomicAdd_system(&mya[p], 0ULL) < r) {}
            }
        }
    }
    __syncthreads();

    // Gathered full rows live in the slot (row i = rank0_shard | ... | rankW-1_shard).
    const half* gbase = (const half*)(arena + slot_off);
    for (int row = 0; row < rows; row++)
    {
        const half* g = gbase + (long)row * hidden;
        float ss = 0.0f;
        for (int d = threadIdx.x; d < hidden; d += blockDim.x)
        {
            float v = __half2float(g[d]);
            ss += v * v;
        }
        red[threadIdx.x] = ss;
        __syncthreads();
        #pragma unroll
        for (int s = P2P_THREADS / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        float inv = rsqrtf(red[0] / (float)hidden + eps);
        __syncthreads();
        const half* res = residual + (long)row * hidden;
        half* o = out + (long)row * hidden;
        for (int d = threadIdx.x; d < hidden; d += blockDim.x)
        {
            float v = __half2float(g[d]) * inv * __half2float(weight[d]);
            o[d] = __float2half(v + __half2float(res[d]));
        }
        __syncthreads();
    }
}

template<typename T>
__global__ void p2p_allreduce_kernel
(
    char* arena,
    const int64_t* peer_ptrs,
    const T* data,
    T* out,
    int ar_off,
    int ar_slot_elems,
    int elems,
    int world,
    int rank,
    int flags_off,
    int acks_off,
    int counter_off,
    int wait_ack
)
{
    __shared__ unsigned long long round;
    if (threadIdx.x == 0)
        round = atomicAdd((unsigned long long*)(arena + counter_off), 1ULL) + 1ULL;
    __syncthreads();
    unsigned long long r = round;

    // Scatter my contribution into every peer's scratch[rank].
    for (int p = 0; p < world; p++)
    {
        T* dst = (T*)((char*)peer_ptrs[p] + ar_off) + rank * ar_slot_elems;
        for (int i = threadIdx.x; i < elems; i += blockDim.x)
            dst[i] = data[i];
    }
    // Fence every thread's writes, THEN barrier, THEN signal (see allgather).
    __threadfence_system();
    __syncthreads();

    // Warp-parallel rendezvous (one lane per peer), as the allgather kernel.
    if (threadIdx.x < 32)
    {
        int p = threadIdx.x;
        if (p < world)
        {
            unsigned long long* f =
                (unsigned long long*)((char*)peer_ptrs[p] + flags_off) + rank;
            atomicExch_system(f, r);
            unsigned long long* myf = (unsigned long long*)(arena + flags_off);
            while (atomicAdd_system(&myf[p], 0ULL) < r) {}
        }
    }
    __syncthreads();

    // Reduce locally (fp32 accumulate; exact for the one-nonzero embedding case).
    const T* base = (const T*)(arena + ar_off);
    for (int i = threadIdx.x; i < elems; i += blockDim.x)
    {
        float acc = 0.0f;
        for (int p = 0; p < world; p++)
            acc += p2p_to_float(base[p * ar_slot_elems + i]);
        out[i] = p2p_from_float(acc, (T*)nullptr);
    }
    __syncthreads();
    // BUG 116 rendezvous AFTER the local reduce. The scratch is now double-buffered
    // (region alternates per call), so within a graph the per-op ack spin is redundant
    // (reuse distance 2 + stream-ordered consumers + graph-start barrier cover the
    // cross-replay hazard) -- deferred-ack like the allgather. Eager ops still spin.
    if (threadIdx.x < 32)
    {
        int p = threadIdx.x;
        if (p < world)
        {
            unsigned long long* a =
                (unsigned long long*)((char*)peer_ptrs[p] + acks_off) + rank;
            atomicExch_system(a, r);
            if (wait_ack)
            {
                unsigned long long* mya = (unsigned long long*)(arena + acks_off);
                while (atomicAdd_system(&mya[p], 0ULL) < r) {}
            }
        }
    }
    __syncthreads();
}

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
)
{
    p2p_allgather_kernel<<<1, P2P_THREADS, 0, stream>>>
        (arena, peer_ptrs, shard, slot_off, shard_bytes, row_bytes, world, rank,
         flags_off, acks_off, counter_off, wait_ack);
}

void p2p_barrier_cuda
(
    char* arena,
    const int64_t* peer_ptrs,
    int acks_off,
    int counter_off,
    int world,
    int rank,
    cudaStream_t stream
)
{
    p2p_barrier_kernel<<<1, 32, 0, stream>>>
        (arena, peer_ptrs, acks_off, counter_off, world, rank);
}

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
    int wait_ack,
    cudaStream_t stream
)
{
    if (dtype == 0)
        p2p_allreduce_kernel<half><<<1, P2P_THREADS, 0, stream>>>
            (arena, peer_ptrs, (const half*)data, (half*)data,
             ar_off, ar_slot_elems, elems, world, rank, flags_off, acks_off, counter_off, wait_ack);
    else
        p2p_allreduce_kernel<float><<<1, P2P_THREADS, 0, stream>>>
            (arena, peer_ptrs, (const float*)data, (float*)data,
             ar_off, ar_slot_elems, elems, world, rank, flags_off, acks_off, counter_off, wait_ack);
}

void p2p_ag_norm_res_cuda
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
    const void* weight,
    const void* residual,
    void* out,
    int rows,
    int hidden,
    float eps,
    cudaStream_t stream
)
{
    p2p_ag_norm_res_kernel<<<1, P2P_THREADS, 0, stream>>>
        (arena, peer_ptrs, shard, slot_off, shard_bytes, row_bytes, world, rank,
         flags_off, acks_off, counter_off, wait_ack,
         (const half*)weight, (const half*)residual, (half*)out, rows, hidden, eps);
}