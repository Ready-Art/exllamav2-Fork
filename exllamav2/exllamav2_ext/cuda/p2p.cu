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
    int counter_off
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

    if (threadIdx.x == 0)
    {
        // Signal completion into each peer's flag slot for this rank...
        for (int p = 0; p < world; p++)
        {
            unsigned long long* f =
                (unsigned long long*)((char*)peer_ptrs[p] + flags_off) + rank;
            atomicExch_system(f, r);
        }
        // ...then wait until every peer has signaled round r into MY flags.
        unsigned long long* myf = (unsigned long long*)(arena + flags_off);
        for (int p = 0; p < world; p++)
            while (atomicAdd_system(&myf[p], 0ULL) < r) {}
        // BUG 116 rendezvous: ack round r and wait for every peer's ack before
        // returning, so no rank can start round r+1 writes while a slower rank
        // is still consuming round r data.
        for (int p = 0; p < world; p++)
        {
            unsigned long long* a =
                (unsigned long long*)((char*)peer_ptrs[p] + acks_off) + rank;
            atomicExch_system(a, r);
        }
        unsigned long long* mya = (unsigned long long*)(arena + acks_off);
        for (int p = 0; p < world; p++)
            while (atomicAdd_system(&mya[p], 0ULL) < r) {}
    }
    __syncthreads();
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
    int counter_off
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

    if (threadIdx.x == 0)
    {
        for (int p = 0; p < world; p++)
        {
            unsigned long long* f =
                (unsigned long long*)((char*)peer_ptrs[p] + flags_off) + rank;
            atomicExch_system(f, r);
        }
        unsigned long long* myf = (unsigned long long*)(arena + flags_off);
        for (int p = 0; p < world; p++)
            while (atomicAdd_system(&myf[p], 0ULL) < r) {}
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
    // BUG 116 rendezvous AFTER the local reduce: no peer may overwrite the shared
    // scratch with round r+1 contributions until every rank finished reading r.
    if (threadIdx.x == 0)
    {
        for (int p = 0; p < world; p++)
        {
            unsigned long long* a =
                (unsigned long long*)((char*)peer_ptrs[p] + acks_off) + rank;
            atomicExch_system(a, r);
        }
        unsigned long long* mya = (unsigned long long*)(arena + acks_off);
        for (int p = 0; p < world; p++)
            while (atomicAdd_system(&mya[p], 0ULL) < r) {}
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
    cudaStream_t stream
)
{
    p2p_allgather_kernel<<<1, P2P_THREADS, 0, stream>>>
        (arena, peer_ptrs, shard, slot_off, shard_bytes, row_bytes, world, rank,
         flags_off, acks_off, counter_off);
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
    cudaStream_t stream
)
{
    if (dtype == 0)
        p2p_allreduce_kernel<half><<<1, P2P_THREADS, 0, stream>>>
            (arena, peer_ptrs, (const half*)data, (half*)data,
             ar_off, ar_slot_elems, elems, world, rank, flags_off, acks_off, counter_off);
    else
        p2p_allreduce_kernel<float><<<1, P2P_THREADS, 0, stream>>>
            (arena, peer_ptrs, (const float*)data, (float*)data,
             ar_off, ar_slot_elems, elems, world, rank, flags_off, acks_off, counter_off);
}