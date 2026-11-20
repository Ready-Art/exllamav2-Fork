#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstring>

#include "ext_p2p.h"
#include "cuda/p2p.cuh"

// Returns true only if THIS device can peer-access every other device in the
// group. The Python side MIN-allreduces the result so the whole group agrees
// before any P2P kernel runs; any failure -> NCCL fallback.
bool p2p_enable_peer_access(int64_t rank, int64_t world)
{
    const c10::cuda::CUDAGuard guard((c10::DeviceIndex)rank);
    for (int p = 0; p < (int)world; p++)
    {
        if (p == (int)rank) continue;
        int can = 0;
        cudaError_t e = cudaDeviceCanAccessPeer(&can, (int)rank, p);
        if (e != cudaSuccess || !can) return false;
        e = cudaDeviceEnablePeerAccess(p, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) return false;
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    }
    return true;
}

pybind11::bytes p2p_ipc_handle(torch::Tensor arena)
{
    TORCH_CHECK(arena.is_cuda() && arena.is_contiguous());
    const c10::cuda::CUDAGuard guard(arena.device());
    cudaIpcMemHandle_t h;
    cudaError_t e = cudaIpcGetMemHandle(&h, arena.data_ptr());
    TORCH_CHECK(e == cudaSuccess, "cudaIpcGetMemHandle failed: ", cudaGetErrorString(e));
    return pybind11::bytes((const char*)&h, sizeof(h));
}

int64_t p2p_ipc_open(const std::string& handle)
{
    TORCH_CHECK(handle.size() == sizeof(cudaIpcMemHandle_t));
    cudaIpcMemHandle_t h;
    std::memcpy(&h, handle.data(), sizeof(h));
    void* ptr = nullptr;
    cudaError_t e = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(e == cudaSuccess, "cudaIpcOpenMemHandle failed: ", cudaGetErrorString(e));
    return (int64_t)ptr;
}

void p2p_allgather
(
    int64_t arena_ptr,
    int64_t peer_ptrs_ptr,
    int64_t shard_ptr,
    int64_t slot_off,
    int64_t shard_bytes,
    int64_t row_bytes,
    int64_t world,
    int64_t rank,
    int64_t flags_off,
    int64_t acks_off,
    int64_t counter_off
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    p2p_allgather_cuda
    (
        (char*)arena_ptr,
        (const int64_t*)peer_ptrs_ptr,
        (const char*)shard_ptr,
        (int)slot_off,
        (int)shard_bytes,
        (int)row_bytes,
        (int)world,
        (int)rank,
        (int)flags_off,
        (int)acks_off,
        (int)counter_off,
        stream
    );
}

void p2p_allreduce
(
    int64_t arena_ptr,
    int64_t peer_ptrs_ptr,
    int64_t data_ptr,
    int64_t ar_off,
    int64_t ar_slot_elems,
    int64_t elems,
    int64_t world,
    int64_t rank,
    int64_t flags_off,
    int64_t acks_off,
    int64_t counter_off,
    int64_t dtype
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    p2p_allreduce_cuda
    (
        (char*)arena_ptr,
        (const int64_t*)peer_ptrs_ptr,
        (void*)data_ptr,
        (int)ar_off,
        (int)ar_slot_elems,
        (int)elems,
        (int)world,
        (int)rank,
        (int)flags_off,
        (int)acks_off,
        (int)counter_off,
        (int)dtype,
        stream
    );
}