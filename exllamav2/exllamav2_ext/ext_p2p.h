#ifndef EXT_P2P_H
#define EXT_P2P_H

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <string>

bool p2p_enable_peer_access(int64_t rank, int64_t world);
// py::bytes, NOT std::string: pybind UTF-8-decodes std::string returns into str,
// which raises on the raw binary cudaIpcMemHandle_t.
pybind11::bytes p2p_ipc_handle(torch::Tensor arena);
int64_t p2p_ipc_open(const std::string& handle);

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
    int64_t counter_off,
    int64_t wait_ack
);

void p2p_barrier
(
    int64_t arena_ptr,
    int64_t peer_ptrs_ptr,
    int64_t acks_off,
    int64_t counter_off,
    int64_t world,
    int64_t rank
);

void p2p_ag_norm_res
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
    int64_t counter_off,
    int64_t wait_ack,
    torch::Tensor weight,
    torch::Tensor residual,
    torch::Tensor out,
    int64_t rows,
    int64_t hidden,
    double eps
);

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
    int64_t dtype,
    int64_t wait_ack
);

#endif