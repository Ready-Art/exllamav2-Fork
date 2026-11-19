#ifndef _ext_tp_h
#define _ext_tp_h

#define BROADCAST_KV 0
#define BROADCAST_ID 1
#define BROADCAST_VC 2
#define BROADCAST_RS 3
#define BROADCAST_Q 4

#define TP_MULTITHREADED

//#include <nccl.h>
#include "cpp/threadpool.h"
#include "cuda/tp.cuh"

class ExtTPContext
{
public:
    std::vector<std::tuple<int, int, int>> kv_split;
    std::vector<std::tuple<int, int, int>> id_split;
    std::vector<std::tuple<int, int, int>> vc_split;
    std::vector<std::tuple<int, int, int>> rs_split;
    std::vector<std::tuple<int, int, int>> q_split;
    std::vector<void*> pinned_temp;
    size_t pinned_size;
    std::vector<cudaStream_t> streams;

    std::vector<int> all_devices;

    ThreadPool* thread_pool;
    ExtTPData* tp_data;

    void* mapped_globals;

    std::vector<cudaEvent_t> sync_events;
    // BUG 40: orders tp_broadcast's D2H (source stream) against its H2Ds (target streams).
    std::vector<cudaEvent_t> bc_events;

    // BUG 86: the REVERSE edge of the broadcast hazard.
    //
    // bc_events[src] orders "source finished writing" before "target reads". It does NOT order
    // "target finished reading" before "source overwrites the buffer next module". Without that second
    // edge, a fast source device reuses its buffer while a slow target is still reading it -- which is
    // exactly the corruption seen when the per-collective barrier was removed.
    //
    // read_events[target] is recorded on the target's stream after its copy, and the source's stream
    // waits on every target's read event before its next write. That is a per-target dependency edge in
    // both directions, which is what the global barrier was approximating.
    std::vector<cudaEvent_t> read_events;

    // BUG 43: CUDA-graph capture support.
    //
    // torch.cuda.graph cannot capture this step (it requires one stream on one device), so the
    // capture is driven from C++ with the raw API. `real_streams` is the per-device stream set
    // the kernels launch on; `capture_stream` is the primary device's stream that the capture
    // begins and ends on. `graph_capture` is true between begin and end.
    std::vector<cudaStream_t> real_streams;
    cudaStream_t capture_stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_capture = false;
    // Fork/join events used to bring each device stream into the capture region.
    std::vector<cudaEvent_t> fork_events;
    std::vector<cudaEvent_t> join_events;
//    std::vector<ncclComm_t> comms;
//    std::vector<int> comms_index;

    ExtTPContext
    (
        std::vector<std::tuple<int, int, int>> _kv_split,
        std::vector<std::tuple<int, int, int>> _id_split,
        std::vector<std::tuple<int, int, int>> _vc_split,
        std::vector<std::tuple<int, int, int>> _rs_split,
        std::vector<std::tuple<int, int, int>> _q_split,
        std::vector<torch::Tensor> _pinned_temp,
        std::vector<cudaStream_t> _streams
    );
    ~ExtTPContext();
};

uintptr_t make_tp_context
(
    const std::vector<std::tuple<int, int, int>> kv_split,
    const std::vector<std::tuple<int, int, int>> id_split,
    const std::vector<std::tuple<int, int, int>> vc_split,
    const std::vector<std::tuple<int, int, int>> rs_split,
    const std::vector<std::tuple<int, int, int>> q_split,
    std::vector<torch::Tensor> pinned_temp,
    std::vector<uintptr_t> streams
);

void free_tp_context(uintptr_t ctx);

void tp_broadcast
(
    uintptr_t tp_context,
    int buffer,
    torch::Tensor source,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int dim,
    int t_device = -1
);

void tp_gather
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &inputs,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int broadcast_type_target,
    int dim,
    int t_device = -1
);

void tp_gather_barrier
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &inputs,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int broadcast_type_target,
    int dim,
    int t_device = -1,
    Barrier* barrier = nullptr
);

void tp_cross_device_barrier
(
    uintptr_t tp_context,
    int broadcast_type,
    int t_device = -1,
    int stage = -1,
    int next_stage = -1
);

// BUG 43: CUDA-graph capture driver for the TP decode step. torch.cuda.graph cannot capture
// multi-device work, so capture is done here with the raw CUDA API.
void tp_graph_capture_begin(uintptr_t tp_context);
void tp_graph_capture_end(uintptr_t tp_context);
void tp_graph_launch(uintptr_t tp_context);
bool tp_graph_ready(uintptr_t tp_context);
int tp_graph_capture_status(uintptr_t tp_context);
void tp_set_capture_stream(uintptr_t tp_context, uintptr_t stream);
void tp_force_end_capture(uintptr_t stream);
void tp_reset_counters(uintptr_t tp_context);

// BUG 87: raise if any spin kernel timed out (collective-ordering deadlock).
void tp_raise_on_timeout(uintptr_t tp_context);

//void tp_all_reduce
//(
//    uintptr_t tp_context,
//    const std::vector<torch::Tensor> &tensors
//);

void tp_all_reduce
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &tensors,
    const std::vector<torch::Tensor> &residuals
);

#endif