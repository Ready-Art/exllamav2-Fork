#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "ext_tp.h"
#include "cpp/util.h"

#define cuda_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// BUG 43: resolve the stream a TP operation must be issued on.
//
// During CUDA-graph capture the primary device's work has to run on the stream the capture is
// active on. Every other device's stream is forked from that stream with an event, so it is part of
// the capture too. If dev0 still used its ordinary stream here, the barrier would record
// `sync_events[dev0]` on an uncaptured stream and every captured device stream that waits on it
// would abort with "dependency created on uncaptured work in another stream".
inline cudaStream_t tp_stream(ExtTPContext* ctx, int dev)
{
    if (ctx->graph_capture && !ctx->all_devices.empty() &&
        dev == ctx->all_devices[0] && ctx->capture_stream)
        return ctx->capture_stream;
    return ctx->streams[dev];
}

// BUG 48: direct device-to-device collectives.
//
// Every collective used to stage through host pinned memory: a D2H followed by one H2D per
// target device. On this machine the four GPUs share a PCIe switch with working peer access on
// all 12 ordered pairs, so the host round trip is pure overhead -- and it is the dominant cost
// of prefill, not a rounding error. Measured on a 1032-token prefill (4 GPUs):
//
//   Memcpy HtoD (Pinned -> Device)   3122 ms   66.6% of all GPU time
//   Memcpy DtoH (Device -> Pinned)    717 ms   15.3%
//   ------------------------------------------------------
//   host-staged copies total        3839 ms   81.9%
//
// A payload sweep (prof_p2p3.py) shows where each strategy wins:
//
//   payload      host-staged   P2P      speedup
//   10.5 KB         120 us    122 us    0.98x     <- decode, a wash
//   43 KB           118 us    123 us    0.96x     <- decode, a wash
//   1 MB            202 us    138 us    1.47x
//   11.1 MB        2112 us   1320 us    1.60x     <- prefill hidden
//   44.4 MB        8429 us   5220 us    1.61x     <- prefill mlp
//
// So P2P is used only above a threshold, and never during graph capture: decode payloads
// are launch-latency bound and P2P does not help there, and the captured decode graph is
// already correct with the host-staged path. This keeps the change strictly additive.
#define TP_P2P_THRESHOLD_BYTES (256 * 1024)
// Inside a captured graph the per-op launch overhead that motivated the 256 KB threshold
// above does not exist, so even a small peer copy beats a host-staged one. Keep a floor
// anyway: a peer copy of a few hundred bytes is still not worth a kernel launch.
#define TP_P2P_THRESHOLD_BYTES_GRAPH (2 * 1024)

// Can every target device read `src_dev`'s memory directly?
inline bool tp_p2p_ok(ExtTPContext* ctx, int src_dev, const std::vector<torch::Tensor> &targets)
{
    if (src_dev < 0) return false;
    for (const auto &t : targets)
    {
        int td = t.device().index();
        if (td < 0 || td == src_dev) continue;
        int can = 0;
        if (cudaDeviceCanAccessPeer(&can, td, src_dev) != cudaSuccess || !can)
            return false;
    }
    return true;
}

// BUG 48c: probe whether cudaMemcpy* actually works on this process's device pointers.
//
// PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True makes torch allocate through the CUDA VMM
// API (cuMemCreate/cuMemMap), so a tensor is a sub-allocation of a large virtual segment. The
// runtime's cudaMemcpy* entry points reject those pointers with cudaErrorInvalidValue -- even for
// a *same-device* copy. Verified directly (tpeer2.py):
//
//   expandable_segments ON : cudaMemcpy2DAsync D2D = invalid argument (same-device too)
//   expandable_segments OFF: cudaMemcpy2DAsync D2D = success
//
// cudaPointerGetAttributes() is NOT a usable discriminator: it reports cudaMemoryTypeDevice for
// VMM pointers too (verified, tattr.py), so there is no way to tell from the pointer alone.
//
// The only reliable test is to try it. This probe issues a 1-byte device-to-device copy between
// two real tensors and caches the verdict for the process. A failure enqueues nothing, so falling
// back is safe. The result is stored on the context and consulted by every collective.

ExtTPContext::ExtTPContext
(
    std::vector<std::tuple<int, int, int>> _kv_split,
    std::vector<std::tuple<int, int, int>> _id_split,
    std::vector<std::tuple<int, int, int>> _vc_split,
    std::vector<std::tuple<int, int, int>> _rs_split,
    std::vector<std::tuple<int, int, int>> _q_split,
    std::vector<torch::Tensor> _pinned_temp,
    std::vector<cudaStream_t> _streams
) :
    kv_split(_kv_split),
    id_split(_id_split),
    vc_split(_vc_split),
    rs_split(_rs_split),
    q_split(_q_split),
    streams(_streams)
{
    for (const auto &pt : _pinned_temp)
    {
        void* ptp = (void*) pt.data_ptr();
        pinned_temp.push_back(ptp);
        pinned_size = pt.numel() * pt.element_size();
    }

    for (int i = 0; i < streams.size(); ++i)
        if (streams[i]) all_devices.push_back(i);

    sync_events.resize(streams.size());
    bc_events.resize(streams.size());
    fork_events.resize(streams.size());
    join_events.resize(streams.size());
    real_streams = streams;
    graph_capture = false;

    // BUG 48b: ENABLE peer access. cudaDeviceCanAccessPeer() only reports whether a pair is
    // *capable*; the copy engine refuses peer transfers until cudaDeviceEnablePeerAccess() has
    // been called for that ordered pair. Nothing in this tree ever called it, so "P2P is
    // available" was true on paper and false in practice: peer copies failed with
    // "invalid argument" and the host-staged path was silently doing all the work.
    //
    // This is why P2P looked like a wash at every payload size in the first benchmark -- the
    // copies were never actually peer copies.
    for (int i = 0; i < streams.size(); ++i)
    {
        if (!streams[i]) continue;
        cudaSetDevice(i);
        for (int j = 0; j < streams.size(); ++j)
        {
            if (i == j || !streams[j]) continue;
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can) continue;
            // Already-enabled is reported as cudaErrorPeerAccessAlreadyEnabled; that is fine.
            cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else if (e != cudaSuccess)
            {
                fprintf(stderr, "TP: cudaDeviceEnablePeerAccess(%d -> %d) failed: %s\n",
                        i, j, cudaGetErrorString(e));
                cudaGetLastError();
            }
        }
    }

    for (int i = 0; i < streams.size(); ++i)
    {
        if (!streams[i]) continue;
        cudaSetDevice(i);
        cuda_check(cudaEventCreateWithFlags(&sync_events[i], cudaEventDisableTiming));
        // BUG 40: separate event used to order tp_broadcast's D2H against its H2Ds.
        cuda_check(cudaEventCreateWithFlags(&bc_events[i], cudaEventDisableTiming));
        // BUG 43: fork/join events for CUDA-graph capture.
        cuda_check(cudaEventCreateWithFlags(&fork_events[i], cudaEventDisableTiming));
        cuda_check(cudaEventCreateWithFlags(&join_events[i], cudaEventDisableTiming));
    }

    #ifdef TP_MULTITHREADED

        int numdevs = all_devices.size();
        thread_pool = new ThreadPool(numdevs);

    #endif

    cudaHostAlloc((void**)&tp_data, sizeof(ExtTPData), cudaHostAllocMapped);
    init_tp_data(tp_data);

//    comms.resize(all_devices.size());
//    ncclCommInitAll(&comms[0], all_devices.size(), &all_devices[0]);
//    comms_index.resize(streams.size());
//    for (int i = 0; i < all_devices.size(); ++i)
//        comms_index[all_devices[i]] = i;

}

ExtTPContext::~ExtTPContext()
{
    #ifdef TP_MULTITHREADED
        delete thread_pool;
    #endif

//    for (int i = 0; i < comms.size(); ++i)
//        ncclCommDestroy(comms[i]);

    for (int i = 0; i < bc_events.size(); ++i)
    {
        if (!bc_events[i]) continue;
        cudaSetDevice(i);
        cudaEventDestroy(bc_events[i]);
    }

    for (int i = 0; i < fork_events.size(); ++i)
    {
        cudaSetDevice(i);
        if (fork_events[i]) cudaEventDestroy(fork_events[i]);
        if (join_events[i]) cudaEventDestroy(join_events[i]);
    }

    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);

    cudaFreeHost(tp_data);
}

uintptr_t make_tp_context
(
    std::vector<std::tuple<int, int, int>> kv_split,
    std::vector<std::tuple<int, int, int>> id_split,
    std::vector<std::tuple<int, int, int>> vc_split,
    std::vector<std::tuple<int, int, int>> rs_split,
    std::vector<std::tuple<int, int, int>> q_split,
    std::vector<torch::Tensor> pinned_temp,
    std::vector<uintptr_t> streams
)
{
    std::vector<cudaStream_t> streams_;
    for (int i = 0; i < streams.size(); ++i)
        streams_.push_back(reinterpret_cast<cudaStream_t> (streams[i]));

    ExtTPContext* ctx = new ExtTPContext
    (
        kv_split,
        id_split,
        vc_split,
        rs_split,
        q_split,
        pinned_temp,
        streams_
    );

    return reinterpret_cast<uintptr_t> (ctx);
}

void free_tp_context
(
    uintptr_t tp_context
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    delete ctx;
}

void tp_broadcast
(
    uintptr_t tp_context,
    int buffer,
    torch::Tensor source,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int dim,
    int t_device
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    size_t size = source.numel() * 2;

    void* source_g = NULL;

    int src_dev = source.device().index();

    // BUG 48: pick the transport once, before any copy is issued.
    // BUG 48d: use torch's tensor copy_ for the device-to-device path, NOT cudaMemcpy*.
    //
    // cudaMemcpy* cannot be used here at all. Under
    // PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torch allocates through the CUDA VMM API,
    // so every tensor is a sub-allocation of a large virtual segment, and the runtime's memcpy
    // entry points reject those pointers with cudaErrorInvalidValue -- even for a same-device
    // copy (verified directly: tpeer2.py). There is no reliable way to detect this from the
    // pointer (cudaPointerGetAttributes reports cudaMemoryTypeDevice either way), so any probe
    // either misfires or is racy.
    //
    // torch's copy_ dispatches to a copy kernel rather than cudaMemcpy, so it works under both
    // allocators. It is also peer-aware: torch issues a peer copy when the two tensors live on
    // different devices with peer access enabled. Using it keeps this path correct regardless of
    // how the allocator is configured.
    // BUG 56: do NOT exclude graph capture here. During capture the host-staged path is
    // strictly worse -- it costs a D2H + event + H2D per collective and those are exactly the
    // 1202 HtoD / 840 DtoH per token that dominated the decode step. A peer copy is one graph
    // node. Use a lower size threshold when capturing, because the eager threshold exists to
    // amortise launch overhead that capture removes.
    size_t p2p_threshold = ctx->graph_capture
        ? (size_t) TP_P2P_THRESHOLD_BYTES_GRAPH
        : (size_t) TP_P2P_THRESHOLD_BYTES;
    bool use_p2p = (size >= p2p_threshold) &&
                   tp_p2p_ok(ctx, src_dev, targets);
    if (!use_p2p)
        TORCH_CHECK(size <= ctx->pinned_size, "Temporary tensor is too small")

    if (src_dev >= 0)
    {
        cudaSetDevice(src_dev);
        // BUG 43: during graph capture the D2H must run on a stream that is part of the
        // capture. tp_stream() resolves dev0 to the capture stream and every other device to
        // its own (forked) stream.
        cudaStream_t stream = ctx->graph_capture
            ? tp_stream(ctx, src_dev)
            : at::cuda::getCurrentCUDAStream().stream();

        // BUG 86 REVERSE EDGE: this device is about to overwrite its buffer, so it must first
        // wait until every target has finished reading the previous contents.
        //
        // Expressed as a spin kernel on mapped memory, NOT as a cudaStreamWaitEvent -- an event
        // wait on work recorded outside the capture region is illegal inside it.
        //
        // The threshold is bc_seq[src_dev], read on the device, so the same graph replays
        // correctly: on replay N the source waits for the targets' rd_seq to reach N-1.
        for (int k = 0; k < targets.size(); ++k)
        {
            int tdev = targets[k].device().index();
            if (tdev < 0 || tdev == src_dev) continue;
            tp_wait_ge_cuda
            (
                stream,
                &ctx->tp_data->rd_seq[tdev][src_dev],
                &ctx->tp_data->bc_seq[src_dev],
                ctx->tp_data,
                src_dev,
                10
            );
        }

        source_g = (void*) source.data_ptr();
        if (!use_p2p)
            cuda_check(cudaMemcpyAsync(ctx->pinned_temp[buffer], source_g, size, cudaMemcpyDeviceToHost, stream));

        // BUG 40 / BUG 86: the copies below are issued on each target's stream, a different
        // stream from the one the source was produced on, so the target must be told when the
        // source is ready. That is the forward edge, published as a counter increment on the
        // source's stream. A target spins on bc_seq[src_dev] before it reads.
        tp_publish_cuda(stream, &ctx->tp_data->bc_seq[src_dev]);
    }

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    auto broadcast_dev = [&] (int i) -> void
    {
        #ifdef TP_MULTITHREADED
            at::InferenceMode guard(true);
        #endif
        int dev = std::get<0>(split[i]);

        void* target = (void*) targets[i].data_ptr();
        if (target == source_g) return;

        cudaSetDevice(dev);
        cudaStream_t stream = tp_stream(ctx, dev);

        // BUG 89: make torch's current stream for this device BE the TP stream, so the copy below
        // lands on the same stream as the ordering kernels.
        //
        // tp_stream() returns a raw cudaStream_t. torch's copy_ dispatches on torch's current stream
        // for the device, which is a DIFFERENT stream unless set explicitly; cudaSetDevice only changes
        // the device, not the stream. Without this the wait/publish kernels and the data copy ran on
        // unrelated streams, so the counters ordered nothing and corruption accumulated over tokens --
        // exactly the observed signature (early tokens fine, later ones degenerate).
        at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(stream, dev));

        // BUG 40 / BUG 86: wait for the source before reading it. Spin on the source's
        // published counter rather than waiting on an event, so this is capturable.
        if (src_dev >= 0 && src_dev != dev)
            tp_wait_gt_cuda
            (
                stream,
                &ctx->tp_data->bc_seq[src_dev],
                &ctx->tp_data->rd_seq[dev][src_dev],
                ctx->tp_data,
                dev,
                11
            );

        if (use_p2p)
        {
            // BUG 48d: torch copy_ (peer-aware, VMM-safe). The counter wait above already
            // ordered this against the source.
            targets[i].copy_(source, true);
        }
        else
        {
            cuda_check(cudaMemcpyAsync(target, ctx->pinned_temp[buffer], size, cudaMemcpyHostToDevice, stream));
        }

        // BUG 86 REVERSE EDGE: publish "this target has finished reading", so the source may
        // overwrite its buffer on the next module.
        tp_publish_cuda(stream, &ctx->tp_data->rd_seq[dev][src_dev]);
    };

    #ifdef TP_MULTITHREADED
    {
        std::vector<std::future<void>> threads;
        for (int i = 0; i < split.size(); ++i)
        {
            int dev = std::get<0>(split[i]);
            if (t_device != -1 && t_device != dev) continue;
            threads.push_back(ctx->thread_pool->enqueue(broadcast_dev, i));
        }
        for (auto &t : threads) t.get();
    }
    #else
    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        if (t_device != -1 && t_device != dev) continue;
        broadcast_dev(i);
    }
    #endif

    // BUG 86: with both edges carried by counters, the per-collective barrier is not needed on the
    // P2P path. The host-staged path still needs it: it shares one pinned buffer across modules,
    // and that hazard is not covered by these counters.
    // EXL2_TP_BC_BARRIER=1 restores the old unconditional behaviour for A/B.
    static int _bc_barrier = -1;
    if (_bc_barrier == -1)
    {
        const char* e = getenv("EXL2_TP_BC_BARRIER");
        _bc_barrier = (e == nullptr) ? 0 : (atoi(e) != 0);
    }
    if (!use_p2p || _bc_barrier)
        tp_cross_device_barrier(tp_context, broadcast_type, t_device);
}

void tp_gather
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &inputs,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int broadcast_type_target,
    int dim,
    int t_device
)
{
    tp_gather_barrier
    (
        tp_context,
        buffer,
        inputs,
        broadcast_type,
        targets,
        broadcast_type_target,
        dim,
        t_device,
        nullptr
    );
}

void tp_gather_barrier
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &inputs,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int broadcast_type_target,
    int dim,
    int t_device,
    Barrier* barrier
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    int out_rows = inputs[0].size(0);
    int out_cols = std::get<2>(split[split.size() - 1]) * dim;
    int esize = inputs[0].element_size();

    // BUG 48: the gather can go device-to-device instead of through pinned host memory.
    //
    // Host-staged: each device D2H's its slice into pinned memory at its column offset, then the
    // assembled full-width row is H2D'd back to every target. The data never needs to be on the
    // host, so at prefill sizes that round trip is the dominant cost of the whole step.
    //
    // P2P: each device writes its slice straight into every *target* device's tensor at the same
    // column offset. Same result, no host involvement.
    //
    // Only when the payload is large enough to amortise the peer-copy latency, and never inside a
    // capture region -- the decode graph was verified against the host-staged form.
    size_t gather_bytes = (size_t) out_rows * out_cols * esize;
    // BUG 48d: torch copy_ again (see the note in tp_broadcast). cudaMemcpy* is unusable
    // under expandable_segments.
    bool gather_p2p = (broadcast_type_target != -2) &&
                      (broadcast_type_target != -1) &&
                      (gather_bytes >= (ctx->graph_capture
                          ? (size_t) TP_P2P_THRESHOLD_BYTES_GRAPH
                          : (size_t) TP_P2P_THRESHOLD_BYTES));
    if (gather_p2p)
    {
        for (int i = 0; i < split.size() && gather_p2p; ++i)
        {
            int dev = std::get<0>(split[i]);
            if (t_device != -1 && t_device != dev) continue;
            for (int k = 0; k < split.size(); ++k)
            {
                int tdev = std::get<0>(split[k]);
                if (t_device != -1 && t_device != tdev) continue;
                if (tdev == dev) continue;
                int can = 0;
                if (cudaDeviceCanAccessPeer(&can, tdev, dev) != cudaSuccess || !can)
                { gather_p2p = false; break; }
            }
        }
    }

    if (gather_p2p)
    {
        // BUG 86: per-target ordering in both directions, carried by mapped-memory counters.
        //
        // The old code wrote each source slice into every target from the SOURCE's stream, so a
        // target's own stream had no ordering against the writes into its memory. That is why the
        // global barrier was needed, and why simply deleting it corrupted the output.
        //
        // Two edges are required, not one:
        //   forward : source finished writing  ->  target may read
        //   reverse : target finished reading  ->  source may overwrite next module
        //
        // Both are expressed as spin kernels on mapped counters, NOT as CUDA events: an event wait on
        // work recorded outside the capture region is illegal inside it, which is exactly how the first
        // attempt at this failed.
        // BUG 86b: dispatch each source's work to its own pool thread, so the spin kernels
        // block only that thread rather than the single Python caller.
        auto gather_src = [&] (int i) -> void
        {
            #ifdef TP_MULTITHREADED
                at::InferenceMode guard(true);
            #endif
            int dev = std::get<0>(split[i]);

            uint8_t* src = (uint8_t*) inputs[i].data_ptr();
            int src_cols = inputs[i].size(1);
            int col_off = std::get<1>(split[i]) * dim;

            cudaSetDevice(dev);
            cudaStream_t src_stream = tp_stream(ctx, dev);

            // BUG 89: align torch's current stream with the TP stream so the copy_ below is
            // ordered by the wait/publish kernels (see the note in broadcast_dev).
            at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(src_stream, dev));

            // REVERSE EDGE: this source must not overwrite inputs[i] until every target that read
            // it last round has finished. rd_seq[tdev] is published by tdev after its read.
            for (int k = 0; k < split.size(); ++k)
            {
                int tdev = std::get<0>(split[k]);
                if (tdev == dev) continue;
                tp_wait_ge_cuda
                (
                    src_stream,
                    &ctx->tp_data->rd_seq[tdev][dev],
                    &ctx->tp_data->bc_seq[dev],
                    ctx->tp_data,
                    dev,
                    20
                );
            }

            for (int k = 0; k < split.size(); ++k)
            {
                int tdev = std::get<0>(split[k]);
                if (t_device != -1 && t_device != tdev) continue;
                // Write this source slice into target k at the source's column offset.
                targets[k].narrow(1, col_off, src_cols).copy_(inputs[i], true);
            }

            // FORWARD EDGE: publish "this source has finished writing into every target".
            tp_publish_cuda(src_stream, &ctx->tp_data->bc_seq[dev]);
        };

        auto gather_dst = [&] (int k) -> void
        {
            #ifdef TP_MULTITHREADED
                at::InferenceMode guard(true);
            #endif
            int tdev = std::get<0>(split[k]);
            cudaSetDevice(tdev);
            cudaStream_t dst_stream = tp_stream(ctx, tdev);

            // BUG 89: align torch's current stream with the TP stream (see broadcast_dev).
            at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(dst_stream, tdev));

            for (int i = 0; i < split.size(); ++i)
            {
                int dev = std::get<0>(split[i]);
                if (dev == tdev) continue;
                tp_wait_gt_cuda
                (
                    dst_stream,
                    &ctx->tp_data->bc_seq[dev],
                    &ctx->tp_data->rd_seq[tdev][dev],
                    ctx->tp_data,
                    tdev,
                    21
                );

                // REVERSE EDGE: publish "this target has finished reading from `dev`".
                // BUG 87: rd_seq is per (reader, source) pair, so the publish must be inside the
                // loop -- one per source, not one per target.
                tp_publish_cuda(dst_stream, &ctx->tp_data->rd_seq[tdev][dev]);
            }
        };

        #ifdef TP_MULTITHREADED
        {
            std::vector<std::future<void>> threads;
            for (int i = 0; i < split.size(); ++i)
            {
                int dev = std::get<0>(split[i]);
                if (t_device != -1 && t_device != dev) continue;
                threads.push_back(ctx->thread_pool->enqueue(gather_src, i));
            }
            for (auto &t : threads) t.get();

            threads.clear();
            for (int k = 0; k < split.size(); ++k)
            {
                int tdev = std::get<0>(split[k]);
                if (t_device != -1 && t_device != tdev) continue;
                threads.push_back(ctx->thread_pool->enqueue(gather_dst, k));
            }
            for (auto &t : threads) t.get();
        }
        #else
        for (int i = 0; i < split.size(); ++i)
        {
            int dev = std::get<0>(split[i]);
            if (t_device != -1 && t_device != dev) continue;
            gather_src(i);
        }
        for (int k = 0; k < split.size(); ++k)
        {
            int tdev = std::get<0>(split[k]);
            if (t_device != -1 && t_device != tdev) continue;
            gather_dst(k);
        }
        #endif

        if (barrier)
            barrier->arrive_and_wait();

        // BUG 86: the per-collective barrier is not needed on this path any more -- both edges are
        // carried by counters. EXL2_TP_GATHER_BARRIER=1 restores it for A/B.
        static int _g_barrier = -1;
        if (_g_barrier == -1)
        {
            const char* e = getenv("EXL2_TP_GATHER_BARRIER");
            _g_barrier = (e == nullptr) ? 0 : (atoi(e) != 0);
        }
        if (_g_barrier)
            tp_cross_device_barrier(tp_context, broadcast_type, t_device);
        return;
    }

    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        if (t_device != -1 && t_device != dev) continue;

        uint8_t* src = (uint8_t*) inputs[i].data_ptr();
        int src_cols = inputs[i].size(1);
        uint8_t* dst = ((uint8_t*) ctx->pinned_temp[buffer]) + std::get<1>(split[i]) * esize * dim;

        cudaSetDevice(dev);
        cuda_check(cudaMemcpy2DAsync
        (
            dst,
            out_cols * esize,
            src,
            src_cols * esize,
            src_cols * esize,
            out_rows,
            cudaMemcpyDeviceToHost,
            tp_stream(ctx, dev)
        ));
    }

    if (broadcast_type_target == -2) return;

    if (barrier)
        barrier->arrive_and_wait();

    tp_cross_device_barrier(tp_context, broadcast_type, t_device);

    if (broadcast_type_target == -1) return;

    size_t size = targets[0].numel() * 2;

    switch(broadcast_type_target)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        if (t_device != -1 && t_device != dev) continue;

        void* target = (void*) targets[i].data_ptr();

        cudaSetDevice(dev);
        cudaStream_t stream = tp_stream(ctx, dev);
        cuda_check(cudaMemcpyAsync(target, ctx->pinned_temp[buffer], size, cudaMemcpyHostToDevice, stream));
    }
}

void tp_cross_device_barrier
(
    uintptr_t tp_context,
    int broadcast_type,
    int t_device,
    int stage,
    int next_stage
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    if (stage == -1)
    {
        stage = ctx->tp_data->next_stage;
        ctx->tp_data->next_stage = (ctx->tp_data->next_stage + 1) % MAX_SYNC_STAGES;
        next_stage = ctx->tp_data->next_stage;
    }

    uint32_t* sync = ctx->tp_data->sync[stage];
    uint32_t* sync_next = ctx->tp_data->sync[next_stage];

//    for (int i = 0; i < ctx->all_devices.size(); ++i)
//    {
//        int dev = ctx->all_devices[i];
//        // if (t_device != -1 && t_device != dev) continue;
//        cross_device_barrier_cuda
//        (
//            ctx->streams[dev],
//            sync,
//            sync_next,
//            ctx->all_devices.size(),
//            i
//        );
//    }

//    for (int i = 0; i < ctx->all_devices.size(); ++i)
//    {
//        int dev = ctx->all_devices[i];
//        cudaSetDevice(dev);
//        // if (t_device != -1 && t_device != dev) continue;
//        cudaStreamSynchronize(ctx->streams[dev]);
//    }

    // ---------------------------------------------------------------------------------------
    // Cross-device barrier.
    //
    // OLD (still selectable with EXL2_TP_SPIN_BARRIER=0): a full mesh of CUDA events --
    //   N x cudaEventRecord + N*(N-1) x cudaStreamWaitEvent
    // i.e. 4 + 12 = 16 event operations per barrier. The model issues 1200 barriers per decode
    // token (480 broadcast + 480 allgather + 240 gather), so that is ~19200 event operations per
    // token. That is the reason decode sits at 41 ms/token with only 17.3 ms of kernel time.
    //
    // NEW (default): one 1-thread spin kernel per device, coordinated by device 0, ported from
    // exllamav3's pg_barrier_inner (exllamav3_ext/parallel/barrier_inner.cuh). A participant
    // does one counter store and one counter poll instead of N-1 cross-device event waits. The
    // counters are monotonic and never reset, so a captured graph replays correctly with no
    // per-step initialisation.
    //
    // The event path is kept behind an env var so this can be A/B'd without a rebuild.
    // ---------------------------------------------------------------------------------------

    {
        static const int _spin_env = -1;
        static int _spin = -1;
        if (_spin == -1)
        {
            const char* e = getenv("EXL2_TP_SPIN_BARRIER");
            _spin = (e == nullptr) ? 1 : (atoi(e) != 0);
        }

        if (_spin)
        {
            uint32_t device_mask = 0;
            for (int i = 0; i < ctx->all_devices.size(); ++i)
                device_mask |= (1u << ctx->all_devices[i]);

            int coordinator = ctx->all_devices[0];

            for (int i = 0; i < ctx->all_devices.size(); ++i)
            {
                int dev = ctx->all_devices[i];
                if (t_device != -1 && t_device != dev) continue;
                cudaSetDevice(dev);
                tp_barrier_cuda
                (
                    tp_stream(ctx, dev),
                    ctx->tp_data,
                    device_mask,
                    dev,
                    coordinator
                );
            }
            return;
        }
    }

    // BUG 43: during graph capture this event dance is exactly what makes the cross-device
    // work part of the graph -- recording on a captured stream creates a graph node and
    // waiting on a captured stream creates the dependency edge. So it must NOT be skipped.
    for (int i = 0; i < ctx->all_devices.size(); ++i)
    {
        int dev_i = ctx->all_devices[i];
        cudaSetDevice(dev_i);
        cuda_check(cudaEventRecord(ctx->sync_events[dev_i], tp_stream(ctx, dev_i)));
    }

    for (int i = 0; i < ctx->all_devices.size(); ++i)
    {
        for (int j = 0; j < ctx->all_devices.size(); ++j)
        {
            if (i == j) continue;
            int dev_i = ctx->all_devices[i];
            int dev_j = ctx->all_devices[j];
            cudaSetDevice(dev_i);
            cuda_check(cudaStreamWaitEvent(tp_stream(ctx, dev_i), ctx->sync_events[dev_j], 0));
        }
    }
}

// BUG 43: CUDA-graph capture driver.
//
// torch.cuda.graph cannot capture the TP decode step: it requires capture to begin and end on
// a single stream on a single device, and the TP step issues work on four devices. The raw CUDA
// API does support it -- begin capture on the primary device's stream, fork the other devices'
// streams with events, run the step, join, then end capture. The resulting graph holds nodes for
// every device and is launched with one cudaGraphLaunch.
//
// This is the same structure exllamav3 uses (exllamav3_ext/graph.cu).
void tp_graph_capture_begin(uintptr_t tp_context)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    TORCH_CHECK(!ctx->graph_capture, "tp_graph_capture_begin: capture already active");

    int dev0 = ctx->all_devices[0];
    cudaSetDevice(dev0);

    // Surface any earlier async fault before starting a new capture.
    cuda_check(cudaDeviceSynchronize());

    ctx->capture_stream = ctx->real_streams[dev0];

    // Global mode: work issued on other streams from this thread is captured too, which is what
    // lets the per-device streams join the graph via the event forks in tp_broadcast and
    // tp_cross_device_barrier.
    cuda_check(cudaStreamBeginCapture(ctx->capture_stream, cudaStreamCaptureModeGlobal));
    ctx->graph_capture = true;

    // Fork every other device's stream from the capture stream, inside the capture region.
    //
    // Without this, the first event wait on a device stream refers to work that is not part of
    // the capture and the runtime rejects it with "dependency created on uncaptured work in
    // another stream". Recording the fork event on the capture stream and waiting on it from each
    // device stream makes those streams part of the capture, so the work the TP kernels launch
    // on them (the GEMMs) is recorded into the graph.
    // One fork event, created on dev0 and recorded on the capture stream. Every other
    // device's stream then waits on it. An event must be recorded on a stream belonging to the
    // device it was created on, which is why this is a single dev0 event rather than one per
    // device; a cross-device wait on it is legal.
    cuda_check(cudaEventRecord(ctx->fork_events[dev0], ctx->capture_stream));
    for (int i = 1; i < ctx->all_devices.size(); ++i)
    {
        int dev = ctx->all_devices[i];
        cudaSetDevice(dev);
        cuda_check(cudaStreamWaitEvent(ctx->real_streams[dev], ctx->fork_events[dev0], 0));
    }
    cudaSetDevice(dev0);
}

void tp_graph_capture_end(uintptr_t tp_context)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    TORCH_CHECK(ctx->graph_capture, "tp_graph_capture_end: no active capture");

    int dev0 = ctx->all_devices[0];

    // Join every forked device stream back into the capture stream before ending it, so all
    // the recorded work is reachable from the stream the capture is ended on. Each join event is
    // recorded on its own device's stream (required) and waited on from the capture stream.
    for (int i = 1; i < ctx->all_devices.size(); ++i)
    {
        int dev = ctx->all_devices[i];
        cudaSetDevice(dev);
        cuda_check(cudaEventRecord(ctx->join_events[dev], ctx->real_streams[dev]));
        cudaSetDevice(dev0);
        cuda_check(cudaStreamWaitEvent(ctx->capture_stream, ctx->join_events[dev], 0));
    }

    cudaSetDevice(dev0);

    cudaGraph_t graph = nullptr;
    cudaError_t st = cudaStreamEndCapture(ctx->capture_stream, &graph);
    ctx->graph_capture = false;
    ctx->capture_stream = nullptr;
    cuda_check(st);

    if (ctx->graph_exec) { cudaGraphExecDestroy(ctx->graph_exec); ctx->graph_exec = nullptr; }
    if (ctx->graph) { cudaGraphDestroy(ctx->graph); ctx->graph = nullptr; }

    ctx->graph = graph;
    cuda_check(cudaGraphInstantiate(&ctx->graph_exec, graph, nullptr, nullptr, 0));
}

void tp_graph_launch(uintptr_t tp_context)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    TORCH_CHECK(ctx->graph_exec, "tp_graph_launch: no graph captured");

    // BUG 87: surface a spin-kernel timeout from the PREVIOUS replay before starting another one.
    // A deadlocked replay no longer hangs the process -- the kernel gives up, records what it was
    // waiting for, and this raises. Checking here rather than immediately after the launch is deliberate:
    // the timeout is only set once the kernel has actually spun for ~4 s, which is long after
    // cudaGraphLaunch returns.
    tp_raise_on_timeout(tp_context);

    int dev0 = ctx->all_devices[0];
    cudaSetDevice(dev0);
    cuda_check(cudaGraphLaunch(ctx->graph_exec, ctx->real_streams[dev0]));
}

bool tp_graph_ready(uintptr_t tp_context)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    return ctx->graph_exec != nullptr;
}

// BUG 87: report a spin-kernel timeout as an exception instead of hanging forever.
//
// Before this, a broken collective ordering pinned every GPU at 100% and produced no output at all,
// which is indistinguishable from a slow run and cost an entire session. The spin kernels now give up
// after TP_SPIN_MAX_ITERS and record what they were waiting for; this drains that record and raises.
void tp_raise_on_timeout(uintptr_t tp_context)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    int dev = -1;
    uint32_t seen = 0, want = 0, kind = 0;
    if (tp_check_timeout(ctx->tp_data, &dev, &seen, &want, &kind))
    {
        const char* what =
            (kind == 10) ? "broadcast source waiting for targets to finish reading (reverse edge)" :
            (kind == 11) ? "broadcast target waiting for source to publish (forward edge)" :
            (kind == 20) ? "gather source waiting for targets to finish reading (reverse edge)" :
            (kind == 21) ? "gather target waiting for source to publish (forward edge)" :
            (kind == 100) ? "barrier coordinator waiting for participants to arrive" :
            (kind == 101) ? "barrier participant waiting for release" : "unknown";
        TORCH_CHECK
        (
            false,
            "TP spin-kernel TIMEOUT on device ", dev, ": ", what,
            " (counter=", seen, ", wanted=", want, ", kind=", kind, ")",
            ". This is a collective-ordering deadlock, not a slow run."
        );
    }
}

// BUG 43: tell the C++ context which stream the (Python-driven) capture is running on, so
// the D2H in tp_broadcast is issued on a stream that belongs to the capture. Pass 0 to clear.
void tp_set_capture_stream(uintptr_t tp_context, uintptr_t stream)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    if (stream == 0)
    {
        ctx->capture_stream = nullptr;
        ctx->graph_capture = false;
        return;
    }
    ctx->capture_stream = reinterpret_cast<cudaStream_t> (stream);
    ctx->graph_capture = true;
}

// Diagnostic: 0 = not capturing, 1 = capture active, 2 = capture invalidated.
// Used to locate the first operation that invalidates a capture.
int tp_graph_capture_status(uintptr_t tp_context)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    if (!ctx->graph_capture) return 0;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaError_t e = cudaStreamGetCaptureInfo(ctx->capture_stream, &st, nullptr);
    if (e != cudaSuccess)
    {
        (void) cudaGetLastError();
        return -1;
    }
    if (st == cudaStreamCaptureStatusActive) return 1;
    if (st == cudaStreamCaptureStatusInvalidated) return 2;
    return 0;
}

//void tp_all_reduce_nccl
//(
//    uintptr_t tp_context,
//    const std::vector<torch::Tensor> &tensors
//)
//{
//    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
//
//    ncclGroupStart();
//
//    for (int i = 0; i < tensors.size(); ++i)
//    {
//        int dev = tensors[i].device().index();
//        int comms_i = ctx->comms_index[dev];
//
//        ncclAllReduce
//        (
//            tensors[i].data_ptr(),
//            tensors[i].data_ptr(),
//            tensors[i].numel(),
//            ncclFloat16,
//            ncclSum,
//            ctx->comms[comms_i],
//            ctx->streams[dev]
//        );
//    }
//
//    ncclGroupEnd();
//}

//void tp_all_reduce
//(
//    uintptr_t tp_context,
//    const std::vector<torch::Tensor> &tensors
//)

void tp_all_reduce
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &tensors,
    const std::vector<torch::Tensor> &residuals
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    size_t size = tensors[0].numel() * tensors[0].element_size();
    size_t num = tensors.size();

    // Reduction via host buffer

    for (int i = 0; i < num; ++i)
    {
        int dev = tensors[i].device().index();
        auto torch_stream = at::cuda::getStreamFromExternal(ctx->streams[dev], dev);
        cudaSetDevice(dev);
        at::cuda::setCurrentCUDAStream(torch_stream);

        if (i > 0)
        {
            int prev_dev = tensors[i - 1].device().index();

            // Copy host buffer to current residual

            cuda_check(cudaStreamWaitEvent
            (
                ctx->streams[dev],
                ctx->sync_events[prev_dev],
                0
            ));

            cuda_check(cudaMemcpyAsync
            (
                residuals[i].data_ptr(),
                ctx->pinned_temp[buffer],
                size,
                cudaMemcpyHostToDevice,
                ctx->streams[dev]
            ));
        }

        // Add current tensor to current residual

        residuals[i].add_(tensors[i]);

        // Copy current residual to host buffer

        cuda_check(cudaMemcpyAsync
        (
            ctx->pinned_temp[buffer],
            residuals[i].data_ptr(),
            size,
            cudaMemcpyDeviceToHost,
            ctx->streams[dev]
        ));

        cuda_check(cudaEventRecord
        (
            ctx->sync_events[dev],
            ctx->streams[dev]
        ));
    }

    // Broadcast result

    int last_dev = tensors[num - 1].device().index();

    for (int i = 0; i < num - 1; ++i)
    {
        int dev = tensors[i].device().index();
        cudaSetDevice(dev);

        cuda_check(cudaStreamWaitEvent
        (
            ctx->streams[dev],
            ctx->sync_events[last_dev],
            0
        ));
        cuda_check(cudaMemcpyAsync
        (
            residuals[i].data_ptr(),
            ctx->pinned_temp[buffer],
            size,
            cudaMemcpyHostToDevice,
            ctx->streams[dev]
        ));
    }
}
