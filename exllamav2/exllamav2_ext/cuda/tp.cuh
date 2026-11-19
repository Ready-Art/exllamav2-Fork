#ifndef _tp_cuh
#define _tp_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#define MAX_SYNC_DEVICES 8
#define MAX_SYNC_STAGES 3

// BUG 87: spin-kernel timeout. A deadlocked collective used to hang the process forever with every
// GPU pinned at 100% and no output at all, which is indistinguishable from a slow run. Each spin kernel
// now gives up after this many backoff iterations (~4 s at the 2000 ns ceiling) and records what it was
// waiting for, so a broken ordering reports instead of hanging.
#define TP_SPIN_MAX_ITERS 2000000

struct ExtTPData
{
    uint32_t sync[MAX_SYNC_STAGES][MAX_SYNC_DEVICES];
    uint32_t next_stage;

    // ---------------------------------------------------------------------------------------
    // Coordinator-based spin barrier (ported from exllamav3's pg_barrier_inner).
    //
    // Replaces a full mesh of CUDA events (N records + N*(N-1) waits = 16 event ops per barrier,
    // 1200 barriers per decode token = ~19200 event ops/token).
    // ---------------------------------------------------------------------------------------
    uint32_t barrier_arrive[MAX_SYNC_DEVICES];
    uint32_t barrier_release[MAX_SYNC_DEVICES];

    // ---------------------------------------------------------------------------------------
    // Per-collective sequence counters (BUG 86/87), the exllamav3 approach.
    //
    // The per-collective barrier cannot simply be deleted: the source buffer is reused by the next
    // module, so there are TWO hazards, not one.
    //
    //   forward : source finished writing  ->  target may read
    //   reverse : target finished reading  ->  source may overwrite
    //
    // CUDA events express both, but an event wait on work recorded OUTSIDE the capture region is
    // illegal inside it ("dependency created on uncaptured work in another stream"). That is why the
    // event-based reverse edge could not be captured.
    //
    // These counters are plain uint32 in host-pinned MAPPED memory. They are published and polled by
    // 1-thread kernels, and a kernel launch IS capturable. That is the whole trick: replace every
    // event record with a publish kernel and every event wait with a spin kernel.
    //
    // Monotonic, never reset, so a captured graph replays correctly: on replay N the counters simply
    // advance from wherever replay N-1 left them. The counters MUST persist across collectives: the
    // reverse edge exists precisely to stop a source overwriting its buffer before the target finishes
    // reading the PREVIOUS collective's contents. Giving each collective a fresh counter (the BUG 88
    // slot experiment) breaks that edge -- a fresh counter starts at 0, so the reverse check passes
    // immediately and the overwrite races the read. Do not reintroduce slots.
    //
    //   bc_seq[s]   : number of times device s has published "my data is ready"
    //   rd_seq[r][s]: number of times reader r has published "I finished reading data from s"
    //
    // BUG 87: rd_seq is indexed by (reader, source), not by reader alone. With a single rd_seq[r],
    // every source compared that one counter against its OWN bc_seq[s]; the counters cross-contaminated
    // and the protocol deadlocked on the second collective:
    //
    //   broadcast #1 from dev0 : bc_seq[0]=1, rd_seq[1]=rd_seq[2]=rd_seq[3]=1
    //   broadcast #2 from dev1 : bc_seq[1]=1, but dev2 waits for bc_seq[1] >= rd_seq[2]+1 = 2. HANG.
    //
    // Indexing by (reader, source) gives every ordered pair its own counter, so a source only ever
    // compares against reads of ITS OWN data, and each collective advances every pair it touches by
    // exactly one on both sides.
    //
    // BUG 89: the counters only order the copy if the copy and the wait/publish kernels run on the
    // SAME stream per device. torch's copy_ dispatches on torch's current stream, so the TP code must
    // setCurrentCUDAStream(tp_stream) before issuing it. Without that the counters order nothing and
    // corruption accumulates over tokens.
    // ---------------------------------------------------------------------------------------
    uint32_t bc_seq[MAX_SYNC_DEVICES];
    uint32_t rd_seq[MAX_SYNC_DEVICES][MAX_SYNC_DEVICES];

    // BUG 87 diagnostics. A spin kernel that exceeds TP_SPIN_MAX_ITERS sets timeout_flag[dev] and
    // records the counter value it saw and the threshold it wanted, then returns instead of spinning
    // forever. The host checks these with tp_check_timeout() and raises, so a broken ordering produces
    // an actionable error rather than wiping out the whole run.
    uint32_t timeout_flag[MAX_SYNC_DEVICES];
    uint32_t timeout_seen[MAX_SYNC_DEVICES];
    uint32_t timeout_want[MAX_SYNC_DEVICES];
    uint32_t timeout_kind[MAX_SYNC_DEVICES];
};

void init_tp_data(ExtTPData* tp_data);

// Legacy full-mesh event barrier (kept for A/B via EXL2_TP_SPIN_BARRIER=0).
void cross_device_barrier_cuda
(
    cudaStream_t stream,
    uint32_t* sync,
    uint32_t* sync_next,
    uint32_t num_devices,
    uint32_t device
);

// Coordinator-based spin barrier. Launches one 1-thread kernel on `stream`; the kernel returns once
// every device in `device_mask` has arrived.
void tp_barrier_cuda
(
    cudaStream_t stream,
    ExtTPData* tp_data,
    uint32_t device_mask,
    int this_device,
    int coordinator_device
);

// Publish "I am done with my half of the collective": increments *counter.
void tp_publish_cuda(cudaStream_t stream, uint32_t* counter);

// Wait until *counter >= *target_ptr. Both are device pointers so the threshold is read on the device
// and is therefore correct on every graph replay.
void tp_wait_ge_cuda
(
    cudaStream_t stream,
    uint32_t* counter,
    uint32_t* target_ptr,
    ExtTPData* tp_data,
    int this_device,
    int kind
);

// Wait until *counter > *target_ptr, i.e. *counter >= *target_ptr + 1.
//
// This is the CONSUMER side of a collective. A target has consumed rd_seq[target][source] broadcasts;
// it must wait for the source to publish broadcast number rd_seq[target][source] + 1. Comparing the
// source's counter against the source's own counter (as an earlier version did) is always true and
// silently removes the ordering entirely -- which is what produced degenerate output.
void tp_wait_gt_cuda
(
    cudaStream_t stream,
    uint32_t* counter,
    uint32_t* target_ptr,
    ExtTPData* tp_data,
    int this_device,
    int kind
);

// BUG 87: returns 1 and fills the out params if any spin kernel timed out since the last call.
int tp_check_timeout(ExtTPData* tp_data, int* dev, uint32_t* seen, uint32_t* want, uint32_t* kind);

#endif
