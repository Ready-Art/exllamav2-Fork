#include "rope.cuh"
#include "tp.cuh"
#include "util.cuh"

#define TP_SYNC_MIN_SLEEP 20
#define TP_SYNC_MAX_SLEEP 2000

void init_tp_data(ExtTPData* tp_data)
{
    for (int i = 0; i < MAX_SYNC_STAGES; ++i)
        for (int j = 0; j < MAX_SYNC_DEVICES; ++j)
            tp_data->sync[i][j] = 0;
    tp_data->next_stage = 0;

    for (int i = 0; i < MAX_SYNC_DEVICES; ++i)
    {
        tp_data->barrier_arrive[i] = 0;
        tp_data->barrier_release[i] = 0;
        tp_data->bc_seq[i] = 0;
        for (int j = 0; j < MAX_SYNC_DEVICES; ++j)
            tp_data->rd_seq[i][j] = 0;
        tp_data->timeout_flag[i] = 0;
        tp_data->timeout_seen[i] = 0;
        tp_data->timeout_want[i] = 0;
        tp_data->timeout_kind[i] = 0;
    }
}

__device__ __forceinline__ uint32_t tp_ldg_cv_u32(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void tp_stg_wt_u32(uint32_t* p, uint32_t v)
{
    asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// ---------------------------------------------------------------------------------------------
// Coordinator-based spin barrier (ported from exllamav3 pg_barrier_inner).
// ---------------------------------------------------------------------------------------------

__global__ void tp_barrier_kernel
(
    ExtTPData* __restrict__ tp_data,
    const uint32_t device_mask,
    const int this_device,
    const int coordinator_device
)
{
    if (threadIdx.x) return;

    uint32_t* arrive = tp_data->barrier_arrive;
    uint32_t* release = tp_data->barrier_release;

    if (this_device == coordinator_device)
    {
        uint32_t pending = device_mask & ~(1u << this_device);
        uint32_t sleep = TP_SYNC_MIN_SLEEP;
        uint32_t iters = 0;
        while (pending)
        {
            uint32_t pending_before = pending;
            for (int i = 0; i < MAX_SYNC_DEVICES; ++i)
            {
                if (!(pending & (1u << i))) continue;
                if (tp_ldg_cv_u32(arrive + i) != tp_ldg_cv_u32(release + i))
                    pending &= ~(1u << i);
            }
            if (pending == pending_before)
            {
                __nanosleep(sleep);
                if (sleep < TP_SYNC_MAX_SLEEP) sleep <<= 1;
                if (++iters > TP_SPIN_MAX_ITERS)
                {
                    tp_stg_wt_u32(tp_data->timeout_flag + this_device, 1u);
                    tp_stg_wt_u32(tp_data->timeout_seen + this_device, pending);
                    tp_stg_wt_u32(tp_data->timeout_want + this_device, device_mask);
                    tp_stg_wt_u32(tp_data->timeout_kind + this_device, 100u);
                    return;
                }
            }
            else
            {
                sleep = TP_SYNC_MIN_SLEEP;
                iters = 0;
            }
        }

        for (int i = 0; i < MAX_SYNC_DEVICES; ++i)
        {
            if (!(device_mask & (1u << i)) || i == this_device) continue;
            tp_stg_wt_u32(release + i, tp_ldg_cv_u32(arrive + i));
        }
    }
    else
    {
        const uint32_t s = tp_ldg_cv_u32(release + this_device) + 1;
        tp_stg_wt_u32(arrive + this_device, s);

        uint32_t sleep = TP_SYNC_MIN_SLEEP;
        uint32_t iters = 0;
        while (tp_ldg_cv_u32(release + this_device) != s)
        {
            __nanosleep(sleep);
            if (sleep < TP_SYNC_MAX_SLEEP) sleep <<= 1;
            if (++iters > TP_SPIN_MAX_ITERS)
            {
                tp_stg_wt_u32(tp_data->timeout_flag + this_device, 1u);
                tp_stg_wt_u32(tp_data->timeout_seen + this_device,
                              tp_ldg_cv_u32(release + this_device));
                tp_stg_wt_u32(tp_data->timeout_want + this_device, s);
                tp_stg_wt_u32(tp_data->timeout_kind + this_device, 101u);
                return;
            }
        }
    }
}

void tp_barrier_cuda
(
    cudaStream_t stream,
    ExtTPData* tp_data,
    uint32_t device_mask,
    int this_device,
    int coordinator_device
)
{
    tp_barrier_kernel<<<1, 32, 0, stream>>>
    (
        tp_data,
        device_mask,
        this_device,
        coordinator_device
    );
}

// ---------------------------------------------------------------------------------------------
// Sequence counters (BUG 86).
//
// These replace cudaEventRecord / cudaStreamWaitEvent for the per-collective ordering. An event wait
// on work recorded outside the capture region is illegal inside it; a kernel that polls mapped memory
// is not. That is the only reason this exists.
//
// The publish is a store, the wait is a spin on a load. Both are single-thread kernels, so they are
// cheap and, more importantly, capturable.
// ---------------------------------------------------------------------------------------------

__global__ void tp_publish_kernel(uint32_t* counter)
{
    if (threadIdx.x) return;
    tp_stg_wt_u32(counter, tp_ldg_cv_u32(counter) + 1);
}

void tp_publish_cuda(cudaStream_t stream, uint32_t* counter)
{
    tp_publish_kernel<<<1, 32, 0, stream>>>(counter);
}

// BUG 87: shared spin body with a timeout. `strict` selects >= vs > (i.e. threshold +  varieties).
__device__ __forceinline__ void tp_spin_body
(
    uint32_t* counter,
    const uint32_t* target,
    ExtTPData* tp_data,
    const int this_device,
    const uint32_t kind,
    const bool strict
)
{
    uint32_t t = tp_ldg_cv_u32(target);
    if (strict) t += 1;

    uint32_t sleep = TP_SYNC_MIN_SLEEP;
    uint32_t iters = 0;
    while (tp_ldg_cv_u32(counter) < t)
    {
        __nanosleep(sleep);
        if (sleep < TP_SYNC_MAX_SLEEP) sleep <<= 1;
        if (++iters > TP_SPIN_MAX_ITERS)
        {
            tp_stg_wt_u32(tp_data->timeout_flag + this_device, 1u);
            tp_stg_wt_u32(tp_data->timeout_seen + this_device, tp_ldg_cv_u32(counter));
            tp_stg_wt_u32(tp_data->timeout_want + this_device, t);
            tp_stg_wt_u32(tp_data->timeout_kind + this_device, kind);
            return;
        }
    }
}

__global__ void tp_wait_ge_kernel
(
    uint32_t* counter,
    const uint32_t* target,
    ExtTPData* tp_data,
    const int this_device,
    const uint32_t kind
)
{
    if (threadIdx.x) return;
    tp_spin_body(counter, target, tp_data, this_device, kind, false);
}

__global__ void tp_wait_gt_kernel
(
    uint32_t* counter,
    const uint32_t* target,
    ExtTPData* tp_data,
    const int this_device,
    const uint32_t kind
)
{
    if (threadIdx.x) return;
    tp_spin_body(counter, target, tp_data, this_device, kind, true);
}

void tp_wait_ge_cuda
(
    cudaStream_t stream,
    uint32_t* counter,
    uint32_t* target_ptr,
    ExtTPData* tp_data,
    int this_device,
    int kind
)
{
    tp_wait_ge_kernel<<<1, 32, 0, stream>>>
    (
        counter, target_ptr, tp_data, this_device, (uint32_t) kind
    );
}

void tp_wait_gt_cuda
(
    cudaStream_t stream,
    uint32_t* counter,
    uint32_t* target_ptr,
    ExtTPData* tp_data,
    int this_device,
    int kind
)
{
    tp_wait_gt_kernel<<<1, 32, 0, stream>>>
    (
        counter, target_ptr, tp_data, this_device, (uint32_t) kind
    );
}

int tp_check_timeout(ExtTPData* tp_data, int* dev, uint32_t* seen, uint32_t* want, uint32_t* kind)
{
    for (int i = 0; i < MAX_SYNC_DEVICES; ++i)
    {
        if (tp_data->timeout_flag[i])
        {
            if (dev) *dev = i;
            if (seen) *seen = tp_data->timeout_seen[i];
            if (want) *want = tp_data->timeout_want[i];
            if (kind) *kind = tp_data->timeout_kind[i];
            tp_data->timeout_flag[i] = 0;
            return 1;
        }
    }
    return 0;
}

// Legacy full-mesh event barrier. Unused.
void cross_device_barrier_cuda
(
    cudaStream_t stream,
    uint32_t* sync,
    uint32_t* sync_next,
    uint32_t num_devices,
    uint32_t device
)
{
}
