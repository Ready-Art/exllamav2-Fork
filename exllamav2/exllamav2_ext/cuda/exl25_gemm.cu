// EXL2.5 fast-lane dispatch + activation transform (reconstruct/cuBLAS path).

#include "exl25_gemm.cuh"
#include "exl25_transform.cuh"
#include "q_matrix.cuh"
#include "util.cuh"
#include "graph.cuh"

#include <mutex>
#include <unordered_map>

// Eligibility: returns the k-block size to use, or 0 to fall back to the legacy EXL2
// kernel (uniform tensors keep q_groups/q_group_map, so the legacy path stays correct).
int exl25_serve_kn(const QMatrix* b, int size_m, int size_n, int size_k)
{
    // Kill switch / A/B gate: EXL2_EXL25=0 serves PLAIN uniform tensors with the legacy
    // kernel (they keep valid q_groups/q_group_map, so legacy is correct, just slower).
    // had/awq tensors are exempt: legacy cannot apply the activation transform, so they
    // always take the fast lane regardless of the gate.
    static int enabled = -1;
    if (enabled < 0) { const char* e = getenv("EXL2_EXL25"); enabled = (e && atoi(e) == 0) ? 0 : 1; }
    if (!enabled && !b->had_k && !b->cuda_awq_scale) return 0;

    if (b->is_gptq || b->exl25_bits == 0) return 0;
    int bits = b->exl25_bits;
    if (bits < 2 || bits > 8 || bits == 7) return 0;
    int gs = b->exl25_groupsize;
    if (gs < 32 || gs > 256 || gs % 32 != 0) return 0;
    if (size_k % gs != 0) return 0;
    if (b->had_k != 0 && b->had_k != 128) return 0;
    if (b->had_k && size_k % 128 != 0) return 0;

    bool ok128 = (size_k % 128 == 0) && gs <= 128;
    bool ok256 = (size_k % 256 == 0) && gs <= 256;
    if (b->had_k) { /* both >= 128, fine */ }
    if (!ok128 && !ok256) return 0;
    if (!ok256) return 128;
    if (!ok128) return 256;

    // Occupancy heuristic: prefer 128 when the 256-grid would underfill the GPU
    // (decode m=1, narrow n), 256 when there is already enough parallelism (less
    // preload/scale overhead per weight).
    int blocks256 = DIVIDE(size_n, 1024) * (size_k / 256);
    int target = 164;  // ~2 waves on 82-SM Ampere
    if (blocks256 < target) return 128;
    return 256;
}

void exl25_gemm_cuda_part
(
    cudaStream_t stream,
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    int m_count,
    bool clear,
    const half* r_weights,
    int r_weights_stride,
    bool mul_r_weights,
    Graph* graph,
    int label,
    int kn
)
{
    fp_exl25_gemm_kernel kernel = pick_exl25_kernel(b->exl25_bits, m_count, kn, r_weights != NULL, mul_r_weights);
    if (!kernel) return;

    dim3 blockDim, gridDim;
    blockDim.x = kn;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = DIVIDE(size_n, kn * 4);
    gridDim.y = DIVIDE(size_m, m_count);
    gridDim.z = DIVIDE(size_k, kn);

    kernel<<<gridDim, blockDim, 0, stream>>>
    (
        a,
        b->cuda_q_weight,
        b->cuda_q_scale,
        b->cuda_q_scale_max,
        c,
        size_m,
        size_n,
        size_k,
        b->exl25_groupsize,
        b->cuda_awq_scale,
        b->had_k,
        clear,
        r_weights,
        r_weights_stride
    );
    if (graph) graph->attach_label(stream, label, 3 /* GEMM_EXL2 */);
}

// ---- activation transform (awq inverse-scale + Hadamard butterfly), out-of-place ----
//
// Used by the reconstruct+cuBLAS path (m > MAX_Q_GEMM_ROWS prefill): cuBLAS must see
// (a * S^-1 * H) to match the pre-rotated stored weights. The small-m fast lane does the
// same transform in-kernel on the shared-memory preload.

__global__ void exl25_transform_kernel
(
    const half* __restrict__ a,
    half* __restrict__ out,
    int size_k,
    const half* __restrict__ awq_scale,
    int had_k
)
{
    __shared__ half sh[128];
    int t = threadIdx.x;
    int idx = blockIdx.y * size_k + blockIdx.x * 128 + t;

    half v = a[idx];
    if (awq_scale) v = __hmul(v, awq_scale[blockIdx.x * 128 + t]);
    sh[t] = v;
    __syncthreads();

    if (had_k)
    {
        #pragma unroll
        for (int h = 1; h < 128; h <<= 1)
        {
            half x = sh[t];
            half y = sh[t ^ h];
            __syncthreads();
            sh[t] = (t & h) ? __hsub(y, x) : __hadd(x, y);
            __syncthreads();
        }
    }
    out[idx] = sh[t];
}

// Grow-only per-device scratch (process lifetime; eager prefill path only -- never
// allocated under graph capture: the fork's graphs stay <= MAX_Q_GEMM_ROWS rows, which
// take the in-kernel-transform lane).
static std::mutex exl25_scratch_mutex;
static std::unordered_map<int, std::pair<half*, size_t>> exl25_scratch;

half* exl25_get_scratch(int device, size_t numel)
{
    std::lock_guard<std::mutex> lock(exl25_scratch_mutex);
    auto it = exl25_scratch.find(device);
    if (it != exl25_scratch.end() && it->second.second >= numel) return it->second.first;

    cudaStream_t stream = NULL;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cs);
    if (cs != cudaStreamCaptureStatusNone)
    {
        fprintf(stderr, " ## EXL2.5 transform scratch growth under graph capture (m > %d captured?) -- refusing\n", MAX_Q_GEMM_ROWS);
        return NULL;
    }

    half* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, numel * sizeof(half));
    if (err != cudaSuccess)
    {
        fprintf(stderr, " ## EXL2.5 transform scratch alloc failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    if (it != exl25_scratch.end()) cudaFree(it->second.first);
    exl25_scratch[device] = { ptr, numel };
    return ptr;
}

void exl25_transform_a
(
    cudaStream_t stream,
    const half* a,
    half* out,
    int size_m,
    int size_k,
    const half* awq_scale,
    int had_k
)
{
    dim3 blockDim(128), gridDim(size_k / 128, size_m);
    exl25_transform_kernel<<<gridDim, blockDim, 0, stream>>>(a, out, size_k, awq_scale, had_k);
}
