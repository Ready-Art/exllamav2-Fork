// Standalone parity + perf bench for the EXPERIMENTAL HMMA INT8 verify kernel
// (int8_hmma.cuh). NOT wired into the extension build/dispatch — dormant source.
// Build (from repo root):
//   nvcc -arch=sm_86 -O3 -I exllamav2/exllamav2_ext/cuda \
//        -o bench_hmma mtp/int8_hmma_bench.cu exllamav2/exllamav2_ext/cuda/int8_decode.cu
// (int8_decode.cu provides int8_merge_cuda for the split-KV merge.)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "int8_hmma.cuh"

void int8_merge_cuda(float* po, float* pl, half* o, int num_head_rows,
                     int batch_size, int num_splits, int head_dim, cudaStream_t stream);

static float timeit(void (*launch)(const half*, const int8_t*, const int8_t*, const half*,
                                   const half*, float*, float*, const int*, int, int, int,
                                   int, int, int, float),
                    const half* q, const int8_t* k8, const int8_t* v8, const half* ks,
                    const half* vs, float* po, float* pl, const int* kvl,
                    int depth, int qlen, int nq, int nkv, int splits, int B, float sc)
{
    for (int i = 0; i < 10; i++)
        launch(q, k8, v8, ks, vs, po, pl, kvl, depth, qlen, nq, nkv, splits, B, sc);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        launch(q, k8, v8, ks, vs, po, pl, kvl, depth, qlen, nq, nkv, splits, B, sc);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / 100;
}

template <int HD, int TK, int NQ, int HSPLIT>
void launch_hmma(const half* q, const int8_t* k8, const int8_t* v8, const half* ks,
                 const half* vs, float* po, float* pl, const int* kvl,
                 int depth, int qlen, int nq, int nkv, int splits, int B, float sc)
{
    int total = nkv * splits * B;
    int smem = 2 * TK * HD * 2;
    cudaFuncSetAttribute(int8_verify_hmma_kernel<HD, TK, NQ, HSPLIT>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int8_verify_hmma_kernel<HD, TK, NQ, HSPLIT><<<total, (NQ / 16) * HSPLIT * 32, smem>>>(
        q, k8, v8, ks, vs, po, pl, kvl, nullptr, depth, qlen, nq, nkv, splits, B, sc);
}

int main()
{
    const int NQ = 8, NKV = 1, HD = 512, QL = 4, B = 1;
    const int HR = NQ * QL;
    float sc = 1.0f / sqrtf((float)HD) * 1.4426950408889634f;
    int maxkv = 25600;
    int depth = ((maxkv + 127) / 128) * 128;
    half *q, *ks, *vs, *o;
    int8_t *k8, *v8;
    int* kvl;
    float *po, *pl;
    cudaMalloc(&q, B * QL * NQ * HD * 2);
    cudaMalloc(&k8, (size_t)B * depth * NKV * HD);
    cudaMalloc(&v8, (size_t)B * depth * NKV * HD);
    cudaMalloc(&ks, (size_t)B * depth * NKV * 2);
    cudaMalloc(&vs, (size_t)B * depth * NKV * 2);
    cudaMalloc(&kvl, 4);
    cudaMalloc(&o, B * HR * HD * 2);
    cudaMalloc(&po, (size_t)B * 256 * HR * HD * 4);
    cudaMalloc(&pl, (size_t)B * 256 * HR * 2 * 4);

    std::vector<half> hq(B * QL * NQ * HD);
    std::vector<int8_t> hk((size_t)B * depth * NKV * HD), hv(hk.size());
    std::vector<half> hks((size_t)B * depth * NKV), hvs(hks.size());
    srand(7);
    auto rf = []() { return (rand() / (float)RAND_MAX - 0.5f) * 2.0f; };
    for (auto& x : hq) x = __float2half(rf() * 0.5f);
    for (size_t i = 0; i < hk.size(); i++) { hk[i] = (int8_t)(rf() * 60); hv[i] = (int8_t)(rf() * 60); }
    for (size_t i = 0; i < hks.size(); i++) { hks[i] = __float2half(0.012f); hvs[i] = __float2half(0.012f); }
    cudaMemcpy(q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(k8, hk.data(), hk.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(v8, hv.data(), hv.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ks, hks.data(), hks.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(vs, hvs.data(), hvs.size() * 2, cudaMemcpyHostToDevice);

    // CPU reference for parity at kv=4096
    int kv = 4096;
    cudaMemcpy(kvl, &kv, 4, cudaMemcpyHostToDevice);
    int splits = 64;
    launch_hmma<HD, 16, HR, 4>(q, k8, v8, ks, vs, po, pl, kvl, depth, QL, NQ, NKV, splits, B, sc);
    int8_merge_cuda(po, pl, o, HR, B, splits, HD, 0);
    cudaDeviceSynchronize();
    std::vector<half> ho(B * HR * HD);
    cudaMemcpy(ho.data(), o, ho.size() * 2, cudaMemcpyDeviceToHost);
    // reference: dequant K/V, per-row causal attention
    double maxerr = 0, dot = 0, na = 0, nb = 0;
    for (int r = 0; r < QL; r++)
    {
        int caus = kv - QL + r + 1;
        for (int h = 0; h < NQ; h++)
        {
            std::vector<float> acc(HD, 0.0f);
            float m = -1e30f, l = 0.0f;
            std::vector<float> scores(caus);
            for (int t = 0; t < caus; t++)
            {
                float s = 0;
                for (int d = 0; d < HD; d++)
                {
                    float qv = __half2float(hq[(r * NQ + h) * HD + d]) * sc;
                    float kvv = (float)hk[t * HD + d] * 0.012f;
                    s += qv * kvv;
                }
                scores[t] = s;
                m = std::max(m, s);
            }
            for (int t = 0; t < caus; t++)
            {
                float p = exp2f(scores[t] - m);
                l += p;
                for (int d = 0; d < HD; d++)
                    acc[d] += p * ((float)hv[t * HD + d] * 0.012f);
            }
            int hr = r * NQ + h;
            for (int d = 0; d < HD; d++)
            {
                float ref = acc[d] / l;
                float got = __half2float(ho[hr * HD + d]);
                dot += ref * got; na += ref * ref; nb += got * got;
                maxerr = std::max(maxerr, (double)fabs(ref - got));
            }
        }
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("PARITY kv=%d cos=%.6f maxerr=%.5f %s\n", kv, cos, maxerr,
           cos > 0.999 ? "PASS" : "FAIL");

    // perf sweep at kv=25600 across TK / HSPLIT / splits
    int kv2 = 25600;
    cudaMemcpy(kvl, &kv2, 4, cudaMemcpyHostToDevice);
#define SWEEP(TK, HS) do { \
    float best = 1e9; int bs = 0; \
    for (int sp : {64, 128, 256}) { \
        float ms = timeit(launch_hmma<HD, TK, HR, HS>, q, k8, v8, ks, vs, po, pl, kvl, \
                          depth, QL, NQ, NKV, sp, B, sc); \
        if (ms < best) { best = ms; bs = sp; } } \
    float gbs = (float)kv2 * NKV * HD * 2 / 1e9f / (best / 1000); \
    printf("TK=%d HS=%d splits=%d %.3f ms  %.0f GB/s\n", TK, HS, bs, best, gbs); \
} while (0)
    SWEEP(16, 4);
    SWEEP(16, 8);
    SWEEP(16, 16);
    SWEEP(32, 8);
    SWEEP(32, 16);
#undef SWEEP
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA ERR %s\n", cudaGetErrorString(e)); return 1; }
    printf("HMMA_BENCH_DONE\n");
    return 0;
}
