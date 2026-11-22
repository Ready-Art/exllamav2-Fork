// EXL2.5 fast-lane instantiations: 5/6/8-bit.

#include "exl25_gemm.cuh"

template <int bits, bool rw, bool mulr>
static fp_exl25_gemm_kernel pick_m_kn(int max_m, int kn)
{
    #define EXL25_PICK_M(M) \
        if (max_m <= M) { \
            if (kn == 128) return exl25_gemm_kernel<bits, M, 128, rw, mulr>; \
            if (kn == 256) return exl25_gemm_kernel<bits, M, 256, rw, mulr>; \
            return NULL; \
        }
    EXL25_PICK_M(1)
    EXL25_PICK_M(2)
    EXL25_PICK_M(4)
    EXL25_PICK_M(8)
    EXL25_PICK_M(16)
    #undef EXL25_PICK_M
    return NULL;
}

template <int bits>
static fp_exl25_gemm_kernel pick_rw(int max_m, int kn, bool rw, bool mulr)
{
    if (!rw && !mulr) return pick_m_kn<bits, false, false>(max_m, kn);
    if ( rw && !mulr) return pick_m_kn<bits, true,  false>(max_m, kn);
    if (!rw &&  mulr) return pick_m_kn<bits, false, true >(max_m, kn);
    return pick_m_kn<bits, true, true>(max_m, kn);
}

fp_exl25_gemm_kernel pick_exl25_kernel_b(int bits, int max_m, int kn, bool r_weights, bool mul_r_weights)
{
    if (bits == 5) return pick_rw<5>(max_m, kn, r_weights, mul_r_weights);
    if (bits == 6) return pick_rw<6>(max_m, kn, r_weights, mul_r_weights);
    if (bits == 8) return pick_rw<8>(max_m, kn, r_weights, mul_r_weights);
    return NULL;
}
