#include "kernel_micro.h"

#define FMLA(A, B, alpha) __asm__ volatile(         \
    /********** K=0 **********/                     \
                                                    \
    "ld1 {v8.4S-v9.4S}, [%[A]], #32\n\t"            \
                                                    \
    "ld1 {v0.4S-v3.4S}, [%[B]], #64\n\t"            \
    "ld1 {v4.4S-v7.4S}, [%[B]], #64\n\t"            \
                                                    \
    /* mult alpha to A */                           \
    "dup v10.4S, %w[alpha]\n\t"                     \
    "fmul v8.4S, v8.4S, v10.4S\n\t"                 \
    "fmul v9.4S, v9.4S, v10.4S\n\t"                 \
                                                    \
    /* MAC: C += alpha_A @ B */                     \
    "fmla v16.4S, v0.4S, v8.4S[0]\n\t"              \
    "fmla v17.4S, v1.4S, v8.4S[0]\n\t"              \
    "fmla v18.4S, v2.4S, v8.4S[0]\n\t"              \
    "fmla v19.4S, v3.4S, v8.4S[0]\n\t"              \
                                                    \
    "fmla v20.4S, v0.4S, v8.4S[1]\n\t"              \
    "fmla v21.4S, v1.4S, v8.4S[1]\n\t"              \
    "fmla v22.4S, v2.4S, v8.4S[1]\n\t"              \
    "fmla v23.4S, v3.4S, v8.4S[1]\n\t"              \
                                                    \
    "fmla v24.4S, v0.4S, v8.4S[2]\n\t"              \
    "fmla v25.4S, v1.4S, v8.4S[2]\n\t"              \
    "fmla v26.4S, v2.4S, v8.4S[2]\n\t"              \
    "fmla v27.4S, v3.4S, v8.4S[2]\n\t"              \
                                                    \
    "fmla v28.4S, v0.4S, v8.4S[3]\n\t"              \
    "fmla v29.4S, v1.4S, v8.4S[3]\n\t"              \
    "fmla v30.4S, v2.4S, v8.4S[3]\n\t"              \
    "fmla v31.4S, v3.4S, v8.4S[3]\n\t"              \
                                                    \
    "fmla v16.4S, v4.4S, v9.4S[0]\n\t"              \
    "fmla v17.4S, v5.4S, v9.4S[0]\n\t"              \
    "fmla v18.4S, v6.4S, v9.4S[0]\n\t"              \
    "fmla v19.4S, v7.4S, v9.4S[0]\n\t"              \
                                                    \
    "fmla v20.4S, v4.4S, v9.4S[1]\n\t"              \
    "fmla v21.4S, v5.4S, v9.4S[1]\n\t"              \
    "fmla v22.4S, v6.4S, v9.4S[1]\n\t"              \
    "fmla v23.4S, v7.4S, v9.4S[1]\n\t"              \
                                                    \
    "fmla v24.4S, v4.4S, v9.4S[2]\n\t"              \
    "fmla v25.4S, v5.4S, v9.4S[2]\n\t"              \
    "fmla v26.4S, v6.4S, v9.4S[2]\n\t"              \
    "fmla v27.4S, v7.4S, v9.4S[2]\n\t"              \
                                                    \
    "fmla v28.4S, v4.4S, v9.4S[3]\n\t"              \
    "fmla v29.4S, v5.4S, v9.4S[3]\n\t"              \
    "fmla v30.4S, v6.4S, v9.4S[3]\n\t"              \
    "fmla v31.4S, v7.4S, v9.4S[3]\n\t"              \
                                                    \
    ::[A] "r"(A), [B] "r"(B), [alpha] "r"(alpha)    \
)


void sgemm_micro_kernel_neon_4x16(
    int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC)
{
    /**
     * ASSUME
     * k is multiple of 8
     * beta == 1
     * incRowC == 1
     */

    // A is column-wise ordered buffer
    // B is row-wise ordered buffer

    int i, j, k;

    // INIT AB = C
    __asm__ volatile(
        // load C to v16-31 (memory-friendly order)
        "ld1 {v16.4S-v19.4S}, [x0], %[incColC]\n\t"
        "ld1 {v20.4S-v23.4S}, [x0], %[incColC]\n\t"
        "ld1 {v24.4S-v27.4S}, [x0], %[incColC]\n\t"
        "ld1 {v28.4S-v31.4S}, [x0], %[incColC]\n\t"
        ::
        [C] "r"(C),
        [incColC] "r"(incColC * 4)
    );

    // COMPUTE C <- C + alpha * AB
    for (k = 0; k < kc/8; k++)
    {
        FMLA(A, B, alpha);
        A += 4*2;
        B += 16*2;
        FMLA(A, B, alpha);
        A += 4*2;
        B += 16*2;
        FMLA(A, B, alpha);
        A += 4*2;
        B += 16*2;
        FMLA(A, B, alpha);
        A += 4*2;
        B += 16*2;
    }

    __asm__ volatile(
        // store C
        "st1 {v16.4S-v19.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v20.4S-v23.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v24.4S-v27.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v28.4S-v31.4S}, [%[C]], %[incColC]\n\t"
        
        ::[C] "r"(C), [incColC] "r"(incColC * 4)
    );

}
