#include "kernel_8x8.h"

void sgemm_kernel_8x8_naive(
    int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC
)
{
    // A is packed columnwise
    // B is packed rowwise
    float AB[NR * MR];

    int i, j, k;

    // init AB
    for (i=0; i<MR; i++) {
        for (j=0; j<NR; j++) {
            AB[i*NR+j] = 0;
        }
    }

    // calc AB
    for (k=0; k<kc; k++) {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                AB[i*NR+j] += A[i] * B[j];
            }
        }
        A += 8;
        B += 8;
    }

    // update C <- beta * C
    if (beta == 0.0) {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                C[i*incRowC+j*incColC] = 0;
            }
        }
    }
    else if (beta != 1.0) {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

    // update C <- alpha * AB
    if (alpha == 1.0) {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                C[i*incColC+j*incRowC] += AB[i*NR+j];
            }
        }
    }
    else {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                C[i*incColC+j*incRowC] += alpha * AB[i*NR+j];
            }
        }
    }
}

void sgemm_kernel_8x8_neon_fullyunroll(
    int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC
) {
    /**
     * ASSUME
     * beta == 1
     * incRowC == 1
    */

    #define KC_UNROLL 8

    int kc_unroll = kc / KC_UNROLL;
    int kc_left = kc % KC_UNROLL;

    int i, j, k;

   // INIT AB = C
   __asm__ volatile (
        // load C to v16-31 (memory-friendly order)
        "ld1 {v16.4S-v17.4S}, [x0], %[incColC]\n\t"
        "ld1 {v18.4S-v19.4S}, [x0], %[incColC]\n\t"
        "ld1 {v20.4S-v21.4S}, [x0], %[incColC]\n\t"
        "ld1 {v22.4S-v23.4S}, [x0], %[incColC]\n\t"
        "ld1 {v24.4S-v25.4S}, [x0], %[incColC]\n\t"
        "ld1 {v26.4S-v27.4S}, [x0], %[incColC]\n\t"
        "ld1 {v28.4S-v29.4S}, [x0], %[incColC]\n\t"
        "ld1 {v30.4S-v31.4S}, [x0], %[incColC]\n\t"
        ::[C]"r"(C), [incColC]"r"(incColC*4)
    );

    // COMPUTE C <- C + alpha * AB
    for(k=0; k<kc; k++) {
        // compute (8, 1) @ (1, 8)
        __asm__ volatile(
            // load A(8, 1) & B(1, 8)
            "ld1 {v0.4S-v1.4S}, [%[A]]\n\t"
            "ld1 {v2.4S-v3.4S}, [%[B]]\n\t"
            
            // mult alpha to A
            "dup v4.4S, %w[alpha]\n\t"
            "fmul v0.4S, v0.4S, v4.4S\n\t"
            "fmul v1.4S, v1.4S, v4.4S\n\t"

            // MAC: C += alpha_A @ B
            "fmla v16.4S, v2.4S, v0.4S[0]\n\t"
            "fmla v17.4S, v3.4S, v0.4S[0]\n\t"
            "fmla v18.4S, v2.4S, v0.4S[1]\n\t"
            "fmla v19.4S, v3.4S, v0.4S[1]\n\t"
            "fmla v20.4S, v2.4S, v0.4S[2]\n\t"
            "fmla v21.4S, v3.4S, v0.4S[2]\n\t"
            "fmla v22.4S, v2.4S, v0.4S[3]\n\t"
            "fmla v23.4S, v3.4S, v0.4S[3]\n\t"

            "fmla v24.4S, v2.4S, v1.4S[0]\n\t"
            "fmla v25.4S, v3.4S, v1.4S[0]\n\t"
            "fmla v26.4S, v2.4S, v1.4S[1]\n\t"
            "fmla v27.4S, v3.4S, v1.4S[1]\n\t"
            "fmla v28.4S, v2.4S, v1.4S[2]\n\t"
            "fmla v29.4S, v3.4S, v1.4S[2]\n\t"
            "fmla v30.4S, v2.4S, v1.4S[3]\n\t"
            "fmla v31.4S, v3.4S, v1.4S[3]\n\t"

            ::[A]"r"(A), [B]"r"(B), [alpha]"r"(alpha)
            :"x0"
        );
        A += MR;
        B += NR;
    }

    __asm__ volatile (
        // store C
        "st1 {v16.4S-v17.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v18.4S-v19.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v20.4S-v21.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v22.4S-v23.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v24.4S-v25.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v26.4S-v27.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v28.4S-v29.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v30.4S-v31.4S}, [%[C]], %[incColC]\n\t"
        ::[C]"r"(C), [incColC]"r"(incColC*4)
    );

    // for (k=0; k<kc_left; k++) {

    // }
}