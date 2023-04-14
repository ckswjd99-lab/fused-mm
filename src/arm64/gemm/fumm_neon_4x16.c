#include "gemm.h"

void sfumm_neon_4x16(
    int m, int n1, int n2, int k,
    float alpha1, float alpha2,
    const float *A, int incRowA, int incColA, float *A_buffer,
    const float *B1, int incRowB1, int incColB1, float *B1_buffer,
    const float *B2, int incRowB2, int incColB2, float *B2_buffer,
    float beta,
    float *C, int incRowC, int incColC, float *C_buffer
) {
    // Assume:
    // m is multiple of 4
    // n1 is multiple of 16
    // n2 is multiple of 16
    // k is multiple of 8
    // incRow is 1
    // C_buffer is size of [FUMM_NC1, FUMM_MC]

    int mb = (m+FUMM_MC-1) / FUMM_MC;
    int nb = (n1+FUMM_NC1-1) / FUMM_NC1;

    int _mc = m % FUMM_MC;
    int _nc = n1 % FUMM_NC1;

    int mc, nc1;
    int i, j;

    float _beta;

    for (j=0; j<nb; ++j) {
        nc1 = (j!=nb-1 || _nc==0) ? FUMM_NC1 : _nc;

        pack_rowwise_neon_4x16(
            k, nc1, 
            &B1[j*FUMM_NC1*incRowB1], incRowB1, incColB1,
            B1_buffer
        );

        pack_rowwise_neon_4x16(
            nc1, n2, 
            &B2[j*FUMM_NC1*incColB2], incRowB2, incColB2,
            B2_buffer
        );

        for (i=0; i<mb; ++i) {
            mc = (i!=mb-1 || _mc==0) ? FUMM_MC : _mc;

            if (j == 0) {
                pack_colwise_neon_4x16(
                    mc, k, 
                    &A[i*FUMM_MC*incColA], incRowA, incColA, &A_buffer[i*FUMM_MC*k]
                );

            }


            sfumm_macro_kernel_neon_4x16(
                mc, nc1, n2, k,
                alpha1, alpha2,
                &A_buffer[i*FUMM_MC*k], B1_buffer, B2_buffer,
                beta,
                &C[i*FUMM_MC*incColC], incRowC, incColC,
                C_buffer
            );
        }
    }
}