#include "kernel_macro.h"

//
//  Packing complete panels from A (i.e. without padding)
//
void pack_NRxk_neon_8x8(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
) {
    // Assume:
    // k is multiple of 8
    // incRowA is 1

    int j;

    for (j=0; j<k/8; ++j) {
        __asm__ volatile (
            "ld1 {v0.4S-v1.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v1.4S}, [%[buffer]], #32\n\t"

            "ld1 {v0.4S-v1.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v1.4S}, [%[buffer]], #32\n\t"

            "ld1 {v0.4S-v1.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v1.4S}, [%[buffer]], #32\n\t"

            "ld1 {v0.4S-v1.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v1.4S}, [%[buffer]], #32\n\t"

            "ld1 {v0.4S-v1.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v1.4S}, [%[buffer]], #32\n\t"

            "ld1 {v0.4S-v1.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v1.4S}, [%[buffer]], #32\n\t"

            "ld1 {v0.4S-v1.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v1.4S}, [%[buffer]], #32\n\t"

            "ld1 {v0.4S-v1.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v1.4S}, [%[buffer]], #32\n\t"

            "prfm PLDL1STRM, [%[A], %[incColA]]\n\t"

            ::[A]"r"(A), [buffer]"r"(buffer), [incColA]"r"(incColA*4)
        );

        buffer += NR*8;
        A      += incColA*8;
    }

}

//
//  Packing panels from A with padding if required
//
void pack_rowwise_neon_8x8(
    int mc, int kc, 
    const float *A, int incRowA, int incColA, float *buffer
) {
    // Assume:
    // mc, kc is multiples of 8
    // incRowA is 1

    int mp  = mc / NR;

    int i, j;

    for (i=0; i<mp; ++i) {
        pack_NRxk_neon_8x8(kc, A, incRowA, incColA, buffer);
        buffer += kc*NR;
        A      += NR;
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
void pack_kxMR_neon_8x8(
    int k, 
    const float *B, int incRowB, int incColB, float *buffer
) {
    // Assume:
    // size of B is multiple of 8
    // incRowB is 1
    int i, j;

    for (i=0; i<k; ++i) {
        buffer[0] = B[0*incColB];
        buffer[1] = B[1*incColB];
        buffer[2] = B[2*incColB];
        buffer[3] = B[3*incColB];
        buffer[4] = B[4*incColB];
        buffer[5] = B[5*incColB];
        buffer[6] = B[6*incColB];
        buffer[7] = B[7*incColB];

        buffer += MR;
        B      += incRowB;
    }
}

//
//  Packing panels from B with padding if required
//
void pack_colwise_neon_8x8(
    int kc, int nc, 
    const float *B, int incRowB, int incColB, float *buffer
) {
    // Assume:
    // mc, kc is multiples of 8
    // incRowA is 1
    int np  = nc / MR;

    int i, j;

    for (j=0; j<np; ++j) {
        pack_kxMR_neon_8x8(kc, B, incRowB, incColB, buffer);
        buffer += kc*MR;
        B      += MR*incColB;
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  
//
void sgemm_macro_kernel_neon_8x8(
    int mc, int nc, int kc,
    float alpha,
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
) {
    // Assume:
    // mc, nc, kc is multiple of 8
    // incRowC is 1

    float _C[MR*NR];

    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

            if (mr==MR && nr==NR) {
                sgemm_micro_kernel_neon_8x8(kc, alpha, &A_buffer[i*kc*MR], &B_buffer[j*kc*NR],
                                   beta,
                                   &C[i*MR*incColC+j*NR*incRowC],
                                   incRowC, incColC);
            } else {
                sgemm_micro_kernel_neon_8x8(kc, alpha, &A_buffer[i*kc*MR], &B_buffer[j*kc*NR],
                                   0.0,
                                   _C, 1, MR);
                sgescal(mr, nr, beta,
                        &C[i*MR*incColC+j*NR*incRowC], incRowC, incColC);
                sgeaxpy(mr, nr, 1.0, _C, 1, MR,
                        &C[i*MR*incColC+j*NR*incRowC], incRowC, incColC);
            }
        }
    }
}
