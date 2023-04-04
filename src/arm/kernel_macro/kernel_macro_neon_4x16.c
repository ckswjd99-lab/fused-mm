#include "kernel_macro.h"

//
//  Packing complete panels from A (i.e. without padding)
//
void pack_NRxk_neon_4x16(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
) {
    // Assume:
    // k is multiple of 8
    // incRowA is 1

    int j;

    for (j=0; j<k/8; ++j) {
        __asm__ volatile (
            "ld1 {v0.4S-v3.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"

            "ld1 {v0.4S-v3.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"

            "ld1 {v0.4S-v3.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"

            "ld1 {v0.4S-v3.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"
            
            "ld1 {v0.4S-v3.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"

            "ld1 {v0.4S-v3.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"

            "ld1 {v0.4S-v3.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"

            "ld1 {v0.4S-v3.4S}, [%[A]], %[incColA]\n\t"
            "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"

            "prfm PLDL1STRM, [%[A], %[incColA]]\n\t"

            ::[A]"r"(A), [buffer]"r"(buffer), [incColA]"r"(incColA*4)
        );

        buffer += 16*8;
        A      += incColA*8;
    }

}

//
//  Packing panels from A with padding if required
//
void pack_rowwise_neon_4x16(
    int kc, int nc, 
    const float *A, int incRowA, int incColA, float *buffer
) {
    // Assume:
    // nc, kc is multiples of 8
    // incRowA is 1

    int mp  = nc / 16;

    int i, j;

    for (i=0; i<mp; ++i) {
        pack_NRxk_neon_4x16(kc, A, incRowA, incColA, buffer);
        buffer += kc*16;
        A      += 16;
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
void pack_kxMR_neon_4x16(
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

        buffer += 4;
        B      += incRowB;
    }
}

//
//  Packing panels from B with padding if required
//
void pack_colwise_neon_4x16(
    int nc, int kc, 
    const float *B, int incRowB, int incColB, float *buffer
) {
    // Assume:
    // mc, kc is multiples of 8
    // incRowA is 1
    int np  = nc / 4;

    int i, j;

    for (j=0; j<np; ++j) {
        pack_kxMR_neon_4x16(kc, B, incRowB, incColB, buffer);
        buffer += kc*4;
        B      += 4*incColB;
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  
//
void sgemm_macro_kernel_neon_4x16(
    int mc, int nc, int kc,
    float alpha,
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
) {
    // Assume:
    // mc is multiple of 4
    // nc is multiple of 16
    // kc is multiple of 8
    // incRowC is 1

    assert(mc % 4 == 0);
    assert(nc % 16 == 0);
    assert(kc % 8 == 0);

    float _C[4*16];

    int mp = (mc+4-1) / 4;
    int np = (nc+16-1) / 16;

    int _mr = mc % 4;
    int _nr = nc % 16;

    int mr, nr;
    int i, j;

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? 16 : _nr;

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? 4 : _mr;

            if (mr==4 && nr==16) {
                sgemm_micro_kernel_neon_4x16(kc, alpha, &A_buffer[i*kc*4], &B_buffer[j*kc*16],
                                   beta,
                                   &C[i*4*incColC+j*16*incRowC],
                                   incRowC, incColC);
            } else {
                sgemm_micro_kernel_neon_4x16(kc, alpha, &A_buffer[i*kc*4], &B_buffer[j*kc*16],
                                   0.0,
                                   _C, 1, 4);
                sgescal(mr, nr, beta,
                        &C[i*4*incColC+j*16*incRowC], incRowC, incColC);
                sgeaxpy(mr, nr, 1.0, _C, 1, 4,
                        &C[i*4*incColC+j*16*incRowC], incRowC, incColC);
            }
        }
    }
}
