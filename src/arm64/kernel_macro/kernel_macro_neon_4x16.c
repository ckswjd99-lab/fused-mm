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
    // k is multiple of 8
    // incRowB is 1
    int i, j;

    for (i=0; i<k/8; ++i) {
        __asm__ volatile (
            "ld1 {v0.4s-v1.4s}, [%x1], %x2\n\t"
            "ld1 {v2.4s-v3.4s}, [%x1], %x2\n\t"
            "ld1 {v4.4s-v5.4s}, [%x1], %x2\n\t"
            "ld1 {v6.4s-v7.4s}, [%x1], %x2\n\t"

            "trn1 v16.4s, v0.4s, v2.4s\n\t"
            "trn1 v17.4s, v4.4s, v6.4s\n\t"
            "trn1 v8.2d, v16.2d, v17.2d\n\t"
            "trn2 v10.2d, v16.2d, v17.2d\n\t"

            "trn2 v18.4s, v0.4s, v2.4s\n\t"
            "trn2 v19.4s, v4.4s, v6.4s\n\t"
            "trn1 v9.2d, v18.2d, v19.2d\n\t"
            "trn2 v11.2d, v18.2d, v19.2d\n\t"

            "st1 {v8.4s-v11.4s}, [%x0], #64\n\t"

            "trn1 v16.4s, v1.4s, v3.4s\n\t"
            "trn1 v17.4s, v5.4s, v7.4s\n\t"
            "trn1 v8.2d, v16.2d, v17.2d\n\t"
            "trn2 v10.2d, v16.2d, v17.2d\n\t"

            "trn2 v18.4s, v1.4s, v3.4s\n\t"
            "trn2 v19.4s, v5.4s, v7.4s\n\t"
            "trn1 v9.2d, v18.2d, v19.2d\n\t"
            "trn2 v11.2d, v18.2d, v19.2d\n\t"

            "st1 {v8.4s-v11.4s}, [%x0], #64\n\t"

            ::"r"(buffer), "r"(B), "r"(incColB*4)
        );
        // buffer[0] = B[0*incColB];
        // buffer[1] = B[1*incColB];
        // buffer[2] = B[2*incColB];
        // buffer[3] = B[3*incColB];

        buffer += 32;
        B      += incRowB*8;
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
//  Pack B2
//
void pack_kfumm_B2_neon_4x16(
    int nc1, int n2,
    const float *B, int incRowB, int incColB, float *buffer
) {
    int i, j;

    for (i=0; i<nc1/16; i++) {
        for (j=0; j<n2/4; j++) {
            buffer[0 + 0*4] = B[0 + 0*incColB];
            buffer[1 + 0*4] = B[1 + 0*incColB];
            buffer[2 + 0*4] = B[2 + 0*incColB];
            buffer[3 + 0*4] = B[3 + 0*incColB];

            buffer[0 + 1*4] = B[0 + 1*incColB];
            buffer[1 + 1*4] = B[1 + 1*incColB];
            buffer[2 + 1*4] = B[2 + 1*incColB];
            buffer[3 + 1*4] = B[3 + 1*incColB];

            buffer[0 + 2*4] = B[0 + 2*incColB];
            buffer[1 + 2*4] = B[1 + 2*incColB];
            buffer[2 + 2*4] = B[2 + 2*incColB];
            buffer[3 + 2*4] = B[3 + 2*incColB];

            buffer[0 + 3*4] = B[0 + 3*incColB];
            buffer[1 + 3*4] = B[1 + 3*incColB];
            buffer[2 + 3*4] = B[2 + 3*incColB];
            buffer[3 + 3*4] = B[3 + 3*incColB];

            buffer[0 + 4*4] = B[0 + 4*incColB];
            buffer[1 + 4*4] = B[1 + 4*incColB];
            buffer[2 + 4*4] = B[2 + 4*incColB];
            buffer[3 + 4*4] = B[3 + 4*incColB];

            buffer[0 + 5*4] = B[0 + 5*incColB];
            buffer[1 + 5*4] = B[1 + 5*incColB];
            buffer[2 + 5*4] = B[2 + 5*incColB];
            buffer[3 + 5*4] = B[3 + 5*incColB];

            buffer[0 + 6*4] = B[0 + 6*incColB];
            buffer[1 + 6*4] = B[1 + 6*incColB];
            buffer[2 + 6*4] = B[2 + 6*incColB];
            buffer[3 + 6*4] = B[3 + 6*incColB];

            buffer[0 + 7*4] = B[0 + 7*incColB];
            buffer[1 + 7*4] = B[1 + 7*incColB];
            buffer[2 + 7*4] = B[2 + 7*incColB];
            buffer[3 + 7*4] = B[3 + 7*incColB];

            buffer[0 + 8*4] = B[0 + 8*incColB];
            buffer[1 + 8*4] = B[1 + 8*incColB];
            buffer[2 + 8*4] = B[2 + 8*incColB];
            buffer[3 + 8*4] = B[3 + 8*incColB];

            buffer[0 + 9*4] = B[0 + 9*incColB];
            buffer[1 + 9*4] = B[1 + 9*incColB];
            buffer[2 + 9*4] = B[2 + 9*incColB];
            buffer[3 + 9*4] = B[3 + 9*incColB];

            buffer[0 + 10*4] = B[0 + 10*incColB];
            buffer[1 + 10*4] = B[1 + 10*incColB];
            buffer[2 + 10*4] = B[2 + 10*incColB];
            buffer[3 + 10*4] = B[3 + 10*incColB];

            buffer[0 + 11*4] = B[0 + 11*incColB];
            buffer[1 + 11*4] = B[1 + 11*incColB];
            buffer[2 + 11*4] = B[2 + 11*incColB];
            buffer[3 + 11*4] = B[3 + 11*incColB];

            buffer[0 + 12*4] = B[0 + 12*incColB];
            buffer[1 + 12*4] = B[1 + 12*incColB];
            buffer[2 + 12*4] = B[2 + 12*incColB];
            buffer[3 + 12*4] = B[3 + 12*incColB];

            buffer[0 + 13*4] = B[0 + 13*incColB];
            buffer[1 + 13*4] = B[1 + 13*incColB];
            buffer[2 + 13*4] = B[2 + 13*incColB];
            buffer[3 + 13*4] = B[3 + 13*incColB];

            buffer[0 + 14*4] = B[0 + 14*incColB];
            buffer[1 + 14*4] = B[1 + 14*incColB];
            buffer[2 + 14*4] = B[2 + 14*incColB];
            buffer[3 + 14*4] = B[3 + 14*incColB];

            buffer[0 + 15*4] = B[0 + 15*incColB];
            buffer[1 + 15*4] = B[1 + 15*incColB];
            buffer[2 + 15*4] = B[2 + 15*incColB];
            buffer[3 + 15*4] = B[3 + 15*incColB];


            buffer += 64;
            B += 4;
        }


        B += 16 * incColB;
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

    for (i=0; i<mp; ++i) {
        mr    = (i!=mp-1 || _mr==0) ? 4 : _mr;

        for (j=0; j<np; ++j) {
            nr    = (j!=np-1 || _nr==0) ? 16 : _nr;

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

void sfumm_macro_kernel_neon_4x16(
    int mc, int nc1, int n2, int k,
    float alpha1, float alpha2,
    float *A_buffer, float *B1_buffer, float *B2_buffer,
    float beta,
    float *C, int incRowC, int incColC, float *C_buffer
) {
    // Assume:
    // mc is multiple of 16
    // nc1 is multiple of 16
    // nc2 is multiple of 16
    // kc is multiple of 8
    // incRowC is 1
    // C_buffer is size of [FUMM_NC1, FUMM_MC]

    assert(mc % 16 == 0);
    assert(nc1 % 16 == 0);
    assert(n2 % 16 == 0);
    assert(k % 8 == 0);
    assert(incRowC == 1);

    int mp = mc / 4;
    int np1 = nc1 / 16;
    int np2 = n2 / 16;

    int mr, nr;
    int i, j, l;

    float *C_buffer_temp = C_buffer;

    for (i=0; i<FUMM_NC1*FUMM_MC/8; i++) {
        __asm__ volatile (
            "dup v0.4s, wzr\n\t"
            "dup v1.4s, wzr\n\t"
            "st1 {v0.4s-v1.4s}, [%x0]\n\t"
            ::"r"(C_buffer_temp)
        );
        C_buffer_temp += 8;
    }

    for (i=0; i<mp; ++i) {
        for (j=0; j<np1; ++j) {
            sgemm_micro_kernel_neon_4x16_pack_colwise(
                k, 
                alpha1, 
                &A_buffer[i*k*4], &B1_buffer[j*k*16],
                1.0,
                &C_buffer[i*4*FUMM_NC1 + j*4*16],
                1, 16
            );
        }
    }
    
    for (j=0; j<FUMM_MC*FUMM_NC1; j++) {
        C_buffer[j];
    }

    for (i=0; i<mp; ++i) {
        for (j=0; j<np2; ++j) {
            sgemm_micro_kernel_neon_4x16(
                nc1,
                alpha2,
                &C_buffer[i*4*nc1], &B2_buffer[j*16*nc1],
                beta,
                &C[i*4*incColC + j*16*incRowC],
                incRowC, incColC
            );
        }
    }


}

void skfumm_macro_kernel_neon_4x16(
    int mc, int nc1, int n2, int k,
    float alpha1, float alpha2,
    float *A_buffer, float *B1_buffer, float *B2_buffer,
    float beta,
    float *C, int incRowC, int incColC, float *C_buffer
) {
    // Assume:
    // mc is multiple of 16
    // nc1 is multiple of 16
    // nc2 is multiple of 16
    // kc is multiple of 8
    // incRowC is 1
    // C_buffer is size of [FUMM_NC1, FUMM_MC]

    assert(mc % 16 == 0);
    assert(nc1 % 16 == 0);
    assert(n2 % 16 == 0);
    assert(k % 8 == 0);
    assert(incRowC == 1);

    int mp = mc / 4;
    int np1 = nc1 / 16;

    int mr, nr;
    int i, j, l;

    for (i=0; i<mp; ++i) {
        for (j=0; j<np1; ++j) {
            sfumm_micro_kernel_neon_4x16(
                k, n2,
                alpha1,
                &A_buffer[i*k*4], &B1_buffer[j*k*16], &B2_buffer[j*n2*16],
                1.0,
                &C[i*incColC*4], 1, incColC
            );
        }
    }
}
