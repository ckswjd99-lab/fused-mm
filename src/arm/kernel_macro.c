#include "kernel_macro.h"

//
//  Packing complete panels from A (i.e. without padding)
//
void pack_NRxk(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
) {
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<NR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += NR;
        A      += incColA;
    }
}

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

    for (j=0; j<k; ++j) {
        __asm__ volatile (
            "ld1 {v0.4S-v3.4S}, [%[A]], #64\n\t"
            "ld1 {v4.4S-v7.4S}, [%[A]]\n\t"
            "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"
            "st1 {v4.4S-v7.4S}, [%[buffer]]\n\t"
            ::[A]"r"(A), [buffer]"r"(buffer)
        );

        buffer += NR;
        A      += incColA;
    }


    // __asm__ volatile (
    //     ".pack_NRxk_neon_8x8_LoopInit:\n\t"
    //     "mov x0, #0\n"
    //     ".pack_NRxk_neon_8x8_LoopCondition:\n\t"
    //     "cmp x0, %[k]\n\t"
    //     "bgt .pack_NRxk_neon_8x8_LoopEnd\n\t"
    //     ".pack_NRxk_neon_8x8_LoopBody:\n\t"
    //     "ld1 {v0.4S-v3.4S}, [%[A]], #64\n\t"
    //     "ld1 {v4.4S-v7.4S}, [%[A]]\n\t"
    //     "sub %[A], %[A], #64\n\t"
    //     "add %[A], %[A], %[incColA]\n\t"
    //     "st1 {v0.4S-v3.4S}, [%[buffer]], #64\n\t"
    //     "st1 {v4.4S-v7.4S}, [%[buffer]], #64\n\t"
    //     "add x0, x0, #1\n\t"
    //     "b .pack_NRxk_neon_8x8_LoopCondition\n"
    //     ".pack_NRxk_neon_8x8_LoopEnd:\n\t"
    //     ::[k]"r"(k), [A]"r"(A), [incColA]"r"(incColA*4), [buffer]"r"(buffer)
    //     :"x0"
    // );
}

//
//  Packing panels from A with padding if required
//
void pack_rowwise(
    int mc, int kc, 
    const float *A, int incRowA, int incColA, float *buffer
) {
    int mp  = mc / NR;
    int _mr = mc % NR;

    int i, j;

    if (incRowA == 1) {
        for (i=0; i<mp; ++i) {
            pack_NRxk(kc, A, incRowA, incColA, buffer);
            buffer += kc*NR;
            A      += NR*incRowA;
        }
    }
    else {
        for (i=0; i<mp; ++i) {
            pack_NRxk(kc, A, incRowA, incColA, buffer);
            buffer += kc*NR;
            A      += NR*incRowA;
        }
    }
    if (_mr>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<NR; ++i) {
                buffer[i] = 0.0;
            }
            buffer += NR;
            A      += incColA;
        }
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
void pack_kxMR(
    int k, 
    const float *B, int incRowB, int incColB, float *buffer
) {
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<MR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += MR;
        B      += incRowB;
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
//  Packing complete panels from B (i.e. without padding)
//
void pack_kxNR_unroll(
    int k, 
    const float *B, int incRowB, int incColB, float *buffer
) {
    int i, j;

    #define UNROLL_kxNR 8

    int unroll_k = k / UNROLL_kxNR;
    int unroll_left = k % UNROLL_kxNR;

    for (i=0; i<unroll_k; i++) {
        __asm__ volatile (
            "ld1 {v0.4S}, [%0], %2\n\t"
            "ld1 {v2.4S}, [%0], %2\n\t"
            "ld1 {v4.4S}, [%0], %2\n\t"
            "ld1 {v6.4S}, [%0], %2\n\t"
            "ld1 {v1.4S}, [%0], %2\n\t"
            "ld1 {v3.4S}, [%0], %2\n\t"
            "ld1 {v5.4S}, [%0], %2\n\t"
            "ld1 {v7.4S}, [%0], %2\n\t"
            "trn1 v8.4S,  v0.4S, v2.4S\n\t"
            "trn2 v9.4S,  v0.4S, v2.4S\n\t"
            "trn1 v10.4S, v4.4S, v6.4S\n\t"
            "trn2 v11.4S, v4.4S, v6.4S\n\t"
            "mov v0.D[0], v8.D[0]\n\t"
            "mov v0.D[1], v10.D[0]\n\t"
            "mov v2.D[0], v9.D[0]\n\t"
            "mov v2.D[1], v11.D[0]\n\t"
            "mov v4.D[0], v8.D[1]\n\t"
            "mov v4.D[1], v10.D[1]\n\t"
            "mov v6.D[0], v9.D[1]\n\t"
            "mov v6.D[1], v11.D[1]\n\t"
            "trn1 v8.4S,  v1.4S, v3.4S\n\t"
            "trn2 v9.4S,  v1.4S, v3.4S\n\t"
            "trn1 v10.4S, v5.4S, v7.4S\n\t"
            "trn2 v11.4S, v5.4S, v7.4S\n\t"
            "mov v1.D[0], v8.D[0]\n\t"
            "mov v1.D[1], v10.D[0]\n\t"
            "mov v3.D[0], v9.D[0]\n\t"
            "mov v3.D[1], v11.D[0]\n\t"
            "mov v5.D[0], v8.D[1]\n\t"
            "mov v5.D[1], v10.D[1]\n\t"
            "mov v7.D[0], v9.D[1]\n\t"
            "mov v7.D[1], v11.D[1]\n\t"
            "st1 {v0.4S-v3.4S}, [%1], #64\n\t"
            "st1 {v4.4S-v7.4S}, [%1], #64\n\t"
            ::"r"(B), "r"(buffer), "r"(incColB)
        );
    }

    for (i=0; i<unroll_left; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
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
//  Packing panels from B with padding if required
//
void pack_colwise(
    int kc, int nc, 
    const float *B, int incRowB, int incColB, float *buffer
) {
    int np  = nc / MR;
    int _nr = nc % MR;

    int i, j;

    if (incRowB == 1) {
        for (j=0; j<np; ++j) {
            pack_kxMR(kc, B, incRowB, incColB, buffer);
            buffer += kc*MR;
            B      += MR*incColB;
        }
    }
    else {
        for (j=0; j<np; ++j) {
            pack_kxMR(kc, B, incRowB, incColB, buffer);
            buffer += kc*MR;
            B      += MR*incColB;
        }
    }
    if (_nr>0) {
        for (i=0; i<kc; ++i) {
            for (j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (j=_nr; j<MR; ++j) {
                buffer[j] = 0.0;
            }
            buffer += MR;
            B      += incRowB;
        }
    }
}

//
//  Compute Y += alpha*X
//
void sgeaxpy(int           m,
        int           n,
        float        alpha,
        const float  *X,
        int           incRowX,
        int           incColX,
        float        *Y,
        int           incRowY,
        int           incColY)
{
    int i, j;


    if (alpha!=1.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incColY+j*incRowY] += alpha*X[i*incColX+j*incRowX];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incColY+j*incRowY] += X[i*incColX+j*incRowX];
            }
        }
    }
}

//
//  Compute X *= alpha
//
void sgescal(int     m,
        int     n,
        float  alpha,
        float  *X,
        int     incRowX,
        int     incColX)
{
    int i, j;

    if (alpha!=0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incColX+j*incRowX] *= alpha;
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incColX+j*incRowX] = 0.0;
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
void sgemm_macro_kernel_naive(
    int mc, int nc, int kc,
    float alpha,
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
) {
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
                sgemm_micro_kernel_naive(kc, alpha, &A_buffer[i*kc*MR], &B_buffer[j*kc*NR],
                                   beta,
                                   &C[i*MR*incColC+j*NR*incRowC],
                                   incRowC, incColC);
            } else {
                sgemm_micro_kernel_naive(kc, alpha, &A_buffer[i*kc*MR], &B_buffer[j*kc*NR],
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

//
//  Macro Kernel for the multiplication of blocks of A and B and ReLU.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
void sgemm_macro_kernel_relu(
    int mc, int nc, int kc,
    float alpha, 
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
) {
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
                sgemm_micro_kernel_relu_naive(kc, alpha, &A_buffer[i*kc*MR], &B_buffer[j*kc*NR],
                                   beta,
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC);
            } else {
                sgemm_micro_kernel_relu_naive(kc, alpha, &A_buffer[i*kc*MR], &B_buffer[j*kc*NR],
                                   0.0,
                                   _C, 1, MR);
                sgescal(mr, nr, beta,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
                sgeaxpy(mr, nr, 1.0, _C, 1, MR,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}