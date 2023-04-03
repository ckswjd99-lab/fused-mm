#include "kernel_macro.h"

//
//  Packing complete panels from A (i.e. without padding)
//
void pack_MRxk(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
) {
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}


//  Packing complete panels from A (i.e. without padding)
//
void pack_MRxk_unroll(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
) {
    int i, j;

    for (j=0; j<k; ++j) {
        __asm__ volatile (
            "ld1 {v0.4S, v1.4S}, [%1]\n\t"
            "st1 {v0.4S, v1.4S}, [%0]\n\t"
            ::"r"(buffer), "r"(A), "r"(incColA)
        );
        buffer += 8;
        A += incColA;
    }
}

//
//  Packing panels from A with padding if required
//
void pack_rowwise(
    int mc, int kc, 
    const float *A, int incRowA, int incColA, float *buffer
) {
    int mp  = mc / MR;
    int _mr = mc % MR;

    int i, j;

    if (incRowA == 1) {
        for (i=0; i<mp; ++i) {
            pack_MRxk_unroll(kc, A, incRowA, incColA, buffer);
            buffer += kc*MR;
            A      += MR*incRowA;
        }
    }
    else {
        for (i=0; i<mp; ++i) {
            pack_MRxk(kc, A, incRowA, incColA, buffer);
            buffer += kc*MR;
            A      += MR*incRowA;
        }
    }
    if (_mr>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<MR; ++i) {
                buffer[i] = 0.0;
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
void pack_kxNR(
    int k, 
    const float *B, int incRowB, int incColB, float *buffer
) {
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
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
void pack_colwise(
    int kc, int nc, 
    const float *B, int incRowB, int incColB, float *buffer
) {
    int np  = nc / NR;
    int _nr = nc % NR;

    int i, j;

    if (incRowB == 1) {
        for (j=0; j<np; ++j) {
            pack_kxNR(kc, B, incRowB, incColB, buffer);
            buffer += kc*NR;
            B      += NR*incColB;
        }
    }
    else {
        for (j=0; j<np; ++j) {
            pack_kxNR(kc, B, incRowB, incColB, buffer);
            buffer += kc*NR;
            B      += NR*incColB;
        }
    }
    if (_nr>0) {
        for (i=0; i<kc; ++i) {
            for (j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (j=_nr; j<NR; ++j) {
                buffer[j] = 0.0;
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

//
//  Compute Y += alpha*X
//
static void
sgeaxpy(int           m,
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
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
            }
        }
    }
}

//
//  Compute X *= alpha
//
static void
sgescal(int     m,
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
                X[i*incRowX+j*incColX] *= alpha;
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = 0.0;
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
void sgemm_macro_kernel(
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
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
                sgeaxpy(mr, nr, 1.0, _C, 1, MR,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  
//
void sgemm_macro_kernel_neon(
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
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC);
            } else {
                sgemm_micro_kernel_naive(kc, alpha, &A_buffer[i*kc*MR], &B_buffer[j*kc*NR],
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