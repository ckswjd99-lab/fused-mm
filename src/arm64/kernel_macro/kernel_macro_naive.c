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
