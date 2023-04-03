#include "gemm.h"

//
//  Compute Y += alpha*X
//
static void sgeaxpy(
    int m, int n, 
    float alpha, 
    const float  *X, int incRowX, int incColX, 
    float *Y, int incRowY, int incColY
) {
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
static void sgescal(
    int     m, int     n,
    float  alpha,
    float  *X, int     incRowX, int     incColX
) {
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
//  Compute C <- beta*C + alpha*A*B
//
void sgemm_naive(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
) {
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    float _beta;

    if (alpha==0.0 || k==0) {
        sgescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            _beta = (l==0) ? beta : 1.0;

            pack_rowwise(
                    nc, kc, 
                    &B[l*KC*incColB+j*NC*incRowB], incRowB, incColB,
                    B_buffer);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_colwise(
                    kc, mc,
                    &A[i*MC*incColA+l*KC*incRowA], incRowA, incColA, A_buffer
                );

                sgemm_macro_kernel(
                    mc, nc, kc, 
                    alpha, A_buffer, B_buffer, 
                    _beta, &C[i*MC*incColC+j*NC*incRowC], incRowC, incColC
                );
            }
        }
    }
}

//
//  Compute C <- beta*C + alpha*A*B
//
void sgemm_neon(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
) {
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    float _beta;

    if (alpha==0.0 || k==0) {
        sgescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            _beta = (l==0) ? beta : 1.0;

            pack_rowwise(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   B_buffer);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_colwise(
                    mc, kc,
                    &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA, A_buffer
                );

                sgemm_macro_kernel(
                    mc, nc, kc, 
                    alpha, A_buffer, B_buffer, 
                    _beta, &C[i*MC*incRowC+j*NC*incColC], incRowC, incColC
                );
            }
        }
    }
}

//
//  Compute C <- beta*C + alpha*A*B
//
void sgemm_relu_neon(
    int m, int n, int k,
    float alpha,
    const float *A, int incRowA, int incColA, float *A_buffer,
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
) {
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    float _beta;

    if (alpha==0.0 || k==0) {
        sgescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            _beta = (l==0) ? beta : 1.0;

            pack_rowwise(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   B_buffer);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_colwise(mc, kc,
                       &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                       A_buffer);

                sgemm_macro_kernel_relu(
                    mc, nc, kc, alpha, A_buffer, B_buffer,
                    _beta, &C[i*MC*incRowC+j*NC*incColC], incRowC, incColC
                );
            }
        }
    }
}