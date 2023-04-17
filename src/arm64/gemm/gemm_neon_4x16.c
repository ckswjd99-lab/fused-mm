#include "gemm.h"

void sgemm_neon_4x16(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
) {
    // Assume:
    // m is multiple of 4
    // n is multiple of 16
    // k is multiple of 8
    // incRow is 1

    // ASSERTION
    assert(m % 4 == 0);
    assert(n % 16 == 0);
    assert(k % 8 == 0);
    assert(incRowA == 1);
    assert(incRowB == 1);
    assert(incRowC == 1);


    int mmb = (m+MMC-1) / MMC;
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mmc = m % MMC;
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


            pack_rowwise_neon_4x16(
                    kc, nc, 
                    &B[l*KC*incColB+j*NC*incRowB], incRowB, incColB,
                    B_buffer);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;
                
                // if (j==0) {
                    pack_colwise_neon_4x16(
                        mc, kc, 
                        &A[i*MC*incColA+l*KC*incRowA], incRowA, incColA, A_buffer/*&A_buffer[i*MC*KC]*/
                    );
                // }

                sgemm_macro_kernel_neon_4x16(
                    mc, nc, kc, 
                    alpha, A_buffer, B_buffer, 
                    _beta, &C[i*MC*incColC+j*NC*incRowC], incRowC, incColC
                );
            }
        }
    }
}
