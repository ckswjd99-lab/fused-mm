#include "gemm.h"

void sgemm_nn(
    int M, int N, int K, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer,
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
) {
    // A_buffer: float[L2_BLOCK_K * L2_BLOCK_M]
    // B_buffer: float[L2_BLOCK_N * L2_BLOCK_K]

    int i, j, k;

    int l2_iter_k = (K + L2_BLOCK_K - 1) / L2_BLOCK_K;
    int l2_iter_n = (N + L2_BLOCK_N - 1) / L2_BLOCK_N;
    int l2_iter_m = (M + L2_BLOCK_MM - 1)/ L2_BLOCK_MM;

    int l2_remainder_k = K % L2_BLOCK_K;
    int l2_remainder_n = N % L2_BLOCK_N;
    int l2_remainder_m = M % L2_BLOCK_MM;

    int l2_k, l2_n, l2_m;
    float _beta;

    // L2 iter for N
    for (j=0; j<l2_iter_n; j++) {
        l2_n = (j != l2_iter_n-1 || l2_remainder_n == 0) ? L2_BLOCK_N : l2_remainder_n;

        // L2 iter for K
        for (k=0; k<l2_iter_k; k++) {
            l2_k = (k != l2_iter_k-1 || l2_remainder_k == 0) ? L2_BLOCK_K : l2_remainder_k;
            _beta = (k == 0) ? beta : 1.0;
        
            // L2 iter for M
            spack_rowwise_8xk(L2_BLOCK_N, L2_BLOCK_K, B, incRowB, incColB, B_buffer);
            for (i=0; i<l2_iter_m; i++) {
                l2_m = (i != l2_iter_m || l2_remainder_m == 0) ? L2_BLOCK_MM : l2_remainder_m;

                spack_colwise_kx8(
                    l2_k, l2_m, 
                    A + i*l2_m*incColA + k*l2_k*incRowA, 
                    incRowA, incColA,
                    A_buffer
                );

                if (incRowC == 1 && beta == 1) {
                    sgemm_block_L2_fullyunroll(
                        l2_m, l2_n, l2_k,
                        alpha, A_buffer, B_buffer,
                        _beta, C+j*L2_BLOCK_N*incRowA+i*L2_BLOCK_MM*incColA, incRowC, incColC
                    );
                }
                else {
                    sgemm_block_L2_naive(
                        l2_m, l2_n, l2_k,
                        alpha, A_buffer, B_buffer,
                        _beta, C, incRowC, incColC
                    );
                }

            }
        }
    }

}