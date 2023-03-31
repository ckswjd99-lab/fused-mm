#include "block_l2.h"

void spack_rowwise_8xk(int nc, int kc, const float *A, int incRowA, int incColA, float *buffer) {
    // naive version
    int i, j;
    
    // unroll 8 columns
    for (i=0; i<(nc&~0x7); i+=8) {
        for (j=0; j<kc; j++) {
            buffer[0] = A[incColA * j + incRowA * (0 + i)];
            buffer[1] = A[incColA * j + incRowA * (1 + i)];
            buffer[2] = A[incColA * j + incRowA * (2 + i)];
            buffer[3] = A[incColA * j + incRowA * (3 + i)];
            buffer[4] = A[incColA * j + incRowA * (4 + i)];
            buffer[5] = A[incColA * j + incRowA * (5 + i)];
            buffer[6] = A[incColA * j + incRowA * (6 + i)];
            buffer[7] = A[incColA * j + incRowA * (7 + i)];
            buffer += 8;
        }
    }

    // remainder columns
    if (nc%8 != 0) {
        for (j=0; j<kc; j++) {
            for (i=0; i<(nc % 8); i++) {
                buffer[i] = A[incColA * j + incRowA * ((nc&~0x7) + i)];
            }
            for (i=(nc % 8); i<8; i++) {
                buffer[i] = 0;
            }
            buffer += 8;
        }
    }
}

void spack_colwise_kx8(int kc, int mc, const float *A, int incRowA, int incColA, float *buffer) {
    // naive version
    int i, j;

    // unroll 8 rows
    for (i=0; i<(mc&~0x7); i+=8) {
        for (j=0; j<kc; j++) {
            buffer[0] = A[incColA * (i + 0) + incRowA * j];
            buffer[1] = A[incColA * (i + 1) + incRowA * j];
            buffer[2] = A[incColA * (i + 2) + incRowA * j];
            buffer[3] = A[incColA * (i + 3) + incRowA * j];
            buffer[4] = A[incColA * (i + 4) + incRowA * j];
            buffer[5] = A[incColA * (i + 5) + incRowA * j];
            buffer[6] = A[incColA * (i + 6) + incRowA * j];
            buffer[7] = A[incColA * (i + 7) + incRowA * j];
            buffer += 8;
        }
    }

    // remainder rows
    if (mc % 8 != 0) {
        for (j=0; j<kc; j++) {
            for (i=0; i<mc%8; i++) {
                buffer[i] = A[incColA * ((mc&~0x7) + i) + incRowA * j];
            }
            for (i=mc%8; i<8; i++) {
                buffer[i] = 0;
            }
            buffer += 8;
        }
    }
}

void sgemm_L2_kernel(int mc, int nc, int kc, float alpha, const float *A_buf, const float *B_buf, float beta, float *C, int incRowC, int incColC) {
    // Assume: A, B buffers are already packed
    // A is packed colwise, B is packed Rowwise
    // Computation only occurs
    int i, j;

    if (mc % 8 == 0 && nc % 8 == 0 && incRowC == 1 && beta == 1) {
        for (i=0; i<mc/8; i++) {
            for (j=0; j<nc/8; j++) {
                sgemm_kernel_8x8_neon_fullyunroll(kc, alpha, A_buf + (8*kc)*i, B_buf + (kc*8)*j, beta, C+8*i*incColC+8*j*incRowC, incRowC, incColC);
            }
        }
    }
    else {
        // UNHAPPY...
    }
}