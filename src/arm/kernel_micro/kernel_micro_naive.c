#include "kernel_micro.h"

void sgemm_micro_kernel_naive(
    int kc,
    float alpha, const float *A, const float *B,
    float beta, float *C, int incRowC, int incColC
) {
    float AB[MR * NR];

    int i, j, k;

    // init AB
    for (i = 0; i < MR; i++) {
        for (j = 0; j < NR; j++) {
            AB[i + MR * j] = 0;
        }
    }

    // calc AB
    for (k = 0; k < kc; k++) {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                AB[i * NR + j] += A[i + k * MR] * B[j + k * NR];
            }
        }
    }

    // update C <- beta * C
    if (beta == 0.0) {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                C[i * incRowC + j * incColC] = 0;
            }
        }
    }
    else if (beta != 1.0) {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                C[i * incRowC + j * incColC] *= beta;
            }
        }
    }

    // update C <- alpha * AB
    if (alpha == 1.0) {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                C[i * incRowC + j * incColC] += AB[i + MR * j];
            }
        }
    }
    else {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                C[i * incRowC + j * incColC] += alpha * AB[i + MR * j];
            }
        }
    }
}

void sgemm_micro_kernel_relu_naive(
    int kc,
    float alpha, const float *A, const float *B,
    float beta, float *C, int incRowC, int incColC
) {
    float AB[MR * NR];

    int i, j, k;

    // init AB
    for (i = 0; i < MR; i++) {
        for (j = 0; j < NR; j++) {
            AB[i + MR * j] = 0;
        }
    }

    // calc AB
    for (k = 0; k < kc; k++) {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                AB[i + j * MR] += A[i + k * MR] * B[j + k * NR];
            }
        }
    }

    // update C <- beta * C
    if (beta == 0.0) {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                C[i * incRowC + j * incColC] = 0;
            }
        }
    }
    else if (beta != 1.0) {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                C[i * incRowC + j * incColC] *= beta;
            }
        }
    }

    // update C <- alpha * AB & ReLU
    if (alpha == 1.0) {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                C[i * incRowC + j * incColC] += AB[i + MR * j];
                if (C[i * incRowC + j * incColC] < 0) C[i * incRowC + j * incColC] = 0;
            }
        }
    }
    else {
        for (i = 0; i < MR; i++) {
            for (j = 0; j < NR; j++) {
                C[i * incRowC + j * incColC] += alpha * AB[i + MR * j];
                if (C[i * incRowC + j * incColC] < 0) C[i * incRowC + j * incColC] = 0;
            }
        }
    }
}
