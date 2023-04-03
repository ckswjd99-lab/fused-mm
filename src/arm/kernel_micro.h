#include "params.h"

void sgemm_micro_kernel_naive(
    int kc,
    float alpha, const float *A, const float *B,
    float beta, float *C, int incRowC, int incColC
);

void sgemm_micro_kernel_relu_naive(
    int kc,
    float alpha, const float *A, const float *B,
    float beta, float *C, int incRowC, int incColC
);

void sgemm_kernel_8x8_neon_fullyunroll(
    int kc, 
    float alpha, const float *A, const float *B, 
    float beta, float *C, int incRowC, int incColC
);