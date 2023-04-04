#include "../../../params.h"

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

void sgemm_micro_kernel_neon_8x8(
    int kc, 
    float alpha, const float *A, const float *B, 
    float beta, float *C, int incRowC, int incColC
);

void sgemm_micro_kernel_neon_4x16(
    int kc, 
    float alpha, const float *A, const float *B, 
    float beta, float *C, int incRowC, int incColC
);