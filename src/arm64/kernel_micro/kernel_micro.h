#include "assert.h"
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

void sgemm_micro_kernel_neon_4x16_pack_colwise(
    int kc, 
    float alpha, const float *A, const float *B, 
    float beta, float *C, int incRowC, int incColC
);

void sfumm_micro_kernel_neon_4x16(
    int k, int n2,
    float alpha, const float *A, const float *B1, const float *B2,
    float beta, float *C, int incRowC, int incColC
);