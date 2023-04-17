#include "../../../params.h"
#include "../kernel_macro/kernel_macro.h"
#include "assert.h"

void sgemm_naive(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
);

void sgemm_neon_8x8(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
);

void sgemm_neon_4x16(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
);

void sfumm_neon_4x16_relu(
    int m, int n1, int n2, int k,
    float alpha1, float alpha2,
    const float *A, int incRowA, int incColA, float *A_buffer,
    const float *B1, int incRowB1, int incColB1, float *B1_buffer,
    float beta,
    float *C, int incRowC, int incColC
);

void skfumm_neon_4x16(
    int m, int n1, int n2, int k,
    float alpha1, float alpha2,
    const float *A, int incRowA, int incColA, float *A_buffer,
    const float *B1, int incRowB1, int incColB1, float *B1_buffer,
    const float *B2, int incRowB2, int incColB2, float *B2_buffer,
    float beta,
    float *C, int incRowC, int incColC, float *C_buffer
);