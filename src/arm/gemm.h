#include "params.h"
#include "kernel_macro.h"

void sgemm_naive(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
);

void sgemm_neon(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
);

void sgemm_relu_neon(
    int m, int n, int k,
    float alpha,
    const float *A, int incRowA, int incColA, float *A_buffer,
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
);

