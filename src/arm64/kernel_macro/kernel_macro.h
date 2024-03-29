#include "../../../params.h"
#include "../kernel_micro/kernel_micro.h"
#include <assert.h>

void pack_NRxk(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_NRxk_neon_8x8(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_NRxk_neon_4x16(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_rowwise(
    int kc, int nc, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_rowwise_neon_8x8(
    int kc, int nc, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_rowwise_neon_4x16(
    int kc, int nc, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_kxMR(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_kxMR_neon_8x8(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_kxMR_neon_4x16(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_colwise(
    int mc, int kc, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_colwise_neon_8x8(
    int mc, int kc, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_colwise_neon_4x16(
    int mc, int kc, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_kfumm_B2_neon_4x16(
    int nc1, int n2,
    const float *B, int incRowB, int incColB, float *buffer
);

void sgeaxpy(
    int m, int n,
    float alpha,
    const float *X, int incRowX, int incColX,
    float *Y, int incRowY, int incColY
);

void sgescal(
    int m, int n,
    float alpha,
    float *X, int incRowX, int incColX
);

void sgemm_macro_kernel_naive(
    int mc, int nc, int kc,
    float alpha,
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
);

void sgemm_macro_kernel_neon_8x8(
    int mc, int nc, int kc,
    float alpha,
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
);

void sgemm_macro_kernel_neon_4x16(
    int mc, int nc, int kc,
    float alpha,
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
);

void sfumm_macro_kernel_neon_4x16(
    int mc, int nc1, int n2, int k,
    float alpha1, float alpha2,
    float *A_buffer, float *B1_buffer, float *B2_buffer,
    float beta,
    float *C, int incRowC, int incColC, float *C_buffer
);

void skfumm_macro_kernel_neon_4x16(
    int mc, int nc1, int n2, int k,
    float alpha1, float alpha2,
    float *A_buffer, float *B1_buffer, float *B2_buffer,
    float beta,
    float *C, int incRowC, int incColC, float *C_buffer
);
