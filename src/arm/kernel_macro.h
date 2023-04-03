#include "params.h"
#include "kernel_micro.h"

void pack_MRxk(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_NRxk(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_MRxk_unroll(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_colwise(
    int mc, int kc, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_kxNR(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_kxMR(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_kxNR_unroll(
    int k, 
    const float *A, int incRowA, int incColA, float *buffer
);

void pack_rowwise(
    int kc, int nc, 
    const float *A, int incRowA, int incColA, float *buffer
);

void sgemm_macro_kernel(
    int mc, int nc, int kc,
    float alpha,
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
);

void sgemm_macro_kernel_neon(
    int mc, int nc, int kc,
    float alpha,
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
);

void sgemm_macro_kernel_relu(
    int mc, int nc, int kc,
    float alpha, 
    float *A_buffer, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
);
