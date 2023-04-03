#include "kernel_8x8.h"

void spack_rowwise_8xk(int nc, int kc, const float *B, int incRowB, int incColB, float *buffer);
void spack_colwise_kx8(int kc, int mc, const float *A, int incRowA, int incColA, float *buffer);
void sgemm_block_L2_naive(int mc, int nc, int kc, float alpha, const float *A_buf, const float *B_buf, float beta, float *C, int incRowC, int incColC);
void sgemm_block_L2_fullyunroll(int mc, int nc, int kc, float alpha, const float *A_buf, const float *B_buf, float beta, float *C, int incRowC, int incColC);