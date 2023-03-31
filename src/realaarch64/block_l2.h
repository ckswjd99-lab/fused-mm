#include "kernel_8x8.h"

void spack_rowwise_8xk(int mc, int kc, const float *A, int incRowA, int incColA, float *buffer);
void spack_colwise_kx8(int mc, int kc, const float *A, int incRowA, int incColA, float *buffer);
void sgemm_L2_kernel(int mc, int nc, int kc, float alpha, const float *A_buf, const float *B_buf, float beta, float *C, int incRowC, int incColC);