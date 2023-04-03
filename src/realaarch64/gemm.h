#include "params.h"
#include "block_l2.h"

void sgemm_nn(
    int M, int N, int K, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer,
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta,
    float *C, int incRowC, int incColC
);