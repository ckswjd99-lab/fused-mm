#include "bench.h"

float A_buffer [KC * MC];
float B_buffer [NC * KC];

extern void sgemm_neon_4x16(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
);

int main() {

    sgemm_neon_4x16(BENCH_M, BENCH_N, BENCH_K, 1.0, A, 1, BENCH_K, A_buffer, B1, 1, BENCH_N, B_buffer, 1.0, C1, 1, BENCH_N);

    return 0;
}