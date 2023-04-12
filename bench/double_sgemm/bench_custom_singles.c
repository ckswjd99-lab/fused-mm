#include "bench.h"
#include "../../params.h"

// #define BENCH_M           64
// #define BENCH_N1          64
// #define BENCH_N2          64
// #define BENCH_K           64

float A [BENCH_K * BENCH_M];
float B1 [BENCH_N1 * BENCH_K];
float C1 [BENCH_M * BENCH_N2];
float B2 [BENCH_N2 * BENCH_N1];
float C2 [BENCH_M * BENCH_N2];

float A_buffer [BENCH_K * FUMM_MC];
float B1_buffer [NC * BENCH_K];
float C1_buffer [BENCH_N1 * FUMM_MC];
float B2_buffer [NC * BENCH_K];

extern void sgemm_neon_4x16(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
);
extern void sfumm_neon_4x16(
    int m, int n1, int n2, int k,
    float alpha1, float alpha2,
    const float *A, int incRowA, int incColA, float *A_buffer,
    const float *B1, int incRowB1, int incColB1, float *B1_buffer,
    const float *B2, int incRowB2, int incColB2, float *B2_buffer,
    float beta,
    float *C, int incRowC, int incColC, float *C_buffer
);
extern int matrix_comp(float *A, float *B, int rows, int cols);


int main() {
    // sgemm_neon_4x16(BENCH_M, BENCH_N, BENCH_K, 1.0, A, 1, BENCH_K, A_buffer, B, 1, BENCH_N, B_buffer, 1.0, C1, 1, BENCH_N);
    sgemm_neon_4x16(BENCH_M, BENCH_N1, BENCH_K, 1.0, A, 1, BENCH_K, A_buffer, B1, 1, BENCH_N1, B1_buffer, 1.0, C1, 1, BENCH_N1);
    sgemm_neon_4x16(BENCH_M, BENCH_N2, BENCH_N1, 1.0, C1, 1, BENCH_N1, C1_buffer, B2, 1, BENCH_N2, B2_buffer, 1.0, C2, 1, BENCH_N2);

    // sfumm_neon_4x16(BENCH_M, BENCH_N1, BENCH_N2, BENCH_K, 1.0, 1.0, A, 1, BENCH_K, A_buffer, B1, 1, BENCH_N1, B1_buffer, B2, 1, BENCH_N2, B2_buffer, 1.0, A, 1, BENCH_N2, C_buffer);

    return 0;
}