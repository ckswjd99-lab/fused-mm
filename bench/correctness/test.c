#include <time.h>
#include <stdlib.h>
#include <cblas.h>
#include "../../params.h"

#define BENCH_M           256
#define BENCH_N           (768*4)
#define BENCH_K           768

float A [BENCH_K * BENCH_M];
float B [BENCH_N * BENCH_K];
float C1 [BENCH_M * BENCH_N];
float C2 [BENCH_M * BENCH_N];

float A_buffer [KC * BENCH_M];
float B_buffer [NC * KC];

extern void sgemm_neon_4x16(
    int m, int n, int k, 
    float alpha, 
    const float *A, int incRowA, int incColA, float *A_buffer, 
    const float *B, int incRowB, int incColB, float *B_buffer,
    float beta, 
    float *C, int incRowC, int incColC
);
extern int matrix_comp(float *A, float *B, int rows, int cols);


int main() {

    srand((unsigned int)time(NULL));

    for (int i=0; i<BENCH_N*BENCH_M; i++) {
        C1[i] = 0;
        C2[i] = 0;
    }
    for (int i=0; i<BENCH_K*BENCH_M; i++) {
        A[i] = ((float)rand())/RAND_MAX*2-1;
    }
    for (int i=0; i<BENCH_N*BENCH_K; i++) {
        B[i] = ((float)rand())/RAND_MAX*2-1;

    }

    sgemm_neon_4x16(BENCH_M, BENCH_N, BENCH_K, 1.0, A, 1, BENCH_K, A_buffer, B, 1, BENCH_N, B_buffer, 1.0, C1, 1, BENCH_N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, BENCH_M, BENCH_N, BENCH_K, 1.0, A, BENCH_K, B, BENCH_N, 1.0, C2, BENCH_N);
    
    printf("same? %d\n", matrix_comp(C1, C2, BENCH_N, BENCH_M));

    return 0;
}