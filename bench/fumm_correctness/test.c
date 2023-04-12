#include <time.h>
#include <stdlib.h>
#include <cblas.h>
#include "../../params.h"

// #define BENCH_M           64
// #define BENCH_N1          64
// #define BENCH_N2          64
// #define BENCH_K           64
#define BENCH_M           128
#define BENCH_N1          (768*4)
#define BENCH_N2          768
#define BENCH_K           768

float A [BENCH_K * BENCH_M];
float B1 [BENCH_N1 * BENCH_K];
float B2 [BENCH_N2 * BENCH_N1];
float C1 [BENCH_M * BENCH_N2];
float C2_1 [BENCH_N1 * BENCH_M];
float C2_2 [BENCH_M * BENCH_N2];

float A_buffer [BENCH_K * BENCH_M];
float B1_buffer [NC * BENCH_K];
float B2_buffer [NC * BENCH_K];
float C_buffer [FUMM_NC1 * FUMM_MC];

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

    // srand((unsigned int)time(NULL));

    for (int i=0; i<BENCH_N2*BENCH_M; i++) {
        C1[i] = 0;
        C2_1[i] = 0;
        C2_2[i] = 0;
    }
    for (int i=0; i<BENCH_K*BENCH_M; i++) {
        A[i] = i % BENCH_K;
        A[i] = ((float)rand())/RAND_MAX-0.5;
    }
    for (int i=0; i<BENCH_N1*BENCH_K; i++) {
        B1[i] = BENCH_N1 - i % BENCH_N1;
        B1[i] = ((float)rand())/RAND_MAX-0.5;
    }
    for (int i=0; i<BENCH_N1*BENCH_N2; i++) {
        B2[i] = i % BENCH_N2;
        B2[i] = ((float)rand())/RAND_MAX-0.5;
    }

    int temp_n1 = BENCH_N1;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, BENCH_M, temp_n1, BENCH_K, 1.0, A, BENCH_K, B1, BENCH_N1, 1.0, C2_1, BENCH_N1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, BENCH_M, BENCH_N2, temp_n1, 1.0, C2_1, BENCH_N1, B2, BENCH_N2, 1.0, C2_2, BENCH_N2);

    // sgemm_neon_4x16(BENCH_M, BENCH_N, BENCH_K, 1.0, A, 1, BENCH_K, A_buffer, B, 1, BENCH_N, B_buffer, 1.0, C1, 1, BENCH_N);
    sfumm_neon_4x16(BENCH_M, BENCH_N1, BENCH_N2, BENCH_K, 1.0, 1.0, A, 1, BENCH_K, A_buffer, B1, 1, BENCH_N1, B1_buffer, B2, 1, BENCH_N2, B2_buffer, 1.0, C1, 1, BENCH_N2, C_buffer);
    
    printf("same? %d\n", matrix_comp(C1, C2_2, BENCH_N2, BENCH_M));

    return 0;
}