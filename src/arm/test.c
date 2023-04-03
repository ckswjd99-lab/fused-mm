#include <arm_neon.h>
#include <time.h>
#include <cblas.h>
#include "params.h"
#include "gemm.h"
#include "utils.h"

#define token_num   256
#define d_model     768
#define d_ff        (768*4)
#define ffn_num     4

#define BENCH_M           256
#define BENCH_N           (768*4)
#define BENCH_K           768

float32_t ffn_buffer    [d_model * token_num * (ffn_num+1)];
float32_t W_ff1         [d_ff * d_model];
float32_t hidden        [d_ff * token_num];
float32_t W_ff2         [d_model * d_ff];

float32_t A [BENCH_K * BENCH_M];
float32_t B1 [BENCH_N * BENCH_K];
float32_t B2 [BENCH_N * BENCH_K];
float32_t C1 [BENCH_M * BENCH_N];
float32_t C2 [BENCH_M * BENCH_N];

float32_t A_buffer [KC * BENCH_M];
float32_t B_buffer [NC * KC];

int main() {

    srand((unsigned int)time(NULL));

    // for (int i=0; i<d_ff*token_num; i++) {
    //     hidden[i] = i;
    // }

    // matrix_init_rand(A, 8 * 8);
    // matrix_init_rand(B, 8 * 8);

    // for (int i=0; i<M*K; i++) {
    //     C1[i] = 0;
    //     C2[i] = 0;

    //     A[i] = ((float)rand())/RAND_MAX*2-1;
    //     B1[i] = B2[i] = ((float)rand())/RAND_MAX*2-1;
    //     // B1[(i/K)+(i%K)*N] = B2[i] = ((float)rand())/RAND_MAX*2-1;
    // }
    for (int i=0; i<BENCH_N*BENCH_M; i++) {
        C1[i] = 0;
        C2[i] = 0;
    }
    for (int i=0; i<BENCH_K*BENCH_M; i++) {
        A[i] = ((float)rand())/RAND_MAX*2-1;
        // A[i] = i;
    }
    for (int i=0; i<BENCH_N*BENCH_K; i++) {
        B1[i] = B2[i] = ((float)rand())/RAND_MAX*2-1;
        // B1[i] = B2[i] = BENCH_N*BENCH_K-i;
        // B1[(i%8)*8+i/8] = B2[i] = i;

    }

    // print_matrix(B1, N, K);
    // print_matrix(B2, K, N);

    // sgemm_kernel_8x8_neon_fullyunroll(K, 1.0, A, B1, 1.0, C1, 1, M);
    sgemm_neon_8x8(BENCH_M, BENCH_N, BENCH_K, 1.0, A, 1, BENCH_K, A_buffer, B1, 1, BENCH_N, B_buffer, 1.0, C1, 1, BENCH_N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, BENCH_M, BENCH_N, BENCH_K, 1.0, A, BENCH_K, B2, BENCH_N, 1.0, C2, BENCH_N);

    // print_matrix(C1, 8, 8);

    // sgemm_relu_neon(d_ff, token_num, d_model, 1.0, W_ff1, 1, d_ff, ffn_buffer, 1, d_model, 0.0, hidden, 1, d_ff);
    // sgemm_neon(d_model, token_num, d_ff, 1.0, W_ff2, 1, d_model, hidden, 1, d_ff, 1.0, ffn_buffer + d_model * token_num, 1, d_model);

    
    printf("same? %d\n", matrix_comp(C1, C2, BENCH_N, BENCH_M));

    return 0;
}