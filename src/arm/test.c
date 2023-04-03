#include <arm_neon.h>
#include <time.h>
#include <cblas.h>
#include "params.h"
#include "gemm.h"
#include "utils.h"

#define token_num   256
#define d_model     786
#define d_ff        (786*4)
#define ffn_num     4

#define M           945
#define N           141
#define K           153

float32_t ffn_buffer    [d_model * token_num * (ffn_num+1)];
float32_t W_ff1         [d_ff * d_model];
float32_t hidden        [d_ff * token_num];
float32_t W_ff2         [d_model * d_ff];

float32_t A [K * M];
float32_t B1 [N * K];
float32_t B2 [N * K];
float32_t C1 [M * N];
float32_t C2 [M * N];

float32_t A_buffer [KC * MC];
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
    for (int i=0; i<N*M; i++) {
        C1[i] = 0;
        C2[i] = 0;
    }
    for (int i=0; i<K*M; i++) {
        A[i] = ((float)rand())/RAND_MAX*2-1;
        // A[i] = i;
    }
    for (int i=0; i<N*K; i++) {
        B1[i] = B2[i] = ((float)rand())/RAND_MAX*2-1;
        // B1[i] = B2[i] = N*K-i;
        // B1[(i%8)*8+i/8] = B2[i] = i;

    }

    // print_matrix(B1, N, K);
    // print_matrix(B2, K, N);

    // sgemm_kernel_8x8_neon_fullyunroll(K, 1.0, A, B1, 1.0, C1, 1, M);
    sgemm_naive(M, N, K, 1.0, A, 1, K, A_buffer, B1, 1, N, B_buffer, 1.0, C1, 1, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B2, N, 1.0, C2, N);

    // print_matrix(C1, 8, 8);

    // sgemm_relu_neon(d_ff, token_num, d_model, 1.0, W_ff1, 1, d_ff, ffn_buffer, 1, d_model, 0.0, hidden, 1, d_ff);
    // sgemm_neon(d_model, token_num, d_ff, 1.0, W_ff2, 1, d_model, hidden, 1, d_ff, 1.0, ffn_buffer + d_model * token_num, 1, d_model);

    
    printf("same? %d\n", matrix_comp(C1, C2, N, M));

    return 0;
}