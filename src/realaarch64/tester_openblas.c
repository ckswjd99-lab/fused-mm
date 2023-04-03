#include <arm_neon.h>
#include <time.h>
#include <cblas.h>
#include "utils.h"
#include "block_l2.h"

#define token_num   256
#define d_model     128
#define d_ff        512
#define ffn_num     4

#define M           512
#define N           512
#define K           512

// float32_t ffn_buffer    [d_model * token_num * (ffn_num+1)];
// float32_t W_ff1         [d_ff * d_model];
// float32_t hidden        [d_ff * token_num];
// float32_t W_ff2         [d_model * d_ff];

float32_t A  [M * K];
// float32_t A1 [K * M];
// float32_t A2 [K * M];
float32_t B  [K * N];
// float32_t B1 [N * K];
// float32_t B2 [K * N];
float32_t C1 [M * N];
// float32_t C2 [M * N];
// float32_t C3 [M * N];

float32_t A_buffer [M * K];
float32_t B_buffer [K * N];


int main() {

    // srand((unsigned int)time(NULL));

    // for (int i=0; i<d_ff*token_num; i++) {
    //     hidden[i] = i;
    // }

    // matrix_init_rand(A, 8 * 8);
    // matrix_init_rand(B, 8 * 8);

    // for (int i=0; i<M*K; i++) {
    //     C1[i] = 0;
    //     C2[i] = 0;

        // A[i] = i;
        // B[i] = i;
        // A[i] = ((float)rand())/RAND_MAX*2-1;
        // B[i] = ((float)rand())/RAND_MAX*2-1;
        // B1[(i/K)+(i%K)*N] = B2[i] = ((float)rand())/RAND_MAX*2-1;
    // }
    // for (int i=0; i<8*8; i++) {
    //     C1[i] = 0;
    //     C2[i] = 0;
    //     C3[i] = 0;
    //     // A1[(i%8)*8+i/8] = A2[i] = i;
    //     // B[i] = 8*8-i;
    //     A1[(i%8)*8+i/8] = A2[i] = ((float)rand())/RAND_MAX*2-1;
    //     B[i] = ((float)rand())/RAND_MAX*2-1;
    //     // B1[(i%8)*8+i/8] = B2[i] = i;
    // }

    // for (int i=0; i<M*K; i++) {
    //     A[i] = i; //((float)rand())/RAND_MAX*2-1;
    //     B[i] = (M*K-i); //((float)rand())/RAND_MAX*2-1;
    //     C1[i] = 0;
    //     C2[i] = 0;
    // }

    // sgemm_kernel_8x8_naive(K, 1.0, A1, B, 1.0, C1, 1, M);
    // sgemm_kernel_8x8_neon_fullyunroll(K, 1.0, A1, B, 1.0, C2, 1, M);
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A2, K, B, N, 1.0, C3, M);

    // print_matrix(C1, 8, 8);

    // sgemm_relu_neon(d_ff, token_num, d_model, 1.0, W_ff1, 1, d_ff, ffn_buffer, 1, d_model, 0.0, hidden, 1, d_ff);
    // sgemm_neon(d_model, token_num, d_ff, 1.0, W_ff2, 1, d_model, hidden, 1, d_ff, 1.0, ffn_buffer + d_model * token_num, 1, d_model);

    
    // printf("same? %d\n", matrix_comp(C1, C2, 8, 8));
    // printf("same? %d\n", matrix_comp(C2, C3, 8, 8));

    spack_colwise_kx8(K, M, A, 1, K, A_buffer);
    spack_rowwise_8xk(N, K, B, 1, N, B_buffer);

    // sgemm_L2_kernel_fullyunroll(M, N, K, 1.0, A_buffer, B_buffer, 1.0, C1, 1, M);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 1.0, C1, N);

    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, 2, 1.0, B, M, A, K, 0.0, C2, M);
    // print_matrix(C2, M, N);

    // print_matrix(C1, 16, 16);
    // int is_same = matrix_comp(C1, C2, N, M);
    // printf("same? %d\n", is_same);

    return 0;
}