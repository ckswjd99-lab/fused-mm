#include <time.h>
#include <stdlib.h>
#include <cblas.h>

#define MC  4
#define NC1 16
#define NC2 4

#define K   1024
#define N2  1024

float A         [MC  * K  ];
float A_packed  [MC  * K  ];
float B1        [K   * NC1];
float B1_packed [K   * NC1];
float AB1       [MC  * NC1];
float B2        [NC1 * N2 ];
float B2_packed [NC1 * N2 ];
float C_ob      [MC  * N2 ];
float C_fumm    [MC  * N2 ];

extern void sfumm_micro_kernel_neon_4x16(
    int kc, int n2,
    float alpha, const float *A, const float *B1, const float *B2,
    float beta, float *C, int incRowC, int incColC
);

extern int matrix_comp(float *A, float *B, int rows, int cols);

int main() {
    srand((unsigned int)time(NULL));

    // INIT A, B1, B2, AB1, C
    for (int i=0; i<MC*K; i++) {
        A[i] = i%K;
        A[i] = ((float)rand())/RAND_MAX*2-1;
    }
    for (int i=0; i<K*NC1; i++) {
        B1[i] = i%NC1;
        B1[i] = ((float)rand())/RAND_MAX*2-1;
    }
    for (int i=0; i<MC*NC1; i++) {
        AB1[i] = 0;
    }
    for (int i=0; i<NC1*N2; i++) {
        B2[i] = i%N2;
        B2[i] = ((float)rand())/RAND_MAX*2-1;
    }
    for (int i=0; i<MC*N2; i++) {
        C_ob[i] = 0;
    }

    // PACK A, B1, B2
    for (int i=0; i<K; i++) {
        A_packed[0 + i*MC] = A[i + 0*K];
        A_packed[1 + i*MC] = A[i + 1*K];
        A_packed[2 + i*MC] = A[i + 2*K];
        A_packed[3 + i*MC] = A[i + 3*K];
    }
    float *B2_pack_temp = B2_packed;
    for (int i=0; i<N2; i+=NC2) {
        for (int j=0; j<NC1; j++) {
            B2_pack_temp[0] = B2[i+0 + j*N2];
            B2_pack_temp[1] = B2[i+1 + j*N2];
            B2_pack_temp[2] = B2[i+2 + j*N2];
            B2_pack_temp[3] = B2[i+3 + j*N2];
            B2_pack_temp += 4;
        }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MC, NC1, K, 1.0, A, K, B1, NC1, 1.0, AB1, NC1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MC, N2, NC1, 1.0, AB1, NC1, B2, N2, 1.0, C_ob, N2);

    sfumm_micro_kernel_neon_4x16(K, N2, 1.0, A_packed, B1, B2_packed, 1.0, C_fumm, 1, N2);

    int correct = matrix_comp(C_ob, C_fumm, MC, N2);
    printf("same? %d\n", correct);
    

    return 0;
}