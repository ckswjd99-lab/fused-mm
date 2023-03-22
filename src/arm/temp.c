#include <arm_neon.h>
#include <time.h>
#include "ulm_neon.h"

#define token_num   256
#define d_model     128
#define d_ff        512
#define ffn_num     4

float32_t ffn_buffer    [d_model * token_num * (ffn_num+1)];
float32_t W_ff1         [d_ff * d_model];
float32_t hidden        [d_ff * token_num];
float32_t W_ff2         [d_model * d_ff];

float32_t A [8 * 8];
float32_t B [8 * 8];
float32_t C [8 * 8];

int main() {

    // for (int i=0; i<64; i++) {
    //     A[i] = 1.0;
    //     B[i] = 2.0;
    // }

    // sgemm_neon(8, 8, 8, 1.0, A, 1, 8, B, 1, 8, 0.0, C, 1, 8);

    float32_t datas[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __asm__ volatile (
        "ld1 {v0.4S, v1.4S, v2.4S, v3.4S}, [%0]\n\t"
        "trn1 v4.4S, v0.4S, v1.4S\n\t"
        "trn2 v5.4S, v0.4S, v1.4S\n\t"
        "trn1 v6.4S, v2.4S, v3.4S\n\t"
        "trn2 v7.4S, v2.4S, v3.4S\n\t"
        "mov v0.D[0], v4.D[0]\n\t"
        "mov v0.D[1], v6.D[0]\n\t"
        "mov v1.D[0], v5.D[0]\n\t"
        "mov v1.D[1], v7.D[0]\n\t"
        "mov v2.D[0], v4.D[1]\n\t"
        "mov v2.D[1], v6.D[1]\n\t"
        "mov v3.D[0], v5.D[1]\n\t"
        "mov v3.D[1], v7.D[1]\n\t"

        ::"r"(datas)
    );

    // sgemm_relu_neon(d_ff, token_num, d_model, 1.0, W_ff1, 1, d_ff, ffn_buffer, 1, d_model, 0.0, hidden, 1, d_ff);
    // sgemm_neon(d_model, token_num, d_ff, 1.0, W_ff2, 1, d_model, hidden, 1, d_ff, 1.0, ffn_buffer + d_model * token_num, 1, d_model);

    return 0;
}