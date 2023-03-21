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

    // sgemm_relu_neon(d_ff, token_num, d_model, 1.0, W_ff1, 1, d_ff, ffn_buffer, 1, d_model, 0.0, hidden, 1, d_ff);
    sgemm_neon(d_model, token_num, d_ff, 1.0, W_ff2, 1, d_model, hidden, 1, d_ff, 0.0, ffn_buffer + d_model * token_num, 1, d_model);

    return 0;
}