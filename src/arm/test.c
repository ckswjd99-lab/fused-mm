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

int main() {

    sgemm_relu_neon(d_ff, token_num, d_model, 1.0, W_ff1, 1, d_ff, ffn_buffer, 1, d_model, 0.0, hidden, 1, d_ff);
    sgemm_neon(d_model, token_num, d_ff, 1.0, W_ff2, 1, d_model, hidden, 1, d_ff, 0.0, ffn_buffer + d_model * token_num, 1, d_model);

    return 0;
}