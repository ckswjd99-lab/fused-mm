#include "params.h"
#include "gemm/gemm.h"
#include "utils.h"

#define BENCH_M           256
#define BENCH_N           (768*4)
#define BENCH_K           768

float A [BENCH_K * BENCH_M];
float B1 [BENCH_N * BENCH_K];
float C1 [BENCH_M * BENCH_N];

float32_t A_buffer [KC * BENCH_M];
float32_t B_buffer [NC * KC];
