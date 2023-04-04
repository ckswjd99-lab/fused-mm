#include "bench.h"

int main() {

    sgemm_neon_4x16(BENCH_M, BENCH_N, BENCH_K, 1.0, A, 1, BENCH_K, A_buffer, B1, 1, BENCH_N, B_buffer, 1.0, C1, 1, BENCH_N);

    return 0;
}