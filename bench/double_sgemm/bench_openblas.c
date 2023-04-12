#include "bench.h"
#include "cblas.h"

float A [BENCH_K * BENCH_M];
float B1 [BENCH_N1 * BENCH_K];
float C1 [BENCH_M * BENCH_N1];
float B2 [BENCH_N2 * BENCH_N1];
float C2 [BENCH_N2 * BENCH_M];

int main() {

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, BENCH_M, BENCH_N1, BENCH_K, 1.0, A, BENCH_K, B1, BENCH_N1, 1.0, C1, BENCH_N1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, BENCH_M, BENCH_N2, BENCH_N1, 1.0, C1, BENCH_N1, B2, BENCH_N2, 1.0, C2, BENCH_N2);

    return 0;
}