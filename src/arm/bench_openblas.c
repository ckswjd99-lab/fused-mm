#include "bench.h"
#include "cblas.h"

int main() {

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, BENCH_M, BENCH_N, BENCH_K, 1.0, A, BENCH_K, B1, BENCH_N, 1.0, C1, BENCH_N);

    return 0;
}