#define MR  8
#define NR  8

void sgemm_kernel_8x8_naive(
    int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC
);

void sgemm_kernel_8x8_fullyunroll(
    int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC
);
// when incRowC == 1