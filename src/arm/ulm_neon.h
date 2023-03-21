#ifndef LEVEL3_DGEMM_NN_H
#define LEVEL3_DGEMM_NN_H 1

void sgemm_neon(
    int m, 
    int n, 
    int k, 
    float alpha, 
    const float *A, 
    int incRowA, 
    int incColA, 
    const float *B, 
    int incRowB, 
    int incColB,
    float beta, 
    float *C, 
    int incRowC, 
    int incColC
);

void sgemm_relu_neon(
    int m, 
    int n, 
    int k, 
    float alpha, 
    const float *A, 
    int incRowA, 
    int incColA, 
    const float *B, 
    int incRowB, 
    int incColB,
    float beta, 
    float *C, 
    int incRowC, 
    int incColC
);

#endif // LEVEL3_DGEMM_NN_H
