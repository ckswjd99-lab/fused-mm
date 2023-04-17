#include "kernel_micro.h"

void sfumm_micro_kernel_neon_4x16(
    int kc, int n2,
    float alpha, const float *A, const float *B1, const float *B2,
    float beta, float *C, int incRowC, int incColC
) {
    assert(kc % 8 == 0);
    assert(n2 % 8 == 0);
    assert(incRowC == 1);

    int i, j;

    // INIT AB1_buffer
    float AB1_buffer[4*16]; // columnwise 4x16
    for (int i=0; i<4*16; i++) {
        AB1_buffer[i] = 0;
    }

    // CALC AB1_buffer
    for (i = 0; i < kc; i++) {
        AB1_buffer[0 + 0*4] += A[0] * B1[0];
        AB1_buffer[1 + 0*4] += A[1] * B1[0];
        AB1_buffer[2 + 0*4] += A[2] * B1[0];
        AB1_buffer[3 + 0*4] += A[3] * B1[0];

        AB1_buffer[0 + 1*4] += A[0] * B1[1];
        AB1_buffer[1 + 1*4] += A[1] * B1[1];
        AB1_buffer[2 + 1*4] += A[2] * B1[1];
        AB1_buffer[3 + 1*4] += A[3] * B1[1];

        AB1_buffer[0 + 2*4] += A[0] * B1[2];
        AB1_buffer[1 + 2*4] += A[1] * B1[2];
        AB1_buffer[2 + 2*4] += A[2] * B1[2];
        AB1_buffer[3 + 2*4] += A[3] * B1[2];

        AB1_buffer[0 + 3*4] += A[0] * B1[3];
        AB1_buffer[1 + 3*4] += A[1] * B1[3];
        AB1_buffer[2 + 3*4] += A[2] * B1[3];
        AB1_buffer[3 + 3*4] += A[3] * B1[3];

        AB1_buffer[0 + 4*4] += A[0] * B1[4];
        AB1_buffer[1 + 4*4] += A[1] * B1[4];
        AB1_buffer[2 + 4*4] += A[2] * B1[4];
        AB1_buffer[3 + 4*4] += A[3] * B1[4];

        AB1_buffer[0 + 5*4] += A[0] * B1[5];
        AB1_buffer[1 + 5*4] += A[1] * B1[5];
        AB1_buffer[2 + 5*4] += A[2] * B1[5];
        AB1_buffer[3 + 5*4] += A[3] * B1[5];

        AB1_buffer[0 + 6*4] += A[0] * B1[6];
        AB1_buffer[1 + 6*4] += A[1] * B1[6];
        AB1_buffer[2 + 6*4] += A[2] * B1[6];
        AB1_buffer[3 + 6*4] += A[3] * B1[6];

        AB1_buffer[0 + 7*4] += A[0] * B1[7];
        AB1_buffer[1 + 7*4] += A[1] * B1[7];
        AB1_buffer[2 + 7*4] += A[2] * B1[7];
        AB1_buffer[3 + 7*4] += A[3] * B1[7];

        AB1_buffer[0 + 8*4] += A[0] * B1[8];
        AB1_buffer[1 + 8*4] += A[1] * B1[8];
        AB1_buffer[2 + 8*4] += A[2] * B1[8];
        AB1_buffer[3 + 8*4] += A[3] * B1[8];

        AB1_buffer[0 + 9*4] += A[0] * B1[9];
        AB1_buffer[1 + 9*4] += A[1] * B1[9];
        AB1_buffer[2 + 9*4] += A[2] * B1[9];
        AB1_buffer[3 + 9*4] += A[3] * B1[9];

        AB1_buffer[0 + 10*4] += A[0] * B1[10];
        AB1_buffer[1 + 10*4] += A[1] * B1[10];
        AB1_buffer[2 + 10*4] += A[2] * B1[10];
        AB1_buffer[3 + 10*4] += A[3] * B1[10];

        AB1_buffer[0 + 11*4] += A[0] * B1[11];
        AB1_buffer[1 + 11*4] += A[1] * B1[11];
        AB1_buffer[2 + 11*4] += A[2] * B1[11];
        AB1_buffer[3 + 11*4] += A[3] * B1[11];

        AB1_buffer[0 + 12*4] += A[0] * B1[12];
        AB1_buffer[1 + 12*4] += A[1] * B1[12];
        AB1_buffer[2 + 12*4] += A[2] * B1[12];
        AB1_buffer[3 + 12*4] += A[3] * B1[12];

        AB1_buffer[0 + 13*4] += A[0] * B1[13];
        AB1_buffer[1 + 13*4] += A[1] * B1[13];
        AB1_buffer[2 + 13*4] += A[2] * B1[13];
        AB1_buffer[3 + 13*4] += A[3] * B1[13];

        AB1_buffer[0 + 14*4] += A[0] * B1[14];
        AB1_buffer[1 + 14*4] += A[1] * B1[14];
        AB1_buffer[2 + 14*4] += A[2] * B1[14];
        AB1_buffer[3 + 14*4] += A[3] * B1[14];

        AB1_buffer[0 + 15*4] += A[0] * B1[15];
        AB1_buffer[1 + 15*4] += A[1] * B1[15];
        AB1_buffer[2 + 15*4] += A[2] * B1[15];
        AB1_buffer[3 + 15*4] += A[3] * B1[15];

        A += 4;
        B1 += 16;
    }

    // ACCUM C
    float C_buffer[4*4];    // rowwise 4x4
    for (i = 0; i < n2; i+=4) {
        for (j = 0; j < 16; j++) {
            C_buffer[j] = 0;
        }

        for (j = 0; j < 16; j++) {
            // B2: rowwise 16x4
            C_buffer[0  + 0*4] += AB1_buffer[0 + j*4] * B2[0 + j*4];
            C_buffer[1  + 0*4] += AB1_buffer[0 + j*4] * B2[1 + j*4];
            C_buffer[2  + 0*4] += AB1_buffer[0 + j*4] * B2[2 + j*4];
            C_buffer[3  + 0*4] += AB1_buffer[0 + j*4] * B2[3 + j*4];

            C_buffer[0  + 1*4] += AB1_buffer[1 + j*4] * B2[0 + j*4];
            C_buffer[1  + 1*4] += AB1_buffer[1 + j*4] * B2[1 + j*4];
            C_buffer[2  + 1*4] += AB1_buffer[1 + j*4] * B2[2 + j*4];
            C_buffer[3  + 1*4] += AB1_buffer[1 + j*4] * B2[3 + j*4];

            C_buffer[0  + 2*4] += AB1_buffer[2 + j*4] * B2[0 + j*4];
            C_buffer[1  + 2*4] += AB1_buffer[2 + j*4] * B2[1 + j*4];
            C_buffer[2  + 2*4] += AB1_buffer[2 + j*4] * B2[2 + j*4];
            C_buffer[3  + 2*4] += AB1_buffer[2 + j*4] * B2[3 + j*4];

            C_buffer[0  + 3*4] += AB1_buffer[3 + j*4] * B2[0 + j*4];
            C_buffer[1  + 3*4] += AB1_buffer[3 + j*4] * B2[1 + j*4];
            C_buffer[2  + 3*4] += AB1_buffer[3 + j*4] * B2[2 + j*4];
            C_buffer[3  + 3*4] += AB1_buffer[3 + j*4] * B2[3 + j*4];
        }

        C[i+0 + 0*incColC] += C_buffer[0  + 0*4];
        C[i+1 + 0*incColC] += C_buffer[1  + 0*4];
        C[i+2 + 0*incColC] += C_buffer[2  + 0*4];
        C[i+3 + 0*incColC] += C_buffer[3  + 0*4];
        
        C[i+0 + 1*incColC] += C_buffer[0  + 1*4];
        C[i+1 + 1*incColC] += C_buffer[1  + 1*4];
        C[i+2 + 1*incColC] += C_buffer[2  + 1*4];
        C[i+3 + 1*incColC] += C_buffer[3  + 1*4];

        C[i+0 + 2*incColC] += C_buffer[0  + 2*4];
        C[i+1 + 2*incColC] += C_buffer[1  + 2*4];
        C[i+2 + 2*incColC] += C_buffer[2  + 2*4];
        C[i+3 + 2*incColC] += C_buffer[3  + 2*4];

        C[i+0 + 3*incColC] += C_buffer[0  + 3*4];
        C[i+1 + 3*incColC] += C_buffer[1  + 3*4];
        C[i+2 + 3*incColC] += C_buffer[2  + 3*4];
        C[i+3 + 3*incColC] += C_buffer[3  + 3*4];

        B2 += 64;
    }

}