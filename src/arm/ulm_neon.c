#include <stdio.h>

#define MC  128
#define KC  128
#define NC  840

#define MR  (4*2)   // never change!
#define NR  (4*2)   // never change!

//
//  Local buffers for storing panels from A, B and C
//
static float _A[MC*KC] __attribute__ ((aligned (16)));
static float _B[KC*NC] __attribute__ ((aligned (16)));
static float _C[MR*NR] __attribute__ ((aligned (16)));

//
//  Packing complete panels from A (i.e. without padding)
//
static void
pack_MRxk(int k, const float *A, int incRowA, int incColA,
          float *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}


//  Packing complete panels from A (i.e. without padding)
//
static void
pack_MRxk_unroll(int k, const float *A, int incRowA, int incColA,
          float *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        __asm__ volatile (
            "ld1 {v0.4S, v1.4S}, [%1]\n\t"
            "st1 {v0.4S, v1.4S}, [%0]\n\t"
            ::"r"(buffer), "r"(A), "r"(incColA)
        );
        buffer += 8;
        A += incColA;
    }
}

//
//  Packing panels from A with padding if required
//
static void
pack_A(int mc, int kc, const float *A, int incRowA, int incColA,
       float *buffer)
{
    int mp  = mc / MR;
    int _mr = mc % MR;

    int i, j;

    if (incRowA == 1) {
        for (i=0; i<mp; ++i) {
            pack_MRxk_unroll(kc, A, incRowA, incColA, buffer);
            buffer += kc*MR;
            A      += MR*incRowA;
        }
    }
    else {
        for (i=0; i<mp; ++i) {
            pack_MRxk(kc, A, incRowA, incColA, buffer);
            buffer += kc*MR;
            A      += MR*incRowA;
        }
    }
    if (_mr>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<MR; ++i) {
                buffer[i] = 0.0;
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR(int k, const float *B, int incRowB, int incColB,
          float *buffer)
{
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR_unroll(int k, const float *B, int incRowB, int incColB,
          float *buffer)
{
    int i, j;

    #define UNROLL_kxNR 8

    int unroll_k = k / UNROLL_kxNR;
    int unroll_left = k % UNROLL_kxNR;

    for (i=0; i<unroll_k; i++) {
        __asm__ volatile (
            "ld1 {v0.4S}, [%0], %2\n\t"
            "ld1 {v2.4S}, [%0], %2\n\t"
            "ld1 {v4.4S}, [%0], %2\n\t"
            "ld1 {v6.4S}, [%0], %2\n\t"
            "ld1 {v1.4S}, [%0], %2\n\t"
            "ld1 {v3.4S}, [%0], %2\n\t"
            "ld1 {v5.4S}, [%0], %2\n\t"
            "ld1 {v7.4S}, [%0], %2\n\t"
            "trn1 v8.4S,  v0.4S, v2.4S\n\t"
            "trn2 v9.4S,  v0.4S, v2.4S\n\t"
            "trn1 v10.4S, v4.4S, v6.4S\n\t"
            "trn2 v11.4S, v4.4S, v6.4S\n\t"
            "mov v0.D[0], v8.D[0]\n\t"
            "mov v0.D[1], v10.D[0]\n\t"
            "mov v2.D[0], v9.D[0]\n\t"
            "mov v2.D[1], v11.D[0]\n\t"
            "mov v4.D[0], v8.D[1]\n\t"
            "mov v4.D[1], v10.D[1]\n\t"
            "mov v6.D[0], v9.D[1]\n\t"
            "mov v6.D[1], v11.D[1]\n\t"
            "trn1 v8.4S,  v1.4S, v3.4S\n\t"
            "trn2 v9.4S,  v1.4S, v3.4S\n\t"
            "trn1 v10.4S, v5.4S, v7.4S\n\t"
            "trn2 v11.4S, v5.4S, v7.4S\n\t"
            "mov v1.D[0], v8.D[0]\n\t"
            "mov v1.D[1], v10.D[0]\n\t"
            "mov v3.D[0], v9.D[0]\n\t"
            "mov v3.D[1], v11.D[0]\n\t"
            "mov v5.D[0], v8.D[1]\n\t"
            "mov v5.D[1], v10.D[1]\n\t"
            "mov v7.D[0], v9.D[1]\n\t"
            "mov v7.D[1], v11.D[1]\n\t"
            "st1 {v0.4S-v3.4S}, [%1], #64\n\t"
            "st1 {v4.4S-v7.4S}, [%1], #64\n\t"
            ::"r"(B), "r"(buffer), "r"(incColB)
        );
    }

    for (i=0; i<unroll_left; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

//
//  Packing panels from B with padding if required
//
static void
pack_B(int kc, int nc, const float *B, int incRowB, int incColB,
       float *buffer)
{
    int np  = nc / NR;
    int _nr = nc % NR;

    int i, j;

    if (incRowB == 1) {
        for (j=0; j<np; ++j) {
            pack_kxNR_unroll(kc, B, incRowB, incColB, buffer);
            buffer += kc*NR;
            B      += NR*incColB;
        }
    }
    else {
        for (j=0; j<np; ++j) {
            pack_kxNR(kc, B, incRowB, incColB, buffer);
            buffer += kc*NR;
            B      += NR*incColB;
        }
    }
    if (_nr>0) {
        for (i=0; i<kc; ++i) {
            for (j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (j=_nr; j<NR; ++j) {
                buffer[j] = 0.0;
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

//
//  Micro kernel for multiplying panels from A and B.
//
static void
sgemm_micro_kernel(int kc,
                   float alpha, const float *A, const float *B,
                   float beta,
                   float *C, int incRowC, int incColC)
{
    float AB[MR*NR];

    int i, j, l;

    //
    //  Compute AB = A*B
    //

    // float32x4_t tmp0, tmp1, tmp2, tmp3;             // v0~3
    // float32x4_t tmp4, tmp5, tmp6, tmp7;             // v4~7

    // float32x4_t ab0_00_11_22_33 = vdupq_n_f32(0);   // v16
    // float32x4_t ab0_01_10_23_32 = vdupq_n_f32(0);   // v17
    // float32x4_t ab0_03_12_21_30 = vdupq_n_f32(0);   // v18
    // float32x4_t ab0_02_13_20_31 = vdupq_n_f32(0);   // v19
    
    // float32x4_t ab1_00_11_22_33 = vdupq_n_f32(0);   // v20
    // float32x4_t ab1_01_10_23_32 = vdupq_n_f32(0);   // v21
    // float32x4_t ab1_03_12_21_30 = vdupq_n_f32(0);   // v22
    // float32x4_t ab1_02_13_20_31 = vdupq_n_f32(0);   // v23

    // float32x4_t ab2_00_11_22_33 = vdupq_n_f32(0);   // v24
    // float32x4_t ab2_01_10_23_32 = vdupq_n_f32(0);   // v25
    // float32x4_t ab2_03_12_21_30 = vdupq_n_f32(0);   // v26
    // float32x4_t ab2_02_13_20_31 = vdupq_n_f32(0);   // v27

    // float32x4_t ab3_00_11_22_33 = vdupq_n_f32(0);   // v28
    // float32x4_t ab3_01_10_23_32 = vdupq_n_f32(0);   // v29
    // float32x4_t ab3_03_12_21_30 = vdupq_n_f32(0);   // v30
    // float32x4_t ab3_02_13_20_31 = vdupq_n_f32(0);   // v31

    __asm__ volatile (
        "dup v16.4S, wzr\n\t"
        "dup v17.4S, wzr\n\t"
        "dup v18.4S, wzr\n\t"
        "dup v19.4S, wzr\n\t"
        "dup v20.4S, wzr\n\t"
        "dup v21.4S, wzr\n\t"
        "dup v22.4S, wzr\n\t"
        "dup v23.4S, wzr\n\t"
        "dup v24.4S, wzr\n\t"
        "dup v25.4S, wzr\n\t"
        "dup v26.4S, wzr\n\t"
        "dup v27.4S, wzr\n\t"
        "dup v28.4S, wzr\n\t"
        "dup v29.4S, wzr\n\t"
        "dup v30.4S, wzr\n\t"
        "dup v31.4S, wzr\n\t"
    );


    for (l=0; l<kc; ++l) {
        int temp;
        __asm__ volatile (
            "ld1 {v0.4S, v1.4S}, [%0]\n\t"
            "ld1 {v2.4S, v3.4S}, [%1]\n\t"
            "fmla v16.4S, v0.4S, v2.4S\n\t"
            "fmla v20.4S, v1.4S, v2.4S\n\t"
            "fmla v24.4S, v0.4S, v3.4S\n\t"
            "fmla v28.4S, v1.4S, v3.4S\n\t"
            "rev64 v4.4S, v2.4S\n\t"
            "rev64 v5.4S, v3.4S\n\t"
            "fmla v17.4S, v0.4S, v4.4S\n\t"
            "fmla v21.4S, v1.4S, v4.4S\n\t"
            "fmla v25.4S, v0.4S, v5.4S\n\t"
            "fmla v29.4S, v1.4S, v5.4S\n\t"
            "ext v6.16B, v4.16B, v4.16B, #8\n\t"
            "ext v7.16B, v5.16B, v5.16B, #8\n\t"
            "fmla v18.4S, v0.4S, v6.4S\n\t"
            "fmla v22.4S, v1.4S, v6.4S\n\t"
            "fmla v26.4S, v0.4S, v7.4S\n\t"
            "fmla v30.4S, v1.4S, v7.4S\n\t"
            "ext v6.16B, v2.16B, v2.16B, #8\n\t"
            "ext v7.16B, v3.16B, v3.16B, #8\n\t"
            "fmla v19.4S, v0.4S, v6.4S\n\t"
            "fmla v23.4S, v1.4S, v6.4S\n\t"
            "fmla v27.4S, v0.4S, v7.4S\n\t"
            "fmla v31.4S, v1.4S, v7.4S\n\t"
            "add %0, %0, #32\n\t"
            "add %1, %1, #32\n\t"
            :"+r"(A), "+r"(B)
        );
    }

    //
    // Save ab
    //

    __asm__ volatile ("st1 {v16.S}[0], [%0]\n\t"::"r"(&AB[0+0*8]));
    __asm__ volatile ("st1 {v17.S}[1], [%0]\n\t"::"r"(&AB[1+0*8]));
    __asm__ volatile ("st1 {v19.S}[2], [%0]\n\t"::"r"(&AB[2+0*8]));
    __asm__ volatile ("st1 {v18.S}[3], [%0]\n\t"::"r"(&AB[3+0*8]));
    __asm__ volatile ("st1 {v20.S}[0], [%0]\n\t"::"r"(&AB[4+0*8]));
    __asm__ volatile ("st1 {v21.S}[1], [%0]\n\t"::"r"(&AB[5+0*8]));
    __asm__ volatile ("st1 {v23.S}[2], [%0]\n\t"::"r"(&AB[6+0*8]));
    __asm__ volatile ("st1 {v22.S}[3], [%0]\n\t"::"r"(&AB[7+0*8]));

    __asm__ volatile ("st1 {v17.S}[0], [%0]\n\t"::"r"(&AB[0+1*8]));
    __asm__ volatile ("st1 {v16.S}[1], [%0]\n\t"::"r"(&AB[1+1*8]));
    __asm__ volatile ("st1 {v18.S}[2], [%0]\n\t"::"r"(&AB[2+1*8]));
    __asm__ volatile ("st1 {v19.S}[3], [%0]\n\t"::"r"(&AB[3+1*8]));
    __asm__ volatile ("st1 {v21.S}[0], [%0]\n\t"::"r"(&AB[4+1*8]));
    __asm__ volatile ("st1 {v20.S}[1], [%0]\n\t"::"r"(&AB[5+1*8]));
    __asm__ volatile ("st1 {v22.S}[2], [%0]\n\t"::"r"(&AB[6+1*8]));
    __asm__ volatile ("st1 {v23.S}[3], [%0]\n\t"::"r"(&AB[7+1*8]));

    __asm__ volatile ("st1 {v19.S}[0], [%0]\n\t"::"r"(&AB[0+2*8]));
    __asm__ volatile ("st1 {v18.S}[1], [%0]\n\t"::"r"(&AB[1+2*8]));
    __asm__ volatile ("st1 {v16.S}[2], [%0]\n\t"::"r"(&AB[2+2*8]));
    __asm__ volatile ("st1 {v17.S}[3], [%0]\n\t"::"r"(&AB[3+2*8]));
    __asm__ volatile ("st1 {v23.S}[0], [%0]\n\t"::"r"(&AB[4+2*8]));
    __asm__ volatile ("st1 {v22.S}[1], [%0]\n\t"::"r"(&AB[5+2*8]));
    __asm__ volatile ("st1 {v20.S}[2], [%0]\n\t"::"r"(&AB[6+2*8]));
    __asm__ volatile ("st1 {v21.S}[3], [%0]\n\t"::"r"(&AB[7+2*8]));

    __asm__ volatile ("st1 {v18.S}[0], [%0]\n\t"::"r"(&AB[0+3*8]));
    __asm__ volatile ("st1 {v19.S}[1], [%0]\n\t"::"r"(&AB[1+3*8]));
    __asm__ volatile ("st1 {v17.S}[2], [%0]\n\t"::"r"(&AB[2+3*8]));
    __asm__ volatile ("st1 {v16.S}[3], [%0]\n\t"::"r"(&AB[3+3*8]));
    __asm__ volatile ("st1 {v22.S}[0], [%0]\n\t"::"r"(&AB[4+3*8]));
    __asm__ volatile ("st1 {v23.S}[1], [%0]\n\t"::"r"(&AB[5+3*8]));
    __asm__ volatile ("st1 {v21.S}[2], [%0]\n\t"::"r"(&AB[6+3*8]));
    __asm__ volatile ("st1 {v20.S}[3], [%0]\n\t"::"r"(&AB[7+3*8]));

    __asm__ volatile ("st1 {v24.S}[0], [%0]\n\t"::"r"(&AB[0+4*8]));
    __asm__ volatile ("st1 {v25.S}[1], [%0]\n\t"::"r"(&AB[1+4*8]));
    __asm__ volatile ("st1 {v27.S}[2], [%0]\n\t"::"r"(&AB[2+4*8]));
    __asm__ volatile ("st1 {v26.S}[3], [%0]\n\t"::"r"(&AB[3+4*8]));
    __asm__ volatile ("st1 {v28.S}[0], [%0]\n\t"::"r"(&AB[4+4*8]));
    __asm__ volatile ("st1 {v29.S}[1], [%0]\n\t"::"r"(&AB[5+4*8]));
    __asm__ volatile ("st1 {v31.S}[2], [%0]\n\t"::"r"(&AB[6+4*8]));
    __asm__ volatile ("st1 {v30.S}[3], [%0]\n\t"::"r"(&AB[7+4*8]));

    __asm__ volatile ("st1 {v25.S}[0], [%0]\n\t"::"r"(&AB[0+5*8]));
    __asm__ volatile ("st1 {v24.S}[1], [%0]\n\t"::"r"(&AB[1+5*8]));
    __asm__ volatile ("st1 {v26.S}[2], [%0]\n\t"::"r"(&AB[2+5*8]));
    __asm__ volatile ("st1 {v27.S}[3], [%0]\n\t"::"r"(&AB[3+5*8]));
    __asm__ volatile ("st1 {v29.S}[0], [%0]\n\t"::"r"(&AB[4+5*8]));
    __asm__ volatile ("st1 {v28.S}[1], [%0]\n\t"::"r"(&AB[5+5*8]));
    __asm__ volatile ("st1 {v30.S}[2], [%0]\n\t"::"r"(&AB[6+5*8]));
    __asm__ volatile ("st1 {v31.S}[3], [%0]\n\t"::"r"(&AB[7+5*8]));
    
    __asm__ volatile ("st1 {v27.S}[0], [%0]\n\t"::"r"(&AB[0+6*8]));
    __asm__ volatile ("st1 {v26.S}[1], [%0]\n\t"::"r"(&AB[1+6*8]));
    __asm__ volatile ("st1 {v24.S}[2], [%0]\n\t"::"r"(&AB[2+6*8]));
    __asm__ volatile ("st1 {v25.S}[3], [%0]\n\t"::"r"(&AB[3+6*8]));
    __asm__ volatile ("st1 {v31.S}[0], [%0]\n\t"::"r"(&AB[4+6*8]));
    __asm__ volatile ("st1 {v30.S}[1], [%0]\n\t"::"r"(&AB[5+6*8]));
    __asm__ volatile ("st1 {v28.S}[2], [%0]\n\t"::"r"(&AB[6+6*8]));
    __asm__ volatile ("st1 {v29.S}[3], [%0]\n\t"::"r"(&AB[7+6*8]));

    __asm__ volatile ("st1 {v26.S}[0], [%0]\n\t"::"r"(&AB[0+7*8]));
    __asm__ volatile ("st1 {v27.S}[1], [%0]\n\t"::"r"(&AB[1+7*8]));
    __asm__ volatile ("st1 {v25.S}[2], [%0]\n\t"::"r"(&AB[2+7*8]));
    __asm__ volatile ("st1 {v24.S}[3], [%0]\n\t"::"r"(&AB[3+7*8]));
    __asm__ volatile ("st1 {v30.S}[0], [%0]\n\t"::"r"(&AB[4+7*8]));
    __asm__ volatile ("st1 {v31.S}[1], [%0]\n\t"::"r"(&AB[5+7*8]));
    __asm__ volatile ("st1 {v29.S}[2], [%0]\n\t"::"r"(&AB[6+7*8]));
    __asm__ volatile ("st1 {v28.S}[3], [%0]\n\t"::"r"(&AB[7+7*8]));

//
//  Update C <- beta*C
//
    if (beta==0.0) {
        for (j=0; j<NR; j++) {
            __asm__ volatile (
                "dup v0.4S, wzr\n\t"
                "dup v1.4S, wzr\n\t"
                "st1 {v0.4S, v1.4S}, [%0]\n\t"
                ::"r"(&C[j*incColC])
            );
        }
    } else if (beta!=1.0) {
        for (j=0; j<NR; j++) {
            __asm__ volatile (
                "ld1 {v0.4S, v1.4S}, [%0]\n\t"
                "dup v2.4S, %w1\n\t"
                "fmul v0.4S, v0.4S, v2.4S\n\t"
                "fmul v1.4S, v1.4S, v2.4S\n\t"
                "st1 {v0.4S, v1.4S}, [%0]\n\t"
                ::"r"(&C[j*incColC]), "r"(beta)
            );
        }
    }

//
//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
//                                  the above layer dgemm_nn)
//
    if (alpha==1.0) {
        for (j=0; j<NR; j++) {
            __asm__ volatile (
                "ld1 {v0.4S, v1.4S}, [%0]\n\t"
                "ld1 {v2.4S, v3.4S}, [%1]\n\t"
                "add v0.4S, v0.4S, v2.4S\n\t"
                "add v1.4S, v1.4S, v3.4S\n\t"
                "st1 {v0.4S, v1.4S}, [%0]\n\t"
                ::"r"(&C[j*incColC]), "r"(&AB[j*MR])
            );
        }
    } else {
        for (j=0; j<NR; j++) {
            __asm__ volatile (
                "ld1 {v0.4S, v1.4S}, [%0]\n\t"
                "ld1 {v2.4S, v3.4S}, [%1]\n\t"
                "dup v4.4S, %w2\n\t"
                "fmul v2.4S, v2.4S, v4.4S\n\t"
                "fmul v3.4S, v3.4S, v4.4S\n\t"
                "add v0.4S, v0.4S, v2.4S\n\t"
                "add v1.4S, v1.4S, v3.4S\n\t"
                "st1 {v0.4S, v1.4S}, [%0]\n\t"
                ::"r"(&C[j*incColC]), "r"(&AB[j*MR]), "r"(alpha)
            );
        }
    }
}

void
sgemm_micro_kernel_naive(int kc,
                   float alpha, const float *A, const float *B,
                   float beta,
                   float *C, int incRowC, int incColC)
{
    float AB[MR*NR];

    int i, j, k;

    // init AB
    for (i=0; i<MR; i++) {
        for (j=0; j<NR; j++) {
            AB[i+MR*j] = 0;
        }
    }

    // calc AB
    for (k=0; k<kc; k++) {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                AB[i+j*MR] += A[i+k*MR] * B[j+k*NR];
            }
        }
    }

    // update C <- beta * C
    if (beta == 0.0) {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                C[i*incRowC+j*incColC] = 0;
            }
        }
    }
    else if (beta != 1.0) {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

    // update C <- alpha * AB
    if (alpha == 1.0) {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                C[i*incRowC+j*incColC] += AB[i+MR*j];
            }
        }
    }
    else {
        for (i=0; i<MR; i++) {
            for (j=0; j<NR; j++) {
                C[i*incRowC+j*incColC] += alpha * AB[i+MR*j];
            }
        }
    }
}

void sgemm_kernel_8x8_neon_fullyunroll(
    int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC
) {
    /**
     * ASSUME
     * beta == 1
     * incRowC == 1
    */

    #define KC_UNROLL 8

    int kc_unroll = kc / KC_UNROLL;
    int kc_left = kc % KC_UNROLL;

    int i, j, k;

   // INIT AB = C
   __asm__ volatile (
        // load C to v16-31 (memory-friendly order)
        "ld1 {v16.4S-v17.4S}, [x0], %[incColC]\n\t"
        "ld1 {v18.4S-v19.4S}, [x0], %[incColC]\n\t"
        "ld1 {v20.4S-v21.4S}, [x0], %[incColC]\n\t"
        "ld1 {v22.4S-v23.4S}, [x0], %[incColC]\n\t"
        "ld1 {v24.4S-v25.4S}, [x0], %[incColC]\n\t"
        "ld1 {v26.4S-v27.4S}, [x0], %[incColC]\n\t"
        "ld1 {v28.4S-v29.4S}, [x0], %[incColC]\n\t"
        "ld1 {v30.4S-v31.4S}, [x0], %[incColC]\n\t"
        ::[C]"r"(C), [incColC]"r"(incColC*4)
    );

    // COMPUTE C <- C + alpha * AB
    for(k=0; k<kc; k++) {
        // compute (8, 1) @ (1, 8)
        __asm__ volatile(
            // load A(8, 1) & B(1, 8)
            "ld1 {v0.4S-v1.4S}, [%[A]]\n\t"
            "ld1 {v2.4S-v3.4S}, [%[B]]\n\t"
            
            // mult alpha to A
            "dup v4.4S, %w[alpha]\n\t"
            "fmul v0.4S, v0.4S, v4.4S\n\t"
            "fmul v1.4S, v1.4S, v4.4S\n\t"

            // MAC: C += alpha_A @ B
            "fmla v16.4S, v0.4S, v2.4S[0]\n\t"
            "fmla v17.4S, v1.4S, v2.4S[0]\n\t"
            "fmla v18.4S, v0.4S, v2.4S[1]\n\t"
            "fmla v19.4S, v1.4S, v2.4S[1]\n\t"
            "fmla v20.4S, v0.4S, v2.4S[2]\n\t"
            "fmla v21.4S, v1.4S, v2.4S[2]\n\t"
            "fmla v22.4S, v0.4S, v2.4S[3]\n\t"
            "fmla v23.4S, v1.4S, v2.4S[3]\n\t"

            "fmla v24.4S, v0.4S, v3.4S[0]\n\t"
            "fmla v25.4S, v1.4S, v3.4S[0]\n\t"
            "fmla v26.4S, v0.4S, v3.4S[1]\n\t"
            "fmla v27.4S, v1.4S, v3.4S[1]\n\t"
            "fmla v28.4S, v0.4S, v3.4S[2]\n\t"
            "fmla v29.4S, v1.4S, v3.4S[2]\n\t"
            "fmla v30.4S, v0.4S, v3.4S[3]\n\t"
            "fmla v31.4S, v1.4S, v3.4S[3]\n\t"

            ::[A]"r"(A), [B]"r"(B), [alpha]"r"(alpha)
            :"x0"
        );
        A += MR;
        B += NR;
    }

    __asm__ volatile (
        // store C
        "st1 {v16.4S-v17.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v18.4S-v19.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v20.4S-v21.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v22.4S-v23.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v24.4S-v25.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v26.4S-v27.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v28.4S-v29.4S}, [%[C]], %[incColC]\n\t"
        "st1 {v30.4S-v31.4S}, [%[C]], %[incColC]\n\t"
        ::[C]"r"(C), [incColC]"r"(incColC*4)
    );

    // for (k=0; k<kc_left; k++) {

    // }
}

//
//  Micro kernel for multiplying panels from A and B then ReLU.
//
static void
sgemm_relu_micro_kernel(int kc,
                   float alpha, const float *A, const float *B,
                   float beta,
                   float *C, int incRowC, int incColC)
{
    float AB[MR*NR] __attribute__ ((aligned (16)));

    int i, j, l;

    //
    //  Compute AB = A*B
    //

    /*

    float32x4_t tmp0, tmp1, tmp2, tmp3;
    float32x4_t tmp4, tmp5, tmp6, tmp7;

    float32x4_t ab0_00_11_22_33 = vdupq_n_f32(0);
    float32x4_t ab0_01_10_23_32 = vdupq_n_f32(0);
    float32x4_t ab0_03_12_21_30 = vdupq_n_f32(0);
    float32x4_t ab0_02_13_20_31 = vdupq_n_f32(0);
    
    float32x4_t ab1_00_11_22_33 = vdupq_n_f32(0);
    float32x4_t ab1_01_10_23_32 = vdupq_n_f32(0);
    float32x4_t ab1_03_12_21_30 = vdupq_n_f32(0);
    float32x4_t ab1_02_13_20_31 = vdupq_n_f32(0);

    float32x4_t ab2_00_11_22_33 = vdupq_n_f32(0);
    float32x4_t ab2_01_10_23_32 = vdupq_n_f32(0);
    float32x4_t ab2_03_12_21_30 = vdupq_n_f32(0);
    float32x4_t ab2_02_13_20_31 = vdupq_n_f32(0);

    float32x4_t ab3_00_11_22_33 = vdupq_n_f32(0);
    float32x4_t ab3_01_10_23_32 = vdupq_n_f32(0);
    float32x4_t ab3_03_12_21_30 = vdupq_n_f32(0);
    float32x4_t ab3_02_13_20_31 = vdupq_n_f32(0);


    for (l=0; l<kc; ++l) {
        
        tmp0 = vld1q_f32(A);
        tmp1 = vld1q_f32(A+4);

        tmp2 = vld1q_f32(B);
        tmp3 = vld1q_f32(B+4);

        ab0_00_11_22_33 = vfmaq_f32(tmp0, tmp2, ab0_00_11_22_33);
        ab1_00_11_22_33 = vfmaq_f32(tmp1, tmp2, ab1_00_11_22_33);
        ab2_00_11_22_33 = vfmaq_f32(tmp0, tmp3, ab2_00_11_22_33);
        ab3_00_11_22_33 = vfmaq_f32(tmp1, tmp3, ab3_00_11_22_33);

        tmp4 = vrev64q_f32(tmp2);
        tmp5 = vrev64q_f32(tmp3);

        ab0_01_10_23_32 = vfmaq_f32(tmp0, tmp4, ab0_01_10_23_32);
        ab1_01_10_23_32 = vfmaq_f32(tmp1, tmp4, ab1_01_10_23_32);
        ab2_01_10_23_32 = vfmaq_f32(tmp0, tmp5, ab2_01_10_23_32);
        ab3_01_10_23_32 = vfmaq_f32(tmp1, tmp5, ab3_01_10_23_32);

        tmp6 = vextq_f32(tmp4, tmp4, 2);
        tmp7 = vextq_f32(tmp5, tmp5, 2);

        ab0_03_12_21_30 = vfmaq_f32(tmp0, tmp6, ab0_03_12_21_30);
        ab1_03_12_21_30 = vfmaq_f32(tmp1, tmp6, ab1_03_12_21_30);
        ab2_03_12_21_30 = vfmaq_f32(tmp0, tmp7, ab2_03_12_21_30);
        ab3_03_12_21_30 = vfmaq_f32(tmp1, tmp7, ab3_03_12_21_30);

        tmp6 = vextq_f32(tmp2, tmp2, 2);
        tmp7 = vextq_f32(tmp3, tmp3, 2);

        ab0_02_13_20_31 = vfmaq_f32(tmp0, tmp6, ab0_02_13_20_31);
        ab1_02_13_20_31 = vfmaq_f32(tmp1, tmp6, ab1_02_13_20_31);
        ab2_02_13_20_31 = vfmaq_f32(tmp0, tmp7, ab2_02_13_20_31);
        ab3_02_13_20_31 = vfmaq_f32(tmp1, tmp7, ab3_02_13_20_31);

        A += 8;
        B += 8;
    }

    //
    // Save ab0
    //

    vst1q_lane_f32(&AB[0+0*8], ab0_00_11_22_33, 0);
    vst1q_lane_f32(&AB[1+1*8], ab0_00_11_22_33, 1);
    vst1q_lane_f32(&AB[2+2*8], ab0_00_11_22_33, 2);
    vst1q_lane_f32(&AB[3+3*8], ab0_00_11_22_33, 3);

    vst1q_lane_f32(&AB[0+1*8], ab0_01_10_23_32, 0);
    vst1q_lane_f32(&AB[1+0*8], ab0_01_10_23_32, 1);
    vst1q_lane_f32(&AB[2+3*8], ab0_01_10_23_32, 2);
    vst1q_lane_f32(&AB[3+2*8], ab0_01_10_23_32, 3);

    vst1q_lane_f32(&AB[0+3*8], ab0_03_12_21_30, 0);
    vst1q_lane_f32(&AB[1+2*8], ab0_03_12_21_30, 1);
    vst1q_lane_f32(&AB[2+1*8], ab0_03_12_21_30, 2);
    vst1q_lane_f32(&AB[3+0*8], ab0_03_12_21_30, 3);

    vst1q_lane_f32(&AB[0+2*8], ab0_02_13_20_31, 0);
    vst1q_lane_f32(&AB[1+3*8], ab0_02_13_20_31, 1);
    vst1q_lane_f32(&AB[2+0*8], ab0_02_13_20_31, 2);
    vst1q_lane_f32(&AB[3+1*8], ab0_02_13_20_31, 3);
    
    //
    // Save ab1
    //

    vst1q_lane_f32(&AB[4+0*8], ab1_00_11_22_33, 0);
    vst1q_lane_f32(&AB[5+1*8], ab1_00_11_22_33, 1);
    vst1q_lane_f32(&AB[6+2*8], ab1_00_11_22_33, 2);
    vst1q_lane_f32(&AB[7+3*8], ab1_00_11_22_33, 3);

    vst1q_lane_f32(&AB[4+1*8], ab1_01_10_23_32, 0);
    vst1q_lane_f32(&AB[5+0*8], ab1_01_10_23_32, 1);
    vst1q_lane_f32(&AB[6+3*8], ab1_01_10_23_32, 2);
    vst1q_lane_f32(&AB[7+2*8], ab1_01_10_23_32, 3);

    vst1q_lane_f32(&AB[4+3*8], ab1_03_12_21_30, 0);
    vst1q_lane_f32(&AB[5+2*8], ab1_03_12_21_30, 1);
    vst1q_lane_f32(&AB[6+1*8], ab1_03_12_21_30, 2);
    vst1q_lane_f32(&AB[7+0*8], ab1_03_12_21_30, 3);

    vst1q_lane_f32(&AB[4+2*8], ab1_02_13_20_31, 0);
    vst1q_lane_f32(&AB[5+3*8], ab1_02_13_20_31, 1);
    vst1q_lane_f32(&AB[6+0*8], ab1_02_13_20_31, 2);
    vst1q_lane_f32(&AB[7+1*8], ab1_02_13_20_31, 3);
    
    //
    // Save ab2
    //

    vst1q_lane_f32(&AB[0+4*8], ab2_00_11_22_33, 0);
    vst1q_lane_f32(&AB[1+5*8], ab2_00_11_22_33, 1);
    vst1q_lane_f32(&AB[2+6*8], ab2_00_11_22_33, 2);
    vst1q_lane_f32(&AB[3+7*8], ab2_00_11_22_33, 3);

    vst1q_lane_f32(&AB[0+5*8], ab2_01_10_23_32, 0);
    vst1q_lane_f32(&AB[1+4*8], ab2_01_10_23_32, 1);
    vst1q_lane_f32(&AB[2+7*8], ab2_01_10_23_32, 2);
    vst1q_lane_f32(&AB[3+6*8], ab2_01_10_23_32, 3);

    vst1q_lane_f32(&AB[0+7*8], ab2_03_12_21_30, 0);
    vst1q_lane_f32(&AB[1+6*8], ab2_03_12_21_30, 1);
    vst1q_lane_f32(&AB[2+5*8], ab2_03_12_21_30, 2);
    vst1q_lane_f32(&AB[3+4*8], ab2_03_12_21_30, 3);

    vst1q_lane_f32(&AB[0+6*8], ab2_02_13_20_31, 0);
    vst1q_lane_f32(&AB[1+7*8], ab2_02_13_20_31, 1);
    vst1q_lane_f32(&AB[2+4*8], ab2_02_13_20_31, 2);
    vst1q_lane_f32(&AB[3+5*8], ab2_02_13_20_31, 3);
    
    //
    // Save ab3
    //

    vst1q_lane_f32(&AB[4+4*8], ab3_00_11_22_33, 0);
    vst1q_lane_f32(&AB[5+5*8], ab3_00_11_22_33, 1);
    vst1q_lane_f32(&AB[6+6*8], ab3_00_11_22_33, 2);
    vst1q_lane_f32(&AB[7+7*8], ab3_00_11_22_33, 3);

    vst1q_lane_f32(&AB[4+5*8], ab3_01_10_23_32, 0);
    vst1q_lane_f32(&AB[5+4*8], ab3_01_10_23_32, 1);
    vst1q_lane_f32(&AB[6+7*8], ab3_01_10_23_32, 2);
    vst1q_lane_f32(&AB[7+6*8], ab3_01_10_23_32, 3);

    vst1q_lane_f32(&AB[4+7*8], ab3_03_12_21_30, 0);
    vst1q_lane_f32(&AB[5+6*8], ab3_03_12_21_30, 1);
    vst1q_lane_f32(&AB[6+5*8], ab3_03_12_21_30, 2);
    vst1q_lane_f32(&AB[7+4*8], ab3_03_12_21_30, 3);

    vst1q_lane_f32(&AB[4+6*8], ab3_02_13_20_31, 0);
    vst1q_lane_f32(&AB[5+7*8], ab3_02_13_20_31, 1);
    vst1q_lane_f32(&AB[6+4*8], ab3_02_13_20_31, 2);
    vst1q_lane_f32(&AB[7+5*8], ab3_02_13_20_31, 3);


//
//  Update C <- beta*C
//
    if (beta==0.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

//
//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
//                                  the above layer dgemm_nn)
//
    if (alpha==1.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*MR];
            }
        }
    } else {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                float added = C[i*incRowC+j*incColC] + alpha*AB[i+j*MR];
                C[i*incRowC+j*incColC] = added > 0 ? added : 0;
            }
        }
    }
    */
}

//
//  Compute Y += alpha*X
//
static void
sgeaxpy(int           m,
        int           n,
        float        alpha,
        const float  *X,
        int           incRowX,
        int           incColX,
        float        *Y,
        int           incRowY,
        int           incColY)
{
    int i, j;


    if (alpha!=1.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
            }
        }
    }
}

//
//  Compute X *= alpha
//
static void
sgescal(int     m,
        int     n,
        float  alpha,
        float  *X,
        int     incRowX,
        int     incColX)
{
    int i, j;

    if (alpha!=0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] *= alpha;
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = 0.0;
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
static void
sgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   float   alpha,
                   float   beta,
                   float   *C,
                   int     incRowC,
                   int     incColC)
{
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

            if (mr==MR && nr==NR) {
                sgemm_micro_kernel_naive(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                   beta,
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC);
            } else {
                sgemm_micro_kernel_naive(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                   0.0,
                                   _C, 1, MR);
                sgescal(mr, nr, beta,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
                sgeaxpy(mr, nr, 1.0, _C, 1, MR,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B and ReLU.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
static void
sgemm_relu_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   float   alpha,
                   float   beta,
                   float   *C,
                   int     incRowC,
                   int     incColC)
{
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

            if (mr==MR && nr==NR) {
                sgemm_relu_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                   beta,
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC);
            } else {
                sgemm_relu_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                   0.0,
                                   _C, 1, MR);
                sgescal(mr, nr, beta,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
                sgeaxpy(mr, nr, 1.0, _C, 1, MR,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}

//
//  Compute C <- beta*C + alpha*A*B
//
void
sgemm_neon(int            m,
         int            n,
         int            k,
         float          alpha,
         const float    *A,
         int            incRowA,
         int            incColA,
         const float    *B,
         int            incRowB,
         int            incColB,
         float          beta,
         float          *C,
         int            incRowC,
         int            incColC)
{
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    float _beta;

    if (alpha==0.0 || k==0) {
        sgescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            _beta = (l==0) ? beta : 1.0;

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   _B);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc,
                       &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                       _A);

                sgemm_macro_kernel(mc, nc, kc, alpha, _beta,
                                   &C[i*MC*incRowC+j*NC*incColC],
                                   incRowC, incColC);
            }
        }
    }
}

//
//  Compute C <- beta*C + alpha*A*B
//
void
sgemm_relu_neon(
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
) {
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    float _beta;

    if (alpha==0.0 || k==0) {
        sgescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            _beta = (l==0) ? beta : 1.0;

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   _B);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc,
                       &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                       _A);

                sgemm_relu_macro_kernel(mc, nc, kc, alpha, _beta,
                                   &C[i*MC*incRowC+j*NC*incColC],
                                   incRowC, incColC);
            }
        }
    }
}