#include "utils.h"

int f32comp_noteq(float32_t a, float32_t b) {
  if (fabs(a-b) < EPSILON) {
    return 0;
  }
  return 1;
}

void matrix_init_rand(float32_t *M, uint32_t numvals) {
  for (int i=0; i<numvals; i++) {
    M[i] = (float)rand()/(float)(RAND_MAX);
  }
}

void matrix_init_zero(float32_t *M, uint32_t numvals) {
  for (int i=0; i<numvals; i++) M[i] = 0;
}

int matrix_comp(float32_t *A, float32_t *B, uint32_t rows, uint32_t cols) {
  float32_t a;
  float32_t b;
  for (int i=0; i<rows; i++) {
    for (int j=0; j<cols; j++) {
      a = A[rows*j + i];
      b = B[rows*j + i];
      if (f32comp_noteq(a, b)) {
        printf("i=%d, j=%d, A=%f, B=%f\n", i, j, a, b);
        return 0;
      }
    }
  }
  return 1;
}

void print_vector(float32_t *src, uint32_t rows) {
    for (int i=0; i<rows; i++) {
        printf("%f\n", src[i]);
    }
}

void print_vector_T(float32_t *src, uint32_t rows) {
    for (int i=0; i<rows; i++) {
        printf("%f\t", src[i]);
    }
    printf("\n");
}

void print_matrix(float32_t *src, uint32_t rows, uint32_t cols) {
    for (int j=0; j<cols; j++) {
        for (int i=0; i<rows; i++) {
            printf("%f\t", src[i + j * rows]);
        }
        printf("\n");
    }
}

