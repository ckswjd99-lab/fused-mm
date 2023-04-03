#include <math.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#define EPSILON 0.001
#define ROWCOL2IDX(R, r, c) (r + R * c)

void matrix_init_rand(float32_t *M, uint32_t numvals);
void matrix_init_zero(float32_t *M, uint32_t numvals);
int matrix_comp(float32_t *A, float32_t *B, uint32_t rows, uint32_t cols);

void print_vector(float32_t *src, uint32_t rows);
void print_vector_T(float32_t *src, uint32_t rows);
void print_matrix(float32_t *src, uint32_t rows, uint32_t cols);
