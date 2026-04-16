#pragma once

#include <cstdint>

// Binary ops
void kernel_add_standard(float * dst, const float * src0, const float * src1, int64_t n);
void kernel_mul_standard(float * dst, const float * src0, const float * src1, int64_t n);

// Unary ops (src1 unused)
void kernel_relu_standard(float * dst, const float * src0, const float * src1, int64_t n);
void kernel_sigmoid_standard(float * dst, const float * src0, const float * src1, int64_t n);
void kernel_silu_standard(float * dst, const float * src0, const float * src1, int64_t n);

// MUL_MAT: C[MxN] = A[MxK] * B^T[NxK]
void kernel_mul_mat_standard(float * C, const float * A, const float * B,
                             int64_t M, int64_t N, int64_t K);
