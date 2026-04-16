#pragma once

#include <cstdint>

// Custom kernel implementations.
// Replace the bodies in kernels-custom.cpp with optimized versions
// (e.g. SIMD, OpenMP, loop tiling, etc.)

// Binary ops
void kernel_add_custom(float * dst, const float * src0, const float * src1, int64_t n);
void kernel_mul_custom(float * dst, const float * src0, const float * src1, int64_t n);

// Unary ops
void kernel_relu_custom(float * dst, const float * src0, const float * src1, int64_t n);
void kernel_sigmoid_custom(float * dst, const float * src0, const float * src1, int64_t n);
void kernel_silu_custom(float * dst, const float * src0, const float * src1, int64_t n);

// MUL_MAT
void kernel_mul_mat_custom(float * C, const float * A, const float * B,
                           int64_t M, int64_t N, int64_t K);
