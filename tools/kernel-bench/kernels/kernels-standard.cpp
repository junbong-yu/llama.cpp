#include "kernels-standard.h"
#include <cmath>

// Binary ops -- mirrors ggml/src/ggml-cpu/binary-ops.cpp scalar paths

void kernel_add_standard(float * dst, const float * src0, const float * src1, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = src0[i] + src1[i];
    }
}

void kernel_mul_standard(float * dst, const float * src0, const float * src1, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = src0[i] * src1[i];
    }
}

// Unary ops -- mirrors ggml/src/ggml-cpu/unary-ops.cpp scalar paths

void kernel_relu_standard(float * dst, const float * src0, const float * /*src1*/, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = src0[i] > 0.f ? src0[i] : 0.f;
    }
}

void kernel_sigmoid_standard(float * dst, const float * src0, const float * /*src1*/, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = 1.f / (1.f + expf(-src0[i]));
    }
}

void kernel_silu_standard(float * dst, const float * src0, const float * /*src1*/, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = src0[i] / (1.f + expf(-src0[i]));
    }
}

// MUL_MAT: C[MxN] = A[MxK] * B^T[NxK]  (ggml convention: B stored as [N x K])

void kernel_mul_mat_standard(float * C, const float * A, const float * B,
                             int64_t M, int64_t N, int64_t K) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}
