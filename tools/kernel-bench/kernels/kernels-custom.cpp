#include "kernels-custom.h"
#include "kernels-standard.h"

// Custom kernel implementations.
//
// HOW TO ADD YOUR OPTIMIZED KERNEL:
//   1. Replace the body of any function below with your optimized implementation.
//   2. Rebuild: cmake --build build --target llama-kernel-bench
//   3. Run: ./build/bin/llama-kernel-bench --output results.json
//   4. The JSON output will show speedup vs the standard baseline.
//
// Example (AVX2 add):
//   #include <immintrin.h>
//   void kernel_add_custom(float * dst, const float * src0, const float * src1, int64_t n) {
//       int64_t i = 0;
//       for (; i + 8 <= n; i += 8) {
//           __m256 a = _mm256_loadu_ps(src0 + i);
//           __m256 b = _mm256_loadu_ps(src1 + i);
//           _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
//       }
//       for (; i < n; i++) { dst[i] = src0[i] + src1[i]; }
//   }

void kernel_add_custom(float * dst, const float * src0, const float * src1, int64_t n) {
    kernel_add_standard(dst, src0, src1, n);
}

void kernel_mul_custom(float * dst, const float * src0, const float * src1, int64_t n) {
    kernel_mul_standard(dst, src0, src1, n);
}

void kernel_relu_custom(float * dst, const float * src0, const float * src1, int64_t n) {
    kernel_relu_standard(dst, src0, src1, n);
}

void kernel_sigmoid_custom(float * dst, const float * src0, const float * src1, int64_t n) {
    kernel_sigmoid_standard(dst, src0, src1, n);
}

void kernel_silu_custom(float * dst, const float * src0, const float * src1, int64_t n) {
    kernel_silu_standard(dst, src0, src1, n);
}

void kernel_mul_mat_custom(float * C, const float * A, const float * B,
                           int64_t M, int64_t N, int64_t K) {
    kernel_mul_mat_standard(C, A, B, M, N, K);
}
