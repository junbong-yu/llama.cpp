// Forward declarations for the SVE kernel plug-point.
// See kernels-sve.cpp for the (user-authored) implementations.

#pragma once
#include <stddef.h>

namespace kernels_sve {
    void relu_f32(const float * x, float * y, size_t n);
    void gelu_f32(const float * x, float * y, size_t n);
    void silu_f32(const float * x, float * y, size_t n);
}
