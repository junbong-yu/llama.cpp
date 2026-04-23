// SVE kernel plug-point — compile path evidence only.
//
// This file proves that the SVE build path is reachable from this codebase
// without shipping any actual kernel logic. It is NOT wired into the default
// build (CMakeLists does not list it), does NOT register anything into the
// kernel-registry, and contains NO algorithm body: the function bodies below
// are intentionally empty and must be filled in by the kernel author.
//
// Build check (from repo root):
//   clang --target=aarch64-linux-gnu -march=armv8-a+sve -O2 \
//         -c tools/kernel-bench/kernels/kernels-sve.cpp \
//         -o /tmp/kernels-sve.o
//
// On Apple clang 17 the native `-march=armv8-a+sve` build also passes; note
// that `svwhilelt_b32` is overloaded on int32/uint32/int64/uint64, so any
// size-typed index passed to it needs an explicit cast (e.g. uint64_t).
// Running the produced object on Apple Silicon (no SVE hardware) raises
// SIGILL — this file only proves the compile path, not execution.

#include "kernels-sve.h"

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#include <stddef.h>

namespace kernels_sve {

// sve_probe — minimal reference to the SVE header so that removing this
// function would immediately surface a broken include path. Not intended to
// be called; kept inline-hidden to avoid symbol-surface noise.
static inline unsigned sve_probe() {
    return svcntw();  // vector length in f32 lanes; depends on hardware.
}

// === TODO(user): put your SVE implementations below ===
// Signatures mirror the kernel-registry unary/binary f32 contract; the caller
// is expected to have already applied the ith/nth thread partition, so these
// functions operate on a contiguous [0, n) slice.

void relu_f32(const float * x, float * y, size_t n) {
    (void) x; (void) y; (void) n;
    // TODO(user): implement ReLU using SVE intrinsics.
}

void gelu_f32(const float * x, float * y, size_t n) {
    (void) x; (void) y; (void) n;
    // TODO(user): implement GeLU using SVE intrinsics.
    // Remember to pick a variant (exact erf / tanh-approx / sigmoid-approx)
    // consistent with what the target model actually uses.
}

void silu_f32(const float * x, float * y, size_t n) {
    (void) x; (void) y; (void) n;
    // TODO(user): implement SiLU using SVE intrinsics.
}

// SwiGLU signature depends on whether the model uses the fused GGML_OP_SWIGLU
// or the composite SILU(x) * gate path. Finalize after kernel-dev-perf
// Phase 1 (analyst FFN trace).

} // namespace kernels_sve

#else
// Non-SVE build: keep the translation unit valid so that unconditionally
// listing this file in a build graph does not break targets compiled without
// -march=armv8-a+sve.
namespace kernels_sve { /* empty */ }
#endif
