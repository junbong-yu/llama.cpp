#pragma once

// Sample custom ggml backend — wraps the standard CPU backend.
//
// This is a minimal example of registering a separate ggml_backend_t that can
// be used alongside CPU/CUDA/Metal/Hexagon for direct performance comparison.
// Internally, it delegates graph_compute to an internal CPU backend, but can
// optionally toggle the extra_buffer_type kernel-swap hook so that a
// dedicated "custom kernel" backend can be measured independently from the
// plain CPU backend within a single benchmark run.

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Create a new custom backend instance.
// If `use_custom_kernels` is true, the extra_buffer_type kernel-swap hook
// (registered by this backend) will run our custom kernel implementations.
GGML_API ggml_backend_t ggml_backend_custom_init(const char * name, bool use_custom_kernels);

// Enable / disable kernel swapping on a running backend
GGML_API void ggml_backend_custom_set_use_custom_kernels(ggml_backend_t backend, bool use_custom_kernels);

// Check if a backend is our custom backend
GGML_API bool ggml_backend_is_custom(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif
