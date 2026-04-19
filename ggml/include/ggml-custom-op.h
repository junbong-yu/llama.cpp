#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_custom_op_init(void);

GGML_BACKEND_API bool ggml_backend_is_custom_op(ggml_backend_t backend);

// register a custom kernel for a specific operation
// Returns true if registration was successful
typedef bool (*ggml_custom_op_kernel_fn)(const struct ggml_compute_params * params, struct ggml_tensor * dst);
typedef bool (*ggml_custom_op_can_handle_fn)(const struct ggml_tensor * dst);

GGML_BACKEND_API bool ggml_backend_custom_op_register_kernel(enum ggml_op                 op,
                                                             const char *                 name,
                                                             ggml_custom_op_kernel_fn     compute,
                                                             ggml_custom_op_can_handle_fn can_handle);

// set number of threads for parallel computation
GGML_BACKEND_API void ggml_backend_custom_op_set_n_threads(ggml_backend_t backend_custom_op, int n_threads);

// backend registry
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_custom_op_reg(void);

#ifdef __cplusplus
}
#endif
