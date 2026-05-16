#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_custom_kernels_buffer_type(void);

#ifdef __cplusplus
}
#endif
