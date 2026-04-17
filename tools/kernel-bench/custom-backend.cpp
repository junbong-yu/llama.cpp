// Sample custom ggml backend — minimal wrapper around the CPU backend.
//
// Demonstrates how to register a separate ggml_backend_t that can be used
// side-by-side with CUDA / Metal / Hexagon for backend-level comparison.
// Internally calls ggml_backend_graph_compute() on a private CPU backend.

#include "custom-backend.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <atomic>
#include <cstring>
#include <string>

struct custom_backend_context {
    ggml_backend_t    internal_cpu = nullptr;
    std::string       name;
    std::atomic<bool> use_custom_kernels{false};
};

// Unique GUID for this backend type (any fixed 16 bytes)
static const ggml_guid custom_backend_guid = {
    0xC5, 0x73, 0x01, 0x42, 0xA8, 0xB1, 0x4C, 0x9F,
    0x8E, 0x5D, 0x2A, 0xF0, 0x13, 0xB6, 0xE7, 0x29,
};

// ---------------------------------------------------------------------------
// iface implementations — most forward to the internal CPU backend
// ---------------------------------------------------------------------------

static const char * ggml_backend_custom_get_name(ggml_backend_t backend) {
    auto * ctx = (custom_backend_context *)backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_custom_free(ggml_backend_t backend) {
    auto * ctx = (custom_backend_context *)backend->context;
    if (ctx->internal_cpu) {
        ggml_backend_free(ctx->internal_cpu);
    }
    delete ctx;
    delete backend;
}

static void ggml_backend_custom_synchronize(ggml_backend_t backend) {
    auto * ctx = (custom_backend_context *)backend->context;
    ggml_backend_synchronize(ctx->internal_cpu);
}

static enum ggml_status ggml_backend_custom_graph_compute(
        ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (custom_backend_context *)backend->context;
    // Delegate to the internal CPU backend.
    // The extra_buffer_type hook (if registered externally, e.g. in
    // inference-bench.cpp) will fire during this call and optionally swap
    // kernels based on ctx->use_custom_kernels.
    return ggml_backend_graph_compute(ctx->internal_cpu, cgraph);
}

// Tensor data access: forward to the internal CPU backend via its buffer.
// Since both backends share the CPU device, the buffer handles are compatible.
static void ggml_backend_custom_set_tensor_async(
        ggml_backend_t backend, struct ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * ctx = (custom_backend_context *)backend->context;
    ggml_backend_tensor_set_async(ctx->internal_cpu, tensor, data, offset, size);
}

static void ggml_backend_custom_get_tensor_async(
        ggml_backend_t backend, const struct ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    auto * ctx = (custom_backend_context *)backend->context;
    ggml_backend_tensor_get_async(ctx->internal_cpu, tensor, data, offset, size);
}

static const struct ggml_backend_i ggml_backend_custom_i = {
    /* .get_name                = */ ggml_backend_custom_get_name,
    /* .free                    = */ ggml_backend_custom_free,
    /* .set_tensor_async        = */ ggml_backend_custom_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_custom_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_custom_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_custom_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

ggml_backend_t ggml_backend_custom_init(const char * name, bool use_custom_kernels) {
    auto * ctx = new custom_backend_context;
    ctx->internal_cpu = ggml_backend_cpu_init();
    if (!ctx->internal_cpu) {
        delete ctx;
        return nullptr;
    }
    ctx->name = name ? name : "Custom";
    ctx->use_custom_kernels.store(use_custom_kernels);

    // Reuse the CPU device for buffer_type compatibility.
    ggml_backend_dev_t cpu_dev = ggml_backend_get_device(ctx->internal_cpu);

    auto * backend = new ggml_backend {
        /* .guid    = */ (ggml_guid_t) &custom_backend_guid,
        /* .iface   = */ ggml_backend_custom_i,
        /* .device  = */ cpu_dev,
        /* .context = */ ctx,
    };
    return backend;
}

void ggml_backend_custom_set_use_custom_kernels(ggml_backend_t backend, bool use_custom_kernels) {
    if (!ggml_backend_is_custom(backend)) return;
    auto * ctx = (custom_backend_context *)backend->context;
    ctx->use_custom_kernels.store(use_custom_kernels);
}

bool ggml_backend_is_custom(ggml_backend_t backend) {
    return backend != nullptr &&
           ggml_guid_matches(backend->guid, (ggml_guid_t)&custom_backend_guid);
}
