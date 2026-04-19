#include "ggml-custom-op.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstring>
#include <future>
#include <mutex>
#include <unordered_map>
#include <vector>

// ==============================================================================
// Custom Kernel Registry
// ==============================================================================

struct ggml_custom_kernel {
    enum ggml_op                 op;
    std::string                  name;
    ggml_custom_op_kernel_fn     compute;
    ggml_custom_op_can_handle_fn can_handle;
};

struct ggml_backend_custom_op_context {
    int                             n_threads = GGML_DEFAULT_N_THREADS;
    std::vector<ggml_custom_kernel> kernels;
    std::mutex                      kernels_mutex;

    // find a registered kernel for the given op; returns nullptr if none found
    const ggml_custom_kernel * find_kernel(const struct ggml_tensor * op) const {
        for (auto & k : kernels) {
            if (k.op == op->op) {
                if (k.can_handle == nullptr || k.can_handle(op)) {
                    return &k;
                }
            }
        }
        return nullptr;
    }
};

// ==============================================================================
// Backend (stream) interface
// ==============================================================================

static const char * ggml_backend_custom_op_get_name(ggml_backend_t backend) {
    return "Custom-Op";
    GGML_UNUSED(backend);
}

static void ggml_backend_custom_op_free(ggml_backend_t backend) {
    ggml_backend_custom_op_context * ctx = (ggml_backend_custom_op_context *) backend->context;
    delete ctx;
    delete backend;
}

// Forward declaration
static void ggml_backend_custom_op_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst);

static enum ggml_status ggml_backend_custom_op_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_custom_op_context * ctx = (ggml_backend_custom_op_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        const ggml_custom_kernel * kernel = ctx->find_kernel(node);
        if (kernel == nullptr) {
            // This should not happen - supports_op() should only return true for
            // ops we have kernels for. Log a warning and skip.
            GGML_LOG_WARN("%s: no kernel registered for op %s\n", __func__, ggml_op_desc(node));
            continue;
        }

        struct ggml_compute_params params = {
            /*.type  =*/GGML_TASK_TYPE_COMPUTE,
            /*.ith   =*/0,
            /*.nth   =*/1,
            /*.wsize =*/0,
            /*.wdata =*/nullptr,
        };

        // For simple kernels, use single-threaded execution.
        // For compute-intensive ops like MUL_MAT, use multi-threaded execution.
        const int nth = ctx->n_threads;

        if (nth <= 1) {
            if (!kernel->compute(&params, node)) {
                GGML_LOG_WARN("%s: kernel %s failed for op %s\n", __func__, kernel->name.c_str(), ggml_op_desc(node));
            }
        } else {
            // Multi-threaded execution: split work across threads
            // The kernel's compute function is responsible for handling the ith/nth split
            std::vector<std::future<void>> tasks;
            for (int ith = 1; ith < nth; ith++) {
                params.ith = ith;
                params.nth = nth;
                tasks.push_back(std::async(std::launch::async, [&kernel, &params, node]() {
                    if (!kernel->compute(&params, node)) {
                        GGML_LOG_WARN("%s: kernel %s failed for op %s (thread %d)\n", __func__, kernel->name.c_str(),
                                      ggml_op_desc(node), params.ith);
                    }
                }));
            }

            // Main thread
            params.ith = 0;
            params.nth = nth;
            if (!kernel->compute(&params, node)) {
                GGML_LOG_WARN("%s: kernel %s failed for op %s (thread 0)\n", __func__, kernel->name.c_str(),
                              ggml_op_desc(node));
            }

            for (auto & task : tasks) {
                task.get();
            }
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i custom_op_backend_i = {
    /* .get_name                = */ ggml_backend_custom_op_get_name,
    /* .free                    = */ ggml_backend_custom_op_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,  // synchronous, no-op
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_custom_op_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

// ==============================================================================
// GUID
// ==============================================================================

static ggml_guid_t ggml_backend_custom_op_guid(void) {
    // Unique GUID for this backend (randomly generated)
    static ggml_guid guid = { 0xc7, 0x2b, 0x9e, 0xf1, 0xa3, 0x45, 0x67, 0x89,
                              0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89 };
    return &guid;
}

// ==============================================================================
// Backend initialization
// ==============================================================================

ggml_backend_t ggml_backend_custom_op_init(void) {
    ggml_backend_custom_op_context * ctx = new ggml_backend_custom_op_context;

    // Drain pending kernels registered before init into this context
    {
        std::lock_guard<std::mutex> lock(g_kernels_mutex);
        ctx->kernels = std::move(g_pending_kernels);
        g_pending_kernels.clear();
    }

    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_custom_op_guid(),
        /* .iface   = */ custom_op_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_custom_op_reg(), 0),
        /* .context = */ ctx,
    };

    g_custom_op_ctx = ctx;

    return backend;
}

bool ggml_backend_is_custom_op(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_custom_op_guid());
}

void ggml_backend_custom_op_set_n_threads(ggml_backend_t backend_custom_op, int n_threads) {
    GGML_ASSERT(ggml_backend_is_custom_op(backend_custom_op));

    ggml_backend_custom_op_context * ctx = (ggml_backend_custom_op_context *) backend_custom_op->context;
    ctx->n_threads                       = n_threads;
}

// Global kernel registry - allows registration before backend init
static std::mutex                      g_kernels_mutex;
static std::vector<ggml_custom_kernel> g_pending_kernels;

bool ggml_backend_custom_op_register_kernel(enum ggml_op                 op,
                                            const char *                 name,
                                            ggml_custom_op_kernel_fn     compute,
                                            ggml_custom_op_can_handle_fn can_handle) {
    std::lock_guard<std::mutex> lock(g_kernels_mutex);

    ggml_custom_kernel kernel;
    kernel.op         = op;
    kernel.name       = name ? name : "";
    kernel.compute    = compute;
    kernel.can_handle = can_handle;

    if (g_custom_op_ctx != nullptr) {
        g_custom_op_ctx->kernels.push_back(kernel);
    } else {
        g_pending_kernels.push_back(kernel);
    }

    return true;
}

// ==============================================================================
// Device interface (ACCEL type)
// ==============================================================================

static const char * ggml_backend_custom_op_device_get_name(ggml_backend_dev_t dev) {
    return "Custom-Op";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_custom_op_device_get_description(ggml_backend_dev_t dev) {
    return "Custom Op Accelerator Backend";
    GGML_UNUSED(dev);
}

static void ggml_backend_custom_op_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // No dedicated memory - uses host memory
    *free  = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_custom_op_device_get_type(ggml_backend_dev_t dev) {
    // ACCEL type: accelerator intended to be used together with the CPU backend
    // This is critical - the scheduler orders backends as GPU -> ACCEL -> CPU
    // and requires the last backend to be CPU type.
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_custom_op_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_custom_op_device_get_name(dev);
    props->description = ggml_backend_custom_op_device_get_description(dev);
    props->type        = ggml_backend_custom_op_device_get_type(dev);
    ggml_backend_custom_op_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_custom_op_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_custom_op_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_custom_op_device_get_buffer_type(ggml_backend_dev_t dev) {
    // Reuse CPU buffer type since we operate on host memory
    return ggml_backend_cpu_buffer_type();
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_custom_op_device_buffer_from_host_ptr(ggml_backend_dev_t dev,
                                                                                void *             ptr,
                                                                                size_t             size,
                                                                                size_t             max_tensor_size) {
    // Reuse CPU buffer from host pointer
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

// Forward declarations for kernel registry access
// We need access to the backend context to find registered kernels.
// Since the context lives in the backend instance, we use a global pointer.
static ggml_backend_custom_op_context * g_custom_op_ctx = nullptr;

static bool ggml_backend_custom_op_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // Only claim ops that have a registered custom kernel.
    // The can_handle function further restricts which specific tensors we handle.
    if (g_custom_op_ctx == nullptr) {
        return false;
    }

    const ggml_custom_kernel * kernel = g_custom_op_ctx->find_kernel(op);
    return kernel != nullptr;

    GGML_UNUSED(dev);
}

static bool ggml_backend_custom_op_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // We support host (CPU) buffer types since we operate on host memory
    return ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static bool ggml_backend_custom_op_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // Return true for ops we want to offload from CPU to our accelerator.
    // This tells the scheduler: "I want to handle this op even though the
    // weights are in host memory - I can do it faster than CPU."
    if (g_custom_op_ctx == nullptr) {
        return false;
    }

    const ggml_custom_kernel * kernel = g_custom_op_ctx->find_kernel(op);
    return kernel != nullptr;

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_custom_op_device_i = {
    /* .get_name             = */ ggml_backend_custom_op_device_get_name,
    /* .get_description      = */ ggml_backend_custom_op_device_get_description,
    /* .get_memory           = */ ggml_backend_custom_op_device_get_memory,
    /* .get_type             = */ ggml_backend_custom_op_device_get_type,
    /* .get_props            = */ ggml_backend_custom_op_device_get_props,
    /* .init_backend         = */ ggml_backend_custom_op_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_custom_op_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_custom_op_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_custom_op_device_supports_op,
    /* .supports_buft        = */ ggml_backend_custom_op_device_supports_buft,
    /* .offload_op           = */ ggml_backend_custom_op_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ==============================================================================
// Backend registry interface
// ==============================================================================

static const char * ggml_backend_custom_op_reg_get_name(ggml_backend_reg_t reg) {
    return "custom-op";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_custom_op_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_custom_op_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_custom_op_device = {
        /* .iface   = */ ggml_backend_custom_op_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_custom_op_device;
    GGML_UNUSED(index);
}

static void * ggml_backend_custom_op_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_custom_op_register_kernel") == 0) {
        return (void *) ggml_backend_custom_op_register_kernel;
    }
    if (std::strcmp(name, "ggml_backend_custom_op_set_n_threads") == 0) {
        return (void *) ggml_backend_custom_op_set_n_threads;
    }
    return NULL;
    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_custom_op_reg_i = {
    /* .get_name         = */ ggml_backend_custom_op_reg_get_name,
    /* .get_device_count = */ ggml_backend_custom_op_reg_get_device_count,
    /* .get_device       = */ ggml_backend_custom_op_reg_get_device,
    /* .get_proc_address = */ ggml_backend_custom_op_get_proc_address,
};

ggml_backend_reg_t ggml_backend_custom_op_reg(void) {
    static struct ggml_backend_reg ggml_backend_custom_op_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_custom_op_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_custom_op_reg;
}

// Dynamic loading entry point
GGML_BACKEND_DL_IMPL(ggml_backend_custom_op_reg)
