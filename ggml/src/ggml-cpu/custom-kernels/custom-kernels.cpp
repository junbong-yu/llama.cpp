// Custom kernel module for A/B performance comparison experiments.
//
// This module intercepts specific GGML ops via the extra_buffer_type mechanism,
// forwarding them to either the original (baseline) implementation or a custom
// replacement.  Toggle individual ops with the CUSTOM_KERNEL_* env-vars at
// startup (defaults: all ON so that the dispatch overhead itself is measurable).
//
// Usage:
//   CUSTOM_KERNEL_MUL_MAT=1  CUSTOM_KERNEL_RMS_NORM=1  \
//   CUSTOM_KERNEL_SOFT_MAX=1 CUSTOM_KERNEL_ROPE=1      ./llama-bench ...

#include "custom-kernels.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "traits.h"
#include "ops.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Env-var helpers – read once at init
// ---------------------------------------------------------------------------
static bool env_enabled(const char * name, bool default_val) {
    const char * v = getenv(name);
    if (!v) return default_val;
    return v[0] == '1' || v[0] == 'y' || v[0] == 'Y';
}

struct custom_kernel_config {
    bool mul_mat;
    bool rms_norm;
    bool soft_max;
    bool rope;
};

static custom_kernel_config get_config() {
    static custom_kernel_config cfg = {
        env_enabled("CUSTOM_KERNEL_MUL_MAT",  true),
        env_enabled("CUSTOM_KERNEL_RMS_NORM",  true),
        env_enabled("CUSTOM_KERNEL_SOFT_MAX",  true),
        env_enabled("CUSTOM_KERNEL_ROPE",      true),
    };
    static bool once = [&]() {
        fprintf(stderr,
            "[custom-kernels] MUL_MAT=%d  RMS_NORM=%d  SOFT_MAX=%d  ROPE=%d\n",
            cfg.mul_mat, cfg.rms_norm, cfg.soft_max, cfg.rope);
        return true;
    }();
    (void)once;
    return cfg;
}

// ---------------------------------------------------------------------------
// Custom kernel implementations  (currently: identical copies / pass-through)
//
//   *** EDIT THESE to experiment with your optimized versions ***
//
// Each function follows the exact same signature as the original in ops.h /
// ggml-cpu.c.  To create a truly independent copy you can paste the original
// source here and modify it.  For the initial baseline comparison the
// functions simply delegate to the originals so the ONLY measurable
// difference is the extra_buffer_type dispatch overhead.
// ---------------------------------------------------------------------------

static void custom_mul_mat(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    // >>> Replace with your optimized MUL_MAT kernel <<<
    ggml_compute_forward_mul_mat(params, dst);
}

static void custom_rms_norm(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    // >>> Replace with your optimized RMS_NORM kernel <<<
    ggml_compute_forward_rms_norm(params, dst);
}

static void custom_soft_max(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    // >>> Replace with your optimized SOFT_MAX kernel <<<
    ggml_compute_forward_soft_max(params, dst);
}

static void custom_rope(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    // >>> Replace with your optimized ROPE kernel <<<
    ggml_compute_forward_rope(params, dst);
}

// ---------------------------------------------------------------------------
// tensor_traits – dispatches compute_forward to our custom kernels
// ---------------------------------------------------------------------------
namespace ggml::cpu::custom_kernels {

class tensor_traits : public ggml::cpu::tensor_traits {
  public:
    bool work_size(int /* n_threads */, const struct ggml_tensor * /* op */, size_t & /* size */) override {
        // Return false → fall back to default work-size calculation.
        // Override this if your custom kernel needs a different scratch size.
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        auto cfg = get_config();

        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (cfg.mul_mat) { custom_mul_mat(params, op); return true; }
                break;
            case GGML_OP_RMS_NORM:
                if (cfg.rms_norm) { custom_rms_norm(params, op); return true; }
                break;
            case GGML_OP_SOFT_MAX:
                if (cfg.soft_max) { custom_soft_max(params, op); return true; }
                break;
            case GGML_OP_ROPE:
                if (cfg.rope) { custom_rope(params, op); return true; }
                break;
            default:
                break;
        }
        return false;
    }
};

static tensor_traits * get_traits_instance() {
    static tensor_traits instance;
    return &instance;
}

// ---------------------------------------------------------------------------
// extra_buffer_type – tells the CPU backend which ops we can handle
// ---------------------------------------------------------------------------
class extra_buffer_type : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t /* dev */, const struct ggml_tensor * op) override {
        auto cfg = get_config();
        switch (op->op) {
            case GGML_OP_MUL_MAT:  return cfg.mul_mat;
            case GGML_OP_RMS_NORM: return cfg.rms_norm;
            case GGML_OP_SOFT_MAX: return cfg.soft_max;
            case GGML_OP_ROPE:     return cfg.rope;
            default: return false;
        }
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        // If any source tensor lives in our buffer, intercept via its extra pointer
        for (int i = 0; i < GGML_MAX_SRC && op->src[i]; i++) {
            if (op->src[i]->extra == (void *) get_traits_instance()) {
                return get_traits_instance();
            }
        }
        return nullptr;
    }
};

} // namespace ggml::cpu::custom_kernels

// ---------------------------------------------------------------------------
// Buffer implementation – plain CPU memory, identical to CPU host buffer.
// The only addition: init_tensor sets tensor->extra to our tensor_traits
// so that ggml_cpu_extra_compute_forward() picks up our custom kernels.
// ---------------------------------------------------------------------------
static void custom_buf_free(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * custom_buf_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static enum ggml_status custom_buf_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::custom_kernels::get_traits_instance();
    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static void custom_buf_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                     uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void custom_buf_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                  const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void custom_buf_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                  void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static void custom_buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i custom_buffer_iface = {
    /* .free_buffer   = */ custom_buf_free,
    /* .get_base      = */ custom_buf_get_base,
    /* .init_tensor   = */ custom_buf_init_tensor,
    /* .memset_tensor = */ custom_buf_memset_tensor,
    /* .set_tensor    = */ custom_buf_set_tensor,
    /* .get_tensor    = */ custom_buf_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ custom_buf_clear,
    /* .reset         = */ nullptr,
};

// ---------------------------------------------------------------------------
// Buffer type interface
// ---------------------------------------------------------------------------
static const char * custom_buft_get_name(ggml_backend_buffer_type_t) {
    return "CUSTOM_KERNELS";
}

static ggml_backend_buffer_t custom_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (!data) {
        fprintf(stderr, "[custom-kernels] failed to allocate buffer of size %zu\n", size);
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, custom_buffer_iface, data, size);
}

static size_t custom_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 64;
}

static bool custom_buft_is_host(ggml_backend_buffer_type_t) {
    return true;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
ggml_backend_buffer_type_t ggml_backend_cpu_custom_kernels_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name      = */ custom_buft_get_name,
            /* .alloc_buffer  = */ custom_buft_alloc_buffer,
            /* .get_alignment = */ custom_buft_get_alignment,
            /* .get_max_size  = */ nullptr,
            /* .get_alloc_size= */ nullptr,
            /* .is_host       = */ custom_buft_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::custom_kernels::extra_buffer_type(),
    };
    return &buft;
}
