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
#include "vec.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

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
// Custom kernel implementations
//
// RMS_NORM and SOFT_MAX: fully independent copies of the original code.
//   You can freely modify these without affecting the baseline.
//   The RMS_NORM below has a fused sum-of-squares + scale optimization:
//     Original: 2 passes (sum-of-squares pass, then memcpy + scale pass)
//     Custom:   1 pass   (fused sum-of-squares → scale in single loop)
//
// MUL_MAT and ROPE: pass-through to original (too large to copy inline).
//   Replace these when you have a custom implementation ready.
// ---------------------------------------------------------------------------

// --- MUL_MAT: pass-through (very complex, ~600 lines) ---
static void custom_mul_mat(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    ggml_compute_forward_mul_mat(params, dst);
}

// --- ROPE: pass-through ---
static void custom_rope(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    ggml_compute_forward_rope(params, dst);
}

// ---------------------------------------------------------------------------
// RMS_NORM — independent copy with fused 1-pass optimization
//
// Original algorithm (ops.cpp):
//   pass 1: sum += x[i]*x[i]             (read x)
//   pass 2: memcpy(y, x, ...)            (read x again, write y)
//   pass 3: ggml_vec_scale_f32(y, scale)  (read y, write y)
//
// Optimized: fuse into 1 pass:
//   pass 1: sum += x[i]*x[i]             (read x)
//           then compute scale
//   pass 2: y[i] = x[i] * scale          (read x once, write y once)
//
// This halves memory traffic on the second pass by avoiding the memcpy.
// ---------------------------------------------------------------------------
static void custom_rms_norm(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    if (src0->type != GGML_TYPE_F32) {
        // fallback for non-f32
        ggml_compute_forward_rms_norm(params, dst);
        return;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                float       * y = (float *) ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // --- pass 1: sum of squares ---
                double sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (double)(x[i00] * x[i00]);
                }

                const float scale = 1.0f / sqrtf((float)(sum / ne00) + eps);

                // --- pass 2: fused copy + scale (saves one full read+write vs memcpy+vec_scale) ---
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    y[i00] = x[i00] * scale;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SOFT_MAX — independent copy (identical to original for now)
//
// Uses the standard numerically-stable softmax:
//   1) find max
//   2) exp(x[i] - max)
//   3) normalize by sum
// ---------------------------------------------------------------------------
static void custom_soft_max(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];

    if (src0->type != GGML_TYPE_F32) {
        ggml_compute_forward_soft_max(params, dst);
        return;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head      = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2((double)n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    float * wp = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    const float * sk = src2 ? (float *)((char *) src2->data) : nullptr;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const int64_t i11 = i01;
                const int64_t i12 = i02 % ne12;
                const int64_t i13 = i03 % ne13;

                // ALiBi
                const uint32_t h = i02;
                const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;

                float * sp = (float *)((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                float * dp = (float *)((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : NULL;
                float       * mp_f32 = src1 ? (float       *)((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : NULL;

                ggml_vec_cpy_f32(ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);

                if (mp_f32) {
                    if (use_f16) {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope * GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } else {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope * mp_f32[i];
                        }
                    }
                }

                float max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                if (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                assert(sum > 0.0);

                if (sk) {
                    sum += (ggml_float) expf(sk[i02] - max);
                }

                sum = 1.0 / sum;
                ggml_vec_scale_f32(ne00, dp, sum);
            }
        }
    }
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
