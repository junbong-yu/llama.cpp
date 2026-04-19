// benchmark-custom-op.cpp
//
// Demo/benchmark for the custom-op backend.
// Shows how to:
//   1. Load the custom-op backend (via GGML_BACKEND_PATH or ggml_backend_load)
//   2. Register a custom MUL_MAT kernel
//   3. Run a graph and compare performance against the CPU backend
//
// Build (with GGML_CUSTOM_OP=ON and GGML_BACKEND_DL=ON):
//   cmake -DGGML_CUSTOM_OP=ON -DGGML_BACKEND_DL=ON ...
//   make
//
// Run:
//   GGML_BACKEND_PATH=./libggml-custom-op.so ./benchmark-custom-op
//   OR
//   ./benchmark-custom-op  (if statically linked)

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef GGML_USE_CUSTOM_OP
#    include "ggml-custom-op.h"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ==============================================================================
// Example custom MUL_MAT kernel (naive matmul for demonstration)
// ==============================================================================

static bool custom_mul_mat_can_handle(const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0)) {
        return false;
    }
    if (!ggml_is_contiguous(src1)) {
        return false;
    }

    return true;
}

static bool custom_mul_mat_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int ith = params->ith;
    const int nth = params->nth;

    // Simple row-parallel: each thread handles a subset of dst rows
    const int64_t dr  = (ne11 + nth - 1) / nth;
    const int64_t ir0 = ith * dr;
    const int64_t ir1 = std::min(ir0 + dr, ne11);

    if (ir0 >= ir1) {
        return true;
    }

    for (int64_t ir = ir0; ir < ir1; ir++) {
        for (int64_t ic = 0; ic < ne01; ic++) {
            float sum = 0.0f;
            for (int64_t ik = 0; ik < ne00; ik++) {
                const float a = ((const float *) src0->data)[ic * ne00 + ik];
                const float b = ((const float *) src1->data)[ir * ne10 + ik];
                sum += a * b;
            }
            ((float *) dst->data)[ir * ne01 + ic] = sum;
        }
    }

    return true;
}

// ==============================================================================
// Benchmark helpers
// ==============================================================================

static double time_graph_compute(ggml_backend_t        backend,
                                 struct ggml_context * ctx,
                                 struct ggml_cgraph *  cgraph,
                                 int                   n_iter) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        enum ggml_status status = ggml_backend_graph_compute(backend, cgraph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed with status %d\n", status);
            return -1.0;
        }
    }
    auto                                      end     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / n_iter;
}

// ==============================================================================
// Main
// ==============================================================================

int main(int argc, char ** argv) {
    int     n_iter = 10;
    int64_t m      = 512;
    int64_t n      = 512;
    int64_t k      = 512;

    if (argc >= 4) {
        m = atol(argv[1]);
        n = atol(argv[2]);
        k = atol(argv[3]);
    }
    if (argc >= 5) {
        n_iter = atoi(argv[4]);
    }

    printf("=== Custom Op Backend Benchmark ===\n");
    printf("Matrix dimensions: M=%lld, N=%lld, K=%lld\n", (long long) m, (long long) n, (long long) k);
    printf("Iterations: %d\n\n", n_iter);

    ggml_backend_load_all();

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        fprintf(stderr, "Failed to initialize CPU backend\n");
        return 1;
    }
    printf("CPU backend: %s\n", ggml_backend_name(cpu_backend));

    ggml_backend_t     custom_backend = nullptr;
    ggml_backend_reg_t custom_reg     = ggml_backend_reg_by_name("custom-op");
    if (custom_reg) {
        ggml_backend_dev_t custom_dev = ggml_backend_reg_dev_get(custom_reg, 0);
        if (custom_dev) {
            custom_backend = ggml_backend_dev_init(custom_dev, nullptr);
        }
    }

    if (!custom_backend) {
        // Try loading via path
#ifdef GGML_USE_CUSTOM_OP
        custom_backend = ggml_backend_custom_op_init();
#endif
    }

    if (!custom_backend) {
        fprintf(stderr,
                "Custom-op backend not available. Run with GGML_BACKEND_PATH pointing to libggml-custom-op.so\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }
    printf("Custom backend: %s\n\n", ggml_backend_name(custom_backend));

    // Register our custom MUL_MAT kernel
#ifdef GGML_USE_CUSTOM_OP
    ggml_backend_custom_op_register_kernel(GGML_OP_MUL_MAT, "naive_mul_mat_f32", custom_mul_mat_compute,
                                           custom_mul_mat_can_handle);
    ggml_backend_custom_op_set_n_threads(custom_backend, 4);
    printf("Registered custom MUL_MAT kernel (naive_matmul_f32)\n\n");
#endif

    // Create compute graph: dst = src0 @ src1
    size_t ctx_size = 0;
    ctx_size += sizeof(float) * m * k * 2;  // src0
    ctx_size += sizeof(float) * k * n * 2;  // src1
    ctx_size += sizeof(float) * m * n * 2;  // dst
    ctx_size += 16 * 1024 * 1024;           // overhead

    struct ggml_init_params params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    struct ggml_tensor * dst  = ggml_mul_mat(ctx, src0, src1);

    ggml_set_name(src0, "src0");
    ggml_set_name(src1, "src1");
    ggml_set_name(dst, "dst");

    // Build compute graph
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, dst);

    // Allocate buffers
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate tensors\n");
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(custom_backend);
        return 1;
    }

    // Initialize src0, src1 with random data
    {
        const float   scale = 1.0f / sqrtf((float) k);
        const int64_t n0    = ggml_nelements(src0);
        const int64_t n1    = ggml_nelements(src1);
        float *       data0 = (float *) ggml_backend_buffer_get_base(buffer) + (src0->data - nullptr) / sizeof(float);
        float *       data1 = (float *) ggml_backend_buffer_get_base(buffer) + (src1->data - nullptr) / sizeof(float);
        // Actually, we need to use ggml_backend_tensor_set for proper offset calculation
        // For simplicity, use the base pointer approach
        float *       buf_base = (float *) ggml_backend_buffer_get_base(buffer);
        float *       d0       = (float *) ((char *) buf_base + ((char *) src0->data - (char *) nullptr));
        float *       d1       = (float *) ((char *) buf_base + ((char *) src1->data - (char *) nullptr));

        // Simple seeded random fill
        unsigned int seed = 42;
        for (int64_t i = 0; i < n0; i++) {
            seed  = seed * 1103515245 + 12345;
            d0[i] = ((float) ((seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * scale;
        }
        seed = 123;
        for (int64_t i = 0; i < n1; i++) {
            seed  = seed * 1103515245 + 12345;
            d1[i] = ((float) ((seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * scale;
        }
    }

    // Run CPU benchmark
    printf("Running CPU backend benchmark...\n");
    ggml_backend_synchronize(cpu_backend);
    double cpu_ms = time_graph_compute(cpu_backend, ctx, cgraph, n_iter);
    printf("  CPU: %.3f ms/iter\n", cpu_ms);

    // For the custom backend, we need to copy tensors to its buffer
    // Since custom backend uses CPU buffers, we can share the same buffer
    // We create a second graph context that uses the custom backend
    printf("\nRunning Custom-Op backend benchmark...\n");

    // Allocate on custom backend
    ggml_backend_buffer_t custom_buffer = ggml_backend_alloc_ctx_tensors(ctx, custom_backend);
    if (!custom_buffer) {
        fprintf(stderr, "Failed to allocate tensors on custom backend\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(custom_backend);
        return 1;
    }

    // Copy data from CPU buffer to custom buffer
    ggml_backend_tensor_copy(src0, src0);
    ggml_backend_tensor_copy(src1, src1);

    ggml_cgraph * cgraph_custom = ggml_new_graph_custom(ctx, cgraph->size, false);
    ggml_build_forward_expand(cgraph_custom, dst);

    ggml_backend_synchronize(custom_backend);
    double custom_ms = time_graph_compute(custom_backend, ctx, cgraph_custom, n_iter);
    printf("  Custom-Op: %.3f ms/iter\n", custom_ms);

    if (cpu_ms > 0 && custom_ms > 0) {
        printf("\n  Speedup: %.2fx (custom vs CPU)\n", cpu_ms / custom_ms);
    }

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(custom_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(custom_backend);

    printf("\n=== Benchmark complete ===\n");
    return 0;
}
