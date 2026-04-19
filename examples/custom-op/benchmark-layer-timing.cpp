// benchmark-layer-timing.cpp
//
// Per-layer operation timing benchmark for llama.cpp models.
// Uses ggml_backend_sched_eval_callback to measure each graph node's
// computation time and aggregates by layer index and operation type.
//
// Build:
//   cmake -DGGML_CUSTOM_OP=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON ...
//   cmake --build build --target benchmark-layer-timing
//
// Usage:
//   ./benchmark-layer-timing -m model.gguf -p "Hello world" -n 32
//   ./benchmark-layer-timing -m model.gguf -p "Hello world" --warmup 1 --repeat 3

#include "ggml-backend.h"
#include "ggml.h"

#ifdef GGML_USE_CUSTOM_OP
#    include "ggml-custom-op.h"
#endif

#include "arg.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// ==============================================================================
// Timing data structures
// ==============================================================================

struct op_timing {
    int64_t total_us = 0;
    int     count    = 0;
};

struct layer_timing {
    int64_t                          total_us = 0;
    std::map<std::string, op_timing> ops;
};

struct timing_context {
    std::map<int, layer_timing>      layers;
    std::map<std::string, op_timing> global_ops;
    int64_t                          prev_time_us = 0;
    int                              n_layers     = 0;
    bool                             collecting   = false;
    std::mutex                       mtx;
};

// ==============================================================================
// Helpers
// ==============================================================================

static int64_t now_us() {
    return ggml_time_us();
}

static bool parse_tensor_name(const char * name, int * layer_idx, char * op_name, size_t op_name_size) {
    if (!name || !name[0]) {
        *layer_idx = -1;
        if (op_name && op_name_size > 0) {
            op_name[0] = '\0';
        }
        return false;
    }

    const char * dash = strrchr(name, '-');
    if (dash && dash != name) {
        char * endp = nullptr;
        long   idx  = strtol(dash + 1, &endp, 10);
        if (endp != dash + 1 && *endp == '\0') {
            *layer_idx        = (int) idx;
            size_t prefix_len = (size_t) (dash - name);
            if (prefix_len >= op_name_size) {
                prefix_len = op_name_size - 1;
            }
            memcpy(op_name, name, prefix_len);
            op_name[prefix_len] = '\0';
            return true;
        }
    }

    *layer_idx = -1;
    size_t len = strlen(name);
    if (len >= op_name_size) {
        len = op_name_size - 1;
    }
    memcpy(op_name, name, len);
    op_name[len] = '\0';
    return false;
}

static const char * ggml_op_name(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
            return "NONE";
        case GGML_OP_ADD:
            return "ADD";
        case GGML_OP_MUL:
            return "MUL";
        case GGML_OP_MUL_MAT:
            return "MUL_MAT";
        case GGML_OP_SILU:
            return "SILU";
        case GGML_OP_RMS_NORM:
            return "RMS_NORM";
        case GGML_OP_SOFT_MAX:
            return "SOFT_MAX";
        case GGML_OP_ROPE:
            return "ROPE";
        case GGML_OP_GLU:
            return "GLU";
        case GGML_OP_concat:
            return "CONCAT";
        case GGML_OP_RESHAPE:
            return "RESHAPE";
        case GGML_OP_VIEW:
            return "VIEW";
        case GGML_OP_PERMUTE:
            return "PERMUTE";
        case GGML_OP_TRANSPOSE:
            return "TRANSPOSE";
        case GGML_OP_CPY:
            return "CPY";
        case GGML_OP_CONT:
            return "CONT";
        case GGML_OP_DIAG_MASK:
            return "DIAG_MASK";
        case GGML_OP_SCALE:
            return "SCALE";
        case GGML_OP_UNARY:
            return "UNARY";
        default:
            {
                static char buf[32];
                snprintf(buf, sizeof(buf), "OP_%d", (int) op);
                return buf;
            }
    }
}

// ==============================================================================
// Eval callback — called for every graph node during compute
// ==============================================================================

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    timing_context * ctx = (timing_context *) user_data;

    if (ask) {
        // Always observe — we need per-node timing
        return true;
    }

    // ask == false: node computation just completed
    int64_t cur_time  = now_us();
    int64_t elapsed   = cur_time - ctx->prev_time_us;
    ctx->prev_time_us = cur_time;

    char op_name[64] = {};
    int  layer_idx   = -1;
    parse_tensor_name(ggml_get_name(t), &layer_idx, op_name, sizeof(op_name));

    // If no parsed op name, use ggml_op type
    if (op_name[0] == '\0') {
        snprintf(op_name, sizeof(op_name), "%s", ggml_op_name(t->op));
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    if (layer_idx >= 0) {
        ctx->layers[layer_idx].total_us += elapsed;
        ctx->layers[layer_idx].ops[op_name].total_us += elapsed;
        ctx->layers[layer_idx].ops[op_name].count++;
        if (layer_idx + 1 > ctx->n_layers) {
            ctx->n_layers = layer_idx + 1;
        }
    }

    ctx->global_ops[op_name].total_us += elapsed;
    ctx->global_ops[op_name].count++;

    return true;
}

// ==============================================================================
// Output formatting
// ==============================================================================

static void print_layer_timing(timing_context * ctx, const char * phase, int n_prompt, int n_gen) {
    printf("\n=== Per-Layer Timing (%s) ===\n", phase);
    if (n_prompt > 0) {
        printf("Prompt tokens: %d\n", n_prompt);
    }
    if (n_gen > 0) {
        printf("Generated tokens: %d\n", n_gen);
    }
    printf("\n");

    int64_t total_all_layers_us = 0;
    for (auto & [li, lt] : ctx->layers) {
        total_all_layers_us += lt.total_us;
    }

    printf("%-8s %12s  %-20s  %s\n", "Layer", "Time (ms)", "Op breakdown", "");
    printf("%-8s %12s  %-20s  %s\n", "-----", "---------", "------------------", "");

    for (int li = 0; li < ctx->n_layers; li++) {
        auto it = ctx->layers.find(li);
        if (it == ctx->layers.end()) {
            continue;
        }

        const layer_timing & lt       = it->second;
        double               layer_ms = lt.total_us / 1000.0;
        double               pct      = total_all_layers_us > 0 ? (100.0 * lt.total_us / total_all_layers_us) : 0.0;

        printf("%-8d %9.3f ms  (%5.1f%%)", li, layer_ms, pct);

        // Top-3 ops by time
        std::vector<std::pair<std::string, op_timing>> sorted_ops(lt.ops.begin(), lt.ops.end());
        std::sort(sorted_ops.begin(), sorted_ops.end(),
                  [](const auto & a, const auto & b) { return a.second.total_us > b.second.total_us; });

        printf("  ");
        int shown = 0;
        for (auto & [name, timing] : sorted_ops) {
            if (shown >= 3) {
                break;
            }
            if (shown > 0) {
                printf(", ");
            }
            printf("%s: %.2fms", name.c_str(), timing.total_us / 1000.0);
            shown++;
        }
        printf("\n");
    }

    printf("\n--- Summary ---\n");
    printf("Total layer compute: %.3f ms\n", total_all_layers_us / 1000.0);
    if (ctx->n_layers > 0) {
        printf("Avg per layer:       %.3f ms\n", (total_all_layers_us / 1000.0) / ctx->n_layers);
    }

    // Find slowest and fastest layers
    int64_t max_time = 0, min_time = INT64_MAX;
    int     max_layer = -1, min_layer = -1;
    for (auto & [li, lt] : ctx->layers) {
        if (lt.total_us > max_time) {
            max_time  = lt.total_us;
            max_layer = li;
        }
        if (lt.total_us < min_time) {
            min_time  = lt.total_us;
            min_layer = li;
        }
    }
    if (max_layer >= 0) {
        printf("Slowest layer:      %d (%.3f ms)\n", max_layer, max_time / 1000.0);
    }
    if (min_layer >= 0) {
        printf("Fastest layer:      %d (%.3f ms)\n", min_layer, min_time / 1000.0);
    }

    printf("\n--- Global Op Distribution ---\n");
    std::vector<std::pair<std::string, op_timing>> sorted_global(ctx->global_ops.begin(), ctx->global_ops.end());
    std::sort(sorted_global.begin(), sorted_global.end(),
              [](const auto & a, const auto & b) { return a.second.total_us > b.second.total_us; });

    printf("%-20s %12s %8s %8s\n", "Op", "Time (ms)", "Count", "% of Total");
    printf("%-20s %12s %8s %8s\n", "--", "--------", "-----", "---------");
    for (auto & [name, timing] : sorted_global) {
        double pct = total_all_layers_us > 0 ? (100.0 * timing.total_us / total_all_layers_us) : 0.0;
        printf("%-20s %9.3f ms %8d %7.1f%%\n", name.c_str(), timing.total_us / 1000.0, timing.count, pct);
    }
}

// ==============================================================================
// Main
// ==============================================================================

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx        = 512;
    params.n_batch      = 512;
    params.n_ubatch     = 512;
    params.n_predict    = 32;
    params.n_gpu_layers = 99;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BENCH)) {
        return 1;
    }

    common_init();

    if (params.model.path.empty()) {
        LOG_ERR("model path is required. Use -m <model_path>\n");
        return 1;
    }

    int n_warmup = 1;
    int n_repeat = 1;

    // Parse extra args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            n_warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            n_repeat = atoi(argv[++i]);
        }
    }

    printf("=== Layer Timing Benchmark ===\n");
    printf("Model: %s\n", params.model.path.c_str());
    printf("Warmup: %d, Repeat: %d\n", n_warmup, n_repeat);
    printf("n_gpu_layers: %d\n\n", params.n_gpu_layers);

    // =====================================================================
    // Load model
    // =====================================================================

    common_init_result llama_init = common_init_from_params(params);
    llama_model *      model      = llama_init.model;
    llama_context *    ctx        = llama_init.context;

    if (!model || !ctx) {
        LOG_ERR("failed to initialize model/context\n");
        return 1;
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_layer     = llama_model_n_layer(model);
    printf("Model info:\n");
    printf("  n_ctx_train: %d\n", n_ctx_train);
    printf("  n_layer:     %d\n", n_layer);
    printf("  n_embd:      %d\n", llama_model_n_embd(model));
    printf("  n_vocab:     %d\n\n", llama_model_n_vocab(model));

    // =====================================================================
    // Load custom-op backend if available
    // =====================================================================

#ifdef GGML_USE_CUSTOM_OP
    ggml_backend_reg_t custom_reg = ggml_backend_reg_by_name("custom-op");
    if (custom_reg) {
        ggml_backend_dev_t custom_dev = ggml_backend_reg_dev_get(custom_reg, 0);
        if (custom_dev) {
            printf("Custom-Op backend: detected (device: %s)\n", ggml_backend_dev_name(custom_dev));
        }
    } else {
        printf("Custom-Op backend: not loaded\n");
    }
#else
    ggml_backend_reg_t custom_reg = ggml_backend_reg_by_name("custom-op");
    if (custom_reg) {
        ggml_backend_dev_t custom_dev = ggml_backend_reg_dev_get(custom_reg, 0);
        if (custom_dev) {
            printf("Custom-Op backend: detected (device: %s)\n", ggml_backend_dev_name(custom_dev));
        }
    }
#endif

    // List backends
    printf("Active backends:\n");
    for (int i = 0; i < llama_backend_count(); i++) {
        ggml_backend_dev_t dev = llama_get_backend_device(i);
        printf("  [%d] %s (%s)\n", i, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    }
    printf("\n");

    // =====================================================================
    // Prepare prompt
    // =====================================================================

    const std::string prompt = params.prompt.empty() ? "Hello, how are you today?" : params.prompt;

    std::vector<llama_token> tokens   = common_tokenize(model, prompt, true, true);
    const int                n_prompt = (int) tokens.size();

    printf("Prompt: \"%s\"\n", prompt.c_str());
    printf("Tokens: %d\n\n", n_prompt);

    // =====================================================================
    // Timing context
    // =====================================================================

    timing_context timing;

    // =====================================================================
    // Warmup
    // =====================================================================

    for (int w = 0; w < n_warmup; w++) {
        printf("Warmup %d/%d...\n", w + 1, n_warmup);

        llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("warmup decode failed\n");
            return 1;
        }
        llama_batch_free(batch);

        // Generate a few tokens
        for (int i = 0; i < params.n_predict; i++) {
            llama_token new_token = llama_sampler_sample(params.sampling.smplers, ctx, -1);
            batch                 = llama_batch_get_one(&new_token, 1);
            if (llama_decode(ctx, batch) != 0) {
                break;
            }
            llama_batch_free(batch);
        }

        llama_kv_cache_clear(ctx);
        llama_synchronize(ctx);
    }

    // =====================================================================
    // Benchmark: prompt evaluation with per-layer timing
    // =====================================================================

    for (int r = 0; r < n_repeat; r++) {
        printf("Run %d/%d: Prompt evaluation...\n", r + 1, n_repeat);

        llama_kv_cache_clear(ctx);

        // Reset timing
        timing.layers.clear();
        timing.global_ops.clear();
        timing.prev_time_us = now_us();
        timing.collecting   = true;

        // Set the eval callback
        llama_set_eval_callback(ctx, eval_callback, &timing);

        llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);

        llama_decode(ctx, batch);
        llama_batch_free(batch);

        llama_synchronize(ctx);
        timing.collecting = false;
        llama_set_eval_callback(ctx, nullptr, nullptr);

        char phase_label[64];
        snprintf(phase_label, sizeof(phase_label), "Prompt Eval Run %d", r + 1);
        print_layer_timing(&timing, phase_label, n_prompt, 0);
    }

    // =====================================================================
    // Benchmark: generation with per-layer timing
    // =====================================================================

    for (int r = 0; r < n_repeat; r++) {
        printf("\nRun %d/%d: Token generation...\n", r + 1, n_repeat);

        // Keep KV from prompt
        timing.layers.clear();
        timing.global_ops.clear();
        timing.prev_time_us = now_us();
        timing.collecting   = true;

        llama_set_eval_callback(ctx, eval_callback, &timing);

        int n_generated = 0;
        for (int i = 0; i < params.n_predict; i++) {
            llama_token new_token = llama_sampler_sample(params.sampling.smplers, ctx, -1);
            llama_batch batch     = llama_batch_get_one(&new_token, 1);
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                break;
            }
            llama_batch_free(batch);
            n_generated++;
        }

        llama_synchronize(ctx);
        timing.collecting = false;
        llama_set_eval_callback(ctx, nullptr, nullptr);

        char phase_label[64];
        snprintf(phase_label, sizeof(phase_label), "Token Gen Run %d", r + 1);
        print_layer_timing(&timing, phase_label, 0, n_generated);
    }

    // =====================================================================
    // Backend distribution (if custom-op is present)
    // =====================================================================

    printf("\n=== Backend Distribution ===\n");
    // The scheduler assigns nodes to backends — we can check which ops went where
    // by looking at global_ops and cross-referencing with backend assignment
    printf("(Per-op timing collected above includes all backends)\n");

    // =====================================================================
    // Cleanup
    // =====================================================================

    llama_free(ctx);
    llama_model_free(model);
    common_cleanup();

    printf("\n=== Benchmark complete ===\n");
    return 0;
}
