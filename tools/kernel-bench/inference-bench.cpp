// inference-bench.cpp — Benchmark custom kernel swapping during actual model inference.
//
// Uses ggml's extra_buffer_type hook to intercept specific ops during inference
// and replace them with custom kernel implementations. Compares standard vs
// custom kernel performance on a real model.

#include "kernel-registry.h"  // for elementwise_kernel_fn typedef
#include "kernels/kernels-standard.h"
#include "kernels/kernels-custom.h"

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

// ggml-cpu internals needed for the hook
#include "ggml-cpu-impl.h"
#include "traits.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Op timing statistics
// ---------------------------------------------------------------------------
struct OpStats {
    std::string name;
    int64_t     count          = 0;
    double      total_time_us  = 0.0;
    int64_t     total_elements = 0;
};

static std::mutex g_stats_mutex;
static std::unordered_map<std::string, OpStats> g_op_stats;
static bool g_use_custom_kernels = false;

static void record_op(const char * name, double time_us, int64_t elements) {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    auto & s = g_op_stats[name];
    s.name = name;
    s.count++;
    s.total_time_us += time_us;
    s.total_elements += elements;
}

static void reset_stats() {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_op_stats.clear();
}

static double now_us() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Custom tensor_traits — intercepts element-wise ops during compute
// ---------------------------------------------------------------------------

// Kernel dispatch table entry
struct KernelDispatch {
    const char *          name;
    ggml_op               op;
    elementwise_kernel_fn standard_fn;
    elementwise_kernel_fn custom_fn;
    bool                  is_binary;  // needs src1
};

static const KernelDispatch g_dispatches[] = {
    {"add", GGML_OP_ADD, kernel_add_standard, kernel_add_custom, true},
    {"mul", GGML_OP_MUL, kernel_mul_standard, kernel_mul_custom, true},
};
static const int N_DISPATCHES = 2;

class custom_tensor_traits : public ggml::cpu::tensor_traits {
public:
    const KernelDispatch * dispatch;

    bool work_size(int /*n_threads*/, const struct ggml_tensor * /*op*/, size_t & /*size*/) override {
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * dst) override {
        if (!dispatch) return false;

        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];

        // Only intercept when all tensors are F32 and contiguous
        if (!src0 || src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            return false;
        }
        if (dispatch->is_binary && (!src1 || src1->type != GGML_TYPE_F32)) {
            return false;
        }
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            return false;
        }
        if (dispatch->is_binary && src1 && !ggml_is_contiguous(src1)) {
            return false;
        }
        // For binary ops with broadcasting, skip (our kernels don't handle it)
        if (dispatch->is_binary && src1 && !ggml_are_same_shape(src0, src1)) {
            return false;
        }

        const int64_t n = ggml_nelements(src0);
        const int ith = params->ith;
        const int nth = params->nth;

        const int64_t chunk = (n + nth - 1) / nth;
        const int64_t start = ith * chunk;
        const int64_t end   = std::min(start + chunk, n);
        if (start >= n) return true;

        const float * src0_data = (const float *)src0->data + start;
        const float * src1_data = (dispatch->is_binary && src1)
                                  ? (const float *)src1->data + start : nullptr;
        float * dst_data = (float *)dst->data + start;
        const int64_t count = end - start;

        elementwise_kernel_fn fn = g_use_custom_kernels ? dispatch->custom_fn : dispatch->standard_fn;

        double t0 = now_us();
        fn(dst_data, src0_data, src1_data, count);
        double t1 = now_us();

        if (ith == 0) {
            record_op(dispatch->name, t1 - t0, n);
        }

        return true;
    }
};

// ---------------------------------------------------------------------------
// Custom extra_buffer_type — returns tensor_traits for ops we want to intercept
// ---------------------------------------------------------------------------

// Static pool of traits objects
static custom_tensor_traits g_traits_pool[N_DISPATCHES];
static bool g_traits_initialized = false;

static void init_traits_pool() {
    if (g_traits_initialized) return;
    for (int i = 0; i < N_DISPATCHES; i++) {
        g_traits_pool[i].dispatch = &g_dispatches[i];
    }
    g_traits_initialized = true;
}

class custom_extra_buffer_type : public ggml::cpu::extra_buffer_type {
public:
    bool supports_op(ggml_backend_dev_t /*dev*/, const struct ggml_tensor * op) override {
        for (int i = 0; i < N_DISPATCHES; i++) {
            if (op->op == g_dispatches[i].op) return true;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        init_traits_pool();
        for (int i = 0; i < N_DISPATCHES; i++) {
            if (op->op == g_dispatches[i].op) {
                return &g_traits_pool[i];
            }
        }
        return nullptr;
    }
};

// ---------------------------------------------------------------------------
// Hook registration
// ---------------------------------------------------------------------------

static custom_extra_buffer_type g_custom_extra;

// Minimal buffer type iface so ggml doesn't crash when querying it
static const char * custom_buft_get_name(ggml_backend_buffer_type_t /*buft*/) {
    return "custom_kernel_hook";
}
static ggml_backend_buffer_t custom_buft_alloc(ggml_backend_buffer_type_t /*buft*/, size_t /*size*/) {
    return nullptr;
}
static size_t custom_buft_get_alignment(ggml_backend_buffer_type_t /*buft*/) {
    return 32;
}

static struct ggml_backend_buffer_type g_custom_buft = {
    /* .iface   = */ {
        /* .get_name       = */ custom_buft_get_name,
        /* .alloc_buffer   = */ custom_buft_alloc,
        /* .get_alignment  = */ custom_buft_get_alignment,
        /* .get_max_size   = */ nullptr,
        /* .get_alloc_size = */ nullptr,
        /* .is_host        = */ nullptr,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

static bool g_hook_registered = false;

static void register_custom_hook() {
    if (g_hook_registered) return;
    g_custom_buft.context = &g_custom_extra;
    auto & bufts = ggml_backend_cpu_get_extra_buffer_types();
    bufts.push_back(&g_custom_buft);
    g_hook_registered = true;
}

static void unregister_custom_hook() {
    if (!g_hook_registered) return;
    auto & bufts = ggml_backend_cpu_get_extra_buffer_types();
    bufts.erase(
        std::remove(bufts.begin(), bufts.end(), (ggml_backend_buffer_type_t)&g_custom_buft),
        bufts.end());
    g_hook_registered = false;
}

// ---------------------------------------------------------------------------
// Model inference helper
// ---------------------------------------------------------------------------

struct InferenceResult {
    double pp_time_ms  = 0;
    double tg_time_ms  = 0;
    double pp_tokens_s = 0;
    double tg_tokens_s = 0;
    int    pp_tokens   = 0;
    int    tg_tokens   = 0;
};

static InferenceResult run_inference(llama_model * model, int n_pp, int n_tg, int n_threads) {
    InferenceResult result;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = n_pp + n_tg;
    ctx_params.n_batch   = n_pp;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to create context\n");
        return result;
    }

    // Prompt processing (pp)
    llama_batch batch = llama_batch_init(n_pp, 0, 1);
    for (int i = 0; i < n_pp; i++) {
        common_batch_add(batch, i + 1, i, {0}, false); // dummy tokens
    }
    batch.logits[batch.n_tokens - 1] = true;

    double t0 = now_us();
    llama_decode(ctx, batch);
    double t1 = now_us();
    result.pp_time_ms  = (t1 - t0) / 1000.0;
    result.pp_tokens   = n_pp;
    result.pp_tokens_s = n_pp / (result.pp_time_ms / 1000.0);

    // Token generation (tg)
    llama_batch_free(batch);
    batch = llama_batch_init(1, 0, 1);

    t0 = now_us();
    for (int i = 0; i < n_tg; i++) {
        common_batch_clear(batch);
        common_batch_add(batch, 0, n_pp + i, {0}, true);
        llama_decode(ctx, batch);
    }
    t1 = now_us();
    result.tg_time_ms  = (t1 - t0) / 1000.0;
    result.tg_tokens   = n_tg;
    result.tg_tokens_s = n_tg / (result.tg_time_ms / 1000.0);

    llama_batch_free(batch);
    llama_free(ctx);
    return result;
}

// ---------------------------------------------------------------------------
// CLI & Main
// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    printf("Usage: %s -m <model.gguf> [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  -m <path>       Model file (required)\n");
    printf("  -pp <n>         Prompt tokens (default: 256)\n");
    printf("  -tg <n>         Generation tokens (default: 64)\n");
    printf("  -t <n>          Threads (default: 4)\n");
    printf("  -r <n>          Repetitions (default: 1)\n");
    printf("  -o <file>       JSON output file\n");
    printf("  -h              Show help\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string output_file;
    int n_pp   = 256;
    int n_tg   = 64;
    int n_threads = 4;
    int n_reps = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) { model_path = argv[++i]; }
        else if (strcmp(argv[i], "-pp") == 0 && i + 1 < argc) { n_pp = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-tg") == 0 && i + 1 < argc) { n_tg = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)  { n_threads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)  { n_reps = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)  { output_file = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { print_usage(argv[0]); return 0; }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Suppress llama logging
    llama_log_set([](enum ggml_log_level, const char *, void *){}, nullptr);

    // Load model (CPU only: ngl=0)
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    fprintf(stderr, "Loading model: %s\n", model_path.c_str());
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }
    fprintf(stderr, "Model loaded.\n\n");

    // Register hook AFTER model loading (avoids interfering with device init)
    register_custom_hook();

    // Warmup run
    fprintf(stderr, "Warmup run...\n");
    g_use_custom_kernels = false;
    reset_stats();
    run_inference(model, n_pp, n_tg, n_threads);

    // --- Run 1: Standard kernels ---
    fprintf(stderr, "\n=== Run 1: Standard kernels ===\n");
    std::unordered_map<std::string, OpStats> standard_stats;
    InferenceResult standard_result;

    for (int rep = 0; rep < n_reps; rep++) {
        g_use_custom_kernels = false;
        reset_stats();
        auto res = run_inference(model, n_pp, n_tg, n_threads);
        if (rep == 0) standard_result = res;
        else {
            standard_result.pp_tokens_s += res.pp_tokens_s;
            standard_result.tg_tokens_s += res.tg_tokens_s;
        }
        {
            std::lock_guard<std::mutex> lock(g_stats_mutex);
            for (auto & [name, stats] : g_op_stats) {
                standard_stats[name].name = name;
                standard_stats[name].count += stats.count;
                standard_stats[name].total_time_us += stats.total_time_us;
                standard_stats[name].total_elements += stats.total_elements;
            }
        }
    }
    standard_result.pp_tokens_s /= n_reps;
    standard_result.tg_tokens_s /= n_reps;

    fprintf(stderr, "  pp: %.1f t/s | tg: %.1f t/s\n",
            standard_result.pp_tokens_s, standard_result.tg_tokens_s);

    // --- Run 2: Custom kernels ---
    fprintf(stderr, "\n=== Run 2: Custom kernels ===\n");
    std::unordered_map<std::string, OpStats> custom_stats;
    InferenceResult custom_result;

    for (int rep = 0; rep < n_reps; rep++) {
        g_use_custom_kernels = true;
        reset_stats();
        auto res = run_inference(model, n_pp, n_tg, n_threads);
        if (rep == 0) custom_result = res;
        else {
            custom_result.pp_tokens_s += res.pp_tokens_s;
            custom_result.tg_tokens_s += res.tg_tokens_s;
        }
        {
            std::lock_guard<std::mutex> lock(g_stats_mutex);
            for (auto & [name, stats] : g_op_stats) {
                custom_stats[name].name = name;
                custom_stats[name].count += stats.count;
                custom_stats[name].total_time_us += stats.total_time_us;
                custom_stats[name].total_elements += stats.total_elements;
            }
        }
    }
    custom_result.pp_tokens_s /= n_reps;
    custom_result.tg_tokens_s /= n_reps;

    fprintf(stderr, "  pp: %.1f t/s | tg: %.1f t/s\n",
            custom_result.pp_tokens_s, custom_result.tg_tokens_s);

    // --- Results ---
    fprintf(stderr, "\n============================================\n");
    fprintf(stderr, "  INFERENCE COMPARISON (pp=%d, tg=%d, threads=%d)\n", n_pp, n_tg, n_threads);
    fprintf(stderr, "============================================\n\n");

    fprintf(stderr, "%-12s %14s %14s %10s\n", "", "Standard", "Custom", "Speedup");
    fprintf(stderr, "%-12s %14s %14s %10s\n", "---", "---", "---", "---");
    fprintf(stderr, "%-12s %11.1f t/s %11.1f t/s %9.2fx\n", "pp",
            standard_result.pp_tokens_s, custom_result.pp_tokens_s,
            custom_result.pp_tokens_s / standard_result.pp_tokens_s);
    fprintf(stderr, "%-12s %11.1f t/s %11.1f t/s %9.2fx\n", "tg",
            standard_result.tg_tokens_s, custom_result.tg_tokens_s,
            custom_result.tg_tokens_s / standard_result.tg_tokens_s);

    fprintf(stderr, "\n--- Per-Op Kernel Timing (hooked ops only) ---\n");
    fprintf(stderr, "%-12s %8s %14s %14s %10s\n",
            "Op", "Calls", "Std Total(us)", "Cust Total(us)", "Speedup");
    fprintf(stderr, "%-12s %8s %14s %14s %10s\n",
            "---", "---", "---", "---", "---");

    for (auto & [name, ss] : standard_stats) {
        auto it = custom_stats.find(name);
        if (it != custom_stats.end()) {
            auto & cs = it->second;
            double speedup = (cs.total_time_us > 0) ? ss.total_time_us / cs.total_time_us : 0;
            fprintf(stderr, "%-12s %8ld %14.1f %14.1f %9.2fx\n",
                    name.c_str(), (long)ss.count,
                    ss.total_time_us, cs.total_time_us, speedup);
        }
    }

    // --- JSON output ---
    if (!output_file.empty()) {
        std::ostringstream json;
        json << std::fixed << std::setprecision(3);
        json << "{\n";
        json << "  \"model\": \"" << model_path << "\",\n";
        json << "  \"config\": { \"pp\": " << n_pp << ", \"tg\": " << n_tg
             << ", \"threads\": " << n_threads << ", \"reps\": " << n_reps << " },\n";

        json << "  \"inference\": {\n";
        json << "    \"standard\": { \"pp_tokens_s\": " << standard_result.pp_tokens_s
             << ", \"tg_tokens_s\": " << standard_result.tg_tokens_s << " },\n";
        json << "    \"custom\": { \"pp_tokens_s\": " << custom_result.pp_tokens_s
             << ", \"tg_tokens_s\": " << custom_result.tg_tokens_s << " },\n";
        json << "    \"speedup\": { \"pp\": "
             << custom_result.pp_tokens_s / standard_result.pp_tokens_s
             << ", \"tg\": "
             << custom_result.tg_tokens_s / standard_result.tg_tokens_s << " }\n";
        json << "  },\n";

        json << "  \"per_op\": [\n";
        bool first = true;
        for (auto & [name, ss] : standard_stats) {
            auto it = custom_stats.find(name);
            if (it == custom_stats.end()) continue;
            auto & cs = it->second;
            if (!first) json << ",\n";
            first = false;
            json << "    { \"op\": \"" << name << "\""
                 << ", \"calls\": " << ss.count
                 << ", \"std_total_us\": " << ss.total_time_us
                 << ", \"custom_total_us\": " << cs.total_time_us
                 << ", \"speedup\": " << (cs.total_time_us > 0 ? ss.total_time_us / cs.total_time_us : 0)
                 << " }";
        }
        json << "\n  ]\n";
        json << "}\n";

        std::ofstream ofs(output_file);
        ofs << json.str();
        fprintf(stderr, "\nJSON written to %s\n", output_file.c_str());
    }

    unregister_custom_hook();
    llama_model_free(model);
    return 0;
}
