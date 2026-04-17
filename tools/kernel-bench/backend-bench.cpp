// backend-bench.cpp — Compare ops across multiple ggml backends.
//
// Runs the same computation graph on the standard CPU backend and on our
// sample custom backend, reporting per-op timing and correctness. The custom
// backend wraps the CPU backend internally, so initial results should be
// identical — swap in optimized kernels via kernels-custom.cpp to observe
// real differences.

#include "custom-backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
static double now_us() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
}

static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    gmtime_r(&t, &tm_buf);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

// ---------------------------------------------------------------------------
// Op definitions
// ---------------------------------------------------------------------------
enum class OpKind { ADD, MUL, MUL_MAT, UNARY_RELU, UNARY_SIGMOID, UNARY_SILU };

struct OpSpec {
    std::string name;
    OpKind      kind;
    // Element-wise: sizes[0] = n_elements. MUL_MAT: sizes = {M, N, K}
    std::vector<int64_t> sizes;
};

struct BenchResult {
    std::string backend;
    std::string op;
    std::string size_label;
    int64_t     n_elements  = 0;
    int         n_runs      = 0;
    double      avg_time_us = 0.0;
    double      min_time_us = 0.0;
    double      max_time_us = 0.0;
    double      bandwidth_gb_s = 0.0;
    double      gflops      = 0.0;
    bool        correct     = true;
    double      max_abs_err = 0.0;
};

// ---------------------------------------------------------------------------
// Build a ggml graph with a single op
// ---------------------------------------------------------------------------
static ggml_tensor * build_op(ggml_context * ctx, const OpSpec & spec) {
    switch (spec.kind) {
        case OpKind::ADD:
        case OpKind::MUL: {
            int64_t n = spec.sizes[0];
            auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            auto * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            return (spec.kind == OpKind::ADD) ? ggml_add(ctx, a, b) : ggml_mul(ctx, a, b);
        }
        case OpKind::MUL_MAT: {
            int64_t M = spec.sizes[0], N = spec.sizes[1], K = spec.sizes[2];
            auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); // [K, M] ggml convention
            auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N); // [K, N]
            return ggml_mul_mat(ctx, a, b); // -> [N, M] in ggml
        }
        case OpKind::UNARY_RELU: {
            int64_t n = spec.sizes[0];
            auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            return ggml_relu(ctx, a);
        }
        case OpKind::UNARY_SIGMOID: {
            int64_t n = spec.sizes[0];
            auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            return ggml_sigmoid(ctx, a);
        }
        case OpKind::UNARY_SILU: {
            int64_t n = spec.sizes[0];
            auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            return ggml_silu(ctx, a);
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Fill tensors deterministically so backends see the same data
// ---------------------------------------------------------------------------
static void fill_tensor_random(ggml_tensor * t, uint32_t seed) {
    size_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) buf[i] = dist(rng);
    ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(float));
}

// ---------------------------------------------------------------------------
// Run a single op on a given backend
// ---------------------------------------------------------------------------
static BenchResult run_op_on_backend(
        ggml_backend_t backend, const std::string & backend_name,
        const OpSpec & spec, double min_time_sec, int warmup,
        std::vector<float> & output_buf) {
    BenchResult res;
    res.backend = backend_name;
    res.op      = spec.name;

    // Build size label
    if (spec.kind == OpKind::MUL_MAT) {
        res.size_label = std::to_string(spec.sizes[0]) + "x" +
                         std::to_string(spec.sizes[1]) + "x" +
                         std::to_string(spec.sizes[2]);
    } else {
        res.size_label = std::to_string(spec.sizes[0]);
    }

    // Context + graph
    ggml_init_params ip = {
        /* mem_size   = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /* mem_buffer = */ nullptr,
        /* no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(ip));

    ggml_tensor * out = build_op(ctx.get(), spec);
    if (!out) { res.correct = false; return res; }

    // Allocate tensors on this backend
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) { res.correct = false; return res; }

    // Fill inputs with deterministic data (same seed across backends)
    if (out->src[0]) fill_tensor_random(out->src[0], 42);
    if (out->src[1]) fill_tensor_random(out->src[1], 123);

    // Build graph
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    // Timing loop
    std::vector<double> times;
    double total = 0.0;
    while (total < min_time_sec * 1e6 && (int)times.size() < 10000) {
        double t0 = now_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        double t1 = now_us();
        double dt = t1 - t0;
        times.push_back(dt);
        total += dt;
        if ((int)times.size() < 10) continue;  // ensure at least 10 runs
    }

    res.n_runs = (int)times.size();
    res.min_time_us = *std::min_element(times.begin(), times.end());
    res.max_time_us = *std::max_element(times.begin(), times.end());
    double sum = 0;
    for (double t : times) sum += t;
    res.avg_time_us = sum / times.size();

    // Metrics
    if (spec.kind == OpKind::MUL_MAT) {
        double M = spec.sizes[0], N = spec.sizes[1], K = spec.sizes[2];
        res.gflops = (2.0 * M * N * K / 1e9) / (res.avg_time_us / 1e6);
        res.n_elements = (int64_t)(M * N);
    } else {
        res.n_elements = spec.sizes[0];
        int bytes_moved = (spec.kind == OpKind::ADD || spec.kind == OpKind::MUL) ? 12 : 8;
        res.bandwidth_gb_s = (spec.sizes[0] * bytes_moved / 1e9) / (res.avg_time_us / 1e6);
    }

    // Copy output for correctness check
    size_t n_out = ggml_nelements(out);
    output_buf.assign(n_out, 0.0f);
    ggml_backend_tensor_get(out, output_buf.data(), 0, n_out * sizeof(float));

    return res;
}

// ---------------------------------------------------------------------------
// Compare two outputs
// ---------------------------------------------------------------------------
static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return 1e9;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------
static std::string result_to_json(const BenchResult & r, const std::string & indent) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << indent << "{\n";
    ss << indent << "  \"backend\": \"" << r.backend << "\",\n";
    ss << indent << "  \"op\": \"" << r.op << "\",\n";
    ss << indent << "  \"size\": \"" << r.size_label << "\",\n";
    ss << indent << "  \"n_elements\": " << r.n_elements << ",\n";
    ss << indent << "  \"n_runs\": " << r.n_runs << ",\n";
    ss << indent << "  \"avg_time_us\": " << r.avg_time_us << ",\n";
    ss << indent << "  \"min_time_us\": " << r.min_time_us << ",\n";
    ss << indent << "  \"max_time_us\": " << r.max_time_us << ",\n";
    ss << indent << "  \"bandwidth_gb_s\": " << r.bandwidth_gb_s << ",\n";
    ss << indent << "  \"gflops\": " << r.gflops << ",\n";
    ss << indent << "  \"correct\": " << (r.correct ? "true" : "false") << ",\n";
    ss << indent << "  \"max_abs_err\": " << std::setprecision(8) << r.max_abs_err << "\n";
    ss << indent << "}";
    return ss.str();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::string output_file;
    double min_time_sec = 0.5;
    int warmup = 3;
    std::string op_filter;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output_file = argv[++i];
        else if (strcmp(argv[i], "--min-time") == 0 && i + 1 < argc) min_time_sec = atof(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--op") == 0 && i + 1 < argc) op_filter = argv[++i];
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  -o <file>         JSON output\n");
            printf("  --op <name>       Filter by op name\n");
            printf("  --min-time <sec>  Minimum benchmark time per op (default: 0.5)\n");
            printf("  --warmup <n>      Warmup iterations (default: 3)\n");
            return 0;
        }
    }

    // Op benchmark specs
    std::vector<OpSpec> ops = {
        {"add",     OpKind::ADD,           {4096}},
        {"add",     OpKind::ADD,           {65536}},
        {"add",     OpKind::ADD,           {1048576}},
        {"mul",     OpKind::MUL,           {4096}},
        {"mul",     OpKind::MUL,           {65536}},
        {"mul",     OpKind::MUL,           {1048576}},
        {"relu",    OpKind::UNARY_RELU,    {65536}},
        {"relu",    OpKind::UNARY_RELU,    {1048576}},
        {"sigmoid", OpKind::UNARY_SIGMOID, {65536}},
        {"sigmoid", OpKind::UNARY_SIGMOID, {1048576}},
        {"silu",    OpKind::UNARY_SILU,    {65536}},
        {"silu",    OpKind::UNARY_SILU,    {1048576}},
        {"mul_mat", OpKind::MUL_MAT,       {64,  64,  64}},
        {"mul_mat", OpKind::MUL_MAT,       {256, 256, 256}},
        {"mul_mat", OpKind::MUL_MAT,       {512, 512, 512}},
    };

    // Initialize backends
    fprintf(stderr, "Initializing backends...\n");
    ggml_backend_t backend_cpu    = ggml_backend_cpu_init();
    ggml_backend_t backend_custom = ggml_backend_custom_init("Custom-CPU", true);
    if (!backend_cpu || !backend_custom) {
        fprintf(stderr, "ERROR: backend init failed\n");
        return 1;
    }

    struct BackendEntry { ggml_backend_t backend; std::string name; };
    std::vector<BackendEntry> backends = {
        {backend_cpu,    "CPU"},
        {backend_custom, "Custom-CPU"},
    };

    fprintf(stderr, "Backends: CPU, Custom-CPU\n\n");

    // Run all ops on all backends, comparing outputs for correctness
    std::vector<BenchResult> all_results;

    for (const auto & spec : ops) {
        if (!op_filter.empty() && spec.name != op_filter) continue;

        std::string size_str = (spec.kind == OpKind::MUL_MAT)
            ? (std::to_string(spec.sizes[0]) + "x" + std::to_string(spec.sizes[1]) +
               "x" + std::to_string(spec.sizes[2]))
            : std::to_string(spec.sizes[0]);

        fprintf(stderr, "[%-8s %-18s]", spec.name.c_str(), size_str.c_str());

        std::vector<float> reference_out;
        for (size_t bi = 0; bi < backends.size(); bi++) {
            std::vector<float> out;
            auto res = run_op_on_backend(backends[bi].backend, backends[bi].name,
                                         spec, min_time_sec, warmup, out);

            if (bi == 0) {
                reference_out = out;
            } else {
                res.max_abs_err = max_abs_diff(reference_out, out);
                res.correct = res.max_abs_err < 1e-4;
            }

            fprintf(stderr, " %s:%.1fus", backends[bi].name.c_str(), res.avg_time_us);
            all_results.push_back(res);
        }
        fprintf(stderr, "\n");
    }

    // Summary table
    fprintf(stderr, "\n=== Backend Comparison ===\n");
    fprintf(stderr, "%-10s %-18s %-12s %12s %12s %8s %s\n",
            "Op", "Size", "Backend", "Avg(us)", "Min(us)", "Runs", "Match");
    for (const auto & r : all_results) {
        fprintf(stderr, "%-10s %-18s %-12s %12.1f %12.1f %8d %s\n",
                r.op.c_str(), r.size_label.c_str(), r.backend.c_str(),
                r.avg_time_us, r.min_time_us, r.n_runs,
                r.correct ? "OK" : "FAIL");
    }

    // Speedup: custom vs CPU baseline
    fprintf(stderr, "\n=== Speedup (Custom-CPU / CPU) ===\n");
    fprintf(stderr, "%-10s %-18s %12s %12s %8s\n", "Op", "Size", "CPU(us)", "Custom(us)", "Speedup");
    for (size_t i = 0; i + 1 < all_results.size(); i += 2) {
        const auto & cpu = all_results[i];
        const auto & cus = all_results[i + 1];
        if (cpu.op != cus.op || cpu.size_label != cus.size_label) continue;
        double speedup = (cus.avg_time_us > 0) ? cpu.avg_time_us / cus.avg_time_us : 0;
        fprintf(stderr, "%-10s %-18s %12.1f %12.1f %7.2fx\n",
                cpu.op.c_str(), cpu.size_label.c_str(),
                cpu.avg_time_us, cus.avg_time_us, speedup);
    }

    // JSON output
    if (!output_file.empty()) {
        std::ostringstream json;
        json << std::fixed << std::setprecision(3);
        json << "{\n";
        json << "  \"framework_version\": \"1.0.0\",\n";
        json << "  \"timestamp\": \"" << get_timestamp() << "\",\n";
        json << "  \"backends\": [\"CPU\", \"Custom-CPU\"],\n";
        json << "  \"config\": { \"min_time_sec\": " << min_time_sec
             << ", \"warmup\": " << warmup << " },\n";

        json << "  \"results\": [\n";
        for (size_t i = 0; i < all_results.size(); i++) {
            json << result_to_json(all_results[i], "    ");
            if (i + 1 < all_results.size()) json << ",";
            json << "\n";
        }
        json << "  ],\n";

        // Comparisons
        json << "  \"comparisons\": [\n";
        bool first = true;
        for (size_t i = 0; i + 1 < all_results.size(); i += 2) {
            const auto & cpu = all_results[i];
            const auto & cus = all_results[i + 1];
            if (cpu.op != cus.op || cpu.size_label != cus.size_label) continue;
            double speedup = (cus.avg_time_us > 0) ? cpu.avg_time_us / cus.avg_time_us : 0;
            if (!first) json << ",\n";
            first = false;
            json << "    {\n";
            json << "      \"op\": \"" << cpu.op << "\",\n";
            json << "      \"size\": \"" << cpu.size_label << "\",\n";
            json << "      \"baseline_backend\": \"" << cpu.backend << "\",\n";
            json << "      \"contender_backend\": \"" << cus.backend << "\",\n";
            json << "      \"baseline_avg_us\": " << cpu.avg_time_us << ",\n";
            json << "      \"contender_avg_us\": " << cus.avg_time_us << ",\n";
            json << "      \"speedup\": " << speedup << ",\n";
            json << "      \"correctness\": " << (cus.correct ? "true" : "false") << ",\n";
            json << "      \"max_abs_err\": " << std::setprecision(8) << cus.max_abs_err << "\n";
            json << "    }";
        }
        json << "\n  ]\n";
        json << "}\n";

        std::ofstream ofs(output_file);
        ofs << json.str();
        fprintf(stderr, "\nJSON written to %s\n", output_file.c_str());
    }

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_custom);
    return 0;
}
