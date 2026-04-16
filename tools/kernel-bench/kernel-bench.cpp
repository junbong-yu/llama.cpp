#include "kernel-registry.h"
#include "bench-harness.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// Default benchmark sizes
static const std::vector<int64_t> DEFAULT_EW_SIZES = {256, 4096, 65536, 1048576, 16777216};

struct MatSize { int64_t M, N, K; };
static const std::vector<MatSize> DEFAULT_MM_SIZES = {
    {32, 32, 32},
    {128, 128, 128},
    {512, 512, 512},
};

// --- JSON helpers (simple manual serialization) ---

static std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"') { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else { out += c; }
    }
    return out;
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

static std::string get_os_info() {
#ifdef __APPLE__
    return "macOS (Darwin)";
#elif defined(__linux__)
    return "Linux";
#elif defined(_WIN32)
    return "Windows";
#else
    return "Unknown";
#endif
}

static std::string get_arch_info() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

static std::string result_to_json(const BenchResult & r, const std::string & indent) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << indent << "{\n";
    ss << indent << "  \"op\": \"" << json_escape(r.op_name) << "\",\n";
    ss << indent << "  \"variant\": \"" << json_escape(r.variant) << "\",\n";
    ss << indent << "  \"size\": \"" << json_escape(r.size_label) << "\",\n";
    ss << indent << "  \"n_elements\": " << r.n_elements << ",\n";
    ss << indent << "  \"n_runs\": " << r.n_runs << ",\n";
    ss << indent << "  \"avg_time_us\": " << r.avg_time_us << ",\n";
    ss << indent << "  \"min_time_us\": " << r.min_time_us << ",\n";
    ss << indent << "  \"max_time_us\": " << r.max_time_us << ",\n";
    ss << indent << "  \"stddev_time_us\": " << r.stddev_time_us << ",\n";
    ss << indent << "  \"bandwidth_gb_s\": " << r.bandwidth_gb_s << ",\n";
    ss << indent << "  \"gflops\": " << r.gflops << ",\n";
    ss << indent << "  \"correctness\": " << (r.correctness_ok ? "true" : "false") << ",\n";
    ss << indent << "  \"max_abs_error\": " << std::setprecision(8) << r.max_abs_error << ",\n";
    ss << indent << "  \"max_rel_error\": " << r.max_rel_error << "\n";
    ss << indent << "}";
    return ss.str();
}

// --- CLI ---

struct CliArgs {
    std::string op_filter;        // empty = all
    std::string variant_filter;   // empty = all
    std::string output_file;      // empty = stdout
    double      min_time_sec = 1.0;
    int         warmup       = 3;
    bool        help         = false;
};

static void print_usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --op <name>        Benchmark specific op (add, mul, relu, sigmoid, silu, mul_mat)\n");
    printf("  --variant <name>   Benchmark specific variant (standard, custom)\n");
    printf("  --output <file>    Write JSON to file (default: stdout)\n");
    printf("  --min-time <sec>   Minimum benchmark time per kernel (default: 1.0)\n");
    printf("  --warmup <n>       Warmup iterations (default: 3)\n");
    printf("  --help             Show this help\n");
}

static CliArgs parse_args(int argc, char ** argv) {
    CliArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--op") == 0 && i + 1 < argc) {
            args.op_filter = argv[++i];
        } else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            args.variant_filter = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args.output_file = argv[++i];
        } else if (strcmp(argv[i], "--min-time") == 0 && i + 1 < argc) {
            args.min_time_sec = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            args.warmup = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            args.help = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            args.help = true;
        }
    }
    return args;
}

// --- Main ---

int main(int argc, char ** argv) {
    CliArgs args = parse_args(argc, argv);
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    KernelRegistry registry = KernelRegistry::create_default();

    BenchConfig config;
    config.warmup_runs  = args.warmup;
    config.min_time_sec = args.min_time_sec;
    BenchHarness harness(config);

    std::vector<BenchResult> all_results;

    // Progress output to stderr (so stdout JSON is clean)
    auto progress = [](const char * fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
    };

    for (const auto & op_name : registry.op_names()) {
        if (!args.op_filter.empty() && args.op_filter != op_name) {
            continue;
        }

        const KernelEntry * ref = registry.get(op_name, "standard");
        if (!ref) {
            progress("WARNING: no standard kernel for '%s', skipping\n", op_name.c_str());
            continue;
        }

        auto variants = registry.get_variants(op_name);

        for (const auto * kern : variants) {
            if (!args.variant_filter.empty() && args.variant_filter != kern->variant) {
                continue;
            }

            if (kern->op_type == KernelOpType::MUL_MAT) {
                for (const auto & sz : DEFAULT_MM_SIZES) {
                    progress("  %-10s %-10s %ldx%ldx%ld ...", op_name.c_str(),
                             kern->variant.c_str(), (long)sz.M, (long)sz.N, (long)sz.K);
                    auto result = harness.bench_mul_mat(*kern, *ref, sz.M, sz.N, sz.K);
                    progress(" %.1f us (%d runs) %s\n", result.avg_time_us, result.n_runs,
                             result.correctness_ok ? "OK" : "FAIL");
                    all_results.push_back(result);
                }
            } else {
                for (int64_t n : DEFAULT_EW_SIZES) {
                    progress("  %-10s %-10s n=%-10ld ...", op_name.c_str(),
                             kern->variant.c_str(), (long)n);
                    auto result = harness.bench_elementwise(*kern, *ref, n);
                    progress(" %.1f us (%d runs) %s\n", result.avg_time_us, result.n_runs,
                             result.correctness_ok ? "OK" : "FAIL");
                    all_results.push_back(result);
                }
            }
        }
    }

    // Build comparisons: for each (op, size) pair, compare custom vs standard
    struct Comparison {
        std::string op, size, baseline, contender;
        double speedup, baseline_avg_us, contender_avg_us;
    };
    std::vector<Comparison> comparisons;

    // Index results by (op, size, variant)
    for (const auto & r : all_results) {
        if (r.variant == "standard") continue;
        // Find corresponding standard result
        for (const auto & s : all_results) {
            if (s.op_name == r.op_name && s.size_label == r.size_label && s.variant == "standard") {
                Comparison cmp;
                cmp.op             = r.op_name;
                cmp.size           = r.size_label;
                cmp.baseline       = "standard";
                cmp.contender      = r.variant;
                cmp.baseline_avg_us   = s.avg_time_us;
                cmp.contender_avg_us  = r.avg_time_us;
                cmp.speedup        = (r.avg_time_us > 0) ? s.avg_time_us / r.avg_time_us : 0.0;
                comparisons.push_back(cmp);
                break;
            }
        }
    }

    // Serialize to JSON
    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{\n";
    json << "  \"framework_version\": \"1.0.0\",\n";
    json << "  \"timestamp\": \"" << get_timestamp() << "\",\n";
    json << "  \"system_info\": {\n";
    json << "    \"os\": \"" << get_os_info() << "\",\n";
    json << "    \"arch\": \"" << get_arch_info() << "\"\n";
    json << "  },\n";
    json << "  \"config\": {\n";
    json << "    \"warmup_runs\": " << config.warmup_runs << ",\n";
    json << "    \"min_time_sec\": " << config.min_time_sec << "\n";
    json << "  },\n";

    // Results array
    json << "  \"results\": [\n";
    for (size_t i = 0; i < all_results.size(); i++) {
        json << result_to_json(all_results[i], "    ");
        if (i + 1 < all_results.size()) json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // Comparisons array
    json << "  \"comparisons\": [\n";
    for (size_t i = 0; i < comparisons.size(); i++) {
        const auto & c = comparisons[i];
        json << "    {\n";
        json << "      \"op\": \"" << json_escape(c.op) << "\",\n";
        json << "      \"size\": \"" << json_escape(c.size) << "\",\n";
        json << "      \"baseline\": \"" << json_escape(c.baseline) << "\",\n";
        json << "      \"contender\": \"" << json_escape(c.contender) << "\",\n";
        json << "      \"speedup\": " << c.speedup << ",\n";
        json << "      \"baseline_avg_us\": " << c.baseline_avg_us << ",\n";
        json << "      \"contender_avg_us\": " << c.contender_avg_us << "\n";
        json << "    }";
        if (i + 1 < comparisons.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";

    // Output
    std::string json_str = json.str();
    if (args.output_file.empty()) {
        std::cout << json_str;
    } else {
        std::ofstream ofs(args.output_file);
        if (!ofs) {
            fprintf(stderr, "ERROR: cannot open output file: %s\n", args.output_file.c_str());
            return 1;
        }
        ofs << json_str;
        progress("Results written to %s\n", args.output_file.c_str());
    }

    // Print summary table to stderr
    progress("\n=== Summary ===\n");
    progress("%-10s %-15s %-10s %12s %12s %8s %s\n",
             "Op", "Size", "Variant", "Avg(us)", "Min(us)", "Runs", "Correct");
    progress("%-10s %-15s %-10s %12s %12s %8s %s\n",
             "---", "---", "---", "---", "---", "---", "---");
    for (const auto & r : all_results) {
        progress("%-10s %-15s %-10s %12.1f %12.1f %8d %s\n",
                 r.op_name.c_str(), r.size_label.c_str(), r.variant.c_str(),
                 r.avg_time_us, r.min_time_us, r.n_runs,
                 r.correctness_ok ? "OK" : "FAIL");
    }

    if (!comparisons.empty()) {
        progress("\n=== Speedup (custom vs standard) ===\n");
        progress("%-10s %-15s %12s %12s %8s\n", "Op", "Size", "Std(us)", "Custom(us)", "Speedup");
        progress("%-10s %-15s %12s %12s %8s\n", "---", "---", "---", "---", "---");
        for (const auto & c : comparisons) {
            progress("%-10s %-15s %12.1f %12.1f %7.2fx\n",
                     c.op.c_str(), c.size.c_str(),
                     c.baseline_avg_us, c.contender_avg_us, c.speedup);
        }
    }

    return 0;
}
