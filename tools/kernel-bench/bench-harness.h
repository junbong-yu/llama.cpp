#pragma once

#include "kernel-registry.h"
#include <cstdint>
#include <string>
#include <vector>

struct BenchConfig {
    int    warmup_runs  = 3;
    double min_time_sec = 1.0;
    double max_abs_eps  = 1e-5;
    double max_rel_eps  = 1e-4;
};

struct BenchResult {
    std::string op_name;
    std::string variant;
    std::string size_label;
    int64_t     n_elements   = 0;
    int         n_runs       = 0;
    double      avg_time_us  = 0.0;
    double      min_time_us  = 0.0;
    double      max_time_us  = 0.0;
    double      stddev_time_us = 0.0;
    double      bandwidth_gb_s = 0.0;
    double      gflops       = 0.0;
    bool        correctness_ok = true;
    double      max_abs_error = 0.0;
    double      max_rel_error = 0.0;
};

class BenchHarness {
public:
    explicit BenchHarness(const BenchConfig & config);

    // Benchmark an element-wise kernel (unary or binary) at a given size.
    // 'reference' is the standard kernel used for correctness checking.
    BenchResult bench_elementwise(const KernelEntry & kernel,
                                  const KernelEntry & reference,
                                  int64_t n_elements);

    // Benchmark a MUL_MAT kernel.
    BenchResult bench_mul_mat(const KernelEntry & kernel,
                              const KernelEntry & reference,
                              int64_t M, int64_t N, int64_t K);

private:
    BenchConfig config_;

    std::vector<float> make_random(int64_t n, float lo = -1.0f, float hi = 1.0f);

    void check_correctness(const float * expected, const float * actual,
                           int64_t n, BenchResult & result);
};
