#include "bench-harness.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static double now_us() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
}

BenchHarness::BenchHarness(const BenchConfig & config) : config_(config) {}

std::vector<float> BenchHarness::make_random(int64_t n, float lo, float hi) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; i++) {
        v[i] = dist(rng);
    }
    return v;
}

void BenchHarness::check_correctness(const float * expected, const float * actual,
                                      int64_t n, BenchResult & result) {
    result.correctness_ok = true;
    result.max_abs_error  = 0.0;
    result.max_rel_error  = 0.0;

    for (int64_t i = 0; i < n; i++) {
        double ae = std::fabs((double)actual[i] - (double)expected[i]);
        double re = 0.0;
        if (std::fabs((double)expected[i]) > 1e-10) {
            re = ae / std::fabs((double)expected[i]);
        }
        if (ae > result.max_abs_error) result.max_abs_error = ae;
        if (re > result.max_rel_error) result.max_rel_error = re;
    }

    if (result.max_abs_error > config_.max_abs_eps && result.max_rel_error > config_.max_rel_eps) {
        result.correctness_ok = false;
    }
}

BenchResult BenchHarness::bench_elementwise(const KernelEntry & kernel,
                                             const KernelEntry & reference,
                                             int64_t n_elements) {
    BenchResult res;
    res.op_name    = kernel.name;
    res.variant    = kernel.variant;
    res.n_elements = n_elements;
    res.size_label = std::to_string(n_elements);

    bool is_binary = (kernel.op_type == KernelOpType::BINARY);

    auto src0 = make_random(n_elements);
    std::vector<float> src1;
    if (is_binary) {
        src1 = make_random(n_elements);
    }
    const float * src1_ptr = is_binary ? src1.data() : nullptr;

    // Correctness check: run reference then kernel, compare outputs
    std::vector<float> ref_out(n_elements);
    std::vector<float> kern_out(n_elements);

    reference.ew_fn(ref_out.data(), src0.data(), src1_ptr, n_elements);
    kernel.ew_fn(kern_out.data(), src0.data(), src1_ptr, n_elements);
    check_correctness(ref_out.data(), kern_out.data(), n_elements, res);

    // Warmup
    for (int w = 0; w < config_.warmup_runs; w++) {
        kernel.ew_fn(kern_out.data(), src0.data(), src1_ptr, n_elements);
    }

    // Timing loop: run until at least min_time_sec elapsed
    std::vector<double> times;
    double total_time_us = 0.0;
    double min_time_us_target = config_.min_time_sec * 1e6;

    while (total_time_us < min_time_us_target || (int)times.size() < 10) {
        double t0 = now_us();
        kernel.ew_fn(kern_out.data(), src0.data(), src1_ptr, n_elements);
        double t1 = now_us();
        double dt = t1 - t0;
        times.push_back(dt);
        total_time_us += dt;

        // Safety: cap at 10000 runs
        if ((int)times.size() >= 10000) break;
    }

    // Statistics
    res.n_runs = (int)times.size();
    res.min_time_us = *std::min_element(times.begin(), times.end());
    res.max_time_us = *std::max_element(times.begin(), times.end());
    res.avg_time_us = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    double var = 0.0;
    for (double t : times) {
        double d = t - res.avg_time_us;
        var += d * d;
    }
    res.stddev_time_us = std::sqrt(var / times.size());

    // Bandwidth: for unary, read src0 + write dst = 2*n*4 bytes
    //            for binary, read src0 + src1 + write dst = 3*n*4 bytes
    int64_t bytes_moved = n_elements * sizeof(float) * (is_binary ? 3 : 2);
    res.bandwidth_gb_s = (bytes_moved / 1e9) / (res.avg_time_us / 1e6);
    res.gflops = 0.0; // element-wise ops are memory-bound

    return res;
}

BenchResult BenchHarness::bench_mul_mat(const KernelEntry & kernel,
                                         const KernelEntry & reference,
                                         int64_t M, int64_t N, int64_t K) {
    BenchResult res;
    res.op_name    = kernel.name;
    res.variant    = kernel.variant;
    res.n_elements = M * N;
    res.size_label = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);

    auto A = make_random(M * K);
    auto B = make_random(N * K);
    std::vector<float> ref_out(M * N);
    std::vector<float> kern_out(M * N);

    // Correctness check
    reference.mm_fn(ref_out.data(), A.data(), B.data(), M, N, K);
    kernel.mm_fn(kern_out.data(), A.data(), B.data(), M, N, K);

    // MUL_MAT needs looser tolerance due to accumulation order
    double saved_abs = config_.max_abs_eps;
    double saved_rel = config_.max_rel_eps;
    config_.max_abs_eps = 1e-3;
    config_.max_rel_eps = 1e-3;
    check_correctness(ref_out.data(), kern_out.data(), M * N, res);
    config_.max_abs_eps = saved_abs;
    config_.max_rel_eps = saved_rel;

    // Warmup
    for (int w = 0; w < config_.warmup_runs; w++) {
        kernel.mm_fn(kern_out.data(), A.data(), B.data(), M, N, K);
    }

    // Timing loop
    std::vector<double> times;
    double total_time_us = 0.0;
    double min_time_us_target = config_.min_time_sec * 1e6;

    while (total_time_us < min_time_us_target || (int)times.size() < 10) {
        double t0 = now_us();
        kernel.mm_fn(kern_out.data(), A.data(), B.data(), M, N, K);
        double t1 = now_us();
        double dt = t1 - t0;
        times.push_back(dt);
        total_time_us += dt;

        if ((int)times.size() >= 10000) break;
    }

    // Statistics
    res.n_runs = (int)times.size();
    res.min_time_us = *std::min_element(times.begin(), times.end());
    res.max_time_us = *std::max_element(times.begin(), times.end());
    res.avg_time_us = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    double var = 0.0;
    for (double t : times) {
        double d = t - res.avg_time_us;
        var += d * d;
    }
    res.stddev_time_us = std::sqrt(var / times.size());

    res.bandwidth_gb_s = 0.0;
    // GFLOPS: 2*M*N*K FLOPs for GEMM
    double flops = 2.0 * M * N * K;
    res.gflops = (flops / 1e9) / (res.avg_time_us / 1e6);

    return res;
}
