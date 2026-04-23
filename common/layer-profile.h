#pragma once

// Per-layer inference profiler for llama.cpp.
//
// Installs a ggml_backend_sched_eval_callback that:
//   1. Measures wall-clock time taken by each graph node and attributes the
//      time to the transformer layer the node belongs to (layer index is
//      parsed from the tensor name suffix, e.g. "l_out-13" -> layer 13).
//   2. Captures per-layer output tensor statistics (shape, dtype, min, max,
//      mean, std, L2-norm, NaN/Inf counts, head/tail sample values and,
//      optionally, the full value array).
//
// Results are written to a JSON file at program exit in the unified
// "layer-profile/v1" schema that is shared across inference engines. See
// docs/layer-profile-schema.md for the full specification.

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct common_params;

struct layer_profile_config {
    std::string output_path;
    bool        dump_full_values = false;
    int         n_sample_values  = 8;

    // Directory for per-layer CSV dumps. When non-empty the profiler
    // writes `<csv_dir>/layer_<il>.csv` with `csv_n_samples` randomly
    // sampled (index,value) pairs for each layer. Sampling uses a seed
    // derived from the current hour so that multiple runs within the
    // same hour pick the same indices.
    std::string csv_dir;
    int         csv_n_samples    = 300;

    std::string engine_name      = "llama.cpp";
    std::string engine_version;
    std::string model_id;
    std::string device;
    std::string dtype;
};

struct layer_profile_stats {
    std::vector<int64_t> shape;     // ne[0..3]
    std::string          dtype;
    int64_t              n_elements = 0;

    float    min     = 0.0f;
    float    max     = 0.0f;
    double   sum     = 0.0;
    double   mean    = 0.0;
    double   std_dev = 0.0;
    double   l2_norm = 0.0;
    int64_t  nan_count = 0;
    int64_t  inf_count = 0;

    std::vector<float> sample_head;
    std::vector<float> sample_tail;
    std::vector<float> full_values;   // populated only if dump_full_values
};

struct layer_profile_entry {
    int         layer_index = -1;
    std::string output_tensor_name;

    int64_t tensor_count  = 0;
    int64_t call_count    = 0;
    int64_t total_time_us = 0;

    layer_profile_stats output;

    // Random-subsample buffer populated when csv_dir is configured.
    // `csv_indices` is generated once on the first output capture and
    // then reused across forward passes; `csv_values` holds the most
    // recently captured values at those indices.
    std::vector<int64_t> csv_indices;
    std::vector<float>   csv_values;
};

struct layer_profiler {
    layer_profile_config config;

    std::mutex mtx;

    std::unordered_map<int, layer_profile_entry> layers;

    // rolling state used to attribute node wall-time to layers
    int64_t prev_tensor_time_us = 0;
    int     prev_tensor_layer   = -2;

    int64_t total_compute_time_us = 0;
    int64_t n_forward_passes      = 0;
    int64_t n_layer_max           = 0;

    // Seed used for CSV index sampling. Derived once at install time
    // from the current hour (0..23) so that every layer in the run
    // shares the same RNG stream.
    uint64_t csv_seed = 0;
};

// Install the profiler on `params`. Replaces any pre-existing
// params.cb_eval with the layer profiler callback and registers an
// atexit() handler that writes the JSON report on program shutdown.
// No-op when params.layer_profile_path is empty.
void common_layer_profiler_install(common_params & params);

// Manually flush the profiler state to disk. Safe to call multiple times.
void common_layer_profiler_flush();

// The eval callback. Exposed for unit tests; normal callers should not
// invoke this directly.
bool common_layer_profiler_cb_eval(ggml_tensor * t, bool ask, void * user_data);
