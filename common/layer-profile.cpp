#include "layer-profile.h"

#include "common.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <unordered_set>

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// global state
// ---------------------------------------------------------------------------

static std::unique_ptr<layer_profiler> g_profiler;
static std::atomic<bool>               g_atexit_registered{false};

// ---------------------------------------------------------------------------
// name parsing helpers
// ---------------------------------------------------------------------------

// llama.cpp tags layer-bound tensors as "<base>-<layer_index>" (see
// llama_context::graph_get_cb in src/llama-context.cpp). We reuse that
// suffix to recover the layer each node belongs to.
static int parse_layer_index(const char * name) {
    if (!name || !*name) {
        return -1;
    }
    const char * dash = std::strrchr(name, '-');
    if (!dash || !*(dash + 1)) {
        return -1;
    }
    int         il  = 0;
    bool        any = false;
    for (const char * p = dash + 1; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return -1;
        }
        il  = il * 10 + (*p - '0');
        any = true;
    }
    return any ? il : -1;
}

static std::string extract_base_name(const char * name) {
    if (!name) {
        return {};
    }
    const char * dash = std::strrchr(name, '-');
    if (!dash) {
        return name;
    }
    for (const char * p = dash + 1; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return name;
        }
    }
    return std::string(name, dash - name);
}

// Tensor name used by llama.cpp model builders as the final per-layer
// residual output. The canonical choice is `l_out`; `ffn_out` is a
// reasonable fallback for the few builders that don't emit `l_out`.
static bool is_layer_output_name(const std::string & base) {
    return base == "l_out" || base == "ffn_out";
}

// ---------------------------------------------------------------------------
// CSV index sampling
// ---------------------------------------------------------------------------

// Seed derived from the current local hour (0..23). Multiple runs within
// the same hour will therefore pick the same 1000 tensor positions, which
// is the behavior explicitly requested by the user.
static uint64_t current_hour_seed() {
    const std::time_t now = std::time(nullptr);
    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now);
#else
    localtime_r(&now, &local_tm);
#endif
    return (uint64_t) local_tm.tm_hour;
}

// Build the list of tensor positions that will be dumped to CSV for a
// given layer. The returned vector is sorted in ascending order so
// that the resulting CSV is easy to diff line-by-line.
//
//   - If the tensor has <= n_samples elements, every position is kept
//     in natural order (no randomness is needed or useful).
//   - Otherwise, draw n_samples *unique* indices uniformly from
//     [0, n_elements) with rejection sampling and then sort them.
static std::vector<int64_t> make_sample_indices(int64_t  n_elements,
                                                uint64_t seed,
                                                int      n_samples) {
    if (n_elements <= 0 || n_samples <= 0) {
        return {};
    }

    if (n_elements <= (int64_t) n_samples) {
        std::vector<int64_t> all;
        all.reserve((size_t) n_elements);
        for (int64_t i = 0; i < n_elements; ++i) {
            all.push_back(i);
        }
        return all;
    }

    std::mt19937_64                        rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, n_elements - 1);

    std::vector<int64_t>        picked;
    std::unordered_set<int64_t> seen;
    picked.reserve((size_t) n_samples);
    seen.reserve((size_t) n_samples);

    while ((int) picked.size() < n_samples) {
        const int64_t idx = dist(rng);
        if (seen.insert(idx).second) {
            picked.push_back(idx);
        }
    }

    std::sort(picked.begin(), picked.end());
    return picked;
}

// ---------------------------------------------------------------------------
// tensor statistics
// ---------------------------------------------------------------------------

static void capture_layer_output(const ggml_tensor          * t,
                                 layer_profile_entry        & e,
                                 const layer_profile_config & cfg,
                                 uint64_t                     csv_seed) {
    layer_profile_stats & s = e.output;

    s.shape.assign(t->ne, t->ne + GGML_MAX_DIMS);
    s.dtype      = ggml_type_name(t->type);
    s.n_elements = (int64_t) ggml_nelements(t);

    // For now we only compute value statistics for the float types the
    // debug tooling already supports. Quantized tensors are recorded with
    // shape/dtype only.
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        return;
    }

    const size_t n_bytes = ggml_nbytes(t);
    std::vector<uint8_t> staging;
    const uint8_t * data_ptr = nullptr;

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (is_host) {
        data_ptr = (const uint8_t *) t->data;
    } else {
        staging.resize(n_bytes);
        ggml_backend_tensor_get(t, staging.data(), 0, n_bytes);
        data_ptr = staging.data();
    }

    auto load_f32 = [&](int64_t idx) -> float {
        if (t->type == GGML_TYPE_F32) {
            return ((const float *) data_ptr)[idx];
        }
        if (t->type == GGML_TYPE_F16) {
            return ggml_fp16_to_fp32(((const ggml_fp16_t *) data_ptr)[idx]);
        }
        return ggml_bf16_to_fp32(((const ggml_bf16_t *) data_ptr)[idx]);
    };

    const int64_t n = s.n_elements;

    double   sum_v = 0.0;
    double   sum_sq = 0.0;
    float    mn  =  std::numeric_limits<float>::infinity();
    float    mx  = -std::numeric_limits<float>::infinity();
    int64_t  nans = 0;
    int64_t  infs = 0;
    int64_t  valid = 0;

    for (int64_t i = 0; i < n; ++i) {
        const float v = load_f32(i);
        if (std::isnan(v)) { ++nans; continue; }
        if (std::isinf(v)) { ++infs; continue; }
        sum_v  += v;
        sum_sq += (double) v * (double) v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        ++valid;
    }

    s.min       = valid > 0 ? mn : 0.0f;
    s.max       = valid > 0 ? mx : 0.0f;
    s.sum       = sum_v;
    s.mean      = valid > 0 ? sum_v / (double) valid : 0.0;
    const double var = valid > 0 ? (sum_sq / (double) valid) - (s.mean * s.mean) : 0.0;
    s.std_dev   = var > 0.0 ? std::sqrt(var) : 0.0;
    s.l2_norm   = std::sqrt(sum_sq);
    s.nan_count = nans;
    s.inf_count = infs;

    const int n_sample = std::min<int>(cfg.n_sample_values, (int) n);
    s.sample_head.resize(n_sample);
    for (int i = 0; i < n_sample; ++i) {
        s.sample_head[i] = load_f32(i);
    }
    s.sample_tail.resize(n_sample);
    for (int i = 0; i < n_sample; ++i) {
        s.sample_tail[i] = load_f32(n - n_sample + i);
    }

    if (cfg.dump_full_values) {
        s.full_values.resize(n);
        for (int64_t i = 0; i < n; ++i) {
            s.full_values[i] = load_f32(i);
        }
    }

    // CSV random subsample: generate the per-layer index list once (on
    // the first observation of this layer's output) and reuse it across
    // subsequent forward passes. Values are refreshed every call so the
    // CSV reflects the last forward pass that was seen.
    if (!cfg.csv_dir.empty()) {
        if (e.csv_indices.empty()) {
            e.csv_indices = make_sample_indices(
                s.n_elements, csv_seed, cfg.csv_n_samples);
        }
        e.csv_values.clear();
        e.csv_values.reserve(e.csv_indices.size());
        for (int64_t idx : e.csv_indices) {
            if (idx < 0 || idx >= s.n_elements) {
                continue;
            }
            e.csv_values.push_back(load_f32(idx));
        }
    }
}

// ---------------------------------------------------------------------------
// eval callback
// ---------------------------------------------------------------------------

bool common_layer_profiler_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * prof = (layer_profiler *) user_data;
    if (!prof) {
        return true;
    }

    // The scheduler invokes the callback twice per node: once with ask=true
    // to request the data, and once with ask=false after the node has been
    // computed and copied to host if necessary. We do our accounting on the
    // ask=false pass so that `ggml_backend_tensor_get` is safe to call.
    if (ask) {
        return true;
    }

    const int64_t now = ggml_time_us();

    std::lock_guard<std::mutex> lock(prof->mtx);

    // Attribute the time delta to the PREVIOUS node. When we see the
    // ask=false pass for node N, node N-1 has just finished executing, so
    // dt = now - prev_time_us is a close approximation of N-1's wall time.
    if (prof->prev_tensor_time_us != 0) {
        const int64_t dt = now - prof->prev_tensor_time_us;
        if (dt > 0) {
            prof->total_compute_time_us += dt;
            if (prof->prev_tensor_layer >= 0) {
                auto & e = prof->layers[prof->prev_tensor_layer];
                e.layer_index    = prof->prev_tensor_layer;
                e.total_time_us += dt;
                e.tensor_count  += 1;
            }
        }
    }

    const int il = parse_layer_index(t->name);
    if (il >= 0) {
        const std::string base = extract_base_name(t->name);
        if (is_layer_output_name(base)) {
            auto & e = prof->layers[il];
            e.layer_index = il;
            if (e.output_tensor_name.empty()) {
                e.output_tensor_name = t->name;
            }
            capture_layer_output(t, e, prof->config, prof->csv_seed);
            e.call_count += 1;
            if (il == 0 && base == "l_out") {
                prof->n_forward_passes += 1;
            }
        }
        if (il + 1 > prof->n_layer_max) {
            prof->n_layer_max = il + 1;
        }
    }

    prof->prev_tensor_layer   = il;
    prof->prev_tensor_time_us = now;
    return true;
}

// ---------------------------------------------------------------------------
// serialization
// ---------------------------------------------------------------------------

static json stats_to_json(const layer_profile_stats & s, bool include_full) {
    json j;
    j["shape"]       = s.shape;
    j["dtype"]       = s.dtype;
    j["n_elements"]  = s.n_elements;
    j["min"]         = s.min;
    j["max"]         = s.max;
    j["sum"]         = s.sum;
    j["mean"]        = s.mean;
    j["std"]         = s.std_dev;
    j["l2_norm"]     = s.l2_norm;
    j["nan_count"]   = s.nan_count;
    j["inf_count"]   = s.inf_count;
    j["sample_head"] = s.sample_head;
    j["sample_tail"] = s.sample_tail;
    if (include_full) {
        j["values"] = s.full_values;
    }
    return j;
}

static void write_layer_csv_files(const layer_profiler & prof,
                                  const std::vector<int> & keys) {
    const std::string & dir = prof.config.csv_dir;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_ERR("%s: failed to create CSV directory '%s': %s\n",
                __func__, dir.c_str(), ec.message().c_str());
        return;
    }

    int written = 0;
    for (int il : keys) {
        const auto it = prof.layers.find(il);
        if (it == prof.layers.end()) {
            continue;
        }
        const auto & e = it->second;
        if (e.csv_values.empty()) {
            continue;
        }

        char fname[64];
        std::snprintf(fname, sizeof(fname), "layer_%03d.csv", il);
        const std::filesystem::path path = std::filesystem::path(dir) / fname;

        std::ofstream ofs(path);
        if (!ofs.is_open()) {
            LOG_ERR("%s: failed to open '%s' for writing\n",
                    __func__, path.string().c_str());
            continue;
        }

        ofs << "index,value\n";
        const size_t n = std::min(e.csv_indices.size(), e.csv_values.size());
        for (size_t i = 0; i < n; ++i) {
            ofs << e.csv_indices[i] << ',' << e.csv_values[i] << '\n';
        }
        ++written;
    }
    LOG_INF("%s: wrote %d per-layer CSV file(s) to %s (seed=%llu)\n",
            __func__, written, dir.c_str(),
            (unsigned long long) prof.csv_seed);
}

void common_layer_profiler_flush() {
    if (!g_profiler) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_profiler->mtx);

    std::vector<int> keys;
    keys.reserve(g_profiler->layers.size());
    for (auto & kv : g_profiler->layers) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());

    // --- JSON profile (optional) ---
    if (!g_profiler->config.output_path.empty()) {
        json root;
        root["schema"]                = "layer-profile/v1";
        root["engine"]                = g_profiler->config.engine_name;
        root["engine_version"]        = g_profiler->config.engine_version;
        root["model_id"]              = g_profiler->config.model_id;
        root["device"]                = g_profiler->config.device;
        root["dtype"]                 = g_profiler->config.dtype;
        root["n_layer"]               = g_profiler->n_layer_max;
        root["n_forward_passes"]      = g_profiler->n_forward_passes;
        root["total_compute_time_us"] = g_profiler->total_compute_time_us;

        json layers_arr = json::array();
        for (int il : keys) {
            const auto & e = g_profiler->layers[il];
            json je;
            je["layer_index"]        = e.layer_index;
            je["output_tensor_name"] = e.output_tensor_name;
            je["tensor_count"]       = e.tensor_count;
            je["call_count"]         = e.call_count;
            je["total_time_us"]      = e.total_time_us;
            je["avg_time_us"]        = e.call_count > 0
                                           ? (double) e.total_time_us / (double) e.call_count
                                           : 0.0;
            je["output"]             = stats_to_json(e.output, g_profiler->config.dump_full_values);
            layers_arr.push_back(je);
        }
        root["layers"] = layers_arr;

        std::ofstream ofs(g_profiler->config.output_path);
        if (!ofs.is_open()) {
            LOG_ERR("%s: failed to open '%s' for writing\n",
                    __func__, g_profiler->config.output_path.c_str());
        } else {
            ofs << root.dump(2) << std::endl;
            LOG_INF("%s: wrote layer profile to %s\n",
                    __func__, g_profiler->config.output_path.c_str());
        }
    }

    // --- per-layer CSV value dumps (optional) ---
    if (!g_profiler->config.csv_dir.empty()) {
        write_layer_csv_files(*g_profiler, keys);
    }
}

// ---------------------------------------------------------------------------
// installation
// ---------------------------------------------------------------------------

static void common_layer_profiler_atexit() {
    common_layer_profiler_flush();
}

void common_layer_profiler_install(common_params & params) {
    if (params.layer_profile_path.empty() && params.layer_values_csv_dir.empty()) {
        return;
    }
    if (g_profiler) {
        LOG_WRN("%s: layer profiler already installed, skipping\n", __func__);
        return;
    }

    g_profiler = std::make_unique<layer_profiler>();
    g_profiler->config.output_path      = params.layer_profile_path;
    g_profiler->config.dump_full_values = params.layer_profile_full;
    g_profiler->config.n_sample_values  = params.layer_profile_samples;
    g_profiler->config.csv_dir          = params.layer_values_csv_dir;
    g_profiler->config.engine_name      = "llama.cpp";
    g_profiler->config.model_id         = params.model.path;
    g_profiler->csv_seed                = current_hour_seed();

    if (params.cb_eval) {
        LOG_WRN("%s: an existing cb_eval was set on common_params and will be "
                "replaced by the layer profiler callback\n", __func__);
    }
    params.cb_eval           = common_layer_profiler_cb_eval;
    params.cb_eval_user_data = g_profiler.get();

    bool expected = false;
    if (g_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(common_layer_profiler_atexit);
    }

    LOG_INF("%s: layer profiler installed "
            "(json=%s, csv_dir=%s, full_values=%d, samples=%d, csv_seed=%llu)\n",
            __func__,
            g_profiler->config.output_path.empty() ? "<off>" : g_profiler->config.output_path.c_str(),
            g_profiler->config.csv_dir.empty()     ? "<off>" : g_profiler->config.csv_dir.c_str(),
            (int) g_profiler->config.dump_full_values,
            g_profiler->config.n_sample_values,
            (unsigned long long) g_profiler->csv_seed);
}
