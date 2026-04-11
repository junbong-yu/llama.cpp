# Job-to-do: Per-layer profiling for llama.cpp

## 1. Goal

Add a way to, for every transformer layer that `llama.cpp` executes:

1. Measure the wall-clock time spent inside that layer, and
2. Capture the value of the layer's output tensor (summary statistics and,
   optionally, the full value array),

and write both into a **JSON file with a unified, cross-engine schema** so
that the same information produced by another inference engine (vLLM,
TensorRT-LLM, MLX, ...) can be diffed against `llama.cpp` layer-by-layer.

The work is delivered as:

- source changes on branch `claude/llamacpp-layer-profiling-xmqUf`,
- a self-contained patch file (`docs/patches/layer-profile.patch`),
- a schema specification (`docs/layer-profile-schema.md`),
- this write-up.

## 2. Files added

| File | Purpose |
| ---- | ------- |
| `common/layer-profile.h` | Profiler data structures and public API. |
| `common/layer-profile.cpp` | Profiler implementation (callback, statistics, JSON emission, atexit flush). |
| `docs/layer-profile-schema.md` | Full specification of the `layer-profile/v1` JSON schema so other engines can produce the same format. |
| `docs/patches/layer-profile.patch` | Standalone `git diff` patch that applies cleanly on top of llama.cpp master. |
| `job-to-do.md` | This document. |

## 3. Files modified

| File | Change |
| ---- | ------ |
| `common/common.h` | Added three fields on `common_params`: `layer_profile_path`, `layer_profile_full`, `layer_profile_samples`. |
| `common/arg.cpp` | Registered three new CLI flags: `--layer-profile`, `--layer-profile-full`, `--layer-profile-samples`. |
| `common/common.cpp` | Call `common_layer_profiler_install(params)` at the top of `common_init_from_params`, before the `llama_context` is constructed. |
| `common/CMakeLists.txt` | Added `layer-profile.cpp` / `layer-profile.h` to the `common` static library. |

## 4. How it works

### 4.1 Hook point

llama.cpp already exposes a per-node graph execution hook via
`ggml_backend_sched_eval_callback`, surfaced on the public `common_params`
struct as `cb_eval` / `cb_eval_user_data`, and already wired through
`common_context_params_from_params` into `llama_cparams`. The layer profiler
reuses this callback — no ggml or `llama_context` changes are required.

The scheduler calls the eval callback twice per graph node:

- `ask=true`: the scheduler asks whether the callback wants this node's data.
  Returning `true` triggers the backend to synchronize and copy the tensor to
  host memory.
- `ask=false`: the node has finished executing; `t->data` (or the backend
  buffer) is safe to read.

The profiler does all its work on the `ask=false` pass.

### 4.2 Attributing time to layers

Graph nodes in llama.cpp are tagged with names like `<base>-<il>` (see
`llama_context::graph_get_cb` in `src/llama-context.cpp`), where `<il>` is
the transformer layer index. For example the final residual output of
layer 13 in the LLaMA graph builder is called `l_out-13`.

Given that naming convention, the profiler's eval callback does the
following on every `ask=false` pass:

1. Read the current wall-clock time `now = ggml_time_us()`.
2. Attribute `now - prev_tensor_time_us` to the **previous** node's layer
   (since the previous node has just finished executing by the time we see
   this pass).
3. Parse the current tensor's layer index by looking at its name suffix.
4. If the tensor's base name is `l_out` or `ffn_out` (the two tensor names
   used by llama.cpp model builders for the final per-layer residual
   output), record its statistics as the layer's output snapshot.
5. Save `(prev_tensor_layer, prev_tensor_time_us) = (il, now)`.

Across multiple forward passes the profiler accumulates total time per
layer, counts the number of times each layer's output was observed, and
keeps the statistics of the most recent output.

### 4.3 Statistics captured per layer output

Implemented in `compute_stats()` in `common/layer-profile.cpp`:

- `shape` — `ne[0..3]`.
- `dtype` — the ggml type name (`f32`, `f16`, `bf16`, `q4_0`, ...).
- `n_elements` — `ggml_nelements(t)`.
- `min`, `max`, `sum`, `mean`, `std`, `l2_norm` — computed over all non-NaN,
  non-Inf values.
- `nan_count`, `inf_count`.
- `sample_head`, `sample_tail` — first/last N flattened values
  (N is configurable via `--layer-profile-samples`, default 8).
- `values` — optional, full flattened value array, populated only when
  `--layer-profile-full` is passed.

For quantized dtypes only `shape`, `dtype` and `n_elements` are captured to
keep the implementation small; statistics would require dequantization which
is out of scope for this patch.

Tensors that live on non-host backends are copied to a staging buffer via
`ggml_backend_tensor_get` before being read (the same pattern used by
`common/debug.cpp`).

### 4.4 Installation and lifecycle

`common_layer_profiler_install(params)` is called at the top of
`common_init_from_params`, before `new common_init_result(params)` creates
the `llama_context`. This is important: `cb_eval` must be set on
`common_params` before it is copied into `llama_context_params` and stored
on the context. The install function:

1. No-ops if `params.layer_profile_path` is empty.
2. Allocates a singleton `layer_profiler` held in a file-static
   `std::unique_ptr`.
3. Sets `params.cb_eval` to the profiler callback and
   `params.cb_eval_user_data` to the singleton pointer.
4. Registers `std::atexit(common_layer_profiler_atexit)` exactly once.

The atexit handler calls `common_layer_profiler_flush()` which serializes
the accumulated data to the configured JSON path using `nlohmann::json`
(already a build-time dependency of `common`).

### 4.5 CLI surface

```
--layer-profile FNAME
    write per-layer timing and output statistics to FNAME as JSON
    (schema: layer-profile/v1, see docs/layer-profile-schema.md)

--layer-profile-full
    also record full per-layer output values in the layer profile JSON
    (warning: large output)

--layer-profile-samples N
    number of head/tail sample values to record per layer output in the
    layer profile JSON (default: 8)

--layer-values-csv DIR
    write one CSV file per layer (layer_<il>.csv) with 300 randomly
    sampled output values to DIR; sampling uses a seed derived from
    the current hour and draws 300 uniform indices per layer
```

Because the flags are attached to the default example group (i.e. no
`.set_examples({...})`), they are available on every llama.cpp tool that
uses `common_params` — `llama-cli`, `llama-perplexity`, `llama-bench`,
`llama-server`, etc.

Typical invocation:

```
./build/bin/llama-cli \
    -m /models/llama-3-8b.gguf \
    -p "The quick brown fox" \
    -n 1 \
    --layer-profile layer-profile.json
```

which produces `layer-profile.json` with the `layer-profile/v1` schema.

## 4.6 Per-layer CSV value dumps (`--layer-values-csv`)

Separately from the JSON profile, the profiler can write one CSV file
per layer containing randomly sampled output values. This is enabled by
a single new flag:

```
--layer-values-csv DIR
```

Either `--layer-profile` (JSON), `--layer-values-csv` (CSV), or both
may be used on the same run — they are independent outputs.

### Sampling procedure

- RNG seed = current local hour (`localtime(now).tm_hour`, range 0..23).
- RNG = `std::mt19937_64` seeded with that value.
- Draw exactly **300** indices uniformly from `[0, n_elements)` for the
  layer's output tensor. Duplicates are permitted so that every layer's
  CSV has exactly 300 rows regardless of tensor size.
- The index list is generated once per layer on the first output
  observation and cached on the layer entry, so subsequent forward
  passes reuse the same indices.
- On every forward pass, the values at those 300 positions are re-read
  from the tensor and overwritten in the profiler state, so the final
  CSV reflects the **last** observed forward pass.

The `csv_n_samples` knob lives on `layer_profile_config` if a reviewer
wants to tweak the sample count without touching the callsite.

### File layout

For each observed layer, one file is written to `DIR`:

```
DIR/
    layer_000.csv
    layer_001.csv
    ...
    layer_031.csv
```

Each file contains exactly one header row plus exactly 300 data rows:

```
index,value
17,0.4213
91,-1.8809
...
```

where `index` is the row-major flattened offset into the layer's output
tensor and `value` is the dequantized `f32` value at that offset.

### Seed rationale

Using the current hour as a seed means:

- Runs within the same hour pick the same indices → the CSV files are
  directly diffable across runs without extra configuration.
- Runs in different hours pick different indices → the sampled coverage
  rotates across the tensor over time so that failures that depend on
  specific positions still have a chance of being hit.

This matches the sampling strategy the user explicitly asked for
(300 uniform indices per layer) and is documented in the schema doc
(`docs/layer-profile-schema.md` §5) so other engines can reproduce
the exact same sampling.

## 5. Output schema (summary)

Full spec lives in `docs/layer-profile-schema.md`. Briefly:

```jsonc
{
  "schema":                "layer-profile/v1",
  "engine":                "llama.cpp",
  "engine_version":        "...",
  "model_id":              "/path/to/model.gguf",
  "device":                "cpu",
  "dtype":                 "f16",
  "n_layer":               32,
  "n_forward_passes":      7,
  "total_compute_time_us": 123456789,
  "layers": [
    {
      "layer_index":        0,
      "output_tensor_name": "l_out-0",
      "tensor_count":       142,
      "call_count":         7,
      "total_time_us":      3421,
      "avg_time_us":        488.7,
      "output": {
        "shape":       [1, 4096, 1, 1],
        "dtype":       "f32",
        "n_elements":  4096,
        "min":         -3.2109,
        "max":          2.8750,
        "sum":         12.125,
        "mean":         0.00296,
        "std":          0.41234,
        "l2_norm":     26.3421,
        "nan_count":    0,
        "inf_count":    0,
        "sample_head": [...],
        "sample_tail": [...]
      }
    }
    /* ... one object per layer, sorted ascending ... */
  ]
}
```

## 6. How another inference engine reproduces the format

`docs/layer-profile-schema.md` has a dedicated "Producer checklist" section.
The minimal steps are:

1. Hook per-graph-node execution (e.g. PyTorch forward hooks, CUDA events,
   JAX profiler).
2. Attribute each node to a layer (engines that know layer indices directly
   from module paths can skip llama.cpp's name-suffix parsing trick).
3. Measure wall time per node and accumulate into `layer.total_time_us`.
4. At the final residual of each layer, compute the same statistics object
   (§3 of the schema).
5. Serialize on shutdown with ordered keys and UTF-8.

## 7. What was built and verified

- [x] Source compiles cleanly (`cmake --build build --target common -j$(nproc)`
      succeeds, including `common/layer-profile.cpp`).
- [x] `libcommon.a` links into the rest of the llama.cpp build.
- [x] Patch file `docs/patches/layer-profile.patch` is complete and
      self-contained (`git apply --check` passes on a clean checkout).
- [ ] **Not verified end-to-end:** a real inference run with
      `--layer-profile` producing a populated JSON file. This requires a
      GGUF model to be available at runtime, which is outside the scope of
      the harness used to author this patch. The reviewer SHOULD run
      `llama-cli --layer-profile ...` against a small model before shipping.

## 8. Known limitations and follow-ups

- **Layer timing is approximate.** The eval callback observes graph nodes
  after the backend reports them, not hardware-exact kernel intervals.
  For more precise GPU timings on CUDA/Metal/Vulkan one could pair this
  with backend-native events (e.g. `cudaEventRecord`). Left as future work.
- **Only `l_out` and `ffn_out` are recognized as layer outputs.** A few
  exotic model builders in `src/models/` may name the final residual
  differently; those layers will have timing data but an empty `output`
  object. A simple fix is to additionally recognize `result_norm` etc. —
  left for follow-up once concrete models are identified.
- **Quantized tensors are not dequantized.** Only shape/dtype/n_elements
  are reported for them. Full statistics would require reusing the
  dequantization paths in `ggml-quants.c`.
- **`common/debug.cpp` already installs a debug eval callback.** If both
  `--layer-profile` and the existing debug flags (`--save-logits`,
  `--tensor-filter`) are used together, the layer profiler currently wins
  (and logs a warning). If both are wanted simultaneously, a chaining
  callback would need to be introduced.

## 9. Summary of the deliverables

| Deliverable | Location |
| ----------- | -------- |
| Patch file | `docs/patches/layer-profile.patch` |
| Schema spec (for other engines) | `docs/layer-profile-schema.md` |
| Source: profiler header | `common/layer-profile.h` |
| Source: profiler implementation | `common/layer-profile.cpp` |
| Source: CLI + wiring | `common/common.h`, `common/arg.cpp`, `common/common.cpp`, `common/CMakeLists.txt` |
| Write-up | `job-to-do.md` (this file) |
