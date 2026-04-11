# Layer Profile Schema (`layer-profile/v1`)

This document specifies a **unified JSON format** for per-layer inference
profiling output. The goal is to allow inference-engine-agnostic comparison of
per-layer execution timings and output activations: given the same model,
tokenizer and prompt, two engines (e.g. `llama.cpp` and another runtime)
should emit files that follow this schema so that a downstream tool can diff
them layer-by-layer.

The llama.cpp reference producer lives in
[`common/layer-profile.{h,cpp}`](../common/layer-profile.cpp) and is activated
by passing `--layer-profile <path>` to any tool that uses `common_params`
(e.g. `llama-cli`).

---

## 1. Top-level object

```jsonc
{
  "schema":                "layer-profile/v1",
  "engine":                "llama.cpp",
  "engine_version":        "b5632",
  "model_id":              "/models/llama-3-8b-f16.gguf",
  "device":                "cpu",
  "dtype":                 "f16",
  "n_layer":               32,
  "n_forward_passes":      7,
  "total_compute_time_us": 123456789,
  "layers": [ /* array of layer objects, see §2 */ ]
}
```

| Field                    | Type    | Required | Description                                                                                                                                          |
| ------------------------ | ------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema`                 | string  | yes      | Must be the literal `"layer-profile/v1"`. Consumers MUST reject unknown schema strings.                                                              |
| `engine`                 | string  | yes      | Name of the inference engine that produced the file (e.g. `"llama.cpp"`, `"vllm"`, `"tensorrt-llm"`, `"mlx"`).                                       |
| `engine_version`         | string  | no       | Free-form version identifier (git hash, release tag, ...).                                                                                           |
| `model_id`               | string  | yes      | Path, HuggingFace repo id or any other identifier of the underlying model. Consumers SHOULD refuse to diff two files with different `model_id`.     |
| `device`                 | string  | no       | Execution device (`"cpu"`, `"cuda:0"`, `"metal"`, ...).                                                                                              |
| `dtype`                  | string  | no       | Dominant compute dtype. Use the same string the engine prints internally (`"f32"`, `"f16"`, `"bf16"`, `"q4_0"`, ...).                                |
| `n_layer`                | integer | yes      | Number of transformer layers observed during profiling. SHOULD equal `hparams.n_layer`.                                                              |
| `n_forward_passes`       | integer | yes      | How many forward passes were accumulated into the timings / stats. A single `decode()` on one token is one pass.                                      |
| `total_compute_time_us`  | integer | yes      | Sum of wall-clock time (microseconds) attributed to the full graph across all recorded forward passes. Equal to the sum of per-layer `total_time_us` only if every graph node could be attributed to a layer. |
| `layers`                 | array   | yes      | Array of per-layer objects, sorted in ascending order of `layer_index`. See §2.                                                                       |

### 1.1 Identity across engines

For two files produced by different engines to be meaningfully comparable, the
following MUST be equal:

- `schema`, `n_layer`, `model_id` (or a documented mapping between model ids)
- the tokenizer and the input prompt used to drive the forward pass
- the requested output dtype (see §3)

The number of forward passes (`n_forward_passes`) does NOT have to match —
consumers should compare per-layer averages (`avg_time_us`) when it differs.

---

## 2. Layer object

Each entry in `layers` describes one transformer layer:

```jsonc
{
  "layer_index":        0,
  "output_tensor_name": "l_out-0",
  "tensor_count":       142,
  "call_count":         7,
  "total_time_us":      3421,
  "avg_time_us":        488.7,
  "output": { /* tensor-stats object, see §3 */ }
}
```

| Field                | Type    | Required | Description                                                                                                                                                                                                                                              |
| -------------------- | ------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `layer_index`        | integer | yes      | 0-indexed transformer layer index.                                                                                                                                                                                                                         |
| `output_tensor_name` | string  | no       | Name of the tensor that was treated as the layer's final residual output (after the last `add`). For llama.cpp this is typically `"l_out-<il>"`. Engines that don't use named tensors SHOULD emit a human-readable identifier such as `"layer.0.output"`. |
| `tensor_count`       | integer | yes      | Number of individual graph nodes that were attributed to this layer during profiling.                                                                                                                                                                      |
| `call_count`         | integer | yes      | Number of times the layer's output tensor was produced (usually equals `n_forward_passes`).                                                                                                                                                                |
| `total_time_us`      | integer | yes      | Total wall-clock time (microseconds) spent inside this layer, summed across every forward pass.                                                                                                                                                            |
| `avg_time_us`        | number  | yes      | `total_time_us / call_count`, for convenience.                                                                                                                                                                                                             |
| `output`             | object  | yes      | Statistics of the layer output tensor captured from the **last** forward pass. See §3.                                                                                                                                                                     |

Layers MUST be sorted by `layer_index` ascending. A layer MAY be omitted from
the list if nothing was attributed to it (e.g. when the profiler was attached
mid-way through inference); consumers MUST handle gaps gracefully.

---

## 3. Tensor-stats object

The `output` field of each layer uses this sub-schema. The same shape is
expected when other engines report intermediate activations.

```jsonc
{
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
  "sample_head": [0.013, -0.221, 0.198, 0.004, -0.112, 0.341, 0.007, -0.055],
  "sample_tail": [-0.089, 0.412, 0.029, -0.004, 0.118, -0.205, 0.002, 0.067],
  "values":      [ /* optional, full value array — see below */ ]
}
```

| Field         | Type          | Required | Description                                                                                                                                                          |
| ------------- | ------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shape`       | int[]         | yes      | Four-element array with the tensor's `ne[0..3]`. Engines with fewer dimensions MUST pad with `1`.                                                                    |
| `dtype`       | string        | yes      | Dtype name as printed by the engine (`"f32"`, `"f16"`, `"bf16"`, `"q4_0"`, ...). Quantized dtypes are permitted but only the metadata fields below will be populated. |
| `n_elements`  | integer       | yes      | Product of `shape` entries.                                                                                                                                          |
| `min`         | number        | yes*     | Minimum non-NaN, non-Inf value. `0` if all elements are non-finite. Not required for quantized dtypes.                                                               |
| `max`         | number        | yes*     | Maximum non-NaN, non-Inf value. `0` if all elements are non-finite.                                                                                                  |
| `sum`         | number        | yes*     | Sum of non-NaN, non-Inf values.                                                                                                                                      |
| `mean`        | number        | yes*     | `sum / valid_count` where `valid_count = n_elements - nan_count - inf_count`.                                                                                        |
| `std`         | number        | yes*     | Population standard deviation computed over valid values.                                                                                                            |
| `l2_norm`     | number        | yes*     | √Σxᵢ² over valid values.                                                                                                                                             |
| `nan_count`   | integer       | yes      | Number of NaN elements.                                                                                                                                              |
| `inf_count`   | integer       | yes      | Number of ±Inf elements.                                                                                                                                             |
| `sample_head` | number[]      | yes      | First N flattened values in row-major order. N is controlled by the producer (llama.cpp default: 8, override via `--layer-profile-samples`).                          |
| `sample_tail` | number[]      | yes      | Last N flattened values in row-major order.                                                                                                                          |
| `values`      | number[]      | no       | Full flattened value array in row-major order. Present only when the producer was configured to dump full values (llama.cpp: `--layer-profile-full`).                |

\* Fields marked `yes*` are required for float dtypes (`f32`, `f16`, `bf16`).
For quantized dtypes the producer MAY emit them as `0` or omit them; consumers
MUST tolerate both.

### 3.1 Float precision

All numeric fields SHOULD be emitted as IEEE-754 doubles in JSON. Producers
MUST NOT emit `NaN` or `Infinity` literals — those are not valid JSON. Instead,
non-finite aggregates should be reported via the `nan_count` / `inf_count`
fields and the corresponding aggregate value should be set to `0`.

### 3.2 Tolerance for cross-engine diffs

When diffing two `layer-profile/v1` files, a reasonable default tolerance for
`f16` activations is:

| Field     | Absolute tolerance | Relative tolerance |
| --------- | ------------------ | ------------------ |
| `mean`    | 1e-4               | 1e-3               |
| `std`     | 1e-4               | 1e-3               |
| `l2_norm` | 1e-3               | 1e-3               |
| `min`/`max` | 1e-3            | 1e-3               |
| `sample_head[i]`, `sample_tail[i]` | 1e-3 | 1e-3 |

Consumers are free to apply their own tolerances. The producer is not
responsible for enforcing them.

---

## 4. Producer checklist (for other engines)

If you want to emit the same schema from another engine (e.g. vLLM, TensorRT-LLM,
MLX, transformers), the minimum work is:

1. **Hook per-graph-node execution.** You need a way to observe each
   operator or kernel as it runs. In llama.cpp this is
   `ggml_backend_sched_eval_callback`; in PyTorch you can use
   `torch.nn.Module` forward hooks or CUDA events; in JAX you can use
   `jax.profiler`.
2. **Attribute each node to a layer.** The llama.cpp reference producer
   parses the tensor name suffix (`<base>-<il>`). Other engines typically
   know the layer index directly from the module path
   (e.g. `model.layers.13.mlp`).
3. **Measure wall time.** Record `t0 = now()` before the kernel launches and
   `t1 = now()` after it completes. Accumulate `t1 - t0` into
   `layer.total_time_us`.
4. **Capture the layer output tensor stats.** At the final residual add of
   each layer, compute the tensor-stats object (§3). Copy the tensor to host
   memory if it lives on an accelerator.
5. **Write the JSON file at shutdown.** Use ordered keys, UTF-8, 2-space
   indentation. llama.cpp registers `std::atexit` for this; a Python reference
   implementation would use `atexit.register(...)`.

See [`common/layer-profile.cpp`](../common/layer-profile.cpp) for a
concrete implementation (≈ 330 lines).

---

## 5. Side-channel: per-layer CSV value dumps

In addition to the JSON profile, the llama.cpp producer can emit a
separate per-layer CSV dump triggered by `--layer-values-csv DIR`. This is
independent of the JSON output: either, both or neither may be enabled on
the same run.

For every transformer layer observed, the producer writes a file
`<DIR>/layer_<il>.csv` (3-digit zero-padded `il`) with exactly two
columns:

```
index,value
17,0.4213
91,-1.8809
...
```

- `index` is the flattened row-major offset into the layer's output
  tensor (`l_out-<il>`).
- `value` is the dequantized `f32` value at that offset.
- Row order matches the order in which indices were drawn from the RNG.

**Sampling procedure.** The producer:

1. Derives a RNG seed from the current local hour
   (`localtime(now).tm_hour`, range `0..23`). This means that two
   profiling runs started in the same clock hour reuse the same sample
   positions, which makes diffing cheap and makes CSV outputs reproducible
   between nearby runs on the same model.
2. Uses the seed to draw a pool of `pool_size = 3000` candidate indices
   uniformly from `[0, n_elements)` with a 64-bit Mersenne Twister.
3. Dedupes the pool in draw order and keeps the first `pick = 1000`
   unique indices. If the pool happens to contain fewer than 1000 unique
   values (extremely rare for any tensor larger than ~10k elements), the
   file is shorter.
4. On every forward pass, re-reads the layer output at those positions
   and overwrites `csv_values`, so the final CSV reflects the last
   observed forward pass.

Consumers that want to cross-check two engines can use the CSV files
directly: given that both engines sampled the same indices (because the
schema below is reproducible across engines that follow it), element-wise
diffs are straightforward.

### 5.1 Cross-engine contract

For another engine to produce a compatible `layer_<il>.csv`:

- It MUST use the same draw procedure: seed = current local hour, RNG =
  `std::mt19937_64`, candidate pool of 3000 uniform integers in
  `[0, n_elements)`, keep first 1000 unique in draw order.
- It MUST use flattened row-major indices into a tensor of the same
  `shape` as the corresponding JSON layer object.
- It MUST emit the header row literally as `index,value` and use a plain
  comma separator with no quoting.
- It SHOULD emit values with enough precision to survive an `f16` round
  trip (≥ 6 significant digits).

Any deviation from the above makes the CSV files non-comparable.

## 6. Versioning

Future non-backwards-compatible changes MUST bump the schema string, e.g.
`layer-profile/v2`. Backwards-compatible additions (new optional fields) do
NOT require a version bump but SHOULD be documented in this file.

Schema version history:

- `layer-profile/v1` — initial release (2026-04-11).
