# How to Run Kernel Benchmark Experiments

Guide for running the ggml kernel swap benchmark experiments with `llama-kernel-bench` and `llama-inference-bench`.

## 1. Build

```bash
cd /Users/junbongyu/src/SR/NNT_LMA/llama.cpp

# First-time CMake configure
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build both benchmark tools
cmake --build . --target llama-kernel-bench llama-inference-bench -j$(sysctl -n hw.ncpu)
```

Binaries land in `build/bin/`.

## 2. Micro-benchmark (isolated kernels)

Runs each registered kernel variant (standard vs custom) on several tensor sizes and compares timing.

```bash
cd /Users/junbongyu/src/SR/NNT_LMA/llama.cpp

# All 6 ops x all sizes, JSON output
./build/bin/llama-kernel-bench --output /tmp/kernel-bench.json

# Filter single op
./build/bin/llama-kernel-bench --op mul_mat --min-time 1.0 --output /tmp/mm.json

# Analyze
python3 scripts/compare-kernels.py /tmp/kernel-bench.json
python3 scripts/compare-kernels.py /tmp/kernel-bench.json --op add
python3 scripts/compare-kernels.py /tmp/kernel-bench.json --format csv > results.csv
```

### CLI options (`llama-kernel-bench`)

| Option | Description | Default |
|--------|-------------|---------|
| `--op <name>` | Filter by op (add, mul, relu, sigmoid, silu, mul_mat) | all |
| `--variant <name>` | Filter by variant (standard, custom) | all |
| `--output <file>` | Write JSON to file | stdout |
| `--min-time <sec>` | Minimum benchmark time per kernel | 1.0 |
| `--warmup <n>` | Warmup iterations | 3 |

## 3. Live inference benchmark (kernel swap during QWen3 inference)

Hooks into ggml's `extra_buffer_type` mechanism to intercept `ADD`/`MUL` ops during actual model inference. Runs inference twice (standard then custom) and reports per-op timing + overall t/s.

### Download model (first time only)

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q8_0.gguf',
               local_dir='models')"
```

### Run

```bash
./build/bin/llama-inference-bench \
    -m models/Qwen3-0.6B-Q8_0.gguf \
    -pp 256 -tg 64 -t 4 -r 3 \
    -o /tmp/inference-bench.json
```

### CLI options (`llama-inference-bench`)

| Option | Description | Default |
|--------|-------------|---------|
| `-m <path>` | Model file (required) | - |
| `-pp <n>` | Prompt processing tokens | 256 |
| `-tg <n>` | Token generation count | 64 |
| `-t <n>` | Threads | 4 |
| `-r <n>` | Repetitions | 1 |
| `-o <file>` | JSON output file | - |

## 4. Experiment workflow (before/after optimization)

```bash
# Step 1: Baseline (before optimization)
./build/bin/llama-kernel-bench --output before.json
./build/bin/llama-inference-bench -m models/Qwen3-0.6B-Q8_0.gguf -r 3 -o infer-before.json

# Step 2: Edit tools/kernel-bench/kernels/kernels-custom.cpp
#         Replace function bodies with your SIMD / OpenMP / optimized implementations

# Step 3: Rebuild
cmake --build build --target llama-kernel-bench llama-inference-bench -j$(sysctl -n hw.ncpu)

# Step 4: Re-run
./build/bin/llama-kernel-bench --output after.json
./build/bin/llama-inference-bench -m models/Qwen3-0.6B-Q8_0.gguf -r 3 -o infer-after.json

# Step 5: Compare before vs after
python3 scripts/compare-kernels.py before.json after.json
```

## 5. Adding a custom kernel

Edit `tools/kernel-bench/kernels/kernels-custom.cpp`. Example (ARM NEON ADD):

```cpp
#include <arm_neon.h>

void kernel_add_custom(float * dst, const float * src0, const float * src1, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a = vld1q_f32(src0 + i);
        float32x4_t b = vld1q_f32(src1 + i);
        vst1q_f32(dst + i, vaddq_f32(a, b));
    }
    for (; i < n; i++) { dst[i] = src0[i] + src1[i]; }
}
```

Rebuild and re-run. The JSON output auto-computes `speedup = standard_avg / custom_avg`.

To add a new variant alongside standard/custom, register it in `kernel-registry.cpp`:

```cpp
reg.register_kernel({"add", "avx2", KernelOpType::BINARY, kernel_add_avx2, nullptr});
```

## 6. Output formats

### `llama-kernel-bench` JSON

```json
{
  "framework_version": "1.0.0",
  "timestamp": "...",
  "system_info": { "os": "...", "arch": "..." },
  "results": [
    { "op": "add", "variant": "standard", "size": "1048576",
      "avg_time_us": 137.4, "bandwidth_gb_s": 91.6, "correctness": true, ... }
  ],
  "comparisons": [
    { "op": "add", "size": "1048576", "baseline": "standard", "contender": "custom",
      "speedup": 1.35, ... }
  ]
}
```

### `llama-inference-bench` JSON

```json
{
  "model": "models/Qwen3-0.6B-Q8_0.gguf",
  "config": { "pp": 256, "tg": 64, "threads": 4, "reps": 3 },
  "inference": {
    "standard": { "pp_tokens_s": 819.4, "tg_tokens_s": 74.4 },
    "custom":   { "pp_tokens_s": 777.8, "tg_tokens_s": 74.3 },
    "speedup":  { "pp": 0.95, "tg": 1.00 }
  },
  "per_op": [
    { "op": "add", "calls": 10920, "std_total_us": 5359.7,
      "custom_total_us": 5558.9, "speedup": 0.96 }
  ]
}
```

## 7. Notes

- The inference-bench uses `-ngl 0` (CPU-only) so the hook can intercept ops; Metal-offloaded ops bypass the CPU path and won't be measured.
- The hook only intercepts F32 contiguous tensors with matching shapes (no broadcasting). Other cases fall through to the standard ggml path.
- Hook registration happens **after** model loading to avoid interfering with backend device initialization.
- For fair comparison, custom kernels split work across threads using `ith`/`nth` from `ggml_compute_params`.
