# How to Run the SVE Kernel A/B Experiment

Runbook for integrating a user-provided SVE kernel, running micro-benchmarks
and live-inference comparisons against the existing llama.cpp/ggml path, and
interpreting the results with variance awareness.

**Prerequisite:** you have filled in the function bodies in
`tools/kernel-bench/kernels/kernels-sve.cpp` (e.g. `kernels_sve::relu_f32`,
`kernels_sve::gelu_f32`, `kernels_sve::silu_f32`). That file ships with empty
placeholders and is not wired into the build by default — Phase 1 below wires
it in.

---

## 0. Which host are you on?

| Host | SVE run? | SVE compile? | What you can do |
|------|----------|--------------|-----------------|
| Apple Silicon (M1/M2/M3) | ❌ (no hardware) | ✅ (Apple clang 17) | Compile-only check + x-compile artifacts |
| x86_64 Linux | ❌ | ❌ (NEON/AVX only) | Skip SVE; run standard vs NEON/AVX variants |
| aarch64 Linux + SVE (Graviton3/4, Neoverse V1/V2, Grace) | ✅ | ✅ | Full A/B loop end-to-end |
| qemu-user-aarch64 + SVE emulation | ✅ (slow) | ✅ (cross) | Full loop for correctness, not perf |

Stop here and pick one. The rest of the runbook works on all four but marks
which steps are skipped on SVE-less hosts.

---

## 1. Wire `kernels-sve.cpp` into the build (one-time)

### 1.1 `tools/kernel-bench/CMakeLists.txt` — add feature-gated source

```cmake
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=armv8-a+sve" KB_HAVE_SVE_COMPILER)

if (KB_HAVE_SVE_COMPILER)
    # Add kernels-sve.cpp to both llama-kernel-bench and llama-inference-bench
    target_sources(llama-kernel-bench    PRIVATE kernels/kernels-sve.cpp)
    target_sources(llama-inference-bench PRIVATE kernels/kernels-sve.cpp)
    set_source_files_properties(kernels/kernels-sve.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8-a+sve")
    target_compile_definitions(llama-kernel-bench    PRIVATE KB_HAVE_SVE=1)
    target_compile_definitions(llama-inference-bench PRIVATE KB_HAVE_SVE=1)
endif()
```

The `set_source_files_properties` call is important: you want `-march=armv8-a+sve`
applied **only to this translation unit** so the rest of the project still
builds with its default ISA level.

### 1.2 `kernel-registry.cpp` — register the SVE variant

Add a header include and extend `create_default()`:

```cpp
#include "kernels/kernels-standard.h"
#include "kernels/kernels-custom.h"
#if defined(KB_HAVE_SVE)
#include "kernels/kernels-sve.h"   // declares kernels_sve::{relu,gelu,silu}_f32
#endif

KernelRegistry KernelRegistry::create_default() {
    KernelRegistry reg;
    // ... existing standard + custom registrations unchanged ...

#if defined(KB_HAVE_SVE)
    reg.register_kernel({"relu", "sve", KernelOpType::UNARY, kernels_sve::relu_f32, nullptr});
    reg.register_kernel({"silu", "sve", KernelOpType::UNARY, kernels_sve::silu_f32, nullptr});
    // gelu slot exists only once you add a gelu entry for "standard"/"custom"
    // in kernels-standard.h/custom.h -- kernel-registry currently has no gelu.
#endif

    return reg;
}
```

> Note: the current registry has no `"gelu"` entry at all. If your experiment
> includes GeLU in micro-bench mode, you must also add a baseline
> `kernel_gelu_standard` in `kernels/kernels-standard.{h,cpp}` first.
> For inference-bench mode the story is different — see Phase 1.4.

### 1.3 Expose the SVE kernels via a header

Create `tools/kernel-bench/kernels/kernels-sve.h` with forward declarations:

```cpp
#pragma once
#include <stddef.h>
namespace kernels_sve {
    void relu_f32(const float * x, float * y, size_t n);
    void gelu_f32(const float * x, float * y, size_t n);
    void silu_f32(const float * x, float * y, size_t n);
}
```

### 1.4 (Inference mode only) Extend the op hook in `inference-bench.cpp`

`custom_tensor_traits::compute_forward` currently dispatches on
`GGML_OP_ADD` / `GGML_OP_MUL`. To route activations through your SVE
kernels, extend the switch (pseudo-diff — adapt to the exact layout of that
file):

```cpp
case GGML_OP_UNARY: {
    switch (ggml_get_unary_op(op)) {
#if defined(KB_HAVE_SVE)
        case GGML_UNARY_OP_RELU:
            kernels_sve::relu_f32(src, dst, n_elements);
            return true;
        case GGML_UNARY_OP_SILU:
            kernels_sve::silu_f32(src, dst, n_elements);
            return true;
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_GELU_ERF:
            kernels_sve::gelu_f32(src, dst, n_elements);
            return true;
#endif
        default: return false;  // fall through to standard path
    }
}
```

The `ith` / `nth` thread split is applied by the caller, so your kernel
receives a contiguous `[0, n)` slice for its partition. SwiGLU needs a
separate case for `GGML_OP_SWIGLU` if the target model uses the fused op —
confirm with the `kernel-dev-perf` orchestrator's analyst phase before
committing to a signature.

---

## 2. Build

### 2.1 SVE-capable host (native, runnable)

```bash
cd /Users/junbongyu/src/sr/llama.cpp
cmake -S . -B build-sve -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-sve --target llama-kernel-bench llama-inference-bench -j
```

Confirm feature detection in the configure output:
```
-- Performing Test KB_HAVE_SVE_COMPILER - Success
-- HAVE_SVE - Success   # also needed from ggml's own probe
```

### 2.2 Cross-compile (aarch64-linux-gnu, not runnable on host)

```bash
cmake -S . -B build-xsve \
      -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build-xsve --target llama-kernel-bench -j
```
(create a toolchain file or pass `-DCMAKE_C/CXX_COMPILER=clang
-DCMAKE_CXX_FLAGS="--target=aarch64-linux-gnu -march=armv8-a+sve"` as
appropriate). Produces aarch64 binaries you can scp to a SVE host.

### 2.3 Apple M1 / M2 / M3 (compile-only evidence)

On M1 the project-level `HAVE_SVE` CMake test fails (the probe links and runs
a program, which SIGILLs on non-SVE hardware), so `KB_HAVE_SVE_COMPILER`
above will succeed but you likely still want to verify the single TU
compiles:

```bash
clang -march=armv8-a+sve -O2 \
      -c tools/kernel-bench/kernels/kernels-sve.cpp -o /tmp/kernels-sve.o
```
Expect a ~700 B object with `kernels_sve::relu_f32` etc. symbols. Running
the binary on M1 produces SIGILL — that is expected, stop here and use the
cross-compile path or move to an SVE host before Phase 3.

---

## 3. Correctness smoke test

Always correctness before performance. `bench-harness` already compares each
kernel's output to the `"standard"` baseline inside `llama-kernel-bench`.

```bash
./build-sve/bin/llama-kernel-bench --op relu --variant sve --output /tmp/sve_smoke.json
python3 scripts/compare-kernels.py /tmp/sve_smoke.json --op relu
```

Look for `OK` in the `correctness` column. If `FAIL`:
- check the tolerance policy (bit-exact for ReLU, `< 1e-4` relative for SiLU/GeLU);
- suspect uninitialized tail lanes from your SVE `svst1_f32` store when `n` is
  not a multiple of the vector length — the predicated `svwhilelt_b32` must
  guard the tail or you'll write garbage past `n`.

Do this on at least 3 different `--size` values (the micro-bench sweeps a
default set) before moving on.

---

## 4. Performance A/B with `run-ab-bench.py`

`scripts/compare-kernels.py` only reads one or two JSON files and has no
multi-seed aggregation. For a noise-aware verdict use the dedicated
orchestrator:

### 4.1 Micro-bench mode (per-kernel)

```bash
python3 scripts/run-ab-bench.py kernel \
    --bin build-sve/bin/llama-kernel-bench \
    --variants standard sve \
    --ops relu silu \
    --seeds 5 \
    --min-time 1.0 \
    --out-dir /tmp/ab_sve_kernel
```

Output:
- `/tmp/ab_sve_kernel/kernel_seed{1..5}.json` — raw runs
- `/tmp/ab_sve_kernel/summary.json` — aggregated
- stdout table with:
  - per-variant `mean / std / cv% / correct` per `(op, size)`
  - speedup of `sve` vs baseline `standard`
  - one of four verdicts:
    - `IMPROVED` — speedup above threshold and outside noise band
    - `REGRESSED` — speedup below `1 - threshold`, outside noise
    - `NOISE` — within the noise floor, no claim to make
    - `NOISY_MEASUREMENT` — CV > 5% on either side, redo with `--min-time`
      bumped or more `--seeds`

Default threshold is 3% speedup with `> 2 × max(cv_baseline, cv_user)` spread
required before flagging improvement or regression.

### 4.2 Inference mode (end-to-end model)

Only meaningful if you completed Phase 1.4 (UNARY hook extension):

```bash
python3 scripts/run-ab-bench.py inference \
    --bin build-sve/bin/llama-inference-bench \
    --model models/Qwen3-0.6B-Q8_0.gguf \
    --pp 256 --tg 64 --threads $(nproc) \
    --seeds 3 \
    --out-dir /tmp/ab_sve_infer
```

Output:
- Model-level `pp / tg tokens/s` (standard vs custom) with stddev + verdict
- Per-op totals with stddev + speedup verdict

The inference binary's own JSON already contains a baseline-vs-custom pair
inside a single run, so seeds here average those same-process pairs for
noise isolation rather than fresh-process variance.

### 4.3 Threshold tuning

If your kernel is a small win (e.g. 1–2%) and you need higher confidence,
bump `--seeds` to 10+ and drop `--threshold` to `0.01`. The verdict is
**deliberately conservative**: it won't call an `IMPROVED` on sub-noise
wins, because publishing those tends to degrade trust in later runs.

---

## 5. Interpreting the numbers

1. **Correctness first.** `NOISE`/`NOISY_MEASUREMENT` verdicts are moot if
   `correct` column has any `FAIL` — fix the kernel before reading speedups.
2. **Op-level speedup vs model-level speedup.** Activations are usually 5–15%
   of total inference time. A 2× ReLU speedup may only move pp/tg by a few
   percent; check both numbers and report both. The inference-mode report
   shows this explicitly.
3. **Size-dependent wins.** SVE typically helps large tensors (≥ 64K
   elements). Small sizes often regress because of vector setup overhead;
   this is normal and usually fine because small activations contribute
   little to total time. Check the size column before declaring a regression.
4. **GeLU variant matters.** If your SVE GeLU implements the exact (erf)
   form but the model uses `GGML_UNARY_OP_GELU_QUICK`, correctness will
   appear to fail (max_rel_error ≈ 5e-3). Confirm the variant with a
   stderr dump on the first intercept; pick one GeLU and stick with it.

---

## 6. Known limits on this repo state

- `kernel-registry` currently has no `"gelu"` baseline. Micro-bench GeLU
  needs a standard implementation added in `kernels-standard.{h,cpp}` first.
  Inference-bench mode bypasses this because it hooks the live ggml path.
- `inference-bench.cpp` hooks only `GGML_OP_ADD` / `GGML_OP_MUL` in the
  shipped branch. Phase 1.4 above must be applied before inference mode
  will route through SVE.
- SwiGLU on QWen3 may be fused (`GGML_OP_SWIGLU`) or composite (`SILU*MUL`)
  depending on the ggml version linked. Trace once with the
  `kernel-dev-perf` orchestrator's analyst phase before investing in
  the fused-op path.
- Apple M1/M2/M3 cannot execute SVE binaries. Treat M-series as a
  compile-check environment and move real A/B runs to Graviton3/4 or
  qemu-user.

---

## 7. Script reference

| Script | Purpose |
|--------|---------|
| `scripts/compare-kernels.py` | Ad-hoc single/pair analysis of one or two `kernel-bench` JSON files. No multi-seed aggregation. |
| `scripts/extract-backend-results.py` | Same role for `backend-bench` JSON. Not used in this runbook. |
| `scripts/run-ab-bench.py` | **This runbook's driver.** Multi-seed, variance-aware A/B orchestration for both `kernel-bench` and `inference-bench`. |
