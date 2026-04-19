# Custom-Op Backend — 테스트 및 검증 방법

## 개요

이 문서는 커스텀 백엔드가 정상적으로 동작하는지 검증하는 방법을 설명한다. 검증은 3가지 레벨로 진행한다:

1. **로딩 검증** — 백엔드가 정상적으로 로드/등록되는지 확인
2. **커널 정합성 검증** — 커스텀 커널의 연산 결과가 CPU 백엔드와 일치하는지 확인
3. **성능 비교** — CPU 백엔드 대비 처리 속도 측정

---

## 1. 빌드 확인

먼저 빌드가 성공했는지 확인:

```bash
cd /path/to/llama.cpp
cmake -B build -DGGML_CUSTOM_OP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-custom-op -j$(nproc)

# 라이브러리가 생성되었는지 확인
ls -la build/src/ggml-custom-op*        # 정적 라이브러리
ls -la build/bin/libggml-custom-op*     # 동적 라이브러리 (GGML_BACKEND_DL=ON인 경우)
```

### 예상 결과

```
build/src/libggml-custom-op.a          # 정적 라이브러리
build/bin/libggml-custom-op.so         # Linux 동적 라이브러리
# 또는
build/bin/libggml-custom-op.dylib      # macOS 동적 라이브러리
```

---

## 2. 백엔드 로딩 검증

### 2.1 정적 로딩 테스트

```c
// test_load.c — 정적 링크로 백엔드 로딩 확인
#include <stdio.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-custom-op.h"

int main(void) {
    // 1. 백엔드 레지스트리에서 검색
    ggml_backend_load_all();

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("custom-op");
    if (reg == NULL) {
        printf("FAIL: custom-op backend not found in registry\n");
        printf("Available backends:\n");
        for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
            ggml_backend_reg_t r = ggml_backend_reg_get(i);
            printf("  - %s\n", ggml_backend_reg_name(r));
        }
        return 1;
    }
    printf("PASS: custom-op backend found: %s\n", ggml_backend_reg_name(reg));

    // 2. 디바이스 확인
    size_t dev_count = ggml_backend_reg_dev_count(reg);
    printf("  Devices: %zu\n", dev_count);
    if (dev_count < 1) {
        printf("FAIL: no devices found\n");
        return 1;
    }

    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    printf("  Device name: %s\n", ggml_backend_dev_name(dev));
    printf("  Device description: %s\n", ggml_backend_dev_description(dev));

    // 3. 디바이스 타입이 ACCEL인지 확인
    enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
    if (dev_type != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        printf("FAIL: expected ACCEL type, got %d\n", dev_type);
        return 1;
    }
    printf("  Device type: ACCEL (correct)\n");

    // 4. 백엔드 초기화
    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (backend == NULL) {
        printf("FAIL: backend init failed\n");
        return 1;
    }
    printf("  Backend name: %s\n", ggml_backend_name(backend));
    printf("PASS: backend initialized successfully\n");

    // 5. 커널 등록 (아직 아무 커널도 없음)
    bool is_custom = ggml_backend_is_custom_op(backend);
    printf("  Is custom-op backend: %s\n", is_custom ? "yes" : "no");

    ggml_backend_free(backend);
    return 0;
}
```

빌드 및 실행:
```bash
gcc test_load.c -o test_load -I../ggml/include -L./build/src -lggml -lggml-custom-op -lpthread -lm
./test_load
```

### 2.2 동적 로딩 테스트

```bash
# 런타임에 .so/.dylib을 로딩하는지 확인
GGML_BACKEND_PATH=./build/bin/libggml-custom-op.so ./my_app
```

`ggml_backend_load_all()`을 호출한 후 `ggml_backend_reg_by_name("custom-op")`이 `NULL`이 아니면 성공.

---

## 3. 커널 정합성 검증

커스텀 커널이 CPU 백엔드와 동일한 결과를 내는지 확인하는 것이 가장 중요한 테스트다.

### 3.1 정합성 테스트 원칙

```
입력 텐서 (동일한 난수 seed)
    │
    ├──→ CPU 백엔드 ──→ 결과_A
    │
    └──→ Custom-Op 백엔드 ──→ 결과_B

max |결과_A - 결과_B| < tolerance (엡실론)
```

### 3.2 정합성 테스트 코드

```c
// test_correctness.c — MUL_MAT 정합성 검증
#include <stdio.h>
#include <math.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-custom-op.h"

#define TOLERANCE 1e-5f
#define MATRIX_SIZE 64

static bool naive_mul_mat_can_handle(const struct ggml_tensor * op) {
    return op->src[0]->type == GGML_TYPE_F32
        && op->src[1]->type == GGML_TYPE_F32
        && ggml_is_contiguous(op->src[0])
        && ggml_is_contiguous(op->src[1]);
}

static bool naive_mul_mat_compute(const struct ggml_compute_params * params,
                                  struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = ne01 > 0 ? dst->ne[1] : 0;
    const int ith = params->ith;
    const int nth = params->nth;
    const int64_t dr = (ne11 + nth - 1) / nth;
    const int64_t ir0 = ith * dr;
    const int64_t ir1 = (ir0 + dr) < ne11 ? (ir0 + dr) : ne11;

    for (int64_t ir = ir0; ir < ir1; ir++) {
        for (int64_t ic = 0; ic < ne01; ic++) {
            float sum = 0.0f;
            for (int64_t ik = 0; ik < ne00; ik++) {
                sum += ((const float *)src0->data)[ic * ne00 + ik]
                     * ((const float *)src1->data)[ir * ne10 + ik];
            }
            ((float *)dst->data)[ir * ne01 + ic] = sum;
        }
    }
    return true;
}

static float max_diff(const float * a, const float * b, int n) {
    float md = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

int main(void) {
    printf("=== MUL_MAT Correctness Test ===\n\n");

    // 커널 등록
    ggml_backend_custom_op_register_kernel(
        GGML_OP_MUL_MAT, "naive_mul_mat_f32",
        naive_mul_mat_compute, naive_mul_mat_can_handle
    );

    ggml_backend_load_all();
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    ggml_backend_t custom_backend = ggml_backend_custom_op_init();
    ggml_backend_custom_op_set_n_threads(custom_backend, 1);

    printf("CPU backend:     %s\n", ggml_backend_name(cpu_backend));
    printf("Custom backend:  %s\n\n", ggml_backend_name(custom_backend));

    int passed = 0, failed = 0;

    // 다양한 행렬 크기로 테스트
    int sizes[] = {16, 32, 64, 128, 256};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < n_sizes; si++) {
        int M = sizes[si];
        int K = sizes[si];
        int N = sizes[si];
        char test_name[64];
        snprintf(test_name, sizeof(test_name), "MUL_MAT %dx%dx%d", M, K, N);

        // 그래프 구성: dst = src0^T @ src1
        size_t ctx_size = ggml_tensor_overhead() * 4 + 1024 * 1024;
        struct ggml_init_params ip = {
            .mem_size = ctx_size,
            .mem_buffer = NULL,
            .no_alloc = true,
        };

        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);

        // CPU에서 실행
        ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
        float * s0 = (float *) ((char *) ggml_backend_buffer_get_base(buf_cpu));
        float * s1 = s0 + (size_t)(M * K);

        // 난수 초기화 (결정적)
        unsigned int seed = 42 + si;
        for (int i = 0; i < M * K; i++) {
            seed = seed * 1103515245 + 12345;
            s0[i] = ((float)((seed >> 16) & 0x7fff) / 32768.0f - 0.5f);
        }
        seed = 123 + si;
        for (int i = 0; i < K * N; i++) {
            seed = seed * 1103515245 + 12345;
            s1[i] = ((float)((seed >> 16) & 0x7fff) / 32768.0f - 0.5f);
        }

        struct ggml_cgraph * cg_cpu = ggml_new_graph(ctx);
        ggml_build_forward_expand(cg_cpu, dst);
        ggml_backend_graph_compute(cpu_backend, cg_cpu);

        // 결과 복사
        int n_out = M * N;
        float * result_cpu = (float *)malloc(sizeof(float) * n_out);
        // GPU 백엔드가 아니면 dst->data에 직접 접근 가능
        memcpy(result_cpu, dst->data, sizeof(float) * n_out);

        // Custom-Op 백엔드에서 실행
        ggml_backend_buffer_t buf_custom = ggml_backend_alloc_ctx_tensors(ctx, custom_backend);
        // 데이터 복사 (CPU → Custom)
        ggml_backend_tensor_copy(src0, src0);
        ggml_backend_tensor_copy(src1, src1);

        struct ggml_cgraph * cg_custom = ggml_new_graph_custom(ctx, cg_cpu->size, false);
        ggml_build_forward_expand(cg_custom, dst);
        ggml_backend_graph_compute(custom_backend, cg_custom);

        float * result_custom = (float *)malloc(sizeof(float) * n_out);
        memcpy(result_custom, dst->data, sizeof(float) * n_out);

        // 비교
        float md = max_diff(result_cpu, result_custom, n_out);
        if (md < TOLERANCE) {
            printf("  PASS: %-25s max_diff=%.2e\n", test_name, md);
            passed++;
        } else {
            printf("  FAIL: %-25s max_diff=%.2e (threshold=%.2e)\n",
                   test_name, md, TOLERANCE);
            failed++;
        }

        free(result_cpu);
        free(result_custom);
        ggml_backend_buffer_free(buf_cpu);
        ggml_backend_buffer_free(buf_custom);
        ggml_free(ctx);
    }

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);

    ggml_backend_free(cpu_backend);
    ggml_backend_free(custom_backend);

    return failed > 0 ? 1 : 0;
}
```

### 3.3 기존 test-backend-ops 활용

llama.cpp에 이미 `tests/test-backend-ops.cpp`가 있어 백엔드 간 정합성 비교를 지원한다:

```bash
# CPU vs Custom-Op 백엔드 비교
./build/bin/test-backend-ops -b1 CPU -b2 Custom-Op -o mul_mat
```

---

## 4. 성능 벤치마크

### 4.1 벤치마크 데모 실행

```bash
cmake -B build -DGGML_CUSTOM_OP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark-custom-op -j$(nproc)

# 기본 (512x512 MUL_MAT, 10회 반복)
./build/bin/benchmark-custom-op

# 사용자 지정 행렬 크기 및 반복 횟수
./build/bin/benchmark-custom-op 1024 1024 1024 20
```

### 4.2 벤치마크 출력 예시

```
=== Custom Op Backend Benchmark ===
Matrix dimensions: M=1024, N=1024, K=1024
Iterations: 20

CPU backend: CPU
Custom backend: Custom-Op

Registered custom MUL_MAT kernel (naive_matmul_f32)

Running CPU backend benchmark...
  CPU: 45.234 ms/iter

Running Custom-Op backend benchmark...
  Custom-Op: 62.871 ms/iter

  Speedup: 0.72x (custom vs CPU)
```

> naive 구현은 CPU의 최적화된 BLAS 커널보다 느릴 수 있다.
> 이것이 정상적인 동작이다 — 커스텀 커널은 **최적화된 구현으로 교체**하기 위한 것이다.

### 4.3 llama-bench를 사용한 전체 모델 벤치마크

커스텀 백엔드가 실제 모델 추론에 미치는 영향을 측정:

```bash
# CPU 전용 (baseline)
./build/bin/llama-bench -m /path/to/model.gguf -p 512 -ngl 0

# 커스텀 백엔드 포함 (MUL_MAT 커널이 교체됨)
# 이는 스케줄러가 자동으로 ACCEL 백엔드에 MUL_MAT를 할당함
```

---

## 5. 디버깅

### 5.1 백엔드 로딩 디버그

```bash
# 디버그 모드 빌드
cmake -B build -DGGML_CUSTOM_OP=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# 실행 시 백엔드 로딩 로그 확인
GGML_LOG_LEVEL=debug ./build/bin/benchmark-custom-op
```

NDEBUG가 정의되지 않은 디버그 빌드에서는 `ggml_backend_reg_by_name("custom-op")` 호출 시 로깅이 활성화된다.

### 5.2 커널 등록 확인

```c
// 등록 직후 확인
ggml_backend_custom_op_register_kernel(GGML_OP_MUL_MAT, "my_kernel", compute_fn, can_handle_fn);
printf("Kernel registered for GGML_OP_MUL_MAT\n");
```

백엔드 초기화 후 `supports_op()`이 제대로 동작하는지 확인:
```c
ggml_backend_t backend = ggml_backend_custom_op_init();
// 백엔드 초기화 후에야 supports_op()이 등록된 커널을 참조한다
```

### 5.3 스케줄러 확인

특정 Op이 어떤 백엔드에 할당되었는지 확인하려면, `ggml_backend_sched`를 직접 구성:

```c
// 스케줄러에 백엔드 순서 명시
ggml_backend_t backends[] = { custom_backend, cpu_backend };
ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, GGML_DEFAULT_GRAPH_SIZE, false);

// 그래프 할당 전에 어떤 백엔드가 어떤 Op를 처리할지 확인
// ggml_backend_sched_graph_compute() 호출 시 내부적으로 할당
```

---

## 6. 다른 백엔드와의 비교 테스트

### 6.1 CPU vs Custom-Op vs CUDA

```c
ggml_backend_t backends[3];
backends[0] = cuda_backend;   // 우선순위 최고
backends[1] = custom_backend; // ACCEL (CPU보다 우선)
backends[2] = cpu_backend;    // 마지막 수단

ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 3, ...);
```

스케줄러의 동작:
- `supports_op()`이 true인 가장 높은 우선순위 백엔드가 해당 Op를 처리
- Custom-Op 백엔드가 MUL_MAT을 등록하면: CUDA가 있으면 CUDA가 처리, 없으면 Custom-Op가 처리, 둘 다 없으면 CPU가 처리
- `offload_op()`이 true이면: 가중치가 호스트 메모리에 있어도 Custom-Op가 처리

### 6.2 의도적 커널 교체 테스트

특정 Op만 다르게 처리하고 싶을 때:

```c
// ADD만 커스텀 커널로 처리, MUL_MAT은 CPU가 처리
ggml_backend_custom_op_register_kernel(
    GGML_OP_ADD,
    "my_add_kernel",
    my_add_compute,
    my_add_can_handle
);
// MUL_MAT은 등록하지 않음 → Custom-Op 백엔드가 supports_op()에서 false 반환 → CPU가 처리
```

---

## 7. 테스트 체크리스트

빌드 후 다음 항목을 순서대로 확인:

| # | 테스트 | 확인 방법 | 기대 결과 |
|---|--------|-----------|----------|
| 1 | 라이브러리 빌드 | `ls build/src/libggml-custom-op.a` | 파일 존재 |
| 2 | 백엔드 등록 | `ggml_backend_reg_by_name("custom-op") != NULL` | non-NULL |
| 3 | 디바이스 타입 | `ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL` | ACCEL |
| 4 | 버퍼 타입 | `ggml_backend_buft_is_host(custom_buft) == true` | true |
| 5 | 커널 등록 | `register_kernel()` 반환값 | true |
| 6 | supports_op (등록된 Op) | `ggml_backend_dev_supports_op(dev, mul_mat_tensor)` | true |
| 7 | supports_op (미등록 Op) | `ggml_backend_dev_supports_op(dev, add_tensor)` | false |
| 8 | offload_op | `ggml_backend_dev_offload_op(dev, mul_mat_tensor)` | true |
| 9 | 그래프 실행 | `ggml_backend_graph_compute()` | GGML_STATUS_SUCCESS |
| 10 | 정합성 | CPU 결과 vs Custom-Op 결과의 max diff | < 1e-5 |

---

## 8. 알려진 제한사항

1. **GPU 백엔드 우선순위**: CUDA/Metal 백엔드가 등록된 Op에 대해서는 항상 GPU 백엔드가 우선순위를 가진다. 커스텀 백엔드는 GPU가 지원하지 않는 Op에 대해서만 처리한다.

2. **단일 커널 per Op**: 현재 구현에서는 각 Op 타입당 하나의 커널만 등록할 수 있다. 여러 커널을 등록하면 마지막 커널이 우선이다 (first-match가 아님).

3. **F16/양자화 타입**: 벤치마크 데모의 naive MUL_MAT 커널은 F32만 지원한다. F16이나 양자화 타입을 처리하려면 커널 내부에서 타입 변환을 구현해야 한다.

4. **다중 백엔드 인스턴스**: 현재 글로벌 컨텍스트 포인터(`g_custom_op_ctx`)를 사용하므로, 백엔드 인스턴스는 하나만 생성해야 한다. 여러 인스턴스를 생성하면 마지막 인스턴스의 컨텍스트만 사용된다.

5. **동적 로딩 시 시드**: `ggml_backend_load_all()`을 호출하기 전에 `register_kernel()`을 호출하면, 커널이 보류 목록에 저장되고 백엔드 초기화 시 컨텍스트로 이동한다. 초기화 후에 `register_kernel()`을 호출하면 직접 컨텍스트에 추가된다.

---

## 9. 레이어별 연산시간 벤치마크

`benchmark-layer-timing`은 실제 모델을 실행하면서 각 레이어의 Op별 연산 시간을 측정하는 도구다. `ggml_backend_sched_eval_callback`을 사용하여 그래프의 모든 노드에 대해 전후 `ggml_backend_synchronize()`를 호출하고 마이크로초 단위로 타이밍을 수집한다.

### 9.1 빌드

```bash
cmake -B build -DGGML_CUSTOM_OP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark-layer-timing -j$(nproc)
```

### 9.2 실행

```bash
# 기본: 모델 로드 후 프롬프트/토큰 생성 타이밍 측정
./build/bin/benchmark-layer-timing -m models/llama-2-7b.Q4_K_M.gguf -p "Hello world"

# 웜업 2회, 반복 3회 측정
./build/bin/benchmark-layer-timing -m models/llama-2-7b.Q4_K_M.gguf -p "Hello world" --warmup 2 --repeat 3

# 생성 토큰 수 지정
./build/bin/benchmark-layer-timing -m models/llama-2-7b.Q4_K_M.gguf -p "Hello world" -n 64
```

### 9.3 출력 형식

```
=== Layer Timing Benchmark ===
Model: models/llama-2-7b.Q4_K_M.gguf
Warmup: 1, Repeat: 1
n_gpu_layers: 99

Model info:
  n_ctx_train: 4096
  n_layer:     32
  n_embd:      4096
  n_vocab:     32000

Custom-Op backend: detected (device: Custom-Op)   # 또는 "not loaded"

Active backends:
  [0] CPU (CPU)
  [1] Custom-Op (Accelerator)

=== Per-Layer Timing (Prompt Eval Run 1) ===
Prompt tokens: 4

Layer       Time (ms)  Op breakdown
-----       ---------  ------------------
    0      2.341 ms  ( 5.1%)  attn_norm: 0.12ms, attn_q: 0.89ms, attn_k: 0.31ms, attn_v: 0.28ms
    1      2.278 ms  ( 4.9%)  attn_norm: 0.11ms, attn_q: 0.87ms, attn_k: 0.30ms, attn_v: 0.27ms
  ...
   31      2.412 ms  ( 5.2%)  attn_norm: 0.13ms, attn_q: 0.91ms, attn_k: 0.32ms, attn_v: 0.29ms

--- Summary ---
Total layer compute: 78.234 ms
Avg per layer:       2.445 ms
Slowest layer:       3 (3.12 ms)
Fastest layer:       15 (1.98 ms)

--- Global Op Distribution ---
Op                     Time (ms)    Count % of Total
--                     --------    ----- ---------
MUL_MAT                 48.123        96    61.5%
RMS_NORM                 4.234        64     5.4%
SILU                     2.891        32     3.7%
...

=== Per-Layer Timing (Token Gen Run 1) ===
Generated tokens: 32
...
```

### 9.4 작동 원리

1. **`llama_context_params.cb_eval`** 설정: `ggml_backend_sched_eval_callback`을 통해 스케줄러가 모든 그래프 노드에 대해 콜백을 호출
2. **노드 이름 파싱**: 각 노드의 이름(예: `"attn_q-7"`)에서 레이어 인덱스(`7`)와 Op 이름(`"attn_q"`)을 추출
3. **시간 누적**: `ask=true` (관찰 여부 질문) → 항상 `true` 반환, `ask=false` (계산 완료 통지) → 이전 노드 이후 경과 시간을 누적
4. **백엔드 동기화**: 콜백이 설정되면 스케줄러가 각 노드 계산 후 `ggml_backend_synchronize()`를 호출하므로 정확한 타이밍 보장

### 9.5 커스텀 백엔드와의 연동

Custom-Op 백엔드가 MUL_MAT 커널을 등록하면, 스케줄러가 해당 Op를 Custom-Op 백엔드에 할당한다. 레이어 타이밍 출력에서 MUL_MAT Op의 시간이 Custom-Op 백엔드에서 처리된 것임을 확인할 수 있다.

현재 구현에서는 백엔드별 시간 구분이 출력에 포함되지 않지만, `eval_callback`에서 `ggml_backend_get_name()`을 호출하여 노드가 어느 백엔드에서 처리되었는지 확인할 수 있다 (스케줄러의 노드-백엔드 매핑 필요).

### 9.6 활용 시나리오

1. **병목 레이어 식별**: 어떤 레이어의 어떤 Op가 전체 추론 시간의 대부분을 차지하는지 파악
2. **커스텀 커널 효과 측정**: MUL_MAT 커널을 교체하기 전후로 레이어 타이밍을 비교
3. **백엔드 오프로딩 튜닝**: `n_gpu_layers` 값을 조정하면서 CPU/GPU 분산 효과 분석
4. **양자화 영향 분석**: Q4_K_M vs Q8_0 등 양자화 포맷별 레이어 성능 비교