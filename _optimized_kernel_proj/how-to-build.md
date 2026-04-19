# Custom-Op Backend — 빌드 방법

## 개요

`ggml-custom-op` 백엔드는 GGML의 동적 백엔드 아키텍처를 활용하여, **코어 코드 수정 없이** 커스텀 Op 커널을 런타임에 주입할 수 있는 플러그인 백엔드다.

**디바이스 타입**: `GGML_BACKEND_DEVICE_TYPE_ACCEL` — GPU가 없는 환경에서 CPU보다 우선순위가 높아, 등록된 Op를 자동으로 가로챈다.

**버퍼**: CPU 버퍼 재사용 — 별도 메모리 관리 없이 호스트 메모리에서 직접 연산.

---

## 1. 사전 요구사항

- CMake ≥ 3.14
- C/C++17 컴파일러 (GCC ≥ 9, Clang ≥ 10, MSVC ≥ 2019)
- llama.cpp 저장소 (이 디렉토리가 `llama.cpp/` 안에 위치)

---

## 2. 정적 링크 빌드 (권장)

백엔드를 메인 라이브러리에 정적으로 포함시키는 방식.

```bash
cd /path/to/llama.cpp

cmake -B build \
  -DGGML_CUSTOM_OP=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

이렇게 하면:
- `libggml.a` (또는 `ggml.lib`)에 custom-op 백엔드가 포함됨
- `libggml-custom-op.a` (또는 `ggml-custom-op.lib`) 별도 라이브러리도 생성됨
- `benchmark-custom-op` 실행 파일이 `build/bin/` 아래에 생성됨

### 빌드 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `GGML_CUSTOM_OP` | `OFF` | custom-op 백엔드 활성화 |
| `GGML_BACKEND_DL` | `OFF` | 동적 백엔드 로딩 활성화 (동적 로딩 시 필요) |
| `CMAKE_BUILD_TYPE` | — | `Release` 권장 (벤치마크 시) |

---

## 3. 동적 로딩 빌드 (런타임 주입)

백엔드를 `.so`/`.dylib`/`.dll` 형태로 빌드하여 런타임에 로딩.

```bash
cmake -B build \
  -DGGML_CUSTOM_OP=ON \
  -DGGML_BACKEND_DL=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

동적 라이브러리는 빌드 출력 디렉토리에 생성됨:
- Linux: `build/bin/libggml-custom-op.so`
- macOS: `build/bin/libggml-custom-op.dylib`
- Windows: `build/bin/ggml-custom-op.dll`

### 런타임 로딩 방법

**방법 A: 환경변수**
```bash
export GGML_BACKEND_PATH=/path/to/libggml-custom-op.so
./my-app
```

**방법 B: 코드에서 로딩**
```c
ggml_backend_reg_t reg = ggml_backend_load("/path/to/libggml-custom-op.so");
// 이후 ggml_backend_load_all() 호출 시 자동 등록됨
```

**방법 C: `ggml_backend_load_all()`에 의한 자동 발견**

> ⚠️ 현재 `ggml_backend_load_all()`은 하드코딩된 백엔드 목록만 검색한다.
> `custom-op`은 이 목록에 없으므로, 방법 A 또는 B를 사용해야 한다.
> `ggml_backend_load_all()`에 추가하려면 `ggml-backend-reg.cpp`의
> `ggml_backend_load_all_from_path()` 함수에 로딩 코드를 추가해야 한다.

---

## 4. 벤치마크 빌드

### 4.1 커스텀 Op 벤치마크 (GGML 그래프 단위)

`examples/custom-op/benchmark-custom-op.cpp`는 커스텀 백엔드와 CPU 백엔드의 MUL_MAT 성능을 비교하는 데모다.

```bash
cmake -B build -DGGML_CUSTOM_OP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark-custom-op -j$(nproc)
```

실행:
```bash
# 기본: 512x512 행렬, 10회 반복
./build/bin/benchmark-custom-op

# 사용자 지정: M=1024, N=1024, K=1024, 20회 반복
./build/bin/benchmark-custom-op 1024 1024 1024 20
```

### 4.2 레이어별 연산시간 벤치마크 (실제 모델)

`examples/custom-op/benchmark-layer-timing.cpp`는 실제 모델을 로드하여 레이어별/Op별 연산 시간을 측정한다.

```bash
cmake -B build -DGGML_CUSTOM_OP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark-layer-timing -j$(nproc)
```

실행:
```bash
./build/bin/benchmark-layer-timing -m models/llama-2-7b.Q4_K_M.gguf -p "Hello world" -n 32
./build/bin/benchmark-layer-timing -m models/qwen2-1.5b.Q4_K_M.gguf --warmup 1 --repeat 3
```

---

## 5. 다른 Op 커널 추가하기

새로운 Op 커널을 추가하려면:

### 5.1 커널 함수 작성

```c
// my_kernel.h
#include "ggml.h"
#include "ggml-custom-op.h"

static bool my_add_can_handle(const struct ggml_tensor * op) {
    // 이 커널이 처리할 수 있는 텐서 조건을 반환
    return op->op == GGML_OP_ADD
        && op->src[0]->type == GGML_TYPE_F32
        && op->src[1]->type == GGML_TYPE_F32;
}

static bool my_add_compute(const struct ggml_compute_params * params,
                           struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t ne = ggml_nelements(dst);
    const int64_t dr = (ne + nth - 1) / nth;
    const int64_t i0 = ith * dr;
    const int64_t i1 = MIN(i0 + dr, ne);

    for (int64_t i = i0; i < i1; i++) {
        ((float *)dst->data)[i] =
            ((const float *)src0->data)[i] +
            ((const float *)src1->data)[i];
    }
    return true;
}
```

### 5.2 커널 등록

```c
// main.c
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-custom-op.h"
#include "my_kernel.h"

int main(void) {
    // 1. 백엔드 초기화 전이나 후에 커널 등록 가능
    ggml_backend_custom_op_register_kernel(
        GGML_OP_ADD,               // 어떤 Op를 가로챌지
        "my_add_f32",              // 커널 이름 (디버깅용)
        my_add_compute,            // compute 함수 포인터
        my_add_can_handle          // can_handle 함수 포인터 (NULL = 모든 텐서)
    );

    // 2. 백엔드 초기화
    ggml_backend_load_all();
    ggml_backend_t custom_backend = ggml_backend_custom_op_init();

    // 3. 스레드 수 설정 (선택)
    ggml_backend_custom_op_set_n_threads(custom_backend, 4);

    // 4. 스케줄러에 백엔드 등록 후 그래프 실행
    // ...
}
```

### 5.3 `can_handle` 콜백

`can_handle` 콜백은 `supports_op()`에서 호출되어, 특정 텐서 구성에 대해서만 커널이 동작하도록 제한한다:

- `can_handle == NULL` → 해당 Op의 모든 텐서를 처리함
- `can_handle != NULL` → 콜백이 `true`를 반환하는 텐서만 처리함

예: MUL_MAT 커널이 F32×F32만 처리하고 싶다면:
```c
static bool my_mul_mat_can_handle(const struct ggml_tensor * op) {
    return op->src[0]->type == GGML_TYPE_F32
        && op->src[1]->type == GGML_TYPE_F32
        && ggml_is_contiguous(op->src[0])
        && ggml_is_contiguous(op->src[1]);
}
```

---

## 6. 프로젝트 파일 구조

```
llama.cpp/
├── ggml/
│   ├── include/
│   │   └── ggml-custom-op.h          # 퍼블릭 API 헤더
│   └── src/
│       ├── ggml-custom-op/
│       │   ├── CMakeLists.txt         # ggml_add_backend_library(ggml-custom-op ...)
│       │   └── ggml-custom-op.cpp     # 백엔드 구현체
│       └── CMakeLists.txt             # ggml_add_backend(CUSTOM_OP) 추가됨
├── examples/
│   └── custom-op/
│       ├── CMakeLists.txt             # 빌드 타겟 (benchmark-custom-op, benchmark-layer-timing)
│       ├── benchmark-custom-op.cpp    # GGML 그래프 단위 벤치마크
│       └── benchmark-layer-timing.cpp # 레이어별 연산시간 벤치마크
└── _optimized_kernel_proj/
    ├── how-to-build.md               # 이 파일
    └── how-to-test.md                 # 테스트 가이드
```

---

## 7. 빌드 트러블슈팅

### "ggml-custom-op.h: No such file or directory"

CMake가 헤더 검색 경로에 `ggml/include/`를 포함하지 않은 경우.
빌드 시스템이 `ggml` 타겟의 `PUBLIC` 인클루드 디렉토리로 설정하는지 확인.

### "undefined reference to ggml_backend_custom_op_register_kernel"

정적 링크 시 `ggml-custom-op` 라이브러리를 링크하지 않은 경우.
링커 명령에 `-lggml-custom-op` (또는 CMake의 `target_link_libraries`)를 추가.

### "GGML_BACKEND_DL is required"

동적 로딩 없이 정적으로 빌드하면 `GGML_BACKEND_DL_IMPL` 매크로가 빈 매크로로 정의됨.
동적 로딩이 필요하면 `-DGGML_BACKEND_DL=ON` 추가.

### macOS에서 "dylib not found"

`GGML_BACKEND_PATH` 경로가 절대 경로인지 확인하거나, `DYLD_LIBRARY_PATH`에 디렉토리를 추가:
```bash
export DYLD_LIBRARY_PATH=/path/to/build/bin:$DYLD_LIBRARY_PATH
```

---

## 8. CMake 통합 (다른 프로젝트에서 사용)

다른 프로젝트에서 `ggml-custom-op`를 서브디렉토리로 포함하는 경우:

```cmake
# CMakeLists.txt
add_subdirectory(path/to/llama.cpp/ggml ggml_build)

target_link_libraries(my_app PRIVATE ggml-custom-op)
target_include_directories(my_app PRIVATE ${CMAKE_SOURCE_DIR}/path/to/llama.cpp/ggml/include)
```

또는 빌드된 라이브러리를 직접 링크:
```cmake
find_package(ggml REQUIRED)
target_link_libraries(my_app PRIVATE ggml::ggml-custom-op)
```