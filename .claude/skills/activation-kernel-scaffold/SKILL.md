---
name: activation-kernel-scaffold
description: "사용자가 제공한 활성 함수(ReLU/GeLU/SwiGLU) 커널을 tools/kernel-bench 구조에 통합하는 스캐폴더. kernels-user.cpp 플러그 포인트 생성, kernel-registry.cpp 등록 코드 제안, inference-bench.cpp 의 UNARY 훅 확장 패치 제안. '사용자 커널 통합', 'kernels-user.cpp 추가', 'variant=user 등록', '훅에 GELU/RELU 추가' 시 사용."
---

# Activation Kernel Scaffold

사용자 커널 구현을 보존하면서 kernel-bench / inference-bench 에 통합 가능한 최소 어댑터 코드를 제안하는 스킬. **파일을 자동 생성·수정하지 않고 diff 를 제시하며**, 사용자 승인 후에만 실제 반영한다.

## 주요 단계

### 1. 타깃 op 분석

사용자가 지정한 op(relu/gelu/swiglu/silu) 별로 ggml 표현을 확인한다:

| op | ggml 표현 | inference-bench 훅 진입점 |
|----|----------|-------------------------|
| ReLU | `GGML_OP_UNARY` + `GGML_UNARY_OP_RELU` | UNARY 훅 내에서 `ggml_get_unary_op() == GGML_UNARY_OP_RELU` 분기 |
| GeLU | `GGML_OP_UNARY` + `GGML_UNARY_OP_{GELU, GELU_QUICK, GELU_ERF}` | 동일. variant 별로 분기 |
| SiLU | `GGML_OP_UNARY` + `GGML_UNARY_OP_SILU` | 동일 |
| SwiGLU (퓨즈드) | `GGML_OP_SWIGLU` (ggml 최신) | `GGML_OP_SWIGLU` 훅 별도 |
| SwiGLU (합성) | `SILU(x) * gate` | SiLU 훅만 + MUL 은 기존 훅 그대로 |

### 2. kernels-user.cpp 플러그 포인트 제안

`tools/kernel-bench/kernels/kernels-user.{h,cpp}` 파일의 템플릿을 제안한다. 사용자는 **함수 본문**만 자신의 구현으로 교체:

```cpp
// kernels-user.h
#pragma once
#include "kernel-registry.h"
namespace kernels_user {
    void relu_f32(const float* x, float* y, size_t n, int ith, int nth);
    void gelu_f32(const float* x, float* y, size_t n, int ith, int nth);
    void silu_f32(const float* x, float* y, size_t n, int ith, int nth);
    // SwiGLU 는 모델 경로 확정 후 시그니처 결정
    void register_all();
}

// kernels-user.cpp (본문은 사용자가 교체)
#include "kernels-user.h"
namespace kernels_user {
    void relu_f32(const float* x, float* y, size_t n, int ith, int nth) {
        // === TODO(user): 당신의 ReLU 구현 ===
        // ith/nth 로 스레드 분할, n 원소 범위만 처리
    }
    // ... gelu_f32, silu_f32 동일 패턴
}
```

### 3. kernel-registry.cpp 등록 diff

기존 파일을 수정하지 않고 **아래 추가 라인을 제안**한다. 사용자 승인 후 kernel-author 가 반영:

```cpp
// kernel-registry.cpp init_standard_kernels() 끝부분에 추가
#include "kernels/kernels-user.h"
void init_user_kernels() {
    register_kernel("relu", "user", kernels_user::relu_f32);
    register_kernel("gelu", "user", kernels_user::gelu_f32);
    register_kernel("silu", "user", kernels_user::silu_f32);
}
```

### 4. inference-bench.cpp 훅 확장 diff

현재 훅은 `GGML_OP_ADD` / `GGML_OP_MUL` 만 받는다. UNARY 추가를 위한 최소 diff 제안 (기존 파일 수정은 사용자 승인 후):

```cpp
// custom_tensor_traits::compute_forward 내부
switch (op->op) {
    case GGML_OP_ADD: /* 기존 */ break;
    case GGML_OP_MUL: /* 기존 */ break;
    case GGML_OP_UNARY: {
        switch (ggml_get_unary_op(op)) {
            case GGML_UNARY_OP_RELU:
                kernels_user::relu_f32(...);
                return true;
            case GGML_UNARY_OP_GELU:
            case GGML_UNARY_OP_GELU_QUICK:
            case GGML_UNARY_OP_GELU_ERF:
                kernels_user::gelu_f32(...);
                return true;
            case GGML_UNARY_OP_SILU:
                kernels_user::silu_f32(...);
                return true;
            default: return false;
        }
    }
    // SwiGLU 는 모델 경로 확정 후 추가
}
```

GeLU variant 가 여러 개 섞일 수 있으므로 **훅 진입 시 실제로 받은 `ggml_unary_op` 를 stderr 로 한 번 덤프** 하는 진단 경로를 권장한다 (첫 실행에서 variant 확인 후 필요하면 세분화).

### 5. CMakeLists.txt 변경 제안

`tools/kernel-bench/CMakeLists.txt` 에 `kernels-user.cpp` 를 추가하는 블록 제안. SVE 같이 조건부 컴파일이 필요한 변형은 `check_cxx_compiler_flag` 로 게이팅.

## 출력 포맷

단일 마크다운 보고서:

```
## 1. 타깃 op 표현 분석
## 2. 제안 파일 (kernels-user.{h,cpp}) 전문
## 3. kernel-registry.cpp diff
## 4. inference-bench.cpp diff
## 5. CMakeLists.txt diff
## 6. 예상 영향 (빌드 타깃, 링크 의존성, 기존 테스트)
```

사용자 확인 후 kernel-author 에이전트가 diff 를 실제 반영한다.

## 제약

- 사용자 커널 로직을 수정하지 않는다 (시그니처만 맞춤, 본문 보존)
- 기존 `kernels-custom.cpp` / `kernels-standard.cpp` 는 수정하지 않는다
- 훅 확장은 **UNARY/SWIGLU 만 추가**, 기존 ADD/MUL 훅은 그대로 유지
