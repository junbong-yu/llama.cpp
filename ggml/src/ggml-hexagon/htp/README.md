# HMX (Hexagon Matrix Extensions) Programming Guide

## 개요

**HMX (Hexagon Matrix Extensions)**는 Qualcomm Snapdragon 디바이스의 Hexagon DSP에서 행렬 곱셈을 가속화하기 위한 전용 하드웨어 유닛입니다. HVX (Hexagon Vector Extensions)와 달리 HMX는 32×32 FP16 타일 기반의 매트릭스 연산에 특화되어 있습니다.

### 지원 하드웨어
- **Hexagon v73+** (Snapdragon 8 Gen 2 및 최신)
- **Hexagon v75** (Snapdragon 8 Gen 3)
- **Hexagon v79/v81** (Snapdragon 8 Elite)

## 아키텍처 개요

```
┌─────────────────────────────────────┐
│  HMX Matrix Processing Unit         │
│  ┌─────────────────────────────┐    │
│  │  activation.hf (32×32 FP16) │    │
│  │  weight.hf     (32×32 FP16) │    │
│  │              ×              │    │
│  │       bias (scales)         │    │
│  │              ↓              │    │
│  │  Accumulator (FP32/FP16)    │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

## 핵심 특징

| 특성 | 설명 |
|------|------|
| **타일 크기** | 32×32 FP16 (2048 bytes) |
| **정렬 요구사항** | 128-byte 정렬 |
| **K 차원** | 32의 배수 필수 |
| **N 차원** | 32의 배수 필수 |
| **M 차명** | 32의 배수 권장 (아닐 시 HVX fallback) |
| **메모리** | VTCM (Vector Tightly Coupled Memory) 필수 |
| **입력 포맷** | FP16 직접 지원 (양자화는 사전 변환 필요) |

## 빌드 설정

### CMakeLists.txt

```cmake
# HMX 지원 확인 (v73+)
if(HEXAGON_ARCH VERSION_GREATER_EQUAL "73")
    target_compile_options(${TARGET_NAME} PRIVATE
        -mhmx              # HMX 명령어 활성화
        -mhvx              # HVX (HMX 필수)
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE
        HTP_HAS_HMX=1      # HMX 매크로 정의
    )
    target_sources(${TARGET_NAME} PRIVATE
        hmx-matmul-ops.c
        hmx-utils.h
        hmx-ops.h
    )
endif()
```

### 컴파일러 플래그

```bash
# Hexagon Clang
hexagon-clang \
    -O3 \
    -mv73 \
    -mhvx \
    -mhmx \
    -DHTP_HAS_HMX=1 \
    -c hmx-matmul-ops.c \
    -o hmx-matmul-ops.o
```

## C-Level API

### 고수준 함수 인터페이스

```c
// hmx-ops.h

/**
 * FP16 weights × FP32 activation 행렬 곱셈
 * @param ctx HTP 컨텍스트
 * @param dst 출력 버퍼 [M×N] float
 * @param activation 입력 활성화 [M×K] float
 * @param permuted_weight 가중치 [K×N] FP16 (tile-permuted)
 * @param m 행 수
 * @param k 내적 차원
 * @param n 열 수
 * @param act_stride 활성화 행 stride
 * @param weight_stride 가중치 행 stride
 * @return 0 성공, -1 실패
 */
int hmx_mat_mul_permuted_w16a32(struct htp_context *ctx,
                                float *restrict dst,
                                const float *activation,
                                const __fp16 *permuted_weight,
                                int m, int k, int n,
                                int act_stride,
                                int weight_stride);

/**
 * 양자화된 weights (Q4_0/Q8_0/IQ4_NL/MXFP4) 행렬 곱셈
 */
int hmx_mat_mul_permuted_qk_0_d16a32(struct htp_context *ctx,
                                      float *restrict dst,
                                      const float *activation,
                                      const uint8_t *permuted_weight,
                                      int m, int k, int n,
                                      int weight_type);
```

### 사용 예시

```c
#include "hmx-ops.h"
#include "htp-ctx.h"

void example_matmul(struct htp_context *ctx) {
    int m = 128, k = 256, n = 512;
    
    // 메모리 할당 (128-byte 정렬)
    float *dst = aligned_alloc(128, m * n * sizeof(float));
    float *act = aligned_alloc(128, m * k * sizeof(float));
    __fp16 *wgt = aligned_alloc(128, k * n * sizeof(__fp16));
    
    // 가중치를 tile-permuted 형태로 변환 (필수)
    convert_to_tile_permuted(wgt, weight_src, k, n);
    
    // HMX 행렬 곱셈 실행
    int ret = hmx_mat_mul_permuted_w16a32(ctx, dst, act, wgt,
                                           m, k, n, k, k);
    if (ret != 0) {
        // 에러 처리
    }
}
```

## Raw Assembly Programming

### 핵심 어셈블리 명령어

| 명령어 | 문법 | 설명 |
|--------|------|------|
| `mxmem` | `mxmem(Rs, Rt)` | 메모리 ↔ HMX 레지스터 전송 |
| `mxmem2` | `mxmem2(Rs)` | 바이어스/스케일 로드 |
| `mxclracc` | `mxclracc.hf` | FP16 어큐뮬레이터 초기화 |
| `acc` | `acc` | 어큐뮬레이터 레지스터 |

### 인라인 어셈블리 헬퍼

```c
// hmx-utils.h

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024  // 32×32
#define HMX_FP16_TILE_SIZE   2048  // 1024 × 2 bytes

// 출력 스케일 설정 (256-byte 정렬 필요)
static inline void hmx_set_output_scales(const void *scales) {
    asm volatile("bias = mxmem2(%0)" :: "r"(scales));
}

// 스케일 초기화 (HVX 사용)
static inline void hmx_init_column_scales(void *out_scales, 
                                          HVX_Vector v_scale) {
    HVX_Vector *pv = (HVX_Vector *)out_scales;
    *pv++ = v_scale;           // 128 bytes
    *pv   = Q6_V_vzero();      // 128 bytes zero padding
}

// 여러 타일 스트리밍 로드
static inline void hmx_load_tiles_fp16(const __fp16 *row_tiles,
                                        const __fp16 *col_tiles,
                                        size_t n_tiles) {
    size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
    asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "weight.hf = mxmem(%2, %3) }\n"
        :: "r"(row_tiles), "r"(limit),
           "r"(col_tiles), "r"(limit)
        : "memory");
}

// 단일 타일 쌍 로드
static inline void hmx_load_tile_pair_fp16(const __fp16 *act_tile,
                                            const __fp16 *wt_tile) {
    asm volatile(
        "{ activation.hf = mxmem(%0, %1)\n"
        "weight.hf = mxmem(%2, %3) }\n"
        :: "r"(act_tile), "r"(2047),   // 단일 타일: 2047
           "r"(wt_tile),  "r"(2047)
        : "memory");
}

// 어큐뮬레이터 결과 저장
static inline void hmx_consume_accumulator_fp16(__fp16 *out) {
    asm volatile(
        "mxmem(%0, %1):after.hf = acc\n"
        :: "r"(out), "r"(0)
        : "memory");
}
```

### 완전한 매트릭스 곱셈 루프

```c
// C = A × B 행렬 곱셈 (타일 단위)
static void core_dot_chunk_fp16(__fp16 *output,
                                const __fp16 *activation,
                                const __fp16 *weight,
                                const __fp16 *scales,
                                int n_row_tiles,   // M/32
                                int n_col_tiles,   // N/32
                                int n_dot_tiles) { // K/32
    // 1. 출력 스케일 설정
    hmx_set_output_scales(scales);
    
    // 2. 타일 단위로 순회
    for (int r = 0; r < n_row_tiles; ++r) {
        for (int c = 0; c < n_col_tiles; ++c) {
            // 3. 어큐뮬레이터 초기화
            asm volatile("mxclracc.hf");
            
            const __fp16 *row_tiles = activation + 
                                      r * n_dot_tiles * 1024;
            const __fp16 *col_tiles = weight + 
                                      c * n_dot_tiles * 1024;
            
            // 4. K 차원 순회하며 타일 곱셈 누적
            for (int k = 0; k < n_dot_tiles; ++k) {
                int offset = k * 1024;
                hmx_load_tile_pair_fp16(row_tiles + offset,
                                        col_tiles + offset);
                // HMX가 자동으로 누적!
            }
            
            // 5. 결과 저장
            __fp16 *out_tile = output + 
                               (r * n_col_tiles + c) * 1024;
            hmx_consume_accumulator_fp16(out_tile);
        }
    }
}
```

### 어셈블리 문법 분석

```asm
activation.hf = mxmem(Rs, Rt):deep
│           │      │    │    │
│           │      │    │    └── :deep = 깊은 파이프라인 스트리밍
│           │      │    └─────── Rt = 메모리 영역 크기 (limit)
│           │      └──────────── Rs = 메모리 주소
│           └─────────────────── .hf = FP16 타입
└─────────────────────────────── activation = HMX 레지스터

mxmem(Rs, Rt):after.hf = acc
│      │   │      │      │
│      │   │      │      └── acc = 어큐뮬레이터
│      │   │      └───────── :after.hf = FP16 변환 후 저장
│      │   └──────────────── Rt = 0 (단일 타일)
│      └──────────────────── Rs = 출력 주소
└─────────────────────────── mxmem = 메모리 저장
```

## 데이터 레이아웃

### Tile-Permuted Format

HMX는 특수한 **Crouton layout**을 사용합니다:

```
원본 행렬 (K×N):
┌─────────────────┐
│ a00 a01 a02 ... │
│ a10 a11 a12 ... │
│ ...             │
└─────────────────┘

Tile-permuted 변환 (32×32 타일):
┌─────────────────────────────┐
│ 타일(0,0) │ 타일(0,1) │ ... │
├─────────────────────────────┤
│ 타일(1,0) │ 타일(1,1) │ ... │
└─────────────────────────────┘

각 타일 내부:
┌────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ ...│  (row-major within tile)
├────┼────┼────┼────┤
│ 32 │ 33 │ 34 │ ...│
└────┴────┴────┴────┘
```

### 변환 함수 예시

```c
// F16 row-major → tile-permuted 변환
void interleave_fp16_to_tiles(__fp16 *dst, const __fp16 *src,
                               int n_cols, int k) {
    const int n_k_tiles = k / 32;
    const HVX_Vector v_scat_base = hvx_vmem(scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    
    for (int r = 0; r < n_cols; r += 2) {
        int ct = r / 32;       // N-dimension tile index
        int local_r = r % 32;  // intra-tile row index
        
        HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, 
                                            Q6_V_vsplat_R(local_r * 4));
        HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
        
        for (int c = 0; c < k; c += 32) {
            int kt = c / 32;
            int tile_idx = ct * n_k_tiles + kt;
            __fp16 *tile_base = dst + tile_idx * 1024;
            
            HVX_Vector v0 = hvx_vmemu(src + r * k + c);
            HVX_Vector v1 = hvx_vmemu(src + (r + 1) * k + c);
            
            // HVX scatter로 타일 메모리에 쓰기
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, 
                               2047, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, 
                               2047, v_off1, v1);
        }
    }
}
```

## HVX + HMX 하이브리드

### 동기화 패턴

**중요:** HVX 스캐터 후 HMX 읽기 전 동기화 필요

```c
// HVX로 VTCM에 쓴 후 HMX가 읽기 전에 동기화
// HVX scatter는 버퍼링되므로 명시적 flush 필요

// 방법 1: vmem load로 scatter 버퍼 flush
(void) *(volatile HVX_Vector *)(vtcm_dst + 
         (end_tile - 1) * HMX_FP16_TILE_N_ELMS);

// 방법 2: 메모리 펜스
__asm__ __volatile__("" ::: "memory");
```

### 역양자화 + HMX

```c
// Q4_0 → FP16 역양자화 후 HMX 연산
void dequantize_and_matmul_q4(struct htp_context *ctx,
                               float *dst, const float *act,
                               const uint8_t *q4_weight,
                               int m, int k, int n) {
    // 1. VTCM 할당
    __fp16 *vtcm_wgt = vtcm_alloc(n * k * sizeof(__fp16));
    __fp16 *vtcm_act = vtcm_alloc(m * k * sizeof(__fp16));
    
    // 2. 역양자화 (HVX 사용)
    dequantize_q4_to_f16_hvx(vtcm_wgt, q4_weight, k, n);
    
    // 3. 동기화 (HVX → HMX)
    hvx_synchronize();
    
    // 4. HMX 연산
    HAP_compute_res_hmx_lock(ctx->vtcm_rctx);
    core_dot_chunk_fp16(vtcm_out, vtcm_act, vtcm_wgt, 
                        scales, m/32, n/32, k/32);
    HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
}
```

## HMX 자원 관리

### 초기화 및 락

```c
#include <HAP_compute_res.h>

// HMX 초기화 (htp_iface_start에서)
#ifdef HTP_HAS_HMX
if (use_hmx) {
    ctx->hmx_enabled = 1;
    ctx->vtcm_scratch_size = ctx->vtcm_size;
    
    // HMX 전원 켜기
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HMX;
    request.hmx.power_up = TRUE;
    HAP_power_set((void*)ctx, &request);
}
#endif

// HMX 연산 시 락 획득
HAP_compute_res_hmx_lock(ctx->vtcm_rctx);
// ... HMX 연산 ...
HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
```

### VTCM 메모리 관리

```c
// VTCM 순차 할당기
static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, 
                                      size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}

// 사용 예시
uint8_t *vtcm_base = ctx->vtcm_base;
uint8_t *vtcm_ptr = vtcm_base;

__fp16 *vtcm_weight = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, 
                                               weight_size);
__fp16 *vtcm_act = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, 
                                            act_size);
__fp16 *vtcm_out = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, 
                                            out_size);
```

## 제약사항

### VTCM 4MB 뱅크 경계

```c
// 중요: mxmem은 4MB 뱅크를 넘나들 수 없음!
// 타일 배치 시 확인 필요

// 안전한 배치
assert((uintptr_t)tile_addr & 0x3FFFFF000 == 
       ((uintptr_t)tile_addr + tile_size) & 0x3FFFFF000);
```

### 정렬 요구사항

| 데이터 | 정렬 |
|--------|------|
| 활성화/가중치 포인터 | 128-byte |
| 출력 포인터 | 128-byte |
| 스케일 | 256-byte |
| VTCM 버퍼 | 128-byte |

### 크기 제한

```c
// M, K, N은 32의 배수여야 함
assert(m % 32 == 0);
assert(k % 32 == 0);
assert(n % 32 == 0);

// 양자화된 weights는 256의 배수 필요
assert(k % 256 == 0);  // for Q4_0/Q8_0
```

## 환경 변수

```bash
# HMX 활성화
export GGML_HEXAGON_USE_HMX=1

# HVX 스레드 수
export GGML_HEXAGON_NHVX=4

# HTP 세션 수 (대형 모델용)
export GGML_HEXAGON_NDEV=2

# 실행
./llama-cli -m model.gguf --device HTP0
```

## 디버깅 및 프로파일링

### 타이머 매크로

```c
// hmx-profile.h

#define ENABLE_PROFILE_TIMERS  // 주석 제거 시 활성화

#if defined(ENABLE_PROFILE_TIMERS)
#  define TIMER_DEFINE(name) int64_t name##_ticks = 0
#  define TIMER_START(name)  \
     int64_t name##_t0 = HAP_perf_get_qtimer_count()
#  define TIMER_STOP(name)   \
     name##_ticks += HAP_perf_get_qtimer_count() - name##_t0
#  define TIMER_US(name)     \
     HAP_perf_qtimer_count_to_us(name##_ticks)
#else
#  define TIMER_DEFINE(name)
#  define TIMER_START(name)
#  define TIMER_STOP(name)
#  define TIMER_US(name)     0LL
#endif

// 사용 예시
TIMER_DEFINE(hmx_core);
TIMER_START(hmx_core);
core_dot_chunk_fp16(...);
TIMER_STOP(hmx_core);
FARF(HIGH, "HMX core: %lld us", TIMER_US(hmx_core));
```

### 상세 로깅

```bash
# 상세 로그 활성화
export GGML_HEXAGON_VERBOSE=1
export GGML_HEXAGON_PROFILE=1
```

## 참고 자료

- [Qualcomm Hexagon SDK](https://developer.qualcomm.com/software/hexagon-dsp-sdk)
- [QNN HTP Backend Extensions](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html)
- [htp-ops-lib (오픈소스 레퍼런스)](https://github.com/haozixu/htp-ops-lib)
- [llama.cpp PR #20693](https://github.com/ggml-org/llama.cpp/pull/20693)

## 라이선스

이 코드는 llama.cpp 프로젝트의 MIT 라이선스를 따릅니다.
