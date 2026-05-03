# llama.cpp 심층 아키텍처 분석 (DEEP)

> 이 문서는 `llama.cpp-architecture.md`의 HIGH-LEVEL 내용을 전제로 한다. 여기서는 실행 의미론, 내부 구현, 코드 레벨의 세부 사항에 집중한다.

---

## 목차

1. [실행 모델 심층 분석](#1-실행-모델-심층-분석)
2. [GGML 연산 시스템](#2-ggml-연산-시스템)
3. [백엔드 추상화 및 스케줄러](#3-백엔드-추상화-및-스케줄러)
4. [llama.cpp의 GGML 위 계층](#4-llamacpp의-ggml-위-계층)
5. [추론 파이프라인 상세](#5-추론-파이프라인-상세)
6. [KV Cache 심층 분석](#6-kv-cache-심층-분석)
7. [Attention 메커니즘](#7-attention-메커니즘)
8. [양자화 상세](#8-양자화-상세)
9. [배치 처리](#9-배치-처리)
10. [데이터 흐름 종합](#10-데이터-흐름-종합)
11. [핵심 파일 레퍼런스](#11-핵심-파일-레퍼런스)

---

## 1. 실행 모델 심층 분석

### 1.1 ggml_tensor: n차원 텐서의 물리적 표현

`ggml_tensor` (`ggml.h:660-692`)는 GGML의 핵심 데이터 구조다. 모든 연산은 이 구조체를 입력으로 받아 새로운 `ggml_tensor`를 반환한다.

```c
struct ggml_tensor {
    enum ggml_type type;                    // 데이터 타입 (F32, F16, Q4_0, ...)
    struct ggml_backend_buffer * buffer;    // 백엔드 버퍼 (GPU/CPU 메모리 위치)

    int64_t ne[GGML_MAX_DIMS];              // shape: [ne0, ne1, ne2, ne3]
    size_t  nb[GGML_MAX_DIMS];              // stride (byte 단위)
                                            //   nb[0] = ggml_type_size(type)
                                            //   nb[1] = nb[0] * (ne[0] / blck_size) + padding
                                            //   nb[i] = nb[i-1] * ne[i-1]

    enum ggml_op op;                        // 이 텐서가 나타내는 연산 (GGML_OP_MUL_MAT 등)
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];  // 연산별 파라미터
    int32_t flags;                          // INPUT, OUTPUT, PARAM, LOSS, COMPUTE

    struct ggml_tensor * src[GGML_MAX_SRC]; // 최대 10개의 소스 텐서 (연산 입력)

    struct ggml_tensor * view_src;          // 뷰인 경우 원본 텐서
    size_t               view_offs;         // 뷰 오프셋 (byte)

    void * data;                            // 실제 데이터 포인터 (할당된 경우)
    char name[GGML_MAX_NAME];               // 텐서 이름 ("blk.0.attn_q.weight" 등)
    void * extra;                           // 백엔드 특화 데이터 (CUDA: ggml_cuda_tensor_extra 등)
    char padding[8];                        // 정렬용 패딩
};
```

**핵심 설계 포인트**:

- **`ne[]` (number of elements)**: GGML은 Fortran 스타일 열우선(column-major) 레이아웃을 사용한다. `ne[0]`이 가장 안쪽 차원이다. 양자화 타입의 경우 `ne[0]`은 블록 단위로 정렬된다.
- **`nb[]` (nbytes stride)**: 각 차원의 바이트 간격. `nb[0]`은 `ggml_type_size(type)`이며, 양자화 타입에서는 `ggml_blck_size(type)`로 나눈 값이 실제 요소 수다.
- **`op` + `src[]`**: 연산 결과 텐서는 `op`에 연산 종류를, `src[]`에 입력 텐서들을 저장한다. 리프 텐서(가중치)는 `op = GGML_OP_NONE`이다.
- **`view_src` + `view_offs`**: `ggml_view_2d` 등으로 생성된 뷰 텐서는 원본을 참조한다. 데이터 복제 없이 슬라이싱이 가능하다.
- **`buffer`**: 텐서가 어느 백엔드 버퍼에 할당되었는지 추적. GPU 텐서는 CUDA/Metal 버퍼를, CPU 텐서는 호스트 버퍼를 가리킨다.
- **`extra`**: 백엔드별 추가 메타데이터. CUDA 백엔드에서는 `ggml_cuda_tensor_extra`가 device pointer와 split 정보를 저장한다.

### 1.2 ggml_cgraph: 계산 그래프

`ggml_cgraph` (`ggml-impl.h:329-347`)는 텐서 연산의 DAG(Directed Acyclic Graph)를 표현한다.

```c
struct ggml_cgraph {
    int size;                              // 최대 노드 수 (기본 GGML_DEFAULT_GRAPH_SIZE = 2048)
    int n_nodes;                           // 활성 노드 수
    int n_leafs;                           // 활성 리프 수

    struct ggml_tensor ** nodes;           // 연산 결과 텐서 (변화 가능한 데이터)
    struct ggml_tensor ** grads;           // 그래디언트 출력 (학습용)
    struct ggml_tensor ** grad_accs;       // 그래디언트 누적기
    struct ggml_tensor ** leafs;           // 상수 텐서 (가중치 등)
    int32_t             * use_counts;      // 각 텐서의 사용 횟수 (해시 테이블 인덱스 기준)

    struct ggml_hash_set visited_hash_set; // 중복 노드 방지용 해시 집합
    enum ggml_cgraph_eval_order order;     // 평가 순서 (GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT 등)
    uint64_t uid;                          // 그래프 고유 식별자 (재사용 판단용)
};
```

**그래프 빌드 메커니즘**:

```c
// 그래프 생성
struct ggml_cgraph * gf = ggml_new_graph(ctx);

// 연산 정의 → 그래프에 추가
struct ggml_tensor * result = ggml_mul_mat(ctx, weight, input);
ggml_build_forward_expand(gf, result);  // result와 그 의존성을 재귀적으로 그래프에 추가
```

`ggml_build_forward_expand()`는 **lazy build-as-you-go** 방식으로 동작한다:

1. 주어진 텐서의 `src[]`를 재귀적으로 순회
2. 각 소스 텐서에 대해 동일하게 재귀 호출 (DFS)
3. 리프 텐서(가중치)에 도달하면 `leafs[]`에 추가
4. 연산 텐서를 `nodes[]`에 추가 (위상 정렬 순서 보장)
5. `visited_hash_set`으로 중복 추가 방지

이 방식은 그래프가 **빌드되는 순간 이미 위상 정렬**되어 있음을 보장한다. 별도의 토폴로지컬 소트 단계가 필요 없다.

### 1.3 그래프 재사용 (Graph Reuse)

llama.cpp는 매 스텝마다 그래프를 새로 빌드하지 않고, **토폴로지가 동일하면 이전 그래프를 재사용**한다.

```cpp
// llama-graph.h: llm_graph_result::can_reuse()
bool can_reuse(const llm_graph_params & params) {
    // 1. ubatch 구조 확인
    bool can_reuse_ubatch =
        ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
        ubatch.n_tokens     == other.ubatch.n_tokens &&
        ubatch.n_seq_tokens == other.ubatch.n_seq_tokens &&
        ubatch.n_seqs       == other.ubatch.n_seqs &&
        ubatch.n_seqs_unq   == other.ubatch.n_seqs_unq;

    // 2. equal_seqs 모드에서는 시퀀스 ID도 일치해야 함
    if (can_reuse_ubatch && ubatch.equal_seqs()) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
        }
    }

    // 3. 컨텍스트 파라미터 확인
    return can_reuse_ubatch &&
        cparams.embeddings  == other.cparams.embeddings &&
        cparams.causal_attn == other.cparams.causal_attn &&
        arch  == other.arch &&
        gtype == other.gtype &&
        cvec  == other.cvec &&
        loras == other.loras;
}
```

**재사용 흐름** (`llama-context.cpp:1171-1241`):

```cpp
llm_graph_result * llama_context::process_ubatch(...) {
    auto * res = gf_res_prev.get();  // 이전 그래프 결과
    auto * gf  = res->get_gf();

    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        // 그래프 재사용: 입력 데이터만 업데이트
        n_reused++;
    } else {
        // 새 그래프 빌드
        res->reset();
        ggml_backend_sched_reset(sched.get());
        gf = model.build_graph(gparams);
        ggml_backend_sched_alloc_graph(sched.get(), gf);
    }

    // 입력 데이터 설정 (텐서 data 포인터에 실제 값 복사)
    res->set_inputs(&ubatch);

    // 그래프 실행
    graph_compute(gf, ubatch.n_tokens > 1);
    return res;
}
```

**`gf_res_prev`와 `gf_res_reserve`**:

- `gf_res_prev`: 직전 ubatch의 그래프 결과. 재사용 후보.
- `gf_res_reserve`: 예비 그래프. ubatch 구조가 바뀔 때(예: n_tokens 변화) 이전 그래프를 보관해두었다가 같은 구조가 다시 나타나면 재사용.

### 1.4 단일 사용 할당 (Single-use Allocation)

GGML 그래프에서 **중간 연산 결과 텐서**는 compute 후에 즉시 해제된다. 이는 `ggml_allocr`(할당기)가 그래프의 `use_counts`를 분석하여 구현한다:

1. 그래프 빌드 시 각 텐서의 `use_count`를 계산 (몇 개의 노드가 이 텐서를 `src[]`로 참조하는가)
2. 스케줄러가 노드를 실행할 때마다 `use_count`를 감소
3. `use_count == 0`이 되면 해당 텐서의 메모리를 즉시 해제 (free list에 반환)
4. 다음 텐서가 같은 메모리 영역을 재사용

이로 인해 **피크 메모리 사용량**이 모든 중간 텐서를 동시에 보유하는 경우보다 크게 줄어든다.

---

## 2. GGML 연산 시스템

### 2.1 핵심 연산 시그니처

#### ggml_mul_mat — 행렬 곱셈

```c
GGML_API struct ggml_tensor * ggml_mul_mat(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,   // [ne03, ne02, n, k] — k열, n행
    struct ggml_tensor  * b);  // [ne03*y, ne02*x, m, k] — k열, m행 (내부적으로 전치됨)
// 결과: [ne03*y, ne02*x, m, n] — m행, n열
```

LLM에서 가장 빈번한 연산. 가중치 행렬 `a`와 활성화 `b`를 곱한다. GGML은 `b`를 내부적으로 전치하여 효율적인 계산을 수행한다. 양자화 가중치의 경우 dequantize + matmul이 fused된다.

**정밀도 제어**:

```c
ggml_mul_mat_set_prec(tensor, GGML_PREC_F32);  // Phi-2 등에서 높은 정밀도 필요 시
```

#### ggml_rms_norm — RMS 정규화

```c
GGML_API struct ggml_tensor * ggml_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    float                 eps);  // epsilon (보통 1e-5 또는 1e-6)
```

RMSNorm 연산: `x / sqrt(mean(x^2) + eps)`. LayerNorm과 달리 평균을 빼지 않아 계산량이 적다. LLaMA, Mistral, Gemma 등 대부분의 현대 LLM이 사용.

#### ggml_rope_ext — RoPE (Rotary Position Embedding) 확장

```c
GGML_API struct ggml_tensor * ggml_rope_ext(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,       // [n_embd_head, n_tokens, n_head, ...] — Q 또는 K
    struct ggml_tensor  * b,       // [n_tokens] — 위치 인덱스 (inp_pos)
    struct ggml_tensor  * c,       // rope freq factors (LLaMA3용, 없으면 nullptr)
    int                   n_dims,  // RoPE 적용 차원 수 (보통 n_embd_head)
    int                   mode,    // GGML_ROPE_TYPE_NORMAL, NEOX, MROPE, VISION 등
    int                   n_ctx_orig,
    float                 freq_base,
    float                 freq_scale,
    float                 ext_factor,   // NTK-aware interpolation 확장 계수
    float                 attn_factor,  // attention 스케일 팩터
    float                 beta_fast,    // YaRN fast decay
    float                 beta_slow);   // YaRN slow decay
```

RoPE는 Q와 K 벡터에 회전 변환을 적용하여 위치 정보를 인코딩한다. `mode` 파라미터로 다양한 변형을 지원:

- `GGML_ROPE_TYPE_NORMAL`: 기본 RoPE
- `GGML_ROPE_TYPE_NEOX`: GPT-NeoX 스타일 (차원 순서 변경)
- `GGML_ROPE_TYPE_MROPE`: 다중 RoPE (Qwen-VL 등 비전 모델)
- `GGML_ROPE_TYPE_VISION`: 비전 특화 RoPE

**LLaMA3 RoPE 팩터**: `c` 파라미터에 레이어별 주파수 팩터를 전달하여 스케일링을 조정한다.

#### ggml_flash_attn_ext — Flash Attention

```c
GGML_API struct ggml_tensor * ggml_flash_attn_ext(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,       // [n_embd_head, n_tokens, n_head, ...]
    struct ggml_tensor  * k,       // [n_embd_head, n_kv, n_head_kv, ...]
    struct ggml_tensor  * v,       // [n_embd_head, n_kv, n_head_kv, ...] (전치됨)
    struct ggml_tensor  * mask,    // causal mask 또는 nullptr
    float                 scale,   // 1/sqrt(n_embd_head)
    float                 max_bias,   // ALiBi 바이어스
    float                 logit_softcap);  // 로짓 소프트캡 (Gemma2 등)
```

Flash Attention v2/v3 알고리즘을 구현. Q, K, V를 받아 attention 결과를 반환한다.

**브로드캐스팅 규칙**:
- `n_head % n_head_kv == 0` (GQA 지원)
- `n_head % ne32 == 0`
- `ne3 % ne33 == 0`

**정밀도 제어**:

```c
ggml_flash_attn_ext_set_prec(tensor, GGML_PREC_F32);
```

**Sinks 지원** (무한 컨텍스트용):

```c
ggml_flash_attn_ext_add_sinks(tensor, sinks);
```

#### ggml_swiglu — SwiGLU 활성화 함수

```c
GGML_API struct ggml_tensor * ggml_swiglu(
    struct ggml_context * ctx,
    struct ggml_tensor  * a);
```

SwiGLU: `SiLU(gate) * up`. LLaMA, Mistral 등 현대 LLM의 FFN에서 사용. `ggml_mul_mat` 3회 (gate, up, down) + `ggml_swiglu` 1회로 구성된다.

**변형**:

- `ggml_swiglu_split`: SwiGLU split (gate와 up이 하나의 텐서에 fused된 경우)
- `ggml_swiglu_swapped`: 게이트와 업 위치가 바뀐 경우
- `ggml_swiglu_oai`: OpenAI 호환 SwiGLU

### 2.2 연산 코드 매핑

| GGML 연산 | LLM 용도 | 주요 파라미터 |
|---|---|---|
| `GGML_OP_MUL_MAT` | Linear 프로젝션 (QKV, FFN, lm_head) | precision |
| `GGML_OP_MUL_MAT_ID` | MoE expert 선택 | expert IDs |
| `GGML_OP_FLASH_ATTN_EXT` | Self-attention | scale, mask, max_bias |
| `GGML_OP_ROPE_EXT` | 위치 인코딩 | n_dims, freq_base, mode |
| `GGML_OP_RMS_NORM` | 정규화 (attn_norm, ffn_norm) | eps |
| `GGML_OP_SWIGLU` | FFN 활성화 | 없음 |
| `GGML_OP_ADD` | Residual connection | 없음 |
| `GGML_OP_GET_ROWS` | KV cache에서 행 추출 | index tensor |
| `GGML_OP_CONT` | 메모리 연속화 | 없음 |
| `GGML_OP_SCALE` | 스칼라 곱 | scale factor |
| `GGML_OP_SILU` | SiLU 활성화 | 없음 |
| `GGML_OP_SOFT_MAX` | Attention softmax | dim, mask |

---

## 3. 백엔드 추상화 및 스케줄러

### 3.1 ggml_backend_sched: 5-Pass 할당 알고리즘

`ggml_backend_sched` (`ggml-backend.cpp:774-828`)는 계산 그래프의 각 노드를 적절한 백엔드(CPU, CUDA, Metal 등)에 할당하고, 백엔드 경계에서 그래프를 분할한다.

```c
struct ggml_backend_sched {
    bool is_reset;
    bool is_alloc;

    int n_backends;
    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids;  // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;       // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids;  // [graph_size]
    int * leaf_backend_ids;  // [graph_size]

    struct ggml_cgraph graph;
    struct ggml_backend_sched_split * splits;  // 그래프 분할 결과
    int n_splits;

    // 파이프라인 병렬 처리 지원
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
};
```

### 3.2 5-Pass 할당 상세

`ggml_backend_sched_split_graph()` (`ggml-backend.cpp:1014`)는 다음 5단계로 백엔드를 할당한다:

**Pass 1: 버퍼 기반 할당**

이미 버퍼가 할당된 텐서(가중치 등)는 해당 버퍼의 백엔드에 할당한다.

```c
// leafs (가중치)
for (int i = 0; i < graph->n_leafs; i++) {
    int * leaf_backend_id = &tensor_backend_id(leaf);
    if (*leaf_backend_id == -1) {
        *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
    }
}
// nodes (연산 결과)
for (int i = 0; i < graph->n_nodes; i++) {
    int * node_backend_id = &tensor_backend_id(node);
    if (*node_backend_id == -1) {
        *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);
    }
}
```

`ggml_backend_sched_backend_id_from_cur()`는 다음 우선순위로 백엔드를 결정한다:
1. 텐서의 `buffer`가 있는 경우 → 해당 버퍼의 백엔드
2. `view_src`가 있는 경우 → view_src의 버퍼 백엔드
3. `GGML_TENSOR_FLAG_INPUT` 플래그 → CPU (마지막 백엔드)
4. `src[]` 중 가중치(`usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS`)가 있는 경우 → 해당 가중치의 백엔드

**Pass 2: 인접 노드 확장 (4방향)**

할당된 노드를 기준으로 인접한 미할당 노드를 같은 백엔드로 확장한다.

```c
// GPU 백엔드 아래로 확장
{
    int cur_backend_id = -1;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (*node_backend_id != -1) {
            if (*node_backend_id == sched->n_backends - 1) {
                cur_backend_id = -1;  // CPU는 확장 중지
            } else {
                cur_backend_id = *node_backend_id;
            }
        } else if (cur_backend_id != -1) {
            ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
        }
    }
}
// GPU 백엔드 위로 확장 (역순)
// 나머지 백엔드 아래로 확장
// 나머지 백엔드 위로 확장 (역순)
```

GPU 백엔드(높은 우선순위)를 먼저 확장하고, CPU(가장 낮은 우선순위)는 마지막으로 확장한다. 이로 인해 GPU 연산이 최대화된다.

**Pass 3: 우선순위 업그레이드 + 미할당 노드 최적 할당**

```c
for (int i = 0; i < graph->n_nodes; i++) {
    if (*node_backend_id == -1) {
        // 미할당: 가장 많은 입력을 지원하는 백엔드 선택
        int n_supported_best = -1;
        for (int b = 0; b < sched->n_backends; b++) {
            if (ggml_backend_supports_op(sched->backends[b], node)) {
                int n_supported = 0;
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (ggml_backend_sched_buffer_supported(sched, src, b)) {
                        n_supported++;
                    }
                }
                if (n_supported > n_supported_best) {
                    n_supported_best = n_supported;
                    *node_backend_id = b;  // "3.best"
                }
            }
        }
    } else {
        // 할당됨: 더 높은 우선순위 백엔드로 업그레이드 가능 확인
        for (int b = 0; b < *node_backend_id; b++) {
            if (sched->bufts[b] == sched->bufts[*node_backend_id] &&
                ggml_backend_supports_op(sched->backends[b], node)) {
                // 모든 src의 버퍼 타입이 호환되면 업그레이드
                *node_backend_id = b;  // "3.upg"
                break;
            }
        }
    }
}
```

**Pass 4: view_src에서 할당 전파**

```c
for (int i = 0; i < graph->n_nodes; i++) {
    if (node->view_src != NULL && *cur_backend_id == -1) {
        *cur_backend_id = tensor_backend_id(node->view_src);  // "4.vsrc"
    }
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        if (*src_backend_id == -1 && src->view_src != NULL) {
            *src_backend_id = tensor_backend_id(src->view_src);  // "4.src"
        }
    }
}
```

**Pass 5: 백엔드 경계에서 분할**

할당이 완료된 후, 백엔드가 바뀌는 지점에서 그래프를 분할한다. 각 split은 동일한 백엔드에서 실행되는 연속된 노드들의 집합이다.

```c
struct ggml_backend_sched_split {
    int backend_id;
    int i_start;  // 시작 노드 인덱스
    int i_end;    // 종료 노드 인덱스
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];  // split 간 입력 텐서
    int n_inputs;
    struct ggml_cgraph graph;  // 이 split의 그래프 뷰
};
```

분할 시 이전 split의 출력 텐서가 다음 split의 입력 텐서로 전달된다. 백엔드가 다른 경우 데이터 복사(copy tensor)가 자동으로 삽입된다.

### 3.3 ggml_backend_reg 자동 감지

각 백엔드는 `ggml_backend_reg` 인터페이스를 구현하여 자동 등록된다:

```c
struct ggml_backend_reg {
    uint32_t api_version;
    struct ggml_backend_reg_i iface;
    void * context;
};

struct ggml_backend_reg_i {
    const char * (*get_name)(ggml_backend_reg_t reg);
    size_t       (*get_device_count)(ggml_backend_reg_t reg);
    ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);
    void *       (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
};
```

백엔드별 등록 함수:
- `ggml_backend_cpu_reg()` — CPU (항상 등록)
- `ggml_backend_cuda_reg()` — CUDA (`ggml-cuda/ggml-cuda.cpp`)
- `ggml_backend_metal_reg()` — Metal (`ggml-metal/ggml-metal.cpp`)
- `ggml_backend_vulkan_reg()` — Vulkan
- `ggml_backend_sycl_reg()` — SYCL
- `ggml_backend_cann_reg()` — CANN (Ascend NPU)
- `ggml_backend_zdnn_reg()` — zDNN (IBM Z)

llama.cpp는 `ggml_backend_load_all()`을 호출하여 사용 가능한 모든 백엔드를 동적으로 로드한다.

### 3.4 Multi-GPU Tensor Parallelism

**Meta Device**: 여러 GPU에 텐서를 분할하여 할당한다. `ggml_backend_sched`는 `n_copies` 파라미터로 복사본 수를 제어한다.

**Split State**: 파이프라인 병렬 처리(`cparams.pipeline_parallel`) 시 여러 백엔드가 동시에 다른 split을 실행한다. `ggml_backend_event_t`로 동기화한다.

```c
ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
```

---

## 4. llama.cpp의 GGML 위 계층

### 4.1 llama_model: 불변 가중치 컨테이너

`llama_model`은 GGUF 파일에서 로드된 가중치를 소유한다. 로드 후 **절대 변경되지 않으며**, 여러 `llama_context`가 공유할 수 있다.

**Data-Oriented Design**: `llama_layer`는 메서드가 없는 순수 데이터 구조체다. 모든 필드가 `ggml_tensor*` 포인터이며, 특정 아키텍처에 없는 필드는 `nullptr`다.

### 4.2 llama_layer: 140+ 아키텍처를 단일 구조체로

```cpp
struct llama_layer {
    // Attention 정규화
    ggml_tensor * attn_norm, * attn_norm_b;
    ggml_tensor * attn_q_norm, * attn_k_norm;

    // Attention 프로젝션
    ggml_tensor * wq, * wk, * wv, * wo;
    ggml_tensor * wqkv;  // Fused QKV (일부 모델)

    // FFN 정규화
    ggml_tensor * ffn_norm, * ffn_norm_b;

    // FFN 프로젝션 (SwiGLU)
    ggml_tensor * ffn_gate, * ffn_up, * ffn_down;

    // MoE
    ggml_tensor * ffn_gate_inp, * ffn_gate_exps;
    ggml_tensor * ffn_down_exps, * ffn_up_exps;

    // SSM (Mamba/RWKV)
    ggml_tensor * ssm_in, * ssm_x, * ssm_dt, * ssm_out;

    // 비트넷 스케일
    ggml_tensor * wq_s, * wk_s, * wv_s, * wo_s;
    // ... 140+ 아키텍처 지원
};
```

### 4.3 llm_graph_context: Template Method 패턴

`llm_graph_context`는 모델별 그래프 빌드가 공유하는 **공통 프리미티브**를 제공한다:

| 메서드 | 기능 |
|---|---|
| `build_inp_embd(tok_embd)` | 입력 토큰 → 임베딩 |
| `build_inp_pos()` | 위치 인덱스 텐서 |
| `build_inp_out_ids()` | 출력 행 인덱스 |
| `build_norm(cur, mw, mb, type, il)` | RMS Norm / Layer Norm |
| `build_qkv(layer, cur, ...)` | Q/K/V 프로젝션 (LoRA 포함) |
| `build_attn(inp, wo, q, k, v, ...)` | Flash Attention + output proj |
| `build_ffn(cur, gate, up, down, ...)` | SwiGLU FFN |
| `build_moe_ffn(cur, ...)` | Mixture of Experts FFN |
| `build_lora_mm(w, cur)` | LoRA 어댑터 적용 matmul |
| `build_cvec(cur, il)` | Control Vector 적용 |

### 4.4 LLaMA 모델 그래프 빌드 (코드 레벨)

`src/models/llama.cpp`의 LLaMA 그래프 빌드를 단계별로 추적한다:

```cpp
template <bool embed>
llm_build_llama<embed>::llm_build_llama(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();

    // 1. 입력 임베딩: 토큰 ID → [n_embd, n_tokens]
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    // 2. 위치 인덱스: [n_tokens]
    ggml_tensor * inp_pos = build_inp_pos();

    // 3. Attention 입력 구조체 생성 (KV cache 유무에 따라 분기)
    using inp_attn_type = std::conditional_t<embed,
        llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;
    inp_attn_type * inp_attn = build_attn_inp_kv();

    // 4. 스케일 계산: 1/sqrt(d_k)
    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // 5. 출력 행 인덱스 (logits를 필요한 토큰만 추출)
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // 6. 레이어 루프
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;  // skip connection용

        // 6a. Pre-attention RMSNorm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

        // 6b. QKV 프로젝션
        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head, n_head, n_head_kv, il);

        // 6c. RoPE 적용
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

        // 6d. Q/K 정규화 (Llama4 등 일부 모델)
        if (hparams.use_kq_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
            Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
        }

        // 6e. Flash Attention + Output Projection
        cur = build_attn(inp_attn,
            model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        // 6f. 마지막 레이어에서 출력 행 필터링
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // 6g. Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);

        // 6h. FFN (Dense 또는 MoE 분기)
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // Dense FFN
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            // MoE FFN
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps, model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps, nullptr,
                n_expert, n_expert_used, LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il, ...);
        }

        // 6i. Residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);

        // 6j. Control Vector 적용
        cur = build_cvec(cur, il);

        // 6k. 다음 레이어 입력으로 전달
        inpL = cur;
    }

    // 7. 최종 RMSNorm
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    res->t_embd = cur;

    // 8. LM Head (output projection) — 임베딩 모드가 아닐 때만
    if constexpr (!embed) {
        cur = build_lora_mm(model.output, cur);
        res->t_logits = cur;
    }

    // 9. 그래프에 최종 결과 추가
    ggml_build_forward_expand(gf, cur);
}
```

### 4.5 LoRA 어댑터 통합

`build_lora_mm()` (`llama-graph.cpp:969-998`)는 기본 matmul에 LoRA를 투명하게 추가한다:

```cpp
ggml_tensor * llm_graph_context::build_lora_mm(
    ggml_tensor * w, ggml_tensor * cur, ggml_tensor * w_s) const {

    // 기본 matmul
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    // 활성화된 모든 LoRA 어댑터에 대해
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) continue;  // 이 텐서에 LoRA가 없으면 스킵

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);
        // scale = alpha * adapter_scale / rank

        // LoRA: B @ (A @ x) * scale
        ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lw->b,
            ggml_mul_mat(ctx0, lw->a, cur));
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    // 스케일 적용 (비트넷 등)
    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }
    return res;
}
```

**LoRA 어댑터 구조** (`llama-adapter.h:63-88`):

```cpp
struct llama_adapter_lora {
    llama_model * model;
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;  // tensor_name → {lora_a, lora_b}
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
    float alpha;  // 전역 alpha 스케일
    std::unordered_map<std::string, std::string> gguf_kv;  // GGUF 메타데이터
    std::vector<llama_token> alora_invocation_tokens;  // aLoRA (activated LoRA) 토큰
};

struct llama_adapter_lora_weight {
    ggml_tensor * a;  // [rank, in_features]
    ggml_tensor * b;  // [out_features, rank]

    float get_scale(float alpha, float adapter_scale) const {
        const float rank = (float) b->ne[0];
        return alpha ? adapter_scale * alpha / rank : adapter_scale;
    }
};
```

---

## 5. 추론 파이프라인 상세

### 5.1 encode() 실행 흐름

`encode()`는 비-캐시형 추론(임베딩, reranking)에 사용된다:

```cpp
int llama_context::encode(const llama_batch & batch_inp) {
    // 1. 배치 할당기 초기화
    balloc->init(batch_inp, model.vocab, nullptr, n_embd,
        cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true);

    const uint32_t n_tokens = balloc->get_n_tokens();

    // 2. 단일 ubatch로 분할 (마이크로 배치 불가)
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);
    GGML_ASSERT(cparams.n_ubatch >= n_tokens);

    // 3. 메모리 컨텍스트 초기화 (KV 캐시 없음)
    auto mctx = memory->init_batch(*balloc, 1, cparams.embeddings);

    // 4. ubatch 처리
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    // 5. 결과 추출 (logits 또는 embeddings)
    // ...
}
```

### 5.2 decode() 실행 흐름

`decode()`는 캐시형 추론(텍스트 생성)에 사용된다:

```cpp
int llama_context::decode(const llama_batch & batch_inp) {
    // 1. 배치 할당기 초기화
    balloc->init(batch_inp, model.vocab, nullptr, n_embd,
        cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, false);

    // 2. KV 캐시 슬롯 할당 (ring buffer에서 빈 공간 찾기)
    auto mctx = memory->init_batch(*balloc, n_ubatch, cparams.embeddings);

    // 3. 출력 버퍼 예약
    output_reserve(n_outputs_all);

    // 4. ubatch 루프
    do {
        const auto & ubatch = mctx->get_ubatch();

        // 출력 개수 계산
        n_outputs = count_outputs(ubatch);

        // ubatch 처리 (process_ubatch)
        const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);

        // logits 추출
        if (t_logits && n_outputs > 0) {
            ggml_backend_tensor_get(t_logits, logits.data, ...);
        }

        // embeddings 추출
        if (t_embd && cparams.embeddings) {
            ggml_backend_tensor_get(t_embd, embd.data, ...);
        }

        n_outputs_prev += n_outputs;
    } while (mctx->next());  // 다음 ubatch로 진행

    return n_outputs_prev;
}
```

### 5.3 llama_batch → llama_ubatch 변환

`llama_batch_allocr`이 논리적 배치를 물리적 ubatch로 분할한다. 세 가지 분할 전략:

**split_simple(n_ubatch)**:
- 단순하게 n_ubatch 토큰씩 자른다
- 시퀀스 경계를 고려하지 않음
- 인코딩(encode)에서 사용

**split_equal(n_ubatch)**:
- 각 ubatch가 동일한 시퀀스 세트를 포함하도록 분할
- `ubatch.equal_seqs() == true` → 그래프 재사용 최적화
- 디코딩에서 선호됨

**split_seq(n_ubatch)**:
- 시퀀스 단위로 분할 (시퀀스가 쪼개지지 않음)
- 시퀀스 길이가 n_ubatch를 초과하면 잘림

```cpp
struct llama_ubatch {
    uint32_t n_tokens;       // 총 토큰 수
    uint32_t n_seq_tokens;   // 시퀀스 세트당 토큰 수
    uint32_t n_seqs;         // 시퀀스 세트 수
    uint32_t n_seqs_unq;     // 고유 시퀀스 ID 수

    llama_token  * token;    // 토큰 ID [n_tokens]
    float        * embd;     // 임베딩 직접 입력 [n_tokens * n_embd]
    llama_pos    * pos;      // 위치 [n_tokens]
    llama_seq_id ** seq_id;  // 시퀀스 ID [n_tokens][LLAMA_MAX_SEQ]
    llama_seq_id * seq_id_unq;  // 고유 시퀀스 ID [n_seqs_unq]
    int8_t       * output;   // 출력 플래그 [n_tokens]
    bool           equal_seqs;  // 시퀀스 세트가 동일한가
    void         * data;     // 데이터 소유권 (재사용 체크용)
};
```

### 5.4 process_ubatch() 상세

```cpp
llm_graph_result * llama_context::process_ubatch(
    const llama_ubatch & ubatch,
    llm_graph_type gtype,
    llama_memory_context_i * mctx,
    ggml_status & ret) {

    // 1. 메모리 컨텍스트 적용 (KV 캐시 업데이트)
    if (mctx && !mctx->apply()) {
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // 2. 그래프 재사용 체크
    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        // 재사용: 파이프라인 병렬 시 동기화 필요
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }
        n_reused++;
    } else {
        // 새 그래프 빌드
        res->reset();
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(),
            cparams.cb_eval, cparams.cb_eval_user_data);

        gf = model.build_graph(gparams);  // 모델별 그래프 빌드

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    // 3. 입력 데이터 설정
    res->set_inputs(&ubatch);

    // 4. 그래프 실행
    const auto status = graph_compute(gf, ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;
    return res;
}
```

---

## 6. KV Cache 심층 분석

### 6.1 Ring Buffer 구현

`llama_kv_cache` (`llama-kv-cache.h`)는 **ring buffer** 방식으로 KV 캐시를 관리한다:

```cpp
class llama_kv_cache {
    struct kv_layer {
        uint32_t il;              // 레이어 인덱스
        ggml_tensor * k;          // K 캐시 텐서 [n_embd_head_k, n_ctx, n_layer]
        ggml_tensor * v;          // V 캐시 텐서 [n_embd_head_v, n_ctx, n_layer]
        std::vector<ggml_tensor *> k_stream;  // 스트림별 K
        std::vector<ggml_tensor *> v_stream;  // 스트림별 V
    };

    bool v_trans = true;  // V 텐서가 전치됨 (성능 최적화)
    const uint32_t n_seq_max = 1;  // 최대 시퀀스 수
    const uint32_t n_stream  = 1;  // 스트림 수
    const uint32_t n_pad = 1;      // 패딩
    const uint32_t n_swa = 0;      // Sliding Window 크기 (0 = 비활성)

    std::vector<uint32_t> v_heads;  // 각 스트림의 현재 헤드 위치
    std::vector<llama_kv_cells> v_cells;  // 셀 데이터 (시퀀스 ID, 위치 추적)
    std::vector<uint32_t> seq_to_stream;  // 시퀀스 ID → 스트림 ID 매핑
    std::vector<kv_layer> layers;  // 레이어별 KV 텐서
    std::unordered_map<int32_t, int32_t> map_layer_ids;  // 모델 레이어 → KV 캐시 레이어
};
```

**`llama_kv_cells`**: 각 셀은 KV 캐시의 단일 위치를 나타낸다. 시퀀스 ID, 위치(pos), 점유 상태를 추적한다.

### 6.2 find_slot() — 슬롯 할당 알고리즘

`find_slot()` (`llama-kv-cache.cpp:817`)은 ring buffer에서 ubatch를 수용할 연속된 공간을 찾는다:

```cpp
llama_kv_cache::slot_info llama_kv_cache::find_slot(
    const llama_ubatch & ubatch, bool cont) const {

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res;
    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];
        const auto stream_id = seq_to_stream[seq_id];
        const auto & cells = v_cells[stream_id];
        const uint32_t head_cur = v_heads[stream_id];

        // head_cur부터 빈 셀 탐색
        uint32_t head_new = head_cur;
        uint32_t n_used = cells.get_used();
        uint32_t n_free = get_size() - n_used;

        if (n_free < n_tokens) {
            // 공간 부족: 가장 오래된 시퀀스 제거 (eviction)
            // ...
        }

        // 빈 셀 찾기 (ring buffer 순회)
        for (uint32_t i = 0; i < cells.size(); ++i) {
            uint32_t idx = (head_cur + i) % cells.size();
            if (cells.is_empty(idx) || cells.can_overwrite(idx, seq_id)) {
                // 빈 셀 발견
                res.idxs[s].push_back(idx);
                if (res.idxs[s].size() == n_tokens) break;
            }
        }

        res.strm.push_back(stream_id);
    }

    return res;
}
```

**`slot_info` 구조**:

```cpp
struct slot_info {
    uint32_t s0;                    // 최소 시퀀스 ID
    uint32_t s1;                    // 최대 시퀀스 ID
    std::vector<uint32_t> strm;     // 스트림 ID [n_seqs]
    std::vector<std::vector<uint32_t>> idxs;  // 셀 인덱스 [n_seqs][n_tokens]
};
```

### 6.3 KV Cache Indexing for Attention

Attention 계산 시 KV 캐시에서 필요한 행을 추출하기 위해 `ggml_get_rows`를 사용한다:

```cpp
// KV 캐시에서 K 행 추출
ggml_tensor * llama_kv_cache::build_input_k_idxs(
    ggml_context * ctx, const llama_ubatch & ubatch) const {
    // k_idxs: ubatch의 각 토큰이 참조해야 할 KV 캐시 인덱스
    // [n_kv] — attention에 필요한 모든 KV 위치
}

// ggml_get_rows로 실제 데이터 추출
ggml_tensor * K_cache = ggml_get_rows(ctx, kv_cache.get_k(ctx, il, n_kv, sinfo), k_idxs);
ggml_tensor * V_cache = ggml_get_rows(ctx, kv_cache.get_v(ctx, il, n_kv, sinfo), v_idxs);
```

**`k_idxs` / `v_idxs` 생성 로직**:
1. 현재 ubatch의 토큰에 해당하는 KV 캐시 셀 인덱스를 수집
2. 동일한 시퀀스의 이전 토큰들도 포함 (causal attention)
3. SWA 모델의 경우 sliding window 내의 토큰만 포함

### 6.4 Sliding Window Attention (iswa)

`llama_kv_cache_iswa_context`는 Sliding Window Attention을 지원한다:

- **Base 캐시**: 전체 컨텍스트 윈도우
- **SWA 캐시**: 슬라이딩 윈도우 내의 최근 토큰만

```cpp
class llama_kv_cache_iswa_context : public llama_memory_context_i {
    llama_kv_cache_context * base;  // 전체 컨텍스트
    llama_kv_cache_context * swa;   // 슬라이딩 윈도우
};
```

`llm_graph_input_attn_kv_iswa`는 두 개의 attention mask를 관리한다:
- `self_kq_mask`: base 캐시용 (전체 컨텍스트)
- `self_kq_mask_swa`: SWA 캐시용 (윈도우 내 토큰만)

---

## 7. Attention 메커니즘

### 7.1 Flash Attention (ggml_flash_attn_ext)

Flash Attention은 O(N^2) 메모리 복잡도를 O(N)으로 줄이는 알고리즘이다. llama.cpp에서는 백엔드별로 최적화된 커널을 사용한다:

**CUDA**: `ggml_flash_attn_ext` → `flash_attn_ext` CUDA 커널
- Tiling: Q, K, V를 블록 단위로 분할
- SRAM 내 Softmax: 중간 결과를 GPU SRAM에 유지
- 재계산(Recomputation): 메모리 절약을 위해 forward pass에서 attention weight 재계산

**Metal**: Apple Silicon의 Metal Performance Shaders 활용
**CPU**: 최적화된 블럭 단위 구현

### 7.2 KV Cache + Attention 결합

Attention 계산의 전체 데이터 흐름:

```
1. Qcur = RoPE(Wq @ norm(x))     [n_embd_head, n_tokens, n_head]
2. Kcur = RoPE(Wk @ norm(x))     [n_embd_head, n_tokens, n_head_kv]
3. Vcur = Wv @ norm(x)           [n_embd_head, n_tokens, n_head_kv]

4. KV 캐시에 저장:
   cpy_k(Kcur, k_idxs) → KV 캐시의 k_idxs 위치에 Kcur 복사
   cpy_v(Vcur, v_idxs) → KV 캐시의 v_idxs 위치에 Vcur 복사

5. KV 캐시에서 읽기:
   K_cache = get_rows(KV_K, k_idxs)  [n_embd_head, n_kv, n_head_kv]
   V_cache = get_rows(KV_V, v_idxs)  [n_embd_head, n_kv, n_head_kv]

6. Flash Attention:
   attn = flash_attn_ext(Qcur, K_cache, V_cache, mask, scale)
   [n_embd_head, n_tokens, n_head]

7. Output Projection:
   output = Wo @ attn  [n_embd, n_tokens]
```

### 7.3 GQA (Grouped Query Attention)

GQA는 쿼리 헤드 수(n_head)와 KV 헤드 수(n_head_kv)를 다르게 설정하여 메모리를 절약한다:

```
n_head = 32, n_head_kv = 4 → 각 KV 헤드가 8개의 쿼리 헤드와 공유

Q: [n_embd_head, n_tokens, 32]
K: [n_embd_head, n_kv, 4]     → broadcast to [n_embd_head, n_kv, 32]
V: [n_embd_head, n_kv, 4]     → broadcast to [n_embd_head, n_kv, 32]
```

Flash Attention 내부에서 브로드캐스팅이 자동으로 처리된다:
- `n_head % n_head_kv == 0` 조건 필요
- CUDA 커널에서 KV를 반복하여 쿼리 헤드 수에 맞춤

---

## 8. 양자화 상세

### 8.1 지원 ggml_type 전체 목록

```c
enum ggml_type {
    GGML_TYPE_F32     = 0,   // 32-bit float (4.0 bytes/element)
    GGML_TYPE_F16     = 1,   // 16-bit float (2.0 bytes/element)
    GGML_TYPE_Q4_0    = 2,   // 4-bit quantized, method 0 (0.5 bytes/element)
    GGML_TYPE_Q4_1    = 3,   // 4-bit quantized, method 1
    GGML_TYPE_Q5_0    = 6,   // 5-bit quantized, method 0
    GGML_TYPE_Q5_1    = 7,   // 5-bit quantized, method 1
    GGML_TYPE_Q8_0    = 8,   // 8-bit quantized, method 0 (1.0 bytes/element)
    GGML_TYPE_Q8_1    = 9,   // 8-bit quantized, method 1
    GGML_TYPE_Q2_K    = 10,  // K-quant 2-bit
    GGML_TYPE_Q3_K    = 11,  // K-quant 3-bit
    GGML_TYPE_Q4_K    = 12,  // K-quant 4-bit
    GGML_TYPE_Q5_K    = 13,  // K-quant 5-bit
    GGML_TYPE_Q6_K    = 14,  // K-quant 6-bit
    GGML_TYPE_Q8_K    = 15,  // K-quant 8-bit (내부용)
    GGML_TYPE_IQ2_XXS = 16,  // Importance Quantization 2-bit XXS
    GGML_TYPE_IQ2_XS  = 17,  // Importance Quantization 2-bit XS
    GGML_TYPE_IQ3_XXS = 18,  // Importance Quantization 3-bit XXS
    GGML_TYPE_IQ1_S   = 19,  // Importance Quantization 1-bit S
    GGML_TYPE_IQ4_NL  = 20,  // Importance Quantization 4-bit NL (no LUT)
    GGML_TYPE_IQ3_S   = 21,  // Importance Quantization 3-bit S
    GGML_TYPE_IQ2_S   = 22,  // Importance Quantization 2-bit S
    GGML_TYPE_IQ4_XS  = 23,  // Importance Quantization 4-bit XS
    GGML_TYPE_I8      = 24,  // 8-bit integer
    GGML_TYPE_I16     = 25,  // 16-bit integer
    GGML_TYPE_I32     = 26,  // 32-bit integer
    GGML_TYPE_I64     = 27,  // 64-bit integer
    GGML_TYPE_F64     = 28,  // 64-bit float
    GGML_TYPE_IQ1_M   = 29,  // Importance Quantization 1-bit M
    GGML_TYPE_BF16    = 30,  // Brain float 16
    GGML_TYPE_TQ1_0   = 34,  // Ternary Quantization 1-bit
    GGML_TYPE_TQ2_0   = 35,  // Ternary Quantization 2-bit
    GGML_TYPE_MXFP4   = 39,  // MXFP4 (1 block) — OpenAI gpt-oss
    GGML_TYPE_NVFP4   = 40,  // NVFP4 (4 blocks, E4M3 scale) — NVIDIA
    GGML_TYPE_Q1_0    = 41,  // 1-bit quantized
    GGML_TYPE_COUNT   = 42,
};
```

### 8.2 K-Quantization 원리

K-quantization은 **슈퍼블록** 개념을 도입하여 양자화 정확도를 향상시킨다:

**Q4_0 구조** (기본 4-bit):
```
블록 크기: 32개 요소
구조: [d: f16] + [qs: 32 * 4 bits = 16 bytes]
총 크기: 18 bytes (32 * 0.5625 bytes/element)
dequant: value[i] = d * (qs[i/2] >> (4*(i%2)) & 0xF - 8)
```

**Q4_K 구조** (K-quant 4-bit):
```
슈퍼블록: 256개 요소 = 8 * 블록(32개)
구조:
  [d: f16] [dmin: f16] — 슈퍼블록 전역 스케일
  [scales: 8 * 6 bits] — 블록별 스케일
  [qs: 256 * 4 bits] — 양자화 값
총 크기: 144 bytes (256 * 0.5625 bytes/element)
```

**IQ2_XXS 구조** (Importance Quantization):
```
블록 크기: 256개 요소
구조:
  [d: f16] — 스케일
  [qs: 256 * 2 bits = 64 bytes] — 양자화 값
  [scales: 4 bits per 32 elements] — 부분 스케일
총 크기: ~74 bytes (256 * 0.289 bytes/element ≈ 2.31 bits/element)
```

**MXFP4** (Microscaling FP4):
```
블록 크기: 32개 요소
구조:
  [scale: E4M3 float8] — 블록 공통 스케일
  [fp4_values: 32 * 4 bits = 16 bytes]
총 크기: 18 bytes
OpenAI gpt-oss 모델에서 사용. NVIDIA RTX 5000 시리즈 하드웨어 가속 지원.
```

**NVFP4** (NVIDIA FP4):
```
4개의 MXFP4 블록을 그룹화
공통 E4M3 스케일 사용
NVIDIA Blackwell 아키텍처 최적화
```

### 8.3 Dequantization during MatMul

`ggml_mul_mat`에서 양자화 가중치는 **on-the-fly dequantization**된다:

```c
// CPU 백엔드 (ggml-cpu)
// Q4_0 matmul:
for (int i = 0; i < ne01; i++) {
    for (int j = 0; j < ne11; j++) {
        float sum = 0;
        for (int l = 0; l < ne00; l += QK4_0) {
            // 블록 디코딩
            float d0 = ((ggml_fp16_t *) src0->data)[l/QK4_0].d;
            const uint8_t * qs = src0->data + l/2;

            for (int k = 0; k < QK4_0; k++) {
                float v0 = d0 * ((qs[k/2] >> (4*(k%2)) & 0xF) - 8);
                float v1 = ((float *) src1->data)[j*ne10 + l + k];
                sum += v0 * v1;
            }
        }
        ((float *) dst->data)[j*ne0 + i] = sum;
    }
}
```

**SIMD 최적화**:
- AVX2/AVX512: 256/512-bit 레지스터로 병렬 dequantize + dot product
- NEON (ARM): 128-bit 레지스터 활용
- CUDA: warp-level primitives로 블록 단위 병렬 처리

---

## 9. 배치 처리

### 9.1 llama_batch vs llama_ubatch

| 특성 | llama_batch | llama_ubatch |
|---|---|---|
| **역할** | 사용자 입력 (논리적) | GPU/CPU 처리 단위 (물리적) |
| **크기 제약** | 무제한 | ≤ `n_ubatch` |
| **시퀀스 구조** | 임의의 시퀀스 혼합 | 균일한 시퀀스 세트 |
| **데이터 소유** | 사용자 메모리 | 내부 할당 또는 뷰 |
| **생성 위치** | 사용자 API 호출 | `llama_batch_allocr::split_*()` |

```cpp
struct llama_batch {
    int32_t n_tokens;
    llama_token  * token;     // 토큰 ID [n_tokens]
    float        * embd;      // 임베딩 직접 입력 [n_tokens * n_embd]
    llama_pos    * pos;       // 위치 [n_tokens]
    int32_t      * n_seq_id;  // 각 토큰의 시퀀스 수 [n_tokens]
    llama_seq_id ** seq_id;   // 시퀀스 ID [n_tokens][n_seq_id[i]]
    int8_t       * logits;    // 출력 필요 플래그 [n_tokens]
};
```

### 9.2 n_batch vs n_ubatch

| 파라미터 | 의미 | 영향 |
|---|---|---|
| `n_batch` | API 레벨 배치 크기 | 한 번의 `decode()` 호출에서 처리할 최대 토큰 수 |
| `n_ubatch` | 하드웨어 레벨 배치 크기 | 실제 GPU/CPU가 한 번에 처리하는 토큰 수 |
| 관계 | `n_ubatch <= n_batch` | `n_batch > n_ubatch`이면 내부적으로 여러 ubatch로 분할 |

**성능 트레이드오프**:
- `n_ubatch`가 크면: GPU 활용도 ↑, 메모리 사용량 ↑
- `n_ubatch`가 작으면: 메모리 사용량 ↓, 그래프 재사용 기회 ↑

---

## 10. 데이터 흐름 종합

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        llama.cpp 추론 파이프라인                             │
└─────────────────────────────────────────────────────────────────────────────┘

  사용자 입력 (텍스트)
       │
       ▼
  ┌─────────────┐
  │ llama_vocab  │  토크나이저 (BPE/SPM/WordPiece)
  │ tokenize()   │  텍스트 → token_ids[]
  └──────┬───────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    llama_batch                           │
  │  token[n_tokens], pos[n_tokens], seq_id[n_tokens][]     │
  │  logits[n_tokens] (출력 필요 플래그)                     │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │              llama_batch_allocr                          │
  │  init(batch) → split_simple/equal/seq(n_ubatch)         │
  │                                                         │
  │  ┌──────────┐  ┌──────────┐         ┌──────────┐       │
  │  │ ubatch[0]│  │ ubatch[1]│  ...    │ ubatch[N]│       │
  │  │ n_tokens │  │ n_tokens │         │ n_tokens │       │
  │  │ ≤n_ubatch│  │ ≤n_ubatch│         │ ≤n_ubatch│       │
  │  └────┬─────┘  └────┬─────┘         └────┬─────┘       │
  └───────┼─────────────┼────────────────────┼──────────────┘
          │             │                    │
          ▼             ▼                    ▼
  ┌───────────────────────────────────────────────────────────┐
  │              llama_context::decode()                       │
  │                                                           │
  │  for each ubatch:                                         │
  │  ┌─────────────────────────────────────────────────────┐ │
  │  │ 1. memory->init_batch()                              │ │
  │  │    - KV 캐시 슬롯 할당 (find_slot, ring buffer)     │ │
  │  │    - slot_info 생성 (idxs, strm)                    │ │
  │  │    - llama_kv_cache_context 반환                    │ │
  │  └────────────────────┬────────────────────────────────┘ │
  │                       │                                  │
  │  ┌────────────────────▼────────────────────────────────┐ │
  │  │ 2. model.build_graph(gparams)                       │ │
  │  │    - llm_graph_context 생성                         │ │
  │  │    - LLaMA: token_embd → [layer: norm→qkv→attn→    │ │
  │  │      residual→norm→ffn→residual] → output_norm→     │ │
  │  │      output                                         │ │
  │  │    - ggml_cgraph 반환 (nodes[], leafs[])            │ │
  │  └────────────────────┬────────────────────────────────┘ │
  │                       │                                  │
  │  ┌────────────────────▼────────────────────────────────┐ │
  │  │ 3. sched->graph_compute(gf)                         │ │
  │  │    - 5-pass 백엔드 할당                             │ │
  │  │    - 그래프 분할 (splits[])                         │ │
  │  │    - 백엔드별 실행 (CUDA→Metal→CPU)                 │ │
  │  │    - 중간 텐서 single-use 해제                      │ │
  │  └────────────────────┬────────────────────────────────┘ │
  │                       │                                  │
  │  ┌────────────────────▼────────────────────────────────┐ │
  │  │ 4. memory->apply()                                  │ │
  │  │    - KV 캐시 업데이트 (cpy_k, cpy_v)                │ │
  │  │    - ggml_get_rows로 k_idxs/v_idxs 기반 저장        │ │
  │  └────────────────────┬────────────────────────────────┘ │
  │                       │                                  │
  │  ┌────────────────────▼────────────────────────────────┐ │
  │  │ 5. logits 추출                                      │ │
  │  │    - ggml_backend_tensor_get(t_logits, logits_buf)  │ │
  │  │    - [n_outputs][n_vocab]                           │ │
  │  └─────────────────────────────────────────────────────┘ │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │                   llama_sampler                          │
  │  temperature → top_k → top_p → min_p → penalty → sample │
  │                                                         │
  │  logits[n_vocab] → argmax / multinomial → next_token    │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
                  다음 토큰 (사용자에게 반환)
                         │
                         ▼
              (반복: 새 토큰을 batch에 추가 → decode)
```

---

## 11. 핵심 파일 레퍼런스

| 파일 경로 | 핵심 내용 |
|---|---|
| `ggml/include/ggml.h` | GGML 공개 API: `ggml_tensor`, 연산 함수, 타입 열거형 |
| `ggml/src/ggml-impl.h` | GGML 내부 구현: `ggml_cgraph`, 메모리 풀, 해시 집합 |
| `ggml/src/ggml.c` | GGML 코어: 텐서 연산 구현, 그래프 빌드, 메모리 관리 |
| `ggml/src/ggml-backend.cpp` | 백엔드 스케줄러: 5-pass 할당, 그래프 분할, 실행 |
| `ggml/src/ggml-cpu/` | CPU 백엔드: SIMD 최적화, 양자화 dequantize |
| `ggml/src/ggml-cuda/` | CUDA 백엔드: Flash Attention, matmul 커널 |
| `ggml/src/ggml-metal/` | Metal 백엔드: Apple Silicon 최적화 |
| `src/llama.cpp` | 공개 C API 구현 (`llama_init_from_file`, `llama_decode` 등) |
| `src/llama-model.cpp` | 모델 로딩: GGUF 파싱, 가중치 로드, 아키텍처 감지 |
| `src/llama-context.cpp` | 추론 실행: `encode()`, `decode()`, `process_ubatch()` |
| `src/llama-graph.cpp` | 그래프 빌드: `llm_graph_context`, `build_*` 프리미티브, LoRA |
| `src/llama-graph.h` | 그래프 인터페이스: `llm_graph_result`, `can_reuse()`, 입력 타입 |
| `src/llama-kv-cache.cpp` | KV 캐시: ring buffer, `find_slot()`, shift/copy |
| `src/llama-kv-cache.h` | KV 캐시 헤더: `llama_kv_cache`, `slot_info`, `kv_layer` |
| `src/llama-batch.cpp` | 배치 분할: `llama_batch_allocr`, `split_*()` |
| `src/llama-memory.cpp` | 메모리 추상화: `llama_memory_i` 팩토리 |
| `src/llama-sampler.cpp` | 토큰 샘플링: temperature, top_p, top_k, penalty |
| `src/llama-vocab.cpp` | 토크나이저: BPE, SPM, WordPiece, HuggingFace 토크나이저 |
| `src/llama-adapter.cpp` | LoRA/Control Vector: `llama_adapter_lora`, `llama_adapter_cvec` |
| `src/llama-adapter.h` | LoRA 헤더: `llama_adapter_lora_weight`, `ab_map` |
| `src/llama-arch.cpp` | 아키텍처 메타데이터: 140+ 모델 이름, 텐서 매핑 |
| `src/models/llama.cpp` | LLaMA 그래프 빌드: 153라인 템플릿 |
| `src/models/*.cpp` | 모델별 그래프 빌드 (114개 파일) |
| `include/llama.h` | 공개 C API 헤더 |

---

## 부록: 코드 레벨 핵심 상수

| 상수 | 값 | 의미 |
|---|---|---|
| `GGML_MAX_DIMS` | 4 | 텐서 최대 차원 수 |
| `GGML_MAX_SRC` | 10 | 연산 최대 소스 텐서 수 |
| `GGML_MAX_NAME` | 64 | 텐서 이름 최대 길이 |
| `GGML_MAX_OP_PARAMS` | 64 | 연산 파라미터 최대 크기 (byte) |
| `GGML_DEFAULT_GRAPH_SIZE` | 2048 | 기본 그래프 최대 노드 수 |
| `GGML_SCHED_MAX_BACKENDS` | 16 | 스케줄러 최대 백엔드 수 |
| `GGML_SCHED_MAX_SPLIT_INPUTS` | 16 | split 최대 입력 텐서 수 |
| `GGML_SCHED_MAX_COPIES` | 4 | 파이프라인 최대 복사본 수 |
| `LLAMA_MAX_SEQ` | 4 | 최대 시퀀스 수 per 토큰 |
