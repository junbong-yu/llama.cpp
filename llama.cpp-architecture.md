# llama.cpp 소프트웨어 아키텍처 분석

## 목차

1. [개요](#1-개요)
2. [전체 계층 구조](#2-전체-계층-구조)
3. [핵심 아키텍처 패턴](#3-핵심-아키텍처-패턴)
   - [3.1 Model-Context 분리](#31-model-context-분리)
   - [3.2 Strategy Pattern — 다형적 메모리 관리](#32-strategy-pattern--다형적-메모리-관리)
   - [3.3 Template Method Pattern — 모델별 그래프 빌드](#33-template-method-pattern--모델별-그래프-빌드)
   - [3.4 Batch Splitting — Logical → Physical 분리](#34-batch-splitting--logical--physical-분리)
4. [데이터 흐름 — 추론 파이프라인](#4-데이터-흐름--추론-파이프라인)
5. [컴포넌트 의존성 그래프](#5-컴포넌트-의존성-그래프)
6. [핵심 타입 및 구조체](#6-핵심-타입-및-구조체)
   - [6.1 ggml_tensor — 기본 텐서](#61-ggml_tensor--기본-텐서)
   - [6.2 ggml_cgraph — 계산 그래프](#62-ggml_cgraph--계산-그래프)
   - [6.3 llama_layer — 레이어 가중치](#63-llama_layer--레이어-가중치)
   - [6.4 llama_model — 모델 컨테이너](#64-llama_model--모델-컨테이너)
   - [6.5 llama_context — 추론 실행기](#65-llama_context--추론-실행기)
   - [6.6 llama_memory_i — 메모리 추상화](#66-llama_memory_i--메모리-추상화)
   - [6.7 llama_batch / llama_ubatch — 배치 처리](#67-llama_batch--llama_ubatch--배치-처리)
7. [설계 결정과 트레이드오프](#7-설계-결정과-트레이드오프)
   - [7.1 왜 Layer가 객체가 아닌가?](#71-왜-layer가-객체가-아닌가)
   - [7.2 왜 Graph가 재사용 가능한가?](#72-왜-graph가-재사용-가능한가)
   - [7.3 왜 C API인가?](#73-왜-c-api인가)
   - [7.4 MECE한 관심사 분리](#74-mece한-관심사-분리)
8. [nntrainer와의 구조 비교](#8-nntrainer와의-구조-비교)
9. [클래스 다이어그램](#9-클래스-다이어그램)

---

## 1. 개요

llama.cpp는 **140개 이상의 LLM 아키텍처**를 지원하는 경량 추론 엔진이다. GGUF 포맷의 모델 파일을 로드하여 CPU/GPU에서 효율적으로 추론을 수행한다. 전체 아키텍처는 **4계층 구조**로 설계되어 있으며, **C API를 통한 완전한 캡슐화**가 특징이다.

- **코드베이스 규모**: ~63개 최상위 디렉토리, ~114개 모델 구현 파일
- **지원 백엔드**: CPU, CUDA, Metal, Vulkan, SYCL, ROCm
- **핵심 설계 원칙**: Model-Context 분리, Data-Oriented Design, Graph 재사용

---

## 2. 전체 계층 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    PUBLIC API LAYER                              │
│  include/llama.h  ──  순수 C API (extern "C")                   │
│  common/          ──  CLI 도구, 예제, 서버                       │
│  ── 모델 로딩, 추론 실행, 토큰 샘플링, KV 캐시 조작              │
├─────────────────────────────────────────────────────────────────┤
│                  APPLICATION LOGIC LAYER                         │
│  src/llama-model.*       ──  모델 로딩 / 가중치 컨테이너         │
│  src/llama-context.*     ──  추론 실행 엔진 + 상태 관리          │
│  src/llama-graph.*       ──  계산 그래프 빌더 인터페이스          │
│  src/llama-memory.*      ──  KV Cache / 추상 메모리 관리         │
│  src/llama-arch.*        ──  140+ 아키텍처 열거형                │
│  src/llama-batch.*       ──  배치 분할 (logical→physical)        │
│  src/llama-sampler.*     ──  토큰 샘플링 파이프라인              │
│  src/llama-vocab.*       ──  토크나이저                          │
│  ── 단일 책임 원칙으로 모듈화된 컴포넌트                          │
├─────────────────────────────────────────────────────────────────┤
│                  MODEL IMPLEMENTATIONS (114 files)               │
│  src/models/llama.cpp, gpt2.cpp, gemma*.cpp, qwen*.cpp, ...     │
│  ── 각 모델 아키텍처별 계산 그래프 빌드 함수                      │
│  ── llm_graph_context의 공통 빌드 프리미티브를 조합              │
├─────────────────────────────────────────────────────────────────┤
│                  COMPUTE ENGINE LAYER (ggml)                     │
│  ggml/include/ggml.h      ──  텐서 연산 라이브러리 API           │
│  ggml/src/ggml.c          ──  400+ 텐서 연산 구현                │
│  ggml/include/ggml-backend.h  ──  백엔드 스케줄러                │
│  ggml/src/ggml-cpu/       ──  CPU 최적화 구현 (SIMD)             │
│  ggml/src/ggml-cuda/      ──  CUDA 백엔드                        │
│  ggml/src/ggml-metal/     ──  Metal 백엔드 (Apple Silicon)       │
│  ggml/src/ggml-vulkan/    ──  Vulkan 백엔드                      │
│  ── llama에 대한 의존성 없음 — 독립적인 텐서 라이브러리           │
└─────────────────────────────────────────────────────────────────┘
```

**의존성 방향**: Public API → Application Logic → Model Implementations → ggml
의존성은 항상 **위에서 아래로**만 흐르며, 하위 계층은 상위 계층을 전혀 알지 못한다.

---

## 3. 핵심 아키텍처 패턴

### 3.1 Model-Context 분리

llama.cpp에서 가장 근본적인 아키텍처 결정은 **`llama_model`(데이터)과 `llama_context`(실행)의 완전한 분리**다.

```
┌─────────────────────────────────┐     ┌──────────────────────────────────┐
│         llama_model             │     │        llama_context             │
│  (Immutable Data Container)     │     │  (Runtime Executor + State)      │
├─────────────────────────────────┤     ├──────────────────────────────────┤
│ llama_hparams   hparams         │──참조▶│ const llama_model&  model        │
│ llama_vocab     vocab           │     │ llama_cparams       cparams      │
│ vector<llama_layer> layers[N]   │     │ llama_memory_i*     memory       │
│ ggml_tensor*    tok_embd        │     │ ggml_backend_sched  sched        │
│ ggml_tensor*    output          │     │ buffer_view<float>  logits       │
│ ggml_tensor*    output_norm     │     │ buffer_view<float>  embd         │
│ ...                             │     │ sampling_info       sampling     │
├─────────────────────────────────┤     ├──────────────────────────────────┤
│ load_arch() / load_hparams()    │     │ encode() / decode()              │
│ load_vocab() / load_tensors()   │     │ graph_reserve() / graph_compute()│
│ build_graph(params) → ggml_cgraph│    │ memory_update() / state_load()   │
└─────────────────────────────────┘     └──────────────────────────────────┘
```

**설계 의도**:
- 하나의 `llama_model`을 여러 `llama_context`가 공유 가능 (멀티 테넌트 서빙)
- 모델은 파일에서 로드된 후 **절대 변경되지 않음** (불변, 완전한 thread-safe)
- 각 Context는 독립적인 샘플링 설정, KV 캐시, 시퀀스 상태를 유지

### 3.2 Strategy Pattern — 다형적 메모리 관리

`llama_memory_i`는 **추상 인터페이스**로, 모델 아키텍처에 따라 6가지 이상의 구현체가 존재한다.

```
llama_memory_i (abstract interface)         사용처
├── llama_kv_cache_context               ── 표준 Transformer (LLaMA, GPT 등)
├── llama_kv_cache_iswa_context          ── Sliding Window Attention (Mistral, Gemma2 등)
├── llama_memory_recurrent_context       ── 순환 신경망 (Mamba, RWKV 등)
├── llama_memory_hybrid_context          ── Transformer+Recurrent 혼합 (Jamba, Granite Hybrid 등)
├── llama_memory_hybrid_iswa_context     ── iSWA+Transformer+Recurrent (Cohere2 등)
└── (향후 확장 가능)
```

**인터페이스 라이프사이클**:
```
init_batch() → next() → apply() → 반복
     │           │         └── KV 캐시 갱신, 메모리 상태 반영
     │           └── 다음 ubatch로 진행
     └── 입력 배치 분할, KV 슬롯 할당
```

### 3.3 Template Method Pattern — 모델별 그래프 빌드

`llm_graph_context`는 `build_norm()`, `build_qkv()`, `build_attn()`, `build_ffn()`, `build_moe_ffn()` 등의 **공통 그래프 빌드 프리미티브**를 제공한다. 114개 모델 파일은 이 프리미티브들의 **호출 순서와 조합**만 정의한다.

```
llm_graph_context (공통 프리미티브 제공)
├── build_inp_embd(tok_embd)          ── 입력 임베딩
├── build_inp_pos()                   ── 위치 인코딩
├── build_norm(cur, mw, mb, type)     ── RMS Norm / Layer Norm
├── build_qkv(layer, cur, ...)        ── Q/K/V 프로젝션
├── build_attn(inp, wo, q, k, v, ...) ── Attention 연산
├── build_ffn(cur, gate, up, down)    ── SwiGLU FFN
├── build_moe_ffn(cur, ...)           ── Mixture of Experts FFN
├── build_rs(s, state_copy, ...)      ── Recurrent State 관리
├── build_cvec(cur, il)               ── Control Vector 적용
├── build_lora_mm(w, cur)             ── LoRA 어댑터 적용
└── build_pooling(cls, ...)           ── 풀링 (분류/임베딩 모델)
```

**예시**: LLaMA 모델의 그래프 빌드 (`src/models/llama.cpp`)
```
1. cur = build_inp_embd(model.tok_embd)
2. FOR each layer (0..n_layer-1):
   a. cur = build_norm(cur, layer.attn_norm, RMS)
   b. qkv = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv)
   c. cur = build_attn(inp_attn, layer.wo, qkv.q, qkv.k, qkv.v, kq_scale)
   d. cur = build_norm(cur, layer.ffn_norm, RMS)
   e. cur = build_ffn(cur, layer.ffn_gate, layer.ffn_up, layer.ffn_down, SiLU)
3. cur = build_norm(cur, model.output_norm, RMS)
4. cur = ggml_mul_mat(ctx0, model.output, cur)  // output projection
```

### 3.4 Batch Splitting — Logical → Physical 분리

```
llama_batch (논리적, 사용자 크기)
        │
        ▼
llama_batch_allocr
  ├── split_simple(n_ubatch)   ── 단순 분할
  ├── split_equal(n_ubatch)    ── 동일 시퀀스 세트 분할
  ├── split_seq(n_ubatch)      ── 시퀀스 세트 단위 분할
  └── split_reset()            ── 상태 리셋
        │
        ▼
llama_ubatch[] (물리적, GPU/CPU 처리 단위)
```

| 개념 | 제약 | 설명 |
|---|---|---|
| `llama_batch` | 무제한 | 사용자 입력 그대로의 논리적 배치 |
| `llama_ubatch` | ≤ `n_ubatch` | 물리적 처리 단위 (GPU/CPU 메모리 제약) |
| `n_batch` | API 파라미터 | 사용자 지정 논리적 배치 크기 |
| `n_ubatch` | 하드웨어 제약 | 실제 한 번에 처리할 수 있는 최대 토큰 수 |

---

## 4. 데이터 흐름 — 추론 파이프라인

```
                    사용자
                      │
                      ▼
            ┌──────────────────┐
            │   llama_batch    │  token_ids[], positions[], seq_ids[]
            └──────┬───────────┘
                   │
                   ▼
        ┌─────────────────────────┐
        │  llama_batch_allocr     │
        │  init() → split_*()     │
        └────────┬────────────────┘
                 │
    ┌────────────┴────────────┐
    ▼                         ▼
┌────────────┐         ┌──────────────┐
│  ubatch[0] │  ....   │  ubatch[N-1] │
└─────┬──────┘         └──────┬───────┘
      │                       │
      ▼                       ▼
┌──────────────────────────────────────────┐
│        llama_context::decode(batch)       │
│                                          │
│  for each ubatch:                        │
│   1. memory->init_batch()  ◀── KV cache  │
│   2. model.build_graph()   ◀── 모델별 구현│
│      └→ llm_graph_context                │
│         └→ ggml_cgraph (계산 그래프)      │
│   3. sched->graph_compute() ◀── GPU/CPU  │
│   4. memory->apply()       ◀── KV 업데이트│
│   5. logits 반환                         │
└─────────────────────┬────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ llama_sampler  │  ── temperature, top_p, top_k 등
              └───────┬───────┘
                      │
                      ▼
              다음 토큰 (사용자에게 반환)
```

---

## 5. 컴포넌트 의존성 그래프

```
include/llama.h ────────────────────────────────────── (Public C API)
       │
       ├──▶ src/llama-model.h
       │         │
       │         ├──▶ src/llama-arch.h ──── 140+ 아키텍처 열거형, 텐서 이름 매핑
       │         ├──▶ src/llama-vocab.h ──── 토크나이저 (BPE, SPM, WordPiece, ...)
       │         ├──▶ src/llama-hparams.h ── 모델 하이퍼파라미터
       │         ├──▶ src/llama-graph.h ──── build_graph() 인터페이스
       │         └──▶ src/llama-memory.h ─── create_memory() 인터페이스
       │
       ├──▶ src/llama-context.h
       │         │
       │         ├──▶ src/llama-cparams.h ── 컨텍스트 파라미터 (n_ctx, n_batch, ...)
       │         ├──▶ src/llama-memory.h ── KV Cache / Recurrent State
       │         ├──▶ src/llama-graph.h ──── 그래프 빌드 + 실행
       │         ├──▶ src/llama-sampler.h ── 토큰 샘플링
       │         └──▶ src/llama-adapter.h ── LoRA / Control Vector
       │
       ├──▶ src/llama-batch.h ──── batch → ubatch 분할
       │
       └──▶ ggml/include/ggml.h ─────── 텐서, 연산, 그래프
            ggml/include/ggml-backend.h ── 스케줄러, 디바이스
            ggml/include/ggml-cpu.h ───── CPU 백엔드
            ggml/include/ggml-opt.h ───── 옵티마이저 (학습용)
```

---

## 6. 핵심 타입 및 구조체

### 6.1 ggml_tensor — 기본 텐서

```c
struct ggml_tensor {
    enum ggml_type type;           // F32, F16, Q4_0, Q8_1, BF16, ...
    struct ggml_backend_buffer * buffer;
    int64_t ne[GGML_MAX_DIMS];    // tensor shape (최대 4차원)
    size_t  nb[GGML_MAX_DIMS];    // strides
    enum ggml_op op;              // 연산 타입 (연산 결과 텐서인 경우)
    int32_t flags;                // INPUT, OUTPUT, PARAM, LOSS, COMPUTE
    struct ggml_tensor * src[GGML_MAX_SRC];  // 소스 텐서 (연산 입력)
    void * data;                  // 실제 데이터 포인터
    char name[GGML_MAX_NAME];     // 텐서 이름 (예: "blk.0.attn_q.weight")
    // ...
};
```

- `ne[]`와 `nb[]`로 n차원 텐서를 표현
- `op`과 `src[]`로 연산 그래프의 노드 역할
- `name`은 모델 로딩 시 텐서 식별자로 사용

### 6.2 ggml_cgraph — 계산 그래프

```c
struct ggml_cgraph {
    int size;      // 최대 노드 수
    int n_nodes;   // 활성 노드 수
    int n_leafs;   // 리프 노드 수 (가중치 등)

    struct ggml_tensor ** nodes;      // 연산 결과 텐서
    struct ggml_tensor ** grads;      // 그래디언트 출력
    struct ggml_tensor ** grad_accs;  // 그래디언트 누적기
    struct ggml_tensor ** leafs;      // 상수 텐서 (가중치)

    int32_t * use_counts;
    enum ggml_cgraph_eval_order order;
    uint64_t uid;  // 그래프 식별자 (캐싱용)
};
```

- `ggml_build_forward_expand(gf, tensor)`로 그래프에 노드 추가
- `ggml_backend_sched_graph_compute(sched, gf)`로 실행
- `uid`를 통한 그래프 재사용 여부 판단

### 6.3 llama_layer — 레이어 가중치

```cpp
struct llama_layer {
    // ── Attention 정규화 ──
    ggml_tensor * attn_norm, * attn_norm_b;
    ggml_tensor * attn_q_norm, * attn_k_norm;  // Q/K 정규화 (일부 모델)

    // ── Attention 프로젝션 ──
    ggml_tensor * wq, * wk, * wv, * wo;        // Q/K/V/O 가중치
    ggml_tensor * wqkv;                         // Fused QKV (일부 모델)

    // ── FFN 정규화 ──
    ggml_tensor * ffn_norm, * ffn_norm_b;

    // ── FFN 프로젝션 ──
    ggml_tensor * ffn_gate, * ffn_up, * ffn_down;  // SwiGLU FFN

    // ── MoE (Mixture of Experts) ──
    ggml_tensor * ffn_gate_inp, * ffn_gate_exps;
    ggml_tensor * ffn_down_exps, * ffn_up_exps;

    // ── SSM (Mamba / RWKV) ──
    ggml_tensor * ssm_in, * ssm_x, * ssm_dt, * ssm_out;

    // ── RWKV Time Mix ──
    ggml_tensor * time_mix_w1, * time_mix_w2;
    ggml_tensor * time_mix_lerp_x, * time_mix_lerp_w, ...;

    // ── 비트넷 / 양자화 스케일 ──
    ggml_tensor * wq_s, * wk_s, * wv_s, * wo_s, ...;

    // ── 기타 아키텍처 특화 텐서 (140+ 모델 지원) ──
    // ...
};
```

- **Data-Oriented Design**: 모든 가중치를 하나의 구조체에 평탄화
- 특정 아키텍처에만 존재하는 텐서는 `nullptr`로 비워둠
- 140개 이상의 아키텍처를 단일 구조체로 커버

### 6.4 llama_model — 모델 컨테이너

```cpp
struct llama_model {
    llm_type type;        // 모델 크기 식별자 (7B, 70B, 405B, ...)
    llm_arch arch;        // 아키텍처 식별자 (LLAMA, GPT2, MAMBA, ...)
    std::string name;

    llama_hparams hparams; // n_embd, n_head, n_layer, n_ctx_train, ...
    llama_vocab   vocab;   // 토크나이저

    // ── 전역 가중치 ──
    ggml_tensor * tok_embd;     // 토큰 임베딩 [vocab_size, n_embd]
    ggml_tensor * output_norm;  // 최종 정규화
    ggml_tensor * output;       // 출력 프로젝션 [n_embd, vocab_size]

    // ── 레이어 가중치 ──
    std::vector<llama_layer> layers;  // layers[0..n_layer-1]

    // ── 디바이스 ──
    std::vector<llama_device> devices;

    // ── GGUF 메타데이터 ──
    std::unordered_map<std::string, std::string> gguf_kv;

    // ── 메서드 ──
    void load_arch(llama_model_loader & ml);
    void load_hparams(llama_model_loader & ml);
    void load_vocab(llama_model_loader & ml);
    bool load_tensors(llama_model_loader & ml);
    ggml_cgraph * build_graph(const llm_graph_params & params) const;
    llama_memory_i * create_memory(...) const;
};
```

### 6.5 llama_context — 추론 실행기

```cpp
struct llama_context {
    // ── 참조 ──
    const llama_model & model;
    llama_cparams cparams;

    // ── 상태 관리 ──
    std::unique_ptr<llama_memory_i> memory;   // KV Cache
    llama_adapter_cvec_ptr  cvec;             // Control Vector
    llama_adapter_loras_ptr loras;            // LoRA adapters

    // ── 실행 인프라 ──
    ggml_backend_sched_t sched;               // 백엔드 스케줄러
    ggml_backend_t backend_cpu;
    std::vector<ggml_backend_ptr> backends;
    ggml_threadpool_t threadpool;

    // ── 출력 버퍼 ──
    buffer_view<float> logits;   // [n_outputs][n_vocab]
    buffer_view<float> embd;     // [n_outputs][n_embd]

    // ── 그래프 재사용 ──
    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // ── 샘플링 ──
    struct sampling_info { ... } sampling;

    // ── 핵심 메서드 ──
    int encode(const llama_batch & batch);
    int decode(const llama_batch & batch);
    uint32_t graph_max_nodes(uint32_t n_tokens) const;
    llm_graph_result * process_ubatch(...);
    void memory_update(bool optimize);
    float * get_logits();
};
```

### 6.6 llama_memory_i — 메모리 추상화

```cpp
struct llama_memory_i {
    virtual ~llama_memory_i() = default;

    // 배치 초기화 → ubatch[] 생성
    virtual llama_memory_context_ptr init_batch(
        llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) = 0;

    // 최대 크기 시뮬레이션 (버퍼 예약용)
    virtual llama_memory_context_ptr init_full() = 0;

    // KV 캐시 갱신 (shift, copy 등)
    virtual llama_memory_context_ptr init_update(
        llama_context * lctx, bool optimize) = 0;

    // 시퀀스 조작
    virtual void clear(bool data) = 0;
    virtual bool seq_rm(llama_seq_id, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp(llama_seq_id src, llama_seq_id dst, ...) = 0;
    virtual void seq_keep(llama_seq_id) = 0;
    virtual void seq_add(llama_seq_id, ..., llama_pos delta) = 0;

    // 상태 저장/복원
    virtual void state_write(llama_io_write_i & io, ...) = 0;
    virtual void state_read(llama_io_read_i & io, ...) = 0;
};
```

### 6.7 llama_batch / llama_ubatch — 배치 처리

```cpp
// 사용자 입력 배치 (논리적)
struct llama_batch {
    int32_t n_tokens;
    llama_token  * token;    // 토큰 ID
    float        * embd;     // 또는 임베딩 직접 입력
    llama_pos    * pos;      // 위치
    int32_t      * n_seq_id; // 시퀀스 수
    llama_seq_id ** seq_id;  // 시퀀스 ID
    int8_t       * logits;   // 출력 여부 플래그
};

// 물리적 처리 단위 (μ-batch)
struct llama_ubatch {
    uint32_t n_tokens;       // 총 토큰 수
    uint32_t n_seq_tokens;   // 시퀀스 세트당 토큰
    uint32_t n_seqs;         // 시퀀스 세트 수
    uint32_t n_seqs_unq;     // 고유 시퀀스 ID 수

    llama_token  * token;
    float        * embd;
    llama_pos    * pos;
    llama_seq_id ** seq_id;
    llama_seq_id * seq_id_unq;
    int8_t       * output;
};
```

---

## 7. 설계 결정과 트레이드오프

### 7.1 왜 Layer가 객체가 아닌가?

대부분의 딥러닝 프레임워크(PyTorch, TensorFlow, nntrainer)는 `Layer`를 클래스/인터페이스로 만들지만, llama.cpp는 `llama_layer`를 **순수 데이터 구조체**(모든 멤버가 `ggml_tensor*` 포인터)로 설계했다.

**이유**:
- 140개 이상의 모델 아키텍처마다 레이어 구조가 완전히 다름
  - 어떤 건 Q/K/V 분리, 어떤 건 fused QKV
  - 어떤 건 MLA (Multi-head Latent Attention), 어떤 건 SSM (State Space Model)
  - MoE, Dense, Hybrid 등 구조적 편차가 극심
- 각 아키텍처별로 상속 계층을 만들면 조합 폭발(combinatorial explosion)로 유지보수 불가능
- 그래프 빌드 함수(`build_*`)가 로직을 담당 → **Data-Oriented Design**
- 하나의 `llama_layer` struct가 140+ 아키텍처의 모든 텐서를 union처럼 포함

### 7.2 왜 Graph가 재사용 가능한가?

llama.cpp는 **계산 그래프 재사용**(Graph Reuse)이라는 독특한 최적화를 수행한다.

```cpp
// 컨텍스트 내부에 이전 그래프 결과 보존
llm_graph_result_ptr gf_res_prev;     // 이전 그래프
llm_graph_result_ptr gf_res_reserve;  // 예약 그래프

// 재사용 가능 여부 판단
bool can_reuse(const llm_graph_params & params) {
    // ubatch 구조 동일? n_tokens, n_seqs 동일?
    // 컨텍스트 설정 동일? embeddings, causal_attn 동일?
    // 아키텍처 동일? arch, gtype, cvec, loras 동일?
    return ubatch_match && params_match;
}
```

추론 시 매 토큰마다 그래프를 새로 만드는 대신:
1. 이전 그래프와 토폴로지가 동일하면 재사용
2. 입력 텐서 데이터만 업데이트 (가중치는 그대로)
3. `ggml_backend_sched_graph_compute()` 재호출

**효과**: 그래프 구축 오버헤드 제거, 특히 small-batch 추론에서 큰 성능 향상

### 7.3 왜 C API인가?

`llama.h`는 순수 C API(`extern "C"`)다.

| 장점 | 설명 |
|---|---|
| **언어 중립적** | Python, Java, Go, Rust, Swift, Node.js 등 모든 언어에서 FFI 호출 가능 |
| **ABI 안정성** | C++ name mangling 회피, 컴파일러 간 호환성 보장 |
| **배포 용이** | `.so` / `.dylib` / `.dll` 공유 라이브러리로 배포 |
| **바인딩 자동화** | `llama.h` 하나만 파싱하면 모든 언어 바인딩 생성 가능 |

내부 구현(`src/`)은 C++17을 사용하지만, 모든 공개 경계는 C로 단단히 캡슐화되어 있다.

### 7.4 MECE한 관심사 분리

각 소스 파일은 [단일 책임 원칙](https://en.wikipedia.org/wiki/Single-responsibility_principle)(SRP)을 엄격히 따른다.

| 파일 | 단일 책임 | 라인 수 |
|---|---|---|
| `llama-model.*` | 모델 파일 → 메모리 로딩, 가중치 컨테이너 | ~640 |
| `llama-context.*` | 추론 실행, 출력 관리, 상태 저장/복원 | ~349 |
| `llama-graph.*` | 계산 그래프 구축 인터페이스 + 입력 처리 | ~1064 |
| `llama-memory.*` | KV Cache / Recurrent State 추상화 | ~122 |
| `llama-batch.*` | 배치 정규화 및 μ-batch 분할 | ~173 |
| `llama-sampler.*` | 토큰 샘플링 파이프라인 | 별도 |
| `llama-vocab.*` | 토크나이저 (BPE, SPM, WordPiece 등) | 별도 |
| `llama-arch.*` | 140+ 아키텍처 메타데이터 정의 | ~638 |
| `llama-cparams.*` | 컨텍스트 파라미터 (n_ctx, n_batch 등) | ~47 |
| `llama-hparams.*` | 모델 하이퍼파라미터 | 별도 |
| `models/*.cpp` | 모델별 그래프 빌드 구현 (114개 파일) | 각 ~200-800 |

---

## 8. nntrainer와의 구조 비교

llama.cpp와 nntrainer는 동일한 5가지 개념(Engine, Context, Model, Graph, Layer)을 가지고 있으나, **추상화 수준**과 **설계 철학**이 근본적으로 다르다.

| 개념 | nntrainer | llama.cpp | 차이점 |
|---|---|---|---|
| **Engine** | `Singleton<Engine>` — Context 등록소 + 객체 팩토리 + 스레드풀 | `ggml_backend_sched` — GPU 백엔드 스케줄러, 연산 분배 | nntrainer: "객체 생성 공장", llama.cpp: "연산 실행 지휘자" |
| **Context** | `Context` (추상) — 백엔드별 Layer/Optimizer 팩토리 | `llama_context` (구상) — 추론 상태·메모리·출력 버퍼 통합 소유 | nntrainer: "컴파일 타임 백엔드 선택", llama.cpp: "런타임 실행 환경" |
| **Model** | `NeuralNetwork` — Graph + Optimizer + Dataset + Loss 통합 | `llama_model` — 순수 가중치 컨테이너 (불변 데이터) | nntrainer: "자기 완결적 학습기", llama.cpp: "데이터 보관소" |
| **Graph** | `NetworkGraph` — LayerNode의 위상 정렬 DAG, forward/backward 실행 | `ggml_cgraph` — 텐서 연산 노드의 DAG, 빌드 후 실행 | nntrainer: "Layer 수준 추상화", llama.cpp: "텐서 연산 수준 추상화" |
| **Layer** | `LayerNode` — 인터페이스 기반 일급 객체, forward/backward 캡슐화 | `llama_layer` — `ggml_tensor*` 집합 (수동적 데이터 구조체) | nntrainer: "객체지향", llama.cpp: "데이터지향" |

**아키텍처 스타일 비교**:

| | nntrainer | llama.cpp |
|---|---|---|
| 주 패러다임 | 객체지향 (상속, 인터페이스, 팩토리) | 데이터지향 (구조체, 자유 함수) |
| 확장 방식 | Layer 인터페이스 구현 + Context 등록 | 모델별 그래프 빌드 함수 추가 |
| 추론/학습 | 학습+추론 모두 지원 | 추론 중심 (학습은 ggml-opt로 보조) |
| 지원 모델 수 | 범용 (모든 딥러닝 모델 구성 가능) | 140+ LLM 아키텍처 (특화) |
| 그래프 표현 | Layer 수준 DAG (위상 정렬) | Tensor 연산 수준 DAG (ggml ops) |
| 메모리 관리 | Manager (그래프 내) | llama_memory_i (1급 추상화) |

---

## 9. 클래스 다이어그램

PlantUML 형식의 클래스 다이어그램은 별도 파일로 제공된다:
- [`llama.cpp-class-diagram.puml`](llama.cpp-class-diagram.puml)

---

## 부록: 아키텍처 패턴 요약

| 아키텍처 스타일 | 적용된 부분 | 설명 |
|---|---|---|
| **Layered Architecture** | C API → App Logic → Models → ggml | 의존성은 항상 위→아래 |
| **Data-Oriented Design** | `llama_layer` | 데이터와 로직을 분리, 구조체는 순수 데이터만 |
| **Strategy Pattern** | `llama_memory_i` | KV Cache 전략을 런타임에 교체 가능 |
| **Template Method** | `build_norm`, `build_qkv`, `build_attn`, `build_ffn` | 공통 알고리즘 골격 제공, 세부는 모델별로 |
| **Model-Context Separation** | `llama_model` ↔ `llama_context` | 불변 데이터 ↔ 가변 실행 상태 |
| **Adapter/Bridge** | `llama_batch_allocr` | `llama_batch` → `llama_ubatch[]` 변환 |
| **Facade** | `include/llama.h` (C API) | 내부 C++ 구현을 단순한 C 인터페이스로 감춤 |
| **Flyweight** | `ggml_cgraph` 재사용 | 그래프 토폴로지 캐싱으로 빌드 오버헤드 제거 |
| **Singleton** | `ggml_backend_reg` | 백엔드 레지스트리 (ggml 내부) |
