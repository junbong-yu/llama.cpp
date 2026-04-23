---
name: kernel-dev-perf
description: "사용자가 구현한 활성 함수(ReLU/GeLU/SwiGLU) 최적화 커널을 llama.cpp 에 통합하고 실제 모델 추론에서 baseline 과 A/B 성능·정합성 비교하는 오케스트레이터. 'ReLU/GeLU/SwiGLU 커널 추가', '내 커널로 모델 돌려봐', 'inference-bench 스왑 실험', 'baseline 과 speedup 비교', 'SIMD/NEON/SVE 활성 함수', '정합성 확인', 'ARM 빌드 가능?', '성능 회귀', 'before/after 비교' 등의 요청 시 사용. 후속 작업(재실행, 추가 variant, tolerance 조정, 다른 모델로 재현)에도 반드시 사용."
---

# Kernel Dev · Perf Orchestrator

사용자가 직접 작성한 활성 함수(ReLU/GeLU/SwiGLU 중심) 최적화 커널을 **실제 모델 추론 경로에 꽂아 넣고 기존 llama.cpp 구현과 A/B 비교** 하는 워크플로우를 전담한다. junbong 의 `kernel-bench-harness` 브랜치가 구축한 3-layer 툴킷(kernel-bench / inference-bench / backend-bench) 중 **inference-bench 경로가 1순위**.

기존 `llama-cpp-dev` 오케스트레이터(analyst/builder/reviewer)는 그대로 유지되며, 본 오케스트레이터는 커널 단위 실험에만 특화된다.

## 실행 모드: 서브 에이전트 (전문가 풀)

에이전트 간 실시간 통신이 구조적으로 불필요하다. 각 전문가가 독립 결과를 반환하면 충분하므로 `Agent` 도구 직접 호출로 진행한다.

## 에이전트 구성

| 에이전트 | subagent_type | 역할 | 호출 조건 |
|---------|--------------|------|----------|
| kernel-author | general-purpose | 사용자 커널 통합, 레지스트리 등록 | "커널 추가", "kernels-user.cpp 에 통합" |
| bench-runner | general-purpose | inference-bench / kernel-bench 실행 | "벤치 돌려줘", "JSON 수집" |
| perf-analyst | general-purpose | JSON 해석, speedup·회귀 분석 | "speedup", "before/after 비교", "회귀" |
| correctness-verifier | general-purpose | tolerance·정합성 판정 | "정합성", "오차", "NaN" |
| platform-gatekeeper | general-purpose | 빌드 매트릭스, ISA 플래그 | "ARM 빌드?", "SVE/NEON 가능?", "크로스 컴파일" |

> 기존 `analyst` 는 필요 시 보조로 호출 (예: 타깃 모델의 FFN 그래프 트레이스). `builder` 는 전체 프로젝트 빌드, `platform-gatekeeper` 는 아키텍처 매트릭스로 분담.

## 워크플로우

### Phase 0: 컨텍스트 확인

1. `_workspace/` 존재 여부 확인 (경로: `.claude/skills/kernel-dev-perf/_workspace/`)
2. 실행 모드 결정:
   - 미존재 → 초기 실행, `_workspace/` 생성
   - 존재 + 후속 요청 → 이전 JSON / 보고서 참조
   - 존재 + 새 입력 → 기존을 `_workspace_prev_<timestamp>/` 로 이동
3. 사용자 입력에서 타깃 op (relu/gelu/swiglu) 와 모델 경로 파싱

### Phase 1: 사전 준비 (필요 시)

SwiGLU 가 포함되고 타깃 모델의 ggml 경로가 불분명할 때만 analyst 호출:

```
Agent(analyst, Explore, opus):
  "<model_path> 의 FFN 이 ggml 그래프에서 어떤 op 시퀀스로 표현되는지,
   SwiGLU 가 fused (GGML_OP_SWIGLU) 인지 composite (SILU+MUL) 인지 보고."
```

### Phase 2: 통합·빌드

병렬 호출:

```
Agent(kernel-author, general-purpose, opus, run_in_background=true):
  사용자 커널 소스를 kernels-user.cpp 에 배치하고
  kernel-registry.cpp 에 variant="user" 로 등록.
  inference-bench.cpp 훅 op 리스트에 UNARY(+SWIGLU) 추가 diff 제안.

Agent(platform-gatekeeper, general-purpose, opus, run_in_background=true):
  host + 요구 ISA 기반 빌드 가능성 매트릭스 작성.
  필요한 CMake 플래그 / feature 탐지 패치 제안.
```

### Phase 3: 정합성 1차 검증

```
Agent(correctness-verifier, general-purpose, opus):
  op 별 tolerance 프리셋 적용하여 kernel-bench 모드로 스모크 테스트.
  FAIL 시 이후 Phase 중단.
```

### Phase 4: 인퍼런스 벤치 실행

```
Agent(bench-runner, general-purpose, opus):
  baseline 설정으로 1회 + user 커널 활성화 1회 (최소 3 seed).
  결과 JSON 을 _workspace/run_<ts>_{baseline,user}.json 에 저장.
```

### Phase 5: 성능 분석

```
Agent(perf-analyst, general-purpose, opus):
  두 JSON 비교, op 별 speedup 표 + 모델 전체 t/s 변화 + 회귀 플래그.
  _workspace/phase5_perf_analyst_report.md 생성.
```

### Phase 6: 결과 통합 및 보고

오케스트레이터가 Phase 2~5 산출물을 종합하여 사용자에게 보고. 회귀 감지 시 다음 실험 제안 포함.

## 데이터 흐름

```
[사용자 커널 + 타깃 op + 모델]
       |
[Phase 1: (선택) analyst FFN 트레이스]
       |
[Phase 2: kernel-author 통합 ∥ platform-gatekeeper 매트릭스]
       |
[Phase 3: correctness-verifier 스모크]
       |   (PASS 시만 진행)
[Phase 4: bench-runner baseline+user 실행]
       |
[Phase 5: perf-analyst JSON 비교]
       |
[Phase 6: 사용자 보고]
```

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| 빌드 실패 | platform-gatekeeper 가 원인(ISA/플래그/링크) 분류 → kernel-author 에 수정 요청 |
| 정합성 FAIL | Phase 4 이후 진행 중단. correctness-verifier 가 실패 패턴을 사용자에게 반환 |
| Speedup 측정 노이즈만 | perf-analyst 가 "개선 없음" 이 아닌 "정밀도 한계" 로 보고, seed 증가 제안 |
| 훅이 op 를 한 번도 잡지 못함 | 모델 경로가 해당 op 를 타지 않는 것 → 다른 모델/layer 제안 |

## AGENTS.md 준수

- 최적화 로직은 사용자 소유. 에이전트는 스캐폴딩·어댑터·분석만 수행.
- 커밋/PR 메시지 자동 생성 없음.
- 사용자가 커널의 정당성을 이해·소유하도록 설명을 동반.

## 테스트 시나리오

### 정상 흐름: ReLU 사용자 커널 A/B 비교
1. 사용자: "내 ReLU 구현을 kernels-user.cpp 에 넣고 QWen3-0.6B 로 비교해줘"
2. Phase 2: kernel-author 통합 + platform-gatekeeper = "Apple M1 + NEON, 빌드 OK"
3. Phase 3: correctness-verifier PASS (bit-exact)
4. Phase 4: bench-runner baseline=2.0s, user=1.9s seed 3회
5. Phase 5: perf-analyst "RELU op speedup 1.12x, 전체 tg t/s 1.03x"
6. 사용자에게 리포트 전달

### 에러 흐름: SVE 커널을 Apple M1 에서 요청
1. 사용자: "SVE 버전 GeLU 넣어줘"
2. Phase 2: platform-gatekeeper "Apple M1 은 SVE 하드웨어 미지원, 컴파일만 가능"
3. kernel-author 가 대안 제시: (a) NEON 변형, (b) 크로스 컴파일로 .o 만 확보
4. 사용자 선택 대기 후 Phase 3 로 진행
