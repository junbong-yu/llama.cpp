---
name: llama-cpp-dev
description: "llama.cpp 개발 지원 오케스트레이터. 코드 분석, 아키텍처 탐색, 빌드/테스트, 코드 리뷰, 변경 영향 분석, 모델 구현 분석 등 llama.cpp 개발과 관련된 모든 작업을 수행. 'llama.cpp 코드 분석', '빌드 에러', '코드 리뷰', '테스트 실행', '모델 구조 분석', '의존성 추적', '변경 영향', 'ggml 구조' 등의 요청 시 이 스킬을 사용할 것. 후속 작업: 결과 수정, 다시 분석, 추가 리뷰, 재빌드, 업데이트, 보완 요청 시에도 반드시 사용."
---

# llama.cpp Dev Orchestrator

llama.cpp 개발 작업을 전문 에이전트에게 라우팅하여 수행하는 오케스트레이터.

## 실행 모드: 서브 에이전트 (전문가 풀 패턴)

요청 유형에 따라 적절한 전문가 에이전트를 선택 호출한다. 에이전트 간 실시간 통신이 불필요하고, 각 전문가가 독립적으로 결과를 반환하면 충분하므로 서브 에이전트 모드를 사용한다.

## 에이전트 구성

| 에이전트 | subagent_type | 역할 | 호출 조건 |
|---------|--------------|------|----------|
| analyst | Explore | 코드 분석, 아키텍처 탐색, 의존성 추적 | 코드 구조/흐름 파악, 심볼 추적, 모델 분석 요청 |
| builder | general-purpose | 빌드 설정, 테스트 실행, 에러 진단 | 빌드/컴파일/테스트 관련 요청 |
| reviewer | general-purpose | 코드 리뷰, 변경 영향 분석 | 코드 검토, diff 분석, 품질 검증 요청 |

## 워크플로우

### Phase 0: 컨텍스트 확인

1. `_workspace/` 디렉토리 존재 여부 확인
2. 실행 모드 결정:
   - `_workspace/` 미존재 -> 초기 실행
   - `_workspace/` 존재 + 사용자가 후속 요청 -> 이전 결과 참조하여 추가 작업
3. 후속 요청 시 이전 산출물 경로를 에이전트 프롬프트에 포함

### Phase 1: 요청 분류 및 라우팅

사용자 요청을 분석하여 적절한 에이전트를 선택한다:

| 요청 유형 | 키워드/패턴 | 라우팅 대상 |
|----------|-----------|-----------|
| 코드 분석 | "구조", "아키텍처", "의존성", "흐름", "어디서", "어떻게 동작" | analyst |
| 모델 분석 | "모델", "src/models/", "아키텍처 구현", "레이어" | analyst |
| 심볼 추적 | "함수", "호출", "참조", "정의", "사용처" | analyst |
| 빌드 | "빌드", "컴파일", "cmake", "에러", "링크" | builder |
| 테스트 | "테스트", "ctest", "실패", "test-" | builder |
| 코드 리뷰 | "리뷰", "검토", "diff", "변경", "PR" | reviewer |
| 품질 검증 | "메모리", "안전성", "성능", "버그" | reviewer |
| 복합 요청 | 여러 유형에 해당 | 해당 에이전트 병렬 호출 |

### Phase 2: 에이전트 호출

**단일 에이전트 호출:**

```
Agent(
  description: "{작업 요약}",
  subagent_type: "{analyst|builder|reviewer 중 선택}",
  model: "opus",
  prompt: "
    당신은 llama.cpp {역할} 전문가입니다.
    .claude/agents/{name}.md의 원칙을 따르세요.

    [작업 지시]
    {사용자 요청 상세}

    [프로젝트 루트]
    /Users/junbongyu/src/SR/NNT_LMA/llama.cpp

    [이전 산출물] (있는 경우)
    {_workspace/ 내 관련 파일 경로}
  "
)
```

**복합 요청 시 병렬 호출:**

여러 에이전트가 필요한 경우 `run_in_background: true`로 병렬 호출:

```
Agent(analyst, run_in_background: true, model: "opus", ...)
Agent(reviewer, run_in_background: true, model: "opus", ...)
```

### Phase 3: 결과 통합

1. 에이전트 반환값 수집
2. 복합 요청이었다면 결과를 통합하여 일관된 보고서 작성
3. 산출물을 `_workspace/{phase}_{agent}_{artifact}.md`에 저장
4. 사용자에게 결과 요약 보고

## 데이터 흐름

```
[사용자 요청]
     |
[Phase 1: 분류]
     |
     ├── 코드 분석 -> Agent(analyst, Explore, opus)
     ├── 빌드/테스트 -> Agent(builder, general-purpose, opus)
     ├── 코드 리뷰 -> Agent(reviewer, general-purpose, opus)
     └── 복합 -> 병렬 Agent 호출
     |
[Phase 3: 결과 통합]
     |
[사용자에게 보고]
```

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| 에이전트 1개 실패 | 1회 재시도. 재실패 시 에러 내용 보고하고 나머지 결과로 진행 |
| 분류 불확실 | analyst를 기본으로 호출 (코드 분석이 가장 범용적) |
| 컨텍스트 초과 | 분석 범위를 좁혀서 재호출 |
| 빌드 타임아웃 | 타임아웃 설정 늘려서 재시도, 또는 부분 빌드로 전환 |

## AGENTS.md 준수 사항

llama.cpp의 AGENTS.md 정책을 준수한다:
- AI는 보조 역할로만 사용. 코드 생성보다 분석/이해/리뷰에 집중
- PR/커밋 메시지를 자동 생성하지 않음
- 사용자가 코드를 이해하고 소유할 수 있도록 가이드 제공

## 테스트 시나리오

### 정상 흐름: 코드 분석

1. 사용자: "llama_decode 함수의 호출 체인을 분석해줘"
2. Phase 1: "심볼 추적" -> analyst 라우팅
3. Phase 2: Agent(analyst, Explore, opus) 호출
4. Phase 3: analyst가 llama_decode의 정의, 호출자, 피호출 함수 목록 반환
5. 결과를 사용자에게 보고

### 정상 흐름: 복합 요청

1. 사용자: "이 변경사항을 리뷰하고 테스트도 돌려줘"
2. Phase 1: "코드 리뷰" + "테스트" -> reviewer + builder 병렬
3. Phase 2: 두 에이전트 병렬 호출
4. Phase 3: 리뷰 결과 + 테스트 결과 통합 보고

### 에러 흐름

1. Phase 2에서 builder가 빌드 타임아웃으로 실패
2. 타임아웃 늘려 1회 재시도
3. 재실패 시 "빌드 타임아웃 - 부분 빌드 또는 수동 실행 권장" 보고
4. reviewer 결과는 정상 전달
