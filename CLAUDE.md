IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## 하네스: llama.cpp 개발 지원

**목표:** llama.cpp 코드 분석, 빌드/테스트, 코드 리뷰를 전문 에이전트 팀으로 체계화

**트리거:** llama.cpp 코드 분석, 빌드, 테스트, 리뷰 등 개발 관련 작업 요청 시 `llama-cpp-dev` 스킬을 사용하라. 단순 질문은 직접 응답 가능.

## 하네스: 커널 개발·성능 검증

**목표:** 사용자가 구현한 활성 함수(ReLU/GeLU/SwiGLU 등) 최적화 커널을 llama.cpp 에 통합하고 실제 모델 추론에서 baseline 과 A/B 성능·정합성 비교

**트리거:** "ReLU/GeLU/SwiGLU 커널 추가", "내 커널로 모델 돌려봐", "inference-bench 스왑 실험", "baseline 과 speedup 비교", "SIMD/NEON/SVE 활성 함수", "정합성 확인", "ARM 빌드 가능?", "before/after 비교" 등 커널 단위 실험 요청 시 `kernel-dev-perf` 스킬을 사용하라. llama.cpp 전반 개발 질문은 `llama-cpp-dev` 유지.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-04-16 | 초기 구성 | 전체 | - |
| 2026-04-23 | 커널 개발·성능 검증 하네스 추가 (에이전트 5: kernel-author, bench-runner, perf-analyst, correctness-verifier, platform-gatekeeper / 스킬 4: kernel-dev-perf, activation-kernel-scaffold, inference-swap-bench, perf-compare) | `.claude/agents/`, `.claude/skills/`, CLAUDE.md | 사용자 커널 A/B 실험 워크플로우 표준화 |
