---
name: perf-compare
description: "inference-bench / kernel-bench 가 생성한 baseline·user JSON 쌍을 분석하여 활성 함수 op 별 speedup, 모델 전체 pp/tg t/s 변화, 회귀 감지 리포트를 작성. 'speedup 확인', 'before/after 비교', '성능 회귀', 'pp/tg 변화', '활성 함수 기여도' 시 사용."
---

# Perf Compare

`scripts/compare-kernels.py` 와 `scripts/extract-backend-results.py` 를 호출하여 JSON 쌍을 정량 비교하고, 결과를 마크다운 리포트로 정리하는 스킬.

## 입력

- `baseline_pattern`: `_workspace/run_<ts>_baseline_seed*.json` (glob)
- `user_pattern`: `_workspace/run_<ts>_user_seed*.json`
- `ops_filter` (optional): 분석 대상 op 집합
- `significance_threshold`: speedup 을 "유의미" 로 보는 기준 (기본 1.03 = 3% 개선)

## 처리 흐름

### 1. Seed 집계

- baseline 과 user 각각 seed 파일들을 평균·표준편차 집계
- op 별 `mean_ms, std_ms, calls` 테이블 생성
- 모델 전체 `pp_tps, tg_tps` 평균

### 2. compare-kernels.py 실행

```bash
python scripts/compare-kernels.py \
  --before _workspace/run_<ts>_baseline_seed1.json \
  --after  _workspace/run_<ts>_user_seed1.json \
  --format markdown \
  > _workspace/phase5_compare_raw.md
```

필요 시 seed 별로 반복 호출 후 평균 합성.

### 3. 해석 레이어

compare-kernels.py 출력에 아래 정보를 덧붙인다:

- **유의성**: speedup 이 `1 ± significance_threshold` 밖 & std 가 평균의 5% 미만이면 "유의"
- **기여도**: op 의 baseline 누적 시간이 모델 전체의 몇 % 인가 → 최적화 ROI 계산
- **정합성**: 메타 JSON 의 `correctness_ok` 가 모두 true 인지 확인. 하나라도 false 면 speedup 수치 대신 경고

### 4. 리포트 생성

`_workspace/phase5_perf_analyst_report.md` 에 마크다운:

```
# Perf Compare Report

## Summary
- 실행 시각 / 커밋 / 모델 / 스레드
- 정합성: PASS / FAIL (자세히)
- 대상 op, seed 수

## Per-op speedup
| op | baseline ms | user ms | speedup | std | 유의 |
|----|-------------|---------|---------|-----|------|
| RELU | ... | ... | 1.12x | 0.3% | O |
| GELU | ... | ... | 0.98x | 0.8% | X (개선 없음) |

## Model-level
- pp t/s: baseline X → user Y (ratio)
- tg t/s: baseline X → user Y (ratio)

## 해석
- op 별 기여도 (전체 런타임 대비 %)
- 유의 개선 op 와 회귀 op 분리
- 다음 실험 제안
```

## 에러 핸들링

- JSON 스키마 불일치: bench-runner 에 재실행 요청 (옵션 누락 가능성)
- 정합성 FAIL: speedup 숫자를 표기하지 않고 경고만 표시
- 기여도가 매우 작은 op 에서만 speedup 이 큼: "전체 모델에 미치는 영향 미미" 주석 추가

## 제약

- **정합성이 통과하지 않은 결과로 speedup 수치를 보고하지 않는다**
- seed 수가 3 미만이면 "측정 정밀도 불충분" 이라고 명시
- backend-bench 결과 처리는 `extract-backend-results.py` 분기로 별도 처리
