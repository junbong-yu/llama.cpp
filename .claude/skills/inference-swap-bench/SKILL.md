---
name: inference-swap-bench
description: "llama-inference-bench 를 이용해 실제 모델 추론 중 baseline vs 사용자 커널 A/B 벤치를 재현 가능하게 실행하고 JSON 을 수집하는 스킬. 'inference-bench 돌려줘', '모델 추론 스왑', 'baseline 과 user 양쪽 JSON 필요', 'pp/tg t/s 비교' 요청 시 사용."
---

# Inference Swap Bench

junbong 의 `llama-inference-bench` 를 활용하여 **baseline / user 커널 두 조건을 동일 하드웨어·동일 모델·동일 스레드 수로 2회 돌려 JSON 을 수집**하는 표준 실행 스킬.

## 입력

- `model_path`: gguf 파일 절대 경로
- `ops`: 대상 op 집합 (`relu,gelu,silu,swiglu` 중 일부)
- `threads` (기본 host 코어 수), `warmup` (기본 3), `min_time` (기본 10s)
- `seeds`: 실행 반복 수 (기본 3)
- `label_prefix`: JSON 파일명 prefix

## 실행 절차

### 1. 전제 확인

- `build-arm/bin/llama-inference-bench` (혹은 대상 빌드 디렉토리) 존재 확인
- `git rev-parse HEAD` 로 현재 커밋 해시 기록 (JSON 메타에 저장)
- 워킹 트리 dirty 여부 확인 후 경고

### 2. Baseline run

사용자 커널을 비활성화(전역 플래그 off 또는 variant 미등록) 상태로 실행:

```bash
for seed in 1 2 3; do
  ./build-arm/bin/llama-inference-bench \
    -m <model_path> \
    --ops <ops> \
    --use-custom-kernels 0 \
    --warmup <warmup> --min-time <min_time> \
    --threads <threads> --seed $seed \
    --json _workspace/run_<ts>_baseline_seed${seed}.json
done
```

### 3. User kernel run

`--use-custom-kernels 1` 로 동일 명령 반복. 나머지 인자는 baseline 과 bit-identical.

### 4. 메타 기록

실행한 커맨드라인 전체, host 정보(`uname -a`, `sysctl machdep.cpu.brand_string`), cmake 빌드 옵션을 `_workspace/run_<ts>_meta.json` 에 저장.

### 5. 정합성 플래그 확인

inference-bench 는 실행 중 정합성 오차가 tolerance 를 넘으면 stderr 에 경고를 남긴다 (bench-harness 로직). 모든 run 의 stderr 에서 해당 경고 유무를 스캔하여 결과 JSON 옆에 `correctness_ok: bool` 로 저장한다.

## 출력

- `_workspace/run_<ts>_baseline_seed{1,2,3}.json`
- `_workspace/run_<ts>_user_seed{1,2,3}.json`
- `_workspace/run_<ts>_meta.json`
- 콘솔 요약 표 (한 눈에 볼 수 있도록 pp/tg t/s 평균만)

## 에러 핸들링

- `--use-custom-kernels` 옵션 미지원 (inference-bench 버전 문제): kernel-author 에 플래그 추가 diff 요청
- 훅이 op 를 한 번도 잡지 못함 (카운터 0): 모델 경로가 해당 op 를 타지 않음. perf-analyst 에 전달 → 다른 layer/모델 제안
- seed 간 분산이 큼 (CV > 5%): min_time 증가 후 재실행

## 재현 체크리스트

- [ ] baseline / user 가 같은 커밋에서 빌드됨
- [ ] 같은 모델·양자화·스레드 수
- [ ] 같은 warmup/min_time
- [ ] 3 seed 이상
- [ ] `correctness_ok: true` 모두 확인

## 제약

- 현재 inference-bench 훅은 CPU extra_buffer_type 기반이므로 `-ngl 0` (CPU only) 권장
- F16/BF16 모델은 훅의 F32 경로 가정과 충돌할 수 있음 → correctness-verifier 가 사전에 확인
