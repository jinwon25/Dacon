# 목표 1185 — v352/v353 TrackMan privileged-distillation 후보

## 결론

Public `1182.94969702`의 v345 위에서, 과거에 독립적으로 고정된 v38
TrackMan privileged-distillation(PFD) 학생 방향이 두 개의 연속 시간 구간에서
다시 양수였다. 이 결과를 바탕으로 다음 두 ZIP을 생성했다.

| 우선순위 | 파일 | 의미 | SHA-256 |
|---|---|---|---|
| 1 | `submit_v353.zip` | 고정 레시피를 2024 전체에 재학습한 2025 배포형 | `EEB3841AA67325A96BB35940AEBE0E9BD0B83BAEB9FCAA30C3055D0474E918B6` |
| 2 | `submit_v352.zip` | 2024년 3–7월 학습 모델을 그대로 보존한 감사형 | `59C4BEFC00841FB3A3FC25CE227D0C044D04733F0218BFEEFE59FD7239E53E67` |

제출 1순위는 **v353**이다. v352는 학습창 확장 위험을 피한 사전 고정 fallback이며,
동일 계열의 Public 강도 탐색 용도로 반복 제출하지 않는다.

## 야구·모델링 가설

현재 입력에는 투구 직전 상황과 누적 제구율은 있지만 현재 투구의 구종, 구속, 회전,
무브먼트와 릴리스 위치는 없다. 공식 제공 TrackMan의 이 물리량은 학습 행에서 제구 성공의
조건부 구조를 더 잘 설명하는 교사를 만들 수 있다. 다만 현재 테스트 투구의 TrackMan 값을
알 수 없으므로, 최종 학생은 물리량 자체가 아니라 그 교사와 안전 교사의 OOF 예측 차이를
공식 pre-pitch 행 피처만으로 학습한다.

- 현재 투구 TrackMan: labelled train의 game-group cross-fit 교사에서만 사용
- 학생 입력: 공식 메인 테이블 43개 행 피처
- 제외: `pitcher_id`, `batter_id`, 두 team ID, 모든 현재 투구 TrackMan 열
- 적용 경로: Regular이면서 team 13이 포함된 `R_ANCHOR`
- 고정 강도: `0.40`
- 추론: 각 행을 독립적으로 처리하며 다른 테스트 행이나 테스트 집계를 사용하지 않음

v345의 R_ANCHOR 저랭크 성분과 PFD 증분의 late-2024 상관은 `0.003795`로 거의 0이었다.
즉 동일 ASOF prior의 단순 재보정이 아니라, 학습 시 물리적 투구 결과를 교사로 본 독립 방향이다.

## 시간축 검증

route, 학생 종류, 강도와 모델 구조는 v38의 2024년 6–7월 선택에서 이미 고정됐다.
v345가 만들어진 뒤에는 이 고정 방향만 새 부모 위에 재기준화했다.

| 학생 학습창 | 정직한 다음 구간 | v345 대비 전체 gain | 세부 월 |
|---|---|---:|---|
| 2024년 3–5월 | 2024년 6–7월 | `+0.976131` | 6월 `+1.274537`, 7월 `+0.574192` |
| 2024년 3–7월, 3 seed | 2024년 8–10월 | `+0.952127` | 8월 `+1.546622`, 9월 `+0.250979`, 10월 적용 행 없음 |

후반 감사의 R_ANCHOR 적용 행만 보면 gain은 `+5.445612`, 적용 행은 13,432개,
전체 행 평균 절대 이동은 `0.000158452`였다. 세 개 refit seed의 전체 gain도 각각
`+0.961678`, `+0.964954`, `+0.929256`으로 모두 양수였다.

불확실성은 남는다. 후반 적용 행의 bootstrap p05는 투수 `-1.6091`, 투수×타자
`-5.5294`, 시간 블록 `-0.4652`이고 White reality-check는 `p=0.0785`였다.
따라서 +0.95를 확정적 Public 증가로 해석하지 않으며, v345를 보존한 작은 상보 증분으로만 쓴다.

## v352와 v353

v352는 2024년 3–7월로 학습해 8–10월을 보지 않은 기존 3-seed 학생 모델을 그대로
패키징한다. 저장 당시 Windows CRLF 변환으로 LightGBM `tree_sizes` 오프셋이 깨져 있었기
때문에, builder가 텍스트를 LF로 정규화하고 `model_str`에서 로드한다. 패키지 런타임은
저장된 late-2024 보정을 최대 오차 `1.40e-16` 이내로 재현했다.

v353은 선택·강도·피처·seed를 바꾸지 않고 공식 2024 정렬 행 전체로 표준 최종 재학습했다.

| 항목 | 값 |
|---|---:|
| 정렬 학습 행 | `216,555` |
| 경기 그룹 | `702` |
| privileged teacher gain | `+338.074174` |
| teacher delta 표준편차 | `0.0241172` |
| 학생 ensemble과 teacher delta 상관 | `0.178310` |

v353 자체를 2024 정답으로 다시 평가하면 재학습 오염이 되므로 그렇게 선택하지 않았다.
승격 근거는 앞의 두 strict-forward 구간이고, 2024 전체 적합은 2025 배포를 위한 표준 refit이다.

## 패키지 감사

| 항목 | v352 | v353 |
|---|---:|---:|
| 크기 | `101,266,381` bytes | `101,267,764` bytes |
| ZIP 멤버 | `194` | `194` |
| CRC | 통과 | 통과 |
| 재빌드 SHA 일치 | 통과 | 통과 |
| 공식 2025 fixture 수식 오차 | `1.11e-16` | `1.11e-16` |
| 보호 경로 parity | `1.11e-16` | `1.11e-16` |
| shuffle / partition 오차 | `0 / 0` | `1.11e-16 / 1.11e-16` |
| 확률 finite / 범위 | 통과 | 통과 |

전체 저장소 테스트는 `727 passed, 22 skipped`다. v345도 실제 부모 v334부터 다시 빌드해
v343 SHA `165C074A...A7C9`, v345 SHA `D44578D...5AAAA`가 원본과 정확히 일치했다.
v353 standalone을 245,789행 proxy로 처음부터 실행한 시간은 `192.887초`였고,
ZIP root·CRC·출력 행 수와 ID·확률 finite/range 검사를 모두 통과했다. 정적 금지 연산은 0개,
singleton/shuffle/partition 최대 오차는 각각 `2.22e-16/1.11e-16/1.11e-16`이었다.

## 재현

```powershell
$alignment = (Resolve-Path `
  '..\..\투구 제구 성공 확률 예측 AI 온라인 해커톤\artifacts\trackman_privileged_20260816').Path

python -m src.archive.v352_trackman_pfd_rebase_v345 `
  --train-csv data/train.csv `
  --v345-axes artifacts/v345_transition_workload_beta_cell_v335_20260831_01/selected_axes.npz `
  --v38-audit <v38-artifact-dir>/audit_late_2024.npz `
  --output-dir artifacts/v352_trackman_pfd_rebase_v345_20260831_01

python -m src.champion.v353_refit_trackman_pfd_students `
  --project . `
  --alignment-dir $alignment `
  --output-dir artifacts/v353_refit_trackman_pfd_students_20260831_01

python -m src.champion.v353_build_refit_trackman_pfd_package `
  --source-zip artifacts/v345_transition_workload_beta_package_20260831_01/submit_v345.zip `
  --model-dir artifacts/v353_refit_trackman_pfd_students_20260831_01 `
  --refit-summary artifacts/v353_refit_trackman_pfd_students_20260831_01/summary.json `
  --validation-summary artifacts/v352_trackman_pfd_rebase_v345_20260831_01/summary.json `
  --output-dir artifacts/v353_refit_trackman_pfd_package_20260831_01

python -m src.audit_v352_trackman_pfd `
  --v345-zip artifacts/v345_transition_workload_beta_package_20260831_01/submit_v345.zip `
  --v352-zip artifacts/v353_refit_trackman_pfd_package_20260831_01/submit_v353.zip `
  --input-csv data/test.csv `
  --output-json artifacts/v353_refit_trackman_pfd_package_20260831_01/runtime_fixture_audit.json
```

최종 전달본은 작업 폴더 루트의 `submit_v353.zip`이며, v345와 그 재현 패키지는 변경하지 않았다.
