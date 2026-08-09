# Rolling-origin damped drift 단일 사전 고정 실험

## 설계

- 후보: `rolling_damped_equal_v1`
- 방법: `damped_3_0.5`와 `damped_3_0.8`의 확률 수준 고정 50:50 평균
- 각 outer year보다 이전 시즌 성공률만 사용하며 validation target은 예측 생성에 사용하지 않음
- 과거 시즌이 3개 미만이면 incumbent의 raw RF/LGB blend로 fallback
- 기존 4개 통계 게이트를 변경 없이 적용

## Walk-forward 결과

| validation | incumbent | candidate | delta | candidate mean | target rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | 0.246945046 | 0.245870442 | -0.001074604 | 0.531739 | 0.532762 |
| 2022 | 0.244169346 | 0.243904601 | -0.000264745 | 0.521709 | 0.528920 |
| 2023 | 0.251137482 | 0.251137967 | +0.000000485 | 0.524528 | 0.499957 |
| 2024 | 0.248460961 | 0.248472963 | +0.000012002 | 0.490793 | 0.486105 |

## 고정 게이트

- recency-weighted delta: **-0.000155463** (통과)
- 최신 연도 delta: **+0.000012002** (실패)
- worst-fold delta: **+0.000012002** (통과)
- 결합 pitcher-season bootstrap P(improve): **1.0000** (통과)

## 결정

사전 고정 게이트를 모두 통과하지 못했다. 이 실험을 근거로 새 ZIP을 생성하지 않으며 damped drift 방향은 종료한다. 기존 `submit.zip`을 유지한다.

## 불확실성

결합 pitcher-season bootstrap delta 95% 구간은 [-0.000407052, -0.000255041]이다. 기존 outer folds는 반복 실험에 사용된 개발 데이터이므로 이 결과를 untouched test 수준의 확증으로 해석하지 않는다.
