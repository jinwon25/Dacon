# LDAPS 공간 방법론 전환 실험 — 2026-07-26

## 결론

제출 `1501731`의 실패 뒤 같은 alpha·coverage 미세 조정을 중단하고, 1차
문헌에 기반한 두 개의 새 공간 방법론을 검증했다.

1. 16개 LDAPS 격자를 발전량 시나리오로 변환한 뒤 공식 score 기대값을
   직접 최대화하는 neighborhood decision
2. 예보 풍향의 상류 0/1.5/3.0/4.5 km 지점에 Gaussian kernel을 놓는
   physics-guided upwind advection power curve

첫 방법은 Q2를 통과했지만 H2에서 세 지표가 모두 하락했다. 두 번째
방법은 Q2에서 FICR과 총점은 올렸으나 1-NMAE가 하락해 사전 게이트에서
종료됐다. 두 실험 모두 `rejected`이며 제출 CSV는 만들지 않았다. 현재
제출 대상은 계속
`artifacts_final/candidates/kma_group2_overlay_alpha2375_20260725.csv`다.

## 참고한 1차 자료

- [Molinder et al. (2018)](https://wes.copernicus.org/articles/3/667/2018/)은
  주변 NWP 격자를 동등하게 가능한 예보로 취급해 작은 기상계의 공간
  위치 오차와 대표성 불확실성을 표현한다.
- [Ye et al., AIRU-WRF (2023)](https://arxiv.org/abs/2303.02246)은
  advection·diffusion을 물리적 kernel로 표현하는 physics-guided
  spatiotemporal wind forecasting을 제안한다.
- [Mandi et al. (2022)](https://proceedings.mlr.press/v162/mandi22a.html)은
  예측 오차만 최소화하지 않고 downstream decision quality를 학습
  목표에 포함하는 decision-focused 관점을 제시한다.
- [Kanninen et al. (2021)](https://wes.copernicus.org/articles/6/1205/2021/)은
  인접 격자 결합의 일반적 이득은 작았고, smoothing과 random forest,
  기상 조건별 방법 선택이 더 유망했다고 보고한다. 이 결과는 본 실험의
  인접 격자 실패 방향과도 일치하지만, 해당 연구는 해상 단일 지점이므로
  산악 풍력단지에 그대로 일반화하지 않는다.

## 실험 1: bias-corrected neighborhood decision

구현:
`experiments/ldaps_neighborhood_decision.py`

### 방법

- LDAPS 10 m와 50 m 바람을 power law로 117 m에 외삽한다.
- Q1의 그룹 3 실측으로 모든 격자가 공유하는 monotone isotonic
  wind-to-power curve를 학습한다.
- 각 시각의 16개 격자를 16개 발전량 시나리오로 변환한다.
- 원시 시나리오 중심의 큰 편향을 제거하기 위해 시나리오 중앙값을
  incumbent에 맞추되 격자 간 편차는 그대로 보존한다.
- 각 시나리오의 발전량 가중 FICR과 NMAE를 사용해 기대 공식 score가
  최대인 action을 고른다.
- Q2에서는 방향, 상위 disagreement coverage, 출력 구간, alpha만
  사전 정의한 작은 격자로 선택한다.
- 변경 비율은 10%, 행별 추가 이동은 1% 용량으로 제한한다.

### Q2 선택

- 방향: 양방향
- disagreement coverage: 10%
- incumbent 출력 구간: 20%–80%
- alpha: 0.10
- group-3 score: `+0.0011092408`
- group-3 1-NMAE: `+0.0000679304`
- group-3 FICR: `+0.0021505513`

### H2 잠금 결과

H1로 power curve를 다시 학습한 뒤 H2를 한 번 평가했다.

| 지표 | H2 변화 |
|---|---:|
| group-3 score | `-0.0006798238` |
| group-3 1-NMAE | `-0.0001604521` |
| group-3 FICR | `-0.0011991954` |
| 변경 행 | `358 / 4,409` (`8.12%`) |
| 최대 추가 이동 | `210 kWh` |

7–9월 score는 `-0.0001271376`, 10–12월은 `-0.0011059740`이었다.
월별로는 7·9·10월만 총점이 비음수였다. 184개 예보 발행 주기를
재표집한 2,000회 bootstrap 결과는 다음과 같다.

- score q05: `-0.0014635718`
- 1-NMAE q05: `-0.0003006545`
- FICR q05: `-0.0027076862`
- 세 구성지표 동시 양수 비율: `0.85%`

Q2의 이득은 후반기에 전혀 재현되지 않아 종료한다.

## 실험 2: upwind advection kernel

구현:
`experiments/ldaps_upwind_advection_power_curve.py`

### 방법

- 그룹 3 터빈 좌표의 중심을 원점으로 잡는다.
- 각 시각의 117 m LDAPS 풍향으로 상류 방향을 계산한다.
- 상류 0/1.5/3.0/4.5 km 지점에 폭 1.5 km Gaussian kernel을 두고
  16개 격자의 hub-height wind vector를 가중 평균한다.
- Q1에서 각 구조의 isotonic power curve를 학습하고 Q2에서 구조와
  희소 적용 정책을 함께 선택한다.
- neighborhood 실험과 같은 10% 변경·1% 이동 상한을 적용한다.

### Q2 결과

세 지표를 모두 개선한 정책은 0개였다. 총점 기준 최상 구조는 상류
3.0 km, 양방향, disagreement 상위 5%, alpha 0.05였다.

| 지표 | Q2 변화 |
|---|---:|
| group-3 score | `+0.0018077993` |
| group-3 1-NMAE | `-0.0004221636` |
| group-3 FICR | `+0.0040377621` |
| 변경 행 | `102 / 2,185` (`4.67%`) |

FICR 경계 적중은 늘었지만 절대오차가 악화됐다. 사전 계약에 따라 H2는
열지 않고 종료했다.

## 병목 해석

두 독립 공간 구성에서 같은 패턴이 반복됐다.

- 큰 disagreement 행을 210 kWh 이동하면 6%·8% 경계 안으로 들어오는
  일부 행 때문에 FICR은 증가한다.
- 그러나 이동 방향이 실제 발전량 오차 방향을 안정적으로 맞히지 못해
  1-NMAE가 하락하거나, Q2에서 맞더라도 H2로 전이되지 않는다.
- 따라서 현재 병목은 threshold action의 세기나 coverage가 아니라
  `잔차 부호를 기간 밖에서 맞히는 점예측 정보`의 부족이다.

이 증거로 다음 항목을 종료한다.

- LDAPS 인접 격자를 확률 시나리오로 직접 사용하는 방식
- 고정 거리 상류 kernel의 직접 power-curve 보정
- FICR만 좋아지고 1-NMAE가 나빠지는 후보의 alpha 축소 재탐색

## 다음 우선순위

다음 실험은 공간 격자 변형을 더 늘리지 않는다. 먼저 현재 incumbent의
정확한 2023+2024 multi-period OOF를 구축해 한 해의 계절 이동에 맞춘
정책이 다음 해에도 유지되는지 확인한다. 그 위에서만 다음 구조를
검증한다.

1. 예측 발전량이 아니라 hub-height wind를 issue-day 안에서 완만하게
   smoothing해 front·ramp의 double penalty를 줄인다.
2. 저층 안정도, 약한 종관류, low-level jet 같은 사전 정의한 기상 조건에
   따라 원본과 smoothing 중 하나를 선택한다.
3. 주 목표는 NMAE 개선으로 두고, FICR은 동시 비악화 게이트로 사용한다.
4. 2023→2024와 2024 상·하반기에서 모두 통과한 경우에만 2025 CSV를
   만든다.

이 순서는 문헌상 유망도와 이번 실패 원인을 모두 반영한다. 2024 한 해
내 재선택만으로는 이미 여러 방법이 과적합됐으므로, 2023 exact lineage가
없으면 다음 후보를 승격하지 않는다.

후속 감사에서 2022 그룹 3 라벨이 0건이어서 causal 2023 exact lineage가
구성 불가능함을 확인했다. 누수 없이 가능한 2024 월별 견고성 계약으로
temporal smoothing을 검증한 결과도 Q2에서 기각됐다. 자세한 내용은
`docs/reports/temporal_smoothing_cleanup_2026-07-26.md`에 기록했다.

## 산출물

- 기계 판정:
  `artifacts_final/diagnostics/ldaps_neighborhood_decision_20260726.json`
- 기계 판정:
  `artifacts_final/diagnostics/ldaps_upwind_advection_power_curve_20260726.json`
- 구현:
  `experiments/ldaps_neighborhood_decision.py`
- 구현:
  `experiments/ldaps_upwind_advection_power_curve.py`
- 테스트:
  `tests/test_ldaps_neighborhood_decision.py`
- 테스트:
  `tests/test_ldaps_upwind_advection_power_curve.py`
