# 상위 10% 개선 사이클 결과 — 2026-08-14

## 현재 기준선과 목표

- 보존 champion: `submit_v2.zip`, Public `763.2665303697`.
- 2026-08-14 공식 리더보드: 942팀, 1위 `1176.5490443799`, 94위 `1079.9157207276`.
- 현재 상위 10% 경계까지 `316.6491903579`점 차이다.
- `r(1-r)≈0.25` 근사에서 필요한 추가 Brier 개선은 약 `-0.000792`다.

## 핵심 원인 발견

`src/top1100_features.py`의 시즌 상태 기준점에 이중 시프트가 있었다.

1. `_prior_table`에서 엔티티별 시즌 endpoint를 `shift(1)`했다.
2. `_merge_prior`가 다시 `allow_exact_matches=False`로 직전 시즌만 선택했다.
3. 결과적으로 2024 행은 2023 말이 아니라 2022 말 누적 상태를 빼고 있었다.

시프트를 제거하고 투수·타자의 직전 시즌 endpoint, 현재 시즌 노출·성공 count, 직전 시즌 posterior와 현재 시즌 posterior를 다시 만들었다. 회귀 테스트에는 2022 행이 정확히 2021 endpoint를 사용하는 예제를 추가했다.

## 검증 결과

모든 표의 delta는 candidate Brier - V2-NESTED-R1 Brier다. 음수가 개선이다. 각 outer year의 학습에는 이전 outer OOF만 사용했다.

### 1. 수정 상태 OOF 잔차 LightGBM

| outer | 선택 근거 | 레시피 | eta | delta |
|---:|---|---|---:|---:|
| 2022 | 사전 고정 | corrected state, R-only fit | 0.25 | -0.000130425 |
| 2023 | 2022 | corrected state, all fit | 0.50 | -0.000309168 |
| 2024 | 2023 | corrected state, all fit | 1.00 | **-0.000442071** |

- 2:3:4 recency-weighted delta: `-0.000328515`.
- season:pitcher 5,000회 paired bootstrap: observed `-0.000295046`, 95% CI `[-0.000344619, -0.000246441]`, 전 resample 개선.
- O24 pitcher bootstrap: `-0.000442071`, 95% CI `[-0.000556413, -0.000323910]`.
- O24에서 8개 월 중 7개, 12개 count 중 11개, 5개 투수 이력 bucket 전부 개선했다. 예외는 표본이 작은 10월과 `0-2` count다.

### 2. 수정 상태 CatBoost 재학습

기존 CB-R1의 2023 delta는 `+0.001308301`로 실패했다. 동일한 350회·depth 7 레시피에 수정 상태 피처를 적용하자 전체 raw delta가 다음처럼 바뀌었다.

| outer | raw 전체 delta | raw R delta | 직전 연도 선택 R-blend delta |
|---:|---:|---:|---:|
| 2021 | -0.000891011 | -0.000110283 | 0.000000000 |
| 2022 | -0.000660023 | -0.000429147 | -0.000313755 |
| 2023 | -0.000272390 | -0.000382470 | -0.000344759 |
| 2024 | -0.000430200 | -0.000331244 | -0.000361119 |

- 순방향 1:2:3:4 weighted delta: `-0.000310626`.
- season:pitcher bootstrap 95% CI: `[-0.000300532, -0.000210608]`.
- 2023 실패가 모델 family 자체보다 시즌 상태 구현 결함에서 비롯됐다는 강한 증거다.

### 3. 조합

- 수정 상태 잔차 + 기존 CB/context 축: O22에서 추가 비중을 고르고 O23에 적용, O23에서 다시 고르고 O24에 적용했다.
- delta: O22 `-0.000130425`, O23 `-0.000311`, O24 **`-0.000491`**.
- 세 축 효과 상관은 R행 기준 residual↔corrected-CB `0.52~0.61`, residual↔legacy-context `0.46~0.47`, corrected-CB↔legacy-context `0.64~0.73`이었다.
- 수정 CatBoost를 강하게 더한 3축 규칙은 O23에서 고른 비중이 O24로 이전되지 않아 승격하지 않았다.
- 선수별 경험베이즈 잔차는 O24 `-0.000456`으로 소폭 이득이 있었지만 legacy-context 조합보다 약했다.
- 최근 1년/2년만 학습한 잔차와 고카디널리티 ID 잔차 LightGBM은 전체 이력·무ID 상태 잔차보다 약해 종료했다.

## 해석

- 이번 사이클의 새롭고 가장 재현성 높은 단일 축은 수정 상태 잔차다.
- 가장 강한 전향 조합의 O24 `-0.000491`은 Brier Skill Score 약 +196점 규모다. Public에 선형 이전된다고 가정하면 약 959점 수준으로, 현재 10% 경계에는 아직 부족하다.
- 실제 Public 분포와 local outer 분포가 다르므로 이 환산은 예측치일 뿐이다.
- 2021~2024는 이전 연구에서 반복 노출된 개발 구간이다. bootstrap은 표본 불확실성만 다루며 family 선택 편향을 제거하지 않는다.
- `legacy-context` recipe 자체는 이전 OOF를 본 뒤 개발됐으므로, 조합 후보는 1회 통제 Public probe 가치가 있지만 확정 champion으로 부르지 않는다.

## 결정

1. `submit_v2.zip`은 immutable champion으로 유지한다.
2. 다음 배포 후보는 수정 상태 잔차를 중심으로 만들고, legacy-context 추가 여부를 분리 가능한 두 ZIP으로 준비한다.
3. 첫 Public probe는 새 단일 축만 넣어 이번 버그 수정의 실제 기여를 식별한다.
4. 단일 축이 Public에서 유의미하게 오를 때만 context 조합을 두 번째로 제출한다.
5. 아직 `submit_v6.zip`은 생성·제출하지 않았다. 최종 모델 재학습, 2025 row-local endpoint lookup 동결, 245,789행 runtime/parity 검증이 남아 있다.

## 주요 산출물

- 수정 상태 잔차: `artifacts/top10_20260814/corrected_state_residual_20260814_01/`
- 확대 eta 및 subgroup: `artifacts/top10_20260814/corrected_state_rescore_20260814_03/`
- 수정 CatBoost OOF: `artifacts/top10_20260814/corrected_cb_oof_20260814_01/`
- legacy-context 보완: `artifacts/top10_20260814/corrected_state_complementarity_20260814_01/`
- 3축 상관·조합: `artifacts/top10_20260814/corrected_family_ensemble_20260814_01/`
- 경험베이즈 잔차: `artifacts/top10_20260814/hierarchical_residual_prior_20260814_01/`
