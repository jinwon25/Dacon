# Top-1100 structural modeling — final decision (2026-08-09)

## 1. 대회와 현재 상황 요약

1. 각 투구 직전의 경기 상황·선수 이력으로 `control_success=1` 확률을 예측한다.
2. 평가는 Brier Skill Score이므로 적중률보다 확률의 보정과 조건부 분해능이 중요하다.
3. 현재 보존된 champion은 `submit_v2.zip`, Public 763.2665303697이다.
4. 763에서 1100으로 가려면 base rate 약 0.5 기준 Brier를 약 0.00084 줄여야 한다.
5. 하나의 전역 calibration이나 blend weight만으로 닫기 어려운 격차다.
6. 이번 cycle은 2019–2023 train, 2024 locked confirmation으로 screen했다.
7. 2024는 과거에 반복 노출된 개발 구간이므로 virgin holdout으로 부르지 않는다.
8. test 행 간 통계·순서·검색·적응은 사용하지 않는다.
9. 원본 외부 야구 데이터는 사용하지 않고, 문헌은 방법론 근거로만 사용한다.
10. 고정 gate를 통과한 후보만 `submit_v6.zip`으로 만들 수 있다.

## 2. 데이터·champion·git 감사

- train: 1475092행 × 49열, SHA-256 `D2081186B458B49F60B082BE480C273135833E15BA59A76D033AF28BCF8763FF`
- formal test sample: 5행 × 48열 (전체 hidden test 재구성 아님)
- Trackman: 1793078행 × 30열, SHA-256 `F7818F9EE0CCEFE7C2CF69FA99EFE6E5CB882D8B886DD96D2394BCF3B53F33A9`
- `submit_v2.zip`: 8,880,825 bytes, SHA-256 `FE368AF109EF0BB8103D728F45794A6192599B691BF08DF02A22F31BD2D438A7` — immutable 보존·검증 PASS.
- 현재 저장소 HEAD는 BARAM 작업 branch이며 기준 commit `9b0cb8401336916f510669021005e48b4db2c92a`는 다른 aimers branch에 존재하지만 현재 HEAD의 조상은 아니다. 사용자 변경분 때문에 branch 전환/reset은 하지 않았다.

## 3. 핵심 구조 발견

1. `asof_pitcher_n` 전이는 1,473,508건 전수에서 100% 일치했다. 현재 season 시작 전 snapshot을 사용해 season 노출·성공 count를 복원할 근거가 확보됐다. 표시 rate의 round-trip 오차 p99.9는 약 0.0058 count 단위, max 약 0.0077이었다.
2. target-free 손잡이 집계는 `1=Left, 2=Right`가 반대 mapping보다 모든 season aggregate share error가 작았다. 이는 aggregate 의미 확인이지 player identity 정답률은 아니다.
3. main game block ↔ Trackman candidate alignment는 4,810/7,228 main block을 매칭했지만 실제 high rule 6.82%가 placebo 7.26–8.27%보다 좋지 않았고 median distance도 개선되지 않았다. 따라서 Trackman 물리 profile 승격을 중단했다.

## 4. baseline·validation 부채

`v2_frozen_replay`는 과거 package recipe parity용 진단 baseline이다. `v2_nested`는 outer target을 early stopping·offset·weight에 쓰지 않는 builder를 만들었지만 전체 strict OOF가 local runtime을 초과해 완성되지 않았다. 따라서 신규 screen을 champion 후보나 1100 근거로 부르지 않는다. frozen D full-v2의 2024 Brier는 0.248440533이며 legacy diagnostic이다.

## 5. 구조 모델 screen (2024, target 미사용)

### 다년 비교표와 CatBoost 해석

| baseline/model | 2021 Brier | 2022 Brier | 2023 Brier | 2024 Brier | recency-weighted delta | worst-fold delta |
|---|---:|---:|---:|---:|---:|---:|
| v2_package_exact (legacy frozen replay) | 0.246865966 | 0.244151170 | 0.251047904 | 0.248440533 | reference | reference |
| v2_honest | NA | NA | NA | NA | NA | NA |
| CatBoost P0-A screen | NA | NA | NA | 0.248273754 | NA | NA |

The CatBoost point estimate is **0.000166533 lower than the legacy 2024 frozen-replay Brier**. This is a real 2024 screen signal, not a family-wide failure. However, 2021–2023 CatBoost OOF, honest v2 OOF, seed replication, pitcher-cluster bootstrap CI, recency-weighted delta and worst-fold delta were not produced; therefore the multi-year promotion gate is **not passed / not evaluable**, rather than “CatBoost was rejected by the 2024 point estimate”.

CatBoost gate status: 2024 point estimate **PASS as a research signal**; four-fold weighted delta **NOT EVALUABLE**; worst-fold **NOT EVALUABLE**; seed rule **NOT RUN**; bootstrap CI **NOT RUN**; subgroup safety **NOT RUN**; deployment candidate **NOT ELIGIBLE**.

| model | Brier | score-equivalent | 판단 |
|---|---:|---:|---|
| v2 package exact (legacy, target-overlap diagnostic) | 0.246247912 | 1424.7 | 비교용, promotion 금지 |
| P0-A full-state LightGBM (120만 train rows) | 0.249260507 | 218.7 | v2 legacy 대비 개선 아님 |
| categorical-safe CatBoost (18만 screen rows) | 0.248273754 | 613.7 | 2024 signal; multi-year gate 미확인 |
| field-aware FM pilot (25만 rows, 5 epochs) | 0.448299226 | 0.0 | 수렴·보정 실패, 종료 |
| one-hot SGD dynamic-logit screen (30만 rows) | 0.343705606 | 0.0 | 종료 |

The package prediction is trained with all seasons and is not an honest outer baseline; the table is therefore a diagnostic comparison only. No candidate satisfies the champion gate (2024 ΔBrier ≤ −0.00010, four-fold weighted ΔBrier ≤ −0.00010, worst fold ≤ +0.00005, cluster bootstrap ≥ .95).

## 6. gate·패키지 판정

- outer nested OOF: **FAIL/BLOCKED** — full strict run not available.
- P0-B linkage placebo separation: **FAIL** — no Trackman promotion.
- structural model screen: **FAIL** — no repeatable honest improvement.
- subgroup/correction/bootstrap gate: **NOT RUN / NOT ELIGIBLE**.
- deployment parity for existing v2: **PASS** (previous package audit; 245,789 rows within seconds, row-order checks passed).
- candidate package: **NOT CREATED**. `submit_v6.zip` does not exist and no DACON submission was made.

## 7. 최종 결론

**신규 후보가 고정 gate를 통과하지 못해 submit_v2.zip을 유지한다.**

이번 cycle의 유효한 산출물은 데이터 계약, as-of state 복원 감사, target-free alignment audit, 구조 모델 screen과 재현 테스트다. CatBoost/FＭ/LightGBM screen 결과만으로 Public score 개선을 주장하거나 submit_v6을 만드는 것은 통계적으로 정직하지 않다.

## 8. 다음 실험 최대 5개

1. 기준 branch에서 4-fold strict nested v2를 run cache와 고정 iteration으로 완주하고 row-level OOF hash를 생성한다.
2. P0-A state를 2019–2023 full train의 ordered LightGBM/DeepFM에 넣되 2021–2023 inner screen → 2024 one-time lock 순서를 지킨다.
3. main에 공식 game_date/game_id가 제공되는지 운영진 Q&A로 확인한 뒤에만 pitch-level Trackman alignment를 재개한다.
4. FM pilot의 saturation 원인을 고친 단일 low-learning-rate seed만 사전 등록해 재시험하고, 개선 없으면 family를 종료한다.
5. strict OOF가 확보되기 전에는 residual stack·calibration·submit_v6을 만들지 않는다.
