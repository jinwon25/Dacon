# 초기 분석 및 실험 결과

모든 아래 성능은 동일한 primary validation인 2019–2023 학습 / 2024 검증에서 실제 실행한 값이다. 서로 다른 split의 점수를 직접 비교하지 않았다.

## 검증 선택

- 비공개 평가는 2025년이고 학습의 최신 시즌은 2024년이므로 1년 forward holdout을 선택했다.
- 메인 데이터에 경기 ID와 정확한 날짜가 없어 game-group split은 구성할 수 없다. season 경계가 같은 경기 혼입을 차단한다.
- Platt/isotonic/base-rate shrinkage와 첫 3-way blend는 2019–2022 학습 모델의 2023 OOF 예측에서 학습하고 2024에 고정 적용했다. 후속 season-trend blend의 작은 가중치 grid는 2024 OOF에서 선택했으므로 해당 점수에 선택 편향 위험이 있다.
- 모든 범주 사전과 base rate는 fold train에서만 학습했다. test 행 간 집계는 없다.

## Primary 모델 비교

| model | Brier Score | local Brier Skill Score |
| --- | ---: | ---: |
| constant_train_rate | 0.251875047 | 0.000 |
| empirical_pitcher_prior | 0.250274458 | 0.000 |
| hierarchical_prior_a50 | 0.250168787 | 0.000 |
| hierarchical_prior_a200 | 0.250193908 | 0.000 |
| hierarchical_prior_a1000 | 0.250376078 | 0.000 |
| official_random_forest | 0.248768795 | 415.574 |
| lgb_official_l31 | 0.249104714 | 281.102 |
| lgb_engineered_l31 | 0.249005893 | 320.661 |
| lgb_engineered_l63 | 0.249168776 | 255.457 |
| lgb_engineered_regularized | 0.249050531 | 302.792 |
| lgb_trackman_context | 0.249033530 | 309.598 |
| lgb_no_asof_ablation | 0.249336741 | 188.220 |

## Calibration 및 blend 비교

| model | Brier Score | local Brier Skill Score |
| --- | ---: | ---: |
| best_lgb_raw | 0.249005893 | 320.661 |
| best_lgb_platt | 0.249020864 | 314.668 |
| best_lgb_isotonic | 0.252250205 | 0.000 |
| best_lgb_base_rate_shrink | 0.249232480 | 229.956 |
| fixed_2023_oof_three_way_blend | 0.249952262 | 0.000 |

## 선택

- 최선 LightGBM variant: `lgb_engineered_l31` (2024 early-stop iteration 54).
- 최종 추론 구조: **3-season train-only logit trend + LightGBM/공식 RF = 0.35/0.65 blend**.
- Trackman context 승격 여부: **False**. 선수 ID 직접 조인은 0%라 금지했으며, context도 이전 시즌 league 집계만 사용했다.

## 한계와 누수 위험

- 2024 단일 시즌 holdout이라 시즌별 모델 순위 안정성을 완전히 확인하지 못했다.
- `asof_*`는 공식적으로 허용된 투구 직전 피처이고 갱신식도 확인했지만 target 관련 과거 집계라 분포 변화에 민감하다. no-asof ablation을 비교 기준으로 유지한다.
- 실제 2025 test의 선수 cold-start 비율은 5행 형식 샘플로 추정할 수 없다.
- Trackman 선수/팀 매핑은 제공되지 않아 선수 단위 물리 피처를 만들지 않았다.

## Train-only season trend calibration

Offset은 fold-train의 연도별 target 비율만으로 적합하며 validation/test batch 평균은 사용하지 않는다.

| candidate | Brier Score | local Brier Skill Score | forecast rate |
| --- | ---: | ---: | ---: |
| lgb_season_logit_w2 | 0.248927372 | 352.094 | 0.470994 |
| rf_season_logit_w2 | 0.248729401 | 431.343 | 0.470994 |
| lgb_season_logit_w3 | 0.248770263 | 414.986 | 0.487724 |
| rf_season_logit_w3 | 0.248548983 | 503.566 | 0.487724 |
| lgb_season_logit_w4 | 0.248949803 | 343.114 | 0.498054 |
| rf_season_logit_w4 | 0.248715095 | 437.070 | 0.498054 |
| lgb_season_logit_w5 | 0.248814606 | 397.235 | 0.491715 |
| rf_season_logit_w5 | 0.248588048 | 487.928 | 0.491715 |
| oof_tuned_trend_blend | 0.248460961 | 538.802 | n/a |

최종 선택: **oof_tuned_trend_blend**, 가중치 `[0.35, 0.65, 0.0]`.
작은 blend grid를 동일한 2024 OOF에서 선택했으므로 `0.248460961`은 선택 편향 가능성이 있으며 이 위험을 유지해서 보고한다.
