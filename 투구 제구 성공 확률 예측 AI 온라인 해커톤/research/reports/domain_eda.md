# 한국 야구 제구 도메인 EDA

## 핵심 발견

- `game_type=R`은 시즌당 약 21.1–22.3만 행, `F`는 약 2.3–3.0만 행이다. 2026-08-12 공식 FAQ 답변으로 `R=Regular(1군 정규시즌)`, `F=Futures(퓨처스리그/2군)`가 확정됐다. 모델의 기존 strata 분리는 이 공식 의미와도 일치한다.
- F 성공률은 2022 0.708749에서 2023 0.472904로 -0.235846 변했지만 R은 같은 기간 0.503691에서 0.503118로 -0.000573만 변했다. 2023 전체 calibration 붕괴의 주된 구조적 원인이다.
- 2024 incumbent는 F에서 raw blend보다 Brier를 크게 줄였고 R에서도 소폭 개선했다. 따라서 전역 drift를 제거하는 대신, 검출된 game-type regime residual만 추가하는 방향이 안전하다.
- `asof_pitcher_success_rate`와 `asof_pitcher_strike_rate`는 동일하지 않으며 최대 절대차는 1.000000이다. reverse/middle/ball/strike 비율도 단순히 합 1인 상호배타 범주가 아니므로 임의로 실패확률로 합치지 않는다.
- 카운트, LI, 주자, 이닝은 투구 의도와 위험 회피를 바꾸지만 현재 투구의 실제 구종·요구 코스가 없으므로 row-local 상호작용만 사용한다.

## 사전 정의할 실험

1. 전역 incumbent에 통계적으로 검출된 game-type 최근 regime residual만 더하는 change-point offset.
2. 현재 engineered LGB에 failure-profile entropy, 최근 middle-rate delta, pitch-mix entropy와 game-type regime category만 추가한 최소 피처 모델.
3. 공식 RF 설정을 고정하고 seed 42/202/777 확률을 단순 평균한다.

CatBoost, large hierarchical backoff, squared-error GBDT와 RF 하이퍼파라미터 탐색은 재개하지 않는다. row_id/CSV 순서, test 집계, TrackMan 선수 매핑은 사용하지 않는다.

## 산출 표

- `domain_game_type_regime.csv`: 시즌×game type 성공률과 시즌 residual.
- `domain_game_type_errors.csv`: game type별 incumbent/raw Brier와 평균 예측.
- `domain_segment_rates.csv`: 시즌 내 상황별 성공률.
- `domain_feature_screen.csv`: 연도별 decile target-rate span 진단. 이는 EDA이며 test 변환 통계로 사용하지 않는다.
