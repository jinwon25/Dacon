# Public 1175 실험 근거

## 한 줄 요약

기존 JY 1172 모델이 높은 확률을 주는 압박 상황 중 일부에서, 서로 다른 계보의 Hyunku형 XGBoost 확률을 30% 혼합해 남은 오차를 줄였다.

## 최종 수식

먼저 JY 1172 챔피언 확률 `p_jy`를 그대로 계산한다.

```text
R_CORE = game_type == R
         and pitcher_team_id != 13
         and batter_team_id != 13

PRESSURE = num_runners_on > 0 or li >= 1.5

ACTIVE = R_CORE and PRESSURE and p_jy >= 0.50
```

최종 확률:

```text
ACTIVE가 아니면: p_final = p_jy
ACTIVE이면:      p_final = 0.70 × p_jy + 0.30 × p_xgb
```

즉 JY의 F·anchor·저압 상황은 보호하고, R 핵심 리그의 주자/고LI 고확률 구간만 수정한다.

## 왜 이 구조를 선택했는가

전체 행 XGB 혼합은 시즌 간 전이가 약했다. 반면 다음 조건을 동시에 만족하는 구간에서는 독립 XGB의 residual 방향이 2022·2023·2024 모두 양수였다.

- R 행
- 팀 13 anchor 제외
- 주자 있음 또는 `LI >= 1.5`
- JY 확률 `>= 0.50`

이 신호는 exact X98 경로뿐 아니라 실제 test에서 만들 수 있는 fallback TrackMan 경로로 재구축했을 때도 유지됐다. 따라서 공개 검증용 피처와 제출 피처의 의미를 맞출 수 있었다.

## Strict OOF 결과

최종 선택 `weight=0.30`:

| 시즌 | 활성 행 | 기준 BSS | 후보 BSS | ΔBSS |
|---:|---:|---:|---:|---:|
| 2022 | 68,486 | 2272.2528 | 2296.0319 | **+23.7791** |
| 2023 late | 11,892 | -60.9708 | -56.7156 | **+4.2552** |
| 2024 | 30,798 | 999.1297 | 1000.6651 | **+1.5353** |

세 시즌 모두 양수이며 단순 평균 ΔBSS는 `+9.8566`이다. 평균은 2022의 큰 개선 영향을 받으므로 Public 예상치로 직접 환산하지 않았다.

전체 sweep은 `evidence/jy_xgb_fallback_active_high50.csv`에 보존했다.

## 2024 안정성

월별 ΔBSS:

| 월 | ΔBSS |
|---:|---:|
| 3 | +6.1970 |
| 4 | +7.1768 |
| 5 | +3.2428 |
| 6 | -2.0523 |
| 7 | +5.0308 |
| 8 | -1.1237 |
| 9 | -4.2874 |
| 10 | -12.8017 |

2024 투수 단위 bootstrap 500회:

- 5%: `-0.5433 BSS`
- 중앙값: `+1.4874 BSS`
- 95%: `+3.6670 BSS`
- `P(ΔBSS > 0) = 0.882`

따라서 로컬 증거만으로는 무위험 후보가 아니었다. 실제 Public 1175 결과가 최종 채택 근거다.

## XGBoost 설정

| 항목 | 값 |
|---|---:|
| rows | 1,475,092 |
| features | 114 |
| n_estimators | 1800 |
| learning_rate | 0.006 |
| max_depth | 10 |
| min_child_weight | 6000 |
| subsample | 0.7 |
| colsample_bytree | 0.5 |
| reg_lambda | 50 |
| reg_alpha | 1 |
| seed | 2028 |

학습 가중치는 최근 시즌을 더 반영하는 지수 감쇠를 사용했다. feature family는 raw/context, season-asof, multiscale success rate, pitcher situation, pitcher-batter matchup, 역할/등판량, prior-season TrackMan profile이다.

## 검증 중 발견하고 수정한 문제

초기 fallback 피처에서 새로운 투수의 `p_ppa` 결측을 현재 추론 배치의 중앙값으로 채우고 있었다. 이는 test의 다른 행에 의존할 수 있어 규칙 위반이다.

수정:

```text
이전: 현재 inference batch의 p_ppa 중앙값
이후: 공식 train의 prior-history p_ppa 중앙값
```

수정 후 결과:

- frozen feature parity 최대 절대 차이: `5.22e-08`
- XGB 단일행 대비 최대 예측 차이: `0.0`
- XGB 셔플 대비 최대 예측 차이: `0.0`
- ZIP CRC: 통과
- 필수 모델·lookup·requirements: 포함

## Public 해석

부모 점수 `1172.1373858439`에서 사용자가 확인한 `1175점대`로 상승했다. 정확한 소수점이 없으므로 개선폭은 약 `+2.9점`으로만 기록한다.

이번 성공의 핵심은 XGB 자체의 절대 성능이 아니라 다음 세 가지다.

1. 기존 JY와 다른 계보의 확률을 사용했다.
2. 전체 행이 아니라 전이가 확인된 active segment만 수정했다.
3. 피처 생성 규약을 OOF와 test에서 동일하게 맞췄다.

앞으로의 실험은 이 Public 1175 릴리스를 새 부모로 삼되, 같은 XGB weight를 재탐색해 Public에 과적합하지 않도록 다른 독립 신호만 추가한다.
