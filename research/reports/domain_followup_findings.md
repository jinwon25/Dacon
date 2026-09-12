# 도메인 기반 후속 실험과 제출 후보

## 결론

기존 모델 class를 교체하지 않고, `game_type`별 구조 변화만 train-only change-point 규칙으로 보정한 후보가 사전 통계 게이트를 통과했다. 2021~2023은 incumbent와 동일하고 2024에서만 Brier를 0.000125059 줄였다. 실제 평가에서는 `R` 행을 그대로 유지하고 `F` 행에만 frozen logit offset -0.1079835303을 적용한다.

최종 추천 파일은 `submit_candidate_game_type_regime_v1_clean.zip`이다. 원본 incumbent와 이전 후보 ZIP은 수정하거나 삭제하지 않았다.

## 원데이터 EDA

| season | F rate | R rate | 전체 rate |
| ---: | ---: | ---: | ---: |
| 2019 | 0.689250 | 0.549490 | 0.564670 |
| 2020 | 0.587774 | 0.526925 | 0.532712 |
| 2021 | 0.703840 | 0.512763 | 0.532762 |
| 2022 | 0.708749 | 0.503691 | 0.528920 |
| 2023 | 0.472904 | 0.503118 | 0.499957 |
| 2024 | 0.459280 | 0.489707 | 0.486105 |

- F는 2022→2023에 -0.235846, R은 -0.000573 변했다. 2023 calibration 붕괴의 핵심 구조다.
- 2023과 2024 F residual은 전체 시즌 대비 각각 -0.027054, -0.026825로 지속됐다.
- `asof_pitcher_success_rate`와 strike rate는 동일하지 않고 실패 구성 비율도 상호배타적으로 합 1이 아니다. reverse/middle/ball을 임의 합산하지 않았다.
- 연도 내 decile 성공률 span은 failure-component total 0.1217, entropy 0.0789, pitchmix entropy 0.0417, recent-middle 변동성 0.0251이었다.
- game type 코드의 공식 의미가 설명서에 없으므로 R/F를 익명 strata로만 사용했다. 실제 test의 F 비율은 조회·추정하지 않았다.

## Change-point 규칙

각 forecast year보다 이전 시즌만 사용한다.

1. game type 연도별 성공률에서 같은 시즌 전체 성공률을 빼 residual을 계산한다.
2. 최근 residual과 과거 평균의 부호가 바뀌고 차이가 5%p 이상이며 표본이 5,000 이상일 때만 활성화한다.
3. 최근 두 시즌이 같은 부호로 지속되면 두 시즌 logit residual 평균을 사용한다.
4. 추론에서는 현재 행의 `game_type`만 lookup하며 다른 test 행을 사용하지 않는다.

이 규칙은 2021~2023 validation에서는 어느 그룹도 활성화하지 않았고, 2024 F에만 -0.108321을 적용했다. 전체 train으로 2025를 예측할 때는 2023~2024 F residual 평균 -0.107984가 고정된다. R offset은 0이다.

## Walk-forward 결과

| validation | incumbent | candidate | delta |
| ---: | ---: | ---: | ---: |
| 2021 | 0.246945046 | 0.246945046 | 0 |
| 2022 | 0.244169346 | 0.244169346 | 0 |
| 2023 | 0.251137482 | 0.251137482 | 0 |
| 2024 | 0.248460961 | 0.248335902 | -0.000125059 |

- 고정 1:2:3:4 recency delta: **-0.000050024**
- 최신 2024 delta: **-0.000125059**
- 최악 fold delta: **0**
- 2024 pitcher-season bootstrap 10,000회 delta CI: **[-0.000190832, -0.000067138]**
- 결합 2021~2024 pitcher-season CI: **[-0.000048626, -0.000016710]**
- pitcher 단위 결합 CI: **[-0.000049379, -0.000016798]**

모든 10,000 resample에서 delta가 음수였지만 이는 실제 개선 확률 100%가 아니라 finite bootstrap empirical proportion이다.

## 월별 민감도

2024년 3~8월은 모두 개선했으나 9월은 +0.0000256, 표본이 작은 10월은 +0.0004695로 악화했다. 연간 F rate는 2023→2024에도 하락해 regime 지속성은 지지되지만, 시즌 후반 적응 또는 구성 변화로 효과가 약해질 위험은 남는다.

## 대안 경로

### RF 3-seed

- 2023 delta +0.000050031, 2024 delta -0.000009413.
- regime 결합 시 2024 -0.000128704였지만 2023 worst-fold 한도 0.000050000을 3.1e-8 초과했다.
- 모델 크기·추론 비용이 약 3배여서 2021·2022 확장과 패키지를 중단했다.

### 최소 domain-profile LGB

| validation | profile+incumbent regime delta | best iteration |
| ---: | ---: | ---: |
| 2021 | -0.000026610 | 31 |
| 2022 | -0.000018095 | 48 |
| 2023 | -0.000040529 | 7 |
| 2024 | -0.000113229 | 52 |

네 fold 모두 개선했지만 최적 iteration이 7~52로 불안정하고, 새 production model의 고정 iteration을 nested 검증하지 않았다. 따라서 유망 연구 후보로만 남기고 이번 ZIP에는 포함하지 않았다.

### 중단 유지

CatBoost, hierarchical backoff, squared-error GBDT, RF 하이퍼파라미터 탐색과 TrackMan ID 추측은 재개하지 않았다.

## 패키지

- 파일: `submit_candidate_game_type_regime_v1_clean.zip`
- SHA-256: `A2CFCD38108047FCD50F11D23F4F9E41C782378305A819F0F75FA8AD6A324D86`
- ZIP / 해제 크기: 4.055MB / 5.034MB
- 실제 archive 245,789행: 12.850초
- child peak RSS: 696.9MB
- 공식 5행 smoke, row order, 확률 범위, 인터넷 부재, batch independence: 통과
- `R` probe와 incumbent 최대 차이: 2.22e-16
- split batch 최대 차이: 2.22e-16

## 남은 불확실성

- outer folds는 이미 다수 실험에 사용된 개발 데이터이며 change-point 가설도 EDA 이후 생성됐다.
- game type F의 공식 의미는 확인되지 않았다.
- 실제 2025 test에 F가 없다면 후보는 사실상 incumbent와 같다.
- 2025에서 F regime이 되돌아오면 해당 행은 악화할 수 있다.
- game ID가 없어 경기 단위 bootstrap은 수행하지 못했다.

이 후보는 한 번의 명확한 가설 검증용 제출로만 사용하고 Public 결과로 offset을 미세조정하지 않는다.

## Public 결과와 사후 결정

- submission ID: 36908
- 제출 파일: `submit_gt_v1.zip`
- 제출 시각: 2026-08-06 20:45:27
- Public Score: 704.1257475401
- 서버 실행 시간: 5초
- incumbent 대비 Public 차이: -45.3992015564

로컬 walk-forward와 cluster bootstrap을 통과했지만 실제 평가 점수는 incumbent보다 크게 낮았다. 따라서 `game_type=F`의 2023~2024 residual을 2025에도 그대로 적용하는 regime-carry 가설은 배포 후보에서 제외한다. 이 결과를 이용한 offset 축소, threshold 변경 또는 중간값 제출은 리더보드 과적합이므로 수행하지 않는다. 기존 `submit.zip`을 incumbent이자 현재 단독 1순위로 유지한다.
