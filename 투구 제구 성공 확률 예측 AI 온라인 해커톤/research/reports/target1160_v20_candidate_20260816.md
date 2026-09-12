# v20 target-1160 champion — 2026-08-16

## 결론

`submit_v20.zip`을 v19의 challenger로 생성한 뒤 DACON에 제출했고, Public 개선을 확인해 새 champion으로 승격했다.

- 부모: `submit_v19.zip`
- 부모 Public: `1144.1518063753`
- v20 SHA-256: `4E50A6B7C7D970AF9B8A3656C4D87F2A7D856321C53416E831C4B8E5B01BAE29`
- ZIP 크기: `32,474,467` bytes (`30.970 MiB`)
- 핵심 순방향 최소 개선: `+14.2516628409` unclipped BSS-equivalent
- v19의 실제 local→Public 전달률을 적용한 중심 추정: 약 `1159.75`
- DACON Public: **1151.472428719**
- v19 대비 Public gain: **+7.3206223437**
- 확인 당시 순위: **12위**
- 제출 ID: `1534122`
- DACON 제출: `isSubmitted=true`, `detail=Success`

사전 중심 추정은 실제 결과보다 `8.2775712810` 높았다. 월별 증분이 8개 중 5개만 양수였던 위험과 Public 전달폭 축소가 일관되므로, v20은 개선에 성공했지만 안정형 후보로 해석하지 않는다.

## v19 잔차에서 확인한 구조

v19의 2024 OOF 잔차에 얕은 LightGBM을 다시 적합하는 방법은 세 시간축 최소 `+0.0385`에 그쳐 기각했다. 반면 선수·팀 단위로 강하게 수축한 empirical-Bayes lookup은 세 시간축에서 반복 전이됐다. 이는 v19가 연속형 경기 상황은 이미 많이 설명하지만, 최근 투수 제구 편차와 일부 도메인별 카운트 구조가 남아 있음을 뜻한다.

과거 pitch-type, 일반 TrackMan profile, context residual 후보를 v19 증분 기준으로 재평가했으나 대부분 2024에서 사라졌다. 최종 조합에는 v19와 직교성이 남은 latent failure-mode와 PFD soft-label 신호만 포함했다.

## 고정된 다섯 신호

모든 가중치는 연속 최적화 결과를 단순한 값으로 반올림해 고정했다.

| 신호 | 학습/추론 조건 | 수축 | 가중치 |
|---|---|---:|---:|
| `pitcher_id × batter_hand × pressure` v19 OOF 잔차 | 전체 도메인 | `alpha=50` | `0.07` |
| `pitcher_team_id × batter_hand` v19 OOF 잔차 | 전체 도메인 | `alpha=3200` | `0.16` |
| `count × pitcher_hand × batter_hand` v19 OOF 잔차 | `R_ANCHOR`만 | `alpha=50` | `0.18` |
| `conditional_mode_lgb_h0.5_pow1.5 - v17` | 전체 도메인 | 기존 mode 모델 | `0.045` |
| PFD `soft_l050 - hard_control` | 전체 도메인 | 2024 TrackMan은 학습 때만 사용 | `0.26` |

`pressure`는 `balls_before==3 → threeball`, 그 외 `strikes_before==2 → twostrike`, 나머지는 `normal`이다. R_ANCHOR의 anchor team ID는 기존 v19와 같은 `13`이다.

## 순방향 검증

| 축 | 적합 구간 → 감사 구간 | v19 대비 gain |
|---|---|---:|
| 가장 중요한 연도 전이 | 2023 전체 → 2024 전체 | **+14.2517** |
| 시즌 내 전이 | 2023 3–7월 → 8–10월 | **+178.0079** |
| 최신 시즌 내 전이 | 2024 3–7월 → 8–10월 | **+20.6529** |

2024 전체 도메인별 gain은 모두 양수였다.

| 도메인 | 행 수 | gain |
|---|---:|---:|
| R_CORE | 178,729 | +9.8066 |
| R_ANCHOR | 44,768 | +27.9184 |
| F | 30,010 | +20.6139 |

2024 월별 gain은 `3월 +23.69`, `4월 -15.46`, `5월 +37.23`, `6월 +33.70`, `7월 +37.40`, `8월 +4.09`, `9월 -14.38`, `10월 -55.10`이다. 10월은 1,671행으로 작지만, 월별 불안정성 때문에 이 후보를 확정 champion으로 표시하지 않는다.

## 군집 부트스트랩

2024 전체 후보와 v19의 paired Brier 차이를 5,000회 재표집했다. p05 gain은 원래 fold의 base-rate reference를 고정해 환산했다.

| 재표집 군집 | P(improvement) | gain p05 | gain p95 |
|---|---:|---:|---:|
| 투수 | 98.52% | +1.4253 | +26.7727 |
| 타자 | 99.86% | +4.6309 | +24.0008 |
| 투수×타자 | 99.90% | +4.5277 | +23.7092 |

전체 개선이 특정 선수 몇 명만으로 만들어졌을 가능성은 낮지만, 월별 regime 위험까지 제거됐다는 뜻은 아니다.

## PFD 누수 방지

TrackMan 현재 투구 물리 특성은 2024 학습행에서 privileged teacher의 game-group OOF 확률을 만드는 데만 썼다. 최종 ZIP에는 다음만 들어간다.

1. 메인 테이블의 추론 안전 특성으로 hard label을 학습한 control student
2. `0.5 × hard label + 0.5 × privileged teacher OOF`를 학습한 soft student
3. 추론 시 `0.26 × (soft - control)` 보정

따라서 ZIP 추론은 TrackMan 파일, 현재 투구 유형, 현재 투구 물리값을 읽지 않는다. 이 구조는 generalized distillation의 training-time privileged information 사용법을 따른다.

## 패키지 검증

`src.validate_v20_target1160`의 모든 게이트가 통과했다.

- 부모 대비 추가 파일: v20 spec + PFD control/soft 모델 2개
- 부모 파일 삭제: 없음
- 부모 파일 변경: `script.py`, `model/hybrid.json`만
- 대표 245,789행: `93.274s`, peak RSS `1,434.9 MiB`
- 확률 범위: `[0.3482202, 0.6467972]`
- 홀수/짝수 배치 재결합 최대 차이: `1.11e-16`
- R_CORE/R_ANCHOR/F 모두 부모와 유의한 예측 차이 존재
- 네트워크 호출 없음
- test 집계 없음
- 추론 시 현재 투구 TrackMan 사용 없음

상세 결과는 `reports/v20_validation.md`와 `reports/v20_validation.json`에 있다.

## Public 중심 추정과 운영 판단

v19의 local 2024 gain 대비 실제 Public gain 전달률은 다음과 같다.

```text
(1144.1518063753 - 1093.3213473808) / 46.3959809015 = 1.09558
1144.1518063753 + 1.09558 × 14.2516628409 ≈ 1159.77
```

이는 한 번의 관측 전달률을 사용한 시나리오일 뿐 신뢰구간이 아니었다. 실제 Public은 `1151.472428719`로 v19보다 `+7.3206223437` 개선됐고, 최신 연도 로컬 gain의 전달률은 약 `0.51367`이었다. v20을 champion으로 승격하되, 남은 1회 제출은 동일 신호의 Public 사후 가중치 조정에 사용하지 않는다. 공식 결과의 상세 근거는 `reports/v20_public_result_20260816.md`에 있다.

## 재현 명령

```powershell
python -m src.archive.v20_residual_overlay_screen --project .
python -m src.archive.v20_model_residual_screen --project .
python -m src.archive.v20_recency_eb_screen --project .
python -m src.train_v20_target1160 --project .
python -m src.package_v20_target1160 --project . --parent submissions/history/submit_v19.zip --output submit_v20.zip
python -m src.validate_v20_target1160 --project . --candidate submit_v20.zip --parent submissions/history/submit_v19.zip
$env:PYTHONPATH=(Resolve-Path '.').Path; pytest -q
```

## 참고 자료

- DACON 공식 평가: https://dacon.io/competitions/official/236743/overview/evaluation
- DACON 공식 규칙: https://dacon.io/competitions/official/236743/overview/rules
- DACON test 행 독립성 공지: https://dacon.io/competitions/official/236743/talkboard/417123
- DACON TrackMan/privileged-information FAQ: https://dacon.io/competitions/official/236743/talkboard/417082
- Lopez-Paz et al. (2016), *Unifying distillation and privileged information*: https://arxiv.org/abs/1511.03643
- Cawley & Talbot (2010), model-selection overfitting: https://www.jmlr.org/papers/v11/cawley10a.html
- Owen (2007), crossed-effects pigeonhole bootstrap: https://doi.org/10.1214/07-AOAS122
- Tashman (2000), rolling-origin forecast evaluation: https://doi.org/10.1016/S0169-2070(00)00065-0
- Murphy (1973), Brier-score decomposition: https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2
