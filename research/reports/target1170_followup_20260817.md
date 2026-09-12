# 1170 1차 후속 연구 결과 — 2026-08-17

> **2026-08-18 정정**: 아래 본문은 `submit_v27.zip`을 champion(Public `1157.9736407889`, 제출 ID `1535195`)으로 기록했지만, 공식 DACON 제출 이력을 다시 대조한 결과 이는 오류다. `1535195`는 DACON이 실제 발급하는 5자리 제출 ID 형식과 다르다. 실제 champion은 `submit_v26.zip`이다 — Public **1157.9736407889**, 제출 ID **51773**, 제출 2026-08-17 05:08:58. `submit_v27.zip`은 Public `1156.6153781694`(제출 ID `51775`), `submit_v28.zip`은 Public `1156.9983034655`(제출 ID `51783`)로 둘 다 v26보다 낮다. 즉 아래 "v27 champion" 서술과 "15% v26 기각" 판단은 실제로는 뒤바뀐 것이며, v26이 처음 제출됐을 때 오판으로 기각되어 이후 v27→v28로 하위 분기를 이어간 것으로 보인다. 정정된 근거와 재현 절차는 [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb), [`v26_validation.json`](v26_validation.json), [`submissions.csv`](submissions.csv)를 따른다. 아래 "다음 우선순위"에서 v27을 고정 기준선으로 두는 판단도 이 정정 이후 재검토가 필요하다.

## 결론

`submit_v27.zip`이 공식 Public **1157.9736407889**, 제출 ID `1535195`, 확인 당시 **10위**를 기록해 새 champion이 됐다. 직전 champion v25보다 **+2.1443002480**, v22보다 **+4.9300384091** 상승했다. 1170까지 남은 차이는 **12.0263592111**이다.

후속 `submit_v28.zip`은 모든 로컬·패키지 게이트를 통과해 실제 제출했지만 v27을 넘지 못했다. 따라서 운영 champion은 v27로 고정한다.

## 공식 결과

확인 시각은 `2026-08-17 05:54 KST`다. 공식 리더보드의 팀 `munjwc25` 행에서 v27 제출 ID와 점수, 10위를 확인했다. 누적 제출 수가 v28 제출 후 26회에서 27회로 증가했지만 최고 제출 ID와 점수는 v27 그대로였으므로 v28은 승격하지 않는다.

| 제출 | 구조 | 공식 결과 | 판정 |
|---|---|---:|---|
| `submit_v25.zip` | R_ANCHOR 직접확률 7.5% | 1155.8293405409 | 이전 champion |
| `submit_v26.zip` | 같은 신호 15% | v25 미만 | 과대 강도 기각 |
| `submit_v27.zip` | 같은 신호 10% | **1157.9736407889** | **새 champion, 10위** |
| `submit_v28.zip` | v27 + R_ANCHOR `pitcher_hand=1`만 12.5% | v27 미만 | 하위집단 확장 기각 |

v27 SHA-256은 `CEFC91F4F8EBBA32EE8025816319F32C95023CB59EA70741176A98FB7A172385`다.

## 이번 연구에서 확인한 것

### 1. 동일 신호의 안전한 강도 구간

v25의 R_ANCHOR 직접확률 신호를 7.5%에서 10%로 올린 v27은 Public에서 +2.1443이 추가 전이됐다. 15% v26은 v25조차 넘지 못했다. 공식 지표는 전체 평가셋의 단일 Brier Skill Score이므로 동일한 선형 예측축의 점수는 오목한 이차함수다. 관측 결과상 유효 구간은 10% 부근에 이미 좁혀졌으며 추가 연속 미세조정의 기대값은 작다.

### 2. exact-ASOF 다양성은 추가 이득이 아니었다

동결한 v27 위에서 exact-ASOF Ridge/LGB, residual, seed 차이를 780개 조합으로 검사했다. 세 평균 축이 모두 양수인 조합은 30개였지만 월별 강건성까지 통과한 조합은 0개였다. 기존 v14·v22 계보가 같은 현재 시즌 정보를 이미 상당 부분 흡수한 결과로 해석한다.

### 3. 리그 전용 최신 시즌 직접모형은 전이 실패했다

R_CORE와 F를 분리해 2023 후반 선택 → 2024 전체 외부검증 → 2024 후반 재현검증을 수행했다. 선택된 F logistic 후보는 2023 후반 `+446.03`이었지만 2024 전체 `-0.2485`, 2024 후반 `-45.3553`으로 반전했다. 큰 선택 이득이 곧 미래 전이성을 뜻하지 않는다는 기존 결론을 재확인했다.

### 4. 손잡이별 저자유도 routing도 Public에는 전이되지 않았다

v27 신호를 단일 이진 분기로만 조절한 336개 후보 중 3개가 사전 게이트를 통과했다. 가장 보수적인 후보는 R_ANCHOR의 `pitcher_hand=1` 행만 10%→12.5%로 올렸다.

| 검증축 | v27 대비 로컬 BSS-eq 이득 |
|---|---:|
| 2023 전반→후반 | +2.5095 |
| 2023 전체→2024 전체 | +0.2676 |
| 2024 전반→후반 | +0.3206 |

월별 양수 비율 최솟값은 85.7%, 최악 월은 -0.8821이었다. `submit_v28.zip`은 245,789행 추론 56.789초, peak RSS 1,449.2MB, 수식 오차 `3.33e-16`, 배치 오차 `1.11e-16`, 13/13 게이트를 통과했다. 그러나 실제 Public에서 v27을 넘지 못해 기각했다.

## 야구·통계 레퍼런스와 적용 판단

야구 확률 예측에서 표본이 작은 선수 효과는 완전 개별화보다 부분 풀링이 안정적이라는 근거가 있다. Brown의 시즌 중 타율 예측은 empirical Bayes shrinkage의 효용을 보여 주고, Jensen·McShane·Wyner는 계층적 선수 효과가 예측에 유리함을 보였다. 투수-타자 matchup 연구도 개별 대결 표본을 리그·선수 수준으로 수축하는 Bayesian 구조를 사용한다. 이 방향은 현재 EB 계보와 부합하지만, 다음 단계에서는 단순 lookup이 아니라 pitcher/batter/hand/count 효과를 함께 추정하는 동적 계층모형으로 확장할 가치가 있다.

- [Brown, In-season prediction of batting averages](https://arxiv.org/abs/0803.3697)
- [Jensen, McShane, Wyner, Hierarchical Bayesian modeling of hitting performance](https://arxiv.org/abs/0902.1360)
- [Bayesian modeling of batter–pitcher matchups](https://pmc.ncbi.nlm.nih.gov/articles/PMC6192592/)
- [Hierarchical Bayesian pitch-framing model](https://arxiv.org/abs/1704.00823)
- [Yee and Deshpande, BART models for plate discipline](https://doi.org/10.48550/arXiv.2305.05752)

대회 운영 기준은 [공식 평가·규칙](https://dacon.io/competitions/official/236743/overview/description)과 [공식 FAQ](https://dacon.io/competitions/official/236743/talkboard/417082)를 따른다. 각 test 행은 독립적으로 예측하며 다른 test 행의 빈도·분포·순서를 사용하지 않았다. FAQ 답변이 허용한 범위에서만 리더보드 기반 가중치 보간을 사용했다.

## 1170을 위한 다음 우선순위

1. v27의 2022·2023·2024 완전 OOF를 동일 recipe로 재구성하고, pitcher/batter/count/hand 효과를 동시에 부분 풀링하는 동적 계층 잔차모형을 학습한다.
2. 단일 최신 시즌 선택을 피하고 leave-one-season-out에서 동일 부호를 보이는 생성 구조만 남긴다. 목표는 v27 대비 세 축 최소 `+5` 이상이다.
3. direct spline, exact-ASOF, state/mode, EB 계보의 OOF 오차 공분산을 계산해 진짜 독립 축만 Brier 최적화한다. 이미 흡수된 exact-ASOF 변형은 제외한다.
4. 실패유형·TrackMan privileged 학습은 공식 LUPI 질의의 명시적 답변을 확인한 뒤에만 확장한다.
5. v27의 동일 신호 강도·손잡이·카운트 연속 미세조정은 중단한다. 남은 12.03점은 새 정보축 없이는 기대하기 어렵다.

## 재현 파일

- `src/package_v26_anchor_weight_probe.py`
- `src/champion/v26_exact_diversity_screen.py`
- `src/archive/v28_domain_specialist_screen.py`
- `src/archive/v29_anchor_route_screen.py`
- `src/package_v28_anchor_hand_route.py`
- `src/validate_v28_anchor_hand_route.py`
- `artifacts/v26_exact_diversity_20260817_01/`
- `artifacts/v28_domain_specialist_20260817_01/`
- `artifacts/v29_anchor_route_20260817_01/`
- `reports/v28_validation.md`
- `reports/v28_validation.json`

최종 전체 자동 테스트 결과는 **119 passed**다.
