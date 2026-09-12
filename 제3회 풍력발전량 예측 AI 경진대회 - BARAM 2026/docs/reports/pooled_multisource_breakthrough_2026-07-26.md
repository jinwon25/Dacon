# Pooled multi-source 돌파 실험 — 2026-07-26

## 결론

실험 설계 당시 공개 확정 최고점은 제출 `1501487`, `0.6440998116`이었다. 이번 작업은
공개 점수를 모델·가중치 선택에 사용하지 않고 다음 두 제출 후보를 만들었다.

1. 엄격 후보
   `artifacts_final/candidates/kma_jma_pooled_all3_g1_umkr_hybrid_g3_strict_20260726.csv`
   - SHA-256:
     `8de3ae05e923905c5299f72c31700e4471b82118edfa5a86b13da7042f618d61`
   - 2024 OOF macro 증분: `+0.0048175728`
   - 단순 공개 전이 추정: `0.6489173844`
2. 그룹 2 near-stable 도전 후보
   `artifacts_final/candidates/kma_jma_pooled_all3_g1g2_umkr_hybrid_g3_nearstable_20260726.csv`
   - SHA-256:
     `3e26d2dcac8374580a2c78f9e7035a3edd85aa0d2b71a3a0a9c5797c88640f35`
   - 2024 OOF macro 증분: `+0.0054508615`
   - 단순 공개 전이 추정: `0.6495506731`
3. 그룹 1 fixed residual-stack 도전 후보
   `artifacts_final/candidates/kma_jma_pooled_all3_g1_residual125_g2near_umkrg3_20260726.csv`
   - SHA-256:
     `e9caadcb3a6f0cf22dc266ba95660c16e0c8c73ee3fca39c7b63b5035a09f442`
   - 2024 OOF macro 증분: `+0.0058673528`
   - 단순 공개 전이 추정: `0.6499671644`
   - 등급: `controlled_exploratory`

두 수치는 제출 점수 보장이 아니다. 현재 2024 검증 데이터는 여러 실험에서 반복
확인된 historical benchmark이므로, 최종 판정에는 새 공개 결과가 필요하다.

## 새 학습 구조

KMA UMRG와 JMA MSM 3×3 stencil의 causal forecast feature를 사용했다. 그룹별
모델을 따로 적합하는 대신 그룹 1·2·3 발전량을 각 설비용량으로 정규화하고 하나의
LightGBM quantile 모델에 쌓았다. 그룹 one-hot 열은 서로 다른 power curve를
표현하고, 공통 날씨-발전 관계는 세 그룹의 표본을 함께 사용한다.

- 공통 quantile 선택: 2023년 1–9월 학습 → 2023년 Q4 평가
- 선택 quantile: `0.80`
- 검증 모델: 2023년 전체 학습 → 2024년 예측
- blend weight 선택: 2024년 Q1만 사용
- 확인: Q2, H2, 3개 seed, 월별 방향, 예보발행 block bootstrap
- production: 2024년 전체 재학습 → 2025년 예측
- 행별 최대 이동: 설비용량의 5%

구현은 `experiments/kma_pooled_group_quantile_blend.py`, 테스트는
`tests/test_kma_pooled_group_quantile_blend.py`에 있다.

## 그룹별 결과

### 그룹 1 — strict

세 그룹 pooled 모델의 그룹 1 weight는 `0.20`이다.

| 구간 | score | 1-NMAE | FICR |
|---|---:|---:|---:|
| Q1 | +0.0133228060 | +0.0017298258 | +0.0249157861 |
| Q2 | +0.0066878188 | +0.0002498970 | +0.0131257406 |
| H2 | +0.0118185097 | +0.0024811825 | +0.0211558369 |
| 전체 | **+0.0111147663** | +0.0017214601 | +0.0205080726 |

- 양수 월: 10/12
- issue-block bootstrap q05: `+0.0066433667`
- bootstrap 양수 비율: 100%
- 모든 seed의 Q1/Q2/H2 score, 1-NMAE, FICR: 양수

따라서 그룹 1은 strict 승격했다.

### 그룹 2 — near-stable

그룹 2 weight는 `0.05`다.

| 구간 | score | 1-NMAE | FICR |
|---|---:|---:|---:|
| Q1 | +0.0010013803 | +0.0000073280 | +0.0019954327 |
| Q2 | +0.0015700994 | +0.0004819088 | +0.0026582899 |
| H2 | +0.0025145254 | +0.0007080832 | +0.0043209675 |
| 전체 | **+0.0018998662** | +0.0004667328 | +0.0033329996 |

- 양수 월: 10/12
- issue-block bootstrap q05: `+0.0007564873`
- bootstrap 양수 비율: 99.25%
- 유일한 strict 실패: seed 202의 Q1 1-NMAE `-0.000017529`
- 그 seed의 Q1 score와 FICR은 모두 양수

엄격 후보에서는 그룹 2를 기존 공개 확정 alpha-0.2375 값으로 유지한다.
도전 후보에서만 명시적 seed-component floor `-0.000025`를 적용한
`near_stable` 등급으로 포함한다.

### 그룹 3 — strict 고해상도 KMA 전문가

pooled 그룹 3은 독립 seed·월 게이트를 통과하지 못해 사용하지 않았다. 대신 이미
검증된 actual UMKR 2023 학습 → locally emulated UMKR 2024 확인 구조의 그룹 3
전문가를 사용한다.

- 전체 score 증분: `+0.0033379520`
- Q1/Q2/H2 양 구성요소
- 양수 월: 11/12
- 모든 사전등록 안정성 게이트 통과

JMA 그룹 3 또는 pooled 그룹 3과의 단순 convex blend도 확인했지만, 최종 오차
상관이 약 `0.999`여서 고해상도 KMA 단독보다 약했다.

## 폐기한 방법

- JMA 기반 공식 FICR 기대효용 Bayes action:
  평균은 개선됐지만 그룹 1은 월 7/12 및 H2 bootstrap 실패, 그룹 3은 H2
  1-NMAE와 bootstrap 실패.
- JMA + KMA UMKR 원천 feature 공동학습 그룹 3:
  모든 안정성 게이트는 통과했지만 전체 증분 `+0.001894`로 기존 두 단독 전문가에
  지배당함.
- 예측 결과의 구간별/행별 라우팅:
  모델 오차 상관이 지나치게 높고 현재 H2 재사용 문제에서 selector 분산만 늘리므로
  생성하지 않음.

### 후속 확인: pooled 모델에 UMKR 원천 feature 직접 추가

세 그룹 pooled 모델에 actual UMKR 두 지점 2023과 locally emulated UMKR
2024를 JMA와 함께 추가했다. 2023 Q4에서 선택된 공통 quantile은 `0.75`였다.

- 그룹 1 전체 score: `+0.008340` — 양수 월 8/12로 탈락
- 그룹 2 전체 score: `+0.001199` — Q2 FICR 및 bootstrap 탈락
- 그룹 3 전체 score: `+0.001936` — strict 통과했지만 기존 UMKR 전문가
  `+0.003338`에 지배당함

그룹 3 보정 이동 상관은 `0.8702`였지만 최종 오차 상관은 `0.999750`이었다.
고정 50:50 결합의 전체 증분도 `+0.003057`로 기존 UMKR 단독보다 낮았다.
따라서 production 후보를 생성하지 않았고 현재 제출 순서를 유지한다.

## Controlled exploratory: 그룹 1 fixed residual stack

strict pooled 그룹 1 위에 direct JMA 그룹 1이 incumbent에서 이동한 양의 1/8만
추가하고, 총 이동은 계속 설비용량의 5%로 제한했다. 비율을 그룹·계절·행별로
달리하지 않았다.

| 구간 | score | 1-NMAE | FICR |
|---|---:|---:|---:|
| Q1 | +0.013667 | +0.001786 | +0.025548 |
| Q2 | +0.007603 | +0.000043 | +0.015162 |
| H2 | +0.013726 | +0.002470 | +0.024982 |
| 전체 | **+0.012364** | +0.001678 | +0.023050 |

- timestamp 기준 양수 월: 11/12
- issue-block bootstrap 10,000회 q05: `+0.005964`
- bootstrap 양수 비율: `99.98%`
- 최대 이동: 설비용량의 5%

그룹 3에도 동일한 1/8 규칙을 적용했지만 전체 score가
`+0.003337 → +0.003145`로 하락해 사용하지 않았다.

이 조합은 두 원천 전문가가 각각 strict였더라도 1/8 비율 자체를 반복 확인된 2024
benchmark에서 선별했으므로 strict로 승격하지 않는다. 구현과 감사 보고서는 각각
`experiments/compose_residual_stack_candidate.py`,
`artifacts_final/diagnostics/kma_jma_pooled_all3_g1_residual125_g2near_umkrg3_20260726.json`
이다.

## 규정 및 재현성

DACON 규칙상 외부 공개 데이터는 합법적·재현 가능하고 각 예측 기준 시점 이전에
공개된 정보만 사용해야 한다. 원격 API 모델 추론은 금지되므로 API는 archived
operational weather forecast의 원본 수집에만 사용했고 발전량 모델 학습·추론은
모두 로컬에서 수행했다.

- DACON 규칙:
  https://dacon.io/competitions/official/236727/overview/rules
- DACON 평가:
  https://dacon.io/competitions/official/236727/overview/evaluation
- JMA MSM 운영 사양:
  https://www.jmbsc.or.jp/jp/online/file/f-online10200.html
- Open-Meteo Previous Runs API:
  https://open-meteo.com/en/docs/previous-runs-api
- Open-Meteo 라이선스:
  https://open-meteo.com/en/licence

JMA 원본 JSON, 요청 URL, checksum, feature CSV, manifest를 2023·2024·2025 각각
보존했다. 선택된 forecast offset에 150분의 보수적 공개 지연을 더한 후에도
예측 기준 시점 위반은 0건이며 최소 여유는 30분이다. 세 manifest 모두
`competition_eligible=true`이고 strict 파일 검증을 통과했다.

## 제출 순서

1. 엄격 후보를 먼저 제출한다.
2. 점수가 0.65 미만이면 그룹 2 열만 다른 near-stable 도전 후보를 제출한다.
3. 그래도 0.65 미만이면 그룹 1 열만 추가로 다른 fixed residual-stack 후보를
   제출한다.
4. 각 연속 결과의 차이를 그룹 2와 그룹 1 stack의 독립 public 효과로 기록한다.
5. 공개 점수로 계절 mask, 행 gate, blend weight를 다시 선택하지 않는다.

관련 예측 결합 연구도 복잡한 regime selector보다 정확성과 오류 다양성이 확인된
모델의 단순·강건한 결합을 권고한다. 이번 결과에서도 예측 평균보다 그룹 열 단위의
독립 전문가 조합이 더 안정적이었다.

- Wang et al. (2023), forecast combination review:
  https://doi.org/10.1016/j.ijforecast.2022.11.005
- Tashman (2000), rolling-origin out-of-sample evaluation:
  https://doi.org/10.1016/S0169-2070(00)00065-0
- Muñoz et al. (2020), feature-driven renewable forecasting and trading:
  https://arxiv.org/abs/1907.07580

## 공개 결과 업데이트 — 2026-07-26 09:16 KST

| 제출 | 구성 | 총점 | 1-NMAE | FiCR |
|---|---|---:|---:|---:|
| 1501926 | pooled G1 + incumbent G2 + UMKR G3 | 0.6450069210 | 0.8754603805 | 0.4145534615 |
| 1501927 | pooled G1 + pooled G2 + incumbent G3 | **0.6458227359** | **0.8758052670** | 0.4158402047 |
| 1501928 | residual G1 + pooled G2 + UMKR G3 | 0.6457889498 | 0.8756700069 | **0.4159078927** |

제출 1501927이 새 공개 최고점이다. 1501928은 FiCR은 `+0.0000676880`
높지만 1-NMAE가 `-0.0001352601` 낮아 총점이 `-0.0000337861` 낮다.

세 결과만으로는 G1 residual, G2 pooled, G3 UMKR의 독립 효과를 식별할 수 없다.
각 효과를 `x`, `y`, `z`라고 하면 공개 결과가 제공하는 식은 다음 두 개뿐이다.

- `x + y = score(1501928) - score(1501926) = +0.0007820288`
- `x + z = score(1501928) - score(1501927) = -0.0000337861`

따라서 다음 단일 제출은 아래 브리지 후보로 고정한다.

`artifacts_final/candidates/kma_jma_pooled_all3_g1g2_umkr_hybrid_g3_nearstable_20260726.csv`

- 구성: pooled G1 + pooled G2 + UMKR G3
- SHA-256:
  `3e26d2dcac8374580a2c78f9e7035a3edd85aa0d2b71a3a0a9c5797c88640f35`
- 행/열 계약: 8,760행,
  `forecast_id, forecast_kst_dtm, kpx_group_1, kpx_group_2, kpx_group_3`
- 기존 composition validator 통과

브리지 점수를 `D`라고 하면 독립 공개 효과가 정확히 다음처럼 복원된다.

- G1 residual 효과: `score(1501928) - D`
- G2 pooled 효과: `D - score(1501926)`
- G3 UMKR 효과: `D - score(1501927)`

평가지표가 그룹별 점수의 macro 평균이므로 그룹 열의 기여는 가법적이다. 브리지 결과는
각 변화의 공개 전이 방향을 진단하고 이미 제출된 후보 중 대표 제출을 선택하는 데만
사용한다. 그 결과로 새 조합, 행별 gate, blend weight를 만들지는 않는다.
