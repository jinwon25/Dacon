# BARAM 2026 우선순위 실행 기록 — 2026-07-25

## 결론

2026-07-25 KST 기준 DACON 코드 공유 탭에는 공식 RandomForest 베이스라인과
공식 평가 산식만 공개되어 있다. 참가자가 새로 공개한 고성능 코드나 해법은 확인되지
않았고, GitHub에서도 대회 ID `236727`, 대회명, `BARAM 2026`으로 직접 연결되는
공개 구현을 찾지 못했다.

따라서 현재 실행 우선순위는 다음과 같다.

1. 예보 발표 issue 전체를 한 블록으로 묶은 nested 검증으로 quantile LightGBM Base v2를 확정한다.
2. train 기간 SCADA 풍속만 사용해 NWP→현장 풍속 보정값을 만들고 Base v2와 동일 OOF에서 A/B한다.
3. 그룹 3의 train 행에서만 출력제한 의심치를 탐지·제외하고 동일 OOF에서 A/B한다.
4. 앞 단계에서 살아남은 모델에만 CatBoost 다양성 또는 KMA UMRG 보정을 결합한다.
5. 온라인 DACON 제출은 자동으로 수행하지 않는다. 잠금 검증을 통과한 CSV와 재현 보고서만 만든다.

## 온라인 코드 공유 점검

- [DACON 코드 공유 탭](https://dacon.io/competitions/official/236727/codeshare)
  - `[Baseline] 기상 예보 데이터 기반 RandomForest 풍력발전량 예측`
  - `평가 산식 코드`
- [공식 대회 규칙](https://dacon.io/competitions/official/236727/overview/rules)은
  각 예측기준시점 이전에 실제로 공개된 정보만 허용한다. 이에 따라 test 기간의 실제
  발전량·SCADA는 사용하지 않고, SCADA는 train 기간 NWP 보정 target으로만 사용한다.
- 공개 코드 검색 결과는 공식 baseline 외에 이 대회의 참가자 해법으로 확인할 수 있는
  저장소가 없었다. 라이선스가 불명확한 타 대회 코드는 복사하지 않았다.

## 방법론 근거와 적용

### 그룹별 quantile GBDT

GEFCom2014 wind track 우승 해법은 발전 구역과 quantile별로 독립적인 gradient
boosting model을 학습했다. 이 대회의 FiCR가 오차 6%와 8%에서 불연속적으로
변하므로, 조건부 중앙값을 고정해서 쓰는 것보다 quantile을 inner holdout에서 직접
선택하는 것이 타당하다.

- Landry et al., *Probabilistic gradient boosting machines for GEFCom2014 wind
  forecasting*: https://doi.org/10.1016/j.ijforecast.2016.02.002

적용 구현:

- `experiments/nested_quantile_base.py`
- alpha 후보 `0.55, 0.60, 0.65, 0.70, 0.75`
- `all`/`eligible_only` 학습 정책과 반복 횟수를 바로 앞 완료 계절에서 선택
- 선택에 사용하지 않은 2024 계절을 outer fold로 평가
- 24시간 purge와 완전한 NWP issue-cycle 블록 유지

### SCADA 기반 NWP 현장 보정

운영 풍력 예측 연구에서는 편향된 NWP를 과거 SCADA로 보정하는 방식이 직접 NWP보다
오차와 평균 bias를 낮출 수 있음을 보였다.

- Narváez et al., *Bias correction of wind power forecasts with SCADA data and
  continuous learning*: https://arxiv.org/abs/2402.13916

적용 구현:

- 과거 완료 계절만 학습하는 rolling OOF 현장 풍속 추정치
- test에는 전체 train 기간으로 학습한 NWP→SCADA 풍속 모델만 적용
- 실제 test SCADA나 예측기준시점 이후 관측값은 사용하지 않음

### 출력제한·비정상 운전 정제

SCADA power curve에서 curtailment, 정지, 이상치는 정상 power curve를 왜곡하며,
이를 분리하거나 robust하게 모델링해야 한다는 근거가 반복해서 보고되어 있다.

- Morrison et al., *Anomaly detection in wind turbine SCADA data for power curve
  cleaning*: https://doi.org/10.1016/j.renene.2021.11.118
- Bull et al., *Bayesian modelling of multivalued power curves from an operational
  wind farm*: https://doi.org/10.1016/j.ymssp.2021.108530
- Zhao et al., *Data-driven correction approach to refine power curve of wind farm
  under wind curtailment*: https://doi.org/10.1109/TSTE.2017.2717021

적용 구현:

- 현재 학습 partition 안에서만 풍속 bin별 중앙 power curve를 추정
- 고풍속·고기대 출력인데 실제 출력이 기대치의 절반 미만인 행만 학습에서 제외
- outer validation 행은 탐지기 학습이나 제외 여부 결정에 사용하지 않음
- 그룹 3 전체 train에서는 17,538개 유효 label 중 342개만 표시되는 보수적 정책

## 실행 결과

### 정식 Quantile LightGBM Base v2

- 산출물: `artifacts_final/base_v2/lgbm_full_20260725`
- 12개 outer fold와 issue-block bootstrap 2,000회를 완료했다.
- L1 기준 대비 macro score는 `0.616652 -> 0.629896`
  (`+0.013244`)였고 FiCR은 `+0.031900` 개선됐다.
- 반면 `1-NMAE`가 `-0.005412` 하락해 단독 승격 조건은 통과하지 못했다.
- bootstrap score q05는 `+0.006813`, 양수 비율은 `99.95%`였다.

### SCADA 풍속 보정 A/B

- 산출물: `artifacts_final/base_v2/ws_cal_full_20260725`
- 비교 보고서:
  `artifacts_final/base_v2/ws_cal_vs_lgbm_full_20260725.json`
- 정식 Base v2 candidate 대비 score `+0.006973`,
  `1-NMAE +0.000439`, FiCR `+0.013507`이었다.
- 그러나 2024-JJA score가 `-0.005375` 하락해 최악 계절 비음수
  조건을 통과하지 못했다. 따라서 SCADA 보정 모델은 승격하지 않았다.

### 그룹 3 출력제한 정제 A/B

- 산출물: `artifacts_final/base_v2/group3_curtailment_full_20260725`
- 비교 보고서:
  `artifacts_final/base_v2/group3_curtailment_vs_lgbm_full_20260725.json`
- 정식 Base v2 candidate 대비 score `+0.002549`,
  `1-NMAE +0.000455`, FiCR `+0.004643`이었다.
- bootstrap score q05는 `+0.000709`, 양수 비율은 `98.8%`였다.
- 2024-MAM score가 `-0.000223`으로 소폭 하락해 단독 승격은 보류했다.

### Incumbent 잠금 블렌드

- 그룹 3 정제 Base v2를 incumbent에 섞는 비율은 2024-Q2에서 `15%`로
  선택됐다.
- 잠금 H2에서 score `0.632811 -> 0.636519` (`+0.003708`),
  `1-NMAE +0.001168`, FiCR `+0.006248`이었다.
- issue-block bootstrap score q05는 `+0.001122`, 양수 비율은 `99.0%`였고
  최악 월을 포함한 모든 승격 조건을 통과했다.
- 최종 로컬 후보:
  `submissions/archive/blend_incumbent_basev2_curtailment_w15_20260725.csv`
- SHA-256:
  `4df42d8042a463a406636c964f2ed239687f6b05facf2d09ae5fc0bacb9f7495`
- 행 수는 8,760개이며 DACON 제출은 수행하지 않았다.

### KMA UMRG와 CatBoost 분기

- KMA UMRG bounded power-curve gate는 앞선 incumbent 후보에 대해 이미
  잠금 검증을 통과했고
  `submissions/blend_best_kma_um_power_curve_gate.csv`를 생성했다.
- 새 15% Base v2 블렌드와 KMA 후보를 직접 합치면 두 변화의 상호작용을
  검증하지 않은 조합이 되므로 합성하지 않았다.
- `KMA UMRG 또는 CatBoost` 분기에서 KMA 후보가 이미 통과했으므로,
  추가 CatBoost 장기 학습은 실행하지 않았다.

## 최종 결정

1. 자동 DACON 제출은 하지 않는다.
2. 새 승격 후보는 `15% 그룹 3 정제 Base v2 + 85% incumbent`로 보존한다.
3. 기존 KMA UMRG 후보는 별도 후보로 유지하고, 두 후보의 합성은 새로운
   OOF 상호작용 검증 없이는 수행하지 않는다.

## 18:55 공개 결과 이후 정정

- KMA UMRG 후보(제출 1501460)는 `0.6422368394`로 새 public best가 됐다.
- 15% Base v2 블렌드(제출 1501461)는 `0.6404014430`으로 실패했다.
- 후자에 누락됐던 test 이동량 조건을 추가해 재평가한 결과, 전체 target cell
  100% 변경 때문에 엄격 승격과 탐색 보존 모두 탈락했다.
- KMA OOF 상호작용을 새로 검증한 그룹 2 국소 오버레이만 탐색 후보로
  보존했다. 상세 근거는
  `docs/reports/tiered_promotion_followup_2026-07-25.md`에 기록했다.
- 그룹 2 국소 오버레이 제출 1501476은 `0.6430629531`로 다시 새 public
  best가 됐다. 동일 gate의 alpha 20% 확장은 H2와 bootstrap 증분이 강하지만
  Q2에서는 alpha 10%보다 낮아 별도 탐색 후보로만 생성했다.
- alpha 20% 제출 1501483도 `0.6435929652`로 공개 개선됐다. 누적 이동
  p95 2.5%·최대 6.0% 상한 안의 마지막 확장인 alpha 23.75%만 후속 탐색
  후보로 생성했으며, 더 강한 확장은 중단했다.
- alpha 23.75% 제출 1501487은 `0.6440998116`으로 다시 공개 개선됐다.
  강도는 더 늘리지 않고 기존 핵심 gate 밖의 상향 행 889개만 추가하는 sparse
  보완 후보를 생성했다.
