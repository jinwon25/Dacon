# 독립 멀티 NWP 및 JMA 2년 year-forward 후속 — 2026-07-28

## 결론

공개 최고는 제출 `1502437`, 점수 `0.6461250914`로 유지한다. 오늘은 제출을
하지 않았다. 검증한 두 신규 구조 모두 strict 승격 대상이 없었고 후보 CSV도
생성하지 않았다.

2026-07-28 공식 리더보드의 10위 점수는 약 `0.66006`으로, 현재 최고와의
차이는 약 `0.01393491`이다. 대회 종료일은 2026-08-14이다. 로컬 기대 이득이
`+0.000216`에 불과하고 issue-block 검증에 실패한 G2 weight-0.10 후보는 이
격차를 해소할 구조적 후보가 아니므로 제출을 보류한다.

- 공식 일정:
  https://dacon.io/competitions/official/236727/overview/description
- 공식 리더보드:
  https://dacon.io/competitions/official/236727/leaderboard

## 1. 공개 양수 G2 factor 확장 재감사

`artifacts_final/candidates/public_positive_pooled_g2_w10_probe_20260727.csv`

- 행: `8,760`
- SHA-256:
  `139a6d4e93d9eaeec3731352f55c9f5e4b145805a5eb14ff0fdcbf978f6192a2`
- 스키마·ID·유한값·범위 검사: 통과
- incumbent factor 대비 2024 full:
  - score `+0.0008554743`
  - 1-NMAE `+0.0002132452`
  - FiCR `+0.0014977034`
- H2:
  - score `+0.0002522133`
  - 1-NMAE `+0.0005048090`
  - FiCR `-0.0000003823`
- issue-cycle 전체:
  - score `-0.0000106283`
  - 1-NMAE `+0.0002193379`
  - FiCR `-0.0002405946`
- bootstrap 양수 비율 `49.06%`, q05 `-0.0015119397`
- 최악 월 2024-09 score `-0.0139826637`
- 공개 단순 외삽 예상점 `0.6463410793`

파일은 재현됐지만 strict promotion은 명확히 실패한다. 이미 공개 양수였던 동일
G2 factor의 작은 강도 확장이라는 점만 탐색 근거이며, 오늘 제출 슬롯을 사용할
근거로는 부족하다.

## 2. 독립 소스 중앙값 앙상블

### 가설

ECMWF IFS, DWD ICON, CMC GEM, KMA UMRG 피처를 한 LightGBM에 결합하면 특정
예보원의 스케일이나 피처 수가 모델을 지배할 수 있다. 각 소스마다 동일한 pooled
quantile 모델을 따로 학습한 뒤 네 예측의 중앙값을 취하면 source outlier를 줄이고
공통 풍력 신호만 남길 수 있다는 가설을 검증했다.

기존 `experiments/multimodel_expanding_quantile_blend.py`에
`--source-aggregation median`을 추가했다.

- Q2: 2024-Q1까지 학습하고 quantile/weight 선택
- Q3: Q2까지 확장 학습 후 확인
- Q4: Q3까지 확장 학습 후 확인
- confirmation seed: `42`, `202`, `2026`
- 공개 점수 및 2025 발전 실측 사용: 없음
- 결과:
  `artifacts_final/diagnostics/multisource_independent_median_20260728.json`

### 결과

| 그룹 | Q2 score | Q3 score | Q4 score | H2 score | 판정 |
|---|---:|---:|---:|---:|---|
| G1 | `+0.0040665` | `-0.0013286` | `-0.0004772` | `-0.0008263` | reject |
| G2 | `+0.0029415` | `+0.0044744` | `-0.0022817` | `+0.0004335` | reject |
| G3 | `+0.0023371` | `+0.0049079` | `-0.0079622` | `-0.0023473` | reject |

G2와 G3는 Q2와 Q3에서 모두 양수였으나 Q4에서 역전했다. G3 Q4 FiCR은
`-0.0132132`로 특히 크게 악화했다. 소스 독립성과 중앙값 결합으로도 역전이
남았으므로 병목은 한 예보원의 outlier보다 계절별 날씨-발전 매핑의
비정상성에 가깝다. 같은 family의 mean 집계나 weight 미세조정은 열지 않는다.

## 3. 2023 이전 archive 감사

배포 대칭 검증을 만들기 위해 ECMWF·DWD·CMC 2023 previous-run 수집 가능성을
감사했다. Open-Meteo 공식 문서상 대부분의 모델은 2024년 1월부터 보존되며,
JMA MSM/GSM은 2018년부터 제공되는 예외다.

- Open-Meteo Previous Runs API:
  https://open-meteo.com/en/docs/previous-runs-api

ECMWF 2023 요청은 day-1 wind 필드가 불완전해 수집기가 중단했다. 부분 파일이나
manifest는 남기지 않았다.

JMA MSM 2022는 7월 초부터 완전했다. 기존 2023–2025와 동일한 3×3 스텐실을
2022-07-04부터 수집했다.

- feature:
  `artifacts_final/external_weather/jma_msm_stencil_2022/features.csv`
- manifest:
  `artifacts_final/external_weather/jma_msm_stencil_2022/manifest.json`
- 4,344시간, 9개 위치, 41개 feature
- timing 위반 `0`
- 최소 공개 여유 `30`분
- feature SHA-256:
  `46be789940a5924b6fc2c9d7eb1a228a51d4f7405b424af9051a9bea08686bc7`
- raw SHA-256:
  `aea98d100b7fba2e82e8544f036f811c5aed66b324cc559e8b3744815da5484d`
- manifest 적격성: `competition_eligible=true`

## 4. JMA MSM 2년 pooled year-forward

`experiments/kma_pooled_group_quantile_blend.py`에 선택적 2022 학습 연도와
연도 간 exact feature-schema 검사를 추가했다.

- alpha 선택:
  `2022 H2 + 2023 Jan–Sep -> 2023 Q4`
- validation:
  `2022 H2 + 2023 전체 -> 2024`
- 선택 alpha: `0.75`
- pooled 학습 행: `39,312`
- 결과:
  `artifacts_final/diagnostics/jma_msm_pooled_2022h2_2023_to_2024_20260728.json`
- 대조:
  `artifacts_final/diagnostics/jma_msm_pooled_2023_to_2024_control_20260728.json`

### 2년 모델

| 그룹 | weight | full score | Q2 score | H2 score | 양수 월 | bootstrap q05 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| G1 | 0.15 | `+0.0035322` | `-0.0017742` | `+0.0033983` | 9/12 | `+0.0008977` | reject |
| G2 | 0.10 | `+0.0024936` | `+0.0003403` | `+0.0024643` | 10/12 | `-0.0011020` | reject |
| G3 | 0.05 | `+0.0018537` | `+0.0030830` | `+0.0008270` | 9/12 | `+0.0001855` | reject |

G3는 Q1/Q2/H2의 score, 1-NMAE, FiCR와 모든 seed가 양수였지만 양수 월
9/12와 bootstrap 양수 비율 96.85%가 strict 기준 10/12와 98%를 넘지 못했다.

### 2023 단년 대조 G3

- full score `+0.0019020`
- Q2 score `+0.0026480`
- H2 score `+0.0018533`
- 양수 월 9/12
- bootstrap q05 `+0.0002633`
- bootstrap 양수 비율 97.10%

2022 H2 추가는 G3 full/H2와 bootstrap을 개선하지 않았다. G1은 Q2가
음수였고 G2는 Q2 1-NMAE와 bootstrap이 음수였다. 따라서 2년 학습 family도
종료하며 생산 모델이나 CSV를 만들지 않는다.

## 결정

1. 공개 incumbent `1502437`을 유지한다.
2. G2 weight-0.10 후보는 재현 가능 상태로 보존하되 제출하지 않는다.
3. 멀티 NWP joint·median 계열은 Q4 역전으로 닫는다.
4. JMA 2022 H2 확장은 추가적인 강도·weight 튜닝 없이 닫는다.
5. 현재 공개 점수의 주 병목은 1-NMAE보다 FiCR이며, 다음 대형 분기는
   2024 반복 튜닝이 아니라 배포 대칭의 독립 신호 또는 명시적인 계절 전환
   불확실성 모델이 확보될 때만 연다.

## 재현

```powershell
python -m pytest `
  tests/test_multimodel_expanding_quantile_blend.py `
  tests/test_kma_pooled_group_quantile_blend.py `
  tests/test_compose_public_positive_factor_expansion.py -q

python -m experiments.multimodel_expanding_quantile_blend `
  --source-aggregation median `
  --validation-only `
  ... # 보고서 sources의 2024/2025 context와 manifest

python -m experiments.kma_pooled_group_quantile_blend `
  --validation-only `
  --targets kpx_group_1,kpx_group_2,kpx_group_3 `
  --context-2022 artifacts_final/external_weather/jma_msm_stencil_2022/features.csv `
  --manifest-2022 artifacts_final/external_weather/jma_msm_stencil_2022/manifest.json `
  --context-2023 artifacts_final/external_weather/jma_msm_stencil_2023/features.csv `
  --manifest-2023 artifacts_final/external_weather/jma_msm_stencil_2023/manifest.json `
  --context-2024 artifacts_final/external_weather/jma_msm_stencil_2024/features.csv `
  --manifest-2024 artifacts_final/external_weather/jma_msm_stencil_2024/manifest.json `
  --select-alpha-2023-q4
```
