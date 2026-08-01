# NOAA 다지점 issue-time 관측 후속 검증

작성일: 2026-08-02

## 결론

새 제출 후보는 없다. 현 공개 최고 `1502437`을 유지한다.

API 키가 필요 없는 NOAA Global Hourly의 인근 4개 관측소를 추가해 기존
태백 단일 ASOS 가설을 확장했다. 모든 관측은 예측 기준시각보다 최소 2시간
이전으로 잘랐다. G3 expert router와 G1·G2 순수 관측 증분 모델 모두 잠금
승격 조건을 통과하지 못했다.

- G3 router: H2 평균 score `+0.0006295`, 1-NMAE `+0.0004056`,
  FiCR `+0.0008535`
- 그러나 동일한 관측 없는 router 대조군보다 score `-0.0001137`,
  FiCR `-0.0002535`
- 7월 score `-0.0008445`, issue bootstrap score q05
  `-0.0009246`
- G1·G2 observation-only increment: Q2 승격 정책 `0 / 9`
- 가장 작은 정책도 score `-0.0002270`, 1-NMAE `-0.0000347`,
  FiCR `-0.0004194`

따라서 H2를 보고 정책을 다시 고르거나 제출 CSV를 만들지 않았다.

## 왜 이 경로를 선택했는가

모기 궤적 대회의 성공처럼 실제 오차 상관을 낮추려면 기존 NWP와 다른
causal 입력이 필요하다. 기존 태백 ASOS 한 지점은 G3 H2 score를
`+0.0024036` 개선했지만, 관측 없는 동일 대조군과의 증분 및 월별 안정성
조건에서 실패했다. 단일 지점의 지형 대표성 부족인지 확인하기 위해 고도와
방향이 다른 관측망을 추가했다.

검토한 풍력 우수사례의 전이 가능성은 다음과 같다.

- [KDD Cup 2022 공식 순위](https://baidukddcup2022.github.io/)와
  [1위 HIK 논문](https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_0518.pdf)은
  최근 실제 발전량, 다중 관계 그래프, time-phased tree의 결합이 핵심이다.
  BARAM test에는 2025 실제 발전량이 없으므로 live-power 전환 규칙은 사용할
  수 없다. 그래프·lead-phase·deep/tree 결합 자체는 기존 공개 실패 계열에서
  이미 검증됐다.
- [GEFCom2014 풍력 우승 방법](https://www.sciencedirect.com/science/article/pii/S0169207016000145)은
  zone/quantile별 GBM과 wind smoothing, cross-sectional 결합을 사용한다.
  현재 프로젝트의 quantile core, temporal smoothing, pooled group 실험과
  중복된다.
- [WindFM 공식 저장소](https://github.com/shiyu-coder/WindFM)는 대규모
  사전학습이라는 새 방향이지만, zero-shot inference가 과거 발전량 context를
  요구한다. 연중 2025 실제 발전량이 없는 현재 test contract에는 직접
  적용할 수 없다.

남은 전이 가능한 독립 입력으로 예측 시점 이전 지상관측을 선택했으며,
[NOAA/NCEI ISD 공식 데이터](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)를
사용했다.

## 데이터와 인과성 감사

풍력단지 중심 `(37.282, 128.963)`에서 장기 시간자료가 있는 네 지점을
사전에 고정했다.

| station | 위치 | 고도 | 단지 거리 |
|---|---|---:|---:|
| 47100099999 | 대관령 | 844.0 m | 49.6 km |
| 47105099999 | 강릉 | 26.1 m | 55.0 km |
| 47121099999 | 영월 | 237.0 m | 45.3 km |
| 47130099999 | 울진 | 51.2 m | 51.0 km |

2023·2024 원자료를 보존하고 2024 frozen OOF의 367개 issue를 구성했다.

- observation feature: 380개
- history windows: 1/3/6/12/24시간
- safe cutoff: `issue time - 120 minutes`
- 관측 join: 1,468개 station-issue
- 미래 관측 사용: 0건
- minimum availability margin: 0분
- 장기 동시 결측은 보간하지 않고 count=0으로 유지

원자료, URL, 다운로드 시각, SHA-256과 관측소 메타데이터는
`artifacts_final/external_weather/noaa_isd/manifest_2024.json`에 기록했다.

## 검증 1: G3 six-hour expert router

기존 관측 router를 그대로 사용했다. Q1에서 utility model을 학습하고 Q2에서
정책을 고른 뒤, 선택된 한 정책만 H1 재학습 후 H2에 열었다. 두 CatBoost
seed가 모두 양수이고 관측 포함 branch가 동일한 no-observation control을
이겨야 한다.

Q2에서 3개 정책이 개발 gate를 통과했고 `moderate_a15_c05`가 선택됐다.
H2에서는 상위 5% six-hour block을 spatiotemporal expert 방향으로 15%
이동했다.

| 비교 | score | 1-NMAE | FiCR |
|---|---:|---:|---:|
| active 대비 NOAA router | +0.0006295 | +0.0004056 | +0.0008535 |
| active 대비 no-observation control | +0.0007432 | +0.0003795 | +0.0011070 |
| NOAA 증분 | **-0.0001137** | +0.0000261 | **-0.0002535** |

7월과 bootstrap이 음수이고 observation branch가 control보다 낮으므로
`rejected`다. 11월은 기존 단일 ASOS와 달리 양수였지만 이는 전체 계약을
완화할 근거가 아니다.

## 검증 2: G1·G2 paired residual difference

G3의 반복 공개 실패 family를 피하기 위해 G3를 완전히 고정했다. G1·G2에는
두 shallow LightGBM residual model을 같은 seed와 구조로 학습했다.

1. control: active + official NWP/time context
2. observation: control + NOAA 380개 issue features
3. 허용 correction: `observation prediction - control prediction`만 사용
4. cap: 설비용량의 0.25/0.50/1.00%
5. alpha: 0.25/0.50/1.00

Q1 학습 후 Q2에서 총 9개 정책을 평가했다. 가장 작은
`alpha=0.25, cap=0.25%`가 가장 덜 나빴지만 세 component가 모두 음수였다.

| component | Q2 delta |
|---|---:|
| score | -0.0002270 |
| 1-NMAE | -0.0000347 |
| FiCR | -0.0004194 |
| worst month score | -0.0005748 |

Q2 적격 정책이 없으므로 H2는 열지 않았다.

## 판단

다지점 지상관측은 예측 시각에 이용 가능한 독립 정보지만, 이번 문제의
12~35시간 발전량 오차를 안정적으로 수정하지 못했다. G3의 평균 개선은
관측 자체가 아니라 기존 expert router의 일반 선택 효과로 설명되며,
G1·G2에서는 순수 관측 증분 방향이 즉시 음수였다.

다음 계열을 종료한다.

1. 동일 router에 관측소 수만 더 늘리는 탐색
2. H2 월을 보고 지점·window·coverage를 다시 선택하는 탐색
3. active 위의 작은 NOAA observation residual correction

새로운 발전량 history나 설비 운영상태가 없는 상황에서는 공개 우수사례의
핵심 live-power 메커니즘을 정직하게 재현할 수 없다. 추가 입력이 생기기
전까지는 `1502437` 유지가 가장 확실하다.

## 재현

```powershell
python -m experiments.build_noaa_isd_issue_features

python -m experiments.kma_observation_block_router `
  --observation-features artifacts_final/external_weather/noaa_isd/issue_features_2024.csv `
  --manifest artifacts_final/external_weather/noaa_isd/manifest_2024.json `
  --output artifacts_final/diagnostics/noaa_isd_multistation_block_router_20260802.json

python -m experiments.noaa_observation_incremental_residual
```

진단 JSON과 원자료는 재현 가능한 artifact 트리에 두고, 코드·테스트·결정
보고서만 Git에 포함한다.
