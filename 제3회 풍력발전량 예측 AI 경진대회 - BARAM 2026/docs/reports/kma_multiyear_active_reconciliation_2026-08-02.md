# KMA 다년 확장·G3 병목 재검증 보고서

작성일: 2026-08-02

## 결론

병목을 세 경로로 다시 점검했다. 2022년 KMA 운영예보 추가, G3의 SCADA
가용성 EDA, 유사 경진대회 우승 방법론의 미적용 영역을 각각 검증했다.

핵심 결론은 다음과 같다.

1. **“2022년을 더 학습하면 전체 성능이 오른다”는 가설은 기각됐다.**
   동일 코드의 2023-only 대조군 예상 macro 증분은 `+0.0040748863`,
   2022+2023 실험은 `+0.0035386742`로 다년 확장이 `-0.0005362121`
   낮았다.
2. **G3 단독 로컬 효과는 재현됐다.** 현 활성 OOF에서 G3만 교체하면
   score `+0.0012430465`, 1-NMAE `+0.0002842032`, FiCR
   `+0.0022018899`였다. Q1, Q2, H2와 12개월 score가 모두 양수였고
   10,000회 40/60 보완 표본과 2,000회 H2 issue-block bootstrap도
   통과했다.
3. **그러나 제출 후보로 승격하지 않는다.** UMKR, JMA GSM, JMA MSM 등
   로컬 양성 G3 외부기상 교체가 공개 점수에서 반복 역전됐고, 현재 정책은
   이 family를 동결한다. G3 단독 로컬 증분을 공개 점수에 그대로 더해도
   `0.6473681379`로 0.65에 미달한다.
4. **새 submission CSV는 생성하지 않았다.** 제출 `1502437`, 공개 점수
   `0.6461250914`를 유지한다.

## 1. 추가 데이터 EDA

### G3 라벨·SCADA 가용성

`experiments.scada_proxy_experiment._hourly_scada`의 공식 +60분 정렬을
사용해 G3 시간별 SCADA와 라벨을 연도별로 다시 대조했다.

| 연도 | G3 SCADA 행 | G3 라벨 행 | 겹치는 행 | 상관계수 |
|---|---:|---:|---:|---:|
| 2022 | 0 | 0 | 0 | 계산 불가 |
| 2023 | 8,759 | 8,759 | 8,759 | 0.9999345 |
| 2024 | 8,784 | 8,778 | 8,778 | 0.9986390 |

2023/2024의 거의 완전한 상관은 SCADA 집계와 라벨 정렬이 올바름을
보여준다. 2022년은 정렬 문제가 아니라 G3 발전단지의 라벨과 SCADA가
동시에 부재한다. 따라서 다음 두 경로를 닫는다.

- 2022 G3 SCADA를 직접 합산해 라벨을 복원하는 경로
- 동일 단지의 2022 SCADA operating state를 사전학습하는 경로

G1/G2의 2022 관측을 G3에 전이하는 경로는 과거 cross-group pseudo-label,
multi-task ordinal 실험에서 H2 방향이 역전돼 이미 폐기됐다.

### 2022 KMA 운영예보 복원

G3의 직접 관측이 없으므로 합법적이고 인과적인 외생변수인 과거 KMA
운영예보를 복원했다.

- provider: Korea Meteorological Administration
- model: UMRG N512
- issue days: 365
- target timestamps: 8,759
- raw objects: 1,460
- variables: 850/700 hPa의 u/v 문맥
- availability audit: 730행, 위반 0건
- 최소 공개 여유: 240분
- feature SHA-256:
  `266520057b743c47be9f8905cb03b93b0f623cb63988464707002933c336f7e7`

API key는 manifest와 URL에 기록하지 않았고, manifest의 모든 소스 URL은
key를 제거한 상태다. 대회 규정상 예측 기준시각 전에 공개된 외부 데이터만
사용했으며 test 실제 발전량은 사용하지 않았다. 현재 외부 데이터 허용
조건은 [DACON 공식 규정](https://dacon.io/competitions/official/236727/overview/rules)을
기준으로 재확인했다.

## 2. 가설 검증 설계

실험군과 대조군은 동일한 pooled quantile 모델, feature schema, alpha 탐색,
Q1-only blend weight 선택, 2024 확인면을 사용한다.

| 항목 | 대조군 | 실험군 |
|---|---|---|
| 학습 연도 | 2023 | 2022+2023 |
| alpha 선택 | 2023 Q4 | 2023 Q4 |
| blend 선택 | 2024 Q1만 | 2024 Q1만 |
| 확인 | Q2, H2, 월, seed, issue bootstrap | 동일 |
| 공개 점수 선택 사용 | 없음 | 없음 |

실험군의 2024 전체 delta가 양수인지 보는 것만으로는 기존 활성 후보와의
중복을 알 수 없다. 그래서 두 번째 단계에서 frozen active OOF를 다시 불러와
각 그룹을 하나씩 정확히 교체했다. 이 단계는 현재 제출 구성과의 실제 증분만
측정한다.

## 3. 다년 확장 ablation 결과

상위 실험 자체의 예상 macro delta는 다음과 같다.

| 모델 | 예상 macro score delta |
|---|---:|
| 2023-only | +0.0040748863 |
| 2022+2023 | +0.0035386742 |
| 다년 - 대조군 | **-0.0005362121** |

각 모델이 Q1에서 고른 표면 전체를 직접 비교해도 2022+2023은 full-year
score `-0.0007517258`, 1-NMAE `-0.0000235723`, FiCR
`-0.0014798793`였다. 추가 연도 자체가 일반화 개선을 만들었다는 주장은
성립하지 않는다.

다만 그룹별로는 G3에만 제한된 개선이 남았다. 이는 2022 G1/G2 라벨을
pooled 학습의 보조 표본으로 사용해 G3의 기상-출력 공통 구조를 더 안정화한
효과로 해석할 수 있지만, G3의 2022 직접 라벨을 복원한 것은 아니다.

## 4. 현 활성 OOF와의 정확한 reconciliation

### 그룹별 교체

| 2022+2023 교체 | full score delta | full 1-NMAE delta | full FiCR delta | 판정 |
|---|---:|---:|---:|---|
| G1만 | -0.0018254935 | -0.0000067245 | -0.0036442625 | 기각 |
| G2만 | -0.0001811810 | -0.0000565645 | -0.0003057974 | 기각 |
| G3만 | **+0.0012430465** | **+0.0002842032** | **+0.0022018899** | 통계 통과, family guard 기각 |
| upstream G1+G3 | -0.0005824469 | +0.0002774787 | -0.0014423726 | 기각 |

upstream 실험이 선택한 G1+G3를 그대로 적용하면 현재 활성 G1을 더 약한
KMA-only G1으로 되돌리기 때문에 전체 score와 FiCR가 음수다. 따라서
upstream의 `projected_public_score=0.6496638`을 현재 제출 개선 예상치로
사용하면 안 된다.

### G3 단독 기간 안정성

| 기간 | score delta | 1-NMAE delta | FiCR delta |
|---|---:|---:|---:|
| Q1 | +0.0010254673 | +0.0002254396 | +0.0018254951 |
| Q2 | +0.0015372131 | +0.0003975686 | +0.0026768576 |
| H2 | +0.0012454792 | +0.0002590953 | +0.0022318630 |
| 전체 | +0.0012430465 | +0.0002842032 | +0.0022018899 |

최악 월은 12월이고 score delta는 `+0.0003371391`이다. 전체 3그룹 기준
평균 이동은 설비용량의 `0.1943%`, p95 `1.0084%`, 최대 `2.8197%`다.

### 40/60 보완 표본과 issue-block

동일 timestamp를 40%와 보완 60%로 나눠 10,000회 평가했다. 아래 값은
세 구성요소 delta의 하위 5% 분위수다.

| 표본 | score q05 | 1-NMAE q05 | FiCR q05 |
|---|---:|---:|---:|
| IID 40% | +0.0005507671 | +0.0001997674 | +0.0008441707 |
| IID 보완 60% | +0.0007962837 | +0.0002266151 | +0.0013226635 |
| 월 층화 40% | +0.0005718670 | +0.0001985908 | +0.0008755604 |
| 월 층화 보완 60% | +0.0007953476 | +0.0002269123 | +0.0013226034 |
| H2 issue-block | +0.0005444426 | +0.0000785360 | +0.0008959200 |

통계 게이트는 통과한다. 그러나 통계적으로 안정된 2024 OOF가 곧 공개 전이를
뜻하지 않는다는 사실이 G3에서 이미 반복 관찰됐다.

## 5. 공개 실패 family guard

식별 가능한 공개 대비에서 G3 교체는 다음과 같이 모두 역전됐다.

| 계열 | 공개 score delta | 공개 FiCR delta |
|---|---:|---:|
| UMKR G3 | -0.0003361416 | -0.0005580425 |
| JMA GSM G3 | -0.0011973661 | -0.0020292405 |
| JMA MSM plateau G3 | -0.0005284116 | -0.0010569446 |
| trajectory TCN G3 | -0.0006674758 | -0.0014950275 |

새 G3 후보는 KMA UMRG 다년 pooled 모델이라는 차이가 있지만, 외부기상으로
G3를 교체하고 로컬 FiCR 경계를 개선하는 동일한 위험 family다. 공개 실패
guard를 무시하려면 독립된 새 연도 G3 라벨 또는 공개 구간과 무관한 추가
확인면이 필요하다. 현재는 둘 다 없다.

수치 규모도 제출을 정당화하지 않는다.

- 현 frozen local score `0.6480116126` + 로컬 delta = `0.6492546591`
- 현 공개 score `0.6461250914` + 완전 전이 가정 = `0.6473681379`
- 완전 전이를 가정해도 공개 0.65까지 `0.0026318621` 부족

따라서 통계 게이트와 별도로 historical public failure guard에서 기각한다.

## 6. 유사 경진대회 우수 사례와 현재 적용 여부

### GEFCom2014

[확률적 풍력 부문 우승 논문](https://www.sciencedirect.com/science/article/pii/S0169207016000145)은
zone/quantile별 gradient boosting, 100m wind smoothing, cross-sectional
정보를 결합했다. 현재 프로젝트에서는 quantile GBM, LDAPS temporal wind
smoothing, cross-group pooled 모델을 각각 검증했다. smoothing과
cross-sectional 후처리를 다시 반복할 새 근거는 없다.

### KDD Cup 2022 Wind Power Forecasting

[공식 순위와 자료](https://baidukddcup2022.github.io/) 및
[1위 HIK 방법](https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_0518.pdf)은
tree와 deep temporal 모델의 상보적 융합을 강조한다. 현재 프로젝트는
LightGBM/CatBoost, TCN, pooled multi-task, trajectory ensemble을 이미
검증했지만 G3 H2/공개 FiCR 전이가 일치하지 않았다. 단순 architecture
ensemble 확대보다 독립 확인면 확보가 먼저다.

### EEM 2017

[우승 방법](https://pureportal.strath.ac.uk/en/publications/cluster-based-regime-switching-ar-for-the-eem-2017-wind-power-for/)은
전일 weather의 k-medians regime과 교차검증 기반 AR 전환을 사용했다.
현재 weather regime, KMeans/analog scenario, issue routing 실험이 같은
가설을 다뤘고 안정된 H2 이득이 없었다. regime 수를 다시 탐색하는 것은
현재 병목을 해소하지 못한다.

이 비교에서 남는 실질적 교훈은 모델 종류를 더 늘리는 것이 아니라, 동일
운영 NWP의 더 긴 causal history를 확보하고 독립 연도로 확인하는 것이다.
이번 2022 확장은 그 가설을 처음으로 직접 검증했고, 전체 개선은 기각했다.

## 7. 산출물과 파일 트리 정리

이번 검증의 canonical 파일은 다음 세 개다.

```text
experiments/
  kma_multiyear_active_reconciliation.py
tests/
  test_kma_multiyear_active_reconciliation.py
docs/reports/
  kma_multiyear_active_reconciliation_2026-08-02.md
```

재현용 대용량 산출물은 역할별 기존 트리에 유지한다.

```text
artifacts_final/
  external_weather/kma_um_regional_context_2022/
    features.csv
    manifest.json
    raw/...
  lineage/
    kma_umrg_full2022_pooled_all3_20260802.npz
    kma_umrg_2023only_pooled_all3_control_20260802.npz
  diagnostics/
    kma_umrg_full2022_pooled_all3_20260802.json
    kma_umrg_2023only_pooled_all3_control_20260802.json
    kma_multiyear_active_reconciliation_20260802.json
```

최종본이 따로 존재하는 smoke 보고서 7개, `base_v2/smoke`, 2022 KMA smoke
다운로드, 빈 실행 로그 2개를 삭제했다. Python `__pycache__`와 pytest cache도
최종 테스트 이후 삭제한다. 기존 연구 코드·보고서·candidate는 사용자 작업과
공개 실패 계보를 보존하기 위해 임의로 이동하거나 삭제하지 않는다.

## 최종 결정

- 승격 후보: 없음
- 신규 제출 파일: 없음
- 유지 제출: `1502437`
- 다음 유효 경로: 새로운 G3 독립 라벨 연도 또는 완전히 다른 causal core
  validation surface가 확보될 때만 G3 family를 재개

정식 진단:
`artifacts_final/diagnostics/kma_multiyear_active_reconciliation_20260802.json`
