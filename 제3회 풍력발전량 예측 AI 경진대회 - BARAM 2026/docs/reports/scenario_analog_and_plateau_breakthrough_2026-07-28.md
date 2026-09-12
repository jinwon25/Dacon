# 잔차 시나리오와 JMA plateau 후속 연구 — 2026-07-28

## 결론

공개 최고는 제출 `1502437`, 점수 `0.6461250914`로 유지한다. 이번 연구에서는
서로 독립적인 세 경로를 확인했다.

1. 24시간 NWP 이슈 단위 analog residual scenario는 정산 효용을 직접 최적화해도
   순방향 H2에서 재현되지 않았다.
2. 이웃 시나리오 65%/80% 합의 게이트도 FiCR 전이를 안정화하지 못했다.
3. 기존 JMA MSM pooled G3의 Q1 근접최상 가중치 plateau를 평균한 `0.040625`
   정책은 기존 엄격 로컬 게이트를 모두 통과했다.

세 번째 경로로 로컬 연구 후보를 생성했고, 이후 단일 G3 controlled probe로
제출 `1504352`에 사용했다. 공개점수는 하락했다.

- 후보:
  `artifacts_final/candidates/jma_msm_g3_plateau_mean_w040625_20260728.csv`
- SHA-256:
  `da0bbb9fba76189be89a95d3b99c6e736403acb8078fbc21747a380f36253900`
- 행 수: `8,760`
- 변경 그룹: G3만
- G1/G2: 공개 incumbent와 bitwise 동일
- 후보 검증기: 통과
- 제출 `1504352`: `0.6455966798`
- 공개 판정: reject

공개 incumbent `1502437` 대비:

- score `-0.0005284116`
- 1-NMAE `+0.0000001213`
- FiCR `-0.0010569446`

G3만 변경됐으므로 implied G3 delta는 각각 `-0.0015852348`,
`+0.0000003639`, `-0.0031708338`이다. 사실상 NMAE는 변하지 않았고
FiCR가 명확히 악화됐다.

## 1. 참고한 방법론

Analog Ensemble은 현재 NWP와 유사한 과거 NWP의 관측 결과를 조건부 예측분포로
사용한다. 원 논문은 흐름 의존 오차를 포착하면서 낮은 실시간 계산비용으로
보정된 확률예보를 만드는 구조를 제시한다.

- Delle Monache et al., *Probabilistic Weather Prediction with an Analog
  Ensemble*: https://doi.org/10.1175/MWR-D-12-00281.1
- Alessandrini et al., *A novel application of an analog ensemble for
  short-term wind power forecasting*:
  https://doi.org/10.1016/j.renene.2014.11.061

독립 시점별 analog는 시간 상관을 훼손할 수 있다. 따라서 이번 구현은 24시간
잔차 경로를 통째로 이웃 표본으로 보존했다. 이는 Schaake shuffle을 통해
시공간 상관을 복원한 연구와, 비정규·상관 오차 시나리오를 조건부로 생성한
연구를 단순화한 형태다.

- Sperati et al., *Gridded probabilistic weather forecasts with an analog
  ensemble*: https://doi.org/10.1002/qj.3137
- Wang et al., *A conditional model of wind power forecast errors and its
  application in scenario generation*:
  https://doi.org/10.1016/j.apenergy.2017.12.039
- Hu et al., *Wind power forecasting errors modelling approach considering
  temporal and spatial dependence*:
  https://doi.org/10.1007/s40565-016-0263-y

여러 표현이나 모델에서 하나의 최적 가중치만 추정하면 추정분산 때문에 단순
평균보다 나빠질 수 있다는 forecast-combination 결과도 반영했다.

- Claeskens et al., *The forecast combination puzzle*:
  https://doi.org/10.1016/j.ijforecast.2015.12.005
- Elliott and Timmermann, *Optimal forecast combinations under general loss
  functions and forecast error distributions*:
  https://doi.org/10.1016/j.jeconom.2003.10.019

## 2. Multi-view analog residual scenario

구현:
`experiments/issue_residual_analog_scenarios.py`

### 구조

- dependency unit: 완전한 NWP issue 하나의 lead 12–35, 총 24시간
- analog 표현:
  - G1/G2/G3 incumbent 발전량 궤적
  - LDAPS 물리변수 + 발전량 궤적
  - GFS 물리변수 + 발전량 궤적
- 각 표현에서 이웃 순위를 독립 계산한 뒤 동일 view weight로 pooling
- 과거 `truth - incumbent` 24시간 잔차 경로를 시나리오로 사용
- 행동 범위: 설비용량 대비 `-1.5%`–`+1.5%`
- 공식 NMAE/FICR의 표본 효용을 최대화
- 6시간 블록 중 예상 효용이 가장 큰 일부에만 적용
- 개발: Q1 archive → Q2
- 고정 적용: H1 archive → H2

### 평균 효용 계열

Q2 월별 최악값까지 양수인 유일 정책은 `k24/a0.50/coverage0.10`이었다.
그러나 H2에서 다음과 같이 반전했다.

- score `-0.00009958`
- 1-NMAE `-0.00002666`
- FiCR `-0.00017250`
- 양수 월 `4/6`
- 최악월 2024-08 `-0.00201933`
- issue bootstrap 양수 비율 `43.2%`
- bootstrap q05 `-0.00103083`

leave-one-month-out 교차적합도 전체 score는 `+0.00015429`였지만 양수 월이
`6/12`, 최악월이 `-0.00231953`이라 안정적이지 않았다.

### 시나리오 합의 계열

비영 행동이 이웃 잔차 시나리오의 최소 65% 또는 80%에서 기준선보다 나을 때만
허용했다. 위험통제 계열에서 Q2가 세 달 모두 양수인 선택 정책은
`k24/a0.50/coverage0.10/p0.65`였다.

H2 결과:

- score `-0.00009548`
- 1-NMAE `+0.00003899`
- FiCR `-0.00022994`
- 양수 월 `4/6`
- issue bootstrap 양수 비율 `41.2%`
- bootstrap q05 `-0.00063327`

합의 게이트는 NMAE 손실을 줄였지만 FiCR 전이를 복구하지 못했다. 이 계열은
후보를 만들지 않고 종료한다.

## 3. JMA MSM micro-weight plateau

기존 2023→2024 JMA MSM pooled G3는 `weight=0.05`에서 Q1/Q2/H2와 모든
시드가 양수였지만 양수 월 `9/12`, bootstrap 양수 비율 `97.1%`로 탈락했다.
기존 가중치 grid의 간격이 `0.025`로 거칠어 Q1 근접최상 구간을 놓칠 가능성을
확인했다.

`experiments/kma_pooled_group_quantile_blend.py`에 다음을 추가했다.

- 사용자 지정 Q1 weight grid
- `plateau_mean` 선택 모드
- Q1 최고점에서 기존 tolerance `0.00025` 안에 있는 가중치를 동일하게 평균
- Q2/H2/월별/시드/부트스트랩은 선택에 사용하지 않음

G3 Q1 near-best plateau:

- `0.0375`
- `0.040625`
- `0.04375`

평균이자 실제 grid 점인 `0.040625`가 선택됐다.

### 로컬 검증

| 구간 | score | 1-NMAE | FiCR |
|---|---:|---:|---:|
| Q1 | `+0.00193131` | `+0.00064641` | `+0.00321620` |
| Q2 | `+0.00363887` | `+0.00053952` | `+0.00673822` |
| H2 | `+0.00165784` | `+0.00047969` | `+0.00283599` |
| full | `+0.00215100` | `+0.00054100` | `+0.00376101` |

- 양수 월: `10/12`
- 음수 월:
  - 2024-07 `-0.00035622`
  - 2024-12 `-0.00011861`
- 모든 seed × Q1/Q2/H2 구성요소 최소값: `+0.00047141`
- issue bootstrap 양수 비율: `99.1%`
- bootstrap q05: `+0.00063262`
- 기존 strict gate: 전부 통과
- 기대 macro score 증분: `+0.00071700`
- 단순 전이 예상 공개점수: `0.64684209`

### 생산 후보 정합성

2023+2024 JMA MSM으로 재학습해 2025를 예측했다.

- 식별자: incumbent와 완전히 동일
- G1/G2: 정확히 무변경
- G3 변경 행: `8,760`
- G3 평균 절대 이동: `95.2188 kWh`
- G3 절대 이동 p95: `258.0594 kWh`
- G3 최대 이동: `576.0715 kWh`
- G3 평균 signed 이동: `+16.4520 kWh`
- 상승/하락 행: `4,201 / 4,559`
- G3 출력 범위: `1,540.3169`–`20,004.1104 kWh`
- non-finite: `0`

## 4. 해석과 공개 결과

이 후보는 숫자상 기존 strict local gate를 모두 통과했다. 하지만
`plateau_mean` 규칙은 기존 `weight=0.05` 확인 보고서를 본 뒤 제안했다.
따라서 Q2/H2는 더 이상 완전히 독립적인 confirmation이라고 부를 수 없다.
보고서와 생산 JSON에 이 노출을 명시했다.

같은 데이터 계열의 JMA GSM G3가 강한 로컬 검증에도 공개점수에서 반전한
전례가 있었고, JMA MSM plateau도 같은 양상을 재현했다.

| submission | score | 1-NMAE | FiCR |
|---:|---:|---:|---:|
| incumbent `1502437` | `0.6461250914` | `0.8757842477` | `0.4164659352` |
| JMA MSM plateau `1504352` | `0.6455966798` | `0.8757843690` | `0.4154089906` |
| delta | `-0.0005284116` | `+0.0000001213` | `-0.0010569446` |

단순 전이 예상 `0.64684209`보다 실제 점수는 `-0.00124541` 낮았다. 로컬
G3 score `+0.00215100` 대비 공개 implied G3 score는 `-0.00158523`으로
전이 비율이 `-0.7370`이다. FiCR 전이 비율도 `-0.8431`이다.

결정:

1. 공개 incumbent `1502437`은 그대로 유지한다.
2. analog residual scenario와 scenario-consensus 계열은 닫는다.
3. JMA MSM plateau와 JMA 외부기상 G3 후처리 계열을 공개 실패로 닫는다.
4. 공개결과에 맞춘 weight·월·행 gate 재조정은 하지 않는다.
5. 후보는 재현용으로 보존하되 다시 제출하거나 다른 미확인 factor와 결합하지
   않는다.

## 5. 재현

```powershell
python -m pytest `
  tests/test_issue_residual_analog_scenarios.py `
  tests/test_kma_pooled_group_quantile_blend.py -q

python -m experiments.issue_residual_analog_scenarios `
  --positive-fractions 0.65,0.80 `
  --n-bootstrap 1000

python -m experiments.kma_pooled_group_quantile_blend `
  --targets kpx_group_1,kpx_group_2,kpx_group_3 `
  --context-2023 artifacts_final/external_weather/jma_msm_stencil_2023/features.csv `
  --manifest-2023 artifacts_final/external_weather/jma_msm_stencil_2023/manifest.json `
  --context-2024 artifacts_final/external_weather/jma_msm_stencil_2024/features.csv `
  --manifest-2024 artifacts_final/external_weather/jma_msm_stencil_2024/manifest.json `
  --context-2025 artifacts_final/external_weather/jma_msm_stencil_2025/features.csv `
  --manifest-2025 artifacts_final/external_weather/jma_msm_stencil_2025/manifest.json `
  --alpha 0.75 `
  --weights 0.025,0.028125,0.03125,0.034375,0.0375,0.040625,0.04375,0.046875,0.05 `
  --weight-selection-mode plateau_mean
```
