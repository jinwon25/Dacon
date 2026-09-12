# 공개 factor 병목 감사와 G1–G2 차등 reconciliation

작성 시각: 2026-07-28 KST

## 결론

제출 `1504352`의 실패를 포함해 식별 가능한 공개 대비를 다시 분해한 결과,
현재 공개 양성으로 확인된 축은 `residual G1`과 `pooled G2` 두 개뿐이다.
최근 G3 교체와 공격적 FiCR 보정은 모두 FiCR 음성이다. 따라서 G3를 동결하고,
기존 두 양성 축의 단순 확대와 다른 G1–G2 차등 성분을 새로 만들었다.

검증 당시 생성한 후보:

`artifacts_final/candidates/group12_difference_reconciliation_unanimous_w10_20260728.csv`

- SHA-256:
  `fab745cbed1a1d60c3d15af15b9590a399fd62f8bf63ec16a6dfa3dd9c4a047a`
- 당시 등급: `controlled_exploratory`
- strict 승격: 실패
- 공개 제출: `1504383`
- G3: incumbent와 완전히 동일
- 각 행의 G1+G2 합: incumbent와 동일

제출 결과 score는 `0.6456906139`로 incumbent보다 낮았다. 이 후보와 그 위의
희소 row gate 계열은 모두 닫는다.

## 1. 제출 `1504352`가 확인한 병목

| submission | score | 1-NMAE | FiCR |
|---:|---:|---:|---:|
| incumbent `1502437` | `0.6461250914` | `0.8757842477` | `0.4164659352` |
| JMA MSM plateau `1504352` | `0.6455966798` | `0.8757843690` | `0.4154089906` |
| macro delta | `-0.0005284116` | `+0.0000001213` | `-0.0010569446` |
| implied G3 delta | `-0.0015852348` | `+0.0000003639` | `-0.0031708338` |

평균오차는 표시 정밀도에서 사실상 움직이지 않았고 FiCR만 크게 하락했다.
이는 G3 외부기상 후처리의 병목이 평균점 예측이 아니라 공개 구간의 6%·8%
정산 경계 전이라는 뜻이다. 공개 결과를 보고 JMA 가중치나 행 gate를 다시 맞추는
것은 중단한다.

## 2. 공개 factor-response 감사

구현:

`experiments/public_factor_response_audit.py`

산출물:

`artifacts_final/diagnostics/public_factor_response_audit_20260728.json`

공개 점수는 세 그룹 점수의 평균이므로 한 그룹만 바뀐 대비의 그룹 내 효과는
macro delta의 세 배다.

| factor | 변경 그룹 | macro score | macro 1-NMAE | macro FiCR | 결정 |
|---|---|---:|---:|---:|---|
| pooled G2 | G2 | `+0.0004796733` | `+0.0002306458` | `+0.0007287007` | 유지 |
| residual G1 | G1 | `+0.0003023555` | `-0.0000210193` | `+0.0006257305` | 유지 |
| UMKR G3 | G3 | `-0.0003361416` | `-0.0001142407` | `-0.0005580425` | 폐기 |
| JMA GSM G3 | G3 | `-0.0011973661` | `-0.0003654917` | `-0.0020292405` | 폐기 |
| directional G1 | G1 | `-0.0020020319` | `-0.0019214720` | `-0.0020825918` | 폐기 |
| directional G2/G3 | G2,G3 | `-0.0010189141` | `-0.0005761858` | `-0.0014616426` | 폐기 |
| trajectory TCN G3 | G3 | `-0.0006674758` | `+0.0001600758` | `-0.0014950275` | 폐기 |
| JMA MSM plateau G3 | G3 | `-0.0005284116` | `+0.0000001213` | `-0.0010569446` | 폐기 |

factorial bridge의 최대 폐합 오차는 `2e-10` 이하이다. 이 표는 공개 결과를
진단에만 사용하며 신규 후보의 행별 gate나 weight를 공개 점수에서 선택하지 않는다.

## 3. 신규 돌파구: G1–G2 차등 성분

2024 활성 OOF에서 용량 정규화 출력을 비교했다.

- 실제 G1/G2 상관: `0.9615586`
- 활성 예측 G1/G2 상관: `0.9886263`
- 활성 G1–G2 정규화 차이 MAE: `0.0569616`
- 공통 오차와 차이 오차의 상관: 약 `-0.0165`

현재 예측은 두 그룹을 실제보다 지나치게 같이 움직이게 한다. 공통 오차와 차등
오차도 사실상 직교하므로, G1/G2 각각의 수준을 다시 예측하는 대신 둘의 차이만
보정하는 축을 만들었다.

다중 풍력단지의 공간 의존성을 사용하면 개별 및 집계 예측을 개선할 수 있다는
근거는 Lenzi, Steinsland, Pinson의
[spatio-temporal wind-power 연구](https://arxiv.org/abs/1704.07606)와
일치한다. Forecast reconciliation은 서로 관련된 시계열의 base forecast를
제약 공간으로 투영하는 관점으로 정리되어 있으며,
[2024년 review](https://doi.org/10.1016/j.ijforecast.2023.10.010),
[geometric interpretation](https://doi.org/10.1016/j.ijforecast.2020.06.004),
[2026년 partial reconciliation note](https://doi.org/10.1016/j.ijforecast.2026.04.001)를
방법론 참고점으로 사용했다.

단, 이 대회에는 별도 상위 합계 예보가 없다. 따라서 여기서는 이론적 MinT 최적성을
주장하지 않고, 이미 검증된 incumbent의 G1+G2 합을 보존하는 partial projection으로
제한한다.

### 모델

구현:

`experiments/group12_difference_reconciliation.py`

1. 제공 LDAPS/GFS에서 G1·G2 발전단지 IDW feature의 평균과 차이를 만든다.
2. 2023년만 학습한 세 LightGBM L1 모델이
   `(G1 / 21600) - (G2 / 21600)`을 예측한다.
3. 세 seed가 incumbent 차이에 대한 보정 방향에 모두 동의하는 행만 사용한다.
4. Q1에서만 작은 weight를 선택한다.
5. G1에는 `+adjustment`, G2에는 `-adjustment`를 적용한다.
6. 각 행의 G1+G2 합과 G3는 고정한다.

weight grid를 `0.0125`부터 `0.20`까지 확장해도 Q1 최적은 내부점 `0.10`으로
유지됐다. Q1 세 달을 모두 양성으로 만드는 보수형 `0.0125`도 별도로 확인했지만,
H2 score `-0.00008223`, bootstrap q05 `-0.00011474`로 탈락했다. 작은 보정이
항상 안전한 것이 아니라 FiCR 경계를 실제로 넘을 충분한 강도가 필요했다.

## 4. 순방향 검증

최종 정책은 `weight=0.10`, `unanimous seed gate`다. gate coverage는
`93.22%`이다.

아래 값은 G1/G2 두 그룹 평균의 incumbent 대비 delta다.

| 구간 | score | 1-NMAE | FiCR |
|---|---:|---:|---:|
| Q1 selection | `+0.00091700` | `+0.00023836` | `+0.00159564` |
| Q2 confirmation | `+0.00060860` | `+0.00010450` | `+0.00111270` |
| H2 confirmation | `+0.00051870` | `+0.00012094` | `+0.00091646` |
| full 2024 | `+0.00065385` | `+0.00014709` | `+0.00116062` |

G3가 고정된 전체 대회 macro로 환산한 2024 로컬 delta는 다음과 같다.

- score: `+0.00043590`
- 1-NMAE: `+0.00009806`
- FiCR: `+0.00077375`

세 seed 모두 Q1/Q2/H2/full score가 양성이고, 10,000회 complete
issue-cycle bootstrap 결과는 다음과 같다.

- 양성 비율: `95.46%`
- q05: `+0.00001783`
- median: `+0.00065208`
- q95: `+0.00132167`
- 이슈 시각 결측으로 bootstrap에서만 제외된 행: `1`

기존 공개 양성 G1/G2 factor movement와 신규 movement의 cosine은
`0.18679`다. 신규 축이 기존 factor의 단순 확대가 아니라는 정량 증거다.

## 5. 남은 위험

- 월별 score 양성은 `7/12`이다.
- 3월 `-0.00135090`, 11월 `-0.00066417` 등 FiCR 경계 손실이 남는다.
- strict 기준인 9/12를 통과하지 못했다.
- 여러 방법을 같은 2024 benchmark에서 반복 개발했으므로 Q2/H2를 완전히
  잠긴 holdout으로 해석할 수 없다.
- 공개 양성 factor와 직교하는 신규 구조이므로 기존 공개 transfer ratio를
  외삽할 근거가 없다.
- public subset에서 양성이어도 private 60%에서 반전할 수 있다.

이 때문에 후보를 `strict`로 승격하지 않고 `controlled_exploratory`로만 보존한다.
회귀 위험을 확률적으로 제한하는 후속 연구에는 covariate shift를 포함한
[distribution-free regression risk assessment](https://proceedings.mlr.press/v230/singh24a.html)
같은 접근을 적용할 수 있으나, 현재 후보에는 conformal 보장을 주장하지 않는다.

## 6. 생산 후보 감사

- 행 수: `8,760`
- validator: 통과
- 변경 행: `8,011`
- G1 평균 절대 이동: `27.1087 kWh`, 용량의 `0.1255%`
- G1 p95 절대 이동: `77.9168 kWh`
- G1 최대 절대 이동: `191.4196 kWh`, 용량의 `0.8862%`
- G1 상승/하락/동일: `5,001 / 3,010 / 749`
- G2 이동: G1과 정확히 반대
- G1+G2 합 최대 차이: 부동소수점 반올림 수준
- G3: incumbent와 완전히 동일

재현 명령:

```powershell
python -m pytest `
  tests/test_public_factor_response_audit.py `
  tests/test_group12_difference_reconciliation.py -q

python -m experiments.public_factor_response_audit

python -m experiments.group12_difference_reconciliation `
  --uncertainty-gate unanimous `
  --n-bootstrap 10000 `
  --write-candidate `
  --report artifacts_final/diagnostics/group12_difference_reconciliation_unanimous_w10_20260728.json
```

## 7. 결정

1. 공개 incumbent는 계속 `1502437`이다.
2. G3 외부기상·trajectory·공격적 FiCR 후처리는 동결한다.
3. G2 weight-0.10 단순 확대는 bootstrap 실패 때문에 후순위로 유지한다.
4. G1–G2 broad reconciliation은 공개 실패로 닫는다.
5. 같은 action의 weight·row gate를 public 결과에 맞춰 재조정하지 않는다.
6. 다음 후보는 새로운 core model이어야 하며 rejected action의 후처리가 아니어야 한다.

## 8. 공개 제출 `1504383`

| submission | score | 1-NMAE | FiCR |
|---:|---:|---:|---:|
| incumbent `1502437` | `0.6461250914` | `0.8757842477` | `0.4164659352` |
| G1/G2 reconciliation `1504383` | `0.6456906139` | `0.8758649682` | `0.4155162596` |
| macro delta | `-0.0004344775` | `+0.0000807205` | `-0.0009496756` |
| implied affected-pair delta | `-0.0006517163` | `+0.0001210808` | `-0.0014245134` |

로컬 G1/G2 평균 delta와 공개 affected-pair delta의 전이율:

- score: `-0.99673`
- 1-NMAE: `+0.82319`
- FiCR: `-1.22737`

score는 거의 같은 크기로 부호가 반전됐지만 평균오차 개선은 실제로 전이됐다.
실패 원인은 이동 방향 전체가 무의미해서가 아니라 public subset의 실제 발전량
가중 6%·8% 정산 경계를 불리하게 넘긴 것이다. 같은 현상은 JMA GSM G3,
trajectory TCN G3, JMA MSM plateau G3에서도 반복됐다. 그러므로 반복 inspection한
2024 FiCR 양성만으로 postprocessor를 승격하지 않는다.

구조화 감사:

`artifacts_final/diagnostics/group12_reconciliation_public_result_20260728.json`

## 9. one-sided selective gain gate

공개 결과는 방법론 계열의 우선순위에만 사용했다. alpha, margin, 행 gate에는
공개 점수를 사용하지 않았다.

고정 broad action의 각 행에 대해 다음 gain을 정의했다.

`(|truth - incumbent| - |truth - proposal|) / capacity`

gain이 실제 양수라면 절대오차가 감소하므로 해당 행의 FiCR unit price도 나빠질
수 없다. January–February로 conditional lower-gain quantile 모델을 학습하고,
March에서만 alpha와 margin을 선택하도록 했다. 이는 reject option을 통해
coverage와 risk를 교환하는
[selective regression](https://proceedings.mlr.press/v162/shah22a.html)과
[conformal regression with reject option](https://proceedings.mlr.press/v230/johansson24a.html)을
참고한 보수적 screen이지만, 본 구현은 conformal 보장을 주장하지 않는다.

결과:

- alpha `0.10`, `0.20`: lower gain이 양수인 행 `0`
- alpha `0.30`, margin `0`:
  - G1 선택 eligible 행 `15`, 실제 positive-gain precision `26.67%`
  - G2 선택 eligible 행 `11`, 실제 positive-gain precision `18.18%`
  - March pair score `-0.00000233`
  - March 1-NMAE `-0.00000466`
- 모든 alpha/margin 정책이 pair·개별 그룹·coverage·precision gate 탈락
- Q2/H2 미개방
- 후보 CSV 미생성

구현과 감사:

- `experiments/group12_selective_gain_gate.py`
- `artifacts_final/diagnostics/group12_selective_gain_gate_20260728.json`

결론적으로 broad action을 희소하게 거르는 방식도 근거가 없다. 이 계열은 완전히
닫고 신규 core model 또는 아직 공개 양성이 확인된 factor만 다룬다.

## 10. 다음 core-model 검증 기반

기존 feature cache와 실제 NWP issue 시각으로 2023 nested OOF 가능성을 다시
감사했다. G1/G2는 2022 라벨이 존재하므로 다음 네 개의 완전한 outer issue-season
fold를 만들 수 있다.

| outer season | G1 train / valid | G2 train / valid |
|---|---:|---:|
| 2023-DJF | `7,897 / 2,160` | `7,897 / 2,160` |
| 2023-MAM | `10,057 / 2,208` | `10,057 / 2,208` |
| 2023-JJA | `12,265 / 2,205` | `12,265 / 2,206` |
| 2023-SON | `14,470 / 2,184` | `14,471 / 2,184` |

각 fold는 직전 complete issue-season을 inner selection으로 사용하고 24시간
purge를 유지할 수 있다. 반면 G3는 2022 라벨이 0건이라 동일한 2023 nested
OOF가 불가능하다.

따라서 다음 연구 계약은 다음과 같이 제한한다.

1. G1/G2 전용 신규 core model만 허용한다.
2. 2023 네 계절에서 구조·손실·regularization을 선택한다.
3. 선택 후 2024 전체를 한 번만 순방향 확인한다.
4. 두 연도에서 score와 FiCR 방향이 모두 일치해야 한다.
5. incumbent 위의 row gate나 rejected action 재사용은 금지한다.
6. 이 조건을 통과하기 전에는 생산 CSV를 만들지 않는다.
