# 평가 병목 재감사와 v244 fallback 확장 후보

## 2026-08-29 Public 결과

- 제출 ID: `73881`
- 제출 파일: `submit_v244.zip`
- 제출 시각: `2026-08-29 17:23:03 KST`
- Public 점수: `1175.9746833121`
- 공식 실행 시간: `110초`
- 목표 `1180`까지의 차이: `4.0253166879점`
- 사전 중심 예측 `1178.3227258660` 대비 오차: `-2.3480425539점`
- 사전 보수적 하한 `1176.6694243479` 대비 오차: `-0.6947410358점`

판정: v244는 목표 `1180`에는 실패했지만, 이후 확인된 기존 챔피언
`bridge027_active50_w030.zip`의 정확 점수 `1175.2737119427`보다
`+0.7009713694점` 높아 새 챔피언으로 승격한다. 기존 fallback 계열에서 추정한
로컬→Public 환산은 절대 개선폭을 크게 과대평가했으므로 폐기한다. 이 한 번의 Public
결과로 route나 가중치를 사후 재조정하지 않으며, 독립적인 검증 계약을 갖춘 새 모델
계열만 후속 후보로 다룬다.

작성 시각: 2026-08-29 KST

## 결론

공식 Brier Skill Score 구현과 행 정렬에는 오류가 없었다. 병목은 **다음 시즌에
배포되는 실제 feature transform과 기존 OOF transform이 달랐고**, 이미 반복해서 본
2024 축을 단일 locked holdout처럼 취급한 검증 설계에 있었다.

이 불일치를 교정한 strict-forward OOF를 새로 학습한 뒤, 현재 Public 1175 fallback의
배포 구간을 유지하면서 세 개의 서로 겹치지 않는 저복잡도 XGB 확장 경로를 만들었다.
source-only dose 선택값은 1.5이며 최종 유효 가중치는 다음과 같다.

- 기존 압박·JY `>=0.50`: XGB `0.30` 유지
- 압박·JY `[0.48, 0.50)`·두 모델 차이 `<=0.02`: XGB `0.45`
- 비압박·투수/타자 동손: XGB `0.15`
- 비압박·투수/타자 반대손·JY `>=0.52`: XGB `0.15`

재현 가능한 실행 ZIP을 만들고 245,789행 규모 감사를 통과했다. 그러나 2024 월별
안정성과 cluster bootstrap 하한은 통과하지 못했으므로, 이 결과는 `Public 1180 확정`
후보가 아니라 **현재까지 가장 강한 제한적 탐색 후보**로 분류한다.

## 1. 공식 평가 기준 감사

공식 점수는 다음과 같다.

```text
Score = max(0, 100000 * (1 - mean((p-y)^2) / (mean(y)*(1-mean(y)))))
```

로컬 `src.metrics`의 구현은 공식 식과 일치한다. 현재 점수대에서는 0 clipping이
작동하지 않으므로 incumbent와 candidate의 비교는 같은 행에서 squared-error 감소를
계산하는 것과 정확히 같다. Brier score가 확률 예측을 직접 평가하는 strictly proper
scoring rule이라는 점도 이 선택과 일치한다.

근거:

- DACON 공식 평가: https://www.dacon.io/competitions/official/236743/overview/evaluation
- Gneiting & Raftery (2007): https://doi.org/10.1198/016214506000001437

따라서 이번 병목은 log-loss/BSS 혼동이나 공식 수식 구현 오류가 아니다.

## 2. 발견한 검증 계약 문제

### 2.1 부모 OOF lineage 불일치

기존 Public1175 evidence는 v84/v148 부모 배열에 fallback XGB를 적용한 값이다. 최신
JY 공식을 역사적으로 재구축하면 fallback active 행 수가 일치하지 않는다.

| 축 | 공개 evidence active | exact-JY 재구축 active |
|---|---:|---:|
| full-2022 | 68,486 | 65,879 |
| late-2023 | 11,892 | 12,266 |
| full-2024 | 30,798 | 30,213 |

따라서 기존 evidence gain은 실제 Public1175 전체 수식의 exact OOF가 아니며, 방향
확인용 proxy로만 사용해야 한다.

### 2.2 OOF와 제출 runtime feature 의미 불일치

기존 OOF builder는 검증 연도 행을 training feature path로 변환했다. 실제 제출
runtime은 과거 공식 train에서 고정한 lookup과 zero-anchor multiscale branch로 미래
시즌을 변환한다. v242는 매 audit year마다 다음 절차를 새로 수행했다.

1. `season < audit_year`만으로 categorical map, anchor, 상황별 투수 prior,
   투수-타자 prior, 역할 및 TrackMan prior를 생성한다.
2. 과거 행은 training transform으로 학습한다.
3. audit 행은 실제 frozen runtime transform으로 변환한다.
4. XGB 1,800개 트리를 같은 파라미터와 시간 감쇠 가중치로 다시 학습한다.

그 결과 XGB 단독 OOF BSS가 크게 낮아졌다.

| audit year | 기존 v217 | runtime-faithful v242 | 차이 |
|---:|---:|---:|---:|
| 2022 | 695.717 | 482.231 | -213.486 |
| 2023 | 725.079 | 426.668 | -298.411 |
| 2024 | 836.247 | 600.939 | -235.308 |

이는 기존 후속 연구가 XGB 자체의 배포 성능을 낙관적으로 본 직접 증거다.

### 2.3 시간 전이와 반복 선택

late-2023은 일부 기존 부모가 같은 시즌 정보를 사용했고, full-2024는 여러 후보군에서
반복적으로 관찰됐다. 같은 행 안에서 하는 bootstrap은 투수/타자 군집 불확실성은
측정하지만 **2024에서 2025로 넘어가는 regime transfer**는 측정하지 못한다.

비정상성을 허용하는 시계열에서는 일반 K-fold보다 시간 순서를 보존하는 sequential
validation이 일반화 오차 추정에 적합하다는 문헌과도 일치한다.

- Bengio & Chapados (2003): https://www.jmlr.org/papers/volume3/bengio03b/bengio03b.pdf
- Bergmeir & Benitez (2012): https://doi.org/10.1016/j.ins.2011.12.028

향후 gate는 후보 메커니즘에 따라 분리한다. 학습형 gate는 엄격한 source/holdout
기준을 유지하되, frozen 신호의 저자유도 blend는 두 transform 계약과 다음 시즌
재학습 축을 필수로 본다. 어느 경우에도 2024 한 축이 source reject를 구제하지 못한다.

## 3. 야구 도메인 가설

현재 fallback의 Public 성공은 XGB를 모든 정규시즌 행에 적용해서가 아니라 주자 또는
고 LI의 압박 구간과 JY 고확률 행에 제한한 데서 나왔다. 새 후보는 이 구조를 보호한다.

비압박에서는 주자가 없고 `LI < 1.5`이므로 상황 변화가 작다. 이 구간에서 투수/타자
handedness를 분리한 것은 전형적인 platoon interaction을 모델 오차 구조에 반영한다.
이는 “동손이면 제구 성공률이 반드시 높다”는 인과 주장이 아니라, 서로 다른 모델의
잔차가 handedness matchup에 따라 달랐다는 경험적 routing이다. 야구의 handedness
효과가 단순 타격 손 하나가 아니라 matchup과 투수 질의 결합이라는 연구와 방향이
맞는다.

- Pawlowski et al. 계열의 handedness 연구: https://pubmed.ncbi.nlm.nih.gov/18298283/
- 최근 matchup 연구: https://doi.org/10.1080/24748668.2023.2255806

## 4. v244 결과

route는 2024를 본 뒤 발견됐으므로 깨끗한 locked-holdout claim을 하지 않는다. 공통
dose만 runtime-faithful full-2022와 late-2023에서 선택했다. 모든 부모/source gain이
양수인 dose 중 exact-JY 최악 source gain을 최대화한 값은 `1.5`였다.

### 4.1 증분 BSS

아래는 모두 현재 Public1175 fallback 대비 증분이다.

| OOF 계약 | 부모 | full-2022 | late-2023 | full-2024 |
|---|---|---:|---:|---:|
| legacy training-transform | evidence proxy | +19.272 | +6.830 | +0.895 |
| legacy training-transform | exact JY | +13.879 | +5.894 | +1.682 |
| runtime-faithful | evidence proxy | +10.401 | +5.288 | +2.305 |
| runtime-faithful | exact JY | +4.879 | +2.284 | +2.840 |

두 transform 계약과 두 부모 계열에서 세 연도 전체 gain이 양수다. 특히 runtime-faithful
exact-JY에서 현재 압박 fallback 자체는 `+3.779/-0.762/-0.371`이었지만, v244 확장을
포함한 최종 조합은 JY 대비 대략 `+8.657/+1.521/+2.469`이 된다.

### 4.2 남은 위험

- full-2024 positive month fraction: legacy exact `5/8`, runtime exact `4/8`
- legacy exact active-route bootstrap p05:
  pitcher `-3.941`, crossed `-9.029`, chronological `-3.690`
- runtime exact active-route bootstrap p05:
  pitcher `-3.283`, crossed `-11.736`, chronological `-2.797`
- supplied dose family Reality Check:
  legacy `p=0.213`, runtime `p=0.156`

평균 gain은 강하지만 독립적인 confirmatory season이 없고 월별 편차가 크다. 이것이
후보를 자동 승격하지 않은 이유다.

### 4.3 공격형 압박 weight 0.55 기각

현재 Public 점수가 정확히 1175.0~1176.0이라면 Brier의 2차식 성질상 압박 fallback
weight의 추정 정점은 약 0.525~0.656이다. 그러나 0.55를 v244와 결합하자
runtime-faithful exact-JY가 `-2.175/-3.360/-1.822`로 세 축 모두 음수가 됐다.
Public 한 점에 맞춘 재튜닝 위험이 너무 커 기각했고, 최종 ZIP은 기존 `0.30`을 유지한다.

## 5. Public 점수 예측

정확한 현 점수가 제공되지 않아 `1175.0 <= incumbent < 1176.0`만 사용했다. 기존
fallback의 공개 evidence full-2024 gain `+1.5353`과 실제 Public 개선폭
`+2.8626~+3.8626`을 같은-family transfer calibration으로 쓰면 배율은
`1.865~2.516`이다.

v244의 legacy full-2024 gain `+0.895~+1.682`를 결합한 휴리스틱 예측은 다음과 같다.

| 구분 | 예상 Public |
|---|---:|
| 보수 | 1176.67 |
| 중앙 | **1178.32** |
| 낙관 | 1180.23 |

이 범위는 통계적 신뢰구간이 아니다. 현재 증거상 1180 초과는 가능한 상단 시나리오지만
가장 가능성 높은 단일 예측은 약 **1178.3**이다. 정확한 incumbent 소수점과 실제 한 번의
제출 결과가 들어오면 같은-family 곡선을 즉시 갱신할 수 있다.

## 6. 실행 후보와 규칙 준수

- ZIP: `JY_fallback_XGB_mechanism_w045_w015/rebuilt/submit_jy_xgb_mechanism_w045_w015.zip`
- SHA-256: `E745AD6632E880771BE075672743DBA5F22814DA5355E476B201FB08E0B60162`
- 압축 크기: 85,664,464 bytes
- 파일 수: 177
- CRC: pass
- static forbidden operation: 0
- singleton/shuffle/partition 최대 오차: 0
- 245,789행 runtime: 190.56초 / 공식 제한 600초
- 확률 finite 및 `[0,1]`: pass

공식 규칙은 각 test 행의 예측이 그 행, 그 행의 파생변수, 공식 train에서 만든
모델/통계만 사용해야 한다고 명시한다. 새 ZIP은 다른 test 행의 평균·빈도·rolling·선수
집계를 사용하지 않으며 동적 독립성 감사도 통과했다.

- DACON 독립 예측 공지: https://dacon.io/competitions/official/236743/talkboard/417123
- 공식 데이터 규모: https://dacon.io/competitions/official/236743/data

## 7. 의사결정

현재 원본 Public1175 release와 재현 builder는 수정하지 않고 보존했다. v244의 Public
점수는 `1175.9746833121`로 목표에 `4.0253166879점` 미달했고 사전 중심 예측보다
`2.3480425539점` 낮았지만 기존 챔피언 대비 `+0.7009713694점`의 실제 개선을 확인했다.
v244를 새 기준선으로 승격한다. 동일 fallback family의 다음 dose 탐색은 중단하며,
이 Public 결과에 맞춘 route 또는 가중치 사후 튜닝은 하지 않는다.
