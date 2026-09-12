# 공개 요인 분해와 JMA GSM 후속 실험 — 2026-07-26

## 결론

현재 공개 최고점은 제출 `1501927`의 `0.6458227359`다. 최근 네 제출을
그룹 하나만 다른 대조쌍으로 재구성한 결과, 아직 제출하지 않은 다음 조합의
공개 점수를 **`0.6461250914`**로 복원했다.

`artifacts_final/candidates/public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv`

- 구성: residual G1 + pooled G2 + incumbent KMA G3
- 행 수: 8,760
- SHA-256:
  `a074f2a1901f9d010922cbc64b945453b67354c858b94e7cc0bafba2870cad2a`
- 기대 공개 지표:
  - score: `0.6461250914`
  - 1-NMAE: `0.8757842476`
  - FICR: `0.4164659352`
- 기존 공개 최고점 대비: `+0.0003023555`
- `0.65`까지 남은 차이: `0.0038749086`

이 값은 공개 평가 표본과 결정론적 그룹 macro 평가가 그대로 유지된다는
조건에서 대수적으로 정확하다. 비공개 평가 표본에서도 같은 이득이 난다는
뜻은 아니다.

독립적인 JMA GSM 신호는 그룹 3에서 엄격 검증을 통과했다. 위 확정 조합의
그룹 3만 GSM pooled 전문가로 바꾼 2순위 후보도 만들었다.

`artifacts_final/candidates/public_factor_g1g2_kma_jma_gsm_g3_strict_20260726.csv`

- SHA-256:
  `30a1229ecacd083a06bf1ca26593f39c5c473565b4090944728574c500d31ed5`
- 1순위 후보와 다른 열: `kpx_group_3` 하나뿐
- 2024 그룹 3 score 개선: `+0.0032619155`
- macro 환산 개선: `+0.0010873052`
- 공개 확정 조합에 이 local 효과가 전이될 경우의 보수적 추정:
  `0.6472123966`

기존 보고서에 기록되는 `0.6499418189`는 세 그룹의 2024 local 개선을
모두 과거 공개 기준점에 더한 단순 투영이다. 그룹 1·2의 공개 전이가 local
추정보다 훨씬 작았으므로 제출 의사결정에는 이 낙관값을 사용하지 않는다.

## 공개 점수 요인 분해

평가 점수는 세 그룹 점수의 macro 평균이다. 다른 두 그룹의 예측 벡터가
완전히 같고 한 그룹만 다른 두 제출의 점수 차이는 그 그룹 교체 효과와
정확히 같다.

| 제출 | G1 | G2 | G3 | score | 1-NMAE | FICR |
|---|---|---|---|---:|---:|---:|
| 1501926 | pooled | incumbent | UMKR | 0.6450069210 | 0.8754603805 | 0.4145534615 |
| 1501927 | pooled | pooled | incumbent | **0.6458227359** | **0.8758052670** | 0.4158402047 |
| 1501928 | residual | pooled | UMKR | 0.6457889498 | 0.8756700069 | **0.4159078927** |
| 1501938 | pooled | pooled | UMKR | 0.6454865943 | 0.8756910263 | 0.4152821622 |

따라서 공개 표본에서 식별되는 효과는 다음과 같다.

| 요인 | score | 1-NMAE | FICR |
|---|---:|---:|---:|
| G1 residual | +0.0003023555 | -0.0000210194 | +0.0006257305 |
| G2 pooled | +0.0004796733 | +0.0002306458 | +0.0007287007 |
| G3 UMKR | -0.0003361416 | -0.0001142407 | -0.0005580425 |

따라서 공개 최고 조합인 pooled G1 + pooled G2 + incumbent G3에서 G1만
residual로 바꾸면 `0.6458227359 + 0.0003023555 = 0.6461250914`다.
구현은 `experiments/compose_public_factor_candidate.py`에 있으며, 대조쌍이
지정 그룹 하나만 다르고 기준 후보가 control과 같은지를 fail-closed로
검사한다.

## 병목 진단

공식 metric 구현 자체는 병목이 아니다. 독립 구현과 100개 무작위·경계
사례를 비교해 동일한 결과를 확인했다. exact OOF도 8,779행, 중복 0,
그룹 정렬 정상, Q2/H2 시점 중첩 0이었다.

실제 병목은 다음 두 가지다.

1. 2024 local 개선의 2025 공개 전이가 크게 축소된다.
2. H2를 포함한 2024 데이터가 프로젝트 전반에서 반복 조회되어 더 이상
   진정한 미사용 holdout이 아니다.

예를 들어 pooled G1의 2024 그룹 score 개선은 약 `+0.0111`이지만 공개
효과는 그보다 훨씬 작다. UMKR G3는 local에서 `+0.003338`이었으나 공개에서
`-0.000336`으로 반전했다. 그러므로 2024 점수만 합산한 `0.65` 근접치는
후보 순위의 참고치일 뿐 보장값으로 취급하면 안 된다.

## 독립 기상 신호 실험

JMA MSM/GSM 과거 운용예보는 예측 대상 시점 이전에 이용 가능했던
고정-offset 자료만 사용했다. 원본 JSON, 요청 URL, checksum, 파생
features, manifest를 연도별로 보존했고 모든 채택 archive는 timing
violation 0건이다.

GSM은 공식 운용시각에 360분의 보수적 공개 지연을 더해 판정했다.
최소 가용 여유가 정확히 0분인 경계 행이 있으므로 규정 위반은 아니지만,
이 불확실성도 GSM 후보를 2순위로 두는 이유다.

| 실험 | 주요 결과 | 판정 |
|---|---|---|
| KMA + JMA GSM 기본 | G3 `+0.0032619`, Q1/Q2/H2 양수, 10/12개월 양수, bootstrap q05 `+0.0011446` | G3 strict 채택 |
| KMA + JMA GSM day3 수정량 | G3 `+0.0028061`; 기본 GSM보다 약함 | strict이지만 기각 |
| KMA + JMA MSM 열역학 | G1 `+0.0113840`이나 9/12개월, seed component 음수 | 기각 |
| KMA + MSM + GSM | G3 `+0.0029549`; 기본 GSM보다 약함 | 기각 |
| JMA GSM 단독 | 승격 그룹 없음 | 기각 |
| KMA+MSM 두 그룹 조합 | all-source G1보다 약하거나 승격 실패 | 기각 |

GSM day3 자료는 2023년 8,760행, 2024년 8,784행, 2025 제출 범위
8,760행을 모두 확보했다. day2−day3 예보 수정량은 인과적이지만 기본 GSM
feature set의 일반화를 개선하지 못했으므로 production 후보에는 넣지
않는다.

## 제출 순서

2026-07-26에는 이미 일일 5회 제출을 사용했다. 외부 제출은 자동으로
수행하지 않았다. 다음 제출 가능 시점의 순서는 아래처럼 고정한다.

1. `public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv`
2. 1번이 예상값과 일치할 때만
   `public_factor_g1g2_kma_jma_gsm_g3_strict_20260726.csv`
3. 1번이 예상값과 의미 있게 다르면 추가 제출을 중단하고, DACON scorer의
   표본·반올림·파일 정렬 가정을 먼저 재감사한다.

2번은 1번과 그룹 3만 달라서 JMA GSM의 독립 공개 효과를 깨끗하게
측정한다. 결과를 본 뒤 월 mask, blend weight, 계절 gate를 다시 고르는
행위는 금지한다.

## 재현성과 검증

핵심 산출물:

- 정확 요인 후보 보고서:
  `artifacts_final/diagnostics/public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.json`
- GSM validation:
  `artifacts_final/diagnostics/kma_jma_gsm_pooled_all3_validation_20260726.json`
- GSM production:
  `artifacts_final/diagnostics/kma_jma_gsm_pooled_all3_production_20260726.json`
- day3 기각 보고서:
  `artifacts_final/diagnostics/kma_jma_gsm_day3_pooled_all3_validation_20260726.json`
- 최종 조합 보고서:
  `artifacts_final/diagnostics/public_factor_g1g2_kma_jma_gsm_g3_strict_20260726.json`

최종 후보 두 파일은 모두 8,760행, 공식 5열 계약, ID·시각 정렬, 수치
유한성 검사를 통과했다. 공개 요인 조합기는 public score 사용 사실과
private transfer 비보장 상태를 manifest에 명시한다.

## 공개 결과 업데이트 — 2026-07-27

| 제출 | 구성 | score | 1-NMAE | FICR |
|---|---|---:|---:|---:|
| 1502437 | residual G1 + pooled G2 + incumbent G3 | **0.6461250914** | 0.8757842477 | **0.4164659352** |
| 1502447 | residual G1 + pooled G2 + JMA GSM G3 | 0.6449277253 | 0.8754187560 | 0.4144366947 |

1502437은 예측값 `0.6461250914`와 표시 정밀도까지 정확히 일치했다.
반면 G3만 바꾼 1502447은 score `-0.0011973661`, 1-NMAE
`-0.0003654917`, FICR `-0.0020292405`로 세 지표가 모두 하락했다.
JMA GSM G3 production 계열은 폐기하고 1502437을 새 기준점으로 동결한다.
