# v321–v323 공개 방법론 독립 재검증

## 목적과 경계

오늘 공개된 `Jungminii-1114/LG_AIMERS_personal_repo`의 Candidate4 설명에서
ExtraTrees와 Beta–Binomial current-season pooling이라는 기존 주력 계보와 다른 표현을
확인했다.

- 원문: https://github.com/Jungminii-1114/LG_AIMERS_personal_repo
- strict blend guide:
  https://github.com/Jungminii-1114/LG_AIMERS_personal_repo/blob/main/docs/JOA_CANDIDATE4_STRICT_BLEND_GUIDE.md

외부 OOF, 외부 모델, 외부 데이터는 사용하지 않았다. 공식 `train.csv`만으로 개념을 독립
구현하고 v290-equivalent 과거 축 위에서 재검증했다.

## v321 — Beta–Binomial current-season pooling

공식 career ASOF count/rate에서 직전 시즌의 정확한 terminal state를 빼 현재 시즌 투수·타자
표본수와 성공 수를 복원했다. 예측연도 Y의 pooling concentration과 convex weight는 Y-1에서만
적합했다. v290 결합 용량은 full-2022와 late-2023 maximin으로 선택했다.

- 선택: R 10%, F 0%
- full-2022: `+14.1446`, 7/7개월 양수
- late-2023: `+4.0290`, 2/3개월 양수
- full-2024 locked: **`-6.1170`**, 3/8개월 양수

source는 강했지만 잠금축에서 반전해 기각했다.

## v322 — 현재 시즌 성숙도 게이트

v321의 10% R dose를 고정하고, 현재 시즌 투구수 연속/하드 게이트와 월 성숙 게이트 9개만
사전 선언했다. full-2022와 late-2023 maximin은 `pitcher_n >= 300`을 선택했다.

- full-2022: `+10.5261`
- late-2023: `+5.6786`
- full-2024 locked: **`-3.8280`**

표본 성숙도는 손실을 줄였지만 방향을 복원하지 못해 기각했다. 잠금 결과에 맞춰 month gate나
threshold를 다시 선택하지 않는다.

## v323 — 300-tree ExtraTrees + Beta strict-forward

68개 공식/시간안전 파생피처로 2021·2022·2023·2024 네 ExtraTrees fold를 만들었다.
각 fold는 이전 시즌만 학습했고, Beta pooling은 Y-1, logit calibration은 직전 OOF만 사용했다.
공개 방법의 최종 설정에 맞춰 300 trees를 한 번만 검증했다.

가장 작은 2.5% 결합부터 source 부호가 갈렸다.

| route | full-2022 | late-2023 | source maximin 선택 |
|---|---:|---:|---:|
| R | +7.9578 | **-0.9206** | 0% |
| F | +2.8056 | **-22.7086** | 0% |

따라서 R/F 모두 0%가 선택됐고 후보는 no-op으로 종료됐다. seed·depth·tree 수·보정법을
추가 탐색하지 않는다.

## 결론

Beta 단독은 두 source에서 전이됐지만 2024에서 반전했고, 성숙도 게이트도 이를 구하지
못했다. ExtraTrees 결합은 locked를 열기 전 source 단계에서 실패했다. 이 공개 방법론 분기는
현재 v290 잔차의 추가 후보가 아니다. 광범위 독립 탐색 후에도 최종 제출 후보는 F 전용 v320
포트폴리오 하나다.

재현 산출물:

```text
artifacts/v321_strict_beta_binomial_complement_20260830_01
artifacts/v322_beta_maturity_gate_20260830_01
artifacts/v323_strict_extra_beta_complement_20260830_01
```
