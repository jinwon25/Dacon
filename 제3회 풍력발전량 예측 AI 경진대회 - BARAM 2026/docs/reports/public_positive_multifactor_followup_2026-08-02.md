# 공개 양수 다중 요인 확장 후속 실험 — 2026-08-02

## 결론

다음 1회 제출 후보를 새로 확정했다.

`artifacts_final/candidates/public_positive_g1w1375_g2w095_g3frozen_20260802.csv`

- 행: `8,760`
- SHA-256: `0eb660b963a91e7355680d6c8b208648a41a5625b7005286fb4cb4b6d374cfa3`
- `CandidateValidator`: 통과
- G1 residual weight: `0.125 -> 0.1375`
- G2 pooled-weather weight: `0.05 -> 0.095`
- G3: 공개 최고 후보와 완전히 동일하게 동결
- 보수적 공개 전달 예상 score: `0.6464122451`
- 현 공개 최고 `0.6461250914` 대비 예상 증가: `+0.0002871537`

이 파일은 **제출 가능한 controlled public probe**다. 다만 재표본 FiCR 하단은
음수이므로 독립 private 성능이 보장되는 robust promotion으로 부르지는 않는다.

## 왜 이 경로를 택했는가

모기 궤적 대회에서 효과가 있었던 여러 방법론 블렌딩 원칙을 그대로 적용하되,
이번 대회에서는 공개 점수로 방향이 확인된 축만 남겼다.

공개 factorial 제출에서 분리된 효과는 다음과 같다.

| 요인 | score | 1-NMAE | FiCR |
|---|---:|---:|---:|
| G1 residual | +0.0003023555 | -0.0000210193 | +0.0006257305 |
| G2 pooled weather | +0.0004796733 | +0.0002306458 | +0.0007287007 |

반면 G3 교체 계열은 JMA GSM, JMA MSM, expanding TCN, KMA 다년도 UMRG까지
연속해서 공개 점수가 하락했다. 따라서 이번 후보에서는 G3를 더 이상 움직이지 않았다.

외부 우수 사례도 NWP 앙상블·서로 다른 예측기 결합을 반복해서 사용한다. 최근
tree 기반 풍력 예측 연구는 weather ensemble이 point forecast에도 이득을 줄 수 있음을
보였고, HEFTCom2024 사례는 sister NWP forecast stacking과 온라인 후처리를 사용했다.
KDD Cup 풍력 예측 우수 사례들도 GBDT와 recurrent model을 결합했다. 이번 실험은 그
아이디어를 무제한 모델 혼합이 아니라, 공개 양수로 확인된 두 축의 작은 확장으로 제한했다.

- Probabilistic Wind Power Forecasting with Tree-Based Machine Learning and Weather Ensembles:
  <https://arxiv.org/abs/2602.13010>
- HEFTCom2024 winning solution:
  <https://arxiv.org/abs/2505.10367>
- KDD Cup 2022 88VIP solution:
  <https://arxiv.org/abs/2208.08952>
- KDD Cup 2022 trymore solution:
  <https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_1286.pdf>

## 탐색과 선택

공개 최고 OOF를 anchor로 두고 다음 작은 격자를 비교했다.

- G1 residual: `0.125`부터 `0.1875`
- G2 pooled: `0.05`부터 `0.12`
- G3: 고정

전체 기간 score가 가장 높은 안정 구간은 G1 `0.1375`, G2 `0.095`였다. 더 큰
G2 `0.10`은 H2 FiCR가 거의 0까지 내려가고, `0.11`부터 H2 score와 FiCR가 모두
음수로 돌아섰다. 따라서 최대 확장 대신 `0.095`에서 멈췄다.

## 2024 OOF 결과

모든 값은 현 공개 최고의 정확한 OOF anchor 대비 macro delta다.

| 구간 | score | 1-NMAE | FiCR |
|---|---:|---:|---:|
| Q1 | +0.0003393524 | -0.0000714072 | +0.0007501119 |
| Q2 | +0.0008445407 | +0.0000411893 | +0.0016478922 |
| H2 | +0.0002038160 | +0.0001548168 | +0.0002528152 |
| 전체 | **+0.0003800051** | **+0.0000664551** | **+0.0006935550** |

- 양수 score 월: `8/12`
- validation p95 이동/설비용량: `0.75797%`
- production p95 이동/설비용량: `0.73475%`
- production 최대 이동/설비용량: `2.31037%`

기존 G2 `0.10` 단독 확장 후보와 비교하면 다음과 같다.

| 후보 | 2024 전체 macro delta | production p95 이동 | 전달 예상 score |
|---|---:|---:|---:|
| G2 `0.10` 단독 | +0.0002851581 | 1.24457% | 0.6463410793 |
| G1 `0.1375` + G2 `0.095` | **+0.0003800051** | **0.73475%** | **0.6464122451** |

새 후보가 로컬 증가폭은 더 크고 production 이동은 더 작다.

## 공개 전달 추정

단순히 공개 factor delta를 weight 배율로 곱하지 않았다. 각 그룹·각 metric마다
2024 base factor 효과와 실제 공개 factor 효과의 비율을 구한 후, 이번 추가 확장의
2024 incremental delta에 그 비율을 적용했다.

예상 추가 macro 효과는 다음과 같다.

| 요인 | score | 1-NMAE | FiCR |
|---|---:|---:|---:|
| G1 추가 확장 | +0.0000179794 | -0.0000024398 | +0.0000378128 |
| G2 추가 확장 | +0.0002691744 | +0.0001010274 | +0.0004214872 |
| 합계 | **+0.0002871537** | **+0.0000985876** | **+0.0004593000** |

예상 공개 지표는 score `0.6464122451`, 1-NMAE `0.8758828353`, FiCR
`0.4169252352`다. 이는 관측값이 아니라 두 개의 공개 factor 점에서 외삽한 값이다.

## 스트레스 테스트와 위험

최종 실행은 40/60 보완집합 `5,000`회와 H2 issue-block bootstrap `2,000`회를
사용했다.

| 검사 | score q05 | 1-NMAE q05 | FiCR q05 |
|---|---:|---:|---:|
| IID public 40% | -0.0001730008 | +0.0000059690 | -0.0004004233 |
| IID private 60% | +0.0000064566 | +0.0000251182 | -0.0000418192 |
| 월 층화 public 40% | -0.0001727792 | +0.0000050883 | -0.0003955073 |
| 월 층화 private 60% | +0.0000020232 | +0.0000243890 | -0.0000517031 |
| H2 issue block | -0.0004746067 | +0.0000281891 | -0.0010545700 |

평균 오차 하단은 모두 양수지만 FiCR 임계값의 불연속성 때문에 score 하단이
일부 음수다. 따라서 이 후보는 다음 조건으로만 제출한다.

1. 하루 제출 기회 중 1회만 사용한다.
2. 현 공개 최고 파일은 계속 보존한다.
3. 새 점수가 현 최고보다 낮으면 weight를 공개 결과에 맞춰 재튜닝하지 않고 이 확장
   계열을 닫는다.
4. 새 점수가 높으면 증가분으로 G1/G2 추가 효과를 재분해하되 G3는 계속 동결한다.

## 실패한 보정축

legacy exact driver와 issue-trajectory TCN의 50/50 미세 보정도 별도로 확인했다.
anchor에 total alpha `0.01`만 넣으면 Q1/Q2/H2/전체의 세 component가 모두
양수였지만 full delta는 `+0.0000910204`에 불과했고, 재표본 하단은 통과하지 못했다.

공개 양수 G1/G2를 함께 확장한 뒤 보정 alpha를 키우면 전 구간 양수 조합은
생겼지만 H2 issue-block FiCR q05가 계속 음수였다. production TCN을 새로 학습해
넣을 만큼의 추가 근거가 없어 제출 후보에는 포함하지 않았다.

## 재현

```powershell
python -m experiments.public_positive_stabilized_blend `
  --group1-weights 0.1125,0.125,0.1375,0.15,0.1625 `
  --group2-weights 0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09 `
  --stabilizer-alphas 0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05

python -m experiments.compose_public_positive_multifactor_candidate

python -m pytest `
  tests/test_public_positive_stabilized_blend.py `
  tests/test_compose_public_positive_multifactor_candidate.py -q
```

진단 JSON은 `artifacts_final/diagnostics/`에, 제출 CSV는
`artifacts_final/candidates/`에 보존한다. 중간 격자·단일축 임시 진단 파일은 최종
산출물에서 제거한다.
