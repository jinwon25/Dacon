# 1170 후속 연구 — 독립 팀원 OOF 부재 가정 (v104~v108)

> **2026-08-22 Public 후속 확인**: 사용자가 명시적으로 승인한 v104 1회 probe는 제출 ID
> `60626`, Public **1162.6302840289**를 기록해 v84 대비 **+1.4282239867** 상승했다.
> 아래의 “기각/추가 제출 없음” 판단은 제출 전 엄격한 로컬 promotion gate 판정으로 보존하되,
> 운영 champion은 `standalone_champion_1162.zip`으로 갱신한다. Public 결과를 이용한 동일 recipe
> 재튜닝은 금지한다. 상세 결과는 [`target1170_v104_public_result_20260822.md`](target1170_v104_public_result_20260822.md)에 있다.

## 결론

팀원의 독립 모델 exact OOF와 동일 모델의 2025 예측은 **없는 것으로 고정**했다. 그 파일을
기다리거나 가정해 만든 weight는 하나도 없다. 내부에서 재현 가능한 strict-forward 예측만으로
source 안정성 mask, 교차 아키텍처 합의, raw player-ID paired ablation, batter-ASOF 제거,
pitcher-balanced 학습을 추가 검증했지만 **1170 이상을 기대할 승격 후보는 확보되지 않았다**.

현재 제출 추천은 계속 Public `1161.2020600422`의 standalone champion ZIP이다. v104~v108은
모두 연구 산출물이며 package와 submission을 만들지 않았다.

## 평가 계약

- recipe 선택: 2020~2021 common-forward 또는 사전 고정된 full-2022 + late-2023 source만 사용
- 감사: exact-v84 full-2022, late-2023, full-2024와 late-2024
- point gate: 모든 축 gain 양수, 양수 월 비율 75% 이상, 최악 월 `>-5`, 최소 domain gain 0 이상
- robust gate: pitcher, crossed pitcher×batter, chronological-block bootstrap p05가 모두 양수이고
  White Reality Check `p<=0.10`
- 모든 신규 실험: `test_csv_read=false`, test aggregate/순서/빈도/분포 미사용, row-local inference
- full-2024는 이미 여러 family가 관찰한 development-contaminated audit이며 virgin holdout으로
  표현하지 않는다.

## 실험 결과

| 실험 | source 결과 | exact/locked 결과 | 판정 |
|---|---|---|---|
| v104 source-stability mask | full22 `+9.9172`, late23 `+8.1216`; 전 월 양수 | full24 `+3.2162`, 양수 월 75%, 최악 `-6.6440`; late24 `+2.5868`, 양수 월 66.7%; full24 crossed p05 `-2.476`, Reality `p=.117` | 기각 |
| v105 LGB+MLP paired-ablation 합의 | full22 `+4.6140`, late23 `+4.0588`, 그러나 source 월 gate 실패 | full24 `+0.6609`, 최악 `-12.5603`; late24 `-2.7968` | 기각 |
| v106 raw player-ID paired ablation | source 최적 eta `0` | 고정 2.5% R_CORE+F dose도 2021 `-8.7186`, 2022 `-5.1175`, 2023 `-3.7893`, 2024 `-6.4010` | 기각 |
| v107 batter cumulative ASOF 제거 | 2020은 월/domain 불안정, 2021 평균만 양수 | exact full22 `-14.9310`; late23 `+5.4150`이나 최소 domain `-14.0502`; full24 `-11.6775` | 기각 |
| v108 inverse-sqrt pitcher-season weighting | source 최적 eta `0` | 승격 dose 없음 | 기각 |

### v104가 1170 후보가 아닌 이유

v104는 기존 최선 연구축 v103의 count, pitcher-history, platoon별 source gain 부호만으로 mask를
고정했다. source에서 `2-0` count는 `+8.32/+44.77`, history `30-199`는 `+18.21/+6.87`이었지만
2024에는 같은 그룹이 손실로 바뀌었다. source subgroup 안정성이 미래 subgroup 안정성을
보장하지 않았고, full/late-2024의 월·bootstrap gate도 통과하지 못했다. 2024 평균 `+3.22`는
현재 1170 gap `8.79794`를 설명하기에도 부족하다.

### v106~v108이 닫은 경로

- raw ID 단독 모델의 약함이 아니라, 동일 rows/weights/seed/hyperparameters에서 ID 다섯 열만
  더한 **순수 증분**이 source에서 0 dose를 선택했다. ID 모델 용량·seed 미세조정은 재개하지 않는다.
- batter handedness와 상황은 남기고 누적 batter-ASOF 네 열만 제거했지만 연도 전이가 반전됐다.
  batter 블록 제거/증폭 모두 제출 근거가 없다.
- ExtraTrees와 별도로 LightGBM 학습분포만 pitcher-balanced로 바꿔도 source dose가 0이었다.
  balance exponent 미세탐색은 기대효과보다 family-wise selection 위험이 크다.

## 남아 있는 내부 최선과 위험

v103 fixed union은 full22 `+9.4508`, late23 `+7.4718`, full24 `+3.1148`, late24 `+4.8156`으로
평균은 모두 양수다. 그러나 full24 최악 월 `-8.1258`, late24 양수 월 66.7%, full24 bootstrap
p05가 pitcher `-0.441`, crossed `-2.614`, block `-1.081`, Reality Check `p=.133`이다. v104 mask도
이 하방위험을 제거하지 못했다. 따라서 현재 증거로는 v103/v104를 standalone 또는 champion
overlay 제출로 추천할 수 없다.

기존 transfer 시나리오가 제시한 범위도 대략 `1164.3~1166.9`로 1170 미만이다. 이는 Public
예측 시나리오일 뿐 승격 증거로 사용하지 않았다.

## 보존·재현

현재 보존해야 할 제출 파일:

```text
artifacts/standalone_champion_1161/standalone_champion_1161.zip
SHA256 C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7
```

신규 실험 재현 예시:

```powershell
python -m src.archive.v106_paired_player_identity --help
python -m src.archive.v107_batter_asof_ablation --help
python -m src.archive.v108_pitcher_balanced_weighting --help
python -m pytest tests/test_v104_source_stability_mask.py `
  tests/test_v105_cross_architecture_ablation.py `
  tests/test_v106_paired_player_identity.py `
  tests/test_v107_batter_asof_ablation.py `
  tests/test_v108_pitcher_balanced_weighting.py -q
```

수치 산출물은 각각 `artifacts/v104_*`~`artifacts/v108_*` 아래에 있으며 git 추적 대상이 아니다.
추천 행동은 **새 제출 없음, 1161.2021 champion 보존**이다. 추가적인 유의미한 승격 검증에는
새로운 독립 exact OOF/2025 예측 또는 아직 보지 않은 shadow labels가 필요하다.
