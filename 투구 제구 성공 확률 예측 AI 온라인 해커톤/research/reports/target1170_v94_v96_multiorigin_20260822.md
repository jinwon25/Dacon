# 1170 후속 multi-origin 계약·context 전이 감사 — 2026-08-22

## 결론

공식 champion은 계속 **1161.2020600422**로 유지한다. v94에서 2020–2024의
strict-forward 공통 부모와 복원 가능한 exact v84 부모를 명시적으로 분리한 OOF 계약을
완성했다. 이 계약으로 v95 저자유도 context residual과 v96 연중 regime 재현을 검증했지만,
2024로 넘어가는 시점에 효과가 반전되거나 서로 다른 연도에서 exact 결과의 부호가 달랐다.
따라서 새 ZIP을 만들거나 DACON에 제출하지 않았다.

- champion: `artifacts/standalone_champion_1161/standalone_champion_1161.zip`
- Public: `1161.2020600422`
- 1170까지: `8.7979399578`
- 이번 사이클 제출: `0`
- test CSV·test 행 집계·Public 기반 recipe 선택: 없음
- 후속 전달 원칙: 항상 `script.py + requirements.txt + model/`만으로 실행되는 standalone

## v94: fidelity-labelled multi-origin OOF 계약

기존 연구의 핵심 병목은 과거 시즌 prediction을 현재 champion과 같은 것처럼 섞어
해석한 데 있었다. v94는 이를 두 evidence tier로 분리했다.

| tier | 축 | 용도 |
|---|---|---|
| common wave0 | full 2020, 2021, 2022, 2023, 2024 | 메커니즘 방향·시간 안정성 |
| exact v84 | full-2022 R_CORE/F 82.70%, late-2023 전체, full-2024 전체 | 배포 champion 대비 marginal gain |

각 NPZ에는 원본 train 행 인덱스, target, parent, exact mask, season, month, domain,
pitcher/batter ID와 SHA-256을 저장했다. train CSV SHA-256은
`D2081186B458B49F60B082BE480C273135833E15BA59A76D033AF28BCF8763FF`다.

공통 부모 평균 잔차 `E[y-p]`는 2020 `-0.02373`, 2021 `+0.03305`, 2022
`+0.01806`, 2023 `-0.02456`, 2024 `-0.00314`였다. 기준 확률의 오차 방향이
계속 바뀌므로, 한 시즌의 global/domain calibration을 다음 시즌에 운반하는 전략이
불안정하다는 기존 결론을 더 긴 시간축에서 재확인했다.

중요한 해석 규칙은 다음과 같다.

1. common wave0 gain을 exact-v84 gain으로 부르지 않는다.
2. `v84_full_2022`는 `exact_mask=true`인 R_CORE/F 행만 exact 비교에 사용한다.
3. 한 audit target을 본 뒤 같은 축의 recipe·route·eta를 다시 선택하지 않는다.
4. test 행의 분포·빈도·순서·다른 행을 어떤 feature에도 사용하지 않는다.

## v95: 다중-origin 저자유도 context transport

현재 행에서 알 수 있는 count, hands, base/out, inning, runners, score/LI, team matchup만
사용했다. source residual은 domain별 평균을 제거해 global offset 운반을 막았고,
10 group family × alpha 3 × eta 3 × route 3, 총 270개 recipe를 오직
2020→2021과 2021→2022 common wave0에서 선택했다.

source gate를 통과한 recipe는 52개였다. 동결된 최종 선택은
`count / R_CORE+F / alpha=5000 / eta=1`이었다.

| 축 | gain | 양수 월 | 최악 월 | 최소 적용 domain |
|---|---:|---:|---:|---:|
| common 2020→2021 | +18.4711 | 100% | +10.1975 | +17.2309 |
| common 2021→2022 | +25.5134 | 100% | +11.8812 | +8.0528 |
| exact v84 full22→late23 | +7.6946 | 100% | +4.9641 | +2.7999 |
| exact v84 late23→full24 | **-7.4283** | 25% | **-21.2276** | **-10.2239** |
| exact v84 late-2024 | **-4.4207** | 33.3% | **-9.2369** | **-6.1669** |

오래된 세 전이에서 강했던 count residual이 2024에서 명확히 반전됐다. 2024 결과를 본 뒤
다른 count bin, alpha, eta를 고르는 것은 family 내부의 same-axis 재탐색이므로 금지했다.

## v96: 연도 경계와 연중 적응 분리

v95 recipe를 그대로 고정한 채 각 시즌 3–7월 residual로 8–10월을 보정했다. 이 실험은
2024 label을 이미 사용하는 진단이며 패키징 근거가 아니다.

| 축 | tier | gain | 양수 월 | 최악 월 |
|---|---|---:|---:|---:|
| 2020 early→late | common | +10.5285 | 100% | +8.9663 |
| 2021 early→late | common | +20.0428 | 100% | +0.2517 |
| 2022 early→late | common | +19.0148 | 100% | +5.5206 |
| 2023 early→late | common | +6.2846 | 66.7% | -9.7969 |
| 2024 early→late | common | +20.2005 | 100% | +19.4459 |
| 2022 early→late | exact v84 | **-3.4258** | 33.3% | -8.5963 |
| 2024 early→late | exact v84 | +2.3382 | 66.7% | -1.4464 |

공통 약한 부모에서는 연중 재적합이 모두 양수지만, 실제 v84 잔차에서는 2022와 2024의
부호가 다르다. 더 중요한 점은 2025 평가 target을 알 수 없어 2025 초반 residual table을
fit할 수 없다는 것이다. 공식 ASOF feature는 현재 행 이전의 선수 상태를 전달하지만,
v95의 count별 champion residual 정답을 제공하지 않는다. 숨은 test label이나 다른 test
행 집계를 이용해 이를 복원하는 방식은 허용되지 않는다.

따라서 v96의 올바른 결론은 “현재 시즌 target으로 online 적응하면 잠재 headroom이 있다”이지,
“2024 effect를 2025에 운반해도 된다”가 아니다.

## 공개 대회 자료 재감사

2026-08-22에 공식 코드공유·토크와 공개 GitHub 구현을 다시 확인했다.

- 공식 코드공유에는 RandomForest baseline 외 새로운 고성능 해법이 공개되지 않았다.
- `hoo743-ui/LG_Aimers09`의 recent-ASOF, CatBoost, TrackMan 상황 편차, season 신호는
  현재 champion 또는 v14/v62/v78/v79 계열에서 이미 더 강한 부모와 시간축으로 검증됐다.
- `thisisacrane/lg-aimers9-pitch-control`의 2023 이전 F 제거는 공개 구현의 regime 완화
  아이디어로 유효하지만, 현재 연구에서는 hard F drop과 F 재학습을 이미 감사했고 v84의
  shared-horizon F residual이 더 높은 Public을 만들었다.
- `x2-qp-cheese/LGAimers`의 CatBoost·LightGBM·Residual MLP 다양성은 방법론 참고 대상이지만,
  현재 champion과 같은 행의 strict-forward exact OOF 및 2025 test prediction bundle이 없어
  v77 blend weight를 정할 수 없다.

외부 공개 코드를 곧바로 답처럼 이식하거나 공개 점수로 blend 비율을 정하지 않는다. 새
독립 모델은 코드·feature·fold cutoff·row_id-aligned OOF·test prediction을 함께 받아야 한다.

## 규정·통계 판정

- 공식 평가행은 서로 독립적으로 예측한다. singleton, shuffle, partition 결과가 같아야 한다.
- 현재 투구의 위치·의도·TrackMan 물리량을 안다고 가정하지 않는다.
- v95/v96은 train OOF만 읽었고 test CSV, test 행 수·분포·집계, Public 점수를 읽지 않았다.
- v95는 exact 2024 gain, 월, domain gate를 모두 실패했다.
- v96은 2024 target을 사용하는 development 진단이고 exact 2022 재현도 음수다.
- 따라서 bootstrap·Reality Check·standalone packaging·DACON 제출 단계로 진행하지 않는다.

## 1170을 위한 다음 우선순위

1. **팀원 독립 exact OOF 확보**: 동일한 `(axis,row_id)`의 2022·2023·2024 OOF와 한 recipe의
   2025 test prediction을 `configs/oof_bundle_contract_v1.json` 형식으로 받는다. 현재 가장
   큰 미개척 headroom은 기존 계보 밖의 오차 공분산이다.
2. **완전 exact multi-origin ladder**: v94 common tier를 배포 champion과 혼동하지 않고,
   비용이 허용될 때 각 origin에서 feature 선택·모델 학습·calibration까지 다시 수행한다.
3. **새 관측 정보가 있는 독립 base만 검토**: 같은 ASOF/TrackMan/FM/CatBoost의 seed·강도
   변형은 중단한다. 새 모델은 기존 champion과 낮은 error-direction correlation을 먼저 보여야 한다.
4. **독립 OOF가 도착하면 v77 순서 고정**: analytic headroom → residual correlation →
   worst-month/domain constrained blend → pitcher/crossed/block bootstrap → Reality Check.
5. 위 gate를 통과한 경우에만 처음부터 standalone ZIP으로 만들고 한 번 제출한다.

현재 로컬 정보만으로 1161.2020600422를 상회할 가능성을 통계적으로 뒷받침하는 새 후보는
없다. 제출 횟수를 추가 tuning signal로 쓰지 않고 champion을 보존하는 것이 최종 판정이다.

## 재현

```powershell
python -m src.archive.v94_multi_origin_champion_contract --help
python -m src.archive.v95_multiorigin_context_transport --help
python -m src.archive.v96_context_regime_replication --help

python -m pytest -q `
  tests/test_v94_multi_origin_champion_contract.py `
  tests/test_v95_multiorigin_context_transport.py `
  tests/test_v96_context_regime_replication.py
```

주요 생성 산출물은 다음 경로에 있으며 Git에는 포함하지 않는다.

- `artifacts/v94_multi_origin_champion_contract_20260822_01/`
- `artifacts/v95_multiorigin_context_transport_20260822_01/`
- `artifacts/v96_context_regime_replication_20260822_01/`

## 참고 링크

- 공식 평가: https://dacon.io/competitions/official/236743/overview/evaluation
- 공식 행 독립성 공지: https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent
- 공식 규정 준수 리마인드: https://dacon.io/competitions/official/236743/talkboard/417094?dtype=recent&page=1
- 공식 코드공유: https://dacon.io/competitions/official/236743/codeshare?dtype=recent&page=1
- 공개 구현 1: https://github.com/hoo743-ui/LG_Aimers09
- 공개 구현 2: https://github.com/thisisacrane/lg-aimers9-pitch-control
- 공개 구현 3: https://github.com/x2-qp-cheese/LGAimers
