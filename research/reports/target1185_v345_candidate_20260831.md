# 목표 1185 — v345 전이·workload·Beta 셀 제출 후보

> **사후 결과 (2026-08-31):** 제출 ID `78710`, Public **1182.94969702**로
> v335 대비 `+1.2396938587` 개선돼 새 챔피언이 됐다. 목표 1185까지는
> `2.05030298`이 남으며, 같은 계열의 route·weight·seed를 Public으로 재튜닝하지 않는다.

## 결론

공식 챔피언은 제출 ID `76835`, Public `1181.7100031613`의 `submit_v335.zip`이다.
현재 제출 1순위는 `submit_v345.zip`이다. v345는 미제출 v343을 부모로 하고, 과거 두
source origin에서 이미 고정한 Beta-Binomial 셀 한 개를 추가한다.

- v343: R_CORE 선수 전이 + 3시드 workload-H1
- v345 추가분: `R_CORE|DEVELOPING|MIXED` 셀의 strict-forward Beta-Binomial 방향 10%
- F, R_ANCHOR, 그 밖의 R_CORE 행: v343과 정확히 동일
- Beta pooling: 2024 label만으로 2025 파라미터를 고정
- 선수 시즌 상태: 공식 train의 2024 말 snapshot과 현재 테스트 행의 공식 ASOF만 사용

Public 결과로 route, weight, seed를 고르지 않았다. full-2024는 두 부모 연구에서 이미
노출된 개발 축이므로 독립 holdout으로 주장하지 않는다.

## 로컬 근거

모든 수치는 동일한 full-2024 행에서 v335 exact OOF와 paired Brier로 비교했다.

| 후보 | v335 대비 gain | 양수 월 | 최악 월 | 전체행 RMS |
|---|---:|---:|---:|---:|
| v343 | `+2.944474` | 7/8 | `-25.410144` | `0.001997742` |
| **v345** | **`+3.390593`** | **6/8** | `-25.410144` | **`0.002001524`** |
| v345의 Beta 순증분 | **`+0.446119`** | — | — | — |

v343 증분과 Beta 증분의 상관은 `-0.055423`이다. Beta 셀은 4,022행이며 v343의 R_CORE
지지집합 안에 있지만, 예측 방향은 거의 무상관이다. Beta 셀 단독의 v335 기준 source
gain은 full-2022 `+0.064111`, late-2023 `+0.702617`, full-2024 `+0.422936`으로 세 축에서
모두 양수다. 다만 full-2022 효과가 작고 월별 꼬리가 불안정하므로 저용량 보완으로만 쓴다.

full-2024 결합 robustness:

| 재표집 | p05 | 양수 확률 |
|---|---:|---:|
| 투수 클러스터 | `+0.839776` | `0.9745` |
| 투수×타자 crossed | `-1.967204` | `0.8805` |
| 시간 블록 | `+1.040747` | `0.9810` |

4개 최종 후보 범위 White Reality Check는 `p=0.024988`이다. 이는 전체 역사 탐색 횟수를
보정하지 않으므로 진단값이다. crossed p05는 음수지만 과거 확정 제출에서 이 조건은 Public
방향과 일치하지 않았으므로 단독 기각 사유로 쓰지 않는다.

## 2025 full-fit

Beta pooling은 2024를 calibration year로 고정해 다음 파라미터를 얻었다.

| 항목 | 값 |
|---|---:|
| concentration | `50.0` |
| pitcher season posterior | `0.6997719892` |
| batter season posterior | `0.2307766581` |
| previous game-type prior | `0.0` |
| pitcher career rate | `0.0380374098` |
| pitcher prev5 rate | `0.0314139429` |
| pitcher terminal snapshots | `792` |
| batter terminal snapshots | `830` |

추론 시 각 행의 투수·타자 누적 성공수에서 공식 train으로 고정한 2024 말 누적 성공수를
빼 현재 시즌 표본을 복원한다. 다른 test 행의 빈도, 평균, 정렬, rolling, 그룹 통계를
사용하지 않는다.

## standalone ZIP과 감사

| 항목 | 값 |
|---|---:|
| 파일 | `artifacts/v345_transition_workload_beta_package_20260831_01/submit_v345.zip` |
| SHA-256 | `D44578DC50220CE84DD4B8489BBAE680AFDCF93F931ED4236287B5A9E6F5AAAA` |
| 크기 | `101,139,163 bytes` |
| 멤버 | `191` |
| CRC | 통과 |
| 공식 루트 | `script.py`, `requirements.txt`, `model/` |
| 245,789행 런타임 | `205.039초 / 600초` |
| 정적 금지 연산 | `0` |
| singleton/shuffle/partition 최대 오차 | `0 / 0 / 0` |
| 혼합 경로 수식 오차 | `0` |
| 혼합 경로 비활성 parity | `0` |
| 2024 저장 축 vs 패키지 Beta 런타임 | `0` |

서로 다른 출력 디렉터리에서 전체 builder를 두 번 실행했고 ZIP SHA-256, 멤버 목록,
멤버 CRC와 크기가 모두 일치했다. 빌더는 2024 strict-forward 저장 축과 동일한 방식으로
Beta 후보별 clip을 적용하는 패키지 런타임을 재계산하며 최대 오차 `0`일 때만 ZIP을 만든다.
전체 pytest 결과는 `706 passed, 22 skipped`이며 저장소 감사도 1,230개 파일에서 통과했다.

기존 scikit-learn 1.7.2 pickle을 package의 고정 1.6.1 환경에서 읽을 때 경고가 발생한다.
이는 v290·v335와 동일한 기존 계보 위험이며 두 패키지는 공식 채점을 통과했다. 신규 Beta
번들은 모델 pickle이 아닌 고정 숫자·snapshot lookup이다.

## 재현

```powershell
python -m src.archive.v345_transition_workload_beta_cell_v335 `
  --train-csv data/train.csv `
  --v343-axes artifacts/v339_transition_workload_portfolio_v335_20260831_01/selected_axes.npz `
  --beta-axes artifacts/v337_beta_cell_rebase_v335_20260831_01/selected_axes.npz `
  --beta-summary artifacts/v337_beta_cell_rebase_v335_20260831_01/summary.json `
  --output-dir artifacts/v345_transition_workload_beta_cell_v335_20260831_01

python -m src.champion.v345_build_transition_workload_beta_package `
  --source-zip artifacts/v343_transition_workload_package_20260831_01/submit_v343_transition_workload.zip `
  --train-csv data/train.csv `
  --beta-axes artifacts/v321_strict_beta_binomial_complement_20260830_01/selected_axes.npz `
  --audit-summary artifacts/v345_transition_workload_beta_cell_v335_20260831_01/summary.json `
  --output-dir artifacts/v345_transition_workload_beta_package_20260831_01

python -m src.audit_standalone_release `
  --package artifacts/v345_transition_workload_beta_package_20260831_01/submit_v345.zip `
  --test-csv data/test.csv `
  --scale-rows 245789 `
  --timeout-seconds 600

python -m src.audit_v345_transition_workload_beta `
  --v335-zip submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip `
  --v343-zip artifacts/v343_transition_workload_package_20260831_01/submit_v343_transition_workload.zip `
  --v345-zip artifacts/v345_transition_workload_beta_package_20260831_01/submit_v345.zip `
  --train-csv data/test.csv `
  --output-json artifacts/v345_transition_workload_beta_package_20260831_01/mixed_route_audit.json
```

## 제출 판단

v345는 v343보다 로컬 효과·투수 bootstrap·시간 bootstrap이 모두 개선됐고, 목표 1185의
필요 증분 `+3.289997`보다 full-2024 paired gain이 처음으로 크다. 로컬 gain을 Public으로
1:1 환산할 수는 없지만 현재 재현 가능한 후보 중 목표 도달 가능성이 가장 높다.

DACON 자격증명이 준비되면 업로드명 `submit_v345.zip`으로 한 번 제출한다. Public 결과를
동일 Beta 셀의 weight·support cut·pitchmix cut 재선택에 사용하지 않는다.
