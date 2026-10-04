# 목표 1185 — v354 late-maturity current-season hierarchy 후보

## 결론

제출 1순위를 v353에서 **`submit_v354.zip`으로** 올린다. v354는 v353 전체를 보존하고,
8월 이후 `R_ANCHOR` 행에만 현재 시즌 투수·타자 성공 표본의 계층적 posterior를 20% 혼합한다.

- 전달본: `submit_v354.zip`
- SHA-256: `564AFE4F14FC5B10FD286D78C371721024D66052A4FBDFC51778D35C10084E3F`
- 크기: `101,299,419 bytes`
- ZIP 멤버: `195`
- 부모: `submit_v353.zip`, SHA `EEB3841A...918B6`

## 야구 가설과 계산

시즌 초의 현재 시즌 제구율은 표본이 작아 전년도·커리어 prior의 영향이 커야 한다. 8월 이후에는
투수의 현재 시즌 표본이 충분히 쌓여 역할 변화, 구위·제구 변화와 ABS 환경 적응을 반영한다.
v354는 이 성숙 구간에서만 다음 확률을 계산한다.

1. 공식 train의 예측 연도 이전 행으로 투수·타자의 전체 누적 성공 수를 동결한다.
2. 현재 행의 공식 ASOF 누적 성공 수에서 이를 빼 현재 시즌의 `n/s`를 행별로 복원한다.
3. 직전 시즌의 선수×도메인 성공률을 투수 80, 타자 120의 강도로 수축한다.
4. 현재 시즌 표본을 강도 80으로 posterior 갱신한다.
5. 투수 75% + 타자 25%로 합친 뒤 현재 v353 확률을 이 값 쪽으로 20% 이동한다.

다른 테스트 행, 테스트 빈도·평균·정렬은 읽지 않는다. `game_month`, 선수 ID와 ASOF 값은 모두
현재 행에 공식 제공된 입력이고, prior lookup은 2024까지의 공식 train으로 고정돼 있다.

## 세 시간축 검증

원시 계층 모델과 `R_ANCHOR/0.20` 레시피는 과거 v39에서 고정됐다. 새 가설은 현재 시즌 표본이
성숙한 8월 이후에만 적용하는 것이다. 동일한 gate와 수식을 각 시점의 incumbent 위에 적용했다.

| 평가축 | 당시 부모 | 전체 gain | 적용행 gain | 적용 월 최악 gain |
|---|---|---:|---:|---:|
| full-2022의 8–10월 | v335 analogue | `+3.758190` | `+60.304271` | `+21.611071` |
| late-2023 | v335 | `+7.267935` | `+41.796968` | `+32.208565` |
| full-2024의 8–9월 | v345 | `+4.210609` | `+79.447666` | `+55.024951` |

세 축의 모든 적용 월이 양수다. locked 2024의 13,432개 적용 행 robustness는 다음과 같다.

| 재표본화 | p05 | 양수 확률 |
|---|---:|---:|
| 투수 cluster | `+34.6785` | `0.9980` |
| 투수×타자 crossed | `+14.4584` | `0.9755` |
| 시간 block | `+34.8550` | `0.9980` |

2후보 White reality-check는 `p=0.002499`다. v345의 post-v335 성분은 R_CORE에만 있어
v354 적용 행과 겹치는 행이 0개다. v352 TrackMan-PFD와의 적용행 증분 상관은 `0.183535`다.

late-2024에서 v352의 감사 PFD를 먼저 적용하고 v354처럼 계층 혼합한 보수적 proxy는
v345 대비 full-2024 `+4.371893`이며, 계층 성분 자체의 v352 proxy 대비 순증분은
`+4.083284`다. v353 PFD 전체 refit을 2024 정답으로 재평가하지 않았으므로 이 값은 v354의
완전한 OOF 점수라고 주장하지 않는다.

## 재현성과 실행 감사

- 2024 runtime hierarchy와 연구 함수의 최대 오차: `0`
- 독립 재빌드 ZIP SHA: 동일
- 혼합 fixture 수식/보호 경로 최대 오차: `0/0`
- fixture shuffle/partition 최대 오차: `1.11e-16/5.55e-17`
- 245,789행 standalone 실행: `200.109초` / 제한 600초
- static 금지 연산: `0`
- singleton/shuffle/partition: `0/0/0`
- 행 수·ID·결측·finite·확률 범위·ZIP root·CRC: 통과
- 전체 테스트: `732 passed, 22 skipped`

## 재현 명령

```powershell
python -m src.archive.v354_late_hierarchy_rebase_v345 `
  --train-csv data/train.csv `
  --v335-axes artifacts/v335_anchor_lowrank_complement_audit_20260830_01/selected_axes.npz `
  --v345-axes artifacts/v345_transition_workload_beta_cell_v335_20260831_01/selected_axes.npz `
  --v352-axes artifacts/v352_trackman_pfd_rebase_v345_20260831_01/selected_axes.npz `
  --output-dir artifacts/v354_late_hierarchy_rebase_v345_20260901_01

python -m src.champion.v354_build_late_hierarchy_package `
  --source-zip artifacts/v353_refit_trackman_pfd_package_20260831_01/submit_v353.zip `
  --train-csv data/train.csv `
  --audit-summary artifacts/v354_late_hierarchy_rebase_v345_20260901_01/summary.json `
  --output-dir artifacts/v354_late_hierarchy_package_20260901_01

python -m src.audit_v354_late_hierarchy `
  --v353-zip artifacts/v353_refit_trackman_pfd_package_20260831_01/submit_v353.zip `
  --v354-zip artifacts/v354_late_hierarchy_package_20260901_01/submit_v354.zip `
  --input-csv data/test.csv `
  --output-json artifacts/v354_late_hierarchy_package_20260901_01/runtime_fixture_audit.json

python -m src.audit_standalone_release `
  --package artifacts/v354_late_hierarchy_package_20260901_01/submit_v354.zip `
  --test-csv data/test.csv `
  --scale-rows 245789 `
  --timeout-seconds 600
```

Public 결과를 이용한 월·강도·경로 재탐색은 하지 않는다. v354가 기대와 다르더라도 그 결과는
모델 계열의 전이 위험을 판단하는 데만 쓰며 테스트 정답이나 분포를 역추정하지 않는다.
