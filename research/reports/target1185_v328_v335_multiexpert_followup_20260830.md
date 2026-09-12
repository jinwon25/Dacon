# 목표 1185 후속 연구 — 다중 전문가·대체 목적함수·제출 포트폴리오 (v328–v335)

> **사후 결과 (2026-08-31):** v335는 제출 ID `76835`, Public
> **1181.7100031613**을 기록해 v290보다 `+4.9529314933` 개선됐고 새 챔피언으로
> 승격됐다. 제출 전 판단은 아래에 원형대로 보존하며, 공식 결과 해석은
> [`v335_public_result_20260831.md`](v335_public_result_20260831.md)를 따른다.

## 결론

공식 챔피언은 여전히 v290 `1176.757071668`이다. v320은 현재 가장 중요한 미제출
탐색 후보지만 1180이나 1185를 보장하지 않는다. 이번 사이클에서는 기존 모델의 파라미터를
반복 조정하지 않고, 야구 역할별 전문가 라우팅, 연속 minimax 포트폴리오, F 신뢰도별 투입량,
4분류 실패유형 목적함수, 선수 이동 상태를 서로 독립된 방법론으로 검증했다.

새로 제출 가능한 파일은 두 개다.

1. **v335 보수형 보완 후보**: v320의 F 포트폴리오를 그대로 두고, 여러 연도에서 재현된
   R_ANCHOR 저랭크 상호작용만 추가한다.
2. **v334 공격형 분산 후보**: v335에 R_CORE 선수 이동 상태 잔차까지 추가한다. 최종
   학습축은 강하지만 leave-one-origin 검증에 실패했으므로 우선순위가 낮다.

## 방법론별 판정

| 버전 | 독립 방향 | 핵심 결과 | 판정 |
|---|---|---|---|
| v328 | 야구 archetype consensus MoE | `+3.693/+3.927/+1.224`, 반대 origin 전이 `-4.313` | source reject |
| v329 | 두 origin 공통 cell router | `+14.120/+11.666/-1.802` | locked reject |
| v330 | leave-one-origin + 최종 3-origin router | 최종 `+5.920/+2.897/+3.485`, LOO 2024 `-2.882` | cross-origin reject |
| v331 | 비음수 연속 minimax 전문가 포트폴리오 | LOO 최소 `-0.171`, R_CORE `-0.679` | reject |
| v332 | F 신뢰도별 direct dose 10→20→30% | source `+119.462`, locked `-0.641` | reject; 20% 포화 |
| v333 | 누적카운터 복원 4-class 실패유형 student | `+0.573/+0.859/-0.143` | locked reject |
| v334 | R_ANCHOR lowrank + R_CORE transition | 최종 `+5.920/+2.897/+3.485` | 공격형 백업 |
| v335 | R_ANCHOR lowrank만 유지 | `+3.191/+2.411/+1.731` | 보수형 후보 |

## v335 보수형 후보

고정된 과거 레시피 `lowrank_s300_r2`, dose `0.50`을 v320의 `R_ANCHOR`에만 적용한다.
R_ANCHOR는 정규시즌에서 투수 또는 타자 팀 ID가 13인 행이며, 다른 R 행과 F 행은 v320과
완전히 동일하다.

- full-2022: `+3.191095`, 양수 월 `6/7`
- late-2023: `+2.411287`, 양수 월 `3/3`
- full-2024: `+1.730744`, 양수 월 `7/7`, 최악 활성 월 `+0.867`
- full-2024 pitcher bootstrap p05 `+1.323`
- chronological block p05 `+1.712`
- crossed pitcher×batter p05 `-5.958`
- 제한된 2후보 Reality Check `p=0.02649`

route/expert 자체는 다중 전문가 화면에서 발견했으므로 마지막 p값이 전체 연구 탐색 횟수를
보정하지 않는다는 선택 편향 경고를 유지한다. 그래도 세 origin의 전체 gain과 월별 방향성이
모두 재현된 점은 v334의 R_CORE 신호보다 낫다.

ZIP:
`artifacts/v335_anchor_lowrank_package_20260830_01/submit_v335_anchor_lowrank_complement.zip`

SHA-256:
`A3BCD933DE3F78DEB9565F537DC3199922BDFD219B2840E4150F939CFF144C7A`

런타임 감사: F parity `0`, R_CORE parity `0`, R_ANCHOR 공식 오차 `0`, shuffle `0`,
partition `5.55e-17`, CRC 통과.

## v334 공격형 분산 후보

v320 위에 다음 두 전문가를 disjoint route로 추가했다.

- R_ANCHOR: finalized lowrank `0.50`
- R_CORE: 2020–2024 forward OOF에서 학습한 `pitcher_status_count` 잔차, alpha `1000`, dose `0.25`

선수 상태는 2025 행 하나의 `pitcher_id`, `pitcher_team_id`, count와 공식 과거 이력만으로
NEW/RETURN/SAME/SWITCH를 결정한다. test의 다른 행, 순서, 집계는 사용하지 않는다.

ZIP:
`artifacts/v334_aggressive_moe_package_20260830_01/submit_v334_aggressive_training_moe.zip`

SHA-256:
`B830E1C32B8AE19CFF44A8FA30B888F758520F5CBFCF13E99B91EF4DC2F6EF67`

런타임 감사: F parity `0`, R 공식 오차 `5.55e-17`, shuffle/partition `1.11e-16`,
transition singleton `0`, CRC 통과. 단, LOO 2024가 `-2.882`이고 crossed p05도
`-1.900`이므로 strict champion이 아니다.

## 제출 의사결정

제출 가능 횟수가 회복되면 한 번에 여러 후보를 올리지 않는다.

1. 아직 Public 미확인인 v320을 먼저 제출해 F 포트폴리오의 실제 방향을 확인한다.
2. v320이 v290보다 개선되면 다음 독립 슬롯은 **v335**가 우선이다.
3. v334는 v335 결과와 무관하게 자동 제출하지 않는다. v335가 개선되거나 R 보완축을
   지지하는 새 증거가 있을 때만 분산 후보로 검토한다.
4. 동일 family의 dose·team route·alpha를 Public 점수에 맞춰 재튜닝하지 않는다.

v335의 full-2024 추가 이득 `+1.731`과 v320의 `+3.809`를 합쳐도 1185 격차
`+8.242928332`를 설명하지 못한다. 따라서 현재 결과는 1185 달성 보장이 아니라,
검증 가능한 제출 포트폴리오를 두 단계로 확장한 것이다.

## 검증 상태

- 전체 테스트: `685 passed, 21 skipped, 1 existing warning`
- 공식 BSS 산식 및 동일 행 paired Brier 비교: 유효
- local→Public 고정 환산, 예상 점수 구간, `1180 확실` 표현: 계속 금지
- full-2024: 반복 연구에 노출된 최종 학습축이며 독립 holdout으로 부르지 않음
