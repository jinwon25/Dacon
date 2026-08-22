# 1159.33522 이후 1170 실행 계획 — 2026-08-22

> **완료 상태**: 이 문서의 Track A를 v84로 구현·제출했고 Public
> `1161.2020600422`를 기록했다. 현재 판단과 후속 계획은
> [`target1170_v84_public_result_20260822.md`](target1170_v84_public_result_20260822.md)를 우선한다.

## 현재 위치

v82를 실제 제출해 Public **1159.33522**를 확인했다. 직전 챔피언 대비
`+1.2606643249`이며 1170까지 `10.66478`이 남았다. 테스트 양성률을 `0.4861`로 놓으면
평균 Brier 약 `2.6641345e-05`의 추가 감소가 필요하다.

full-2024 OOF gain `+1.2596972212`와 Public gain의 차이는 `+0.0009671037`에
불과했다. 이는 exact parent와 row-local OOF 재현의 유효성을 지지하지만 clean transfer
관측 한 건뿐이므로 다른 family의 점수 환산계수로 사용하지 않는다.

## v83 새 부모 조건부 감사

v82 exact OOF를 새 부모로 고정하고 기존 frozen shift를 그 위에 더했다. 이 감사는
same-axis label을 사용한 headroom 진단이며 가중치 선택이나 제출 근거 단독 사용은 금지한다.

| 방향 | full-2024 gain | late-2024 gain | full/late 양수 월 | 핵심 판정 |
|---|---:|---:|---:|---|
| strict 추가 10% | -0.0763 | -0.7871 | 37.5% / 33.3% | 증량 금지 |
| v50 low-rank context | +2.5390 | +0.1340 | 37.5% / 33.3% | late·domain 실패 |
| v56 shared-horizon FM | +1.0181 | +2.9166 | 75% / 100% | 다음 복원 1순위 |

v56은 F-only, v82는 R_CORE-only라 활성 행이 겹치지 않는다. v56 최소 domain gain은
full/late 모두 `0`, 최악 월은 `-4.7126/+0.3596`이다. 반면 2022 선택축 최악 월
`-6.5616`, consensus gate `0/32`, FM family의 2024 반복 확인이라는 위험이 있다.

## Track A — v84 고정 v56 최종 모델

가장 먼저 실행한다. 새 탐색이 아니라 기존 레시피를 2025 추론용으로 복원한다.

고정값:

- family: main-effect-free pairwise FM
- rank: `16`
- risk: `source_domain_equal`
- route: `F`
- blend: logit damping `0.10`
- correction clip: `±0.25`
- test 집계: 사용 금지

구현 순서:

1. 기존 v56 `outer_full_2024.npz`로 full/late 수식 parity를 다시 확인한다.
2. 한 해 이동한 두 최신 legal OOF source를 고정한다. source 선택 규칙은 과거 v56과
   동일한 full-year + late-next-year 구조로 두고, target/row_id/parent hash를 manifest에
   기록한다.
3. 2025 최종 FM encoder, embedding weight, source-period×domain centre를 export한다.
4. `standalone_champion_1159.zip`과 한 payload로 결합한다. 과거 ZIP을 런타임에 읽지 않는다.
5. singleton/shuffle/partition `1e-12`, 245,789행 내부 120초, 공식 600초, 확률·CRC·루트
   구조를 검사한다.
6. OOF 레시피와 2025 export parity가 맞을 때만 별도 v84 probe ZIP을 만든다.

중단 조건:

- 기존 rank/risk/route/damping 중 하나라도 다시 선택해야 하면 중단
- source parent exact parity가 안 맞으면 중단
- F 외 domain이 바뀌면 중단
- standalone 120초 soft guard 또는 행 독립성 실패 시 중단
- Public 결과를 보고 damping을 0.05/0.15로 바꾸지 않음

## Track B — 팀원 exact OOF 앙상블

v77 계약이 최우선 병렬 입력이다.

1. `configs/oof_bundle_contract_v1.json` 형식으로 팀원별 동일 row exact temporal OOF를 받는다.
2. v82 부모와 residual correlation, analytic headroom을 먼저 계산한다.
3. 상관이 기존 계열 수준인 `0.99998+`이거나 full/late 어느 한 축의 headroom이 0이면 종료한다.
4. 통과 후보만 월·R_CORE/R_ANCHOR/F 비악화 constrained blend와 세 bootstrap, Reality
   Check로 보낸다.

Public 점수나 제출 ZIP만 있는 모델은 weight 학습에 사용하지 않는다.

## Track C — 새 직교 기반모형

v82 성공은 strict 모델의 다중 시간척도 history, group EB, tree residual, low-rank
pitcher-context 결합이 기존 챔피언의 R_CORE 잔차를 보완했음을 보여준다. 같은 strict weight를
늘리는 대신 다음 두 family만 신규 후보로 허용한다.

1. **계층 동적 확률모형**: 투수·타자·팀·카운트의 Beta-Binomial/상태공간 posterior를
   origin 이전 데이터만으로 갱신하고 posterior mean뿐 아니라 uncertainty interaction을
   사용한다. 현재 행의 ASOF와 동결 state만 입력한다.
2. **cross-fitted multi-task outcome model**: success/middle/reverse/way-off의 상호배타적
   구조를 먼저 예측하고 success를 marginalize한다. family 선택·calibration을 outer fold
   내부에서 완전히 반복한다.

TrackMan IVB/profile, strict 강도, v50 저랭크, v53/FM rank, TabM seed·크기, 전역 calibration,
domain×count lookup은 다시 탐색하지 않는다.

## 제출 정책

- 현재 champion: `standalone_champion_1159.zip`, Public `1159.33522`
- 1170 승격 기준: `configs/evaluation_v3.json`
- 탐색적 Public probe: 사용자가 명시 승인하고 recipe가 Public/2024 사후 튜닝이 아니며
  standalone·행 독립성 감사를 통과한 경우 한 번만 허용
- probe 성공 여부와 무관하게 동일 family 연속 weight line-search 금지
- 현재 우선순위는 **v84 고정 v56 복원 → 팀 OOF → 새 직교 family** 순서다.

## 재현

```powershell
python -m src.v83_post_public_rebase_audit `
  --project <private-project> `
  --final-parent-dir artifacts/v61_final_gate_oof_20260822_01 `
  --output-dir artifacts/v83_post_public_rebase_20260822_01
```

compact 결과는 `conditional_headroom.csv`와 `summary.json` 두 파일만 보존한다.
