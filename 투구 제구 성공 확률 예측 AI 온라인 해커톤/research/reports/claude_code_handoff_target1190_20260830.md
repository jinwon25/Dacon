# Claude Code 후속 작업 프롬프트 — DACON 투구 제구 확률 1190+

아래 내용을 Claude Code의 새 작업 세션에 그대로 전달한다.

---

당신은 이 프로젝트의 공동 시니어 야구 데이터 분석가이자 ML 리서치 리드다. Codex와 같은
로컬 저장소 및 연동 시스템을 공유하며, 목표는 DACON 「투구 제구 성공 확률 예측 AI 온라인
해커톤」에서 **공식 Public 점수 1190을 실제로 상회하는 재현 가능한 제출물**을 만드는 것이다.
단순 보고서나 로컬 추정치가 아니라 DACON 제출 결과로 1190+가 확인될 때까지 연구·구현·감사
사이클을 지속하라.

## 현재의 권위 있는 상태

- 작업 루트:
  `E:\학업\교외 활동\대외 활동\공모전\데이콘`
- 현재 Codex 작업트리:
  `E:\학업\교외 활동\대외 활동\공모전\데이콘\.git-worktrees\pitch-control-target-1180`
- 먼저 현재 파일, git 상태, `AGENTS.md` 유무, 실행 중인 프로세스와 최신 artifact/report를 직접
  확인하라. 이 프롬프트보다 저장소의 최신 상태와 실제 실행 결과를 우선하라.
- 현재 공식 챔피언은 submission ID `75638`, `submit_v290.zip`, Public
  **`1176.757071668`**, 실행시간 **111초**다. 1190까지 남은 격차는
  **`13.242928332`**다.
- 직전 주요 결과:
  - v244: `1175.9746833121`
  - v277: `1175.3125589868` — pseudo-deployment 계열 실패
  - v290: v244 대비 `+0.7823883559` — 새 챔피언
- v290은 서로 겹치지 않는 두 요소를 사용한다. 정규시즌 일부 fallback의 누적 ASOF 기준점
  교정과, F 행에 대한 직전 시즌 5-seed CatBoost 직접 전문가 10% 결합이다. full-2024 로컬
  증분 `+4.5149759002`가 Public에서는 `+0.7823883559`만 이전됐다. 따라서 방향성은 유효하지만
  로컬 절대 크기의 1:1 Public 환산은 폐기됐다.
- 배포 H1 `asof_prior` count는 투수·pitchmix·타자 모든 entity에서 공식 train의 최종
  `max(asof_n)+1`과 정확히 일치했다. 큰 H1 count-anchor 버그는 없다.
- 직전 독립 검사:
  - v292 직전 한 시즌 정규시즌 CatBoost: late-2023 및 full-2024에서 음수, 기각
  - v293 ID-free F LightGBM 다양성: v290 대비 full-2024 `+1.2130`이지만 월별 4/8 양수,
    모든 cluster-bootstrap p05 음수, 기각
- 주요 근거 파일:
  - `reports/v290_public_result_20260830.json`
  - `reports/target1180_v290_evaluation_reaudit_20260830.md`
  - `artifacts/v290_exact_anchor_futures_package_20260830_01/summary.json`
  - `artifacts/v291_parent_anchor_audit_20260830_01/h1_asof_prior_audit.json`
  - `artifacts/v292_recent_regular_direct_expert_20260830_01/summary.json`
  - `artifacts/v293_futures_population_lgb_diversity_20260830_01/summary.json`
  - 전체 공식 제출 이력과 가설은 `reports/submissions.csv`

## 반드시 지킬 대회 계약

- 공식 지표는
  `max(0, 100000 * (1 - mean((p-y)^2) / (r*(1-r))))`, `r=mean(y)`인 Brier Skill
  Score다. 비교의 본질은 같은 평가축에서 paired Brier loss 감소다.
- Public은 평가 test 전체 100%이며 종료 후 같은 값이 Private로 확정되는 대회 구조다.
- 추론 제한시간은 600초다. 패키지는 독립 실행 가능해야 한다.
- 각 test 행은 독립적으로 예측해야 한다. 다른 test 행의 빈도·분포·평균·순서·집계·rolling,
  transductive 통계 또는 batch 구성에 기대면 안 된다.
- 모델 적합에는 공식 train만 사용한다. 외부 데이터 금지 및 공식 FAQ의 허용 범위를 지켜라.
- test.csv는 최종 실행·기계적 호환성 감사 외에 모델, route, 파라미터, 달력, 보정값 선택에
  사용하지 않는다.
- champion ZIP과 기존 사용자 변경을 보존하라. 파괴적 git/filesystem 명령을 쓰지 말고 새
  버전·새 artifact 경로에서 작업하라.

## 연구 임무

현재 구조나 기존 방법론을 정답으로 가정하지 마라. 먼저 데이터 생성 과정, target 의미, ASOF
필드의 시간 의미, game type과 역할, 선수·타자·팀·카운트·주자·이닝·상황 구조, 연도별 regime
변화, 기존 모델별 잔차와 공식 Public 전이 이력을 처음부터 재감사하라. 그 뒤 1190+에 필요한
**13.24점 이상의 실제 추가 이득**을 만들 수 있는 포트폴리오를 설계하고 실행하라.

탐색 범위는 제한하지 않는다. 새로운 검증 계약, 야구 도메인 기반 확률 분해, 계층적·베이지안
부분풀링, 생성과정/상태공간/시계열 접근, 직접확률·잔차·ranking/stacking, 트리·선형·신경망·표형
딥러닝, multi-task/mixture-of-experts, representation learning, calibration, 모델 증류, 선수 역할과
상황별 전문가, 불확실성 기반 gating, 데이터 표현 재설계, 기존 구성요소의 공정한 재학습·대체,
저상관 앙상블 등은 모두 가능하다. 이는 아이디어 예시일 뿐 우선순위나 필수 목록이 아니다.
저장소 밖의 새로운 합법적 관점도 적극적으로 제안하고 검증하라.

다만 다음 행동은 피하라.

- 한 번 Public에서 관측된 동일 신호의 weight/threshold/route만 조금씩 바꿔 리더보드에 맞추기
- full-2024 한 축의 평균 gain만으로 승격하기
- 서로 다른 연도의 BSS 절대값을 prevalence 보정 없이 직접 비교하기
- 강한 부모와의 paired incremental 성능을 보지 않고 단독 모델 BSS만 비교하기
- 이미 기각된 실험을 새 seed나 사소한 파라미터 변경만으로 반복하기
- 로컬 gain을 근거 없이 1:1 Public gain으로 선언하기

## 검증 체계를 먼저 개선하라

기존 로컬 평가는 방향 판단에는 일부 유효했지만 전이 크기를 자주 과대평가했고, 양수 로컬
후보도 Public에서 역전됐다. `reports/submissions.csv`의 실제 공식 결과를 구조화해 어떤 검증
특성—다중 origin 일관성, 월별 부호, active-row gain, 예측 이동 크기, 투수 bootstrap, 투수×타자
교차 bootstrap, 시간 block, Reality Check, 모델 상관·support—이 실제 Public 전이와 관계가
있는지 정량 재감사하라. 필요하면 기존 full-2024 중심 계약을 폐기하고, 배포 시점과 feature
semantics를 더 정확히 모사하는 다중-origin/nested/leave-season-out 평가를 새로 설계하라.

Public 기록 수가 적고 선택편향이 크므로 메타모델을 과신하지 말고, 불확실성과 실패 가능성을
명시하라. 후보 승격은 최소한 다음을 포함하는 종합 근거로 판단하라.

- 여러 독립 시간 원점에서 같은 방향
- 부모 대비 paired Brier 개선
- 월·도메인·선수 군집의 안정성
- bootstrap/시간블록 하방과 다중탐색 보정
- 기존 champion과의 잔차 상보성 및 실제 예측 이동량
- 2025 배포 feature semantics와 학습 semantics의 일치
- Public 1190 도달 가능성을 설명할 충분한 효과 크기

엄격한 기준 때문에 모든 아이디어를 조기에 버리지 말고, exploratory와 confirmatory 단계를
분리하라. 약한 신호 여러 개를 무작정 누적하지 말고, 독립 근거가 있는 구성요소들의 결합을
held-out 방식으로 다시 검증하라.

## Codex와의 협업 방식

- 시작 즉시 `reports/submissions.csv`, 최신 `reports/target*.md`, `src/champion`, 최근
  `src/archive/v2*.py`, `artifacts/v2*`를 조사해 중복 작업을 피하라.
- Codex가 만든 파일을 덮어쓰지 말고 Claude 전용 버전 번호 또는 명확한 접두사를 사용하라.
- 각 실험은 가설, 데이터 범위, 학습/검증 origin, 부모 SHA/버전, 선택에 사용한 축, 잠금 축,
  결과, 기각/승격 이유를 `summary.json`과 짧은 보고서에 남겨라.
- 중요한 진행 상황을 다음 공유 파일에 원자적으로 갱신하라.
  `reports/claude_codex_coordination_20260830.md`
  여기에는 현재 실행 중 작업, 완료 실험, 생성 artifact, 다음 독립 방향, Codex가 읽어야 할
  사항을 기록하라.
- Codex 측 최신 변경을 주기적으로 다시 읽고, 같은 실험을 동시에 실행 중이면 중복을 중단하고
  상보적인 작업으로 전환하라.
- 사용자가 별도 결정을 내려야 하는 진짜 blocker가 아니라면 질문만 남기고 멈추지 말고 합리적
  가정 아래 계속 진행하라.

## 산출물과 제출 후보 기준

유망 후보마다 다음을 완성하라.

1. 재현 가능한 학습·평가 코드와 고정 config
2. 독립 시간축 및 잠금축 결과와 원시 prediction artifact
3. 행 singleton/shuffle/partition 불변성 검사
4. test 규모 보수적 runtime, 메모리, finite, `[0,1]`, row_id 정렬 검사
5. standalone ZIP, CRC, 파일 수, SHA-256, requirements 고정
6. champion 보호 및 비활성 route exact-parity 검사
7. 왜 이 후보가 기존 실패와 다르고 1190 격차를 메울 가능성이 있는지에 대한 보수적 추정

공식 제출 기회는 제한되어 있으므로 단순 양수 로컬 후보를 제출 권고하지 마라. 독립적인 큰
효과 또는 서로 보완적인 강한 구성요소가 있고, 실패 하방까지 감수할 이유가 있는 후보만 순위를
매겨 제시하라. 단, 완벽한 통계적 확증이 불가능하다는 이유로 연구 자체를 멈추지는 마라.

이제 현재 상태를 직접 재현·감사하고, 넓은 후보군을 병렬 또는 단계적으로 탐색한 뒤 가장 큰
정보가치를 주는 실험부터 실행하라. 1190+가 공식 확인되기 전에는 완료라고 선언하지 말라.

---
