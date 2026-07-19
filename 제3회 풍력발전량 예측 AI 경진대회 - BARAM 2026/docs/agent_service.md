# Competition Scientist 에이전트 서비스

## 목표와 설계 원칙

현재 배포는 BARAM 2026을 대상으로 하지만, 대회별 차이는 `.agents/competition.json`에
격리했다. 새 Kaggle·DACON 대회에서는 대회 프로필, 검증 정책 어댑터, 플랫폼 제출
어댑터만 교체하고 실험 트리·작업 큐·감사 로그는 그대로 재사용하는 구조다.

여러 LLM이 자유롭게 대화하는 시스템이 아니라 하나의 결정론적 오케스트레이터가 역할
작업을 순서대로 라우팅한다. LLM은 연구와 가설·코드 제안을 담당하고, 상태 전이와 실행,
검증, 제출은 Python 정책이 담당한다.

```text
Rule Parser -> Data Analyst -> Validation approval
                     |
Research -> Experiment Planner -> Modeling -> local safe runner
                                      |
                               Validation -> Critic
                                      |
                  local_best ----------+---------- submission_candidate
                                                          |
                                       Submission Guard -> platform adapter
                                                          |
                                      score sync -> select/archive -> next branch
```

## 반영한 공개 레퍼런스

- [AIDE](https://github.com/WecoAI/aideml)의 코드 탐색 트리에서 부모·자식 실험과 실패
  분기 보존을 가져왔다. 단, 임의 코드를 무제한 실행하지 않고 allowlist Python 모듈과
  단일 변경 계약으로 제한했다.
- [OpenAI MLE-bench](https://github.com/openai/mle-bench)의 대회 단위 평가, 자원 예산,
  반복 실행 관점을 반영했다. 논문의 75개 Kaggle 평가와 16.9% bronze 결과는 완전
  자율화보다 검증·선택 안전장치가 중요하다는 근거로 사용했다.
- [MLAgentBench](https://github.com/snap-stanford/MLAgentBench)의 관찰-가설-수정-실행-해석
  루프를 역할 작업 큐로 구현했다.
- [DSBench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/50e9ad960ae78b741a6b4fea533f2eaf-Abstract-Conference.html)와
  [AgentDS](https://arxiv.org/abs/2603.19005)의 한계를 반영해 검증 전략과 규정 예외는
  명시적 승인 대상으로 뒀다.
- [DrivenData의 실제 대회 에이전트 실험](https://drivendata.co/blog/ai-agents-data-science-competitions)에서
  최고 실행과 에이전트가 고른 최종 실행이 달랐던 사례를 반영해 `local_best`와
  `submission_candidate`를 서로 다른 선택 상태로 저장한다.
- 장기 실행 상태와 사람 승인 흐름은 [LangGraph](https://github.com/langchain-ai/langgraph),
  실험 메타데이터 관점은 [MLflow](https://github.com/mlflow/mlflow), HPO 이력·조기 중단은
  [Optuna](https://github.com/optuna/optuna)를 참고했다. 현재 코어는 의존성 부담을 줄이기
  위해 SQLite와 표준 라이브러리 상태 머신으로 구현되어 있고, 이후 이들 도구와 연결할 수 있다.

## 현재 구현된 안전장치

- 대회 규칙, ID/타깃, 허용 범위, 시간·그룹 컬럼을 대회 프로필로 구조화한다.
- 승인된 검증 계획 ID가 없는 실험 등록을 차단한다.
- 자식 실험은 `parent_run_id`와 한 가지 `change_summary`를 필수로 남긴다.
- 실행은 `experiments.*` allowlist 모듈만 shell 없이 수행한다.
- 입력·출력 SHA-256, Git 상태, Python 환경, stdout/stderr, 재시도 시도를 기록한다.
- fold 점수, 평균·표준편차, OOF 경로, 시간·메모리, 누수·규정 위험을 평가 계약에 저장한다.
- `local_best`와 `submission_candidate`를 별도로 관리한다.
- 후보 CSV는 sample과 컬럼·행·ID 순서가 정확히 같고, 값이 finite·허용 범위일 때만 통과한다.
- 동일 파일 해시 재제출, 일일/전체 예산, 최소 제출 간격을 차단한다.
- 풍력 배포는 자동 제출이 설정되어 있지만 로컬 execute 플래그와 환경 자격증명이 모두
  없으면 외부 API를 호출하지 않는다.
- public 결과는 주 목적함수가 아니라 실패 family의 다음 탐색 범위를 줄이는 보조 신호로만 쓴다.
- 공개 실패에는 구체 모델명과 함께 일반화된 `family_group`, 변경 `direction`, 실제 변경 행
  비율을 저장한다. 같은 계열·같은 방향의 다음 후보는 실패 변경률의 25% 이하로 범위를
  축소하지 않으면 자동 승격하지 않는다. 반대 방향이나 구조적으로 다른 모델은 이 근거만으로
  차단하지 않으며, 별도의 로컬 검증을 그대로 통과해야 한다.

## 최초 초기화

PowerShell 기준:

```powershell
python -m agent_service init
python -m agent_service status
python -m agent_service validation-add .agents/examples/baram_validation_plan.json
python -m agent_service approval-record .agents/examples/approve_baram_validation.json
python -m agent_service public-record .agents/examples/public_1494307.json
python -m agent_service public-record .agents/examples/public_1494535.json
python -m agent_service leaderboard-sync
python -m agent_service task-add research .agents/examples/next_research_task.json
```

검증 계획 승인 예제의 `subject_id`는 새 DB에서 첫 계획이 1인 경우다. 기존 DB에서는
`python -m agent_service list validation_plans`로 실제 ID를 확인한 뒤 JSON을 수정한다.

## 실험 루프

```powershell
python -m agent_service hypothesis-add .agents/examples/phase_regime_hypothesis.json
python -m agent_service run-register .agents/examples/phase_regime_run.json
python -m agent_service run-execute 1
python -m agent_service run-evaluate 1
python -m agent_service tree
python -m agent_service list decisions
python -m agent_service list selections
```

실제 자식 실험 JSON에는 기존 run의 `parent_run_id`와 한 가지 `change_summary`를 넣는다.
평가 결과가 수치 게이트를 통과하면 `local_best`는 객관적 측정으로 갱신된다. 현재 자동화
배포에서는 통과 run이 `submission_candidate`로도 선택되지만, 사람 승인 배포에서는
`run-select`를 별도로 실행해야 한다.

`rejected` 결과는 `local_best`를 갱신하지 않는다. 과거 선택을 무효화할 때는 DB 행을
삭제하지 않고 비활성화해 `selection.deactivated` 감사 이벤트와 사유를 남긴다.

## 자동 제출과 결과 피드백

```powershell
$env:DACON_API_TOKEN = '<token>'
$env:DACON_TEAM_NAME = '<team name>'
python -m agent_service auto-cycle
python -m agent_service auto-cycle --execute-submissions
```

첫 명령은 항상 dry-run 검사다. 두 번째만 외부 제출을 허용한다. 공식 DACON 제출 API는
파일 전송 성공 여부는 반환하지만 채점 결과 조회 기능은 제공하지 않으므로, 결과가 나온 뒤
`submissions/results.csv`에 행을 추가하고 `leaderboard-sync`를 실행한다. 제출 성공 후 생성된
`leaderboard` 작업을 별도 worker가 완료하는 방식도 가능하다.

서버 모드는 기본적으로 localhost이며 외부 바인딩에는 `BARAM_AGENT_TOKEN` bearer token이
필수다. HTTP `/v1/auto-cycle`은 의도적으로 실제 제출을 수행할 수 없다.

```powershell
python -m agent_service serve --host 127.0.0.1 --port 8765
```

대시보드용 읽기 API는 `/v1/status`와 `/v1/tree`를 제공한다. 전자는 run/task 상태,
현재 선택, public best, 자동 제출 설정을 한 번에 반환한다.

## 새 대회로 확장

1. 대회 루트에 `.agents/competition.json`을 작성한다.
2. sample submission, ID/타깃 범위, 시간·그룹 구조, 외부 데이터와 API 규정을 명시한다.
3. EDA가 제안한 검증 계획을 등록하고 사람이 한 번 승인한다.
4. 대회 metric을 표준 `Evaluation`으로 바꾸는 adapter를 추가한다.
5. 플랫폼 제출 adapter를 연결한다. DACON adapter는 구현되어 있고 Kaggle adapter는 아직 없다.
6. 리더보드 점수 없이 OOF 기준으로 실험 트리를 탐색한 후 제출 예산 안에서 probe한다.

원본 행을 원격 LLM에 보내는 기능은 제공하지 않는다. 규정이 확인되기 전에는 스키마와 집계
통계만 연구·계획 에이전트에 전달한다.
