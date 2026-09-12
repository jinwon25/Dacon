# 데이콘 공모전 저장소 — 공통 지침

## 출력 언어

**사용자에게 보이는 모든 텍스트는 한국어로 쓴다.** 여기에는 진행 상황 보고, 중간 요약,
결과 해설, 계획, 질문이 모두 포함된다. 터미널에 영어 문장이 그대로 뜨지 않게 한다.

한국어로 쓰지 않는 것(그대로 둔다):
- 코드, 변수·함수·파일 경로, 커밋 메시지, 로그 원문
- 대회 공식 용어·지표명(`Brier Skill Score`, `SMAPE`, `OOF`, `LB`, `BSS`, `R-Hit@1cm` 등)
- 라이브러리·모델 이름(LightGBM, CatBoost, XGBoost 등)

기술 용어는 억지로 번역하지 말고 원어를 그대로 쓰되, 문장 자체는 한국어로 만든다.
예: "full-2024 축에서 paired Brier 이득이 +4.51이지만 월별 부호가 6/8입니다."

## 서브에이전트 사용

`.claude/agents/`에 4개가 정의되어 있다. 기본적으로 사용자가 요청할 때만 띄우되,
아래 상황에는 먼저 제안하라.

| 에이전트 | 모델 | 언제 |
|---|---|---|
| `repo-scout` | haiku | 파일·산출물 위치만 찾으면 될 때 |
| `exp-log-reader` | sonnet | 여러 run의 지표를 표로 모을 때 |
| `ml-strategist` | opus | 피처 설계·CV 설계·리크 진단·블렌딩 전략 |
| `codex-second-opinion` | sonnet→Codex MCP | 중요한 판단의 교차 검증 |

**대회마다 저장소 레이아웃이 다르다.** 에이전트에 작업을 넘길 때는 해당 대회의
디렉터리 규칙을 프롬프트에 명시해서 넘겨라. 아래 "대회별 레이아웃" 참조.

## 대회별 레이아웃

- **투구 제구 성공 확률 예측** (`.git-worktrees/pitch-control-target-1180/`)
  - 실험 산출물: `artifacts/<vNNN>_<이름>_<날짜>_01/summary.json`
  - 공식 제출 이력: `reports/submissions.csv` (권위 있는 단일 출처)
  - 보고서: `reports/*.md`, 챔피언 코드: `src/champion/`, 실험 코드: `src/archive/vNNN_*.py`
  - `models/`·`logs_*.log`·`docs/WORK_LOG.md` 는 **없다**
- **모기 / BARAM / 그 외**
  - `models/<run>/run_metadata.json`, `docs/reports/`, 저장소 루트의 `logs_*.log`

## 검증 원칙 (전 대회 공통)

- OOF 개선이 곧 LB 개선이 아니다. **변환률**(로컬 이득 대비 실제 LB 이득)을 항상 같이 본다.
- 신규 앙상블 멤버의 레버는 "직교성 AND 준수한 OOF"다. 상관계수를 실제로 계산해 근거로 댄다.
- pseudo-labeling은 기본적으로 거절한다. 제안하려면 왜 이번엔 다른지를 먼저 입증한다.
