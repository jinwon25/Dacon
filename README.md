# 투구 제구 성공 확률 예측

LG Aimers 9기 `투구 제구 성공 확률 예측 AI 온라인 해커톤`의 3인 팀 연구 저장소다.
공식 데이터만 사용해 각 투구의 `P(control_success = 1)`을 예측했고, 시간 순서와
평가 행 독립성 규칙을 지키는 것을 성능보다 우선했다. 대회는 종료됐으며, 이 저장소는
최종 제출물과 그에 이르기까지의 실험 기록을 보존한다.

## 최종 결과

| 항목 | 값 |
|---|---:|
| 최종 Public 챔피언 | **submit_v345 / 1182.94969702** |
| 제출 ID | `78710` |
| 제출 시각 | `2026-08-31 22:17:29 KST` |
| 직전 v335 대비 | **+1.2396938587** |
| 챔피언 파일 | `submissions/releases/v345/submit_v345.zip` |
| SHA-256 | `D44578DC50220CE84DD4B8489BBAE680AFDCF93F931ED4236287B5A9E6F5AAAA` |
| ZIP 크기 | `101,139,163 bytes` |
| 공식 runtime | `130초` |
| 전체 테스트 | `741 passed, 20 skipped` |

v345는 v335가 보존하던 `R_CORE`에 선수 전이, 3시드 workload-H1, 고정 Beta 셀을
추가한 row-local 패키지다. `F`와 `R_ANCHOR`는 v335와 동일하다. 상세 근거와 감사
결과는 [`research/reports/v345_public_result_20260831.md`](research/reports/v345_public_result_20260831.md)에 있다.

Private Score 재현용 최종 제출 패키지는 [`final_submission/`](final_submission/)에 있다.

## 가장 먼저 할 일

```powershell
git clone https://github.com/jinwon25/pitch-control-probability.git
cd pitch-control-probability
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

공식 DACON 원본은 저장소에 포함하지 않는다. 팀원이 직접 내려받아 다음 위치에 둔다.

```text
data/train.csv
data/trackman_history.csv
data/test.csv
data/sample_submission.csv
```

파일 스키마와 해시는 [`data/README.md`](data/README.md)를 확인한다. 현재 운영 상태는
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md), 장기 인계 메모는
[`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md)가 단일 원본이다.

## 최신 챔피언은 단일 ZIP으로 실행되는가

그렇다. 확정 `submit_v335_anchor_lowrank_complement.zip`은 루트에 아래 세 항목만 가진
독립 릴리스다.

```text
submit_v335_anchor_lowrank_complement.zip
├─ script.py
├─ requirements.txt
└─ model/                 # v290 계보, F 전문가와 frozen low-rank lookup 포함
```

실행할 때 v104·v124·v142 같은 과거 ZIP, 저장소의 `src/`, 학습 코드, 인터넷 연결을
요구하지 않는다. `script.py`는 자기 ZIP의 `model/`만 읽고, 공식 입력 두 파일만
`data/`에서 읽어 `output/submission.csv`를 만든다.

### 독립 실행

아래 예시는 챔피언 ZIP 하나만 깨끗한 폴더에 풀어 실행한다.

```powershell
$release = Join-Path $PWD "run_v335"
New-Item -ItemType Directory -Force $release | Out-Null
Expand-Archive `
  submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip `
  $release -Force
New-Item -ItemType Directory -Force (Join-Path $release "data") | Out-Null
Copy-Item data/test.csv (Join-Path $release "data/test.csv")
Copy-Item data/sample_submission.csv (Join-Path $release "data/sample_submission.csv")
python -m pip install -r (Join-Path $release "requirements.txt")
python (Join-Path $release "script.py")
```

성공하면 `run_v335/output/submission.csv`가 생성된다. 확정
패키지 의존성은 ZIP 내부 `requirements.txt`에 고정돼 있다.

```text
lightgbm==4.6.0
catboost==1.2.8
joblib==1.5.1
numpy==2.2.6
pandas==2.2.3
scikit-learn==1.6.1
xgboost==3.2.0
```

### 단일 릴리스 감사

다음 명령은 ZIP을 임시 디렉터리에 풀고 **그 안의 코드와 모델만으로** 구조·CRC·출력
스키마·확률 범위·셔플/분할 행 독립성·전체 규모 추론을 검사한다.

```powershell
python -m src.archive.audit_standalone_release `
  --package submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip `
  --test-csv data/test.csv `
  --scale-rows 245789 `
  --timeout-seconds 300
```

v335의 route별 감사는 로컬 생성 artifact에 보존했다. 제출 전·공식 실행 결과는 다음과 같다.

| 검사 | 결과 |
|---|---:|
| 구조 감사 | 통과 |
| v320 대비 F parity | `0.0` |
| v320 대비 R_CORE parity | `0.0` |
| R_ANCHOR 공식 최대 오차 | `0.0` |
| shuffle / partition 최대 오차 | `0.0` / `5.55e-17` |
| DACON 공식 추론 | `111초` |
| 공식 제한 | `600초` |

내부 120초는 후보 간 비교용 soft guard이며 대회 공식 제한은 600초다.

ZIP 실행과 소스 재구축은 다른 작업이다. 실행에는 ZIP 하나면 충분하지만, 동일 ZIP을
처음부터 다시 만들려면 v290 부모, v319 low-rank lookup, 공식 train과 forward OOF가
필요하다. 핵심 재구축·감사 진입점은 다음과 같다.

```powershell
python -m src.champion.v319_finalize_futures_lowrank --help
python -m src.champion.v320_build_futures_portfolio_package --help
python -m src.archive.v335_anchor_lowrank_complement_audit --help
python -m src.champion.v335_build_anchor_lowrank_package --help
python -m src.archive.audit_v335_anchor_lowrank --help
```

공식 원본 데이터, 인증정보, 그리고 대회 데이터에서 파생된 모든 산출물은 Git에 올리지
않는다. 제출 ZIP·OOF 번들·lookup·선수 연결표가 여기에 해당한다. OOF에는 train target과
선수 ID가, 제출 ZIP에는 선수별 집계 prior가 들어 있어 공개 배포는 대회 데이터
재배포에 해당하기 때문이다. 이 산출물들은 공식 DACON 팀원에게만 별도 경로로 전달하며,
규칙과 해시는 [`docs/ARTIFACT_HANDOFF.md`](docs/ARTIFACT_HANDOFF.md)와
[`docs/STANDALONE_CHAMPION.md`](docs/STANDALONE_CHAMPION.md)에 있다. 저장소에는 각
산출물의 manifest만 남겨 무엇이 어디에 있는지 추적할 수 있게 한다.

## 문제와 평가

한 행은 한 번의 투구 상황이다. 목표는 정답 클래스를 자르는 것이 아니라 성공 확률을
정확히 보정하는 것이다. 대회 평가는 Brier Skill Score 계열이며 Brier Score가 낮을수록
Public 점수는 높다.

```text
Brier Score = mean((예측 확률 - 실제 정답)^2)
BSS = max(0, 100000 × (1 - 모델 Brier Score / 기준 Brier Score))
```

극단적으로 자신 있는 오답의 손실이 크므로, 최신 시즌에 과적합한 단일 고용량 모델보다
강한 부모를 보존하면서 검증된 하위 영역에만 작은 보정을 더하는 전략을 사용했다.

## 최종 모델 계보

```text
v22  도메인 calibration + row-local 선수 ASOF prior
  → v25/v26  R_ANCHOR post-break 보정
    → TrackMan-ASOF gate (1158.0746)
      → v82  EXP-021 strict, R_CORE 10% (1159.3352)
        → v84  shared pairwise FM, F 10% (1161.2021)
          → v104 source-stable R_CORE (1162.6303)
            → v124 Public-quadratic stack (1164.2949)
              → v142 H1 + sign-stable C3 (1169.6278)
                → v148 conservative bridge (1170.3015)
                  → v167 fixed original H1 affine (1172.0772)
                    → jy_runners_high_li_bridge027 (1172.1374)
                      → v244 mechanism-aware fallback (1175.9747)
                        → v290 exact-anchor + recent F (1176.7571)
                          → v320 F direct + low-rank portfolio (미제출)
                            → v335 R_ANCHOR low-rank complement (1181.7100)
```

계보는 연구 설명용이며 실행 의존성 체인이 아니다. 최신 champion ZIP 안에는 최종 추론에
필요한 구성요소가 모두 들어 있다.

### 최신 v335 champion의 핵심

- `R_CORE`: v290의 exact-anchor/fallback 계보를 수치적으로 보존한다.
- `R_ANCHOR`: team 13이 관여한 정규시즌 행에 frozen `lowrank_s300_r2`를 0.50 추가한다.
- `F`: 최근 5-seed direct expert를 총 20% 혼합하고 같은 low-rank 신호를 0.50 추가한다.
- low-rank는 투수 ID와 현재 count·타자손을 행별로 매핑하며 test 집계를 사용하지 않는다.
- 중간 v320을 제출하지 않았으므로 Public 개선을 F와 R_ANCHOR에 분해해 귀속하지 않는다.

패키지 manifest에 고정된 최종 수식은 다음과 같다.

```text
F = game_type == "F"
R_ANCHOR = game_type == "R" and (pitcher_team_id == 13 or batter_team_id == 13)
output[F] = v290_pre_F + 0.20 × (recent_F - v290_pre_F) + 0.50 × lowrank
output[R_ANCHOR] = v320[R_ANCHOR] + 0.50 × lowrank
output[R_CORE] = v290[R_CORE]
```

모델 변화, 로컬 증분과 실제 Public 전이는
[`research/reports/v335_public_result_20260831.md`](research/reports/v335_public_result_20260831.md)에 정리돼 있다.

## 과거 JY 순차 게이트 연구

v167은 Public `1172.0772380321`의 기준 main submit으로 고정했다. 이후 같은 계열을
무작정 넓히지 않고, 제출권을 아끼기 위해 `runners_on`에서 시작해 순차적으로 추가 여부를
판단했다.

첫 제출 `submit_jy_runners_gate.zip`은 `R_CORE` 중 `num_runners_on > 0`인 행에만
bridge025/H1/C3 recent 조합을 덮어썼고 Public `1172.1199939043`을 기록했다. 기준 대비
`+0.0427558722`라 row-local game-state gate 방향이 맞는 것으로 판단했다.

두 번째 제출 `submit_jy_runners_high_li_bridge027.zip`은 그 위에 `li >= 1.5` high-leverage
행만 추가하고, pressure-count 확장은 제외했다. bridge025 방향은 `1.2`배로 키워 effective
bridge를 만들었다. 순차 감사에서 `runners_or_high_li_bridge027`은 locked 2024 gain
`0.638287`로 grid 내 최고였고, 패키지 구조 감사에서 보호 행 최대 차이 `0.0`을 통과했다.
실제 Public은 `1172.1373858439`로 v167 대비 `+0.0601478118`, 직전 runners 후보 대비
`+0.0173919396` 개선됐다.

상세 근거는
[`research/reports/jy_runners_high_li_bridge027_public_result_20260828.md`](research/reports/jy_runners_high_li_bridge027_public_result_20260828.md),
순차 감사 표는
[`research/reports/jy_runners_high_li_bridge027_sequential_gate_audit.csv`](research/reports/jy_runners_high_li_bridge027_sequential_gate_audit.csv)에 있다.

## 누수 방지와 승격 규칙

- 평가 시즌보다 미래인 target을 특징·보정값·가중치 선택에 사용하지 않는다.
- test 전체의 평균, 순위, 빈도, 분포, 순서나 그룹 통계를 예측에 사용하지 않는다.
- 각 test 행은 그 행의 입력과 공식 train으로 미리 고정한 모델만 사용한다.
- calibration과 blend는 시간 순서를 지킨 OOF에서 결정한다.
- 2024 label은 반복 연구에 사용됐으므로 `development_contaminated`로 취급한다.
- Public 점수로 같은 계열의 강도나 route를 사후 미세 조정하지 않는다.
- 새 후보는 [`research/configs/evaluation_v3.json`](research/configs/evaluation_v3.json)의 다중 시간축,
  월·domain, crossed/block bootstrap, Reality Check와 패키지 감사를 통과해야 한다.

행 순서나 배치 크기를 바꿔도 같은 행의 예측이 유지돼야 한다. 공식 규칙과 구현 근거는
[`docs/REFERENCE_MAP.md`](docs/REFERENCE_MAP.md)를 따른다.

## 재현과 검증

빠른 코드 검증:

```powershell
python -m pytest -q
python scripts/audit_repository.py --include-untracked
git diff --check
```

챔피언 수치 계약과 패키징 회귀 테스트:

```powershell
python -m pytest -q `
  tests/test_v124_public_quadratic_stack.py `
  tests/test_v154_maturity_delta.py `
  tests/test_audit_standalone_release.py
```

최신 JY champion 연구·패키징 재현의 진입점:

```powershell
python scripts\build_jy_next_variants.py
python scripts\audit_jy_sequential_gates.py
python scripts\audit_jy_next_final.py
```

최종 제출 이력과 해시는 [`research/reports/submissions.csv`](research/reports/submissions.csv)에 기록한다.

## 폴더 구조

```text
final_submission/   Private Score 재현용 최종 제출 패키지 (v345)
  code/             학습 코드와 고정 입력
  inference/        리더보드에 제출한 추론본
  verification/     구성요소별 검증 결과
src/
  champion/         실제 제출된 버전으로 이어지는 계보
  core/             축·계약·패키징 공통 코드
  team_assets/      팀원 고정 자산 (JY fallback XGB)
  archive/          종료된 실험 모듈
tests/
  champion/ core/ archive/   src 구조를 그대로 반영
research/
  reports/          실험 결과와 제출 원장
  configs/          동결된 실험 및 패키징 계약
submissions/releases/   해시 고정된 확정 제출 ZIP
docs/               프로젝트 현황·인계·규칙
notebooks/          탐색 워크벤치와 챔피언 재현 노트북
scripts/            저장소 보조 감사
data/               공식 원본 데이터, Git 제외
artifacts/          모델·OOF·패키지 산출물, 원칙적으로 Git 제외
.github/            협업·CI 설정
```

읽는 순서는 이 README → [`final_submission/README.md`](final_submission/README.md) →
[`src/champion/README.md`](src/champion/README.md)를 권한다. 앞의 둘이 최종
결과물이고, 세 번째가 거기에 이르는 코드 계보다.

과거 v11~v167의 상세 실험과 실패 기록은 삭제하지 않고 `research/reports/`와
[`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md)에 보존한다. 현재 판단에는 최신 상태 문서와
JY champion 최종 보고서를 우선한다.

## 주요 문서

| 문서 | 용도 |
|---|---|
| [`final_submission/README.md`](final_submission/README.md) | 최종 제출 패키지 구성과 검증 범위 |
| [`final_submission/03_REPRODUCE.md`](final_submission/03_REPRODUCE.md) | v345 재현 절차 |
| [`src/champion/README.md`](src/champion/README.md) | 제출된 버전으로 이어지는 코드 계보 |
| [`research/reports/v345_public_result_20260831.md`](research/reports/v345_public_result_20260831.md) | 최종 챔피언 v345 승격 근거 |
| [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | 대회 기간 중 운영 기준 |
| [`docs/STANDALONE_CHAMPION.md`](docs/STANDALONE_CHAMPION.md) | 단일 ZIP 실행·감사·전달 |
| [`research/reports/v335_public_result_20260831.md`](research/reports/v335_public_result_20260831.md) | 1180 달성과 v335 승격 근거 |
| [`research/reports/jy_runners_high_li_bridge027_public_result_20260828.md`](research/reports/jy_runners_high_li_bridge027_public_result_20260828.md) | 1172.137 갱신 근거 |
| [`research/reports/target1180_v167_public_result_20260825.md`](research/reports/target1180_v167_public_result_20260825.md) | v167 기준선 근거 |
| [`research/reports/target1170_v142_v148_public_result_20260823.md`](research/reports/target1170_v142_v148_public_result_20260823.md) | 직전 1170 달성 근거 |
| [`research/reports/target1173_v149_v154_final_submission_20260823.md`](research/reports/target1173_v149_v154_final_submission_20260823.md) | 마지막 제출 연구와 비승격 판정 |
| [`research/reports/submissions.csv`](research/reports/submissions.csv) | 제출 ID·점수·해시 원장 |
| [`submissions/README.md`](submissions/README.md) | 제출 파일 보관 규칙과 계보 |
| [`docs/EXPERIMENT_WORKFLOW.md`](docs/EXPERIMENT_WORKFLOW.md) | 공통 실험·검증 절차 |
| [`docs/GITHUB_START_GUIDE.md`](docs/GITHUB_START_GUIDE.md) | 팀원 GitHub 시작 안내 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 브랜치·리뷰·커밋 규칙 |

## Git·데이터 정책

- 원본 DACON 데이터, 인증정보, 개인 쿠키·키는 Git과 LFS에 올리지 않는다.
- 대회 데이터에서 파생된 산출물(모델 ZIP·OOF·lookup·선수 연결표)은 예외 없이 Git에서
  제외한다. 저장소가 public이므로 해시 고정 예외도 두지 않는다.
- 이 정책은 `scripts/audit_repository.py`가 강제하며 `tests/test_repository_audit.py`가
  검증한다. 새 산출물을 추적하려면 먼저 그 파일이 대회 데이터를 담고 있지 않은지 확인한다.
- `main`은 검증된 기준선으로 유지하고 새 연구는 별도 브랜치에서 수행한다.
- 챔피언을 덮어쓰지 않고 버전별 새 디렉터리에 패키징한다.
- DACON 제출은 검증된 단일 ZIP 하나를 지정 담당자가 수행하고 결과를 즉시 원장에 남긴다.

대회 종료 시점의 공식 기준은 v345 Public **1182.94969702**이며, 저장소 정리 후
전체 테스트는 로컬에서 `741 passed, 20 skipped`로 재현된다.
