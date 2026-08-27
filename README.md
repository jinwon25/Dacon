# 투구 제구 성공 확률 예측

LG Aimers 9기 `투구 제구 성공 확률 예측 AI 온라인 해커톤`의 팀 연구 저장소다.
공식 데이터만 사용해 각 투구의 `P(control_success = 1)`을 예측하며, 시간 순서와
평가 행 독립성 규칙을 지키는 것을 성능보다 우선한다.

## 최종 현황

마지막 갱신: `2026-08-28 03:45 KST`

| 항목 | 값 |
|---|---:|
| 공식 Public 챔피언 | **jy_runners_high_li_bridge027 / 1172.1373858439** |
| 공개 순위 | 제출 이력 기준 확인 |
| 제출 ID | DACON UI 확인값 미기록 |
| 목표 1170 초과분 | **+2.1373858439** |
| 직전 v167 대비 | **+0.0601478118** |
| 1180까지 | **7.8626141561** |
| 챔피언 파일 | `submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip` |
| SHA-256 | `4C924E046091304BFC73B50BE51110BDF1351DFD9B577738CC6B65A8DFF43C9E` |
| ZIP 크기 / 파일 수 | `83,675,308 bytes` / `172` |
| 현재 CI 기준 검사 | data-free pytest `57 passed` 로컬 재현 |

최신 챔피언은 v167/main submit을 기준으로, `R_CORE` 중 `num_runners_on > 0` 또는
`li >= 1.5`인 행에만 bridge025를 약 `0.27` 강도로 키우고 H1/C3 recent 조합을 덮어쓴
row-local 후속 패키지다. 다른 행은 기준 main submit과 수치적으로 동일하게 보존한다.
2026-08-28 03:38:21 KST 제출에서 Public `1172.1373858439`를 확인했다. 상세 근거와
재현 절차는
[`reports/jy_runners_high_li_bridge027_public_result_20260828.md`](reports/jy_runners_high_li_bridge027_public_result_20260828.md)에 있다.

## 가장 먼저 할 일

```powershell
git clone https://github.com/Lg-Aimers-chungang/hackathon.git
cd hackathon
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

그렇다. 확정 `submit_jy_runners_high_li_bridge027.zip`은 루트에 아래 세 항목만 가진
독립 릴리스다.

```text
submit_jy_runners_high_li_bridge027.zip
├─ script.py
├─ requirements.txt
└─ model/                 # v124 부모, H1, C3와 모든 하위 모델 포함
```

실행할 때 v104·v124·v142 같은 과거 ZIP, 저장소의 `src/`, 학습 코드, 인터넷 연결을
요구하지 않는다. `script.py`는 자기 ZIP의 `model/`만 읽고, 공식 입력 두 파일만
`data/`에서 읽어 `output/submission.csv`를 만든다.

### 독립 실행

아래 예시는 챔피언 ZIP 하나만 깨끗한 폴더에 풀어 실행한다.

```powershell
$release = Join-Path $PWD "run_jy_runners_high_li_bridge027"
New-Item -ItemType Directory -Force $release | Out-Null
Expand-Archive `
  submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip `
  $release -Force
New-Item -ItemType Directory -Force (Join-Path $release "data") | Out-Null
Copy-Item data/test.csv (Join-Path $release "data/test.csv")
Copy-Item data/sample_submission.csv (Join-Path $release "data/sample_submission.csv")
python -m pip install -r (Join-Path $release "requirements.txt")
python (Join-Path $release "script.py")
```

성공하면 `run_jy_runners_high_li_bridge027/output/submission.csv`가 생성된다. 확정
패키지 의존성은 ZIP 내부 `requirements.txt`에 고정돼 있다.

```text
lightgbm==4.6.0
catboost==1.2.8
joblib==1.5.1
numpy==2.2.6
pandas==2.2.3
scikit-learn==1.6.1
```

### 단일 릴리스 감사

다음 명령은 ZIP을 임시 디렉터리에 풀고 **그 안의 코드와 모델만으로** 구조·CRC·출력
스키마·확률 범위·셔플/분할 행 독립성·전체 규모 추론을 검사한다.

```powershell
python -m src.audit_standalone_release `
  --package submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip `
  --test-csv data/test.csv `
  --scale-rows 245789 `
  --timeout-seconds 300
```

최종 후보의 구조 감사는 `reports/jy_runners_high_li_bridge027_package_audit.json`에
보존했다. 제출 전 probe 감사 결과는 다음과 같다.

| 검사 | 결과 |
|---|---:|
| 구조 감사 | 통과 |
| 기준 main submit 대비 보호 행 최대 차이 | `0.0` |
| 후보 active fraction, probe | `0.625` |
| 이전 runners 후보 active fraction, probe | `0.375` |
| 새로 열린 fraction, probe | `0.25` |
| 3,600행 probe 추론 | `9.723초` |
| 공식 제한 | `600초` |

내부 120초는 후보 간 비교용 soft guard이며 대회 공식 제한은 600초다.

ZIP 실행과 소스 재구축은 다른 작업이다. 실행에는 ZIP 하나면 충분하지만, 동일 ZIP을
처음부터 다시 만들려면 해시가 고정된 부모 artifact와 공식 train/OOF가 필요하다.
재구축 진입점은 `scripts/build_jy_next_variants.py`이며, 순차 감사와 최종 패키지 감사는
아래 스크립트로 재현한다.

```powershell
python scripts\build_jy_next_variants.py
python scripts\audit_jy_sequential_gates.py
python scripts\audit_jy_next_final.py
```

공식 원본 데이터, 인증정보와 일반 대용량 ZIP·OOF는 Git에 올리지 않는다. 다만 PRIVATE
협업 저장소에서 팀원에게 전달하기 위해 검증된 v167 ZIP과 최신 JY champion ZIP은 해시
고정 예외로 명시 추적한다. clone 후 SHA-256을 대조하고, 기존 1161 LFS 및 1170 OOF 번들은
문서화된 해시 고정 allowlist로 유지한다. 실행 및 감사는 위 절차로 수행한다. 자세한 인계 규칙은
[`docs/STANDALONE_CHAMPION.md`](docs/STANDALONE_CHAMPION.md)에 있다.

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
```

계보는 연구 설명용이며 실행 의존성 체인이 아니다. 최신 champion ZIP 안에는 최종 추론에
필요한 구성요소가 모두 들어 있다.

### 최신 JY champion의 핵심

- `R_CORE`: 1군 정규시즌 중 익명 anchor team 13이 관여하지 않는 안정 영역이다.
- `R_ANCHOR`: team 13이 관여한 정규시즌 영역으로, 불안정한 변화점을 보호한다.
- `F`: 퓨처스리그 영역이며 별도 source-balanced FM 계보를 사용한다.
- v167/main submit을 기준 출력으로 먼저 재현한다.
- `R_CORE` 중 `num_runners_on > 0` 또는 `li >= 1.5`인 행만 추가 보정한다.
- active 행에서는 bridge025 방향을 `1.2`배로 키워 effective bridge를 만들고,
  H1 `0.16`, C3 recent `0.25` 조합을 적용한다.
- active가 아닌 행은 기준 main submit과 수치적으로 동일하게 보존한다.

패키지 manifest에 고정된 최종 수식은 다음과 같다.

```text
active = R_CORE and (num_runners_on > 0 or li >= 1.5)
effective_bridge = parent + 1.2 × (bridge025 - parent)
output[active] = 0.84 × effective_bridge + 0.16 × H1 + 0.50 × C3_recent025
output[~active] = main submit output
```

모델 변화, 로컬 증분과 실제 Public 전이는
[`reports/jy_runners_high_li_bridge027_public_result_20260828.md`](reports/jy_runners_high_li_bridge027_public_result_20260828.md)에 정리돼 있다.

## 최신 JY 순차 게이트 연구

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
[`reports/jy_runners_high_li_bridge027_public_result_20260828.md`](reports/jy_runners_high_li_bridge027_public_result_20260828.md),
순차 감사 표는
[`reports/jy_runners_high_li_bridge027_sequential_gate_audit.csv`](reports/jy_runners_high_li_bridge027_sequential_gate_audit.csv)에 있다.

## 누수 방지와 승격 규칙

- 평가 시즌보다 미래인 target을 특징·보정값·가중치 선택에 사용하지 않는다.
- test 전체의 평균, 순위, 빈도, 분포, 순서나 그룹 통계를 예측에 사용하지 않는다.
- 각 test 행은 그 행의 입력과 공식 train으로 미리 고정한 모델만 사용한다.
- calibration과 blend는 시간 순서를 지킨 OOF에서 결정한다.
- 2024 label은 반복 연구에 사용됐으므로 `development_contaminated`로 취급한다.
- Public 점수로 같은 계열의 강도나 route를 사후 미세 조정하지 않는다.
- 새 후보는 [`configs/evaluation_v3.json`](configs/evaluation_v3.json)의 다중 시간축,
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

최종 제출 이력과 해시는 [`reports/submissions.csv`](reports/submissions.csv)에 기록한다.

## 폴더 구조

```text
.github/       협업·CI 설정
configs/       동결된 실험 및 패키징 계약
data/          공식 원본 데이터, Git 제외
docs/          프로젝트 현황·인계·규칙
notebooks/     과거 및 탐색 워크벤치
reports/       실험 결과와 제출 원장
scripts/       저장소 보조 감사
src/           학습·평가·패키징·추론 구현
submissions/   제출 계보와 PRIVATE 전달용 확정 ZIP
tests/         데이터 비의존 및 로컬 통합 테스트
artifacts/     모델·OOF·패키지 산출물, 원칙적으로 Git 제외
```

과거 v11~v167의 상세 실험과 실패 기록은 삭제하지 않고 `reports/`와
[`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md)에 보존한다. 현재 판단에는 최신 상태 문서와
JY champion 최종 보고서를 우선한다.

## 주요 문서

| 문서 | 용도 |
|---|---|
| [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | 현재 챔피언과 운영 기준 |
| [`docs/STANDALONE_CHAMPION.md`](docs/STANDALONE_CHAMPION.md) | 최신 단일 ZIP 실행·감사·전달 |
| [`reports/jy_runners_high_li_bridge027_public_result_20260828.md`](reports/jy_runners_high_li_bridge027_public_result_20260828.md) | 1172.137 갱신 근거 |
| [`reports/target1180_v167_public_result_20260825.md`](reports/target1180_v167_public_result_20260825.md) | v167 기준선 근거 |
| [`reports/target1170_v142_v148_public_result_20260823.md`](reports/target1170_v142_v148_public_result_20260823.md) | 직전 1170 달성 근거 |
| [`reports/target1173_v149_v154_final_submission_20260823.md`](reports/target1173_v149_v154_final_submission_20260823.md) | 마지막 제출 연구와 비승격 판정 |
| [`reports/submissions.csv`](reports/submissions.csv) | 제출 ID·점수·해시 원장 |
| [`submissions/README.md`](submissions/README.md) | 제출 파일 보관 규칙과 계보 |
| [`docs/EXPERIMENT_WORKFLOW.md`](docs/EXPERIMENT_WORKFLOW.md) | 공통 실험·검증 절차 |
| [`docs/GITHUB_START_GUIDE.md`](docs/GITHUB_START_GUIDE.md) | 팀원 GitHub 시작 안내 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 브랜치·리뷰·커밋 규칙 |

## Git·데이터 정책

- 원본 DACON 데이터, 인증정보, 개인 쿠키·키는 Git과 LFS에 올리지 않는다.
- 모델 ZIP·OOF·임시 cache는 원칙적으로 Git에서 제외한다. PRIVATE 팀 전달용 v167 ZIP,
  최신 JY champion ZIP, 문서화된 기존 1161 LFS·1170 OOF 번들만 해시가 고정된 예외다.
- `main`은 검증된 기준선으로 유지하고 새 연구는 별도 브랜치에서 수행한다.
- 챔피언을 덮어쓰지 않고 버전별 새 디렉터리에 패키징한다.
- DACON 제출은 검증된 단일 ZIP 하나를 지정 담당자가 수행하고 결과를 즉시 원장에 남긴다.

2026-08-28 전달 정책 반영 후 data-free CI 묶음은 로컬에서 `57 passed`로 재현됐으며,
현재 공식 기준은 jy_runners_high_li_bridge027 Public **1172.1373858439**이다.
