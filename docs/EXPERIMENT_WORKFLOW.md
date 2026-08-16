# 팀 실험 실행 워크플로

이 문서는 현재 Public champion v20을 기준으로 실험을 재현하고 새 후보를 평가하는 공통 절차다. Python 모듈이 계산의 단일 원본이고, [`../notebooks/experiment_workbench.ipynb`](../notebooks/experiment_workbench.ipynb)는 그 모듈을 순서대로 호출하고 결과를 확인하는 얇은 실행 화면이다. 노트북 안에 학습 로직을 복사하지 않는다.

## 가장 먼저: 파일과 실행 노트북

현재 수치와 파일 위치는 [`PROJECT_STATUS.md`](PROJECT_STATUS.md), [`../submissions/README.md`](../submissions/README.md)에서 먼저 확인한다. Git에는 제출 ZIP과 모델이 없으므로 동일한 공식 DACON 팀의 private artifact 저장소에서 필요한 파일을 받아 SHA-256을 대조한다.

| 작업 | 받을 파일 |
|---|---|
| champion 실행·비교 | `submit_v20.zip` |
| v20 재패키징 | `submissions/history/submit_v19.zip` + `artifacts/v20_target1160_final_20260816/` |
| v20 재학습 | DACON 원본 데이터 + v19 OOF/alignment artifact + v20 활성 학습 코드 |

제출 ZIP은 수동으로 풀거나 다시 압축하지 않는다. v19 이하의 계보와 Drive 구조는 [`ARTIFACT_HANDOFF.md`](ARTIFACT_HANDOFF.md), v20 최종 해시는 [`../reports/v20_public_result_20260816.md`](../reports/v20_public_result_20260816.md)에 있다.

Python 스크립트를 실행하는 공식 단일 노트북은 다음 파일이다.

```text
notebooks/experiment_workbench.ipynb
```

이 노트북에서 경로 확인 → 테스트 → screen → 강건 평가 → 학습 → 패키징 → 검증 순서로 실행한다.

## 1. 환경 준비

프로젝트 루트에서 Python 3.11 가상환경을 만들고 의존성을 설치한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install jupyterlab
```

JupyterLab은 노트북을 열 때만 필요한 선택 의존성이다. CI는 노트북 JSON과 코드 셀을 직접 검사하므로 Jupyter를 설치하지 않는다.

## 2. 필요한 입력과 보관 위치

| 작업 단계 | 필요한 입력 | Git 포함 여부 |
|---|---|---|
| 데이터 비의존 테스트 | 저장소 코드만 | 포함 |
| v14/v15 로컬 평가 | `data/train.csv`, v13/v14 OOF cache | 제외 |
| pressure EB screen | `data/train.csv`, 2022~2024 부모 OOF cache | 제외 |
| v17 residual 학습 | 위 입력과 선택된 recipe | 결과 제외 |
| ZIP 재패키징 | 검증된 부모 ZIP, residual spec | 제외 |
| 제출 검증 | 후보·부모 ZIP, `data/train.csv`, `data/test.csv` | 제외 |

원본 데이터는 각 팀원이 DACON에서 직접 내려받는다. ZIP·모델·OOF·`artifacts/` 산출물은 Git에 올리지 않고, 공식 팀원만 접근할 수 있는 제한된 artifact 공간에서 SHA-256과 함께 전달한다.

## 3. 빠른 상태 점검

아래 단계는 데이터 없이도 실행할 수 있다.

```powershell
python scripts/audit_repository.py --include-untracked
python -m pytest -q `
  tests/test_metrics.py `
  tests/test_local_scorecard.py `
  tests/test_robust_local_evaluation.py `
  tests/test_v16_residual.py `
  tests/test_experiment_notebook.py
```

데이터와 private artifact까지 준비됐다면 전체 테스트를 실행한다.

```powershell
python -m pytest -q
```

## 4. 워크벤치 노트북 사용

```powershell
jupyter lab notebooks/experiment_workbench.ipynb
```

처음 열면 위에서 아래로 실행하되 기본값을 유지한다.

- `DRY_RUN = True`: 실행할 명령만 출력한다.
- `RUN_HEAVY = False`: screen·학습·강건 평가를 실행하지 않는다.
- `RUN_PACKAGING = False`: ZIP 생성과 end-to-end 검증을 실행하지 않는다.

먼저 경로와 명령을 확인하고, 필요한 데이터·artifact가 모두 있을 때만 해당 스위치를 명시적으로 바꾼다. 출력 디렉터리와 보고서 이름은 실험마다 새 이름을 사용한다. 이미 존재하는 챔피언 ZIP과 확정 보고서는 덮어쓰지 않는다.

- 테스트와 실제 명령 실행: `DRY_RUN = False`
- screen·학습·강건 평가 실행: `RUN_HEAVY = True`
- ZIP 생성·end-to-end 검증 실행: `RUN_PACKAGING = True`

## 5. pressure EB 연구 순서

역사적 모듈 이름에는 `v16`이 남아 있지만, 이 코드에서 선택된 pressure recipe와 correction weight `1.5`를 적용해 만든 제출이 v17이다.

### 5.1 후보 축 screen

```powershell
python -m src.v16_residual_calibration_screen `
  --output-dir artifacts/v19_residual_calibration_<run_id>

python -m src.v16_multiseason_screen `
  --output-dir artifacts/v19_multiseason_<run_id>
```

그룹 축, decay, alpha와 weight를 동시에 넓게 탐색한 뒤 최고 한 점만 보고 선택하지 않는다. 2023·2024 최소 gain, 최신 전이, 월·팀 집중도와 교차 bootstrap을 함께 본다.

### 5.2 선택 후보 강건 평가

아래는 v17에서 사용한 pressure 후보를 별도 보고서 이름으로 다시 평가하는 예다.

```powershell
python -m src.evaluate_v16_robust `
  --selected multi_pitcher_batter_hand_pressure_d1_a3200_w1.5 `
  --artifact-dir v16_multiseason_20260815_02 `
  --parent-f-alpha 0.15 `
  --output-json reports/v19_pressure_audit_<run_id>.json `
  --output-markdown reports/v19_pressure_audit_<run_id>.md
```

`--artifact-dir`는 프로젝트의 `artifacts/` 아래 디렉터리 이름이다. screen 출력의 실제 후보명과 cache 파일이 일치하는지 확인한다.

### 5.3 최종 residual spec 학습

```powershell
python -m src.train_v16_residual `
  --output-dir artifacts/v19_residual_final_<run_id> `
  --group pitcher_batter_hand_pressure `
  --decay 1 `
  --alpha 3200 `
  --correction-weight 1.5 `
  --candidate-name v19_pressure_eb_<run_id>
```

이 명령은 `v16_residual_spec.json`을 만든다. 파일명은 추론 호환성 때문에 유지되며 후보 버전은 spec 내부와 실험 보고서에 기록한다.

## 6. v17 재패키징과 검증

현재 패키저는 v14를 부모로 pressure residual 한 층을 추가하는 구조다. v17을 재현할 때는 SHA-256이 정확히 일치하는 v14 부모와 v17 residual artifact를 사용한다.

```powershell
python -m src.package_v16_residual `
  --parent submit_v14.zip `
  --output submit_v17_rebuild.zip `
  --artifact-dir v17_residual_final_20260815 `
  --expected-parent-sha256 AC8E135FBA8DBF41E331A8F4E2AAA658F0AF6675DD5F3B6C89EF51E9BC02977E

python -m src.validate_v16_residual `
  --candidate submit_v17_rebuild.zip `
  --parent submit_v14.zip `
  --report reports/v17_rebuild_validation_<run_id>.md
```

새 후보를 v17 ZIP 위에 다시 쌓을 때 이 패키저를 그대로 사용하면 lineage 검증의 `added member` 가정이 깨진다. 다음 후보는 다음 중 하나로 구성한다.

1. v14에서 출발해 검증된 최종 residual spec 하나를 넣어 전체 후보를 다시 만든다.
2. v17 위에 새로운 종류의 overlay를 추가할 경우 별도 패키저와 lineage test를 먼저 구현한다.

기존 `submit_v17.zip`을 덮어쓰거나 수동으로 ZIP 내부를 편집하지 않는다.

## 7. 후보 승격 기준

새 후보는 다음 조건을 모두 만족해야 제출 검토 대상으로 올린다.

1. 누수 없는 rolling-origin OOF만 사용한다.
2. v20 행별 예측을 고정 incumbent로 paired 비교한다.
3. 최근 두 전이의 전체 gain 방향이 양수다.
4. 월·투수·타자·투수×타자와 연속 투구 block bootstrap을 통과한다.
5. 같은 final family 안에서 selection-aware Reality Check를 통과한다.
6. `R_CORE`, `R_ANCHOR`, F 기여를 분리해 한 영역의 손실을 평균으로 숨기지 않는다.
7. ZIP lineage, offline 실행, 120초 제한, 메모리와 batch invariance를 통과한다.
8. 팀 리뷰 후 지정된 제출 담당자만 DACON에 올린다.

Public 결과는 모델 선택 규칙을 사후 미세조정하는 학습 데이터로 사용하지 않는다. v17/v18에서 확인한 결정은 “pressure EB 채택, 공격적 F 확대 기각” 수준으로만 고정한다.

## 8. 실험 기록 규칙

각 실험은 최소한 다음을 남긴다.

- 가설과 야구적 근거
- 부모 버전과 SHA-256
- 학습 시즌, OOF 경로와 시간 경계
- 후보군 전체와 선택 규칙
- 시즌·도메인·월별 gain
- bootstrap p05, 개선 확률과 Reality Check
- 생성한 artifact 경로와 SHA-256
- 채택·기각 결정 및 다음 행동

코드는 기능 브랜치에 커밋하고, 모델·OOF·제출 ZIP은 커밋하지 않는다. 제출 결과는 [`../reports/submissions.csv`](../reports/submissions.csv)에 즉시 추가한다.
