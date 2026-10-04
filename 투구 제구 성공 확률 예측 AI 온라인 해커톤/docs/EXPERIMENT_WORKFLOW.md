# 팀 실험 실행 워크플로

> **2026-08-28 현재 기준**: 운영 champion은 `jy_runners_high_li_bridge027` Public
> `1172.1373858439`이며 전달 파일은 PRIVATE 저장소의
> `submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip`이다. 아래 v26·v1161 절차는 역사
> 재현용이다. 새 후보 평가는
> [`../research/configs/evaluation_v3.json`](../research/configs/evaluation_v3.json)과
> [`../research/reports/evaluation_reaudit_20260822.md`](../research/reports/evaluation_reaudit_20260822.md)를
> 우선한다.

> **2026-08-18 정정**: 이 문서는 이전에 champion을 v27로 기록했으나 공식 DACON 제출 이력 재대조 결과 오류였다. 실제 champion은 `submit_v26.zip`이다(Public `1157.9736407889`, 제출 ID `51773`). 근거와 재현 절차는 [`../research/reports/target1170_followup_20260817.md`](../research/reports/target1170_followup_20260817.md) 상단과 [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb)에 있다.

이 문서는 현재 standalone champion을 고정 incumbent로 두고 새 후보를 평가하는 공통
절차다. Python 모듈이 계산의 단일 원본이다. v26 재현 노트북과
`experiment_workbench.ipynb`는 과거 계보 감사용이며 새 후보의 승인 화면이 아니다.

## 가장 먼저: 파일과 실행 노트북

현재 수치와 파일 위치는 [`PROJECT_STATUS.md`](PROJECT_STATUS.md), [`../submissions/README.md`](../submissions/README.md)에서 먼저 확인한다. champion ZIP과 OOF evidence는 clone에 포함되지 않는다. 팀 전용 경로에서 받은 뒤 각각 SHA-256을 대조한다.

| 작업 | 받을 파일 |
|---|---|
| champion 실행·비교 | `submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip` |
| v167 기준선 비교 | `submissions/releases/v167/submit_v167.zip` |
| 새 후보 로컬 비교 | 팀 전용 경로로 받은 `artifacts/oof_champion_1161/`의 fidelity label + 후보 exact temporal OOF |
| v26 재패키징 | `submit_v25.zip` + `src/package_v26_anchor_weight_probe.py` (기본값 `--probe-eta 0.15`) |
| v25 기반모형 재학습 | DACON 원본 데이터 + v22 OOF artifact + v25 활성 학습 코드 |

제출 ZIP은 수동으로 풀거나 다시 압축하지 않는다. v19 이하의 계보와 Drive 구조는 [`ARTIFACT_HANDOFF.md`](ARTIFACT_HANDOFF.md), v26 최종 해시와 공식 결과는 [`../research/reports/target1170_followup_20260817.md`](../research/reports/target1170_followup_20260817.md) 상단 정정 안내와 [`../research/reports/v26_validation.json`](../research/reports/v26_validation.json)에 있다.

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
| pressure EB screen | `data/train.csv`, 2022–2024 부모 OOF cache | 제외 |
| v17 residual 학습 | 위 입력과 선택된 recipe | 결과 제외 |
| ZIP 재패키징 | 검증된 부모 ZIP, residual spec | ZIP은 저장소에 두지 않음 |
| 제출 검증 | 후보·부모 ZIP, `data/train.csv`, `data/test.csv` | 데이터·후보 제외 |

원본 데이터는 각 팀원이 DACON에서 직접 내려받는다. champion ZIP과 OOF evidence는
저장소가 아니라 팀 전용 Drive로 공유하며, 모든 파일은 SHA-256과 sensitivity manifest를
동반한다. 후보·중간 모델·대규모 캐시도 마찬가지로 올리지 않는다.

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
python -m src.archive.v16_residual_calibration_screen `
  --output-dir artifacts/v19_residual_calibration_<run_id>

python -m src.archive.v16_multiseason_screen `
  --output-dir artifacts/v19_multiseason_<run_id>
```

그룹 축, decay, alpha와 weight를 동시에 넓게 탐색한 뒤 최고 한 점만 보고 선택하지 않는다. 2023·2024 최소 gain, 최신 전이, 월·팀 집중도와 교차 bootstrap을 함께 본다.

### 5.2 선택 후보 강건 평가

아래는 v17에서 사용한 pressure 후보를 별도 보고서 이름으로 다시 평가하는 예다.

```powershell
python -m src.archive.evaluate_v16_robust `
  --selected multi_pitcher_batter_hand_pressure_d1_a3200_w1.5 `
  --artifact-dir v16_multiseason_20260815_02 `
  --parent-f-alpha 0.15 `
  --output-json research/reports/v19_pressure_audit_<run_id>.json `
  --output-markdown research/reports/v19_pressure_audit_<run_id>.md
```

`--artifact-dir`는 프로젝트의 `artifacts/` 아래 디렉터리 이름이다. screen 출력의 실제 후보명과 cache 파일이 일치하는지 확인한다.

### 5.3 최종 residual spec 학습

```powershell
python -m src.archive.train_v16_residual `
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
python -m src.archive.package_v16_residual `
  --parent submit_v14.zip `
  --output submit_v17_rebuild.zip `
  --artifact-dir v17_residual_final_20260815 `
  --expected-parent-sha256 AC8E135FBA8DBF41E331A8F4E2AAA658F0AF6675DD5F3B6C89EF51E9BC02977E

python -m src.archive.validate_v16_residual `
  --candidate submit_v17_rebuild.zip `
  --parent submit_v14.zip `
  --report research/reports/v17_rebuild_validation_<run_id>.md
```

새 후보를 v17 ZIP 위에 다시 쌓을 때 이 패키저를 그대로 사용하면 lineage 검증의 `added member` 가정이 깨진다. 다음 후보는 다음 중 하나로 구성한다.

1. v14에서 출발해 검증된 최종 residual spec 하나를 넣어 전체 후보를 다시 만든다.
2. v17 위에 새로운 종류의 overlay를 추가할 경우 별도 패키저와 lineage test를 먼저 구현한다.

기존 `submit_v17.zip`을 덮어쓰거나 수동으로 ZIP 내부를 편집하지 않는다.

## 7. 후보 승격 기준

새 후보는 다음 조건을 모두 만족해야 제출 검토 대상으로 올린다.

1. 고정 incumbent는 Public 1158 standalone의 동일 행 exact OOF다.
2. feature 선택, tuning, calibration을 outer fold 안에서 모두 반복한 `nested_outer` 또는
   recipe를 보기 전에 잠근 `locked_shadow` 축이 서로 다르게 최소 2개 있어야 한다.
3. 반복 열람한 2024 축은 `development_contaminated` 진단이며 primary evidence가 아니다.
4. 모든 primary 축에서 gain `>0`, 양수 월 `>=75%`, 최악 월 `>-5`,
   `R_CORE/R_ANCHOR/F` 최소 gain `>=0`을 만족한다.
5. pitcher, crossed pitcher×batter, 연속 투구 block bootstrap p05가 모두 양수다.
6. final family의 실행 trial을 빠짐없이 ledger에 남기고 White Reality Check
   `p<=0.10`을 통과한다.
7. exact parent parity와 recipe 동결 시점을 증명한다.
8. 통과 후보만 처음부터 standalone ZIP으로 만든다. 과거 제출 ZIP·저장소 코드·다른
   artifact에 실행 의존하지 않는다.
9. 오프라인, 행 독립성, 메모리, 공식 600초와 내부 soft guard 120초를 통과한다.
10. 팀 리뷰 후 지정된 제출 담당자만 DACON에 올린다.

`src/evaluation_contract.py`가 2–7번을 기계적으로 판정한다. local→Public projection은
clean transfer 관측 3개 전까지 억제하며, 이후에도 승격 근거로 사용하지 않는다. Public
결과는 모델 선택 규칙이나 blend weight를 사후 미세조정하는 학습 데이터로 쓰지 않는다.

팀원 모델을 혼합할 때는 `research/configs/oof_bundle_contract_v1.json` 형식의 exact temporal OOF만
받고 `src/archive/v77_team_oof_constrained_blend.py`로 정렬·SHA-256·strict-forward 경계와 source
월·domain 비악화 제약을 검증한다. 제출 ZIP이나 Public 점수만 있는 모델의 weight는 정하지
않는다. v77 결과도 bootstrap·Reality Check를 통과하기 전에는 승격 근거가 아니다.

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

코드는 기능 브랜치에 커밋한다. 모델·OOF·제출 ZIP은 저장소에 커밋하지 않는다. 제출 결과는 [`../research/reports/submissions.csv`](../research/reports/submissions.csv)에 즉시 추가한다.
