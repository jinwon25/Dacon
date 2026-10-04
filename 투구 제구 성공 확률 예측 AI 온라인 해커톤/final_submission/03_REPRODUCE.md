> **공개본 실행 조건**: 공식 데이터 외에 비공개 체크포인트·OOF·모델 입력이 필요합니다. 아래는 원 대회 제출 패키지의 절차이며 공개 clone에서 그대로 전체 실행할 수 없습니다. 동결 체크포인트의 동일 재조립과 모든 모델의 신규 학습은 별개입니다.

# Private Score 재현 실행 방법

모든 명령은 `code/`에서 실행한다. 데이터 경로와 출력 경로는 인자로 받는다.
네트워크는 최초 의존성 설치에만 사용하고 학습·추론은 로컬에서 실행한다.
공식 train.csv, trackman_history.csv, test.csv, sample_submission.csv를 `data/`에 둔다.

## 1. 입력 검사

```powershell
python -m pip install -r requirements-dev.txt
python train.py preflight --data-dir data
```

preflight는 공식 데이터·고정 체크포인트 해시와 환경을 확인한다.
전체 학습 재현의 통과 판정은 아니다.

## 2. 최종 ZIP 재빌드 및 검증

```powershell
python train.py rebuild --data-dir data --output-dir artifacts/rebuild_01
python train.py verify --data-dir data --package artifacts/rebuild_01/submit_v345.zip
```

새 폴더만 허용한다. v343 체크포인트는 그대로 사용하며 Beta-Binomial 보정을
공식 train에서 새로 적합한다. 원 v345 SHA-256과 다르면 실패한다.
verify의 245,789행 실행은 공개 샘플 기반 부하 검사이며 숨겨진 test가 아니다.

## 2-1. 초기 V2 OOF와 부모 native 모델

초기 residual 계보에 쓰인 V2 OOF도 새 출력 폴더에서 다시 학습할 수 있다.
참조 폴더는 선택 사항이며, 지정해도 신규 학습이 끝난 뒤 숫자 배열 비교에만 쓴다.

```powershell
python train_initial_oof_fresh.py --data-project . --output-dir artifacts/initial_oof_01
python train_parent_initial.py --component corrected --data-project . --oof-project artifacts/initial_oof_01 --output-dir artifacts/corrected_01
python train_parent_initial.py --component advanced --data-project . --oof-project artifacts/initial_oof_01 --output-dir artifacts/advanced_01
python train_parent_initial.py --component legacy --data-project . --output-dir artifacts/legacy_01
python train_parent_initial.py --component futures_initial --data-project . --output-dir artifacts/futures_initial_01
```

V2 OOF는 매 검증 연도보다 앞선 시즌만 학습하고, 반복 호출되는 같은 fold의
LightGBM/RF 예측은 해당 실행의 메모리 안에서만 재사용한다. 외부·평가 데이터는
사용하지 않는다. corrected와 advanced 모델은 이 명령이 남긴 해시 보고서 및
OOF 파일을 다시 확인한 뒤 적합한다.

초기 legacy CatBoost는 당시 사용한 endpoint-shift 피처 사양을
`legacy_state/src/top1100_features.py`에 별도로 보존했다. corrected residual의
수정 ASOF 사양은 `src/corrected_top1100_features.py`다. 둘을 교체하면 다른
모델이 되므로 공통 이름 때문에 합치지 않는다.

원 제출 native 자산과 비교:

```powershell
python validate_parent_native_fit.py --component legacy --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/legacy_01/model --output-dir artifacts/legacy_check_01
python validate_parent_native_fit.py --component futures_initial --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/futures_initial_01/artifacts/f_regime/final --output-dir artifacts/futures_initial_check_01
python validate_parent_native_fit.py --component corrected --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/corrected_01/model --output-dir artifacts/corrected_check_01
python validate_parent_native_fit.py --component advanced --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/advanced_01/model --output-dir artifacts/advanced_check_01
python train_parent_base.py --data-dir data --output-dir artifacts/parent_base_01
python validate_parent_native_fit.py --component base --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_base_01 --output-dir artifacts/parent_base_check_01
python train_parent_trackman.py --data-dir data --output-dir artifacts/parent_trackman_01
python validate_parent_native_fit.py --component trackman --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_trackman_01 --output-dir artifacts/parent_trackman_check_01
python train_parent_recency_rf.py --data-dir data --output-dir artifacts/parent_recency_rf_01
python validate_parent_native_fit.py --component recency_rf --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_recency_rf_01 --output-dir artifacts/parent_recency_rf_check_01
python -m src.train_recent_exact_overlay --project . --source-artifact-root reproduction_inputs/recent_exact_oof --output-dir artifacts/parent_recent_exact_01
python validate_parent_native_fit.py --component recent_exact --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_recent_exact_01 --output-dir artifacts/parent_recent_exact_check_01
python -m src.train_v14_refinement --project . --output-dir artifacts/parent_v14_01
python validate_parent_native_fit.py --component v14_refinement --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_v14_01 --output-dir artifacts/parent_v14_check_01
python -m src.train_state_mode_joint_2025 --project . --output-dir artifacts/parent_joint_state_01
python validate_parent_native_fit.py --component joint_state_mode --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_joint_state_01 --output-dir artifacts/parent_joint_state_check_01
python -m src.train_v20_target1160 --project . --output-dir artifacts/parent_v20_01 --teacher-folds 3
python validate_parent_native_fit.py --component v20_pfd --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_v20_01 --output-dir artifacts/parent_v20_check_01
python -m src.train_v21_context_state_eb --project . --output-dir artifacts/parent_v21_01
python validate_parent_native_fit.py --component v21_eb --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_v21_01 --output-dir artifacts/parent_v21_check_01
python -m src.train_v22_low_variance_overlay --project . --profile balanced --output-dir artifacts/parent_v22_01
python validate_parent_native_fit.py --component v22_low_variance --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_v22_01 --output-dir artifacts/parent_v22_check_01
python -m src.train_v25_postbreak_anchor --project . --output-dir artifacts/parent_v25_01
python validate_parent_native_fit.py --component v25_anchor --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/parent_v25_01 --output-dir artifacts/parent_v25_check_01
```

이 검사는 모델 본체·피처 사양·학습 통계의 일치를 확인한다. 후속 세대가 변경한
앙상블 dose와 부모 파일명 주석은 별도로 표시하며 전체 v124 조립의 재현과
동일시하지 않는다.

recent-exact의 LightGBM·Ridge·전처리 상태는 새로 적합한다. 다만 그 입력인
2024 corrected/advanced/legacy OOF 3개는 `reproduction_inputs/recent_exact_oof/`에
동결해 둔 공식 train 파생값이다. 각 파일 SHA-256은 실행 manifest에 기록되며,
이 명령 하나가 해당 OOF 자체까지 새로 적합한다는 뜻은 아니다.

joint state/mode와 v25가 사용하는 고정 선택 요약, v20이 사용하는 2024 joint
OOF·TrackMan 정렬은 `reproduction_inputs/`에 포함했다. 모두 공식 제공 데이터의
파생 입력이며 실행 manifest가 SHA-256을 확인 가능한 형태로 남긴다.
v14·v20·v25의 native 모델 상태는 일치하지만 제출 spec의 최종 dose는 이후
v124 quadratic stack에서 바뀌었다. validator는 이 dose와 조립 주석을 별도
필드로 보고하고 모델 적합 동일성과 혼동하지 않는다.
v21 EB와 v22 low-variance도 재생성된 효과·anchor 상태는 일치하며,
제출 spec의 후속 v124 dose만 같은 방식으로 분리한다.

## 3. 동결 lookup

wave0 OOF를 캐시 없이 새로 학습하려면 다음을 실행한다.

```powershell
python train_wave0_fresh.py --data-dir data --output-dir artifacts/wave0_fresh_01
```

2020–2024 각 검증 연도보다 앞선 시즌으로 LGB/RF를 학습한다.
LightGBM의 조기 종료에는 해당 검증 연도 라벨을 사용하므로,
모델 선택까지 독립된 성능 평가로 해석하지 않는다. 모든 행은 공식 train 소속이다.
결과는 `artifacts/wave0_fresh_01/artifacts/followup/oof/`에 생성된다.
이 출력으로 저차원·선수 전이 조회표를 학습할 수 있다.

```powershell
python -m src.champion.v319_finalize_futures_lowrank --train-csv data/train.csv --wave0-oof-dir artifacts/wave0_fresh_01/artifacts/followup/oof --output-dir artifacts/lowrank_fresh_01
python -m src.champion.v334_finalize_player_transition --train-csv data/train.csv --oof-dir artifacts/wave0_fresh_01/artifacts/followup/oof --output-dir artifacts/transition_fresh_01
```

참조 v345 ZIP과 비교하며 두 조회표를 함께 학습하려면 다음을 사용할 수 있다.

```powershell
python train_auxiliary_lookups.py --train-csv data/train.csv --oof-dir artifacts/wave0_fresh_01/artifacts/followup/oof --reference-zip ORIGINAL_SUBMITTED_V345.zip --output-dir artifacts/auxiliary_validation_01
```

실측: 새 OOF 5개 구간의 예측 차이 6.67e-16 이내, 저차원 배열 차이 0,
선수 전이 보정값 최대 차이 3.47e-18. 참조 OOF를 검사기로 제공할 때는
`train_wave0_fresh.py --reference-oof-dir 경로`를 사용한다.
참조 OOF는 학습에 사용하지 않고 신규 학습 결과와 대조하는 데만 사용한다.

```powershell
python train.py lookups --data-dir data --output-dir artifacts/lookups_01
```

공식 train과 TrackMan으로 lookup을 재생성·대조한다. XGBoost 학습과는 별개다.

## 4. H1 학습

```powershell
python -m venv .venv-h1
.venv-h1/Scripts/python.exe -m pip install -r h1/requirements_verified.txt
.venv-h1/Scripts/python.exe train.py train-h1 --data-dir data --output-dir artifacts/h1_01
.venv-h1/Scripts/python.exe validate_h1_fresh_fit.py --model artifacts/h1_01/cat_asof_xl.pkl --train-csv data/train.csv --test-csv data/test.csv --output-dir artifacts/h1_validation_01
```

원 저장 설정인 depth 8 / 3시드 / 고정 center를 쓴다.
버전이 맞지 않으면 학습 시작 전에 중단한다.
세 모델 전체 신규 학습 후 트리·분할 경계·잎 값이 원 저장 모델과 일치했다.
공식 train에서 고르게 추출한 32,768행과 공개 test 5행의 예측 최대 차이는 0이다.
고정 alpha/center를 재추정한 검증이나 전체 v345 신규 학습 검증은 아니다.

## 4-1. 등판량 H1 학습

CatBoost 1.2.8인 기본 검증 환경에서 실행한다. 원 제출 ZIP은 해시 확인 후
H1의 고정 전처리 사양을 읽는 데 사용하며, 세 학습기는 모두 새로 적합한다.
원 제출 ZIP은 2절 재빌드로 생성할 수도 있다.

```powershell
python train_workload_fresh.py --data-dir data --reference-zip ORIGINAL_SUBMITTED_V345.zip --output-dir artifacts/workload_01
python validate_h1_fresh_fit.py --model artifacts/workload_01/workload_h1_bundle.joblib --train-csv data/train.csv --test-csv data/test.csv --workload-reference-zip ORIGINAL_SUBMITTED_V345.zip --output-dir artifacts/workload_validation_01
```

82개 H1 피처에 19개 등판량 피처를 더하고 depth 8 / 1200 iterations /
seed 42, 43, 44로 공식 학습 행 전체를 학습한다.
비율에서 복원한 최소 분모는 실제 투구 수를 유일하게 식별하는 값이 아니라,
관측 비율과 양립하는 최소 표본 크기 추정치다.

## 4-2. C3의 신규 OOF 및 테이블 학습

```powershell
python train_c3_fresh.py --data-dir data --reference-zip ORIGINAL_SUBMITTED_V345.zip --output-dir artifacts/c3_01
```

CatBoost 1.2.8, 고정 v131 depth 6 설정으로 2020–2024 각 연도보다 앞선
공식 학습 행에서 H1 OOF를 새로 생성한다. 배포 H1의 depth 8 설정과 구분한다.
이 OOF로 최근 1·2·3년 및 확장 창의 C3 테이블을 다시 계산한다.
테이블 설정은 실제 배포 계보의 v148 설정을 사용한다.
원 C3 테이블은 학습에 사용하지 않고 마지막 결과 비교에만 읽는다.
비교 결과는 `c3_fresh_fit.json`에 저장하며, 허용한 로컬 차이 1e-12를
넘으면 실패하고 원 제출 모델을 유지한다. 창·평활 상수의 재선정은 실행하지 않는다.

## 5. strict 학습

```powershell
python train.py train-strict --check --data-dir data --output-dir artifacts/strict_01
python train.py train-strict --data-dir data --output-dir artifacts/strict_01
```

4개 시즌 LGB/HGB OOF → 팀 효과 → 투수·카운트 효과 → 저차원 효과 → 최종 적합 순서다.
`--stage`로 개별 단계를 선택할 수 있다. `--check`는 학습을 실행하지 않는다.
결과는 `artifacts/strict_01/submit_exp021_strict.zip`이다.
이 결과를 전체 v345 계보에 연결한 신규 학습 검증은 완료하지 않았다.

## 5-1. 퓨처스 5시드 전체 학습

CatBoost 1.2.8인 기본 검증 환경에서 실행한다. H1의 1.2.10 환경과 구분한다.
선택 기록은 패키지 내부 JSON으로 고정하고 해시를 검사한다.

```powershell
python train.py train-futures --data-dir data --output-dir artifacts/futures_01
python validate_futures_fresh_fit.py --reference-zip ORIGINAL_SUBMITTED_V345.zip --candidate-dir artifacts/futures_01 --train-csv data/train.csv --output-dir artifacts/futures_validation_01
```

`ORIGINAL_SUBMITTED_V345.zip`은 원 제출 ZIP 경로로 바꾼다. 해당 파일은
2절 재빌드 명령으로도 생성할 수 있으며 검사기는 원 제출 SHA-256을 강제한다.
2024년 F 학습 행 30,010개로 5시드를 새로 학습했고 트리·CTR·경계·예측이 일치했다.
기존 v288 선택 기준과 고정 혼합 비율을 다시 탐색한 것은 아니다.

## 6. fallback 및 테스트

fallback XGBoost의 원 GPU 학습 행렬과 현재 114피처 재구성 행렬의 완전한 동일성은 아직 확인하지 못했다.
float32 계산 순서를 보완한 재구성 피처에서 원 모델의 분할 경계
8,666개가 모두 확인됐다. 이는 원 학습 행렬 전체의 동일성을 증명하지 않는다.
XGBoost 3.2.0 CPU 신규 학습도 완료했으나 같은 동결 추론 피처로 비교한
32,773행의 예측 최대 차이 0.04281276, RMS 차이 0.00854285로 불일치했다.
GPU/CPU 차이와 미복원 학습 조건 중 어느 것이 원인인지는 확정하지 않는다.
`fallback_xgb/build_fallback_xgb_model.py`는 기본 실행을 차단하며,
`--allow-reconstructed-training`을 명시한 경우에만 재구성 실험을 허용한다.
실험 모델은 원 제출 ZIP에 자동으로 반영하지 않는다.
XGBoost 버전 3.2.0을 강제하며 CUDA 요청 시 실제 GPU를 검사한다.
GPU가 없을 때 CPU로 조용히 전환해 GPU 학습으로 기록하지 않고 즉시 중단한다.
공통 시즌 평활 함수는 `common_features/season.py`에 있다.
검증 시즌 이전 데이터로만 lookup을 만드는 OOF는
`src.archive.v242_runtime_faithful_fallback_oof`를 사용한다.

```powershell
python fallback_routes/build_candidate_zip.py --source reproduction_inputs/bridge027.zip --asset-dir fallback_xgb --output artifacts/v244_01.zip
python fallback_xgb/check_training_precision.py --train-csv data/train.csv --trackman-csv data/trackman_history.csv --output-dir artifacts/xgb_precision_01
python fallback_xgb/build_fallback_xgb_model.py --allow-reconstructed-training --device cpu --train-csv data/train.csv --trackman-csv data/trackman_history.csv --output-dir artifacts/xgb_fit_01
python fallback_xgb/check_fresh_fit.py --model artifacts/xgb_fit_01/fallback_xgb.json --train-csv data/train.csv --test-csv data/test.csv --output-dir artifacts/xgb_validation_01
python -m pytest -q -p no:cacheprovider
```

CUDA 검증은 NVIDIA CUDA GPU 환경에 별도 가상환경을 만든 뒤
`requirements.txt`를 설치하고 다음처럼 실행한다.

```powershell
python fallback_xgb/build_fallback_xgb_model.py --allow-reconstructed-training --device cuda:0 --train-csv data/train.csv --trackman-csv data/trackman_history.csv --output-dir artifacts/xgb_gpu_fit_01
python fallback_xgb/check_fresh_fit.py --model artifacts/xgb_gpu_fit_01/fallback_xgb.json --train-csv data/train.csv --test-csv data/test.csv --output-dir artifacts/xgb_gpu_validation_01
```

이 절차는 팀의 고정 114피처 계약으로 fallback 모델을 다시 학습한다. 다만 비교
보고서가 일치하기 전에는 GPU를 사용했다는 사실만으로 동일 재현으로 판정하지 않는다.

단위 테스트 통과, 체크포인트 재빌드, 전체 원데이터 신규 학습 재현은 각각 구분한다.

## 7. 제출 구성 최종 점검

```powershell
python check_submission_contents.py --package-dir .. --code-only
python check_submission_contents.py --package-dir .. --pptx SOLUTION.pptx --attendance PARTICIPATION.csv
```

첫 명령은 이번 코드 범위 검사다. PPT와 참가 여부는 팀에서 별도로 첨부한다.
두 번째 명령은 자료가 모두 준비된 후 선택적으로 실행할 전체 구성 검사다.
CSV는 이 로컬 점검기의 편의 형식이지 공식 제출 양식을 추가로 요구하는 것이 아니다.

참가 여부 CSV의 열은 `member,phase3_attendance`이고 참석 값은
`attending` 또는 `not_attending`이다. 팀원 실명과 실제 참가 여부를 사용한다.
PPT/CSV 경로는 실제 준비한 파일로 바꾼다. 미해결·누락 항목이 있으면 종료 코드 2로
중단한다. PPT 구조 및 CSV 값 검사는 내용의 정확성·참가 자격·공식 승인을 증명하지 않는다.
코드 범위 검사와 exact fresh-fit 진단은 분리되며, 상세 상태는 `verification/training_coverage.json`에 기록된다.

## 부록 A. v56 / v104 FM·conditional native 자산 재학습

`reproduction_inputs/parent_v104/`의 7개 배열은 공식 train에서 만든
고정 OOF·정렬 입력이며 README에 크기와 SHA-256을 기록했다. 평가 배치의
평균·빈도·순위는 포함하지 않는다. 다음 명령은 v56 shared FM을 다시 적합한다.

```powershell
python -m src.champion.v84_fixed_v56_export --project . --source-artifact-root reproduction_inputs/parent_v104 --output-dir artifacts/parent_v56_fm_01
```

v104의 older/recent R-FM과 conditional/baseline LightGBM 쌍은 다음과 같이
재학습한다. 앞 절에서 만든 1161 체크포인트를 부모 ZIP으로 사용한다.

```powershell
python -m src.archive.v109_build_v104_public_probe --project . --source-artifact-root reproduction_inputs/parent_v104 --train-csv data/train.csv --data-dir data --parent-zip artifacts/standalone_champion_1161/standalone_champion_1161.zip --v97-config configs/v97_conditional_direct.json --config configs/v109_v104_public_probe.json --output-dir artifacts/parent_v104_01 --timeout 600
python validate_parent_v104_assets.py --reference-zip ORIGINAL_SUBMITTED_V345.zip --v56-dir artifacts/parent_v56_fm_01 --v104-assets-dir artifacts/parent_v104_01/_training_assets --output-dir artifacts/parent_v104_check_01
```

공개 5행 `test.csv`는 feature-spec export 호환성과 예측 parity 검사에만
사용한다. 타깃을 사용하지 않으며 숨겨진 평가 배치의 집계는 수행하지 않는다.
2026-09-06 재검증에서는 두 parent 트리의 일반 파일이 모두 바이트 단위로
일치했고 세 FM 임베딩의 최대 배열 차이는 0이었다.
