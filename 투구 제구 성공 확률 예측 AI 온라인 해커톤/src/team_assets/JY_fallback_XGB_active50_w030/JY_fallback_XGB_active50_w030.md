# JY fallback XGB active50 w0.30 — Public 1175 재현 가이드

마지막 정리: 2026-08-29 KST

## 확정 결과

이 릴리스는 Public `1172.1373858439` JY 챔피언 위에 독립 XGBoost 확률을 제한적으로 혼합한 후보다. 사용자가 확인한 Public 결과는 **1175점대**다. 정확한 소수점 점수는 전달받지 않았으므로 문서에서는 `1175 reported`로 기록한다.

최종 제출 파일:

`submit_jy_fallback_xgb_active50_w030_public1175.zip`

현재 보존본 SHA-256:

`A7BF242D5D4003CAF0B19857C83441CFD49838E353826AF29435AC0C3763A8C7`

부모 JY ZIP:

`parent/submit_jy_runners_high_li_bridge027_public1172.zip`

부모 SHA-256:

`4C924E046091304BFC73B50BE51110BDF1351DFD9B577738CC6B65A8DFF43C9E`

## 1. 바로 추론하기

ZIP을 새 폴더에 풀고 공식 `test.csv`, `sample_submission.csv`를 `data/`에 넣는다.

```powershell
$run = Join-Path $PWD "run_public1175"
New-Item -ItemType Directory -Force $run | Out-Null
Expand-Archive .\submit_jy_fallback_xgb_active50_w030_public1175.zip $run -Force
New-Item -ItemType Directory -Force (Join-Path $run "data") | Out-Null
Copy-Item <공식데이터경로>\test.csv (Join-Path $run "data\test.csv")
Copy-Item <공식데이터경로>\sample_submission.csv (Join-Path $run "data\sample_submission.csv")
python -m pip install -r (Join-Path $run "requirements.txt")
python (Join-Path $run "script.py")
```

결과는 `run_public1175/output/submission.csv`에 생성된다.

## 2. 제출 ZIP 다시 만들기

이 릴리스 폴더는 부모 ZIP과 frozen XGB 자산을 함께 보존하므로 공식 train을 다시 학습하지 않고 동일한 추론 수식의 ZIP을 재조립할 수 있다.

```powershell
python .\scripts\build_jy_xgb_active_high50_zip.py
```

출력:

`rebuilt/submit_jy_fallback_xgb_active50_w030.zip`

ZIP 압축 시각 때문에 재빌드 ZIP의 바이트 SHA는 달라질 수 있다. 검증 대상은 부모 파일 보존, 수식, frozen model/lookup 해시, CRC 및 예측 parity다.

## 3. 모델을 처음부터 다시 학습하기

완전 재학습에는 다음이 필요하다.

- 공식 `train.csv`
- 공식 `trackman_history.csv`
- `pitcher_map.csv`
- 원본 JY 1172 부모 ZIP
- Python 3.11
- `numpy`, `pandas`, `scikit-learn`, `xgboost 3.2.0`, JY 부모 requirements

재학습 순서:

```powershell
python scripts\rebuild_xgb_fallback_oof.py
python scripts\evaluate_jy_xgb_fallback_active.py
python scripts\build_jy_fallback_xgb_asset.py
python scripts\build_jy_fallback_xgb_lookup_asset.py
python scripts\check_fallback_xgb_feature_parity.py
python scripts\smoke_jy_fallback_xgb_asset.py
python scripts\build_jy_xgb_active_high50_zip.py
```

`rebuild_xgb_fallback_oof.py`와 학습 스크립트는 공개 Hyunku 재현 코드의 `repro_v10a_core.py`를 호출한다. 저장소를 다른 위치로 옮길 경우 스크립트 상단의 프로젝트 탐색 경로를 새 위치에 맞춰야 한다. 제출 추론 ZIP 자체는 이 외부 코드나 공식 train을 요구하지 않는다.

## 4. 행 독립성 규칙

최종 XGB runtime은 test의 다른 행을 집계하지 않는다.

- 카테고리 vocabulary: 공식 train에서 고정
- pitcher/batter anchor: 공식 train에서 고정
- situation·matchup·TrackMan lookup: 공식 train에서 고정
- 결측 대체값: 공식 train에서 고정
- test 행 간 rolling, groupby, 평균, 중앙값, 빈도, 순위: 사용하지 않음

개발 중 `p_ppa` 결측을 현재 배치 중앙값으로 채우는 문제가 발견되었고, 이를 공식 train 기반 고정값으로 교체했다. 수정 후 XGB 모듈의 단일행·셔플 예측 최대 차이는 `0.0`이었다.

## 5. 보존 정책

Public 1175 제출 ZIP, 부모 1172 ZIP, frozen asset, evidence CSV, 빌드 스크립트는 삭제하지 않는다. 새 실험은 이 폴더를 수정하지 않고 별도 후보 폴더에서 시작한다.

