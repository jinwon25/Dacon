# fallback XGBoost

`fallback_xgb_frozen_runtime.py`가 고정 lookup으로 행별 114피처를 만들고
`fallback_xgb.json`에서 보조 확률을 계산한다.
최종 혼합 경로는 `../fallback_routes/` 및 v345 추론 코드에 고정돼 있다.

## 파일과 실행 범위

- `feature_columns.json`: 모델 입력 114열과 순서
- `fallback_lookups.joblib`: 공식 train/TrackMan 기반 고정 통계
- `pitcher_map.csv`: 공식 선수 ID와 TrackMan ID 대응
- `build_fallback_xgb_lookups.py`: 고정 통계 재생성
- `build_fallback_xgb_model.py`: 명시적 허용 시 재구성 모델을 학습하는 실험 경로
- `check_fallback_xgb_feature_parity.py`: lookup·입력 열·모델 입력 폭 대조
- `smoke_row_independence.py`: 행 독립성 점검

`code/`에서 실행한다.

```powershell
python train.py lookups --data-dir data --output-dir artifacts/lookups_01
python fallback_xgb/check_fallback_xgb_feature_parity.py --skip-lookups --test-csv data/test.csv
```

lookup·피처 폭 일치만으로 원본 학습 행렬이나 booster 재학습 동일성을 증명하지 않는다.
원 GPU 학습 결과와의 정확한 booster 동일성은 아직 확인하지 못했다.
`--allow-reconstructed-training`은 별도 실험의 명시적 선택이며 최종 추론 ZIP을 수정하지 않는다.

예전 `rebuild_fallback_xgb_oof.py`는 실행하지 않는다.
OOF는 검증 시즌 이전 데이터로 lookup을 적합하는
`src.archive.v242_runtime_faithful_fallback_oof`를 사용한다.
학습용 전체 프레임 집계 함수는 평가 행에 적용하지 않는다.
