# 과거 상태 특징 정의

이 특징 정의는 `src/corrected_top1100_features.py`와 분리합니다. 과거 CatBoost 축은 이전 endpoint-shift 방식으로 학습했으므로 정의를 바꾸면 다른 모델이 됩니다. 두 방식 모두 공식 학습 이력을 사용하며 평가 배치의 통계를 사용하지 않습니다.

`code/`에서 실행하는 `src.finalize_legacy_cb_axis`는 `--research-project legacy_state`를 받습니다. 학습 자료 위치는 `--data-project`로 별도 지정합니다. 열 이름이 같다는 이유로 과거 경로의 특징 정의를 수정된 정의로 대체하지 않습니다.
