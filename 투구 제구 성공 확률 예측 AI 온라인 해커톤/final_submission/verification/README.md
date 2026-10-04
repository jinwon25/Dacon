# 검증 결과

- `tests_summary.json`: 최종 통합 코드의 전체 테스트 실행 요약이다.
- `entrypoints_summary.json`: 재현 문서의 프로젝트 진입점 30개 로드 검사 결과다.

- `xgb_final_sensitivity.json`: 253,507-row controlled pseudo-2025 audit. Replacing only fallback XGBoost with the CPU fresh fit leaves 188,256 rows unchanged; final MAE 0.000202543 and RMS 0.000761740. It is not a hidden-score estimate or proof of byte-identical CUDA training.
- `rebuild.json`: 통합 프로젝트에서 v343 체크포인트와 공식 train으로 Beta를 다시
  적합했다. v345 SHA-256 일치, CRC 통과, Beta 253,507행 최대 오차 0.0.
- `feature_check.json`: 공통화한 시즌 평활 8피처를 공식 train 1,475,092행 전체에서
  이전 계산과 비교해 float32 결과가 모두 정확히 일치했다.
- H1 번들의 세 모델 설정 대조 통과. depth 8 / 1200 iterations / seed 42·43·44.
- `h1_fresh_fit.json`: H1 3시드를 실제 신규 학습해 트리·경계·잎 값 일치.
  32,773행의 입력과 예측도 동일하며 예측 최대 차이 0.
- `xgb_training_precision.json`: 공식 train 1,475,092행의 재구성 114피처에서
  원 XGBoost 분할 경계 8,666개 모두 확인. 원 행렬 전체의 동일성 증명은 아니다.
- `xgb_fresh_fit.json`: XGBoost 3.2.0 CPU 전체 신규 학습은 원 모델과 불일치.
  32,773행의 예측 최대 차이 0.04281276, RMS 차이 0.00854285.
- `futures_fresh_fit.json`: 퓨처스 5시드 전체 신규 학습의 트리·CTR·경계와
  학습 행 30,010개의 예측이 원 모델과 일치, 최대 차이 0.
- 저차원 조회표 11개 배열, 선수 전이 조회표 9개 배열을 공식 train과 보존된
  wave0 OOF로 재계산해 모두 정확히 일치했다. OOF 자체의 신규 학습은 아니다.
- 이후 `wave0_fresh_fit.json`에서 2020–2024 OOF도 모두 신규 학습했다.
  LGB 예측 최대 차이 0, RF/혼합 최대 차이 6.67e-16 이내다.
- `lookup_refit_parity.json`: 새 OOF만 입력으로 두 조회표를 다시 학습했다.
  저차원 11개 배열은 정확히 일치, 선수 전이는 보정값 최대 차이 3.47e-18이며
  나머지 8개 배열은 정확히 일치했다. 기존 OOF 파일은 비교에만 사용했다.
- `training_coverage.json`: 각 런타임 구성요소의 실제 검증 범위와 미완료 단계.
- `initial_oof_fresh_fit.json`: 초기 V2 OOF 4개 연도 fold를 캐시 없이 새로
  학습했으며 숫자 참조와 최대 4.44e-16 이내로 일치했다.
- `legacy_fresh_fit.json`, `futures_initial_fresh_fit.json`,
  `corrected_fresh_fit.json`, `advanced_fresh_fit.json`: 부모 계보의 해당 native
  모델·피처 사양·고정 학습 통계가 신규 학습 결과와 일치한다. 후속 조립 dose와
  v124/v124_bridge 전체 신규 적합까지 검증했다는 뜻은 아니다.
- `parent_base_training.json`, `parent_base_fresh_fit.json`: 공식 train 전체로
  기본 LightGBM 54 rounds와 RandomForest 100 trees를 새로 적합했다. LightGBM
  dump, FeatureBuilder 예측 상태, RandomForest의 모든 트리 노드가 정확히 일치했다.
  최신 FeatureBuilder가 추가한 빈 `drop_columns` 직렬화 기본값은 별도로 기록했다.
- `parent_trackman_training.json`, `parent_trackman_fresh_fit.json`: 공식
  train·TrackMan으로 투수 프로필을 다시 만들고 41-round LightGBM을 적합했다.
  모델·피처 사양·프로필 CSV가 모두 수치 차이 0으로 일치했다.
- `parent_recency_rf_training.json`, `parent_recency_rf_fresh_fit.json`:
  공식 train 전체와 시즌 half-life 1.0 가중치로 RandomForest 100개 트리를
  새로 적합했으며 모든 트리 상태가 수치 차이 0으로 일치했다.
- `parent_recent_exact_training.json`, `parent_recent_exact_fresh_fit.json`:
  2024 공식 train과 해시 고정 OOF 3개로 recent-exact의 5개 native 자산을
  다시 적합했고 모두 수치 차이 0으로 일치했다. 이 실행은 입력 OOF 3개를
  새로 적합하지 않으며 그 사실과 입력 해시를 training 보고서에 기록했다.
- `parent_v14_training.json`, `parent_v14_fresh_fit.json`: anchor Ridge,
  F-trend LightGBM, 전처리 상태가 정확히 일치했다. 제출 spec의 1.3825배
  최종 dose와 선택 주석은 모델 적합 비교에서 분리해 기록했다.
- `parent_joint_state_training.json`, `parent_joint_state_fresh_fit.json`:
  state 회귀 8개, failure-mode 분류 3개, outcome 1개, preprocess/spec까지
  14개 native 자산을 공식 train 전체로 재학습해 모두 수치 차이 0으로 일치했다.
- `parent_v20_training.json`, `parent_v20_fresh_fit.json`: 공식
  train·TrackMan과 해시 고정 joint OOF·정렬 입력으로 teacher 3-fold 및 PFD
  모델을 재학습했다. 모델·EB 상태는 일치하고 v124의 0.7875 dose는 분리했다.
- `parent_v25_training.json`, `parent_v25_fresh_fit.json`: 공식 train과
  해시 고정 선택 감사 요약으로 post-break anchor 모델을 재학습해 일치했다.
  제출의 최종 0.3285 dose 및 리더보드 선택 주석은 별도로 기록했다.
- `parent_v21_training.json`, `parent_v21_fresh_fit.json`: 공식 train과
  해시 고정 joint OOF에서 6개 context EB 표를 재생성해 일치했다.
  v124의 최종 0.575 dose는 별도 기록했다.
- `parent_v22_training.json`, `parent_v22_fresh_fit.json`: model-free
  low-variance 사양을 재생성해 anchor·prior 상태가 일치했다.
  v124의 최종 0.7025 dose는 별도 기록했다.
- `parent_v104_native_fit.json`: 공식 train과 해시 고정된 공식-train 파생
  OOF 7개로 v56 shared FM, v104의 older/recent R-FM, conditional/baseline
  LightGBM 및 bank/spec을 새로 학습했다. `model/v124`와
  `model/v124_bridge` 양쪽에서 모든 일반 파일은 바이트 단위로, 세 FM의
  모든 임베딩 배열은 최대 차이 0으로 일치했다.
- `workload_training.json`, `workload_fresh_fit.json`: 등판량 H1 3시드 전체를
  새로 적합했고 트리·경계·잎 값 및 32,773행 예측이 일치했다.
- `c3_oof_training.json`, `c3_fresh_fit.json`: 5개 연도 OOF를 새로 적합한 뒤
  실제 배포 v148 설정으로 C3 테이블을 재생성했고 값이 모두 일치했다.
- `strict_fresh_fit.json`: strict EXP-021의 6단계를 새로 실행해 9개 native
  자산과 공개 샘플 예측이 일치했다.
- 통합 `train.py lookups` 실행으로 공식 train·TrackMan 기반 lookup 재생성 대조 통과.
  카테고리, 앵커, 상황, 매치업, 등판량, TrackMan 프로필의 최대 오차 모두 0.
- strict 6단계 import/경로 및 실행 소스 13개 무결성 검사 통과.

`runtime_audit.json`은 동일 SHA-256인 원 v345 추론 ZIP의 기존 검사 결과다.
공개 5행의 단일행·셔플·분할 최대 차이 1.11e-16,
공개 샘플 기반 245,789행 부하 실행 316.193초다. 숨겨진 평가 데이터 실행이 아니다.
이번 폴더 통합 후 이 부하 검사를 재실행했다고 주장하지 않는다.

전체 원데이터 신규 학습 재현과 위 검사를 구분한다.
남은 재현 항목은 `../04_SUBMISSION_CHECKLIST.md`에 있다.
중복 재빌드 ZIP은 정리했으며 재빌드 보고서의 기존 출력 경로는 남아 있지 않을 수 있다.
원 추론 ZIP은 프로젝트 원본 위치에 보존했다. 이 학습 패키지에서는
`code/train.py rebuild`로 재생성하고 위 SHA-256과 대조한다.
