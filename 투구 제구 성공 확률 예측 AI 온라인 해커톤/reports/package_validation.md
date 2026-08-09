# 제출 패키지 검증

- 검증 시각: 2026-08-09T14:08:57.983268+09:00
- ZIP: `submit_v5.zip`
- ZIP 크기 / 압축 해제 크기: **12.316 MB / 14.300 MB**
- 최상위 구조: **model/, script.py, requirements.txt**
- 멤버: `['model/ensemble.json', 'model/feature_spec.json', 'model/hybrid.json', 'model/lgb_model.txt', 'model/metadata.json', 'model/rf_model.joblib', 'model/rf_r_only.joblib', 'model/rf_recency_h1.joblib', 'model/trackman_feature_spec.json', 'model/trackman_lgb_model.txt', 'model/trackman_pitcher_profiles.csv', 'requirements.txt', 'script.py']`
- 공식 5행 sample 실행: **통과** (`Saved C:\Users\yun72_92xubzr\AppData\Local\Temp\aimers9_verify_81ucqh9x\package\output\submission.csv rows=5 elapsed=0.623s`)
- 출력 컬럼: **['row_id', 'control_success']**, 행 수: **5**
- test `row_id` 값·순서 보존: **통과**
- 확률 numeric/finite/[0,1]: **통과**
- 대표 **245,789행** CSV 로드+피처+모델 추론: **23.014초**
- 대표 배치 peak RSS: **1441.5 MB**
- 평가 10분 제한 대비: **통과** (동일 행 수 실측이 600초 미만)
- 평가 28GB RAM 제한 대비: **통과**
- 인터넷 호출 정적 검사: **없음**
- test 내부 groupby/value_counts/rank/rolling/expanding 정적 검사: **없음**
- 최종 feature set: **engineered + hybrid**; test 전체 통계 사용: **없음**
- 검증 환경: Python **3.11.2**, pandas **2.2.3**, numpy **2.2.6**, LightGBM **4.6.0**

주의: 대표 배치는 실제 비공개 target이 없는 관계로 학습 데이터의 첫 245,789행을 오직 실행 시간·메모리·shape 검증에만 사용했다. 이 배치에서 성능 점수는 계산하지 않았다.
