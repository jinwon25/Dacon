# 도메인 후보 제출 패키지 검증

- 후보 ZIP: `submit_candidate_game_type_regime_v1_clean.zip`
- ZIP / 압축 해제 크기: **4.055 MB / 5.034 MB**
- 최상위 구조: **model/, script.py, requirements.txt**
- 멤버: `['model/ensemble.json', 'model/feature_spec.json', 'model/game_type_offsets.json', 'model/lgb_model.txt', 'model/metadata.json', 'model/rf_model.joblib', 'requirements.txt', 'script.py']`
- 공식 5행 smoke test: **통과**, 4.914초 (`Saved C:\Users\yun72_92xubzr\AppData\Local\Temp\aimers9_candidate_verify_254syd4v\submit_candidate_game_type_regime_v1_clean\output\submission.csv rows=5 elapsed=0.665s`)
- 245,789행 실제 archive 추론: **12.850초**, child peak RSS **696.9 MB**
- 출력 행 수·row_id 순서·numeric/finite/[0,1]: **통과**
- 동일 20행을 전체/교차 2개 batch로 나눈 예측의 수치 일치: **True**, 최대 절대차 **2.220e-16**
- `R` probe가 incumbent와 수치 동일: **True**, 최대 절대차 **2.220e-16**
- `F` probe가 frozen negative offset으로 모두 하향: **True**
- frozen offset: `{'F': -0.10798353030903143, 'R': 0.0}`
- test 전체 집계·순서·빈도 사용: **없음**
- 인터넷 호출: **없음**
- feature set: **engineered**

대표 배치는 실제 비공개 target 성능 평가가 아니라 archive 실행 시간·메모리·shape 검증에만 사용했다.
