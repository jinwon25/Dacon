# v21 실행·배치 불변성 검사

이 보고서는 과거 패키지의 실행 시간(Runtime), 최대 메모리(peak), 부모와의 계보, 입력 분할에 따른 예측 차이를 점검합니다. max difference는 최대 절대 차이이며, All gates는 이 실행 계약의 통과 여부입니다. 공식 점수 향상이나 모델 성능의 독립 검증을 뜻하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: submit_v21 validation

- Candidate / parent: `submit_v21.zip` / `submit_v20.zip`
- Runtime: **51.120s**, peak **1402.2MB**
- Batch max absolute difference: **1.110e-16**
- Domain comparison: `{'R_CORE': {'rows': 128, 'mean_difference': 0.0023886396074623583, 'mean_abs_difference': 0.005392807885470714, 'max_abs_difference': 0.015852316917530473}, 'R_ANCHOR': {'rows': 128, 'mean_difference': -2.812144418686078e-05, 'mean_abs_difference': 0.001958328959140024, 'max_abs_difference': 0.005032151647333394}, 'F': {'rows': 128, 'mean_difference': 0.0009392128906768656, 'mean_abs_difference': 0.002447103682266815, 'max_abs_difference': 0.008031164521857093}}`
- All gates: **True**
