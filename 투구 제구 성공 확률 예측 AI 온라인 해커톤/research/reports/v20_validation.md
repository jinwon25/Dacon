# v20 실행·배치 불변성 검사

이 보고서는 과거 패키지의 실행 시간(Runtime), 최대 메모리(peak), 부모와의 계보, 입력 분할에 따른 예측 차이를 점검합니다. max difference는 최대 절대 차이이며, All gates는 이 실행 계약의 통과 여부입니다. 공식 점수 향상이나 모델 성능의 독립 검증을 뜻하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: submit_v20 target-1160 validation

- Candidate / parent: `submit_v20.zip` / `submit_v19.zip`
- Representative runtime: **93.274s**, peak **1434.9MB**
- Batch max absolute difference: **1.110e-16**
- Domain comparison: `{'R_CORE': {'rows': 128, 'mean_difference': -0.00162243600635446, 'mean_abs_difference': 0.0033164965504028946, 'max_abs_difference': 0.011880317713646005}, 'R_ANCHOR': {'rows': 128, 'mean_difference': -0.0022676701647402534, 'mean_abs_difference': 0.004915171857563495, 'max_abs_difference': 0.013219025387634642}, 'F': {'rows': 128, 'mean_difference': -0.0008482012734493112, 'mean_abs_difference': 0.0033142862890073997, 'max_abs_difference': 0.01168155116069769}}`
- All gates: **True**
