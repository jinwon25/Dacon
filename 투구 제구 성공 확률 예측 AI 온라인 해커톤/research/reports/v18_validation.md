# v18 실행·배치 불변성 검사

이 보고서는 과거 패키지의 실행 시간(Runtime), 최대 메모리(peak), 부모와의 계보, 입력 분할에 따른 예측 차이를 점검합니다. max difference는 최대 절대 차이이며, All gates는 이 실행 계약의 통과 여부입니다. 공식 점수 향상이나 모델 성능의 독립 검증을 뜻하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: v16 residual package validation

- Candidate / parent: `submit_v18.zip` / `submit_v15.zip`
- Representative runtime: **31.223s**, peak **1450.6MB**
- Batch max absolute difference: **1.110e-16**
- Domain comparison: `{'R_CORE': {'rows': 96, 'mean_difference': 0.0018756426424827854, 'mean_abs_difference': 0.004896523907406345, 'max_abs_difference': 0.015257234526876107}, 'R_ANCHOR': {'rows': 64, 'mean_difference': -8.673617379884035e-18, 'mean_abs_difference': 1.0408340855860843e-17, 'max_abs_difference': 1.1102230246251565e-16}, 'F': {'rows': 64, 'mean_difference': 0.0, 'mean_abs_difference': 0.0, 'max_abs_difference': 0.0}}`
- All gates: **True**
