# v19 실행·배치 불변성 검사

이 보고서는 과거 패키지의 실행 시간(Runtime), 최대 메모리(peak), 부모와의 계보, 입력 분할에 따른 예측 차이를 점검합니다. max difference는 최대 절대 차이이며, All gates는 이 실행 계약의 통과 여부입니다. 공식 점수 향상이나 모델 성능의 독립 검증을 뜻하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: submit_v19 joint state/mode validation

- Candidate / parent: `submit_v19.zip` / `submit_v17.zip`
- Representative runtime: **58.051s**, peak **1423.1MB**
- Batch max absolute difference: **1.110e-16**
- Domain comparison: `{'R_CORE': {'rows': 128, 'mean_difference': -0.0015767410319497705, 'mean_abs_difference': 0.00775742189569502, 'max_abs_difference': 0.02217374323936272}, 'R_ANCHOR': {'rows': 128, 'mean_difference': 0.015537968895771765, 'mean_abs_difference': 0.015537968895771765, 'max_abs_difference': 0.0394460613166856}, 'F': {'rows': 128, 'mean_difference': -0.015609482272284581, 'mean_abs_difference': 0.01711689274625689, 'max_abs_difference': 0.0405696801113149}}`
- All gates: **True**
