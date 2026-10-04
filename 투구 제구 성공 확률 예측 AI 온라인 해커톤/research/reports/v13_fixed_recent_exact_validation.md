# v13 실행·배치 불변성 검사

이 보고서는 과거 패키지의 실행 시간(Runtime), 최대 메모리(peak), 부모와의 계보, 입력 분할에 따른 예측 차이를 점검합니다. max difference는 최대 절대 차이이며, All gates는 이 실행 계약의 통과 여부입니다. 공식 점수 향상이나 모델 성능의 독립 검증을 뜻하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: v13 recent exact-ASOF submission validation

- Candidate / parent: `submit_v13_fixed.zip` / `submit_v11.zip`
- ZIP size: **25.906 MB** compressed, **65.854 MB** extracted
- Exact parent lineage: **True**
- Official sample smoke: **pass**, 8.974s, peak 273.5 MB
- Representative 245,789-row archive inference: **pass**, 32.193s, peak 1418.4 MB
- Split-batch invariance: **True**, max absolute difference 1.110e-16
- R_ANCHOR unchanged from v11: **True**
- R_CORE overlay active: **True**
- F overlay active: **True**
- Offline and row-local static audit: **True**
- Domain comparison: `{'R_CORE': {'rows': 64, 'mean_difference': -0.04711409366785897, 'mean_abs_difference': 0.04711409366785897, 'max_abs_difference': 0.06392963458762524}, 'R_ANCHOR': {'rows': 64, 'mean_difference': 0.0, 'mean_abs_difference': 1.734723475976807e-17, 'max_abs_difference': 1.1102230246251565e-16}, 'F': {'rows': 64, 'mean_difference': -0.024816964800865943, 'mean_abs_difference': 0.024816964800865943, 'max_abs_difference': 0.039756114857424474}}`
- All gates: **True**
