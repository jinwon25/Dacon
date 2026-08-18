# 보고서 빠른 안내

최신 의사결정은 아래 문서부터 확인한다.

1. [`target1170_followup_20260817.md`](target1170_followup_20260817.md): 상단 정정 안내 먼저 확인 — 실제 champion은 v26(공식 Public 1157.97364, 제출 ID `51773`), v27/v28 판정, 신규 구조 실험과 1170 후속 전략
2. [`v26_validation.md`](v26_validation.md), [`v26_validation.json`](v26_validation.json): champion v26 실행 시간, 메모리, 계보, 수식·배치 불변성 근거
3. [`v28_validation.md`](v28_validation.md), [`v28_validation.json`](v28_validation.json): v28 실행 시간, 메모리, 계보, 수식·배치 불변성 및 기각 근거
5. [`v25_public_result_20260817.md`](v25_public_result_20260817.md): v26 직접 부모의 post-break R_ANCHOR 모델과 세 시간축 검증
6. [`v24_semantic_eda_20260817.md`](v24_semantic_eda_20260817.md): 공식 R/F 의미, ASOF·실패유형 복원, team 13 구조와 커뮤니티 감사
7. [`v22_public_result_20260817.md`](v22_public_result_20260817.md): v25 직접 부모의 공식 Public 결과
8. [`v23_structural_audit_20260817.md`](v23_structural_audit_20260817.md): v22 이후 여섯 구조 모델의 시간축 감사와 기각 근거
9. [`dacon_official_compliance_audit_20260816.md`](dacon_official_compliance_audit_20260816.md): 공식 규정·FAQ 대비 준수 감사
10. [`submissions.csv`](submissions.csv): 전체 공식 제출 이력의 단일 기준표

## 보관 원칙

- `target1170_*`, `v27_*`, `v28_*`, `v29_*`: 현재 champion v26과 이번 후속 연구·기각 기록
- `v25_*`: v26 직접 부모와 post-break 연구 기록
- `v22_*`: v25 직접 부모와 직전 연구 기록
- `v24_*`: 공식 의미·심층 EDA와 커뮤니티 활용 감사, 추가 기각 기록
- `v23_*`: v22 이후 구조 후보의 감사·기각 기록
- `v21_*`와 `target1200_*`: v22 직접 부모와 직전 연구 기록
- `v20_*`와 `target1160_*`: v21 직접 부모와 이전 champion 기록
- `v19_*`와 `target1150_*`: v20 직접 부모와 이전 champion 기록
- `v17_*`, `v18_*`, `v13_*`: 직접 계보와 비교 근거
- 날짜가 붙은 나머지 보고서: 과거 실험·기각 가설의 재현 기록
- `top1100/`: 2026-08-09 연구 사이클 기록

보고서는 생성 산출물과 달리 실험 판단 근거이므로 삭제하지 않는다. 실제 ZIP 위치는 [`../submissions/README.md`](../submissions/README.md)에서 확인한다.
