# 보고서 빠른 안내

- [`target1170_v94_v96_multiorigin_20260822.md`](target1170_v94_v96_multiorigin_20260822.md): **최신 multi-origin 감사** — v94 fidelity 계약, v95 context 연도 전이 반전, v96 연중 적응 진단과 공개 자료 재감사
- [`target1170_v92_v93_pipeline_audit_20260822.md`](target1170_v92_v93_pipeline_audit_20260822.md): **최신 데이터·파이프라인 전수 감사** — v92/v93 기각, v87 의존성 bootstrap, TrackMan 한계와 exact multi-origin runner 우선순위

최신 의사결정은 아래 문서부터 확인한다.

- [`target1170_v85_v91_followup_20260822.md`](target1170_v85_v91_followup_20260822.md): **최신 후속 검증** — v85~v91 기각, 1161 champion 유지, 다음 nested 구조 모델 경로

1. [`target1170_v84_public_result_20260822.md`](target1170_v84_public_result_20260822.md): **최신 결과·실행 계획** — Public 1161.2020600422, standalone 승격, 1170 후속 경로
2. [`target1170_v83_post_public_plan_20260822.md`](target1170_v83_post_public_plan_20260822.md): v84 제출 전 고정 v56 선택 근거
3. [`target1170_v79_v81_followup_20260822.md`](target1170_v79_v81_followup_20260822.md): champion offset GLM, exact OOF 공분산 stack, 환경 안정 shallow GBDT 기각
4. [`target1170_v77_v78_followup_20260822.md`](target1170_v77_v78_followup_20260822.md): 팀 exact OOF 계약·constrained blend와 환경 안정 Ridge 기각
5. [`evaluation_reaudit_20260822.md`](evaluation_reaudit_20260822.md): 공식 평가 재감사, 선택 편향, v75/v76과 1170 실행 우선순위
6. [`trackman_deep_dive_20260822.md`](trackman_deep_dive_20260822.md): TrackMan 전수조사, v65~v74 시간축 실험과 폐쇄 경로
7. [`target1170_research_update_20260822.md`](target1170_research_update_20260822.md): 최종 gate exact OOF, v62~v64 직교·count·물리 모델 결과, 팀 브랜치 감사와 최신 문헌
8. [`target1170_eta15_followup_20260822.md`](target1170_eta15_followup_20260822.md): 1158.0746 단일 릴리스, eta=0.15 재감사, v58~v60 결과와 공식 행 독립성
9. [`0819_jy_tmgate.md`](0819_jy_tmgate.md): 직전 champion의 최종 TrackMan-ASOF gate와 Public `+0.1009148862`
10. [`../docs/STANDALONE_CHAMPION.md`](../docs/STANDALONE_CHAMPION.md): 과거 ZIP 없이 실행·전달하는 단일 패키지 절차
11. [`target1170_followup_20260817.md`](target1170_followup_20260817.md): 역사 기록 — v26(1157.97364), v27/v28 판정과 당시 후속 전략
12. [`v26_validation.md`](v26_validation.md), [`v26_validation.json`](v26_validation.json): 직접 부모 v26의 실행 시간, 메모리, 계보와 배치 불변성
13. [`v28_validation.md`](v28_validation.md), [`v28_validation.json`](v28_validation.json): v28 실행·계보·수식·배치 불변성 및 기각 근거
14. [`v25_public_result_20260817.md`](v25_public_result_20260817.md): v26 직접 부모의 post-break R_ANCHOR 모델과 세 시간축 검증
15. [`v24_semantic_eda_20260817.md`](v24_semantic_eda_20260817.md): 공식 R/F 의미, ASOF·실패유형 복원, team 13 구조와 커뮤니티 감사
16. [`v22_public_result_20260817.md`](v22_public_result_20260817.md): v25 직접 부모의 공식 Public 결과
17. [`v23_structural_audit_20260817.md`](v23_structural_audit_20260817.md): v22 이후 구조 후보의 감사·기각 근거
18. [`dacon_official_compliance_audit_20260816.md`](dacon_official_compliance_audit_20260816.md): 공식 규정·FAQ 대비 준수 감사
19. [`submissions.csv`](submissions.csv): 전체 공식 제출 이력의 단일 기준표

## 보관 원칙

- `target1170_v84_*`, `target1170_v83_*`, `target1170_v82_*`, `target1170_v79_v81_*`, `target1170_v77_v78_*`, `evaluation_reaudit_*`, `trackman_deep_dive_*`, `target1170_eta15_*`, `0819_*`: 현재 champion 1161.2020600422와 최신 후속 연구
- 나머지 `target1170_*`, `v27_*`~`v60_*`: 과거 기준선 및 기각 실험 기록
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
