# 보고서 빠른 안내

이 디렉터리는 최종 선택까지의 연구 기록입니다. 먼저 [v345 공식 결과](v345_public_result_20260831.md)와 [최종 재현 안내](../../final_submission/README.md)를 읽고, 아래 목록은 날짜별 의사결정 이력으로 확인합니다. 과거 목록의 “최신·현재”는 작성 당시 표현입니다. 용어는 [프로젝트 문서 안내](../../docs/README.md)를 참고합니다.

## 제출 원장 규칙

[`submissions.csv`](submissions.csv)는 제출 이력의 **권위 있는 단일 출처**다. 집계가
어긋나지 않도록 다음을 지킨다.

- **제출 1건당 정확히 1행.** 같은 제출을 나중에 다른 관점에서 다시 기록할 일이 생기면
  새 행을 추가하지 말고 기존 행의 `local_validation`·`notes`를 확장한다.
- `submission_id`가 비는 경우가 있다. 리더보드 응답이 ID를 주지 않은 제출이며, 그 사정을
  `notes`에 적는다(현재 8건).
- `public_score`가 `<`로 시작하면 best-only 리더보드에서 개별 점수가 노출되지 않아
  상한만 아는 제출이다.
- 판단 근거는 원장이 아니라 이 디렉터리의 보고서에 남기고, 원장에는 링크나 파일명을 적는다.

2026-09-12에 v335(제출 `76835`)가 v290 계보와 v320 포트폴리오 두 관점으로 중복 기록된
것을 한 행으로 병합했다. 두 기록의 내용은 모두 보존했다.


- [`v355_public_result_20260901.md`](v355_public_result_20260901.md): **v355 기각** — Public 1177.9105932872, v345 대비 -5.0391037328; TrackMan-PFD+후반기 계층 결합 계열은 분해 제출·공개점수 재튜닝 없이 종료
- [`target1185_v354_late_hierarchy_candidate_20260901.md`](target1185_v354_late_hierarchy_candidate_20260901.md): **다음 제출 1순위 v354** — v353 + 8월 이후 R_ANCHOR 계층 posterior, 세 시간축 `+3.7582/+7.2679/+4.2106`, 강한 cluster·시간 bootstrap과 재현 ZIP
- [`target1185_v352_v353_trackman_pfd_candidate_20260831.md`](target1185_v352_v353_trackman_pfd_candidate_20260831.md): **v354 직접 부모** — v345와 거의 직교한 training-only TrackMan-PFD, 두 연속 시간 구간 `+0.9761/+0.9521`, 전체-2024 최종 refit과 재현 ZIP
- [`v345_public_result_20260831.md`](v345_public_result_20260831.md): **현재 공식 챔피언** — Public 1182.94969702, v335 대비 +1.2396938587, 1185까지 2.05030298
- [`target1185_v345_candidate_20260831.md`](target1185_v345_candidate_20260831.md): **현 챔피언 재현 근거** — v343 + 고정 Beta 셀, v335 대비 full-2024 `+3.390593`, 재현 SHA·207초 full-scale 감사
- [`target1190_v343_transition_workload_candidate_20260831.md`](target1190_v343_transition_workload_candidate_20260831.md): **v345 부모 연구** — v335 위 선수 전이+3시드 workload, full-2024 `+2.944474`, 7/8개월 개선, v343 런타임 감사
- [`v335_public_result_20260831.md`](v335_public_result_20260831.md): **직전 공식 챔피언** — Public 1181.7100031613, v290 대비 +4.9529314933, v335 구조·감사·귀속 한계
- [`target1185_v328_v335_multiexpert_followup_20260830.md`](target1185_v328_v335_multiexpert_followup_20260830.md): v328–v335 다중 전문가 스크린과 v335 제출 전 근거
- [`target1180_structural_followup_20260830.md`](target1180_structural_followup_20260830.md): **최신 1180 후속 연구** — v264 배포계약 기각, pseudo-deployment v277 탐색 challenger, 독립 실행 감사
- [`target1170_v94_v96_multiorigin_20260822.md`](target1170_v94_v96_multiorigin_20260822.md): **최신 multi-origin 감사** — v94 fidelity 계약, v95 context 연도 전이 반전, v96 연중 적응 진단과 공개 자료 재감사
- [`target1170_v92_v93_pipeline_audit_20260822.md`](target1170_v92_v93_pipeline_audit_20260822.md): **최신 데이터·파이프라인 전수 감사** — v92/v93 기각, v87 의존성 bootstrap, TrackMan 한계와 exact multi-origin runner 우선순위

2026-08-22 당시 의사결정은 아래 문서에 보존한다.

- [`target1170_v85_v91_followup_20260822.md`](target1170_v85_v91_followup_20260822.md): **최신 후속 검증** — v85–v91 기각, 1161 champion 유지, 다음 nested 구조 모델 경로

1. [`target1170_v84_public_result_20260822.md`](target1170_v84_public_result_20260822.md): **최신 결과·실행 계획** — Public 1161.2020600422, standalone 승격, 1170 후속 경로
2. [`target1170_v83_post_public_plan_20260822.md`](target1170_v83_post_public_plan_20260822.md): v84 제출 전 고정 v56 선택 근거
3. [`target1170_v79_v81_followup_20260822.md`](target1170_v79_v81_followup_20260822.md): champion offset GLM, exact OOF 공분산 stack, 환경 안정 shallow GBDT 기각
4. [`target1170_v77_v78_followup_20260822.md`](target1170_v77_v78_followup_20260822.md): 팀 exact OOF 계약·constrained blend와 환경 안정 Ridge 기각
5. [`evaluation_reaudit_20260822.md`](evaluation_reaudit_20260822.md): 공식 평가 재감사, 선택 편향, v75/v76과 1170 실행 우선순위
6. [`trackman_deep_dive_20260822.md`](trackman_deep_dive_20260822.md): TrackMan 전수조사, v65–v74 시간축 실험과 폐쇄 경로
7. [`target1170_research_update_20260822.md`](target1170_research_update_20260822.md): 최종 gate exact OOF, v62–v64 직교·count·물리 모델 결과, 팀 브랜치 감사와 최신 문헌
8. [`target1170_eta15_followup_20260822.md`](target1170_eta15_followup_20260822.md): 1158.0746 단일 릴리스, eta=0.15 재감사, v58–v60 결과와 공식 행 독립성
9. [`0819_jy_tmgate.md`](0819_jy_tmgate.md): 직전 champion의 최종 TrackMan-ASOF gate와 Public `+0.1009148862`
10. [`../../docs/STANDALONE_CHAMPION.md`](../../docs/STANDALONE_CHAMPION.md): 과거 ZIP 없이 실행·전달하는 단일 패키지 절차
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

- `target1170_v84_*`, `target1170_v83_*`, `target1170_v82_*`, `target1170_v79_v81_*`, `target1170_v77_v78_*`, `evaluation_reaudit_*`, `trackman_deep_dive_*`, `target1170_eta15_*`, `0819_*`: 2026-08-22 당시 champion 1161.2020600422와 당시 후속 연구
- 나머지 `target1170_*`, `v27_*`\~`v60_*`: 과거 기준선 및 기각 실험 기록
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

보고서는 생성 산출물과 달리 실험 판단 근거이므로 삭제하지 않는다. 실제 ZIP 위치는 [`../../submissions/README.md`](../../submissions/README.md)에서 확인한다.
