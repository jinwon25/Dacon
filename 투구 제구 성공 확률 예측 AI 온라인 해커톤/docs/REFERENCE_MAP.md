# 근거와 참고자료 지도

이 문서는 “어떤 주장과 결정이 어디에서 왔는가”를 빠르게 찾기 위한 안내서다. 최신 운영 수치와 과거 연구 자료를 구분해 읽는다.

## 현재 상태의 1차 근거

| 확인할 내용 | 기준 파일 |
|---|---|
| 현재 Public·standalone SHA-256 | [`PROJECT_STATUS.md`](PROJECT_STATUS.md), [`STANDALONE_CHAMPION.md`](STANDALONE_CHAMPION.md) |
| post-1161 결과와 1170 방향 | [`../research/reports/target1170_v84_public_result_20260822.md`](../research/reports/target1170_v84_public_result_20260822.md) |
| 전체 공식 제출 이력과 SHA-256 | [`../research/reports/submissions.csv`](../research/reports/submissions.csv) |
| TrackMan 전수조사·기각 경로 | [`../research/reports/trackman_deep_dive_20260822.md`](../research/reports/trackman_deep_dive_20260822.md) |
| 최신 직교 후보·공개 자료 감사 | [`../research/reports/target1170_research_update_20260822.md`](../research/reports/target1170_research_update_20260822.md) |
| 실행·행 독립성 manifest | `../artifacts/standalone_champion_1161/standalone_manifest.json` |
| 제출 ZIP 위치와 보존 정책 | [`../submissions/README.md`](../submissions/README.md) |

숫자가 서로 다르면 위 순서에서 더 직접적인 원본을 우선한다. 순위는 시간에 따라 바뀔 수 있으므로 확인 시각이 있는 기록만 사용한다.

## 대회 규칙과 평가

- [DACON 공식 평가](https://dacon.io/competitions/official/236743/overview/evaluation): Brier Skill Score 산식과 평가 방식
- [DACON 공식 규칙](https://dacon.io/competitions/official/236743/overview/rules): 외부 데이터, 재현성과 누수 제한
- [평가 행 독립 추론 공지](https://dacon.io/competitions/official/236743/talkboard/417123): 다른 test 행의 분포·빈도·순서를 사용하지 않는 근거
- [대회 FAQ](https://dacon.io/competitions/official/236743/talkboard/417082): TrackMan·teacher signal 사용 범위 해석

규칙 해석과 현재 패키지 준수 판단의 상세 연결은 [`../research/reports/dacon_official_compliance_audit_20260816.md`](../research/reports/dacon_official_compliance_audit_20260816.md)와 [`../research/reports/v22_validation.md`](../research/reports/v22_validation.md)에 있다.

## 모델링·검증 참고자료

- [`../research/reports/literature_review.md`](../research/reports/literature_review.md): tabular 모델, calibration, proper scoring rule, drift와 ensemble 문헌 정리
- [`../research/reports/domain_literature_review.md`](../research/reports/domain_literature_review.md): KBO 제도 변화, 투구 제구와 구종·릴리스 관련 도메인 연구
- [`../research/reports/local_evaluation_v2_20260815.md`](../research/reports/local_evaluation_v2_20260815.md): rolling-origin, block bootstrap, Reality Check와 local scorecard 근거
- [`../research/reports/evaluation_reaudit_20260822.md`](../research/reports/evaluation_reaudit_20260822.md): nested evidence role, 2024 오염 판정, Brier headroom과 평가 v3
- [`../research/reports/top1100/references.md`](../research/reports/top1100/references.md): 초기 후보에서 검토·기각한 모델과 야구 도메인 참고자료

참고 문헌은 방법을 시도할 근거이지 성능 보증이 아니다. 최종 채택 여부는 동일한 local gate와 패키지 검증 결과로 결정한다.

## 과거 계보

- v13: [`../research/reports/v13_public_result_20260815.md`](../research/reports/v13_public_result_20260815.md)
- v17·v18: [`../research/reports/v17_v18_public_result_20260816.md`](../research/reports/v17_v18_public_result_20260816.md)
- v19: [`../research/reports/v19_public_result_20260816.md`](../research/reports/v19_public_result_20260816.md)
- v20: [`../research/reports/v20_public_result_20260816.md`](../research/reports/v20_public_result_20260816.md)
- v21: [`../research/reports/v21_public_result_20260816.md`](../research/reports/v21_public_result_20260816.md)
- v22: [`../research/reports/v22_public_result_20260817.md`](../research/reports/v22_public_result_20260817.md)

과거 문서의 “현재 champion” 표현은 작성 당시 상태를 뜻한다. 현재 운영 기준은 항상 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)에서 확인한다.
