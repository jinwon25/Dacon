# 근거와 참고자료 지도

이 문서는 “어떤 주장과 결정이 어디에서 왔는가”를 빠르게 찾기 위한 안내서다. 최신 운영 수치와 과거 연구 자료를 구분해 읽는다.

## 현재 상태의 1차 근거

| 확인할 내용 | 기준 파일 |
|---|---|
| 공식 제출 ID·점수·순위·API 응답 | [`../reports/v19_public_result_20260816.md`](../reports/v19_public_result_20260816.md) |
| 전체 공식 제출 이력과 SHA-256 | [`../reports/submissions.csv`](../reports/submissions.csv) |
| v19 로컬 게이트와 후보 선택 | [`../reports/target1150_joint_candidate_20260816.md`](../reports/target1150_joint_candidate_20260816.md) |
| 실행·메모리·배치 불변성 | [`../reports/v19_validation.json`](../reports/v19_validation.json) |
| 최종 학습 파일 해시 | `../artifacts/state_mode_joint_final_20260816/manifest.json` |
| 제출 ZIP 위치와 보존 정책 | [`../submissions/README.md`](../submissions/README.md) |

숫자가 서로 다르면 위 순서에서 더 직접적인 원본을 우선한다. 순위는 시간에 따라 바뀔 수 있으므로 확인 시각이 있는 기록만 사용한다.

## 대회 규칙과 평가

- [DACON 공식 평가](https://dacon.io/competitions/official/236743/overview/evaluation): Brier Skill Score 산식과 평가 방식
- [DACON 공식 규칙](https://dacon.io/competitions/official/236743/overview/rules): 외부 데이터, 재현성과 누수 제한
- [평가 행 독립 추론 공지](https://dacon.io/competitions/official/236743/talkboard/417123): 다른 test 행의 분포·빈도·순서를 사용하지 않는 근거
- [대회 FAQ](https://dacon.io/competitions/official/236743/talkboard/417082): TrackMan·teacher signal 사용 범위 해석

규칙 해석과 v19 준수 판단의 상세 연결은 [`../reports/target1150_joint_candidate_20260816.md`](../reports/target1150_joint_candidate_20260816.md)에 있다.

## 모델링·검증 참고자료

- [`../reports/literature_review.md`](../reports/literature_review.md): tabular 모델, calibration, proper scoring rule, drift와 ensemble 문헌 정리
- [`../reports/domain_literature_review.md`](../reports/domain_literature_review.md): KBO 제도 변화, 투구 제구와 구종·릴리스 관련 도메인 연구
- [`../reports/local_evaluation_v2_20260815.md`](../reports/local_evaluation_v2_20260815.md): rolling-origin, block bootstrap, Reality Check와 local scorecard 근거
- [`../reports/top1100/references.md`](../reports/top1100/references.md): 초기 후보에서 검토·기각한 모델과 야구 도메인 참고자료

참고 문헌은 방법을 시도할 근거이지 성능 보증이 아니다. 최종 채택 여부는 동일한 local gate와 패키지 검증 결과로 결정한다.

## 과거 계보

- v13: [`../reports/v13_public_result_20260815.md`](../reports/v13_public_result_20260815.md)
- v17·v18: [`../reports/v17_v18_public_result_20260816.md`](../reports/v17_v18_public_result_20260816.md)
- v19: [`../reports/v19_public_result_20260816.md`](../reports/v19_public_result_20260816.md)

과거 문서의 “현재 champion” 표현은 작성 당시 상태를 뜻한다. 현재 운영 기준은 항상 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)에서 확인한다.
