# v4/v5 후보 패키지 검증

검증일: 2026-08-09

| 파일 | SHA-256 | 변경 | 검증 결과 |
|---|---|---|---|
| `submit_v4.zip` | `7D41C2AEA487F772DF878CD53DEFB79EE9C1AFD9BF6EE2968EC8ECA19EE54E6A` | R-only RF 25%, Trackman 0% | 통과 |
| `submit_v5.zip` | `0EF5A2E393D796243F8D2F834C6C9C3E78FB77A942E71CB1FCC09046C333907C` | R-only RF 25%, Trackman 5% | 통과 |

공통 확인 사항:

- 파일명 13자(40자 제한 이내), 최상위 구조 `model/`, `script.py`, `requirements.txt`.
- 공식 sample 5행 실행, row_id 순서 보존, finite numeric 확률 `[0,1]` 모두 통과.
- 245,789행 대표 배치 검사도 통과. v4 약 7.35초/747MB, v5 약 23.01초/1,442MB peak RSS.
- 인터넷 호출 및 test 내부 groupby/value_counts/rank/rolling/expanding 정적 검사 없음.

v4는 R-only branch의 공개 효과를 분리하는 1차 제출용이다. v5는 v4가 개선된 경우에만 Trackman 5% 상호작용을 확인하는 2차 제출용으로 보류한다.
