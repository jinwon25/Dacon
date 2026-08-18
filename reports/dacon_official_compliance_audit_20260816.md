# DACON 공식 규정 준수 감사 — 2026-08-16

> **2026-08-17 보완:** 공식 FAQ는 `R=Regular(1군 정규시즌)`, `F=Futures(퓨처스리그/2군)`라고 확인했다. 익명 투수·타자 TrackMan ID 대응, 과거 TrackMan 요약, 행별 ASOF에서 동결 train 누적 차감, 학습 범위 내 현재 투구 보조 타깃과 리더보드 기반 고정 가중치 선택도 허용된다고 답변했다. 최신 LUPI 세부 질의는 확인 시점에 미답변이며, 상세 감사는 [`v24_semantic_eda_20260817.md`](v24_semantic_eda_20260817.md)를 따른다.

## 감사 결론

v21은 확인한 공식 규칙과 운영진 답변에 위배되는 요소가 없다. 모든 예측은 평가 행 하나의 투구 직전 정보와 train에서 미리 동결한 artifact만 사용한다.

## 확인한 공식 조건

| 조건 | 공식 내용 | v21 확인 |
|---|---|---|
| 평가 | Brier Skill Score, 음수는 0 처리 | 확률 직접 최적화, `[0,1]` 범위 |
| 평가 범위 | Public 100%, 종료 후 Private가 Public으로 전환 | 전체 Public 결과를 진단에만 사용 |
| 행 독립성 | 각 test 행은 다른 test 행과 독립적으로 예측 | 분할 배치 오차 `1.11e-16` |
| test 정보 | 다른 test 행·분포·그룹·순서·집계 사용 금지 | 사용 없음 |
| 외부 데이터 | 제공 데이터 외 외부 데이터 금지 | 사용 없음 |
| TrackMan | 과거 프로필·train 학습 보조 허용, 현재 test 투구 TrackMan 불가 | v20 PFD의 학습 teacher에서만 사용 |
| 실행 환경 | Ubuntu 22.04.5, Python 3.11.15, L4, 6 vCPU, 28GB RAM | Python 3.11 로컬 재현, CPU 추론 |
| 시간·메모리 | 설치 10분, 추론 10분 | 실제 245,789행 `51.12초`, `1.40GB` |
| ZIP | 압축 10GB 이하, 해제 32GB 이하 | `31.089 MiB` |
| 오프라인 | 설치 후 인터넷 없이 추론 | 네트워크 코드 없음 |

## v21 데이터 계보

1. v19 OOF는 2024 각 행의 정답을 직접 본 in-sample 예측이 아니라 시간 순서를 지킨 검증 예측이다.
2. v21 lookup은 `2024 target - v19 OOF`만 집계한다.
3. 월 감쇠, group sum, effective count, alpha, weight를 학습 시 동결한다.
4. 평가 시 현재 행의 `pitcher_id`, 손 조합, 카운트, 이닝, 주자 상황, ASOF 최근 경기 비율만으로 동결 key를 조회한다.
5. 평가 데이터 전체의 평균·빈도·정렬·이전/다음 행을 참조하지 않는다.

## 자동 게이트

- archive lineage, sample/full rows, 확률 범위: 통과
- runtime `<120s`, memory `<4GB`: 통과
- batch invariance, 세 도메인 변화: 통과
- network free, row-local, test aggregate 미사용, external data 미사용: 통과

## 공식 출처

- 대회 설명: https://dacon.io/competitions/official/236743/overview/description
- 평가: https://dacon.io/competitions/official/236743/overview/evaluation
- 규칙: https://dacon.io/competitions/official/236743/overview/rules
- 데이터·실행 환경: https://dacon.io/competitions/official/236743/data
- FAQ: https://dacon.io/competitions/official/236743/talkboard/417082
- 독립 예측 공지: https://dacon.io/competitions/official/236743/talkboard/417123
- 규칙 재안내: https://dacon.io/competitions/official/236743/talkboard/417094
- 제출·팀 병합 공지: https://dacon.io/competitions/official/236743/talkboard/417093
