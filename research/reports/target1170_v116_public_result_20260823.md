# v116 Public 결과 — 2026-08-23

## 결론

v116 `submit_v116.zip`의 Public 점수는 사용자 제출 화면 기준 **1161.5978**이다. 현재
champion v104 `1162.6302840289`보다 **-1.0324840289** 낮으므로 v116을 기각하고
`standalone_champion_1162.zip`을 유지한다.

## 사전 근거와 실제 전이

v116은 2025 label이나 Public 점수를 전혀 사용하지 않고, 2022와 late-2023을 서로 교차해
ridge 강도와 세 가지 실패유형 가중치를 동결했다. exact v104 OOF 대비 gain은 full-2022
`+3.6698`, late-2023 `+12.1839`, full-2024 `+5.4305`, late-2024 `+6.2139`였고,
full-2024 8개 월 중 7개가 양수였다. 반면 투수 및 투수×타자 강건성 p05는 음수였다.

Public에서는 로컬 평균 개선과 반대로 `-1.0325`가 나왔다. 따라서 역사적 구종 선택과
투수·타자손·카운트별 reverse/middle/wayoff 실패분해는 2025로 안정적으로 전이되는 직교축이
아니다. 이 결과를 이용해 mask, 가중치, ridge 강도나 적용 범위를 재조정하지 않으며 v113~v116
failure-prior 계열을 종료한다.

## 패키지·제출 감사

- 제출 파일: `artifacts/v117_v116_public_probe_20260823_01/submit_v116.zip`
- SHA-256: `44D7C1C2104A5795E42BE7FB22E03AA106D9AB8D8877F68A7800D65BC7733FF6`
- DACON API: `isSubmitted=true`, `detail=Success`
- 추론 감사: formula 오차 `5.55e-17`, shuffle/partition 최대 `1.11e-16`
- 245,789행 실행 시간: `115.887초`
- Public 점수 출처: 사용자 제출 화면. 공식 best-only leaderboard는 v104만 표시한다.

## 다음 결정

v82와 v84는 local gain이 Public으로 같은 방향 전이했지만, v116은 반전했다. 이후 후보는
v104의 성공축인 strict current-state R_CORE 및 F interaction과 독립이면서, 최소 두 temporal
origin에서 exact-parent gain과 그룹 강건성을 함께 만족해야 한다. 같은 failure-prior 신호의
Public 기반 미세조정은 금지한다.
