# 목표 점수와 확률 오차 개선 규모

당시 목표와 Brier 오차 개선의 근사 환산, 사전에 정한 통과 조건을 기록합니다. 정답 비율에 의존하는 근사치이며 실제 공개 점수나 보장된 개선으로 읽지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Score ladder

At base rate near 0.5, approximate score-equivalent improvements are:

| ΔBrier | Score gain |
|---:|---:|
| -0.00005 | +20 |
| -0.00010 | +40 |
| -0.00025 | +100 |
| -0.00050 | +200 |
| -0.00084 | +336 |

The 763.27→1100 gap therefore requires roughly `-0.00084` Brier, not a single calibration or blend-weight adjustment. The first promotion gate is fixed at 2024 Δ≤-0.00010, recency-weighted four-fold Δ≤-0.00010, worst fold≤+0.00005 and pitcher-season bootstrap negative probability≥0.95.
