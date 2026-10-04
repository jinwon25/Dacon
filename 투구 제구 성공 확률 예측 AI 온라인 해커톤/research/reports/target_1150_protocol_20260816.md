# 당시 목표 점수와 후보 검증 기준

v17을 기준으로 추가 개선을 요구한 과거 계획입니다. 목표 점수·통과 기준은 달성 결과가 아니며 단일 로컬–공개 환산 계수로 성능을 확정하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Public 1150 local decision protocol

- Current champion: `submit_v17.zip`, Public **1093.3213473808**.
- Required gain: **+56.6786526192** Public BSS points.
- Every new axis is compared incrementally with the reconstructed v17 analogue on 2022→2023 and 2023→2024 forward folds.
- “Near 1150” requires minimum fold gain **+40**, mean gain **+50**, latest-fold gain **+45**, worst supported bootstrap p05 **+10**, at least 70% positive months, and no domain worse than -10.
- A single local-to-Public multiplier is not used. The one clean v17→v18 leaderboard contrast already showed that a small local improvement can reverse on Public.
- Because 2024 has been repeatedly inspected, passing the gate creates a controlled submission candidate; it is not independent proof of 1150.

This protocol deliberately separates a useful incremental discovery from the stronger claim that the 1150 target is locally within reach.
