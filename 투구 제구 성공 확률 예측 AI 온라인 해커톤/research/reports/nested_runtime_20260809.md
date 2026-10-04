# 중첩 시간 검증의 실행 한계

엄격한 중첩 학습은 실행 시간 안에 전체 산출물을 만들지 못했습니다. 부분 예측을 완성된 학습 외 예측으로 간주하지 않았다는 실행 기록입니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Nested runtime audit

- Strict implementation: `src/nested_v2.py`.
- Attempt: full inner callback + outer fixed fit over 2019–2024, six vCPU configuration.
- Result: local execution window exceeded 15 minutes before a complete artifact was emitted; process was stopped to preserve the workspace. No partial prediction was treated as OOF.
- Fast implementation: `src/nested_fast.py` fixes iteration/offset from the immediately preceding cached inner season and performs outer fixed fitting. The first 2021 fold also exceeded the available interactive execution window before producing a complete four-fold artifact.
- Consequence: `v2_nested` and all residual recipes are marked **BLOCKED**, not scored or promoted. Legacy caches remain `v2_frozen_replay` diagnostics only because their outer early stopping is target-dependent.
- This is a computational blocker, not evidence that a candidate improves or worsens Brier. A clean offline run with a longer budget is required before any v6 consideration.
