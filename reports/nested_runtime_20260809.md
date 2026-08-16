# Nested runtime audit

- Strict implementation: `src/nested_v2.py`.
- Attempt: full inner callback + outer fixed fit over 2019–2024, six vCPU configuration.
- Result: local execution window exceeded 15 minutes before a complete artifact was emitted; process was stopped to preserve the workspace. No partial prediction was treated as OOF.
- Fast implementation: `src/nested_fast.py` fixes iteration/offset from the immediately preceding cached inner season and performs outer fixed fitting. The first 2021 fold also exceeded the available interactive execution window before producing a complete four-fold artifact.
- Consequence: `v2_nested` and all residual recipes are marked **BLOCKED**, not scored or promoted. Legacy caches remain `v2_frozen_replay` diagnostics only because their outer early stopping is target-dependent.
- This is a computational blocker, not evidence that a candidate improves or worsens Brier. A clean offline run with a longer budget is required before any v6 consideration.
