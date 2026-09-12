# CatBoost research status

CatBoost is retained as a research candidate.

- 2024 fixed outer screen: Brier `0.248273754`.
- Legacy v2 frozen-replay 2024 Brier: `0.248440533`.
- Point estimate delta: `-0.000166533`.
- 2021–2023 CatBoost OOF: not run.
- Honest v2 OOF: blocked by strict nested runtime.
- Recency-weighted delta, worst-fold delta, seed replication and pitcher-cluster bootstrap: not evaluable.

Interpretation: this is a promising 2024 signal, not a failed model family and not a submission candidate. The next permitted experiment is a fixed CatBoost logit-offset residual with at most two preregistered configurations, trained and selected only from prior-origin OOF.
