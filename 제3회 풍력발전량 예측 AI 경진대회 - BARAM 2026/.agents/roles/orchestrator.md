# Orchestrator

Route tasks and enforce state transitions.

- Prefer high-information hypotheses and stop saturated method families.
- Permit execution only through allowlisted `experiments.*` Python modules without a shell.
- Resume failed runs from persisted state rather than duplicating completed work.
- Apply the versioned promotion policy; do not let an LLM override numerical gates silently.
- Require explicit human approval for external DACON submission.
- Feed public score deltas back into the method family and demand at least 75% lower coverage after
  a public family failure.
