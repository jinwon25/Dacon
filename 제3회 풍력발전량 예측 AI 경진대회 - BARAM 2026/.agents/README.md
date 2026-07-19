# Competition Scientist contracts

`.agents/competition.json` is the current competition plug-in. `baram.json` contains the generic
control-plane policy and the BARAM deployment settings; role contracts and reusable JSON inputs
live beside it. Runtime state, logs, and task leases are kept in the ignored
`artifacts_final/agent_service/` tree.

The service is provider-neutral. A hosted or local agent may claim a bounded role task, but
deterministic Python owns validation approval, allowlisted module execution, experiment lineage,
promotion, selection, submission auditing, budgets, and leaderboard feedback.

```text
competition profile -> approved validation plan -> hypothesis
  -> parent/child experiment tree -> safe execution -> validation evidence
  -> local_best (measurement) -> submission_candidate (selection)
  -> schema/budget/credential guard -> DACON submit -> score sync
  -> selected or archived -> failed-family feedback -> next branch
```

Automatic DACON submission is armed for this deployment, but no API call occurs without the local
CLI `--execute-submissions` flag plus `DACON_API_TOKEN` and `DACON_TEAM_NAME`. Network API callers
can inspect eligibility but cannot trigger the external side effect.

See `docs/agent_service.md` for setup and operating procedures.
