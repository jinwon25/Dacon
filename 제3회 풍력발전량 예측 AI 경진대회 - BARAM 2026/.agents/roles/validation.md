# Validation agent

Audit the experiment independently from the modeling agent.

- Confirm report/candidate timestamps, row alignment, capacities, and groups left unchanged.
- Select parameters only before the locked period; never retune after inspecting locked results.
- Require exact OOF, forward time locking, seed or policy-neighborhood stability, monthly stability,
  and day-level bootstrap.
- Record changed-row ratio and P95 movement relative to capacity.
- Emit the standardized `Evaluation`; the deterministic promotion policy makes the final gate.
- Treat public leaderboard results as stronger evidence than local validation.
