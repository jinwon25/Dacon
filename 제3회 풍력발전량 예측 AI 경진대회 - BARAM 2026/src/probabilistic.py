"""Distribution post-processing and official-metric Bayes point actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_QUANTILE_LEVELS = np.asarray(
    [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    dtype=float,
)


def enforce_noncrossing(quantiles: np.ndarray) -> np.ndarray:
    """Apply the documented row-wise monotone repair for crossing quantiles."""

    values = np.asarray(quantiles, dtype=float)
    if values.ndim != 2:
        raise ValueError("quantiles must be a two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("quantiles contain non-finite values")
    return np.sort(values, axis=1)


def quantile_scenario_weights(levels: np.ndarray) -> np.ndarray:
    """Return probability masses for a discrete quantile-function quadrature.

    Boundaries lie halfway between adjacent quantile levels, with outer mass
    extending to zero and one. This avoids treating the intentionally uneven
    0.05/0.10/.../0.90/0.95 grid as equally spaced samples.
    """

    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or len(levels) < 2:
        raise ValueError("at least two quantile levels are required")
    if not np.all((levels > 0.0) & (levels < 1.0)) or np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
    edges = np.concatenate(([0.0], 0.5 * (levels[:-1] + levels[1:]), [1.0]))
    weights = np.diff(edges)
    return weights / weights.sum()


def settlement_price(error_kwh: np.ndarray, capacity: float) -> np.ndarray:
    """Official unit price: 4 inside 6%, 3 inside 8%, otherwise zero."""

    error = np.asarray(error_kwh, dtype=float)
    return np.where(error <= 0.06 * capacity, 4.0, np.where(error <= 0.08 * capacity, 3.0, 0.0))


def expected_metric_utility(
    actions: np.ndarray,
    scenarios: np.ndarray,
    *,
    capacity: float,
    mean_eligible_generation: float,
    scenario_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the conditional utility exactly proportional to group Score.

    For eligible outcomes, multiplying the prediction-dependent part of the
    official group Score by ``2 * capacity`` gives

    ``-|p-y| + capacity/(4*mean_y) * y * price(|p-y|)``.

    The returned expectation has shape ``(rows, actions)``.
    """

    actions = np.asarray(actions, dtype=float)
    scenarios = np.asarray(scenarios, dtype=float)
    if actions.ndim != 2 or scenarios.ndim != 2 or actions.shape[0] != scenarios.shape[0]:
        raise ValueError("actions and scenarios must be aligned two-dimensional arrays")
    if capacity <= 0.0 or mean_eligible_generation <= 0.0:
        raise ValueError("capacity and mean eligible generation must be positive")
    if not np.isfinite(actions).all() or not np.isfinite(scenarios).all():
        raise ValueError("actions and scenarios must be finite")
    if scenario_weights is None:
        weights = np.full(scenarios.shape[1], 1.0 / scenarios.shape[1])
    else:
        weights = np.asarray(scenario_weights, dtype=float)
        if weights.shape != (scenarios.shape[1],) or np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("scenario weights must be non-negative and sum to one")
    error = np.abs(actions[:, :, None] - scenarios[:, None, :])
    eligible = scenarios[:, None, :] >= 0.10 * capacity
    kappa = capacity / (4.0 * mean_eligible_generation)
    utility = -error + kappa * scenarios[:, None, :] * settlement_price(error, capacity)
    utility = np.where(eligible, utility, 0.0)
    return np.sum(utility * weights[None, None, :], axis=2)


@dataclass(frozen=True)
class BayesActionResult:
    action: np.ndarray
    reference: np.ndarray
    expected_advantage: np.ndarray
    selected_candidate: np.ndarray
    candidate_count: int


def metric_aware_bayes_action(
    quantiles: np.ndarray,
    *,
    capacity: float,
    mean_eligible_generation: float,
    levels: np.ndarray = DEFAULT_QUANTILE_LEVELS,
    point_candidates: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    offset_grid: tuple[float, ...] = (-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04),
    batch_size: int = 512,
) -> BayesActionResult:
    """Choose the expected official-utility maximizer for each row.

    Candidate actions contain every quantile, each quantile shifted by the
    6%/8% cliffs, the p50/reference neighborhood, and optional point-model
    predictions. Ties are resolved toward the reference to limit movement.
    """

    levels = np.asarray(levels, dtype=float)
    scenarios = enforce_noncrossing(quantiles)
    if scenarios.shape[1] != len(levels):
        raise ValueError("quantile columns and levels do not match")
    p50_position = int(np.argmin(np.abs(levels - 0.50)))
    anchor = scenarios[:, p50_position] if reference is None else np.asarray(reference, dtype=float)
    if anchor.shape != (len(scenarios),):
        raise ValueError("reference must have one value per row")
    optional = np.empty((len(scenarios), 0), dtype=float)
    if point_candidates is not None:
        optional = np.asarray(point_candidates, dtype=float)
        if optional.ndim == 1:
            optional = optional[:, None]
        if optional.ndim != 2 or optional.shape[0] != len(scenarios):
            raise ValueError("point candidates must align with quantile rows")
    cliff_offsets = capacity * np.asarray([-0.08, -0.06, 0.06, 0.08])
    shifted = (scenarios[:, :, None] + cliff_offsets[None, None, :]).reshape(len(scenarios), -1)
    local = anchor[:, None] + capacity * np.asarray(offset_grid)[None, :]
    candidates = np.clip(np.concatenate([anchor[:, None], optional, scenarios, shifted, local], axis=1), 0.0, capacity)
    scenario_weights = quantile_scenario_weights(levels)
    action = np.empty(len(scenarios), dtype=float)
    advantage = np.empty(len(scenarios), dtype=float)
    selected = np.empty(len(scenarios), dtype=int)
    for start in range(0, len(scenarios), batch_size):
        stop = min(start + batch_size, len(scenarios))
        utility = expected_metric_utility(
            candidates[start:stop],
            scenarios[start:stop],
            capacity=capacity,
            mean_eligible_generation=mean_eligible_generation,
            scenario_weights=scenario_weights,
        )
        best = utility.max(axis=1, keepdims=True)
        on_plateau = utility >= best - 1e-10
        distance = np.abs(candidates[start:stop] - anchor[start:stop, None])
        choice = np.argmin(np.where(on_plateau, distance, np.inf), axis=1)
        row = np.arange(stop - start)
        anchor_utility = utility[:, 0]
        action[start:stop] = candidates[start:stop][row, choice]
        advantage[start:stop] = utility[row, choice] - anchor_utility
        selected[start:stop] = choice
    return BayesActionResult(action, anchor.copy(), advantage, selected, candidates.shape[1])


def shrink_action(reference: np.ndarray, action: np.ndarray, alpha: float, capacity: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return np.clip((1.0 - alpha) * np.asarray(reference, dtype=float) + alpha * np.asarray(action, dtype=float), 0.0, capacity)
