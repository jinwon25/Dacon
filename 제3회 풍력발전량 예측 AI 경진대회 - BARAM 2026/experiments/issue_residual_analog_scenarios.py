"""Issue-level analog residual scenarios for the group-3 settlement objective.

The experiment treats one complete 24-hour NWP issue as the dependency unit.
For each query issue, three independently standardized feature views rank the
historical issue archive:

* the three incumbent power trajectories;
* LDAPS weather plus the power trajectories;
* GFS weather plus the power trajectories.

The nearest residual trajectories from the three views are pooled with equal
view weight.  They form a nonparametric predictive distribution that preserves
the observed 24-hour error paths.  A bounded action is then selected against
the official NMAE/FICR settlement utility and conservatively blended into only
the highest-advantage six-hour blocks.

This is a diagnostic experiment.  It never writes a submission.  In particular,
the 2024 H2 period was inspected while the method was prototyped on 2026-07-28,
so the report must not describe it as pristine confirmation evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import (
    assign_issue_blocks,
    evaluate_blocked_rolling,
    load_issue_times,
)
from experiments.spatiotemporal_consensus_promotion import _rolling_finesweep_base
from src.metrics import CAPACITY_KWH, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
Q2_START = pd.Timestamp("2024-04-01")
H2_START = pd.Timestamp("2024-07-01")
EXPECTED_LEADS = np.arange(12, 36, dtype=int)
ACTION_RATIOS = np.asarray(
    (
        -0.0150,
        -0.0125,
        -0.0100,
        -0.0075,
        -0.0050,
        -0.0025,
        0.0,
        0.0025,
        0.0050,
        0.0075,
        0.0100,
        0.0125,
        0.0150,
    ),
    dtype=float,
)
VIEW_NAMES = ("power", "ldaps_power", "gfs_power")
LDAPS_COLUMNS = (
    "ldaps__kpx_group_3__hub_ws117__idw",
    "ldaps__kpx_group_3__hub_u117__idw",
    "ldaps__kpx_group_3__hub_v117__idw",
    "ldaps__kpx_group_3__surface_0_sp__idw",
)
GFS_COLUMNS = (
    "gfs__kpx_group_3__hub_ws117__idw",
    "gfs__kpx_group_3__hub_u117__idw",
    "gfs__kpx_group_3__hub_v117__idw",
    "gfs__kpx_group_3__surface_0_gust__idw",
    "gfs__kpx_group_3__surface_0_sp__idw",
)


@dataclass(frozen=True)
class Policy:
    neighbors: int
    alpha: float
    coverage: float
    min_positive_fraction: float

    @property
    def name(self) -> str:
        return (
            f"k{self.neighbors}_a{int(round(self.alpha * 100))}_"
            f"c{int(round(self.coverage * 100))}_"
            f"p{int(round(self.min_positive_fraction * 100))}"
        )


@dataclass
class IssueTrajectories:
    issues: pd.DatetimeIndex
    representatives: pd.DatetimeIndex
    rows: list[np.ndarray]
    views: dict[str, np.ndarray]
    residual_ratio: np.ndarray


def policies(
    min_positive_fractions: tuple[float, ...] = (0.0, 0.65, 0.80),
) -> tuple[Policy, ...]:
    """Small, literature-guided policy grid fixed before the formal run."""
    return tuple(
        Policy(neighbors, alpha, coverage, min_positive_fraction)
        for neighbors in (12, 24, 36)
        for alpha in (0.25, 0.50)
        for coverage in (0.10, 0.25, 0.50)
        for min_positive_fraction in min_positive_fractions
    )


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_issue_trajectories(
    index: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
    raw_views: dict[str, np.ndarray],
    truth: np.ndarray,
    base: np.ndarray,
) -> IssueTrajectories:
    """Collect only finite, complete lead-12..35 issue trajectories."""
    if len(index) != len(issue_times):
        raise ValueError("index and issue_times must have the same length")
    if set(raw_views) != set(VIEW_NAMES):
        raise ValueError(f"Expected views {VIEW_NAMES}, got {tuple(raw_views)}")
    if any(len(values) != len(index) for values in raw_views.values()):
        raise ValueError("Every raw feature view must align with index")
    truth = np.asarray(truth, dtype=float)
    base = np.asarray(base, dtype=float)
    if truth.shape != (len(index),) or base.shape != (len(index),):
        raise ValueError("truth and base must be aligned one-dimensional arrays")

    lead = ((index - issue_times) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    issue_values = np.asarray(issue_times)
    complete_issues: list[pd.Timestamp] = []
    representatives: list[pd.Timestamp] = []
    rows: list[np.ndarray] = []
    views: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_NAMES}
    residuals: list[np.ndarray] = []

    for issue in pd.unique(issue_times):
        positions = np.flatnonzero(issue_values == issue)
        positions = positions[np.argsort(lead[positions])]
        if len(positions) != len(EXPECTED_LEADS):
            continue
        if not np.array_equal(lead[positions], EXPECTED_LEADS):
            continue
        if not np.isfinite(truth[positions]).all() or not np.isfinite(base[positions]).all():
            continue
        if any(not np.isfinite(raw_views[name][positions]).all() for name in VIEW_NAMES):
            continue
        complete_issues.append(pd.Timestamp(issue))
        representatives.append(pd.Timestamp(index[positions[len(positions) // 2]]))
        rows.append(positions)
        for name in VIEW_NAMES:
            views[name].append(np.asarray(raw_views[name][positions], dtype=float))
        residuals.append((truth[positions] - base[positions]) / CAPACITY)

    if not complete_issues:
        raise ValueError("No complete finite issue trajectories")
    stacked_views = {name: np.stack(values) for name, values in views.items()}
    return IssueTrajectories(
        issues=pd.DatetimeIndex(complete_issues),
        representatives=pd.DatetimeIndex(representatives),
        rows=rows,
        views=stacked_views,
        residual_ratio=np.stack(residuals),
    )


def _standardization_scales(
    trajectories: IssueTrajectories,
    train_positions: np.ndarray,
) -> dict[str, np.ndarray]:
    scales: dict[str, np.ndarray] = {}
    for name in VIEW_NAMES:
        values = trajectories.views[name][train_positions]
        scale = values.reshape(-1, values.shape[2]).std(axis=0)
        scale[scale < 1e-7] = 1.0
        scales[name] = scale
    return scales


def _rank_pool_neighbors(
    trajectories: IssueTrajectories,
    train_positions: np.ndarray,
    query_position: int,
    neighbors: int,
    scales: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool equally weighted nearest-neighbour ranks from independent views."""
    if query_position in set(train_positions.tolist()):
        raise ValueError("A query issue cannot occur in its analog archive")
    selected: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for name in VIEW_NAMES:
        values = trajectories.views[name]
        distance = np.sqrt(
            np.mean(
                ((values[train_positions] - values[query_position]) / scales[name]) ** 2,
                axis=(1, 2),
            )
        )
        count = min(int(neighbors), len(train_positions))
        nearest_local = np.argpartition(distance, count - 1)[:count]
        nearest = train_positions[nearest_local]
        view_weight = 1.0 / np.maximum(distance[nearest_local], 1e-6)
        view_weight /= view_weight.sum()
        selected.append(nearest)
        weights.append(view_weight / len(VIEW_NAMES))
    pooled_positions = np.concatenate(selected)
    pooled_weights = np.concatenate(weights)
    pooled_weights /= pooled_weights.sum()
    return pooled_positions, pooled_weights


def _expected_actions(
    base_path: np.ndarray,
    residual_scenarios: np.ndarray,
    scenario_weights: np.ndarray,
    eligible_probability: float,
    eligible_generation: float,
    min_positive_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximise a sample analogue of the official row-level score."""
    scenario_generation = np.clip(
        base_path[:, None] + residual_scenarios.T * CAPACITY,
        0.0,
        CAPACITY,
    )
    candidates = np.clip(
        base_path[:, None] + ACTION_RATIOS[None, :] * CAPACITY,
        0.0,
        CAPACITY,
    )
    error = (
        np.abs(scenario_generation[:, :, None] - candidates[:, None, :]) / CAPACITY
    )
    eligible = scenario_generation >= 0.10 * CAPACITY
    units = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    weights = scenario_weights[None, :, None]
    scenario_utility = (
        -0.5
        * eligible[:, :, None]
        * error
        / max(float(eligible_probability), 1e-6)
        + 0.5
        * eligible[:, :, None]
        * scenario_generation[:, :, None]
        * units
        / 4.0
        / max(float(eligible_generation), 1.0)
    )
    zero_position = int(np.flatnonzero(np.isclose(ACTION_RATIOS, 0.0))[0])
    scenario_advantage = (
        scenario_utility - scenario_utility[:, :, zero_position, None]
    )
    positive_fraction = (
        (scenario_advantage > 0.0) * scenario_weights[None, :, None]
    ).sum(axis=1)
    utility = (scenario_utility * weights).sum(axis=1)
    eligible_action = positive_fraction >= float(min_positive_fraction)
    eligible_action[:, zero_position] = True
    utility = np.where(eligible_action, utility, -np.inf)
    # Settlement thresholds create broad utility plateaus.  NumPy's argmax
    # would select the first (most negative) action on an exact tie, causing
    # needless movement.  Resolve ties toward the smallest absolute action.
    best_utility = utility.max(axis=1, keepdims=True)
    tied = np.isclose(utility, best_utility, rtol=0.0, atol=1e-12)
    action_order = np.argsort(np.abs(ACTION_RATIOS), kind="stable")
    best = np.empty(len(base_path), dtype=int)
    for row in range(len(base_path)):
        best[row] = int(next(position for position in action_order if tied[row, position]))
    advantage = utility[np.arange(len(base_path)), best] - utility[:, zero_position]
    return ACTION_RATIOS[best], advantage


def apply_policy(
    trajectories: IssueTrajectories,
    base: np.ndarray,
    train_mask: np.ndarray,
    query_mask: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit an analog archive and change only selected complete query blocks."""
    train_positions = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    query_positions = np.flatnonzero(np.asarray(query_mask, dtype=bool))
    if not len(train_positions) or not len(query_positions):
        raise ValueError("Analog train and query issue sets must be non-empty")
    if set(train_positions).intersection(query_positions):
        raise ValueError("Analog train and query issues overlap")
    base = np.asarray(base, dtype=float)
    scales = _standardization_scales(trajectories, train_positions)
    train_truth = np.concatenate(
        [
            base[trajectories.rows[position]]
            + trajectories.residual_ratio[position] * CAPACITY
            for position in train_positions
        ]
    )
    train_truth = np.clip(train_truth, 0.0, CAPACITY)
    train_eligible = train_truth >= 0.10 * CAPACITY
    eligible_probability = float(train_eligible.mean())
    eligible_generation = float((train_truth * train_eligible).mean())

    decisions: list[tuple[int, np.ndarray, np.ndarray]] = []
    neighbor_support: list[int] = []
    for query_position in query_positions:
        neighbors, weights = _rank_pool_neighbors(
            trajectories,
            train_positions,
            int(query_position),
            policy.neighbors,
            scales,
        )
        action, advantage = _expected_actions(
            base[trajectories.rows[query_position]],
            trajectories.residual_ratio[neighbors],
            weights,
            eligible_probability,
            eligible_generation,
            policy.min_positive_fraction,
        )
        decisions.append((int(query_position), action, advantage))
        neighbor_support.append(int(len(np.unique(neighbors))))

    blocks: list[tuple[float, int, int, np.ndarray]] = []
    for query_position, action, advantage in decisions:
        for phase in range(4):
            phase_slice = slice(phase * 6, (phase + 1) * 6)
            blocks.append(
                (
                    float(advantage[phase_slice].mean()),
                    query_position,
                    phase,
                    action,
                )
            )
    keep_count = int(np.floor(policy.coverage * len(blocks)))
    eligible_blocks = [item for item in blocks if item[0] > 0.0]
    ranked = sorted(
        eligible_blocks,
        key=lambda item: (item[0], -item[1], -item[2]),
        reverse=True,
    )
    selected = {(item[1], item[2]) for item in ranked[:keep_count]}

    candidate = base.copy()
    changed_rows = 0
    action_counts: dict[str, int] = {}
    for _, query_position, phase, action in blocks:
        if (query_position, phase) not in selected:
            continue
        phase_slice = slice(phase * 6, (phase + 1) * 6)
        row_positions = trajectories.rows[query_position][phase_slice]
        movement = policy.alpha * action[phase_slice] * CAPACITY
        candidate[row_positions] = np.clip(
            candidate[row_positions] + movement, 0.0, CAPACITY
        )
        changed_rows += int(np.count_nonzero(np.abs(movement) > 1e-12))
        for value in action[phase_slice]:
            key = f"{float(value):+.4f}"
            action_counts[key] = action_counts.get(key, 0) + 1
    return candidate, {
        "train_issues": int(len(train_positions)),
        "query_issues": int(len(query_positions)),
        "available_blocks": int(len(blocks)),
        "positive_advantage_blocks": int(len(eligible_blocks)),
        "selected_blocks": int(len(selected)),
        "selected_block_coverage": float(len(selected) / max(len(blocks), 1)),
        "changed_rows": changed_rows,
        "mean_unique_neighbor_support": float(np.mean(neighbor_support)),
        "action_counts_before_alpha": action_counts,
        "eligible_probability": eligible_probability,
        "eligible_generation_kwh_per_archive_row": eligible_generation,
    }


def issue_row_mask(
    trajectories: IssueTrajectories,
    issue_mask: np.ndarray,
    row_count: int,
) -> np.ndarray:
    output = np.zeros(row_count, dtype=bool)
    for position in np.flatnonzero(np.asarray(issue_mask, dtype=bool)):
        output[trajectories.rows[int(position)]] = True
    return output


def evaluate_period(
    index: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    before = evaluate_group(truth[rows], base[rows], CAPACITY)
    after = evaluate_group(truth[rows], candidate[rows], CAPACITY)
    delta = {
        "score": float(after.score - before.score),
        "one_minus_nmae": float(after.one_minus_nmae - before.one_minus_nmae),
        "ficr": float(after.ficr - before.ficr),
    }
    month_blocks, _ = assign_issue_blocks(index, issue_times)
    monthly: dict[str, dict[str, float]] = {}
    for block in dict.fromkeys(month_blocks[rows]):
        mask = rows & (month_blocks == block)
        month_before = evaluate_group(truth[mask], base[mask], CAPACITY)
        month_after = evaluate_group(truth[mask], candidate[mask], CAPACITY)
        monthly[str(block)] = {
            "score": float(month_after.score - month_before.score),
            "one_minus_nmae": float(
                month_after.one_minus_nmae - month_before.one_minus_nmae
            ),
            "ficr": float(month_after.ficr - month_before.ficr),
        }
    monthly_scores = [item["score"] for item in monthly.values()]
    return {
        "base": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": delta,
        "monthly": monthly,
        "months_improved": int(sum(value > 0.0 for value in monthly_scores)),
        "months_evaluated": int(len(monthly_scores)),
        "worst_month_score_delta": float(min(monthly_scores)),
    }


def development_qualifies(evaluation: dict[str, Any]) -> bool:
    """Conservative Q2 selection gate, including every issue-centre month."""
    delta = evaluation["delta"]
    return bool(
        delta["score"] > 0.0
        and delta["one_minus_nmae"] > 0.0
        and delta["ficr"] > 0.0
        and evaluation["worst_month_score_delta"] >= 0.0
    )


def crossfit_issue_months(
    trajectories: IssueTrajectories,
    base: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deployment-shaped sensitivity: every issue month is held out in turn."""
    month = trajectories.representatives.to_period("M").astype(str).to_numpy()
    candidate = np.asarray(base, dtype=float).copy()
    folds: list[dict[str, Any]] = []
    for block in dict.fromkeys(month):
        query = month == block
        train = ~query
        fold_candidate, routing = apply_policy(
            trajectories, base, train, query, policy
        )
        rows = issue_row_mask(trajectories, query, len(base))
        candidate[rows] = fold_candidate[rows]
        folds.append(
            {
                "held_out_issue_month": str(block),
                "train_issues": int(train.sum()),
                "query_issues": int(query.sum()),
                "routing": routing,
            }
        )
    return candidate, {"scheme": "leave-one-issue-centre-month-out", "folds": folds}


def load_validation_inputs(
    labels_path: Path,
    driver_path: Path,
    feature_path: Path,
    issue_path: Path,
) -> tuple[
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    IssueTrajectories,
]:
    driver = np.load(driver_path)
    _, index, truth, group3 = _rolling_finesweep_base(labels_path, driver_path)
    issue_times = load_issue_times(issue_path, index)
    feature_cache = pd.read_pickle(feature_path).reindex(index)
    missing = [
        column
        for column in (*LDAPS_COLUMNS, *GFS_COLUMNS)
        if column not in feature_cache
    ]
    if missing:
        raise KeyError(f"Missing analog feature columns: {missing}")
    power = np.column_stack(
        [
            driver["kpx_group_1__exact_base"].astype(float)
            / CAPACITY_KWH["kpx_group_1"],
            driver["kpx_group_2__exact_base"].astype(float)
            / CAPACITY_KWH["kpx_group_2"],
            group3 / CAPACITY,
        ]
    )
    ldaps = feature_cache[list(LDAPS_COLUMNS)].to_numpy(dtype=float)
    gfs = feature_cache[list(GFS_COLUMNS)].to_numpy(dtype=float)
    raw_views = {
        "power": power,
        "ldaps_power": np.column_stack([ldaps, power]),
        "gfs_power": np.column_stack([gfs, power]),
    }
    trajectories = build_issue_trajectories(
        index, issue_times, raw_views, truth, group3
    )
    return index, issue_times, truth, group3, trajectories


def run(
    labels_path: Path,
    driver_path: Path,
    feature_path: Path,
    issue_path: Path,
    n_bootstrap: int,
    min_positive_fractions: tuple[float, ...] = (0.0, 0.65, 0.80),
) -> dict[str, Any]:
    index, issue_times, truth, base, trajectories = load_validation_inputs(
        labels_path, driver_path, feature_path, issue_path
    )
    q1 = trajectories.representatives < Q2_START
    q2 = (trajectories.representatives >= Q2_START) & (
        trajectories.representatives < H2_START
    )
    h1 = trajectories.representatives < H2_START
    h2 = trajectories.representatives >= H2_START
    q2_rows = issue_row_mask(trajectories, q2, len(index))
    h2_rows = issue_row_mask(trajectories, h2, len(index))

    development: list[dict[str, Any]] = []
    policy_grid = policies(min_positive_fractions)
    for policy in policy_grid:
        candidate, routing = apply_policy(trajectories, base, q1, q2, policy)
        evaluation = evaluate_period(
            index, issue_times, truth, base, candidate, q2_rows
        )
        development.append(
            {
                "policy": asdict(policy),
                "name": policy.name,
                "routing": routing,
                "evaluation": evaluation,
                "qualifies": development_qualifies(evaluation),
            }
        )
    eligible = [item for item in development if item["qualifies"]]
    selected = max(
        eligible,
        key=lambda item: (
            item["evaluation"]["worst_month_score_delta"],
            item["evaluation"]["delta"]["score"],
        ),
        default=None,
    )

    locked: dict[str, Any] | None = None
    crossfit: dict[str, Any] | None = None
    if selected is not None:
        policy = Policy(**selected["policy"])
        candidate, routing = apply_policy(trajectories, base, h1, h2, policy)
        blocked = evaluate_blocked_rolling(
            truth,
            base,
            candidate,
            index,
            issue_times,
            h2_rows,
            n_bootstrap=n_bootstrap,
            seed=20260728,
        )
        locked = {
            "routing": routing,
            "evaluation": evaluate_period(
                index, issue_times, truth, base, candidate, h2_rows
            ),
            "blocked_robustness": blocked,
            "pristine_confirmation": False,
            "reason": (
                "2024 H2 was inspected during method prototyping on 2026-07-28; "
                "retain this as discovery evidence only"
            ),
        }
        crossfit_candidate, crossfit_routing = crossfit_issue_months(
            trajectories, base, policy
        )
        complete_rows = issue_row_mask(
            trajectories,
            np.ones(len(trajectories.issues), dtype=bool),
            len(index),
        )
        crossfit = {
            "routing": crossfit_routing,
            "evaluation": evaluate_period(
                index,
                issue_times,
                truth,
                base,
                crossfit_candidate,
                complete_rows,
            ),
            "interpretation": (
                "cross-fitted structural sensitivity, not a chronological "
                "or pristine promotion fold"
            ),
        }

    automatic_candidate_allowed = False
    return {
        "method": (
            "multi-view rank-pooled analog residual trajectories with "
            "settlement-aware six-hour actions"
        ),
        "sources": [
            "https://doi.org/10.1175/MWR-D-12-00281.1",
            "https://doi.org/10.1016/j.renene.2014.11.061",
            "https://doi.org/10.1016/j.apenergy.2022.118769",
            "https://doi.org/10.1016/j.apenergy.2017.12.039",
            "https://doi.org/10.1002/qj.3137",
        ],
        "lineage": {
            "labels": labels_path.as_posix(),
            "driver": driver_path.as_posix(),
            "features": feature_path.as_posix(),
            "issue_source": issue_path.as_posix(),
            "oof_rows": int(len(index)),
            "complete_issue_cycles": int(len(trajectories.issues)),
            "complete_issue_rows": int(24 * len(trajectories.issues)),
            "discarded_boundary_or_incomplete_rows": int(
                len(index) - 24 * len(trajectories.issues)
            ),
            "lead_hours": [int(EXPECTED_LEADS.min()), int(EXPECTED_LEADS.max())],
            "views": {
                "power": [
                    "group1 exact-base trajectory",
                    "group2 exact-base trajectory",
                    "group3 finesweep incumbent trajectory",
                ],
                "ldaps_power": [*LDAPS_COLUMNS, "power view"],
                "gfs_power": [*GFS_COLUMNS, "power view"],
            },
        },
        "validation_contract": {
            "development": "Q1 archive -> Q2 query; policy selected on Q2",
            "locked_application": "same policy, H1 archive -> H2 query",
            "dependency_unit": "complete 24-hour NWP issue",
            "action_unit": "six-hour lead block",
            "formal_grid_size": int(len(policy_grid)),
            "minimum_scenario_positive_fractions": list(
                min_positive_fractions
            ),
            "q2_strict_gate": (
                "overall score/NMAE/FICR positive and worst issue-month "
                "score delta non-negative"
            ),
            "h2_exposure_disclosure": (
                "H2 was viewed during pre-formal prototyping; it is discovery "
                "rather than pristine confirmation evidence"
            ),
        },
        "development": development,
        "selected_policy": None
        if selected is None
        else {"name": selected["name"], **selected["policy"]},
        "locked_h2_discovery": locked,
        "leave_month_out_crossfit": crossfit,
        "promotion": {
            "automatic_candidate_allowed": automatic_candidate_allowed,
            "candidate_written": False,
            "reason": (
                "no automatic candidate from an H2-exposed discovery branch; "
                "require a genuinely independent confirmation or a controlled "
                "public probe"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--features",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument("--issues", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "issue_residual_analog_scenarios_20260728.json"
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--positive-fractions",
        default="0,0.65,0.80",
        help="comma-separated minimum weighted scenario agreement levels",
    )
    args = parser.parse_args()
    positive_fractions = tuple(
        float(value.strip())
        for value in args.positive_fractions.split(",")
        if value.strip()
    )
    if not positive_fractions or any(
        value < 0.0 or value > 1.0 for value in positive_fractions
    ):
        raise ValueError("positive fractions must be within [0, 1]")
    report = run(
        Path(args.labels),
        Path(args.driver),
        Path(args.features),
        Path(args.issues),
        n_bootstrap=args.n_bootstrap,
        min_positive_fractions=positive_fractions,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_to_builtin(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            _to_builtin(
                {
                    "selected_policy": report["selected_policy"],
                    "locked_h2_discovery": report["locked_h2_discovery"],
                    "promotion": report["promotion"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
