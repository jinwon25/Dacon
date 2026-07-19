"""Robust forward router for the deployable group-3 structural expert.

The router is intentionally diagnostic-only.  It selects policies with monthly
expanding-window backtests from February through June, freezes one policy, and
opens July through December once.  No submission is written by this module.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.dynamic_router_oracle import load_experts
from experiments.weather_similarity_router import (
    H2_START,
    ROOT,
    build_blocks,
    evaluate_period,
    period_blocks,
    utility_table,
)


OUTPUT = (
    ROOT
    / "artifacts_final"
    / "diagnostics"
    / "robust_weather_router_20260718.json"
)
LOCKED_END = pd.Timestamp("2025-01-01")


@dataclass(frozen=True)
class RobustPolicy:
    neighbors: int
    min_positive_fraction: float
    risk_penalty: float
    alpha: float
    coverage: float
    require_positive_source_months: bool

    @property
    def name(self) -> str:
        month_guard = "mg1" if self.require_positive_source_months else "mg0"
        return (
            f"k{self.neighbors}_p{int(self.min_positive_fraction * 100)}_"
            f"r{int(self.risk_penalty * 100)}_a{int(self.alpha * 100)}_"
            f"c{int(self.coverage * 100)}_{month_guard}"
        )


def policies() -> tuple[RobustPolicy, ...]:
    return tuple(
        RobustPolicy(k, positive, risk, alpha, coverage, month_guard)
        for k, positive, risk, alpha, coverage, month_guard in itertools.product(
            (12, 24, 36),
            (0.60, 0.70),
            (0.25, 0.50),
            (0.10, 0.20),
            (0.05, 0.10),
            (False, True),
        )
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
    return value


def route_robust(
    frame: pd.DataFrame,
    rows_by_block: dict[str, np.ndarray],
    experts: dict[str, np.ndarray],
    train_blocks: np.ndarray,
    query_blocks: np.ndarray,
    train_utility: pd.DataFrame,
    policy: RobustPolicy,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Route only when local mean, dispersion, wins, and optional month floor agree."""
    metadata = {"representative_ns", "phase", "rows"}
    columns = [column for column in frame.columns if column not in metadata]
    train_x = frame.loc[train_blocks, columns].to_numpy(dtype=float)
    query_x = frame.loc[query_blocks, columns].to_numpy(dtype=float)
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train_x = (train_x - center) / scale
    query_x = (query_x - center) / scale
    train_phase = frame.loc[train_blocks, "phase"].to_numpy(dtype=int)

    choices: list[dict[str, Any]] = []
    for query_position, block in enumerate(query_blocks):
        phase = int(frame.loc[block, "phase"])
        candidate_positions = np.flatnonzero(train_phase == phase)
        distances = np.sqrt(
            np.mean(
                (train_x[candidate_positions] - query_x[query_position]) ** 2,
                axis=1,
            )
        )
        k = min(policy.neighbors, len(candidate_positions))
        nearest_local = np.argpartition(distances, k - 1)[:k]
        nearest_positions = candidate_positions[nearest_local]
        nearest_blocks = train_blocks[nearest_positions]
        weights = 1.0 / np.maximum(distances[nearest_local], 1e-6)
        weights /= weights.sum()
        utility = train_utility.loc[nearest_blocks]
        utility_values = utility.to_numpy(dtype=float)
        expected = (utility_values * weights[:, None]).sum(axis=0)
        centered = utility_values - expected[None, :]
        dispersion = np.sqrt((centered**2 * weights[:, None]).sum(axis=0))
        robust = expected - policy.risk_penalty * dispersion
        positive_fraction = (utility_values > 0.0).mean(axis=0)

        source_months = pd.to_datetime(
            frame.loc[nearest_blocks, "representative_ns"].to_numpy(dtype=np.int64)
        ).to_period("M")
        month_floors: list[float] = []
        for expert_position in range(utility_values.shape[1]):
            grouped = pd.Series(
                utility_values[:, expert_position], index=source_months
            ).groupby(level=0)
            supported = [
                float(values.mean())
                for _, values in grouped
                if len(values) >= 2
            ]
            month_floors.append(min(supported) if supported else -np.inf)

        winner_position = int(np.argmax(robust))
        source_month_floor = float(month_floors[winner_position])
        eligible = bool(
            expected[winner_position] > 0.0
            and robust[winner_position] > 0.0
            and positive_fraction[winner_position]
            >= policy.min_positive_fraction
            and (
                not policy.require_positive_source_months
                or source_month_floor >= 0.0
            )
        )
        choices.append(
            {
                "block": block,
                "expert": str(utility.columns[winner_position]),
                "expected_utility": float(expected[winner_position]),
                "robust_utility": float(robust[winner_position]),
                "positive_fraction": float(positive_fraction[winner_position]),
                "source_month_floor": source_month_floor,
                "eligible": eligible,
            }
        )

    eligible_choices = [choice for choice in choices if choice["eligible"]]
    keep_count = int(np.floor(policy.coverage * len(query_blocks)))
    selected = {
        choice["block"]
        for choice in sorted(
            eligible_choices,
            key=lambda item: item["robust_utility"],
            reverse=True,
        )[:keep_count]
    }
    base = experts["incumbent_finesweep"]
    output = base.copy()
    selected_experts: dict[str, int] = {}
    selected_rows = 0
    for choice in choices:
        if choice["block"] not in selected:
            continue
        rows = rows_by_block[choice["block"]]
        expert = choice["expert"]
        output[rows] = np.clip(
            base[rows] + policy.alpha * (experts[expert][rows] - base[rows]),
            0.0,
            None,
        )
        selected_experts[expert] = selected_experts.get(expert, 0) + 1
        selected_rows += len(rows)
    return output, {
        "selected_blocks": len(selected),
        "available_blocks": len(query_blocks),
        "coverage": len(selected) / max(len(query_blocks), 1),
        "selected_rows": selected_rows,
        "selected_experts": selected_experts,
    }


def _month_blocks(frame: pd.DataFrame, month: pd.Period) -> np.ndarray:
    start = month.start_time
    end = (month + 1).start_time
    return period_blocks(frame, start, end)


def _component_floor(evaluation: dict[str, Any]) -> float:
    return min(
        value[component]
        for value in evaluation["months"].values()
        for component in ("score", "one_minus_nmae", "ficr")
    )


def _qualifies(evaluation: dict[str, Any]) -> bool:
    delta = evaluation["delta"]
    return bool(
        delta["score"] > 0.0
        and delta["one_minus_nmae"] >= 0.0
        and delta["ficr"] >= 0.0
        and _component_floor(evaluation) >= 0.0
    )


def main() -> None:
    index, truth, all_experts = load_experts()
    experts = {
        "incumbent_finesweep": all_experts["incumbent_finesweep"],
        "spatiotemporal_seed_mean": all_experts["spatiotemporal_seed_mean"],
    }
    issue = load_issue_times(ROOT / "data" / "train" / "gfs_train.csv", index)
    frame, rows_by_block = build_blocks(index, issue, experts)

    development_months = tuple(pd.period_range("2024-02", "2024-06", freq="M"))
    policy_records: list[dict[str, Any]] = []
    for policy in policies():
        combined = experts["incumbent_finesweep"].copy()
        routing_by_month: dict[str, Any] = {}
        query_union: list[str] = []
        for month in development_months:
            train_blocks = period_blocks(frame, None, month.start_time)
            query_blocks = _month_blocks(frame, month)
            train_utility = utility_table(
                truth, experts, rows_by_block, train_blocks
            )
            candidate, routing = route_robust(
                frame,
                rows_by_block,
                experts,
                train_blocks,
                query_blocks,
                train_utility,
                policy,
            )
            query_rows = np.concatenate(
                [rows_by_block[name] for name in query_blocks]
            )
            combined[query_rows] = candidate[query_rows]
            routing_by_month[str(month)] = routing
            query_union.extend(query_blocks.tolist())

        evaluation = evaluate_period(
            index,
            truth,
            experts["incumbent_finesweep"],
            combined,
            rows_by_block,
            np.asarray(query_union, dtype=str),
        )
        policy_records.append(
            {
                "policy": policy,
                "routing_by_month": routing_by_month,
                "evaluation": evaluation,
                "qualifies": _qualifies(evaluation),
            }
        )

    eligible = [record for record in policy_records if record["qualifies"]]
    selected = max(
        eligible,
        key=lambda record: (
            _component_floor(record["evaluation"]),
            record["evaluation"]["delta"]["score"],
        ),
        default=None,
    )
    locked = None
    promoted = False
    if selected is not None:
        h1 = period_blocks(frame, None, H2_START)
        h2 = period_blocks(frame, H2_START, LOCKED_END)
        h1_utility = utility_table(truth, experts, rows_by_block, h1)
        candidate, routing = route_robust(
            frame,
            rows_by_block,
            experts,
            h1,
            h2,
            h1_utility,
            selected["policy"],
        )
        evaluation = evaluate_period(
            index,
            truth,
            experts["incumbent_finesweep"],
            candidate,
            rows_by_block,
            h2,
        )
        promoted = _qualifies(evaluation)
        locked = {
            "routing": routing,
            "evaluation": evaluation,
            "promoted": promoted,
        }

    report = {
        "method": "deployable structural expert robust six-hour weather router",
        "contract": {
            "development": "monthly expanding-window February-June",
            "selection": "all aggregate components and every monthly component non-negative",
            "locked": "frozen policy fit on H1 utility and evaluated on H2 once",
            "deployable_expert": "retained two-seed spatiotemporal mean",
            "submission_created": False,
        },
        "development_months": [str(month) for month in development_months],
        "policy_count": len(policy_records),
        "qualified_policy_count": len(eligible),
        "selected_policy": None
        if selected is None
        else {"name": selected["policy"].name, **selected["policy"].__dict__},
        "selected_development": None
        if selected is None
        else {
            "routing_by_month": selected["routing_by_month"],
            "evaluation": selected["evaluation"],
        },
        "locked": locked,
        "decision": (
            "promotion gate passed; deployment construction must be audited separately"
            if promoted
            else "rejected; no submission created"
        ),
        "top_development": [
            {
                "policy": {
                    "name": record["policy"].name,
                    **record["policy"].__dict__,
                },
                "evaluation": record["evaluation"],
                "qualifies": record["qualifies"],
            }
            for record in sorted(
                policy_records,
                key=lambda record: record["evaluation"]["delta"]["score"],
                reverse=True,
            )[:10]
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(_to_builtin(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            _to_builtin(
                {
                    "qualified_policy_count": len(eligible),
                    "selected_policy": report["selected_policy"],
                    "locked": locked,
                    "decision": report["decision"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
