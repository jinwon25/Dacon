"""Forward weather-similarity router for existing group-3 OOF experts.

Inspired by the KDD Cup 2022 dynamic ensemble and the GEFCom weather-similarity
approach.  Q1 selects one conservative kNN routing policy on Q2.  H1 then fits
the same frozen policy and H2 is opened once.  The script is diagnostic only
and never creates a submission.
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
from experiments.dynamic_router_oracle import load_experts, row_contribution
from src.metrics import CAPACITY_KWH, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
CAPACITY = CAPACITY_KWH["kpx_group_3"]
OUTPUT = ROOT / "artifacts_final" / "diagnostics" / "weather_similarity_router_20260718.json"
Q2_START = pd.Timestamp("2024-04-01")
H2_START = pd.Timestamp("2024-07-01")

WEATHER_COLUMNS = (
    "ldaps__kpx_group_3__hub_ws117__idw",
    "ldaps__kpx_group_3__hub_u117__idw",
    "ldaps__kpx_group_3__hub_v117__idw",
    "ldaps__kpx_group_3__surface_0_sp__idw",
    "gfs__kpx_group_3__hub_ws117__idw",
    "gfs__kpx_group_3__hub_u117__idw",
    "gfs__kpx_group_3__hub_v117__idw",
    "gfs__kpx_group_3__surface_0_gust__idw",
    "gfs__kpx_group_3__surface_0_sp__idw",
)


@dataclass(frozen=True)
class Policy:
    neighbors: int
    min_positive_fraction: float
    alpha: float
    coverage: float

    @property
    def name(self) -> str:
        return (
            f"k{self.neighbors}_p{int(self.min_positive_fraction * 100)}_"
            f"a{int(self.alpha * 100)}_c{int(self.coverage * 100)}"
        )


def policies() -> tuple[Policy, ...]:
    return tuple(
        Policy(k, positive, alpha, coverage)
        for k, positive, alpha, coverage in itertools.product(
            (12, 24, 36),
            (0.55, 0.65),
            (0.25, 0.50),
            (0.10, 0.20, 0.25),
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


def build_blocks(
    index: pd.DatetimeIndex,
    issue: pd.DatetimeIndex,
    experts: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    feature_cache = pd.read_pickle(
        ROOT / "artifacts_final" / "feature_cache" / "features_train.pkl"
    ).reindex(index)
    missing = [column for column in WEATHER_COLUMNS if column not in feature_cache]
    if missing:
        raise KeyError(f"Missing router weather columns: {missing}")
    if feature_cache[list(WEATHER_COLUMNS)].isna().any().any():
        raise ValueError("Router weather features contain missing values")

    lead = ((index - issue) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    phase = ((lead - 12) // 6).astype(int)
    if not np.isin(phase, np.arange(4)).all():
        raise ValueError("Unexpected six-hour lead phase")
    issue_ns = issue.astype("int64").to_numpy()
    keys = np.asarray(
        [f"{issue_value}:{phase_value}" for issue_value, phase_value in zip(issue_ns, phase)],
        dtype=object,
    )

    rows_by_block: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for key in pd.unique(keys):
        rows = np.flatnonzero(keys == key)
        rows_by_block[str(key)] = rows
        record: dict[str, Any] = {
            "block": str(key),
            "representative_ns": int(index[rows].asi8[len(rows) // 2]),
            "phase": int(phase[rows[0]]),
            "rows": int(len(rows)),
        }
        for column in WEATHER_COLUMNS:
            values = feature_cache.iloc[rows][column].to_numpy(dtype=float)
            short = column.replace("__kpx_group_3__", "__g3__")
            record[f"{short}__mean"] = float(values.mean())
            record[f"{short}__std"] = float(values.std())
            record[f"{short}__slope"] = float(values[-1] - values[0])
        base = experts["incumbent_finesweep"][rows]
        record["base__mean"] = float(base.mean() / CAPACITY)
        record["base__std"] = float(base.std() / CAPACITY)
        record["base__slope"] = float((base[-1] - base[0]) / CAPACITY)
        for name, prediction in experts.items():
            if name == "incumbent_finesweep":
                continue
            difference = (prediction[rows] - base) / CAPACITY
            record[f"{name}__diff_mean"] = float(difference.mean())
            record[f"{name}__diff_std"] = float(difference.std())
            record[f"{name}__diff_slope"] = float(difference[-1] - difference[0])
        records.append(record)
    frame = pd.DataFrame.from_records(records).set_index("block").sort_values(
        "representative_ns"
    )
    return frame, rows_by_block


def period_blocks(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> np.ndarray:
    representative = pd.to_datetime(frame["representative_ns"])
    mask = representative < end
    if start is not None:
        mask &= representative >= start
    return frame.index[mask].to_numpy(dtype=str)


def utility_table(
    truth: np.ndarray,
    experts: dict[str, np.ndarray],
    rows_by_block: dict[str, np.ndarray],
    block_names: np.ndarray,
) -> pd.DataFrame:
    period_rows = np.zeros(len(truth), dtype=bool)
    for name in block_names:
        period_rows[rows_by_block[name]] = True
    base_contribution = row_contribution(
        truth, experts["incumbent_finesweep"], period_rows
    )
    values: dict[str, list[float]] = {}
    for expert_name, prediction in experts.items():
        if expert_name == "incumbent_finesweep":
            continue
        delta = row_contribution(truth, prediction, period_rows) - base_contribution
        values[expert_name] = [float(delta[rows_by_block[name]].sum()) for name in block_names]
    return pd.DataFrame(values, index=block_names)


def route(
    frame: pd.DataFrame,
    rows_by_block: dict[str, np.ndarray],
    experts: dict[str, np.ndarray],
    train_blocks: np.ndarray,
    query_blocks: np.ndarray,
    train_utility: pd.DataFrame,
    policy: Policy,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = {"representative_ns", "phase", "rows"}
    columns = [column for column in frame.columns if column not in metadata]
    train_x = frame.loc[train_blocks, columns].to_numpy(dtype=float)
    query_x = frame.loc[query_blocks, columns].to_numpy(dtype=float)
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train_x = (train_x - center) / scale
    query_x = (query_x - center) / scale

    choices: list[dict[str, Any]] = []
    for query_position, block in enumerate(query_blocks):
        phase = int(frame.loc[block, "phase"])
        same_phase = frame.loc[train_blocks, "phase"].to_numpy(dtype=int) == phase
        candidate_positions = np.flatnonzero(same_phase)
        distances = np.sqrt(
            np.mean((train_x[candidate_positions] - query_x[query_position]) ** 2, axis=1)
        )
        k = min(policy.neighbors, len(candidate_positions))
        nearest_local = np.argpartition(distances, k - 1)[:k]
        nearest_positions = candidate_positions[nearest_local]
        nearest_blocks = train_blocks[nearest_positions]
        weights = 1.0 / np.maximum(distances[nearest_local], 1e-6)
        weights /= weights.sum()
        neighbor_utility = train_utility.loc[nearest_blocks]
        expected = (neighbor_utility.to_numpy(dtype=float) * weights[:, None]).sum(axis=0)
        positive_fraction = (neighbor_utility.to_numpy(dtype=float) > 0.0).mean(axis=0)
        winner_position = int(np.argmax(expected))
        choices.append(
            {
                "block": block,
                "expert": str(neighbor_utility.columns[winner_position]),
                "expected_utility": float(expected[winner_position]),
                "positive_fraction": float(positive_fraction[winner_position]),
                "eligible": bool(
                    expected[winner_position] > 0.0
                    and positive_fraction[winner_position] >= policy.min_positive_fraction
                ),
            }
        )

    eligible = [choice for choice in choices if choice["eligible"]]
    keep_count = int(np.floor(policy.coverage * len(query_blocks)))
    selected = {
        choice["block"]
        for choice in sorted(eligible, key=lambda item: item["expected_utility"], reverse=True)[
            :keep_count
        ]
    }
    output = experts["incumbent_finesweep"].copy()
    selected_experts: dict[str, int] = {}
    selected_rows = 0
    for choice in choices:
        if choice["block"] not in selected:
            continue
        rows = rows_by_block[choice["block"]]
        expert = choice["expert"]
        output[rows] = np.clip(
            output[rows] + policy.alpha * (experts[expert][rows] - output[rows]),
            0.0,
            CAPACITY,
        )
        selected_experts[expert] = selected_experts.get(expert, 0) + 1
        selected_rows += len(rows)
    return output, {
        "selected_blocks": int(len(selected)),
        "available_blocks": int(len(query_blocks)),
        "coverage": float(len(selected) / max(len(query_blocks), 1)),
        "selected_rows": int(selected_rows),
        "selected_experts": selected_experts,
    }


def evaluate_period(
    index: pd.DatetimeIndex,
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    rows_by_block: dict[str, np.ndarray],
    block_names: np.ndarray,
) -> dict[str, Any]:
    rows = np.zeros(len(index), dtype=bool)
    for name in block_names:
        rows[rows_by_block[name]] = True
    before = evaluate_group(truth[rows], base[rows], CAPACITY)
    after = evaluate_group(truth[rows], candidate[rows], CAPACITY)
    months: dict[str, Any] = {}
    for month in sorted(pd.unique(index[rows].to_period("M").astype(str))):
        month_rows = rows & (index.to_period("M").astype(str) == month)
        if not (truth[month_rows] >= 0.10 * CAPACITY).any():
            continue
        month_before = evaluate_group(truth[month_rows], base[month_rows], CAPACITY)
        month_after = evaluate_group(truth[month_rows], candidate[month_rows], CAPACITY)
        months[str(month)] = {
            "score": month_after.score - month_before.score,
            "one_minus_nmae": month_after.one_minus_nmae - month_before.one_minus_nmae,
            "ficr": month_after.ficr - month_before.ficr,
        }
    return {
        "base": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": {
            "score": after.score - before.score,
            "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
            "ficr": after.ficr - before.ficr,
        },
        "months": months,
        "worst_month_score_delta": min(value["score"] for value in months.values()),
    }


def main() -> None:
    index, truth, all_experts = load_experts()
    retained = (
        "incumbent_finesweep",
        "lineage_exact_base",
        "lineage_blend_v1",
        "lineage_weighted_member",
        "spatiotemporal_seed17",
        "spatiotemporal_seed29",
        "spatiotemporal_seed_mean",
    )
    experts = {name: all_experts[name] for name in retained}
    issue = load_issue_times(ROOT / "data" / "train" / "gfs_train.csv", index)
    frame, rows_by_block = build_blocks(index, issue, experts)
    q1 = period_blocks(frame, None, Q2_START)
    q2 = period_blocks(frame, Q2_START, H2_START)
    h1 = period_blocks(frame, None, H2_START)
    h2 = period_blocks(frame, H2_START, pd.Timestamp("2025-01-02"))

    q1_utility = utility_table(truth, experts, rows_by_block, q1)
    development: list[dict[str, Any]] = []
    for policy in policies():
        candidate, routing = route(
            frame, rows_by_block, experts, q1, q2, q1_utility, policy
        )
        evaluation = evaluate_period(
            index,
            truth,
            experts["incumbent_finesweep"],
            candidate,
            rows_by_block,
            q2,
        )
        qualifies = bool(
            evaluation["delta"]["score"] > 0.0
            and evaluation["delta"]["one_minus_nmae"] >= 0.0
            and evaluation["delta"]["ficr"] >= 0.0
            and evaluation["worst_month_score_delta"] >= 0.0
        )
        development.append(
            {
                "policy": policy,
                "routing": routing,
                "evaluation": evaluation,
                "qualifies": qualifies,
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
    if selected is not None:
        h1_utility = utility_table(truth, experts, rows_by_block, h1)
        candidate, routing = route(
            frame,
            rows_by_block,
            experts,
            h1,
            h2,
            h1_utility,
            selected["policy"],
        )
        locked = {
            "routing": routing,
            "evaluation": evaluate_period(
                index,
                truth,
                experts["incumbent_finesweep"],
                candidate,
                rows_by_block,
                h2,
            ),
        }

    report = {
        "method": "phase-restricted weather-similarity dynamic expert router",
        "sources": [
            "https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_0518.pdf",
            "https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_1286.pdf",
            "https://www.sciencedirect.com/science/article/pii/S0169207013000836",
        ],
        "contract": {
            "development": "Q1 utility and weather similarity route Q2",
            "locked": "same selected policy, H1 utility and weather similarity route H2 once",
            "block": "complete six-hour lead phase within one NWP issue",
            "fallback": "incumbent finesweep",
            "submission_created": False,
        },
        "features": list(WEATHER_COLUMNS),
        "experts": list(experts),
        "block_counts": {"q1": len(q1), "q2": len(q2), "h1": len(h1), "h2": len(h2)},
        "development_candidates": [
            {
                "policy": item["policy"].__dict__,
                "name": item["policy"].name,
                "routing": item["routing"],
                "evaluation": item["evaluation"],
                "qualifies": item["qualifies"],
            }
            for item in development
        ],
        "selected_policy": None
        if selected is None
        else {"name": selected["policy"].name, **selected["policy"].__dict__},
        "locked": locked,
        "decision": (
            "reject before locked H2 because no Q2 policy passed every component/month gate"
            if selected is None
            else "locked H2 evaluated once; promotion still requires positive score, both components, and every month"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(_to_builtin(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "selected_policy": report["selected_policy"],
        "locked": locked,
        "qualified_q2_policies": len(eligible),
        "best_q2": max(
            (
                {
                    "name": item["policy"].name,
                    "delta": item["evaluation"]["delta"],
                    "worst_month": item["evaluation"]["worst_month_score_delta"],
                }
                for item in development
            ),
            key=lambda item: item["delta"]["score"],
        ),
    }
    print(json.dumps(_to_builtin(summary), ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
