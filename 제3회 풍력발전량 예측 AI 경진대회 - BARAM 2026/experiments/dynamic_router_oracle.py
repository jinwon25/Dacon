"""Upper-bound audit for issue/lead-block routing among existing group-3 experts.

This is a diagnostic, not a selectable model.  It uses locked H2 truth only to
measure whether a KDD-Cup-style dynamic ensemble has enough theoretical
headroom to justify a new forward-validated router.  It never writes a
submission and must not be used to choose test rows or routing thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.spatiotemporal_consensus_promotion import _rolling_finesweep_base
from src.metrics import CAPACITY_KWH, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
CAPACITY = CAPACITY_KWH["kpx_group_3"]
H2_START = pd.Timestamp("2024-07-01")
OUTPUT = ROOT / "artifacts_final" / "diagnostics" / "dynamic_router_oracle_20260718.json"


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


def row_contribution(
    truth: np.ndarray,
    prediction: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Return additive contributions whose sum plus 0.5 is the official score."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    rows = np.asarray(rows, dtype=bool)
    eligible = rows & (truth >= 0.10 * CAPACITY)
    if not eligible.any():
        raise ValueError("No eligible rows in the requested period")
    count = int(eligible.sum())
    generation_sum = float(truth[eligible].sum())
    error_rate = np.abs(truth - prediction) / CAPACITY
    price = np.where(error_rate <= 0.06, 4.0, np.where(error_rate <= 0.08, 3.0, 0.0))
    contribution = np.zeros(len(truth), dtype=float)
    contribution[eligible] = (
        -0.5 * error_rate[eligible] / count
        + 0.5 * truth[eligible] * price[eligible] / (4.0 * generation_sum)
    )
    return contribution


def _align_neural(
    path: Path,
    index: pd.DatetimeIndex,
    fallback: np.ndarray,
    group_position: int = 2,
) -> np.ndarray:
    cache = np.load(path, allow_pickle=False)
    source_index = pd.DatetimeIndex(pd.to_datetime(cache["timestamps_ns"]))
    positions = source_index.get_indexer(index)
    output = np.asarray(fallback, dtype=float).copy()
    present = positions >= 0
    normalized = cache["prediction"].astype(float)[positions[present], group_position]
    output[present] = np.clip(normalized * CAPACITY, 0.0, CAPACITY)
    return output


def load_experts() -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, np.ndarray]]:
    _, index, truth, base = _rolling_finesweep_base(
        ROOT / "data" / "train" / "train_labels.csv",
        ROOT / "artifacts_final" / "lineage" / "exact_driver_oof.npz",
    )
    experts: dict[str, np.ndarray] = {"incumbent_finesweep": base.astype(float)}

    lineage = np.load(ROOT / "artifacts_final" / "lineage" / "exact_driver_oof.npz")
    lineage_index = pd.DatetimeIndex(
        pd.to_datetime(lineage["kpx_group_3__valid_index_ns"])
    )
    if not lineage_index.equals(index):
        raise ValueError("Lineage and rolling-finesweep indexes differ")
    for key in ("exact_base", "stack5", "blend_v1", "weighted_member"):
        experts[f"lineage_{key}"] = lineage[f"kpx_group_3__{key}"].astype(float)

    trajectory = np.load(
        ROOT / "artifacts_final" / "structural_20260718" / "locked_predictions.npz"
    )
    trajectory_index = pd.DatetimeIndex(pd.to_datetime(trajectory["index_ns"]))
    if not trajectory_index.equals(index):
        raise ValueError("Trajectory and rolling-finesweep indexes differ")
    experts["trajectory_residual"] = trajectory["locked_candidate"].astype(float)

    seed17 = _align_neural(
        ROOT / "artifacts_final" / "spatiotemporal" / "validation_predictions.npz",
        index,
        base,
    )
    seed29 = _align_neural(
        ROOT
        / "artifacts_final"
        / "spatiotemporal_seed29"
        / "validation_predictions.npz",
        index,
        base,
    )
    experts["spatiotemporal_seed17"] = seed17
    experts["spatiotemporal_seed29"] = seed29
    experts["spatiotemporal_seed_mean"] = 0.5 * (seed17 + seed29)

    for name, prediction in experts.items():
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid expert surface: {name}")
    return index, truth.astype(float), experts


def oracle_route(
    truth: np.ndarray,
    experts: dict[str, np.ndarray],
    rows: np.ndarray,
    block_id: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    names = list(experts)
    contributions = np.vstack(
        [row_contribution(truth, experts[name], rows) for name in names]
    )
    output = experts["incumbent_finesweep"].copy()
    selected_blocks: dict[str, int] = {name: 0 for name in names}
    selected_rows: dict[str, int] = {name: 0 for name in names}
    unique_blocks = pd.unique(block_id[rows])
    for block in unique_blocks:
        block_rows = rows & (block_id == block)
        block_utility = contributions[:, block_rows].sum(axis=1)
        winner = int(np.argmax(block_utility))
        name = names[winner]
        output[block_rows] = experts[name][block_rows]
        selected_blocks[name] += 1
        selected_rows[name] += int(block_rows.sum())
    metric = evaluate_group(truth[rows], output[rows], CAPACITY)
    base_metric = evaluate_group(
        truth[rows], experts["incumbent_finesweep"][rows], CAPACITY
    )
    return output, {
        "blocks": int(len(unique_blocks)),
        "selected_blocks": selected_blocks,
        "selected_rows": selected_rows,
        "metric": metric.to_dict(),
        "delta": {
            "score": metric.score - base_metric.score,
            "one_minus_nmae": metric.one_minus_nmae - base_metric.one_minus_nmae,
            "ficr": metric.ficr - base_metric.ficr,
        },
    }


def main() -> None:
    index, truth, experts = load_experts()
    issue = load_issue_times(ROOT / "data" / "train" / "gfs_train.csv", index)
    h2 = index >= H2_START
    lead = ((index - issue) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    if not np.isin(lead, np.arange(12, 36)).all():
        raise ValueError("Unexpected lead-hour range")

    issue_id = issue.astype("int64").to_numpy()
    phase_id = np.asarray(
        [f"{item}:{int((hour - 12) // 6)}" for item, hour in zip(issue_id, lead)],
        dtype=object,
    )
    row_id = np.arange(len(index), dtype=np.int64)

    expert_metrics: dict[str, Any] = {}
    base_metric = evaluate_group(
        truth[h2], experts["incumbent_finesweep"][h2], CAPACITY
    )
    for name, prediction in experts.items():
        metric = evaluate_group(truth[h2], prediction[h2], CAPACITY)
        expert_metrics[name] = {
            "metric": metric.to_dict(),
            "delta": {
                "score": metric.score - base_metric.score,
                "one_minus_nmae": metric.one_minus_nmae - base_metric.one_minus_nmae,
                "ficr": metric.ficr - base_metric.ficr,
            },
        }

    _, issue_oracle = oracle_route(truth, experts, h2, issue_id)
    _, phase_oracle = oracle_route(truth, experts, h2, phase_id)
    _, row_oracle = oracle_route(truth, experts, h2, row_id)

    report = {
        "purpose": "truth-only upper-bound audit; never a selectable router",
        "period": {
            "start": str(index[h2].min()),
            "end": str(index[h2].max()),
            "rows": int(h2.sum()),
            "issues": int(pd.unique(issue_id[h2]).size),
            "lead_hours": [int(lead.min()), int(lead.max())],
        },
        "incumbent": base_metric.to_dict(),
        "experts": expert_metrics,
        "oracle": {
            "issue_24h": issue_oracle,
            "lead_phase_6h": phase_oracle,
            "row": row_oracle,
        },
        "decision_rule": (
            "Proceed to a separately forward-validated router only if the 24h or 6h "
            "oracle has material headroom; this report cannot select features, thresholds, "
            "experts, test rows, or a submission."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(_to_builtin(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_to_builtin(report["oracle"]), ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
