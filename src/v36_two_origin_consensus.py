"""Two-origin exact-recipe consensus screen above v27.

Unlike v35, this protocol fixes the same signal, domain, toward-parent
direction, and blend weight on both 2022 and late-2023.  Only recipes that are
positive on both origins are eligible.  Full-2024 remains outside selection;
late-2024 is a secondary stability check of the single frozen recipe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _load_bank, _metadata
from src.v35_three_stage_multibank import _candidate


KEYS = ("signal", "domain", "weight")


def select_consensus(
    stage1: pd.DataFrame, stage2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Select without consulting any 2024 metric."""
    left = stage1.loc[stage1["direction"].eq("toward_parent")].copy()
    right = stage2.loc[stage2["direction"].eq("toward_parent")].copy()
    merged = left.merge(right, on=list(KEYS), suffixes=("_2022", "_2023"))
    if merged.empty:
        raise ValueError("no exact recipe is shared by the two selection origins")
    merged["minimum_gain"] = merged[["gain_2022", "gain_2023"]].min(axis=1)
    merged["minimum_worst_month"] = merged[
        ["worst_month_gain_2022", "worst_month_gain_2023"]
    ].min(axis=1)
    merged["minimum_applied_domain"] = merged[
        ["applied_domain_gain_2022", "applied_domain_gain_2023"]
    ].min(axis=1)
    merged["consensus_score"] = merged[
        ["minimum_gain", "minimum_worst_month", "minimum_applied_domain"]
    ].min(axis=1)
    merged["passes_consensus_gate"] = (
        merged["gain_2022"].gt(0.0)
        & merged["gain_2023"].gt(0.0)
        & merged["positive_month_fraction_2022"].ge(0.75)
        & merged["positive_month_fraction_2023"].eq(1.0)
        & merged["minimum_worst_month"].gt(0.0)
        & merged["minimum_applied_domain"].gt(0.0)
    )
    merged = merged.sort_values(
        ["passes_consensus_gate", "consensus_score", "minimum_gain"],
        ascending=False,
    ).reset_index(drop=True)
    passing = merged.loc[merged["passes_consensus_gate"]]
    chosen = passing.iloc[0] if len(passing) else merged.iloc[0]
    return merged, chosen


def run(project: Path, selection_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    selection_dir = (project / selection_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1 = pd.read_csv(selection_dir / "stage1_2022_metrics.csv")
    stage2 = pd.read_csv(selection_dir / "stage2_2023_metrics.csv")
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "direction": "toward_parent",
        "domain": str(chosen["domain"]),
        "weight": float(chosen["weight"]),
    }

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    del raw
    bank24 = _load_bank(project, 2024, {recipe["signal"]})
    late_mask = _metadata(project, 2024)["month"] >= 8
    results: dict[str, dict[str, object]] = {}
    for axis_name in ("outer_full_2024", "replication_late_2024"):
        frame = axes[axis_name]
        local_bank = (
            bank24
            if axis_name == "outer_full_2024"
            else {name: value[late_mask] for name, value in bank24.items()}
        )
        if any(len(value) != len(frame) for value in local_bank.values()):
            raise ValueError(f"bank row mismatch: {axis_name}")
        candidate, active = _candidate(frame, local_bank, [recipe])
        result = diagnostics(frame, v27_parent(frame), candidate, active)
        results[axis_name] = result
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=v27_parent(frame),
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )
    gates = {
        "consensus_gate": bool(chosen["passes_consensus_gate"]),
        "outer_gain_at_least_5": results["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": results["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": results["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "outer_minimum_domain_nonnegative": results["outer_full_2024"][
            "minimum_domain_gain"
        ]
        >= 0.0,
        "replication_gain_positive": results["replication_late_2024"]["gain"] > 0.0,
        "replication_month_fraction_at_least_two_thirds": results[
            "replication_late_2024"
        ]["positive_month_fraction"]
        >= 2.0 / 3.0,
        "replication_worst_month_above_minus_10": results[
            "replication_late_2024"
        ]["worst_month_gain"]
        > -10.0,
    }
    summary = {
        "protocol": "V36_TWO_ORIGIN_EXACT_RECIPE_CONSENSUS_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "selection": "same recipe positive on 2022 and late-2023; no 2024 labels",
        "consensus_candidate_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "chosen": {
            **recipe,
            "gain_2022": float(chosen["gain_2022"]),
            "gain_late_2023": float(chosen["gain_2023"]),
            "worst_month_2022": float(chosen["worst_month_gain_2022"]),
            "worst_month_late_2023": float(chosen["worst_month_gain_2023"]),
            "consensus_score": float(chosen["consensus_score"]),
        },
        "audits": results,
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("artifacts/v35_three_stage_multibank_20260817_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v36_two_origin_consensus_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.selection_dir, args.output_dir)


if __name__ == "__main__":
    main()
