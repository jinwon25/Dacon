"""Test the frozen v68 TrackMan ridge families outside R_ANCHOR.

This is a breadth diagnosis, not a new hyperparameter search.  The exact same
source-only ridge procedure and feature families are repeated in ALL, R_CORE,
and F to determine whether profile transfer failed because R_ANCHOR has a
narrow pitcher sample or because historical TrackMan profiles do not transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.axes import _cached_v25_axes
from src.v68_trackman_profile_ridge import FEATURE_SETS, _transition


DOMAINS = ("ALL", "R_CORE", "F")


def run(
    project: Path,
    profiles_path: Path,
    current_oof_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    profiles_path = profiles_path.resolve()
    current_oof_dir = current_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    profiles = pd.read_csv(profiles_path, low_memory=False)
    profile23 = profiles.loc[profiles["origin"].eq(2023)].drop(columns="origin")
    profile24 = profiles.loc[profiles["origin"].eq(2024)].drop(columns="origin")
    axes = _cached_v25_axes(project, raw)
    late23 = axes["selection_late_2023"]
    full24 = axes["outer_full_2024"]
    replication24 = axes["replication_late_2024"]
    cache23 = np.load(current_oof_dir / "selection_late_2023.npz")
    cache24 = np.load(current_oof_dir / "outer_full_2024.npz")
    cache_rep = np.load(current_oof_dir / "replication_late_2024.npz")
    parent23 = cache23["final_gate_parent"].astype(np.float64)
    parent24 = cache24["final_gate_parent"].astype(np.float64)
    parent_rep = cache_rep["final_gate_parent"].astype(np.float64)
    early_mask = full24["game_month"].le(7).to_numpy()
    early24 = full24.loc[early_mask].reset_index(drop=True)
    parent_early24 = parent24[early_mask]

    rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        for feature_name in FEATURE_SETS:
            full_result, _ = _transition(
                late23,
                parent23,
                profile23,
                full24,
                parent24,
                profile24,
                feature_name,
                domain=domain,
            )
            rows.append(
                {
                    "domain": domain,
                    "feature_set": feature_name,
                    "transition": "late23_to_full24",
                    **full_result,
                }
            )
            replication_result, _ = _transition(
                early24,
                parent_early24,
                profile24,
                replication24,
                parent_rep,
                profile24,
                feature_name,
                domain=domain,
            )
            rows.append(
                {
                    "domain": domain,
                    "feature_set": feature_name,
                    "transition": "early24_to_late24",
                    **replication_result,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust = (
        metrics.groupby(["domain", "feature_set"], observed=True)
        .agg(
            minimum_gain=("gain", "min"),
            mean_gain=("gain", "mean"),
            minimum_month_fraction=("positive_month_fraction", "min"),
            worst_month_gain=("worst_month_gain", "min"),
            minimum_source_pitchers=("source_pitchers", "min"),
            maximum_abs_shift=("mean_abs_shift", "max"),
        )
        .reset_index()
        .sort_values(["minimum_gain", "mean_gain"], ascending=False)
    )
    robust.to_csv(output_dir / "robust_summary.csv", index=False)
    eligible = robust.loc[
        robust["minimum_gain"].gt(0)
        & robust["minimum_month_fraction"].ge(0.5)
        & robust["worst_month_gain"].gt(-0.5)
    ]
    result = {
        "protocol": "V69_FROZEN_TRACKMAN_PROFILE_DOMAIN_BREADTH_V1",
        "parent": "exact public-1158 temporal OOF",
        "domains": list(DOMAINS),
        "feature_families_unchanged_from_v68": True,
        "summary": robust.to_dict(orient="records"),
        "eligible_recipes": eligible[["domain", "feature_set"]].to_dict(
            orient="records"
        ),
        "eligible_for_packaging": bool(len(eligible)),
        "decision": "research_only_due_reused_2024_audits",
        "row_local_inference": True,
        "other_test_rows_used": False,
        "test_distribution_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.profiles, args.current_oof_dir, args.output_dir)


if __name__ == "__main__":
    main()
