"""Frozen three-axis audit for the post-break R_ANCHOR overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.v23_multiyear_direct_screen import _joint_domain
from src.v23_postbreak_gam_screen import SPECS, _fit_predict
from src.v23_structural_residual_screen import (
    _derived,
    _load_axis,
    apply_v22_recipe,
)
from src.train_v25_postbreak_anchor import APPLY_DOMAIN, ETA, MODEL_NAME


def _early_to_late_2024(project: Path, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    year = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    derived = _derived(year, _joint_domain(year))
    fit = derived.loc[derived["game_month"].le(7)].reset_index(drop=True)
    audit = derived.loc[derived["game_month"].ge(8)].reset_index(drop=True)
    with np.load(
        project / "artifacts" / "champion_oof_20260817_01" / "y2024_early_to_late.npz",
        allow_pickle=True,
    ) as saved:
        target = saved["target"].astype(np.float64)
        v21 = saved["v21"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    if not np.array_equal(target, audit["control_success"].to_numpy(np.float64)):
        raise ValueError("2024 early-to-late target order mismatch")
    audit["target"] = target
    audit["v21"] = v21
    audit["domain3"] = domain
    audit["v22"] = apply_v22_recipe(
        v21,
        domain,
        pd.to_numeric(audit["asof_pitcher_success_rate"], errors="coerce").to_numpy(),
        pd.to_numeric(audit["asof_batter_success_rate"], errors="coerce").to_numpy(),
    )
    return fit, audit


def _evaluate(frame: pd.DataFrame, direct: np.ndarray) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)
    parent = frame["v22"].to_numpy(np.float64)
    apply_mask = frame["domain3"].astype(str).eq(APPLY_DOMAIN).to_numpy()
    candidate = parent.copy()
    candidate[apply_mask] = np.clip(
        parent[apply_mask] + ETA * (direct[apply_mask] - parent[apply_mask]),
        0.001,
        0.999,
    )

    def gain(mask: np.ndarray) -> float:
        local_target = target[mask]
        reference = float(np.mean(local_target) * (1.0 - np.mean(local_target)))
        paired_mse_gain = float(
            np.mean((local_target - parent[mask]) ** 2)
            - np.mean((local_target - candidate[mask]) ** 2)
        )
        if reference <= 0.0:
            return 1_000_000.0 * paired_mse_gain
        return 100_000.0 * paired_mse_gain / reference

    all_mask = np.ones(len(frame), dtype=bool)
    months = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        applied_rows = int(np.sum(mask & apply_mask))
        months.append(
            {
                "month": int(month),
                "rows": int(mask.sum()),
                "applied_rows": applied_rows,
                "gain": gain(mask),
            }
        )
    active_months = [row for row in months if row["applied_rows"] > 0]
    groups = []
    for column in ("pitcher_hand", "batter_hand", "count_state"):
        for value in sorted(frame[column].astype(str).unique()):
            mask = apply_mask & frame[column].astype(str).eq(value).to_numpy()
            if mask.any():
                groups.append(
                    {
                        "column": column,
                        "value": value,
                        "rows": int(mask.sum()),
                        "gain": gain(mask),
                    }
                )
    return {
        "gain": gain(all_mask),
        "applied_rows": int(apply_mask.sum()),
        "applied_domain_gain": gain(apply_mask),
        "positive_active_month_fraction": float(
            np.mean([row["gain"] > 0.0 for row in active_months])
        ),
        "worst_active_month_gain": float(min(row["gain"] for row in active_months)),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
        "months": months,
        "groups": groups,
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    model_spec = next(spec for spec in SPECS if spec.name == MODEL_NAME)

    selection_audit = _load_axis(project, "y2023_early_to_late", raw)
    year23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    year23 = _derived(year23, _joint_domain(year23))
    selection_fit = year23.loc[year23["game_month"].le(7)].reset_index(drop=True)
    selection_direct = _fit_predict(selection_fit, selection_audit, model_spec)
    selection = _evaluate(selection_audit, selection_direct)

    outer_audit = _load_axis(project, "y2023_to_y2024", raw)
    outer_direct = _fit_predict(year23, outer_audit, model_spec)
    outer = _evaluate(outer_audit, outer_direct)

    replication_fit, replication_audit = _early_to_late_2024(project, raw)
    replication_direct = _fit_predict(replication_fit, replication_audit, model_spec)
    replication = _evaluate(replication_audit, replication_direct)

    gates = {
        "selection_gain_positive": selection["gain"] > 0.0,
        "selection_all_active_months_positive": selection[
            "positive_active_month_fraction"
        ]
        == 1.0,
        "selection_applied_domain_positive": selection["applied_domain_gain"] > 0.0,
        "outer_gain_positive": outer["gain"] > 0.0,
        "outer_active_month_fraction_at_least_085": outer[
            "positive_active_month_fraction"
        ]
        >= 0.85,
        "outer_worst_month_above_minus_10": outer["worst_active_month_gain"] > -10.0,
        "outer_applied_domain_positive": outer["applied_domain_gain"] > 0.0,
        "replication_gain_positive": replication["gain"] > 0.0,
        "replication_all_active_months_positive": replication[
            "positive_active_month_fraction"
        ]
        == 1.0,
        "replication_applied_domain_positive": replication["applied_domain_gain"] > 0.0,
    }
    summary = {
        "protocol": "V25_POSTBREAK_ANCHOR_THREE_AXIS_V1",
        "frozen_candidate": {
            "model": MODEL_NAME,
            "eta": ETA,
            "apply_domain": APPLY_DOMAIN,
        },
        "selection_early2023_to_late2023": selection,
        "outer_full2023_to_full2024": outer,
        "replication_early2024_to_late2024": replication,
        "gates": {key: bool(value) for key, value in gates.items()},
        "eligible_for_leaderboard_probe": bool(all(gates.values())),
        "promotion_rule": "public score must exceed 1153.0436023798 before champion promotion",
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/v25_postbreak_anchor_audit_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
