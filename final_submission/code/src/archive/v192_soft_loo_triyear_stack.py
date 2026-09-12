"""Finalize the unique v191 budget passing a predeclared soft LOO gate.

Every held-origin gain must be positive, every held month fraction at least
0.25, every worst month above -0.75, and every domain gain non-negative.  The
final tri-year optimization is performed only after this budget is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v191_leave_one_year_minimax_stack import (
    AXES,
    quadratic_gain_form,
    solve_minimax,
    stack_prediction,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V192_SOFT_LOO_TRIYEAR_STACK_V1"


def soft_budget_table(v191_summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    detail = v191_summary["leave_one_year_detail"]
    for budget_row in v191_summary["budget_ranking"]:
        budget = float(budget_row["budget"])
        metrics_by_axis = [
            detail[f"b{budget:g}__held_{axis}"]["metrics"] for axis in AXES
        ]
        passed = all(
            item["gain"] > 0.0
            and item["positive_month_fraction"] >= 0.25
            and item["worst_month_gain"] > -0.75
            and item["minimum_domain_gain"] >= 0.0
            for item in metrics_by_axis
        )
        rows.append(
            {
                "budget": budget,
                "soft_loo_passed": passed,
                "minimum_held_gain": min(item["gain"] for item in metrics_by_axis),
                "mean_held_gain": float(np.mean([item["gain"] for item in metrics_by_axis])),
                "minimum_month_fraction": min(
                    item["positive_month_fraction"] for item in metrics_by_axis
                ),
                "worst_month_gain": min(item["worst_month_gain"] for item in metrics_by_axis),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["soft_loo_passed", "minimum_held_gain", "mean_held_gain"],
        ascending=[False, False, False], kind="stable",
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v158_path: Path,
    v165_summary: Path,
    library_root: Path,
    v191_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    v191 = json.loads(v191_summary_path.read_text(encoding="utf-8"))
    if v191.get("protocol") != "V191_LEAVE_ONE_YEAR_MINIMAX_STACK_V1":
        raise ValueError("expected v191 LOO evidence")
    budget_ranking = soft_budget_table(v191)
    passing = budget_ranking.loc[budget_ranking["soft_loo_passed"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_ranking.to_csv(output_dir / "soft_budget_ranking.csv", index=False, encoding="utf-8-sig")
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "soft_loo_reject",
            "budget_ranking": budget_ranking.to_dict(orient="records"),
            "restrictions": restrictions(),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    selected_budget = float(passing.iloc[0]["budget"])

    _context, raw_frames, post4 = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, raw_frames, post4, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    frozen = json.loads(v165_summary.read_text(encoding="utf-8"))
    families = [str(name) for name in frozen["families"]]
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {axis: saved[axis].astype(np.float64) for axis in AXES}
    signed_directions: dict[str, np.ndarray] = {}
    for axis_name in AXES:
        columns = []
        for family in families:
            with np.load(library_root / family / "selected_axes.npz", allow_pickle=False) as saved:
                columns.append(saved[axis_name].astype(np.float64) - v158_base[axis_name])
        raw = np.column_stack(columns)
        signed_directions[axis_name] = np.column_stack([raw, -raw])

    forms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    month_names: list[str] = []
    for axis_name in AXES:
        axis = axes[axis_name]
        exact = np.asarray(axis["exact_mask"], dtype=bool)
        forms[axis_name] = quadratic_gain_form(
            axis["target"], parents[axis_name], signed_directions[axis_name], exact
        )
        for month in sorted(np.unique(axis["game_month"]).tolist()):
            key = f"{axis_name}|month={int(month)}"
            mask = exact & np.asarray(axis["game_month"]).__eq__(month)
            forms[key] = quadratic_gain_form(
                axis["target"], parents[axis_name], signed_directions[axis_name], mask
            )
            month_names.append(key)
    solved = solve_minimax(
        forms, list(AXES), month_names, selected_budget,
        seed=19200 + int(selected_budget * 1000),
    )
    weight = solved.pop("weight")
    candidates = {
        axis: stack_prediction(parents[axis], signed_directions[axis], weight)
        for axis in AXES
    }
    final_metrics = {
        axis: metrics(axes[axis], parents[axis], candidates[axis]) for axis in AXES
    }
    exact24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
    robust = _robustness(
        axes["full_2024"], parents["full_2024"], candidates["full_2024"],
        exact24, [candidates["full_2024"], parents["full_2024"]],
    )
    point_pass = bool(
        final_metrics["full_2022"]["gain"] > 0.0
        and final_metrics["full_2022"]["positive_month_fraction"] >= (6.0 / 7.0)
        and final_metrics["late_2023"]["gain"] > 0.0
        and final_metrics["late_2023"]["positive_month_fraction"] >= (2.0 / 3.0)
        and final_metrics["full_2024"]["gain"] > 0.0
        and final_metrics["full_2024"]["positive_month_fraction"] >= 0.625
        and min(item["worst_month_gain"] for item in final_metrics.values()) > -2.0
        and min(item["minimum_domain_gain"] for item in final_metrics.values()) >= 0.0
    )
    robust_pass = bool(
        robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] <= 0.10
    )
    net = weight[:len(families)] - weight[len(families):]
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=candidates["full_2022"],
        late_2023=candidates["late_2023"],
        full_2024=candidates["full_2024"],
        parent_full_2024=parents["full_2024"],
        signed_weight=weight,
        family_names=np.asarray(families),
        net_weight=net,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_pass" if point_pass and robust_pass else (
            "point_pass_robust_reject" if point_pass else "final_point_reject"
        ),
        "soft_loo_gate": {
            "gain_strict_min": 0.0,
            "positive_month_fraction_min": 0.25,
            "worst_month_gain_strict_min": -0.75,
            "minimum_domain_gain": 0.0,
        },
        "selected_budget_by_soft_loo": selected_budget,
        "budget_ranking": budget_ranking.to_dict(orient="records"),
        "final_solver": solved,
        "final_l1_weight": float(weight.sum()),
        "final_net_weights": {
            name: float(value) for name, value in zip(families, net) if abs(value) > 1e-8
        },
        "final_metrics": final_metrics,
        "robustness_2024": robust,
        "point_gate_passed": point_pass,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": bool(point_pass and robust_pass),
        "evidence_tier": "soft_leave_one_year_out",
        "parity": parity,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "soft_gate_frozen_before_final_triyear_fit": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_components_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v158-path", type=Path, required=True)
    parser.add_argument("--v165-summary", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--v191-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v158_path,
        args.v165_summary, args.library_root, args.v191_summary, args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
