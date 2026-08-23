"""Census and forward constrained stack of available exact-row OOF predictions.

Only candidates with matching late-2023 and full-2024 OOF are allowed into
the forward stack.  Additional 2024-only archives are used for same-axis
headroom diagnostics and can never receive a fitted deployment weight here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics
from src.v77_team_oof_constrained_blend import (
    evaluate_axes,
    fit_robust_blend,
    single_candidate_headroom,
)
from src.core.oof_bank import (
    _archive,
    _assert_target,
    _load_common_candidates,
    _load_full24_diagnostic_candidates,
    _load_parent_axis,
    rebase_frozen_candidate,
)


def _load_late24_diagnostic_candidates(
    project: Path,
    parent_axis: dict[str, np.ndarray],
    common: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    artifacts = project / "artifacts"
    target = parent_axis["target"]
    parent = parent_axis["parent"]
    output = dict(common)
    registries = {
        "v37_latest_catboost_shift": ("v37_latest_catboost_20260817_01", "audit_late_2024.npz"),
        "v46_sparse_logit_shift": ("v46_sparse_logit_20260817_01", "replication_late_2024.npz"),
        "v47_pitcher_role_shift": ("v47_pitcher_role_20260817_01", "replication_late_2024.npz"),
        "v48_level_transition_shift": ("v48_level_transition_20260817_01", "replication_late_2024.npz"),
        "v50_low_rank_context_shift": ("v50_low_rank_pitcher_context_20260817_01", "replication_late_2024.npz"),
        "v57_independent_blend_shift": ("v57_public_strict_blend_20260817_01", "replication_late_2024.npz"),
    }
    for name, (directory, filename) in registries.items():
        saved = _archive(artifacts / directory / filename)
        _assert_target(target, saved, name)
        output[name] = rebase_frozen_candidate(
            parent, saved["v27"], saved["candidate"]
        )

    tabm = _archive(
        artifacts / "v45_tabm_mini_20260817_01" / "audit_predictions.npz"
    )
    if not np.array_equal(target, np.asarray(tabm["late_target"], dtype=np.float64)):
        raise ValueError("target/order mismatch: v45/late_2024")
    output["v45_tabm_shift"] = rebase_frozen_candidate(
        parent, tabm["late_parent"], tabm["late_candidate"]
    )
    return output


def _frame(
    axis_name: str,
    axis: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    keep: np.ndarray | None = None,
) -> pd.DataFrame:
    if keep is None:
        keep = np.ones(len(axis["target"]), dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    data: dict[str, Any] = {
        "axis": np.repeat(axis_name, int(keep.sum())),
        "target": axis["target"][keep],
        "incumbent_probability": axis["parent"][keep],
        "domain3": axis["domain3"][keep],
        "month": axis["month"][keep],
    }
    for name, value in candidates.items():
        data[f"candidate__{name}"] = np.asarray(value, dtype=np.float64)[keep]
    return pd.DataFrame(data)


def _fit_transition(
    source: pd.DataFrame,
    audit: pd.DataFrame,
    candidate_names: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    fitted = fit_robust_blend(
        source,
        candidate_names,
        [str(source["axis"].iloc[0])],
        max_total_candidate_weight=float(config["max_total_candidate_weight"]),
        ridge=float(config["ridge"]),
        max_group_brier_increase=float(config["max_group_brier_increase"]),
        min_group_rows=int(config["min_group_rows"]),
    )
    evaluated = evaluate_axes(
        pd.concat([source, audit], ignore_index=True),
        candidate_names,
        fitted["weights"],
    )
    metrics = {row["axis"]: row for row in evaluated}
    audit_name = str(audit["axis"].iloc[0])
    result = metrics[audit_name]
    gates = config["gates"]
    tolerance = float(config["nonzero_weight_tolerance"])
    checks = {
        "nonzero_candidate_weight": sum(fitted["weights"].values()) > tolerance,
        "gain_strictly_positive": float(result["gain"]) > 0.0,
        "positive_month_fraction": float(result["positive_month_fraction"])
        >= float(gates["positive_month_fraction_min"]),
        "worst_month_gain": float(result["worst_month_gain"])
        > float(gates["worst_month_gain_min_exclusive"]),
        "minimum_domain_gain": float(result["minimum_domain_gain"])
        >= float(gates["minimum_domain_gain_min"]),
    }
    return {
        "fit": fitted,
        "source_metrics": metrics[str(source["axis"].iloc[0])],
        "audit_metrics": result,
        "gates": checks,
        "passes_numeric_gate": bool(all(checks.values())),
    }


def _headroom_rows(
    axis_name: str,
    axis: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    keep: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if keep is None:
        keep = np.ones(len(axis["target"]), dtype=bool)
    target = axis["target"][keep]
    parent = axis["parent"][keep]
    parent_error = parent - target
    rows = []
    for name, values in candidates.items():
        candidate = np.asarray(values, dtype=np.float64)[keep]
        headroom = single_candidate_headroom(target, parent, candidate)
        error = candidate - target
        residual_correlation = (
            0.0
            if np.std(error) == 0.0 or np.std(parent_error) == 0.0
            else float(np.corrcoef(parent_error, error)[0, 1])
        )
        raw = diagnostics(
            pd.DataFrame(
                {
                    "target": target,
                    "game_month": axis["month"][keep],
                    "domain3": axis["domain3"][keep],
                }
            ),
            parent,
            candidate,
            np.ones(len(target), dtype=bool),
        )
        rows.append(
            {
                "axis": axis_name,
                "candidate": name,
                "raw_gain": float(raw["gain"]),
                "raw_worst_month_gain": float(raw["worst_month_gain"]),
                "raw_minimum_domain_gain": float(raw["minimum_domain_gain"]),
                "optimal_candidate_weight_same_axis": headroom[
                    "optimal_candidate_weight"
                ],
                "oracle_gain_same_axis": headroom["unclipped_bss_gain"],
                "prediction_correlation": headroom["prediction_correlation"],
                "residual_correlation": residual_correlation,
                "shift_rms": headroom["shift_rms"],
                "diagnostic_only": True,
            }
        )
    return rows


def run(
    project: Path,
    final_parent_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    final_parent_dir = final_parent_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = {
        name: _load_parent_axis(final_parent_dir, name)
        for name in ("late_2023", "full_2024", "late_2024")
    }
    common = {
        name: _load_common_candidates(project, axes[name], name)
        for name in axes
    }
    expected = list(config["common_candidates"])
    for axis_name, candidates in common.items():
        if list(candidates) != expected:
            raise ValueError(f"candidate registry mismatch on {axis_name}")

    # Metadata and labels must match the full-2024 late slice.  Parent and
    # candidates intentionally differ because the replication models refit on
    # early-2024 before predicting late-2024.
    late_mask = axes["full_2024"]["month"] >= 8
    for key in ("target", "domain3", "month"):
        if not np.array_equal(axes["full_2024"][key][late_mask], axes["late_2024"][key]):
            raise ValueError(f"full/late 2024 parity failure: {key}")

    late23_source = _frame("late_2023_source", axes["late_2023"], common["late_2023"])
    full24_audit = _frame("full_2024_audit", axes["full_2024"], common["full_2024"])
    early_mask = axes["full_2024"]["month"] <= 7
    early24_source = _frame(
        "early_2024_source", axes["full_2024"], common["full_2024"], early_mask
    )
    late24_audit = _frame(
        "late_2024_audit", axes["late_2024"], common["late_2024"]
    )
    transitions = {
        "late23_to_full24": _fit_transition(
            late23_source, full24_audit, expected, config
        ),
        "early24_to_late24": _fit_transition(
            early24_source, late24_audit, expected, config
        ),
    }

    broad = _load_full24_diagnostic_candidates(
        project, axes["full_2024"], common["full_2024"]
    )
    broad_late = _load_late24_diagnostic_candidates(
        project, axes["late_2024"], common["late_2024"]
    )
    headroom = [
        *_headroom_rows("full_2024", axes["full_2024"], broad),
        *_headroom_rows("late_2024", axes["late_2024"], broad_late),
    ]
    pd.DataFrame(headroom).to_csv(output_dir / "same_axis_headroom.csv", index=False)
    weight_rows = []
    for transition, result in transitions.items():
        for candidate, weight in result["fit"]["weights"].items():
            weight_rows.append(
                {"transition": transition, "candidate": candidate, "weight": weight}
            )
    pd.DataFrame(weight_rows).to_csv(output_dir / "forward_weights.csv", index=False)

    numeric_pass = all(item["passes_numeric_gate"] for item in transitions.values())
    result = {
        "protocol": config["protocol"],
        "candidate_count_forward": len(expected),
        "candidate_count_same_axis_diagnostic": {
            "full_2024": len(broad),
            "late_2024": len(broad_late),
        },
        "transitions": transitions,
        "gates": {
            "both_forward_transitions_pass": numeric_pass,
            "bootstrap_pass": False,
            "reality_check_pass": False,
        },
        "same_axis_headroom_is_diagnostic_only": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_weights": False,
        "eligible_for_packaging": False,
        "promotion_block": (
            "Forward gates failed."
            if not numeric_pass
            else "Bootstrap and family-wide Reality Check remain required."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.final_parent_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()
