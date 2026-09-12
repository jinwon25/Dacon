"""Audit the exact-anchor fallback inside all frozen v244 routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
)
from src.archive.v241_mechanism_aware_fallback_expansion import (
    paired_metrics,
    route_masks,
)
from src.archive.v248_fixed_route_dual_tree_fallback import (
    AXES,
    SOURCE_AXES,
    _load_axes,
    compose,
)


PROTOCOL = "V285_EXACT_ANCHOR_FIXED_ROUTE_AUDIT_V1"
WEIGHTS = (0.25, 0.50, 0.75, 1.00)


def run(
    train_csv: Path,
    stale_oof_dir: Path,
    exact_oof_dir: Path,
    exact_axes: Path,
    v244_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _context, frames, _correction = _load_year_context(train_csv)
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = _load_axes(contract_dir, bridge_oof)
    with np.load(exact_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"jy_parent_{axis}"].astype(np.float64) for axis in AXES
        }
    with np.load(v244_axes, allow_pickle=False) as saved:
        expected_v244 = {
            axis: saved[f"candidate_runtime_faithful_exact_parent_{axis}"].astype(
                np.float64
            )
            for axis in AXES
        }
    stale_full = {
        year: align_regular_prediction(
            frames[year], stale_oof_dir / f"runtime_faithful_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    exact_full = {
        year: align_regular_prediction(
            frames[year], exact_oof_dir / f"training_parity_exact_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    stale = {
        "full_2022": stale_full[2022],
        "late_2023": stale_full[2023][late23],
        "full_2024": stale_full[2024],
    }
    exact = {
        "full_2022": exact_full[2022],
        "late_2023": exact_full[2023][late23],
        "full_2024": exact_full[2024],
    }
    routes = {
        axis: route_masks(parents[axis], stale[axis], axis_frames[axis])
        for axis in AXES
    }
    baseline: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    for axis in AXES:
        baseline[axis], active[axis] = compose(parents[axis], stale[axis], routes[axis])
        parity[axis] = float(np.max(np.abs(baseline[axis] - expected_v244[axis])))
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch: {axis} {parity[axis]}")

    details: dict[str, dict[str, Any]] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for weight in WEIGHTS:
        key = f"w{weight:g}"
        per_axis = {}
        for axis in AXES:
            fallback = (1.0 - weight) * stale[axis] + weight * exact[axis]
            candidate, candidate_active = compose(parents[axis], fallback, routes[axis])
            if not np.array_equal(candidate_active, active[axis]):
                raise AssertionError("v244 route support changed")
            predictions[(key, axis)] = candidate
            per_axis[axis] = paired_metrics(
                axes[axis], baseline[axis], candidate, active[axis]
            )
        source_pass = all(per_axis[axis]["gain"] > 0.0 for axis in SOURCE_AXES)
        rows.append(
            {
                "key": key,
                "weight": weight,
                "source_gate_passed": source_pass,
                "source_min_gain": min(per_axis[a]["gain"] for a in SOURCE_AXES),
                "source_mean_gain": float(
                    np.mean([per_axis[a]["gain"] for a in SOURCE_AXES])
                ),
                "locked_gain": per_axis["full_2024"]["gain"],
            }
        )
        details[key] = per_axis
    eligible = [row for row in rows if row["source_gate_passed"]]
    selected = max(
        eligible if eligible else rows,
        key=lambda row: (row["source_min_gain"], row["source_mean_gain"]),
    )
    selected_key = str(selected["key"])
    selected_weight = float(selected["weight"])
    robustness = _robustness(
        axes["full_2024"],
        baseline["full_2024"],
        predictions[(selected_key, "full_2024")],
        active["full_2024"],
        [predictions[(f"w{weight:g}", "full_2024")] for weight in WEIGHTS],
    )
    locked = details[selected_key]["full_2024"]
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    numeric_promote = bool(
        selected["source_gate_passed"]
        and locked["gain"] >= 4.0
        and robust_pass
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"baseline_{axis}": baseline[axis] for axis in AXES},
        **{
            f"candidate_{axis}": predictions[(selected_key, axis)]
            for axis in AXES
        },
        **{f"active_{axis}": active[axis] for axis in AXES},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "mechanistic_challenger" if not numeric_promote else "numeric_promote",
        "selected": selected,
        "selected_metrics": details[selected_key],
        "all_weights": rows,
        "v244_reconstruction_max_abs": parity,
        "locked_robustness": robustness,
        "numeric_promotion_gate_passed": numeric_promote,
        "eligible_for_exploratory_packaging": bool(
            selected["source_gate_passed"] and locked["gain"] > 0.0
        ),
        "mechanistic_basis": {
            "career_cumulative_asof_confirmed": True,
            "v244_runtime_used_season_start_anchor": True,
            "exact_runtime_uses_latest_prior_end_anchor": True,
            "standalone_exact_vs_stale_bss_gain": {
                "2022": 215.197922,
                "2023": 299.776700,
                "2024": 238.695224,
            },
        },
        "restrictions": {
            "official_train_only": True,
            "v244_routes_and_route_weights_frozen": True,
            "exact_anchor_dose_selected_on_2022_and_late2023_only": True,
            "full_2024_used_for_selection": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--stale-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.stale_oof_dir,
        args.exact_oof_dir,
        args.exact_axes,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
