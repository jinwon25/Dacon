"""Audit one preregistered pitch-mix release-variance dose in frozen v244 routes.

The candidate model is structurally different from the v244 fallback because it
adds strictly prior-season, pitch-type-specific TrackMan release and movement
moments.  This audit deliberately evaluates one fixed 25% model dose and keeps
all deployed v244 row routes unchanged; it is not a Public-score weight search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
)
from src.archive.v241_mechanism_aware_fallback_expansion import (
    paired_metrics,
    route_masks,
)
from src.archive.v248_fixed_route_dual_tree_fallback import ROUTE_WEIGHTS, compose
from src.archive.v253_fixed_route_command_dispersion_audit import (
    AXES,
    SOURCE_AXES,
    _load_axes,
)


PROTOCOL = "V256_PITCHMIX_RELEASE_FIXED_AUDIT_V1"
PITCHMIX_WEIGHT = 0.25


def mix_pitchmix(
    base_xgb: np.ndarray,
    pitchmix_xgb: np.ndarray,
    weight: float = PITCHMIX_WEIGHT,
) -> np.ndarray:
    """Convexly mix the frozen base and pitch-mix TrackMan models."""

    base_xgb = np.asarray(base_xgb, dtype=np.float64)
    pitchmix_xgb = np.asarray(pitchmix_xgb, dtype=np.float64)
    if base_xgb.shape != pitchmix_xgb.shape:
        raise ValueError("fallback arrays have different shapes")
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    return (1.0 - float(weight)) * base_xgb + float(weight) * pitchmix_xgb


def run(
    train_csv: Path,
    base_xgb_oof_dir: Path,
    pitchmix_xgb_oof_dir: Path,
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
            axis: saved[f"candidate_runtime_faithful_exact_jy_{axis}"].astype(
                np.float64
            )
            for axis in AXES
        }

    base_full = {
        year: align_regular_prediction(
            frames[year],
            base_xgb_oof_dir / f"hyunku_runtime_faithful_xgb_{year}.npy",
        )
        for year in (2022, 2023, 2024)
    }
    pitchmix_full = {
        year: align_regular_prediction(
            frames[year],
            pitchmix_xgb_oof_dir / f"pitchmix_release_xgb_{year}.npy",
        )
        for year in (2022, 2023, 2024)
    }
    base = {
        "full_2022": base_full[2022],
        "late_2023": base_full[2023][late23],
        "full_2024": base_full[2024],
    }
    pitchmix = {
        "full_2022": pitchmix_full[2022],
        "late_2023": pitchmix_full[2023][late23],
        "full_2024": pitchmix_full[2024],
    }
    routes = {
        axis: route_masks(parents[axis], base[axis], axis_frames[axis])
        for axis in AXES
    }

    baselines: dict[str, np.ndarray] = {}
    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    results: dict[str, Any] = {}
    for axis in AXES:
        baselines[axis], active[axis] = compose(
            parents[axis], base[axis], routes[axis]
        )
        parity[axis] = float(np.max(np.abs(baselines[axis] - expected_v244[axis])))
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")
        mixed = mix_pitchmix(base[axis], pitchmix[axis])
        candidates[axis], candidate_active = compose(
            parents[axis], mixed, routes[axis]
        )
        if not np.array_equal(candidate_active, active[axis]):
            raise AssertionError("fixed route support changed")
        results[axis] = paired_metrics(
            axes[axis], baselines[axis], candidates[axis], active[axis]
        )

    source_pass = all(results[axis]["gain"] > 0.0 for axis in SOURCE_AXES)
    locked_pass = results["full_2024"]["gain"] > 0.0
    robustness = _robustness(
        axes["full_2024"],
        baselines["full_2024"],
        candidates["full_2024"],
        active["full_2024"],
        [candidates["full_2024"], baselines["full_2024"].copy()],
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.05
    )
    np.savez_compressed(
        output_dir / "candidate_axes.npz",
        **{f"parent_{axis}": baselines[axis] for axis in AXES},
        **{f"candidate_{axis}": candidates[axis] for axis in AXES},
        **{f"active_{axis}": active[axis] for axis in AXES},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "local_gate_pass"
            if source_pass and locked_pass and robust_pass
            else "source_reject"
            if not source_pass
            else "locked_or_robust_reject"
        ),
        "pitchmix_weight": PITCHMIX_WEIGHT,
        "route_weights": ROUTE_WEIGHTS,
        "v244_reconstruction_max_abs": parity,
        "results": results,
        "source_gate_passed": source_pass,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": bool(source_pass and locked_pass and robust_pass),
        "eligible_for_packaging": False,
        "restrictions": {
            "one_preregistered_mix_weight_only": True,
            "v244_parent_routes_and_weights_frozen": True,
            "full_2024_not_used_for_feature_or_weight_selection": True,
            "full_2024_development_contaminated": True,
            "reality_check_is_family_local_not_lifetime_global": True,
            "test_csv_read": False,
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
    parser.add_argument("--base-xgb-oof-dir", type=Path, required=True)
    parser.add_argument("--pitchmix-xgb-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.base_xgb_oof_dir,
        args.pitchmix_xgb_oof_dir,
        args.exact_axes,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "pitchmix_weight": result["pitchmix_weight"],
                "v244_reconstruction_max_abs": result[
                    "v244_reconstruction_max_abs"
                ],
                "results": result["results"],
                "robustness": result["robustness"],
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
