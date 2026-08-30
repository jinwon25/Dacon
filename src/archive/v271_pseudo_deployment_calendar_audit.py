"""Audit the simplest pseudo-deployment ensemble in the inherited calendar.

The model formula is v268 ``equal_all``: an equal base/pseudo-XGB fallback on
every frozen v244 route.  The April--September window is inherited unchanged
from v70/v261.  No route or weight is selected in this module.  Because v268
results were already observed, this is explicitly exploratory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import align_regular_prediction
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics, route_masks
from src.archive.v248_fixed_route_dual_tree_fallback import ROUTE_WEIGHTS, compose
from src.archive.v253_fixed_route_command_dispersion_audit import AXES, SOURCE_AXES, _load_axes


PROTOCOL = "V271_PSEUDO_DEPLOYMENT_CALENDAR_AUDIT_V1"
PSEUDO_WEIGHT = 0.50
FIRST_ACTIVE_MONTH = 4
LAST_ACTIVE_MONTH = 9


def calendar_equal_fallback(
    base: np.ndarray,
    pseudo: np.ndarray,
    active: np.ndarray,
    month: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base, dtype=np.float64)
    pseudo = np.asarray(pseudo, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    month = np.asarray(month)
    if not (base.shape == pseudo.shape == active.shape == month.shape):
        raise ValueError("calendar equal arrays have different shapes")
    selected = active & (month >= FIRST_ACTIVE_MONTH) & (month <= LAST_ACTIVE_MONTH)
    fallback = base.copy()
    fallback[selected] = (
        (1.0 - PSEUDO_WEIGHT) * base[selected]
        + PSEUDO_WEIGHT * pseudo[selected]
    )
    return fallback, selected


def run(
    train_csv: Path,
    base_xgb_oof_dir: Path,
    pseudo_oof_dir: Path,
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

    year_by_axis = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    results: dict[str, Any] = {}
    parity: dict[str, float] = {}
    baselines: dict[str, np.ndarray] = {}
    candidates: dict[str, np.ndarray] = {}
    selected: dict[str, np.ndarray] = {}
    for axis in AXES:
        year = year_by_axis[axis]
        base = align_regular_prediction(
            frames[year],
            base_xgb_oof_dir / f"hyunku_runtime_faithful_xgb_{year}.npy",
        )
        pseudo = align_regular_prediction(
            frames[year], pseudo_oof_dir / f"pseudo_deployment_xgb_{year}.npy"
        )
        if axis == "late_2023":
            base = base[late23]
            pseudo = pseudo[late23]
        routes = route_masks(parents[axis], base, axis_frames[axis])
        baselines[axis], active = compose(parents[axis], base, routes)
        parity[axis] = float(
            np.max(np.abs(baselines[axis] - expected_v244[axis]))
        )
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")
        fallback, selected[axis] = calendar_equal_fallback(
            base,
            pseudo,
            active,
            axis_frames[axis]["game_month"].to_numpy(),
        )
        candidates[axis], candidate_active = compose(parents[axis], fallback, routes)
        if not np.array_equal(candidate_active, active):
            raise AssertionError("v244 route support changed")
        results[axis] = paired_metrics(
            axes[axis], baselines[axis], candidates[axis], selected[axis]
        )

    source_pass = all(results[axis]["gain"] > 0.0 for axis in SOURCE_AXES)
    locked_pass = results["full_2024"]["gain"] > 0.0
    robustness = _robustness(
        axes["full_2024"],
        baselines["full_2024"],
        candidates["full_2024"],
        selected["full_2024"],
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
        **{f"active_{axis}": selected[axis] for axis in AXES},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "exploratory_all_numeric_gates_pass"
            if source_pass and locked_pass and robust_pass
            else "source_reject"
            if not source_pass
            else "locked_or_robust_reject"
        ),
        "pseudo_weight": PSEUDO_WEIGHT,
        "calendar_window": [FIRST_ACTIVE_MONTH, LAST_ACTIVE_MONTH],
        "route_weights": ROUTE_WEIGHTS,
        "v244_reconstruction_max_abs": parity,
        "results": results,
        "source_gate_passed": source_pass,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_exploratory_packaging": bool(source_pass and locked_pass),
        "eligible_for_champion_promotion": False,
        "restrictions": {
            "v268_equal_all_formula_frozen": True,
            "calendar_inherited_from_v70_v261": True,
            "v268_source_and_locked_results_observed_before_this_audit": True,
            "exploratory_not_confirmatory": True,
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
    parser.add_argument("--pseudo-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.base_xgb_oof_dir,
        args.pseudo_oof_dir,
        args.exact_axes,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
