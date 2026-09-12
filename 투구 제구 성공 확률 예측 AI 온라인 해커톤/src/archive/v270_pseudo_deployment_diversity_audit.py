"""Audit a fixed XGB/LightGBM pseudo-deployment diversity ensemble.

The two pseudo-deployment learners are averaged equally, then averaged equally
with the deployed base XGB inside every frozen v244 route.  There is one
candidate and no weight or route grid.  Because the v267 XGB source results
were known before this diversity branch, the result remains exploratory even
if all numerical gates pass.
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


PROTOCOL = "V270_PSEUDO_DEPLOYMENT_DIVERSITY_AUDIT_V1"
LEARNER_WEIGHT = 0.50
BASE_ENSEMBLE_WEIGHT = 0.50


def diversity_fallback(
    base: np.ndarray,
    pseudo_xgb: np.ndarray,
    pseudo_lgbm: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base, dtype=np.float64)
    pseudo_xgb = np.asarray(pseudo_xgb, dtype=np.float64)
    pseudo_lgbm = np.asarray(pseudo_lgbm, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if not (base.shape == pseudo_xgb.shape == pseudo_lgbm.shape == active.shape):
        raise ValueError("diversity arrays have different shapes")
    expert = (
        (1.0 - LEARNER_WEIGHT) * pseudo_xgb + LEARNER_WEIGHT * pseudo_lgbm
    )
    fallback = base.copy()
    fallback[active] = (
        (1.0 - BASE_ENSEMBLE_WEIGHT) * base[active]
        + BASE_ENSEMBLE_WEIGHT * expert[active]
    )
    return fallback, active.copy()


def _available_axes(xgb_dir: Path, lgbm_dir: Path) -> tuple[str, ...]:
    mapping = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    return tuple(
        axis
        for axis, year in mapping.items()
        if (xgb_dir / f"pseudo_deployment_xgb_{year}.npy").exists()
        and (lgbm_dir / f"pseudo_deployment_lgbm_{year}.npy").exists()
    )


def run(
    train_csv: Path,
    base_xgb_oof_dir: Path,
    pseudo_xgb_oof_dir: Path,
    pseudo_lgbm_oof_dir: Path,
    exact_axes: Path,
    v244_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    available = _available_axes(pseudo_xgb_oof_dir, pseudo_lgbm_oof_dir)
    if not available:
        raise FileNotFoundError("no matching pseudo-deployment diversity folds")
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
            axis: saved[f"jy_parent_{axis}"].astype(np.float64) for axis in available
        }
    with np.load(v244_axes, allow_pickle=False) as saved:
        expected_v244 = {
            axis: saved[f"candidate_runtime_faithful_exact_jy_{axis}"].astype(
                np.float64
            )
            for axis in available
        }

    axis_year = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    results: dict[str, Any] = {}
    parity: dict[str, float] = {}
    baselines: dict[str, np.ndarray] = {}
    candidates: dict[str, np.ndarray] = {}
    selected: dict[str, np.ndarray] = {}
    for axis in available:
        year = axis_year[axis]
        base = align_regular_prediction(
            frames[year],
            base_xgb_oof_dir / f"hyunku_runtime_faithful_xgb_{year}.npy",
        )
        pseudo_xgb = align_regular_prediction(
            frames[year],
            pseudo_xgb_oof_dir / f"pseudo_deployment_xgb_{year}.npy",
        )
        pseudo_lgbm = align_regular_prediction(
            frames[year],
            pseudo_lgbm_oof_dir / f"pseudo_deployment_lgbm_{year}.npy",
        )
        if axis == "late_2023":
            base = base[late23]
            pseudo_xgb = pseudo_xgb[late23]
            pseudo_lgbm = pseudo_lgbm[late23]
        routes = route_masks(parents[axis], base, axis_frames[axis])
        baselines[axis], active = compose(parents[axis], base, routes)
        parity[axis] = float(
            np.max(np.abs(baselines[axis] - expected_v244[axis]))
        )
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")
        fallback, selected[axis] = diversity_fallback(
            base, pseudo_xgb, pseudo_lgbm, active
        )
        candidates[axis], candidate_active = compose(parents[axis], fallback, routes)
        if not np.array_equal(candidate_active, active):
            raise AssertionError("v244 route support changed")
        results[axis] = paired_metrics(
            axes[axis], baselines[axis], candidates[axis], selected[axis]
        )

    source_ready = all(axis in available for axis in SOURCE_AXES)
    source_pass = source_ready and all(
        results[axis]["gain"] > 0.0
        and results[axis]["positive_month_fraction"] >= (2.0 / 3.0)
        for axis in SOURCE_AXES
    )
    locked_ready = "full_2024" in available
    locked_pass = locked_ready and results["full_2024"]["gain"] > 0.0
    robustness = None
    robust_pass = False
    if source_pass and locked_ready:
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
        **{f"parent_{axis}": baselines[axis] for axis in available},
        **{f"candidate_{axis}": candidates[axis] for axis in available},
        **{f"active_{axis}": selected[axis] for axis in available},
    )
    status = (
        "incomplete_source"
        if not source_ready
        else "source_reject"
        if not source_pass
        else "source_pass_pending_locked"
        if not locked_ready
        else "local_gate_pass"
        if locked_pass and robust_pass
        else "locked_or_robust_reject"
    )
    summary = {
        "protocol": PROTOCOL,
        "status": status,
        "available_axes": list(available),
        "learner_weight": LEARNER_WEIGHT,
        "base_ensemble_weight": BASE_ENSEMBLE_WEIGHT,
        "route_weights": ROUTE_WEIGHTS,
        "v244_reconstruction_max_abs": parity,
        "results": results,
        "source_gate_ready": source_ready,
        "source_gate_passed": source_pass,
        "locked_point_ready": locked_ready,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": bool(source_pass and locked_pass and robust_pass),
        "restrictions": {
            "single_fixed_diversity_formula": True,
            "v247_and_v267_hyperparameters_frozen": True,
            "v244_parent_routes_and_weights_frozen": True,
            "xgb_source_results_observed_before_diversity_branch": True,
            "exploratory_not_confirmatory": True,
            "full_2024_not_used_for_formula_selection": True,
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
    parser.add_argument("--pseudo-xgb-oof-dir", type=Path, required=True)
    parser.add_argument("--pseudo-lgbm-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.base_xgb_oof_dir,
        args.pseudo_xgb_oof_dir,
        args.pseudo_lgbm_oof_dir,
        args.exact_axes,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
