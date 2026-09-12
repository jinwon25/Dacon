"""Select one low-complexity LightGBM confirmation scope inside frozen v244."""

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
from src.archive.v248_fixed_route_dual_tree_fallback import (
    ALPHAS,
    AXES,
    SOURCE_AXES,
    compose,
    mix_fallback,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V249_DUAL_TREE_CONSENSUS_SCOPE_V1"
SCOPES = (
    "all_active",
    "model_close_01",
    "model_close_02",
    "same_direction",
    "same_direction_close_02",
    "same_direction_lgbm_conservative",
)


def scope_masks(
    parent: np.ndarray,
    xgb: np.ndarray,
    lightgbm: np.ndarray,
    active: np.ndarray,
) -> dict[str, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    xgb = np.asarray(xgb, dtype=np.float64)
    lightgbm = np.asarray(lightgbm, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    finite = np.isfinite(xgb) & np.isfinite(lightgbm)
    close01 = np.abs(lightgbm - xgb) <= 0.01
    close02 = np.abs(lightgbm - xgb) <= 0.02
    same_direction = (xgb - parent) * (lightgbm - parent) > 0.0
    conservative = np.abs(lightgbm - parent) < np.abs(xgb - parent)
    return {
        "all_active": active & finite,
        "model_close_01": active & finite & close01,
        "model_close_02": active & finite & close02,
        "same_direction": active & finite & same_direction,
        "same_direction_close_02": active & finite & same_direction & close02,
        "same_direction_lgbm_conservative": (
            active & finite & same_direction & conservative
        ),
    }


def select_candidate(results: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, float] | None:
    eligible: list[tuple[float, float, float, str]] = []
    for scope in SCOPES:
        for alpha in ALPHAS:
            key = f"{alpha:g}"
            items = [results[scope][key][axis] for axis in SOURCE_AXES]
            gains = [item["gain"] for item in items]
            if all(gain > 0.0 for gain in gains) and all(
                item["positive_month_fraction"] >= 2.0 / 3.0 for item in items
            ):
                eligible.append((min(gains), float(np.mean(gains)), -alpha, scope))
    if not eligible:
        return None
    _minimum, _mean, negative_alpha, scope = max(eligible)
    return scope, -negative_alpha


def restrictions() -> dict[str, bool]:
    return {
        "v244_parent_routes_and_weights_frozen": True,
        "six_predeclared_consensus_scopes_only": True,
        "scope_and_alpha_selected_on_full_2022_and_late_2023_only": True,
        "source_month_majority_required": True,
        "full_2024_locked_from_selection": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def _load_axes(contract_dir: Path, bridge_oof: Path) -> dict[str, dict[str, np.ndarray]]:
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    return {
        name: {**axis, "exact_mask": np.ones(len(axis["target"]), dtype=bool)}
        for name, axis in axes.items()
    }


def run(
    train_csv: Path,
    xgb_oof_dir: Path,
    lightgbm_oof_dir: Path,
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
            axis: saved[f"jy_parent_{axis}"].astype(np.float64)
            for axis in AXES
        }
    with np.load(v244_axes, allow_pickle=False) as saved:
        expected_v244 = {
            axis: saved[f"candidate_runtime_faithful_exact_jy_{axis}"].astype(np.float64)
            for axis in AXES
        }
    xgb_full = {
        year: align_regular_prediction(
            frames[year], xgb_oof_dir / f"hyunku_runtime_faithful_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    lgbm_full = {
        year: align_regular_prediction(
            frames[year], lightgbm_oof_dir / f"runtime_faithful_lgbm_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    xgb = {
        "full_2022": xgb_full[2022],
        "late_2023": xgb_full[2023][late23],
        "full_2024": xgb_full[2024],
    }
    lightgbm = {
        "full_2022": lgbm_full[2022],
        "late_2023": lgbm_full[2023][late23],
        "full_2024": lgbm_full[2024],
    }

    baselines: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    scopes: dict[str, dict[str, np.ndarray]] = {}
    parity: dict[str, float] = {}
    mixed_outputs: dict[tuple[str, str], np.ndarray] = {}
    for axis in AXES:
        routes = route_masks(parents[axis], xgb[axis], axis_frames[axis])
        baselines[axis], active[axis] = compose(parents[axis], xgb[axis], routes)
        parity[axis] = float(np.max(np.abs(baselines[axis] - expected_v244[axis])))
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")
        scopes[axis] = scope_masks(
            parents[axis], xgb[axis], lightgbm[axis], active[axis]
        )
        for alpha in ALPHAS:
            fallback = mix_fallback(xgb[axis], lightgbm[axis], alpha)
            mixed_outputs[(f"{alpha:g}", axis)], _ = compose(
                parents[axis], fallback, routes
            )

    results: dict[str, dict[str, dict[str, Any]]] = {scope: {} for scope in SCOPES}
    predictions: dict[tuple[str, str, str], np.ndarray] = {}
    for scope in SCOPES:
        for alpha in ALPHAS:
            key = f"{alpha:g}"
            results[scope][key] = {}
            for axis in AXES:
                mask = scopes[axis][scope]
                candidate = baselines[axis].copy()
                candidate[mask] = mixed_outputs[(key, axis)][mask]
                predictions[(scope, key, axis)] = candidate
                results[scope][key][axis] = paired_metrics(
                    axes[axis], baselines[axis], candidate, mask
                )

    selected = select_candidate(results)
    locked_pass = False
    robust_pass = False
    robustness = None
    if selected is not None:
        scope, alpha = selected
        key = f"{alpha:g}"
        locked_pass = results[scope][key]["full_2024"]["gain"] > 0.0
        family = [
            predictions[(candidate_scope, f"{candidate_alpha:g}", "full_2024")]
            for candidate_scope in SCOPES
            for candidate_alpha in ALPHAS
        ] + [baselines["full_2024"].copy()]
        robustness = _robustness(
            axes["full_2024"],
            baselines["full_2024"],
            predictions[(scope, key, "full_2024")],
            scopes["full_2024"][scope],
            family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] < 0.05
        )
        np.savez_compressed(
            output_dir / "selected_candidate_axes.npz",
            **{f"parent_{axis}": baselines[axis] for axis in AXES},
            **{
                f"candidate_{axis}": predictions[(scope, key, axis)]
                for axis in AXES
            },
            **{f"active_{axis}": scopes[axis][scope] for axis in AXES},
        )

    summary = {
        "protocol": PROTOCOL,
        "status": (
            "robust_candidate"
            if selected is not None and locked_pass and robust_pass
            else "locked_or_robust_reject"
            if selected is not None
            else "source_reject"
        ),
        "scopes": list(SCOPES),
        "alphas": list(ALPHAS),
        "selected": None if selected is None else {
            "scope": selected[0], "alpha": selected[1]
        },
        "v244_reconstruction_max_abs": parity,
        "results": results,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": bool(selected is not None and locked_pass and robust_pass),
        "eligible_for_packaging": False,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--xgb-oof-dir", type=Path, required=True)
    parser.add_argument("--lightgbm-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.xgb_oof_dir,
        args.lightgbm_oof_dir,
        args.exact_axes,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    selected = result["selected"]
    print(json.dumps({
        "status": result["status"],
        "selected": selected,
        "selected_results": None if selected is None else result["results"]
        [selected["scope"]][f"{selected['alpha']:g}"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
