"""Audit XGB/LightGBM fallback mixing inside the frozen promoted v244 routes."""

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
from src.core.contract import _load_contract_axis


PROTOCOL = "V248_FIXED_ROUTE_DUAL_TREE_FALLBACK_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
ALPHAS = (0.25, 0.50, 0.75, 1.00)
ROUTE_WEIGHTS = {
    "deployed": 0.30,
    "pressure_boundary_agreement": 0.45,
    "nonpressure_same_hand": 0.15,
    "nonpressure_opposite_hand_high52": 0.15,
}


def mix_fallback(xgb: np.ndarray, lightgbm: np.ndarray, alpha: float) -> np.ndarray:
    xgb = np.asarray(xgb, dtype=np.float64)
    lightgbm = np.asarray(lightgbm, dtype=np.float64)
    if xgb.shape != lightgbm.shape:
        raise ValueError("fallback model arrays have different shapes")
    return (1.0 - float(alpha)) * xgb + float(alpha) * lightgbm


def compose(
    parent: np.ndarray,
    fallback: np.ndarray,
    routes: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    output = parent.copy()
    active = np.zeros(len(parent), dtype=bool)
    for name, weight in ROUTE_WEIGHTS.items():
        mask = np.asarray(routes[name], dtype=bool)
        if not np.isfinite(fallback[mask]).all():
            raise ValueError(f"missing fallback predictions on route {name}")
        output[mask] = np.clip(
            parent[mask] + weight * (fallback[mask] - parent[mask]),
            0.001,
            0.999,
        )
        active |= mask
    return output, active


def select_alpha(results: dict[str, dict[str, Any]]) -> float | None:
    eligible: list[tuple[float, float, float]] = []
    for alpha in ALPHAS:
        key = f"{alpha:g}"
        gains = [results[key][axis]["gain"] for axis in SOURCE_AXES]
        if all(gain > 0.0 for gain in gains):
            eligible.append((min(gains), float(np.mean(gains)), -alpha))
    return None if not eligible else -max(eligible)[2]


def restrictions() -> dict[str, bool]:
    return {
        "v244_parent_and_routes_frozen": True,
        "v244_route_weights_frozen": True,
        "xgb_and_lightgbm_share_runtime_feature_contract": True,
        "model_mix_selected_on_full_2022_and_late_2023_only": True,
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

    routes = {
        axis: route_masks(parents[axis], xgb[axis], axis_frames[axis])
        for axis in AXES
    }
    baselines: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    for axis in AXES:
        baselines[axis], active[axis] = compose(parents[axis], xgb[axis], routes[axis])
        parity[axis] = float(np.max(np.abs(baselines[axis] - expected_v244[axis])))
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")

    results: dict[str, dict[str, Any]] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for alpha in ALPHAS:
        key = f"{alpha:g}"
        results[key] = {}
        for axis in AXES:
            fallback = mix_fallback(xgb[axis], lightgbm[axis], alpha)
            candidate, candidate_active = compose(parents[axis], fallback, routes[axis])
            if not np.array_equal(candidate_active, active[axis]):
                raise AssertionError("fixed route support changed")
            predictions[(key, axis)] = candidate
            results[key][axis] = paired_metrics(
                axes[axis], baselines[axis], candidate, active[axis]
            )

    selected = select_alpha(results)
    locked_pass = False
    robust_pass = False
    robustness = None
    if selected is not None:
        key = f"{selected:g}"
        locked_pass = results[key]["full_2024"]["gain"] > 0.0
        family = [
            predictions[(f"{alpha:g}", "full_2024")] for alpha in ALPHAS
        ] + [baselines["full_2024"].copy()]
        robustness = _robustness(
            axes["full_2024"],
            baselines["full_2024"],
            predictions[(key, "full_2024")],
            active["full_2024"],
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
            **{f"candidate_{axis}": predictions[(key, axis)] for axis in AXES},
            **{f"active_{axis}": active[axis] for axis in AXES},
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
        "alphas": list(ALPHAS),
        "route_weights": ROUTE_WEIGHTS,
        "selected_alpha": selected,
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
    selected = result["selected_alpha"]
    print(json.dumps({
        "status": result["status"],
        "selected_alpha": selected,
        "v244_reconstruction_max_abs": result["v244_reconstruction_max_abs"],
        "selected_results": None if selected is None else result["results"][f"{selected:g}"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
