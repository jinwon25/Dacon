"""Mechanism-aware expansion of the frozen Public1175 fallback XGB.

This is a research/confirmation audit, not a clean locked-holdout claim.  The
three low-complexity routes were discovered while inspecting all three OOF
axes, including 2024.  We therefore require directional agreement on both the
historical evidence parent and the independently reconstructed exact-row-region parent,
and report dependence-aware uncertainty without marking the result packageable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    XGB_THRESHOLD,
    XGB_WEIGHT,
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.core.contract import _load_contract_axis
from src.robust_local_evaluation import paired_score_summary


PROTOCOL = "V241_MECHANISM_AWARE_FALLBACK_EXPANSION_V1"
AXES = ("full_2022", "late_2023", "full_2024")
PARENT_FAMILIES = ("evidence_proxy", "exact_parent")

BOUNDARY_LOW = 0.48
BOUNDARY_HIGH = 0.50
BOUNDARY_AGREEMENT = 0.02
BOUNDARY_WEIGHT = 0.30
SAME_HAND_WEIGHT = 0.10
OPPOSITE_HAND_MINIMUM = 0.52
OPPOSITE_HAND_WEIGHT = 0.10


def route_masks(parent: np.ndarray, xgb: np.ndarray, frame) -> dict[str, np.ndarray]:
    """Return mutually exclusive deployed and expansion routes."""
    parent = np.asarray(parent, dtype=np.float64)
    xgb = np.asarray(xgb, dtype=np.float64)
    if len(parent) != len(xgb) or len(parent) != len(frame):
        raise ValueError("fallback route inputs have different lengths")

    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    core = regular & ~(
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    pressure = pressure_gate(frame)
    same_hand = frame["pitcher_hand"].astype(str).eq(
        frame["batter_hand"].astype(str)
    ).to_numpy()
    finite = np.isfinite(xgb)

    routes = {
        "deployed": core & pressure & (parent >= XGB_THRESHOLD) & finite,
        "pressure_boundary_agreement": (
            core
            & pressure
            & (parent >= BOUNDARY_LOW)
            & (parent < BOUNDARY_HIGH)
            & (np.abs(xgb - parent) <= BOUNDARY_AGREEMENT)
            & finite
        ),
        "nonpressure_same_hand": core & ~pressure & same_hand & finite,
        "nonpressure_opposite_hand_high52": (
            core
            & ~pressure
            & ~same_hand
            & (parent >= OPPOSITE_HAND_MINIMUM)
            & finite
        ),
    }
    names = tuple(routes)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if np.any(routes[left] & routes[right]):
                raise AssertionError(f"fallback routes overlap: {left}, {right}")
    return routes


def apply_candidate(
    parent: np.ndarray,
    xgb: np.ndarray,
    routes: dict[str, np.ndarray],
    enabled: tuple[str, ...] = (
        "pressure_boundary_agreement",
        "nonpressure_same_hand",
        "nonpressure_opposite_hand_high52",
    ),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen fallback and selected disjoint expansion routes."""
    parent = np.asarray(parent, dtype=np.float64)
    xgb = np.asarray(xgb, dtype=np.float64)
    baseline = parent.copy()
    deployed = routes["deployed"]
    baseline[deployed] = np.clip(
        parent[deployed] + XGB_WEIGHT * (xgb[deployed] - parent[deployed]),
        0.001,
        0.999,
    )

    weights = {
        "pressure_boundary_agreement": BOUNDARY_WEIGHT,
        "nonpressure_same_hand": SAME_HAND_WEIGHT,
        "nonpressure_opposite_hand_high52": OPPOSITE_HAND_WEIGHT,
    }
    candidate = baseline.copy()
    changed = np.zeros(len(parent), dtype=bool)
    for name in enabled:
        if name not in weights:
            raise ValueError(f"unknown expansion route: {name}")
        active = routes[name]
        candidate[active] = np.clip(
            parent[active] + weights[name] * (xgb[active] - parent[active]),
            0.001,
            0.999,
        )
        changed |= active
    return baseline, candidate, changed


def paired_metrics(
    axis: dict[str, np.ndarray],
    baseline: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(axis["target"], dtype=np.float64)
    exact = np.asarray(axis["exact_mask"], dtype=bool)
    active = np.asarray(active, dtype=bool) & exact
    overall = paired_score_summary(target[exact], candidate[exact], baseline[exact])
    routed = paired_score_summary(target[active], candidate[active], baseline[active])
    months = []
    for month in sorted(np.unique(np.asarray(axis["game_month"])[active]).tolist()):
        mask = active & (np.asarray(axis["game_month"]) == month)
        item = paired_score_summary(target[mask], candidate[mask], baseline[mask])
        months.append(
            {
                "month": int(month),
                "n_rows": int(mask.sum()),
                "gain": item["unclipped_bss_equivalent_gain"],
            }
        )
    month_gains = np.asarray([item["gain"] for item in months], dtype=np.float64)
    return {
        "gain": overall["unclipped_bss_equivalent_gain"],
        "candidate_brier": overall["candidate_brier"],
        "baseline_brier": overall["incumbent_brier"],
        "active_gain": routed["unclipped_bss_equivalent_gain"],
        "active_rows": int(active.sum()),
        "mean_abs_shift_active": float(
            np.mean(np.abs(candidate[active] - baseline[active]))
        ),
        "positive_month_fraction": float(np.mean(month_gains > 0.0)),
        "worst_month_gain": float(month_gains.min()),
        "months": months,
    }


def _full_axis(axis: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {**axis, "exact_mask": np.ones(len(axis["target"]), dtype=bool)}


def _parents_from_exact_axes(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {
            name: saved[f"jy_parent_{name}"].astype(np.float64)
            for name in AXES
        }


def restrictions() -> dict[str, bool]:
    return {
        "official_bss_exactly_recomputed": True,
        "fallback_xgb_model_frozen": True,
        "deployed_route_and_weight_frozen": True,
        "expansion_routes_pairwise_disjoint": True,
        "two_parent_families_required": True,
        "full_2024_seen_during_route_discovery": True,
        "clean_locked_holdout_claim": False,
        "test_csv_read": False,
        "public_score_used_for_route_selection": False,
    }


def run(
    train_csv: Path,
    fallback_oof_dir: Path,
    fallback_filename_pattern: str,
    contract_dir: Path,
    bridge_oof: Path,
    exact_axes: Path,
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
    axes = {
        "full_2022": _full_axis(_load_contract_axis(contract_dir / "v84_full_2022.npz")),
        "late_2023": _full_axis(_load_contract_axis(contract_dir / "v84_late_2023.npz")),
        "full_2024": _full_axis(_load_contract_axis(bridge_oof)),
    }
    xgb_full = {
        year: align_regular_prediction(
            frames[year],
            fallback_oof_dir / fallback_filename_pattern.format(year=year),
        )
        for year in (2022, 2023, 2024)
    }
    xgb = {
        "full_2022": xgb_full[2022],
        "late_2023": xgb_full[2023][late23],
        "full_2024": xgb_full[2024],
    }
    parent_families = {
        "evidence_proxy": {
            name: np.asarray(axes[name]["parent"], dtype=np.float64)
            for name in AXES
        },
        "exact_parent": _parents_from_exact_axes(exact_axes),
    }

    route_names = (
        "pressure_boundary_agreement",
        "nonpressure_same_hand",
        "nonpressure_opposite_hand_high52",
    )
    results: dict[str, Any] = {}
    saved_arrays: dict[str, np.ndarray] = {}
    locked_family: list[np.ndarray] = []
    exact_locked = None
    exact_locked_base = None
    exact_locked_active = None
    exact_locked_axis = None

    for family_name in PARENT_FAMILIES:
        results[family_name] = {}
        for axis_name in AXES:
            parent = parent_families[family_name][axis_name]
            if len(parent) != len(axes[axis_name]["target"]):
                raise ValueError(f"parent/axis mismatch: {family_name}/{axis_name}")
            routes = route_masks(parent, xgb[axis_name], axis_frames[axis_name])
            baseline, combined, changed = apply_candidate(parent, xgb[axis_name], routes)
            components: dict[str, Any] = {}
            component_candidates = []
            for route_name in route_names:
                _, candidate, active = apply_candidate(
                    parent, xgb[axis_name], routes, (route_name,)
                )
                components[route_name] = paired_metrics(
                    axes[axis_name], baseline, candidate, active
                )
                component_candidates.append(candidate)
            results[family_name][axis_name] = {
                "deployed_rows": int(routes["deployed"].sum()),
                "components": components,
                "combined": paired_metrics(
                    axes[axis_name], baseline, combined, changed
                ),
            }
            saved_arrays[f"baseline_{family_name}_{axis_name}"] = baseline
            saved_arrays[f"candidate_{family_name}_{axis_name}"] = combined
            saved_arrays[f"active_{family_name}_{axis_name}"] = changed
            if family_name == "exact_parent" and axis_name == "full_2024":
                exact_locked = combined
                exact_locked_base = baseline
                exact_locked_active = changed
                exact_locked_axis = axes[axis_name]
                locked_family = component_candidates + [baseline.copy()]

    directional_agreement = all(
        results[family][axis]["combined"]["gain"] > 0.0
        and all(
            results[family][axis]["components"][route]["gain"] > 0.0
            for route in route_names
        )
        for family in PARENT_FAMILIES
        for axis in AXES
    )
    assert exact_locked is not None
    assert exact_locked_base is not None
    assert exact_locked_active is not None
    assert exact_locked_axis is not None
    robustness = _robustness(
        exact_locked_axis,
        exact_locked_base,
        exact_locked,
        exact_locked_active,
        locked_family,
    )
    robust_positive = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
    )

    np.savez_compressed(output_dir / "candidate_axes.npz", **saved_arrays)
    summary = {
        "protocol": PROTOCOL,
        "fallback_filename_pattern": fallback_filename_pattern,
        "status": (
            "research_candidate_dual_parent_robust"
            if directional_agreement and robust_positive
            else "research_candidate_needs_confirmation"
        ),
        "routes": {
            "deployed": {"minimum_parent": XGB_THRESHOLD, "weight": XGB_WEIGHT},
            "pressure_boundary_agreement": {
                "parent_interval": [BOUNDARY_LOW, BOUNDARY_HIGH],
                "maximum_abs_model_gap": BOUNDARY_AGREEMENT,
                "weight": BOUNDARY_WEIGHT,
            },
            "nonpressure_same_hand": {"weight": SAME_HAND_WEIGHT},
            "nonpressure_opposite_hand_high52": {
                "minimum_parent": OPPOSITE_HAND_MINIMUM,
                "weight": OPPOSITE_HAND_WEIGHT,
            },
        },
        "results": results,
        "directional_agreement_all_components_all_axes": directional_agreement,
        "locked_exact_2024_robustness": robustness,
        "locked_exact_2024_robust_positive": robust_positive,
        "eligible_for_release_build": bool(directional_agreement and robust_positive),
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
    parser.add_argument("--fallback-oof-dir", type=Path, required=True)
    parser.add_argument(
        "--fallback-filename-pattern",
        default="fallback_xgb_oof_{year}.npy",
    )
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.fallback_oof_dir,
        args.fallback_filename_pattern,
        args.contract_dir,
        args.bridge_oof,
        args.exact_axes,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "directional_agreement": result[
                    "directional_agreement_all_components_all_axes"
                ],
                "exact_parent": {
                    axis: result["results"]["exact_parent"][axis]["combined"]
                    for axis in AXES
                },
                "robustness": result["locked_exact_2024_robustness"],
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
