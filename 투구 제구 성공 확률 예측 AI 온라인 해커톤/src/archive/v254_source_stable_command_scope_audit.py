"""Select a command-dispersion scope on 2022/late-2023, then audit 2024 once.

The finite scope/dose family is declared in this file.  Full-2024 is never
used to select a member.  Reality Check receives every declared member.
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
from src.archive.v253_fixed_route_command_dispersion_audit import _load_axes, mix_fallback


PROTOCOL = "V254_SOURCE_STABLE_COMMAND_SCOPE_AUDIT_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
MIX_WEIGHTS = (0.15, 0.25, 0.40, 0.60)


def scope_masks(
    parent: np.ndarray,
    base: np.ndarray,
    command: np.ndarray,
    routes: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Finite, outcome-free masks fixed before source-label evaluation."""

    active = np.logical_or.reduce(list(routes.values()))
    deployed = routes["deployed"]
    boundary = routes["pressure_boundary_agreement"]
    same = routes["nonpressure_same_hand"]
    opposite = routes["nonpressure_opposite_hand_high52"]
    delta_base = base - parent
    delta_command = command - parent
    disagreement = np.abs(command - base)
    same_parent_direction = delta_base * delta_command >= 0.0
    same_center_direction = (base - 0.5) * (command - 0.5) >= 0.0
    return {
        "all": active,
        "deployed": deployed,
        "pressure": deployed | boundary,
        "boundary": boundary,
        "nonpressure": same | opposite,
        "same_hand": same,
        "opposite_hand": opposite,
        "same_parent_direction": active & same_parent_direction,
        "opposite_parent_direction": active & ~same_parent_direction,
        "same_center_direction": active & same_center_direction,
        "command_toward_center": active & (np.abs(command - 0.5) < np.abs(base - 0.5)),
        "command_more_extreme": active & (np.abs(command - 0.5) >= np.abs(base - 0.5)),
        "disagreement_le_005": active & (disagreement <= 0.005),
        "disagreement_le_010": active & (disagreement <= 0.010),
        "disagreement_le_020": active & (disagreement <= 0.020),
        "base_low50": active & (base < 0.50),
        "base_high50": active & (base >= 0.50),
        "parent_mid": active & (parent >= 0.45) & (parent < 0.55),
        "parent_tail": active & ((parent < 0.45) | (parent >= 0.55)),
    }


def compose_scoped(
    parent: np.ndarray,
    base: np.ndarray,
    command: np.ndarray,
    routes: dict[str, np.ndarray],
    scope: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline, active = compose(parent, base, routes)
    mixed = mix_fallback(base, command, weight)
    candidate = baseline.copy()
    selected = np.zeros(len(parent), dtype=bool)
    for name, route_weight in ROUTE_WEIGHTS.items():
        mask = np.asarray(routes[name], dtype=bool) & np.asarray(scope, dtype=bool)
        candidate[mask] = np.clip(
            parent[mask] + route_weight * (mixed[mask] - parent[mask]),
            0.001,
            0.999,
        )
        selected |= mask
    return baseline, candidate, selected & active


def select_candidate(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, float, str]] = []
    for name, axes in results.items():
        source = [axes[axis] for axis in SOURCE_AXES]
        if not all(item["gain"] > 0.0 for item in source):
            continue
        if not all(item["positive_month_fraction"] >= (2.0 / 3.0) for item in source):
            continue
        if not all(item["worst_month_gain"] > -5.0 for item in source):
            continue
        gains = [item["gain"] for item in source]
        active_rows = min(item["active_rows"] for item in source)
        if active_rows < 5000:
            continue
        eligible.append((min(gains), float(np.mean(gains)), active_rows, name))
    return None if not eligible else max(eligible)[3]


def run(
    train_csv: Path,
    base_xgb_oof_dir: Path,
    command_xgb_oof_dir: Path,
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
            frames[year], base_xgb_oof_dir / f"hyunku_runtime_faithful_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    command_full = {
        year: align_regular_prediction(
            frames[year], command_xgb_oof_dir / f"command_dispersion_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    base = {
        "full_2022": base_full[2022],
        "late_2023": base_full[2023][late23],
        "full_2024": base_full[2024],
    }
    command = {
        "full_2022": command_full[2022],
        "late_2023": command_full[2023][late23],
        "full_2024": command_full[2024],
    }
    routes = {
        axis: route_masks(parents[axis], base[axis], axis_frames[axis])
        for axis in AXES
    }
    scopes = {
        axis: scope_masks(parents[axis], base[axis], command[axis], routes[axis])
        for axis in AXES
    }
    baselines = {
        axis: compose(parents[axis], base[axis], routes[axis])[0] for axis in AXES
    }
    parity = {
        axis: float(np.max(np.abs(baselines[axis] - expected_v244[axis])))
        for axis in AXES
    }
    if max(parity.values()) > 1e-12:
        raise ValueError(f"v244 reconstruction mismatch: {parity}")

    results: dict[str, dict[str, Any]] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    selected_masks: dict[tuple[str, str], np.ndarray] = {}
    scope_names = list(scopes["full_2022"])
    for scope_name in scope_names:
        for weight in MIX_WEIGHTS:
            name = f"{scope_name}__w{weight:g}"
            results[name] = {}
            for axis in AXES:
                _baseline, candidate, selected_mask = compose_scoped(
                    parents[axis], base[axis], command[axis], routes[axis],
                    scopes[axis][scope_name], weight,
                )
                predictions[(name, axis)] = candidate
                selected_masks[(name, axis)] = selected_mask
                results[name][axis] = paired_metrics(
                    axes[axis], baselines[axis], candidate, selected_mask
                )

    selected = select_candidate(results)
    robustness = None
    locked_pass = False
    robust_pass = False
    if selected is not None:
        locked_pass = results[selected]["full_2024"]["gain"] > 0.0
        family = [
            predictions[(name, "full_2024")] for name in results
        ] + [baselines["full_2024"].copy()]
        robustness = _robustness(
            axes["full_2024"], baselines["full_2024"],
            predictions[(selected, "full_2024")],
            selected_masks[(selected, "full_2024")], family,
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
            **{f"candidate_{axis}": predictions[(selected, axis)] for axis in AXES},
            **{f"active_{axis}": selected_masks[(selected, axis)] for axis in AXES},
        )
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "local_gate_pass"
            if selected is not None and locked_pass and robust_pass
            else "locked_or_robust_reject"
            if selected is not None
            else "source_reject"
        ),
        "scope_count": len(scope_names),
        "mix_weights": list(MIX_WEIGHTS),
        "family_trial_count": len(results),
        "selected_candidate": selected,
        "selection_rule": "maximin source gain after sign/month/worst-month/support gates",
        "v244_reconstruction_max_abs": parity,
        "selected_results": None if selected is None else results[selected],
        "all_results": results,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": bool(selected is not None and locked_pass and robust_pass),
        "eligible_for_packaging": False,
        "restrictions": {
            "finite_scope_and_dose_family_preregistered_in_code": True,
            "selection_uses_full_2022_and_late_2023_only": True,
            "full_2024_not_used_for_selection": True,
            "all_76_candidates_in_local_reality_check": True,
            "lifetime_global_reality_check_unavailable": True,
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
    parser.add_argument("--command-xgb-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.base_xgb_oof_dir, args.command_xgb_oof_dir,
        args.exact_axes, args.v244_axes, args.contract_dir, args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "family_trial_count": result["family_trial_count"],
        "selected_candidate": result["selected_candidate"],
        "selected_results": result["selected_results"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
