"""Audit fixed structural policies for the pseudo-deployment XGB.

The three policies are qualitative routing choices, not a blend-weight grid:
an equal ensemble on every v244 route, the same ensemble only when the new
model is less extreme, and the ensemble only when both models lie on the same
side of 0.5.  One policy is selected on 2022 and late-2023; 2024 stays locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import align_regular_prediction
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics, route_masks
from src.archive.v248_fixed_route_dual_tree_fallback import ROUTE_WEIGHTS, compose
from src.archive.v253_fixed_route_command_dispersion_audit import AXES, SOURCE_AXES, _load_axes


PROTOCOL = "V268_PSEUDO_DEPLOYMENT_ROUTE_AUDIT_V1"
POLICIES = ("equal_all", "toward_center", "same_center_side")
EQUAL_WEIGHT = 0.50


def policy_fallback(
    base: np.ndarray,
    pseudo: np.ndarray,
    active: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base, dtype=np.float64)
    pseudo = np.asarray(pseudo, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if not (base.shape == pseudo.shape == active.shape):
        raise ValueError("policy arrays have different shapes")
    if policy == "equal_all":
        selected = active.copy()
    elif policy == "toward_center":
        selected = active & (np.abs(pseudo - 0.5) < np.abs(base - 0.5))
    elif policy == "same_center_side":
        selected = active & ((base - 0.5) * (pseudo - 0.5) >= 0.0)
    else:
        raise ValueError(f"unknown policy: {policy}")
    fallback = base.copy()
    fallback[selected] = (
        (1.0 - EQUAL_WEIGHT) * base[selected] + EQUAL_WEIGHT * pseudo[selected]
    )
    return fallback, selected


def select_policy(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, int, str]] = []
    for policy in POLICIES:
        if not all(axis in results[policy] for axis in SOURCE_AXES):
            continue
        source = [results[policy][axis] for axis in SOURCE_AXES]
        if not all(item["gain"] > 0.0 for item in source):
            continue
        if not all(item["positive_month_fraction"] >= (2.0 / 3.0) for item in source):
            continue
        if not all(item["worst_month_gain"] > -5.0 for item in source):
            continue
        active_rows = min(int(item["active_rows"]) for item in source)
        if active_rows < 5000:
            continue
        gains = [float(item["gain"]) for item in source]
        eligible.append((min(gains), float(np.mean(gains)), active_rows, policy))
    return None if not eligible else max(eligible)[3]


def _available_axes(pseudo_dir: Path) -> tuple[str, ...]:
    mapping = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    return tuple(
        axis
        for axis, year in mapping.items()
        if (pseudo_dir / f"pseudo_deployment_xgb_{year}.npy").exists()
    )


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
    available = _available_axes(pseudo_oof_dir)
    if not available:
        raise FileNotFoundError("no pseudo-deployment OOF folds")

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
            axis: saved[f"candidate_runtime_faithful_exact_parent_{axis}"].astype(
                np.float64
            )
            for axis in available
        }

    axis_year = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    base: dict[str, np.ndarray] = {}
    pseudo: dict[str, np.ndarray] = {}
    for axis in available:
        year = axis_year[axis]
        base_full = align_regular_prediction(
            frames[year],
            base_xgb_oof_dir / f"runtime_faithful_xgb_{year}.npy",
        )
        pseudo_full = align_regular_prediction(
            frames[year], pseudo_oof_dir / f"pseudo_deployment_xgb_{year}.npy"
        )
        if axis == "late_2023":
            base_full = base_full[late23]
            pseudo_full = pseudo_full[late23]
        base[axis] = base_full
        pseudo[axis] = pseudo_full

    parity: dict[str, float] = {}
    baselines: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    routes: dict[str, dict[str, np.ndarray]] = {}
    for axis in available:
        routes[axis] = route_masks(parents[axis], base[axis], axis_frames[axis])
        baselines[axis], active[axis] = compose(parents[axis], base[axis], routes[axis])
        parity[axis] = float(
            np.max(np.abs(baselines[axis] - expected_v244[axis]))
        )
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")

    results: dict[str, dict[str, Any]] = {policy: {} for policy in POLICIES}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    selected_masks: dict[tuple[str, str], np.ndarray] = {}
    for policy in POLICIES:
        for axis in available:
            fallback, selected = policy_fallback(
                base[axis], pseudo[axis], active[axis], policy
            )
            candidate, candidate_active = compose(
                parents[axis], fallback, routes[axis]
            )
            if not np.array_equal(candidate_active, active[axis]):
                raise AssertionError("v244 route support changed")
            selected &= active[axis]
            results[policy][axis] = paired_metrics(
                axes[axis], baselines[axis], candidate, selected
            )
            predictions[(policy, axis)] = candidate
            selected_masks[(policy, axis)] = selected

    source_ready = all(axis in available for axis in SOURCE_AXES)
    selected_policy = select_policy(results) if source_ready else None
    locked_ready = "full_2024" in available
    locked_pass = bool(
        selected_policy is not None
        and locked_ready
        and results[selected_policy]["full_2024"]["gain"] > 0.0
    )
    robustness = None
    robust_pass = False
    if selected_policy is not None and locked_ready:
        family = [predictions[(policy, "full_2024")] for policy in POLICIES]
        robustness = _robustness(
            axes["full_2024"],
            baselines["full_2024"],
            predictions[(selected_policy, "full_2024")],
            selected_masks[(selected_policy, "full_2024")],
            family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] < 0.05
        )

    if selected_policy is not None:
        np.savez_compressed(
            output_dir / "selected_candidate_axes.npz",
            **{f"parent_{axis}": baselines[axis] for axis in available},
            **{
                f"candidate_{axis}": predictions[(selected_policy, axis)]
                for axis in available
            },
            **{
                f"active_{axis}": selected_masks[(selected_policy, axis)]
                for axis in available
            },
        )
    status = (
        "incomplete_source"
        if not source_ready
        else "source_reject"
        if selected_policy is None
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
        "policies": list(POLICIES),
        "equal_weight": EQUAL_WEIGHT,
        "selection_rule": (
            "source positivity and stability gates, then maximin 2022/late-2023 gain"
        ),
        "selected_policy": selected_policy,
        "v244_reconstruction_max_abs": parity,
        "results": results,
        "source_gate_ready": source_ready,
        "source_gate_passed": selected_policy is not None,
        "locked_point_ready": locked_ready,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": bool(locked_pass and robust_pass),
        "restrictions": {
            "three_structural_policies_preregistered": True,
            "single_equal_weight_not_a_weight_grid": True,
            "v244_parent_routes_and_weights_frozen": True,
            "full_2024_not_used_for_policy_selection": True,
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
