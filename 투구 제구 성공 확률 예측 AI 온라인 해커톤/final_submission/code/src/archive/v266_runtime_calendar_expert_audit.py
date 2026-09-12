"""Audit the frozen v261 formula with runtime-faithful expert predictions."""

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
from src.archive.v254_source_stable_command_scope_audit import scope_masks


PROTOCOL = "V266_RUNTIME_CALENDAR_EXPERT_AUDIT_V1"
COMMAND_WEIGHT = 0.60
BATTER_WEIGHT = 0.25
FIRST_ACTIVE_MONTH = 4
LAST_ACTIVE_MONTH = 9


def compose_runtime_calendar_experts(
    parent: np.ndarray,
    base: np.ndarray,
    command: np.ndarray,
    batter: np.ndarray,
    routes: dict[str, np.ndarray],
    month: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply command first, batter on its complement, only in April--September."""

    parent = np.asarray(parent, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    batter = np.asarray(batter, dtype=np.float64)
    month = np.asarray(month)
    if not (
        parent.shape == base.shape == command.shape == batter.shape == month.shape
    ):
        raise ValueError("calendar expert arrays have different shapes")
    baseline, active = compose(parent, base, routes)
    command_scope = scope_masks(parent, base, command, routes)[
        "command_toward_center"
    ] & active
    complement_scope = active & ~command_scope
    calendar = (month >= FIRST_ACTIVE_MONTH) & (month <= LAST_ACTIVE_MONTH)
    command_selected = command_scope & calendar
    batter_selected = complement_scope & calendar
    fallback = base.copy()
    fallback[command_selected] = (
        (1.0 - COMMAND_WEIGHT) * base[command_selected]
        + COMMAND_WEIGHT * command[command_selected]
    )
    fallback[batter_selected] = (
        (1.0 - BATTER_WEIGHT) * base[batter_selected]
        + BATTER_WEIGHT * batter[batter_selected]
    )
    candidate, candidate_active = compose(parent, fallback, routes)
    if not np.array_equal(candidate_active, active):
        raise AssertionError("v244 route support changed")
    selected = command_selected | batter_selected
    return baseline, candidate, selected, command_selected


def _available_axes(command_dir: Path, batter_dir: Path) -> tuple[str, ...]:
    mapping = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    return tuple(
        axis
        for axis, year in mapping.items()
        if (command_dir / f"command_dispersion_xgb_{year}.npy").exists()
        and (batter_dir / f"batter_trackman_xgb_{year}.npy").exists()
    )


def run(
    train_csv: Path,
    base_xgb_oof_dir: Path,
    command_oof_dir: Path,
    batter_oof_dir: Path,
    exact_axes: Path,
    v244_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    available = _available_axes(command_oof_dir, batter_oof_dir)
    if not available:
        raise FileNotFoundError("no matching command/batter runtime OOF folds")

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
    command: dict[str, np.ndarray] = {}
    batter: dict[str, np.ndarray] = {}
    for axis in available:
        year = axis_year[axis]
        base_full = align_regular_prediction(
            frames[year],
            base_xgb_oof_dir / f"runtime_faithful_xgb_{year}.npy",
        )
        command_full = align_regular_prediction(
            frames[year], command_oof_dir / f"command_dispersion_xgb_{year}.npy"
        )
        batter_full = align_regular_prediction(
            frames[year], batter_oof_dir / f"batter_trackman_xgb_{year}.npy"
        )
        if axis == "late_2023":
            base_full = base_full[late23]
            command_full = command_full[late23]
            batter_full = batter_full[late23]
        base[axis] = base_full
        command[axis] = command_full
        batter[axis] = batter_full

    results: dict[str, Any] = {}
    parity: dict[str, float] = {}
    baselines: dict[str, np.ndarray] = {}
    candidates: dict[str, np.ndarray] = {}
    selected: dict[str, np.ndarray] = {}
    command_selected: dict[str, np.ndarray] = {}
    for axis in available:
        routes = route_masks(parents[axis], base[axis], axis_frames[axis])
        (
            baselines[axis],
            candidates[axis],
            selected[axis],
            command_selected[axis],
        ) = compose_runtime_calendar_experts(
            parents[axis],
            base[axis],
            command[axis],
            batter[axis],
            routes,
            axis_frames[axis]["game_month"].to_numpy(),
        )
        parity[axis] = float(
            np.max(np.abs(baselines[axis] - expected_v244[axis]))
        )
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")
        results[axis] = paired_metrics(
            axes[axis], baselines[axis], candidates[axis], selected[axis]
        )
        results[axis]["command_rows"] = int(command_selected[axis].sum())
        results[axis]["batter_rows"] = int(
            (selected[axis] & ~command_selected[axis]).sum()
        )

    source_ready = all(axis in available for axis in SOURCE_AXES)
    source_pass = source_ready and all(
        results[axis]["gain"] > 0.0 for axis in SOURCE_AXES
    )
    locked_ready = "full_2024" in available
    locked_pass = locked_ready and results["full_2024"]["gain"] > 0.0
    robustness = None
    robust_pass = False
    if locked_ready:
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
        "source_reject"
        if source_ready and not source_pass
        else "source_pass_pending_locked"
        if source_pass and not locked_ready
        else "local_gate_pass"
        if source_pass and locked_pass and robust_pass
        else "locked_or_robust_reject"
        if source_pass and locked_ready
        else "incomplete_source"
    )
    summary = {
        "protocol": PROTOCOL,
        "status": status,
        "available_axes": list(available),
        "command_weight": COMMAND_WEIGHT,
        "batter_weight": BATTER_WEIGHT,
        "calendar_window": [FIRST_ACTIVE_MONTH, LAST_ACTIVE_MONTH],
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
            "v261_formula_frozen_before_runtime_reaudit": True,
            "v244_parent_routes_and_weights_frozen": True,
            "all_expert_audit_base_features_runtime_faithful": True,
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
    parser.add_argument("--command-oof-dir", type=Path, required=True)
    parser.add_argument("--batter-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.base_xgb_oof_dir,
        args.command_oof_dir,
        args.batter_oof_dir,
        args.exact_axes,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
