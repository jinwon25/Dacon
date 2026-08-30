"""Select a pseudo-deployment TrackMan expert on clean source axes.

Five structural experts are declared: pseudo base, command, batter, the equal
command/batter pair, and the equal three-model ensemble.  Each replaces the
pseudo component in the same v271 50% fallback formula and inherited
April--September calendar.  No blend-weight grid is searched.
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
from src.archive.v271_pseudo_deployment_calendar_audit import calendar_equal_fallback


PROTOCOL = "V275_PSEUDO_TRACKMAN_EXPERT_ROUTE_AUDIT_V1"
EXPERTS = ("base", "command", "batter", "command_batter", "all_equal")


def expert_predictions(
    base: np.ndarray, command: np.ndarray, batter: np.ndarray
) -> dict[str, np.ndarray]:
    base = np.asarray(base, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    batter = np.asarray(batter, dtype=np.float64)
    if not (base.shape == command.shape == batter.shape):
        raise ValueError("pseudo expert arrays have different shapes")
    return {
        "base": base,
        "command": command,
        "batter": batter,
        "command_batter": 0.5 * (command + batter),
        "all_equal": (base + command + batter) / 3.0,
    }


def select_expert(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, int, str]] = []
    for expert in EXPERTS:
        if not all(axis in results[expert] for axis in SOURCE_AXES):
            continue
        source = [results[expert][axis] for axis in SOURCE_AXES]
        if not all(item["gain"] > 0.0 for item in source):
            continue
        if not all(item["positive_month_fraction"] >= (2.0 / 3.0) for item in source):
            continue
        active_rows = min(int(item["active_rows"]) for item in source)
        if active_rows < 5000:
            continue
        gains = [float(item["gain"]) for item in source]
        eligible.append((min(gains), float(np.mean(gains)), active_rows, expert))
    return None if not eligible else max(eligible)[3]


def _available_axes(command_dir: Path, batter_dir: Path) -> tuple[str, ...]:
    mapping = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    return tuple(
        axis
        for axis, year in mapping.items()
        if (command_dir / f"pseudo_command_xgb_{year}.npy").exists()
        and (batter_dir / f"pseudo_batter_xgb_{year}.npy").exists()
    )


def run(
    train_csv: Path,
    base_xgb_oof_dir: Path,
    pseudo_base_oof_dir: Path,
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
        raise FileNotFoundError("no matching pseudo TrackMan expert folds")
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
            axis: saved[f"candidate_runtime_faithful_exact_jy_{axis}"].astype(np.float64)
            for axis in available
        }

    year_by_axis = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    parity: dict[str, float] = {}
    baselines: dict[str, np.ndarray] = {}
    candidates: dict[tuple[str, str], np.ndarray] = {}
    selected_masks: dict[tuple[str, str], np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {expert: {} for expert in EXPERTS}
    for axis in available:
        year = year_by_axis[axis]
        deployed_base = align_regular_prediction(
            frames[year],
            base_xgb_oof_dir / f"hyunku_runtime_faithful_xgb_{year}.npy",
        )
        pseudo_base = align_regular_prediction(
            frames[year], pseudo_base_oof_dir / f"pseudo_deployment_xgb_{year}.npy"
        )
        command = align_regular_prediction(
            frames[year], command_oof_dir / f"pseudo_command_xgb_{year}.npy"
        )
        batter = align_regular_prediction(
            frames[year], batter_oof_dir / f"pseudo_batter_xgb_{year}.npy"
        )
        if axis == "late_2023":
            deployed_base = deployed_base[late23]
            pseudo_base = pseudo_base[late23]
            command = command[late23]
            batter = batter[late23]
        routes = route_masks(parents[axis], deployed_base, axis_frames[axis])
        baselines[axis], active = compose(parents[axis], deployed_base, routes)
        parity[axis] = float(np.max(np.abs(baselines[axis] - expected_v244[axis])))
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")
        experts = expert_predictions(pseudo_base, command, batter)
        for expert in EXPERTS:
            fallback, selected = calendar_equal_fallback(
                deployed_base,
                experts[expert],
                active,
                axis_frames[axis]["game_month"].to_numpy(),
            )
            candidate, candidate_active = compose(parents[axis], fallback, routes)
            if not np.array_equal(candidate_active, active):
                raise AssertionError("v244 route support changed")
            candidates[(expert, axis)] = candidate
            selected_masks[(expert, axis)] = selected
            results[expert][axis] = paired_metrics(
                axes[axis], baselines[axis], candidate, selected
            )

    source_ready = all(axis in available for axis in SOURCE_AXES)
    selected_expert = select_expert(results) if source_ready else None
    locked_ready = "full_2024" in available
    locked_pass = bool(
        selected_expert is not None
        and locked_ready
        and results[selected_expert]["full_2024"]["gain"] > 0.0
    )
    robustness = None
    robust_pass = False
    if selected_expert is not None and locked_ready:
        robustness = _robustness(
            axes["full_2024"],
            baselines["full_2024"],
            candidates[(selected_expert, "full_2024")],
            selected_masks[(selected_expert, "full_2024")],
            [candidates[(expert, "full_2024")] for expert in EXPERTS]
            + [baselines["full_2024"].copy()],
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] < 0.05
        )
    if selected_expert is not None:
        np.savez_compressed(
            output_dir / "selected_candidate_axes.npz",
            **{f"parent_{axis}": baselines[axis] for axis in available},
            **{
                f"candidate_{axis}": candidates[(selected_expert, axis)]
                for axis in available
            },
            **{
                f"active_{axis}": selected_masks[(selected_expert, axis)]
                for axis in available
            },
        )
    status = (
        "incomplete_source"
        if not source_ready
        else "source_reject"
        if selected_expert is None
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
        "experts": list(EXPERTS),
        "selected_expert": selected_expert,
        "selection_rule": "source positivity and monthly majority, then maximin gain",
        "v244_reconstruction_max_abs": parity,
        "results": results,
        "source_gate_passed": selected_expert is not None,
        "locked_point_ready": locked_ready,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": bool(locked_pass and robust_pass),
        "restrictions": {
            "five_structural_experts_preregistered": True,
            "all_ensembles_use_equal_weights": True,
            "v271_route_weight_and_calendar_frozen": True,
            "full_2024_not_used_for_expert_selection": True,
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
    parser.add_argument("--pseudo-base-oof-dir", type=Path, required=True)
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
        args.pseudo_base_oof_dir,
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
