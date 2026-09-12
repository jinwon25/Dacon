"""Audit a command specialist plus independent complement experts.

V254 selected one source-stable specialist before this experiment: use 60% of
the command-dispersion model only where it moves probability toward 0.5.  This
experiment preserves that specialist exactly and lets one fixed-dose feature
expert handle only the remaining v244-active rows.  The declared complement
family contains three singles, their three equal pairs, and the command-only
reference.  Selection uses full-2022 and late-2023 only.
"""

from __future__ import annotations

import argparse
import itertools
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
from src.archive.v254_source_stable_command_scope_audit import scope_masks


PROTOCOL = "V260_MECHANISM_COMPLEMENT_EXPERTS_V1"
COMMAND_WEIGHT = 0.60
COMPLEMENT_WEIGHT = 0.25
COMPLEMENT_STEMS = {
    "pitchmix": "pitchmix_release_xgb",
    "count": "count_trackman_xgb",
    "batter": "batter_trackman_xgb",
}
COMPLEMENT_SUBSETS = ((),) + tuple((name,) for name in COMPLEMENT_STEMS) + tuple(
    itertools.combinations(COMPLEMENT_STEMS, 2)
)


def compose_experts(
    parent: np.ndarray,
    base: np.ndarray,
    command: np.ndarray,
    complements: list[np.ndarray],
    routes: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compose disjoint command and complement experts in frozen v244 routes."""

    parent = np.asarray(parent, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    if parent.shape != base.shape or base.shape != command.shape:
        raise ValueError("parent/base/command arrays have different shapes")
    converted = [np.asarray(value, dtype=np.float64) for value in complements]
    if any(value.shape != base.shape for value in converted):
        raise ValueError("complement arrays have different shapes")
    baseline, active = compose(parent, base, routes)
    command_scope = scope_masks(parent, base, command, routes)[
        "command_toward_center"
    ]
    complement_scope = active & ~command_scope
    fallback = base.copy()
    fallback[command_scope] = (
        (1.0 - COMMAND_WEIGHT) * base[command_scope]
        + COMMAND_WEIGHT * command[command_scope]
    )
    selected = command_scope.copy()
    if converted:
        average = np.mean(np.stack(converted, axis=0), axis=0)
        fallback[complement_scope] = (
            (1.0 - COMPLEMENT_WEIGHT) * base[complement_scope]
            + COMPLEMENT_WEIGHT * average[complement_scope]
        )
        selected |= complement_scope
    candidate, candidate_active = compose(parent, fallback, routes)
    if not np.array_equal(candidate_active, active):
        raise AssertionError("frozen v244 route support changed")
    return baseline, candidate, selected & active, command_scope


def select_candidate(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, int, str]] = []
    for name, axes in results.items():
        source = [axes[axis] for axis in SOURCE_AXES]
        if not all(item["gain"] > 0.0 for item in source):
            continue
        if not all(item["positive_month_fraction"] >= (2.0 / 3.0) for item in source):
            continue
        if not all(item["worst_month_gain"] > -5.0 for item in source):
            continue
        gains = [float(item["gain"]) for item in source]
        eligible.append((min(gains), float(np.mean(gains)), -len(name), name))
    return None if not eligible else max(eligible)[3]


def run(
    train_csv: Path,
    base_xgb_oof_dir: Path,
    command_oof_dir: Path,
    pitchmix_oof_dir: Path,
    count_oof_dir: Path,
    batter_oof_dir: Path,
    exact_axes: Path,
    v244_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    complement_dirs = {
        "pitchmix": pitchmix_oof_dir,
        "count": count_oof_dir,
        "batter": batter_oof_dir,
    }
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

    def load(stem: str, directory: Path) -> dict[int, np.ndarray]:
        return {
            year: align_regular_prediction(
                frames[year], directory / f"{stem}_{year}.npy"
            )
            for year in (2022, 2023, 2024)
        }

    base_full = load("hyunku_runtime_faithful_xgb", base_xgb_oof_dir)
    command_full = load("command_dispersion_xgb", command_oof_dir)
    complement_full = {
        name: load(COMPLEMENT_STEMS[name], directory)
        for name, directory in complement_dirs.items()
    }

    def axes_from(full: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "full_2022": full[2022],
            "late_2023": full[2023][late23],
            "full_2024": full[2024],
        }

    base = axes_from(base_full)
    command = axes_from(command_full)
    complements = {
        name: axes_from(full) for name, full in complement_full.items()
    }
    routes = {
        axis: route_masks(parents[axis], base[axis], axis_frames[axis])
        for axis in AXES
    }

    baselines: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    selected_masks: dict[tuple[str, str], np.ndarray] = {}
    command_rows: dict[str, int] = {}
    results: dict[str, dict[str, Any]] = {}
    for subset in COMPLEMENT_SUBSETS:
        name = "command_only" if not subset else "command+" + "+".join(subset)
        results[name] = {}
        for axis in AXES:
            baseline, candidate, selected_mask, command_scope = compose_experts(
                parents[axis],
                base[axis],
                command[axis],
                [complements[item][axis] for item in subset],
                routes[axis],
            )
            if axis not in baselines:
                baselines[axis] = baseline
                parity[axis] = float(
                    np.max(np.abs(baseline - expected_v244[axis]))
                )
                if parity[axis] > 1e-12:
                    raise ValueError(
                        f"v244 reconstruction mismatch on {axis}: {parity[axis]}"
                    )
                command_rows[axis] = int(command_scope.sum())
            predictions[(name, axis)] = candidate
            selected_masks[(name, axis)] = selected_mask
            results[name][axis] = paired_metrics(
                axes[axis], baseline, candidate, selected_mask
            )

    selected = select_candidate(results)
    robustness = None
    locked_pass = False
    robust_pass = False
    if selected is not None:
        locked_pass = results[selected]["full_2024"]["gain"] > 0.0
        family = [
            predictions[(
                "command_only" if not subset else "command+" + "+".join(subset),
                "full_2024",
            )]
            for subset in COMPLEMENT_SUBSETS
        ] + [baselines["full_2024"].copy()]
        robustness = _robustness(
            axes["full_2024"],
            baselines["full_2024"],
            predictions[(selected, "full_2024")],
            selected_masks[(selected, "full_2024")],
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
                f"candidate_{axis}": predictions[(selected, axis)] for axis in AXES
            },
            **{
                f"active_{axis}": selected_masks[(selected, axis)] for axis in AXES
            },
        )
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "local_gate_pass"
            if selected is not None and locked_pass and robust_pass
            else "source_reject"
            if selected is None
            else "locked_or_robust_reject"
        ),
        "command_weight": COMMAND_WEIGHT,
        "complement_weight": COMPLEMENT_WEIGHT,
        "candidate_subsets": [list(value) for value in COMPLEMENT_SUBSETS],
        "family_trial_count": len(COMPLEMENT_SUBSETS),
        "selected_candidate": selected,
        "selection_rule": (
            "preserve v254 command specialist; maximize minimum source gain "
            "after source/month gates"
        ),
        "command_specialist_rows": command_rows,
        "v244_reconstruction_max_abs": parity,
        "selected_results": None if selected is None else results[selected],
        "all_results": results,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": bool(selected is not None and locked_pass and robust_pass),
        "eligible_for_packaging": False,
        "restrictions": {
            "v254_command_specialist_frozen": True,
            "command_and_complement_scopes_disjoint": True,
            "one_fixed_complement_weight": True,
            "full_2024_not_used_for_candidate_selection": True,
            "full_2024_development_contaminated": True,
            "all_declared_candidates_in_local_reality_check": True,
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
    parser.add_argument("--pitchmix-oof-dir", type=Path, required=True)
    parser.add_argument("--count-oof-dir", type=Path, required=True)
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
        args.pitchmix_oof_dir,
        args.count_oof_dir,
        args.batter_oof_dir,
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
                "selected_candidate": result["selected_candidate"],
                "selected_results": result["selected_results"],
                "robustness": result["robustness"],
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
