"""Preregistered fixed-dose audit of four independent feature families.

The finite family is declared before v255/v257/v258 locked results are read:
four single models and all six equal-weight pairs.  Every candidate replaces a
fixed 25% of the fallback XGB inside the already frozen v244 routes.  Candidate
selection uses only full-2022 and late-2023; full-2024 is opened once for the
selected candidate and all ten candidates enter the local Reality Check.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context
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


PROTOCOL = "V259_INDEPENDENT_FEATURE_FAMILY_FIXED_DOSE_AUDIT_V1"
TOTAL_REPLACEMENT_WEIGHT = 0.25
MODEL_STEMS = {
    "command": "command_dispersion_xgb",
    "pitchmix": "pitchmix_release_xgb",
    "count": "count_trackman_xgb",
    "batter": "batter_trackman_xgb",
}
SUBSETS = tuple((name,) for name in MODEL_STEMS) + tuple(
    itertools.combinations(MODEL_STEMS, 2)
)


def blend_feature_models(
    base: np.ndarray,
    models: list[np.ndarray],
    total_weight: float = TOTAL_REPLACEMENT_WEIGHT,
) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    if not models:
        raise ValueError("at least one feature model is required")
    if not 0.0 <= float(total_weight) <= 1.0:
        raise ValueError("total_weight must be in [0, 1]")
    converted = [np.asarray(model, dtype=np.float64) for model in models]
    if any(model.shape != base.shape for model in converted):
        raise ValueError("feature model arrays have different shapes")
    feature_average = np.mean(np.stack(converted, axis=0), axis=0)
    return (1.0 - float(total_weight)) * base + float(total_weight) * feature_average


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
    candidate_dirs = {
        "command": command_oof_dir,
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
            axis: saved[f"candidate_runtime_faithful_exact_parent_{axis}"].astype(
                np.float64
            )
            for axis in AXES
        }
    base_full = {
        year: align_regular_prediction(
            frames[year],
            base_xgb_oof_dir / f"runtime_faithful_xgb_{year}.npy",
        )
        for year in (2022, 2023, 2024)
    }
    model_full = {
        name: {
            year: align_regular_prediction(
                frames[year], directory / f"{MODEL_STEMS[name]}_{year}.npy"
            )
            for year in (2022, 2023, 2024)
        }
        for name, directory in candidate_dirs.items()
    }
    base = {
        "full_2022": base_full[2022],
        "late_2023": base_full[2023][late23],
        "full_2024": base_full[2024],
    }
    models = {
        name: {
            "full_2022": predictions[2022],
            "late_2023": predictions[2023][late23],
            "full_2024": predictions[2024],
        }
        for name, predictions in model_full.items()
    }
    routes = {
        axis: route_masks(parents[axis], base[axis], axis_frames[axis])
        for axis in AXES
    }
    baselines: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    for axis in AXES:
        baselines[axis], active[axis] = compose(
            parents[axis], base[axis], routes[axis]
        )
        parity[axis] = float(np.max(np.abs(baselines[axis] - expected_v244[axis])))
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")

    predictions: dict[tuple[str, str], np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}
    for subset in SUBSETS:
        name = "+".join(subset)
        results[name] = {}
        for axis in AXES:
            mixed = blend_feature_models(
                base[axis], [models[item][axis] for item in subset]
            )
            candidate, candidate_active = compose(
                parents[axis], mixed, routes[axis]
            )
            if not np.array_equal(candidate_active, active[axis]):
                raise AssertionError("frozen v244 route support changed")
            predictions[(name, axis)] = candidate
            results[name][axis] = paired_metrics(
                axes[axis], baselines[axis], candidate, active[axis]
            )

    selected = select_candidate(results)
    locked_pass = False
    robust_pass = False
    robustness = None
    if selected is not None:
        locked_pass = results[selected]["full_2024"]["gain"] > 0.0
        family = [
            predictions[("+".join(subset), "full_2024")] for subset in SUBSETS
        ] + [baselines["full_2024"].copy()]
        robustness = _robustness(
            axes["full_2024"],
            baselines["full_2024"],
            predictions[(selected, "full_2024")],
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
            **{
                f"candidate_{axis}": predictions[(selected, axis)] for axis in AXES
            },
            **{f"active_{axis}": active[axis] for axis in AXES},
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
        "model_families": list(MODEL_STEMS),
        "candidate_subsets": [list(subset) for subset in SUBSETS],
        "family_trial_count": len(SUBSETS),
        "total_replacement_weight": TOTAL_REPLACEMENT_WEIGHT,
        "selected_candidate": selected,
        "selection_rule": (
            "maximize minimum source gain after source/month gates; "
            "full_2024 opened once"
        ),
        "v244_reconstruction_max_abs": parity,
        "selected_results": None if selected is None else results[selected],
        "all_results": results,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": bool(selected is not None and locked_pass and robust_pass),
        "eligible_for_packaging": False,
        "restrictions": {
            "four_feature_families_and_pair_set_preregistered_in_code": True,
            "one_fixed_total_replacement_weight": True,
            "v244_parent_routes_and_weights_frozen": True,
            "full_2024_not_used_for_family_selection": True,
            "full_2024_development_contaminated": True,
            "all_ten_candidates_in_local_reality_check": True,
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
