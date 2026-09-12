"""Add v199 workload features to the proven exact H1 CatBoost recipe.

This is a paired single-seed screen.  The baseline prediction is the frozen
seed-42 forward OOF from v157; the augmented twin uses the same H1 features,
hyperparameters, training rows, and seed plus row-local workload features.
Only the paired H1 delta is applied above the exact row-region parent.  A passing result
must still be confirmed with the full three-seed ensemble before packaging.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import (
    H1_ACTIVE_WEIGHT,
    H1_BASE_WEIGHT,
    _load_year_context,
    affine,
    metrics,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v201_paired_workload_booster import workload_feature_frame
from src.champion.v131_catboost_h1_independent_oof import _fit_year, _prepare_features
from src.core.contract import _load_contract_axis


PROTOCOL = "V203_H1_WORKLOAD_FEATURE_SCREEN_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
AUDIT_YEARS = (2022, 2023, 2024)
SCALES = (0.25, 0.5, 1.0)
MODEL_CONFIG = {
    "iterations": 1200,
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 100.0,
    "border_count": 32,
    "seed": 42,
}


def h1_top_delta(
    baseline_raw: np.ndarray,
    augmented_raw: np.ndarray,
    correction: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    baseline = affine(np.asarray(baseline_raw, dtype=np.float64) + correction)
    augmented = affine(np.asarray(augmented_raw, dtype=np.float64) + correction)
    champion_gate = (
        pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(frame["li"], errors="coerce").fillna(1.0).ge(1.5)
    ).to_numpy()
    weights = np.where(champion_gate, H1_ACTIVE_WEIGHT, H1_BASE_WEIGHT)
    return weights * (augmented - baseline)


def apply_delta(
    parent: np.ndarray,
    direction: np.ndarray,
    exact: np.ndarray,
    domain3: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = (
        np.asarray(exact, dtype=bool)
        & np.asarray(domain3).astype(str).__eq__("R_CORE")
        & np.not_equal(np.asarray(direction, dtype=np.float64), 0.0)
    )
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(scale) * np.asarray(direction)[active],
        0.001,
        0.999,
    )
    return output, active


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.5
        and result["worst_month_gain"] > -3.0
        and result["minimum_domain_gain"] >= 0.0
    )


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "strictly_prior_season_fits": True,
        "paired_same_h1_recipe_and_seed": True,
        "source_scale_selection_before_locked_2024": True,
        "three_seed_confirmation_required": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    component_root: Path,
    baseline_checkpoint_dir: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, component_root
    )
    workload = workload_feature_frame(train)
    augmented_features = list(h1_features) + list(workload.columns)
    for column in workload:
        if column in train:
            raise ValueError(f"workload feature collision: {column}")
        train[column] = workload[column].to_numpy(np.float32)
    del workload
    gc.collect()

    augmented: dict[int, np.ndarray] = {}
    baseline: dict[int, np.ndarray] = {}
    for year in AUDIT_YEARS:
        baseline_path = baseline_checkpoint_dir / f"h1_year{year}_seed42.npy"
        baseline[year] = np.load(baseline_path, allow_pickle=False).astype(np.float64)
        checkpoint = output_dir / f"augmented_h1_year{year}_seed42.npy"
        if checkpoint.exists():
            augmented[year] = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            if len(augmented[year]) != int(np.sum(season == year)):
                raise ValueError(f"invalid augmented checkpoint: {year}")
            print(f"[v203] resumed augmented fold {year}", flush=True)
        else:
            print(f"[v203] fitting augmented exact-H1 fold {year}", flush=True)
            augmented[year] = _fit_year(
                train, target, season, year, augmented_features, MODEL_CONFIG,
                f"workload-H1-seed42",
            )
            np.save(checkpoint, augmented[year], allow_pickle=False)
        if baseline[year].shape != augmented[year].shape:
            raise ValueError(f"paired fold shape mismatch: {year}")
    del train
    gc.collect()

    _context, raw_frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, raw_frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    full_directions = {
        year: h1_top_delta(
            baseline[year], augmented[year], correction[year], raw_frames[year]
        )
        for year in AUDIT_YEARS
    }
    directions = {
        "full_2022": full_directions[2022],
        "late_2023": full_directions[2023][late23],
        "full_2024": full_directions[2024],
    }

    rows: list[dict[str, Any]] = []
    detail: dict[str, dict[str, Any]] = {}
    candidates: dict[float, dict[str, np.ndarray]] = {}
    active_masks: dict[float, dict[str, np.ndarray]] = {}
    for scale in SCALES:
        detail[str(scale)], candidates[scale], active_masks[scale] = {}, {}, {}
        row: dict[str, Any] = {"scale": scale}
        for axis in AXES:
            candidate, active = apply_delta(
                parents[axis], directions[axis], axes[axis]["exact_mask"],
                axes[axis]["domain3"], scale,
            )
            candidates[scale][axis], active_masks[scale][axis] = candidate, active
            score = metrics(axes[axis], parents[axis], candidate)
            detail[str(scale)][axis] = score
            row[f"{axis}_gain"] = score["gain"]
            row[f"{axis}_month_fraction"] = score["positive_month_fraction"]
            row[f"{axis}_worst_month"] = score["worst_month_gain"]
        row["source_gate_passed"] = all(
            source_gate(detail[str(scale)][axis]) for axis in SOURCE_AXES
        )
        row["source_min_gain"] = min(
            detail[str(scale)][axis]["gain"] for axis in SOURCE_AXES
        )
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain"],
        ascending=[False, False], kind="stable",
    )
    table.to_csv(output_dir / "source_scale_screen.csv", index=False, encoding="utf-8-sig")
    passing = table.loc[table["source_gate_passed"]]
    paired_diagnostics = {
        str(year): {
            "raw_delta_mean_abs": float(np.mean(np.abs(augmented[year] - baseline[year]))),
            "raw_delta_p99_abs": float(np.quantile(np.abs(augmented[year] - baseline[year]), 0.99)),
            "raw_correlation": float(np.corrcoef(augmented[year], baseline[year])[0, 1]),
        }
        for year in AUDIT_YEARS
    }
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "model_config": MODEL_CONFIG,
            "paired_diagnostics": paired_diagnostics,
            "screen": table.to_dict(orient="records"),
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected_scale = float(passing.iloc[0]["scale"])
        candidate = candidates[selected_scale]["full_2024"]
        active = active_masks[selected_scale]["full_2024"]
        family = [
            candidates[float(scale)]["full_2024"] for scale in passing["scale"]
        ]
        family.append(parents["full_2024"].copy())
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], candidate, active, family
        )
        locked = detail[str(selected_scale)]["full_2024"]
        point_pass = bool(
            locked["gain"] > 0.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -3.0
            and locked["minimum_domain_gain"] >= 0.0
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"], candidate=candidate,
            direction=directions["full_2024"], active=active,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "confirm_three_seed" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "model_config": MODEL_CONFIG,
            "paired_diagnostics": paired_diagnostics,
            "selected_scale": selected_scale,
            "source": {
                axis: detail[str(selected_scale)][axis] for axis in SOURCE_AXES
            },
            "locked_2024": locked,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_three_seed_confirmation": bool(point_pass and robust_pass),
            "eligible_for_packaging": False,
            "screen": table.to_dict(orient="records"),
            "parity": parity,
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
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.trackman_csv, args.component_root,
        args.baseline_checkpoint_dir, args.contract_dir, args.v104_path,
        args.h1_path, args.c3_path, args.v160_path, args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
