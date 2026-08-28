"""Strict full-year audit of the paired workload-augmented H1 model.

Unlike the legacy full-pipeline axes, every primary prediction here comes from
a model fit only on seasons preceding the audit season.  Full-2022 and
full-2023 select a conservative paired dose; full-2024 is then opened once.
The development-contaminated exact JY axes are retained only as a secondary
veto and can never promote the candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context, affine, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v203_h1_workload_feature_screen import apply_delta, h1_top_delta
from src.core.contract import _load_contract_axis


PROTOCOL = "V205_H1_WORKLOAD_STRICT_FORWARD_AUDIT_V1"
STRICT_AXES = ("full_2022", "full_2023", "full_2024")
SOURCE_AXES = ("full_2022", "full_2023")
SCALES = (0.25, 0.5, 1.0)


def strict_axis(
    frame: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
) -> dict[str, np.ndarray]:
    n_rows = len(frame)
    if len(target) != n_rows or len(parent) != n_rows:
        raise ValueError("strict axis arrays are not aligned")
    return {
        "target": np.asarray(target, dtype=np.float64),
        "parent": np.asarray(parent, dtype=np.float64),
        "exact_mask": np.ones(n_rows, dtype=bool),
        "domain3": np.full(n_rows, "R_CORE", dtype="U6"),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(np.int64),
        "batter_id": frame["batter_id"].to_numpy(np.int64),
    }


def apply_component_delta(
    parent: np.ndarray,
    augmented: np.ndarray,
    scale: float,
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    return np.clip(
        parent + float(scale) * (np.asarray(augmented, dtype=np.float64) - parent),
        0.001,
        0.999,
    )


def primary_source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.60
        and result["worst_month_gain"] > -5.0
    )


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "primary_axes_strictly_forward_full_year": True,
        "source_scale_selection_before_full_2024": True,
        "contaminated_pipeline_axes_can_only_veto": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def run(
    train_csv: Path,
    augmented_dir: Path,
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
    context, raw_frames, correction = _load_year_context(train_csv)
    season = context["season"].to_numpy(np.int16)
    target_all = context["control_success"].to_numpy(np.float64)
    years = (2022, 2023, 2024)
    baseline_raw = {
        year: np.load(
            baseline_checkpoint_dir / f"h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64)
        for year in years
    }
    augmented_raw = {
        year: np.load(
            augmented_dir / f"augmented_h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64)
        for year in years
    }
    baseline = {
        year: affine(baseline_raw[year] + correction[year]) for year in years
    }
    augmented = {
        year: affine(augmented_raw[year] + correction[year]) for year in years
    }
    strict = {
        f"full_{year}": strict_axis(
            raw_frames[year], target_all[season == year], baseline[year]
        )
        for year in years
    }
    for year in years:
        if baseline[year].shape != augmented[year].shape:
            raise ValueError(f"paired checkpoint mismatch: {year}")

    details: dict[str, dict[str, Any]] = {}
    candidates: dict[float, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        details[str(scale)], candidates[scale] = {}, {}
        row: dict[str, Any] = {"scale": scale}
        for axis_name, year in zip(STRICT_AXES, years):
            candidate = apply_component_delta(baseline[year], augmented[year], scale)
            candidates[scale][axis_name] = candidate
            score = metrics(strict[axis_name], baseline[year], candidate)
            details[str(scale)][axis_name] = score
            row[f"{axis_name}_gain"] = score["gain"]
            row[f"{axis_name}_month_fraction"] = score["positive_month_fraction"]
            row[f"{axis_name}_worst_month"] = score["worst_month_gain"]
        row["source_gate_passed"] = all(
            primary_source_gate(details[str(scale)][axis]) for axis in SOURCE_AXES
        )
        row["source_min_gain"] = min(
            details[str(scale)][axis]["gain"] for axis in SOURCE_AXES
        )
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain"],
        ascending=[False, False], kind="stable",
    )
    table.to_csv(output_dir / "strict_source_scale_screen.csv", index=False, encoding="utf-8-sig")
    passing = table.loc[table["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "strict_source_reject",
            "strict_screen": table.to_dict(orient="records"),
            "eligible_for_packaging": False,
            "restrictions": restrictions(),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    selected_scale = float(passing.iloc[0]["scale"])
    locked_candidate = candidates[selected_scale]["full_2024"]
    locked = details[str(selected_scale)]["full_2024"]
    family = [
        candidates[float(scale)]["full_2024"] for scale in passing["scale"]
    ]
    family.append(baseline[2024].copy())
    robust = _robustness(
        strict["full_2024"], baseline[2024], locked_candidate,
        np.ones(len(baseline[2024]), dtype=bool), family,
    )
    locked_point_pass = bool(
        primary_source_gate(locked)
        and locked["positive_month_fraction"] >= 0.625
    )
    robust_pass = bool(
        robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] <= 0.10
    )

    # Secondary full-pipeline view.  These axes have high fidelity but are
    # development-contaminated, so they may veto an obvious regression only.
    pipeline_axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_jy_parents(
        pipeline_axes, raw_frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    full_top_delta = {
        year: h1_top_delta(
            baseline_raw[year], augmented_raw[year], correction[year], raw_frames[year]
        )
        for year in years
    }
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    pipeline_direction = {
        "full_2022": full_top_delta[2022],
        "late_2023": full_top_delta[2023][late23],
        "full_2024": full_top_delta[2024],
    }
    pipeline_metrics: dict[str, Any] = {}
    pipeline_candidates: dict[str, np.ndarray] = {}
    for axis_name in pipeline_axes:
        candidate, _ = apply_delta(
            parents[axis_name], pipeline_direction[axis_name],
            pipeline_axes[axis_name]["exact_mask"],
            pipeline_axes[axis_name]["domain3"], selected_scale,
        )
        pipeline_candidates[axis_name] = candidate
        pipeline_metrics[axis_name] = metrics(
            pipeline_axes[axis_name], parents[axis_name], candidate
        )
    pipeline_veto = bool(
        pipeline_metrics["full_2024"]["gain"] < -0.5
        or pipeline_metrics["full_2024"]["minimum_domain_gain"] < -0.75
    )
    confirm = bool(locked_point_pass and robust_pass and not pipeline_veto)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        strict_full_2022=candidates[selected_scale]["full_2022"],
        strict_full_2023=candidates[selected_scale]["full_2023"],
        strict_full_2024=locked_candidate,
        pipeline_full_2024=pipeline_candidates["full_2024"],
        pipeline_parent_full_2024=parents["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirm_three_seed" if confirm else (
            "pipeline_veto" if pipeline_veto else (
                "strict_point_pass_robust_reject" if locked_point_pass else "strict_locked_reject"
            )
        ),
        "selected_scale": selected_scale,
        "strict_source": {
            axis: details[str(selected_scale)][axis] for axis in SOURCE_AXES
        },
        "strict_locked_2024": locked,
        "strict_robustness_2024": robust,
        "strict_locked_point_passed": locked_point_pass,
        "strict_robust_gate_passed": robust_pass,
        "secondary_contaminated_pipeline_metrics": pipeline_metrics,
        "secondary_pipeline_veto": pipeline_veto,
        "eligible_for_three_seed_confirmation": confirm,
        "eligible_for_packaging": False,
        "strict_screen": table.to_dict(orient="records"),
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
    parser.add_argument("--augmented-dir", type=Path, required=True)
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
        args.train_csv, args.augmented_dir, args.baseline_checkpoint_dir,
        args.contract_dir, args.v104_path, args.h1_path, args.c3_path,
        args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
