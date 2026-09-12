"""Confirm the preregistered workload-H1 candidate with three paired seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_row_region_exact_contract_reaudit import (
    BRIDGE_SCALE,
    H1_BASE_WEIGHT,
    _load_year_context,
    _locked_contract,
    _source_contracts,
    affine,
    c3_mix,
    compose,
    gate_library,
    metrics,
    overwrite_gate,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v205_h1_workload_strict_forward_audit import (
    apply_component_delta,
    load_batter_ids_by_year,
    strict_axis,
)
from src.archive.v206_augmented_h1_weight_scope_audit import source_gate
from src.core.contract import _load_contract_axis


PROTOCOL = "V209_H1_WORKLOAD_MULTISEED_AUDIT_V1"
SEEDS = (42, 43, 44)
YEARS = (2022, 2023, 2024)
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
PRIMARY_ACTIVE_WEIGHT = 0.18
DIAGNOSTIC_ACTIVE_WEIGHTS = (0.21, 0.26)
STRICT_COMPONENT_SCALE = 0.10


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "strictly_prior_season_h1_fits": True,
        "paired_three_seed_confirmation": True,
        "primary_weight_and_scope_frozen_from_v206": True,
        "diagnostic_weights_cannot_promote": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def load_seed_predictions(
    year: int,
    seed: int,
    baseline_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.load(
        baseline_dir / f"h1_year{year}_seed{seed}.npy", allow_pickle=False
    ).astype(np.float64)
    augmented_dir = seed42_dir if seed == 42 else multiseed_dir
    augmented = np.load(
        augmented_dir / f"augmented_h1_year{year}_seed{seed}.npy",
        allow_pickle=False,
    ).astype(np.float64)
    if baseline.shape != augmented.shape:
        raise ValueError(f"paired seed shape mismatch: {year}/{seed}")
    return baseline, augmented


def run(
    train_csv: Path,
    baseline_checkpoint_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context, frames, correction = _load_year_context(train_csv)
    season = context["season"].to_numpy(np.int16)
    target = context["control_success"].to_numpy(np.float64)
    batter_ids = load_batter_ids_by_year(train_csv, season)
    for year in YEARS:
        frames[year] = frames[year].assign(batter_id=batter_ids[year])
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    sources = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    locked, locked_parity = _locked_contract(
        axes["full_2024"], frames[2024], correction[2024],
        h1_path, c3_path, v160_path,
    )
    if parity["locked_runtime_parity"] != locked_parity:
        raise ValueError("locked parity changed between reconstructions")

    baseline_raw: dict[int, dict[int, np.ndarray]] = {}
    augmented_raw: dict[int, dict[int, np.ndarray]] = {}
    for year in YEARS:
        baseline_raw[year], augmented_raw[year] = {}, {}
        for seed in SEEDS:
            base, aug = load_seed_predictions(
                year, seed, baseline_checkpoint_dir, seed42_dir, multiseed_dir
            )
            baseline_raw[year][seed] = base
            augmented_raw[year][seed] = aug

    baseline_mean = {
        year: np.mean([baseline_raw[year][seed] for seed in SEEDS], axis=0)
        for year in YEARS
    }
    augmented_mean = {
        year: np.mean([augmented_raw[year][seed] for seed in SEEDS], axis=0)
        for year in YEARS
    }
    baseline_h1 = {
        year: affine(baseline_mean[year] + correction[year]) for year in YEARS
    }
    augmented_h1 = {
        year: affine(augmented_mean[year] + correction[year]) for year in YEARS
    }
    h1_parity = {
        "full_2022": float(
            np.max(np.abs(baseline_h1[2022] - sources["full_2022"]["h1"]))
        ),
        "late_2023": float(
            np.max(
                np.abs(
                    baseline_h1[2023][late23]
                    - sources["late_2023"]["h1"]
                )
            )
        ),
        "full_2024": float(
            np.max(np.abs(baseline_h1[2024] - locked["h1"]))
        ),
    }
    if max(h1_parity.values()) > 1e-12:
        raise ValueError(f"three-seed H1 parity failed: {h1_parity}")

    strict_seed: dict[str, dict[str, Any]] = {}
    strict_ensemble: dict[str, Any] = {}
    for year in YEARS:
        frame = frames[year]
        audit_axis = strict_axis(frame, target[season == year], baseline_h1[year])
        ensemble_candidate = apply_component_delta(
            baseline_h1[year], augmented_h1[year], STRICT_COMPONENT_SCALE
        )
        strict_ensemble[str(year)] = metrics(
            audit_axis, baseline_h1[year], ensemble_candidate
        )
        strict_seed[str(year)] = {}
        for seed in SEEDS:
            base = affine(baseline_raw[year][seed] + correction[year])
            augmented = affine(augmented_raw[year][seed] + correction[year])
            candidate = apply_component_delta(
                base, augmented, STRICT_COMPONENT_SCALE
            )
            seed_axis = strict_axis(frame, target[season == year], base)
            strict_seed[str(year)][str(seed)] = metrics(
                seed_axis, base, candidate
            )

    source_parts: dict[str, dict[str, np.ndarray]] = {}
    for axis_name in SOURCE_AXES:
        values = sources[axis_name]
        source_parts[axis_name] = {
            "component": values["component"],
            "c3_active": c3_mix(values["sign"], values["recent"], 0.25),
        }
    with np.load(bridge_oof, allow_pickle=False) as saved:
        historical_parent = saved["parent"].astype(np.float64)
        bridge025_top = saved["v142_full_2024"].astype(np.float64) + 0.25 * (
            saved["v138_full_2024"].astype(np.float64)
            - saved["v142_full_2024"].astype(np.float64)
        )
    component_delta = (bridge025_top - historical_parent) / (
        1.0 - H1_BASE_WEIGHT
    )
    locked_bridge_component = (
        locked["component"] + BRIDGE_SCALE * component_delta
    )

    weight_results: dict[str, dict[str, Any]] = {}
    weight_candidates: dict[str, dict[str, np.ndarray]] = {}
    weight_active: dict[str, dict[str, np.ndarray]] = {}
    for weight in (PRIMARY_ACTIVE_WEIGHT, *DIAGNOSTIC_ACTIVE_WEIGHTS):
        key = str(weight)
        weight_results[key], weight_candidates[key], weight_active[key] = {}, {}, {}
        for axis_name in AXES:
            if axis_name == "full_2022":
                h1 = augmented_h1[2022]
            elif axis_name == "late_2023":
                h1 = augmented_h1[2023][late23]
            else:
                h1 = augmented_h1[2024]
            if axis_name in SOURCE_AXES:
                component = source_parts[axis_name]["component"]
                c3_active = source_parts[axis_name]["c3_active"]
            else:
                component = locked_bridge_component
                c3_active = locked["c3_active"]
            proposal = compose(
                component, h1, c3_active, axes[axis_name], h1_weight=weight
            )
            candidate, active = overwrite_gate(
                parents[axis_name], proposal, axes[axis_name],
                axis_frames[axis_name], gate_library()["runners_or_high_li"],
            )
            weight_candidates[key][axis_name] = candidate
            weight_active[key][axis_name] = active
            weight_results[key][axis_name] = metrics(
                axes[axis_name], parents[axis_name], candidate
            )

    primary_key = str(PRIMARY_ACTIVE_WEIGHT)
    primary = weight_results[primary_key]
    source_pass = all(source_gate(primary[axis]) for axis in SOURCE_AXES)
    locked_result = primary["full_2024"]
    point_pass = bool(
        locked_result["gain"] > 0.0
        and locked_result["positive_month_fraction"] >= 0.625
        and locked_result["worst_month_gain"] > -3.0
        and locked_result["minimum_domain_gain"] >= 0.0
    )
    robust = _robustness(
        axes["full_2024"], parents["full_2024"],
        weight_candidates[primary_key]["full_2024"],
        weight_active[primary_key]["full_2024"],
        [
            parents["full_2024"].copy(),
            weight_candidates[primary_key]["full_2024"],
        ],
    )
    robust_pass = bool(
        robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] <= 0.10
    )
    seed_sign_pass = all(
        strict_seed[str(year)][str(seed)]["gain"] > 0.0
        for year in YEARS for seed in SEEDS
    )
    ensemble_sign_pass = all(
        strict_ensemble[str(year)]["gain"] > 0.0 for year in YEARS
    )
    confirm = bool(
        source_pass and point_pass and robust_pass
        and seed_sign_pass and ensemble_sign_pass
    )
    np.savez_compressed(
        output_dir / "primary_axis.npz",
        parent=parents["full_2024"],
        candidate=weight_candidates[primary_key]["full_2024"],
        active=weight_active[primary_key]["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_full_fit" if confirm else (
            "robust_reject" if not robust_pass else "point_or_seed_reject"
        ),
        "primary_active_weight": PRIMARY_ACTIVE_WEIGHT,
        "primary_scope": "runners_or_high_li",
        "strict_component_scale": STRICT_COMPONENT_SCALE,
        "h1_parity_max_abs": h1_parity,
        "strict_seed": strict_seed,
        "strict_ensemble": strict_ensemble,
        "all_nine_seed_year_gains_positive": seed_sign_pass,
        "all_three_ensemble_year_gains_positive": ensemble_sign_pass,
        "source": {axis: primary[axis] for axis in SOURCE_AXES},
        "source_gate_passed": source_pass,
        "locked_2024": locked_result,
        "locked_point_passed": point_pass,
        "robustness": robust,
        "robust_gate_passed": robust_pass,
        "diagnostic_weights_non_promoting": {
            str(weight): weight_results[str(weight)]
            for weight in DIAGNOSTIC_ACTIVE_WEIGHTS
        },
        "eligible_for_full_fit": confirm,
        "eligible_for_packaging": False,
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
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed42-dir", type=Path, required=True)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.baseline_checkpoint_dir, args.seed42_dir,
        args.multiseed_dir, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
