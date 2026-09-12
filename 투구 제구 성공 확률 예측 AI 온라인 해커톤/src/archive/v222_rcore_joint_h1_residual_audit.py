"""Audit the three-seed joint-H1 delta as an all-R_CORE residual.

Unlike the JY recipe, which replaces H1 only inside its pressure gate, this
candidate adds only the paired prediction difference caused by the frozen
v214/v215 workload features.  The small residual is applied to all R_CORE
rows, then the Public1175 fallback is replayed unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import (
    _load_year_context,
    affine,
    metrics,
    rcore_mask,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v205_h1_workload_strict_forward_audit import (
    load_batter_ids_by_year,
)
from src.archive.v216_joint_workload_h1_multiseed_audit import (
    load_joint_seed_predictions,
)
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.core.contract import _load_contract_axis


PROTOCOL = "V222_RCORE_JOINT_H1_RESIDUAL_AUDIT_V1"
YEARS = (2022, 2023, 2024)
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
SEEDS = (42, 43, 44)
SCALES = (0.02, 0.05, 0.10)


def apply_rcore_delta(
    parent: np.ndarray,
    direction: np.ndarray,
    axis: dict[str, np.ndarray],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if parent.shape != direction.shape:
        raise ValueError("parent/direction shape mismatch")
    active = rcore_mask(axis)
    output = parent.copy()
    output[active] = np.clip(
        output[active] + float(scale) * direction[active], 0.001, 0.999
    )
    return output, active


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= (2.0 / 3.0)
        and result["worst_month_gain"] > -5.0
        and result["minimum_domain_gain"] >= 0.0
    )


def select_scale(results: dict[str, dict[str, Any]]) -> float | None:
    eligible: list[tuple[float, float, float]] = []
    for key, axes in results.items():
        if all(source_gate(axes[name]) for name in SOURCE_AXES):
            gains = [axes[name]["gain"] for name in SOURCE_AXES]
            eligible.append((min(gains), float(np.mean(gains)), float(key)))
    return None if not eligible else max(eligible)[2]


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "paired_three_seed_joint_h1_delta_only": True,
        "all_rcore_scope_fixed_before_audit": True,
        "source_scale_selection_before_locked_2024": True,
        "public1175_fallback_formula_frozen": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    baseline_checkpoint_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
    fallback_oof_dir: Path,
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
    parents, parity = exact_jy_parents(
        axes, frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )

    direction_full: dict[int, np.ndarray] = {}
    for year in YEARS:
        baseline, joint = [], []
        for seed in SEEDS:
            base, candidate = load_joint_seed_predictions(
                year, seed, baseline_checkpoint_dir, seed42_dir, multiseed_dir
            )
            baseline.append(base)
            joint.append(candidate)
        baseline_h1 = affine(np.mean(baseline, axis=0) + correction[year])
        joint_h1 = affine(np.mean(joint, axis=0) + correction[year])
        direction_full[year] = joint_h1 - baseline_h1
    directions = {
        "full_2022": direction_full[2022],
        "late_2023": direction_full[2023][late23],
        "full_2024": direction_full[2024],
    }

    exact_results: dict[str, dict[str, Any]] = {}
    exact_candidates: dict[str, dict[str, np.ndarray]] = {}
    exact_active: dict[str, dict[str, np.ndarray]] = {}
    for scale in SCALES:
        key = f"{scale:.2f}"
        exact_results[key], exact_candidates[key], exact_active[key] = {}, {}, {}
        for name in AXES:
            candidate, active = apply_rcore_delta(
                parents[name], directions[name], axes[name], scale
            )
            exact_candidates[key][name] = candidate
            exact_active[key][name] = active
            exact_results[key][name] = metrics(axes[name], parents[name], candidate)
    selected = select_scale(exact_results)
    selected_key = None if selected is None else f"{selected:.2f}"

    full_xgb = {
        year: align_regular_prediction(
            frames[year], fallback_oof_dir / f"hyunku_fallback_xgb_{year}.npy"
        )
        for year in YEARS
    }
    xgb = {
        "full_2022": full_xgb[2022],
        "late_2023": full_xgb[2023][late23],
        "full_2024": full_xgb[2024],
    }
    evidence_parent = {name: axes[name]["parent"].astype(np.float64) for name in AXES}
    pressure = {name: pressure_gate(axis_frames[name]) for name in AXES}
    public1175_proxy = {
        name: apply_fallback(evidence_parent[name], xgb[name], pressure[name])[0]
        for name in AXES
    }
    proxy_results: dict[str, dict[str, Any]] = {}
    proxy_candidates: dict[str, dict[str, np.ndarray]] = {}
    for scale in SCALES:
        key = f"{scale:.2f}"
        proxy_results[key], proxy_candidates[key] = {}, {}
        for name in AXES:
            transported, _active = apply_rcore_delta(
                evidence_parent[name], directions[name], axes[name], scale
            )
            candidate, fallback_active = apply_fallback(
                transported, xgb[name], pressure[name]
            )
            proxy_candidates[key][name] = candidate
            result = metrics(axes[name], public1175_proxy[name], candidate)
            result["fallback_active_rows"] = int(fallback_active.sum())
            proxy_results[key][name] = result

    locked = None if selected_key is None else exact_results[selected_key]["full_2024"]
    proxy_locked = None if selected_key is None else proxy_results[selected_key]["full_2024"]
    robustness = None
    point_pass = False
    robust_pass = False
    if selected_key is not None:
        point_pass = bool(
            locked["gain"] > 0.0
            and locked["positive_month_fraction"] >= 0.875
            and locked["worst_month_gain"] > 0.0
            and locked["minimum_domain_gain"] >= 0.0
        )
        family = [
            exact_candidates[f"{scale:.2f}"]["full_2024"] for scale in SCALES
        ] + [parents["full_2024"].copy()]
        robustness = _robustness(
            axes["full_2024"], parents["full_2024"],
            exact_candidates[selected_key]["full_2024"],
            exact_active[selected_key]["full_2024"], family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] <= 0.10
        )
    proxy_pass = bool(
        proxy_locked is not None
        and proxy_locked["gain"] >= 2.5
        and proxy_locked["positive_month_fraction"] >= 0.75
        and proxy_locked["worst_month_gain"] > -3.0
    )
    confirm = bool(point_pass and robust_pass and proxy_pass)
    if selected_key is not None:
        np.savez_compressed(
            output_dir / "selected_axes.npz",
            **{
                f"exact_parent_{name}": parents[name] for name in AXES
            },
            **{
                f"exact_candidate_{name}": exact_candidates[selected_key][name]
                for name in AXES
            },
            **{
                f"public1175_proxy_{name}": public1175_proxy[name] for name in AXES
            },
            **{
                f"proxy_candidate_{name}": proxy_candidates[selected_key][name]
                for name in AXES
            },
        )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_full_fit" if confirm else (
            "source_reject" if selected_key is None else (
                "locked_point_reject" if not point_pass else (
                    "robust_reject" if not robust_pass else "proxy_effect_reject"
                )
            )
        ),
        "exact_jy_results": exact_results,
        "selected_scale_from_sources": selected,
        "locked_2024": locked,
        "locked_point_passed": point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "public1175_proxy_results": proxy_results,
        "public1175_proxy_locked_2024": proxy_locked,
        "public1175_proxy_effect_passed": proxy_pass,
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
    parser.add_argument("--fallback-oof-dir", type=Path, required=True)
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
        args.multiseed_dir, args.fallback_oof_dir, args.contract_dir,
        args.v104_path, args.h1_path, args.c3_path, args.v160_path,
        args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
