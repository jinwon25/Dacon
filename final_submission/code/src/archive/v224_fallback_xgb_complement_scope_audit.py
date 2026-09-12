"""Audit unused R_CORE segments of the Public1175 fallback XGB.

The published formula and its 30% weight remain frozen.  Candidate scopes are
disjoint complements of the deployed pressure-and-parent>=0.50 support.  Scope
selection uses full-2022 and late-2023 only; full-2024 remains locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v205_h1_workload_strict_forward_audit import load_batter_ids_by_year
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    XGB_WEIGHT,
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.core.contract import _load_contract_axis


PROTOCOL = "V224_FALLBACK_XGB_COMPLEMENT_SCOPE_AUDIT_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
SCOPES = (
    "pressure_low50",
    "nonpressure_high50",
    "nonpressure_low50",
    "nonpressure_all",
    "all_inactive_rcore",
)


def complement_scopes(
    parent: np.ndarray,
    frame,
) -> dict[str, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    pressure = pressure_gate(frame)
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    rcore = regular & ~(
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    high = parent >= 0.50
    deployed = pressure & high
    values = {
        "pressure_low50": rcore & pressure & ~high,
        "nonpressure_high50": rcore & ~pressure & high,
        "nonpressure_low50": rcore & ~pressure & ~high,
        "nonpressure_all": rcore & ~pressure,
        "all_inactive_rcore": rcore & ~deployed,
    }
    if any(np.any(mask & deployed) for mask in values.values()):
        raise ValueError("complement scope overlaps deployed fallback")
    return values


def apply_additional_fallback(
    public1175: np.ndarray,
    xgb: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    output = np.asarray(public1175, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    if not np.isfinite(np.asarray(xgb)[active]).all():
        raise ValueError("additional XGB scope contains missing predictions")
    output[active] = np.clip(
        (1.0 - XGB_WEIGHT) * output[active]
        + XGB_WEIGHT * np.asarray(xgb, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output


def full_axis(axis: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {**axis, "exact_mask": np.ones(len(axis["target"]), dtype=bool)}


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= (2.0 / 3.0)
        and result["worst_month_gain"] > -5.0
        and result["minimum_domain_gain"] >= 0.0
    )


def select_scope(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, str]] = []
    for scope, axes in results.items():
        if all(source_gate(axes[name]) for name in SOURCE_AXES):
            gains = [axes[name]["gain"] for name in SOURCE_AXES]
            eligible.append((min(gains), float(np.mean(gains)), scope))
    return None if not eligible else max(eligible)[2]


def restrictions() -> dict[str, bool]:
    return {
        "fallback_xgb_model_frozen": True,
        "fallback_weight_frozen_at_public1175": True,
        "deployed_scope_unchanged": True,
        "candidate_scopes_disjoint_from_deployed": True,
        "source_only_scope_selection": True,
        "locked_2024_not_used_for_selection": True,
        "test_csv_read": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    fallback_oof_dir: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context, frames, _correction = _load_year_context(train_csv)
    season = context["season"].to_numpy(np.int16)
    batter_ids = load_batter_ids_by_year(train_csv, season)
    for year in (2022, 2023, 2024):
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
    score_axes = {name: full_axis(axes[name]) for name in AXES}
    parent = {name: axes[name]["parent"].astype(np.float64) for name in AXES}
    full_xgb = {
        year: align_regular_prediction(
            frames[year], fallback_oof_dir / f"fallback_xgb_oof_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    xgb = {
        "full_2022": full_xgb[2022],
        "late_2023": full_xgb[2023][late23],
        "full_2024": full_xgb[2024],
    }
    pressure = {name: pressure_gate(axis_frames[name]) for name in AXES}
    public1175 = {
        name: apply_fallback(parent[name], xgb[name], pressure[name])[0]
        for name in AXES
    }
    scopes = {
        name: complement_scopes(parent[name], axis_frames[name]) for name in AXES
    }
    results: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    for scope in SCOPES:
        results[scope], candidates[scope] = {}, {}
        for name in AXES:
            active = scopes[name][scope]
            candidate = apply_additional_fallback(public1175[name], xgb[name], active)
            candidates[scope][name] = candidate
            result = metrics(score_axes[name], public1175[name], candidate)
            result["active_rows"] = int(active.sum())
            results[scope][name] = result
    selected = select_scope(results)
    locked = None if selected is None else results[selected]["full_2024"]
    point_pass = bool(
        locked is not None
        and locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.75
        and locked["worst_month_gain"] > -5.0
        and locked["minimum_domain_gain"] >= 0.0
    )
    robustness = None
    robust_pass = False
    if selected is not None:
        passing = [
            scope for scope in SCOPES
            if all(source_gate(results[scope][name]) for name in SOURCE_AXES)
        ]
        family = [candidates[scope]["full_2024"] for scope in passing]
        family.append(public1175["full_2024"].copy())
        robustness = _robustness(
            score_axes["full_2024"], public1175["full_2024"],
            candidates[selected]["full_2024"], scopes["full_2024"][selected], family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] <= 0.10
        )
    confirm = bool(point_pass and robust_pass)
    if selected is not None:
        np.savez_compressed(
            output_dir / "selected_axes.npz",
            **{f"public1175_{name}": public1175[name] for name in AXES},
            **{f"candidate_{name}": candidates[selected][name] for name in AXES},
            **{f"active_{name}": scopes[name][selected] for name in AXES},
        )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_release_build" if confirm else (
            "source_reject" if selected is None else (
                "locked_point_reject" if not point_pass else "robust_reject"
            )
        ),
        "fixed_additional_weight": XGB_WEIGHT,
        "scope_results": results,
        "selected_scope_from_sources": selected,
        "locked_2024": locked,
        "locked_point_passed": point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_release_build": confirm,
        "eligible_for_packaging": False,
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
    parser.add_argument("--fallback-oof-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.fallback_oof_dir, args.contract_dir,
        args.bridge_oof, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "selected": result["selected_scope_from_sources"],
        "source": None if result["selected_scope_from_sources"] is None else {
            name: result["scope_results"][result["selected_scope_from_sources"]][name]
            for name in SOURCE_AXES
        },
        "locked": result["locked_2024"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
