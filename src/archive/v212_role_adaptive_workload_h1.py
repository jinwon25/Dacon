"""Audit baseball-role adaptive deployment of the three-seed workload H1.

The workload-augmented H1 is positive on every strict year/seed diagnostic, but
its signal is diluted by the frozen top-level ensemble.  This experiment keeps
the previously frozen low/high H1 weights and varies only where the high dose
is used.  The six row-local starter/reliever workload regimes were defined in
v202, before the augmented-H1 predictions existed.  Full-2022 and late-2023
select a recipe before full-2024 is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import (
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
    rcore_mask,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v202_workload_gated_h1_affine import WORKLOAD_COLUMNS, role_gates
from src.archive.v206_augmented_h1_weight_scope_audit import source_gate
from src.archive.v209_h1_workload_multiseed_audit import (
    SEEDS,
    load_seed_predictions,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V212_ROLE_ADAPTIVE_WORKLOAD_H1_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
LOW_WEIGHT = 0.18
HIGH_WEIGHT = 0.26
MODES = ("low_then_high", "high_only")


def apply_role_transport(
    parent: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    axis: dict[str, np.ndarray],
    frame: pd.DataFrame,
    role: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fixed high-dose proposal inside one row-local workload role."""

    parent = np.asarray(parent, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    role = np.asarray(role, dtype=bool)
    jy = gate_library()["runners_or_high_li"](frame)
    base_active = rcore_mask(axis) & jy
    high_active = base_active & role
    output = parent.copy()
    if mode == "low_then_high":
        output[base_active] = low[base_active]
    elif mode != "high_only":
        raise ValueError(f"unknown role transport mode: {mode}")
    output[high_active] = high[high_active]
    return np.clip(output, 0.001, 0.999), high_active


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "fixed_v202_baseball_role_gates": True,
        "fixed_v209_three_seed_workload_h1": True,
        "fixed_low_and_high_weights": True,
        "source_selection_before_locked_2024": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


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
    _context, raw_frames, correction = _load_year_context(train_csv)
    workload = pd.read_csv(
        train_csv, usecols=list(WORKLOAD_COLUMNS), low_memory=False
    )
    workload_season = workload["season"].to_numpy(np.int16)
    frames = {
        year: workload.loc[workload_season == year].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
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
        axes, raw_frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    sources = _source_contracts(
        axes, raw_frames, correction, v104_path, h1_path, c3_path
    )
    locked, locked_parity = _locked_contract(
        axes["full_2024"], raw_frames[2024], correction[2024],
        h1_path, c3_path, v160_path,
    )
    if parity["locked_runtime_parity"] != locked_parity:
        raise ValueError("locked parity changed between reconstructions")

    augmented_h1: dict[int, np.ndarray] = {}
    for year in (2022, 2023, 2024):
        augmented = []
        for seed in SEEDS:
            _baseline, prediction = load_seed_predictions(
                year, seed, baseline_checkpoint_dir, seed42_dir, multiseed_dir
            )
            augmented.append(prediction)
        augmented_h1[year] = affine(
            np.mean(augmented, axis=0) + correction[year]
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
    locked_component = locked["component"] + BRIDGE_SCALE * component_delta

    proposals: dict[str, dict[str, np.ndarray]] = {"low": {}, "high": {}}
    for axis_name in AXES:
        if axis_name == "full_2022":
            h1 = augmented_h1[2022]
        elif axis_name == "late_2023":
            h1 = augmented_h1[2023][late23]
        else:
            h1 = augmented_h1[2024]
        if axis_name in SOURCE_AXES:
            component = source_parts[axis_name]["component"]
            c3 = source_parts[axis_name]["c3_active"]
        else:
            component = locked_component
            c3 = locked["c3_active"]
        proposals["low"][axis_name] = compose(
            component, h1, c3, axes[axis_name], h1_weight=LOW_WEIGHT
        )
        proposals["high"][axis_name] = compose(
            component, h1, c3, axes[axis_name], h1_weight=HIGH_WEIGHT
        )

    proposal_artifact: dict[str, np.ndarray] = {}
    for axis_name in AXES:
        jy_active = (
            rcore_mask(axes[axis_name])
            & gate_library()["runners_or_high_li"](axis_frames[axis_name])
        )
        proposal_artifact[f"{axis_name}_parent"] = parents[axis_name]
        proposal_artifact[f"{axis_name}_target"] = axes[axis_name]["target"]
        proposal_artifact[f"{axis_name}_month"] = axes[axis_name]["game_month"]
        proposal_artifact[f"{axis_name}_pitcher_id"] = axes[axis_name][
            "pitcher_id"
        ]
        proposal_artifact[f"{axis_name}_batter_id"] = axes[axis_name][
            "batter_id"
        ]
        proposal_artifact[f"{axis_name}_active"] = jy_active
        for dose in ("low", "high"):
            candidate = parents[axis_name].copy()
            candidate[jy_active] = proposals[dose][axis_name][jy_active]
            proposal_artifact[f"{axis_name}_{dose}"] = candidate
    np.savez_compressed(output_dir / "weight_proposals.npz", **proposal_artifact)

    gates = {axis: role_gates(axis_frames[axis]) for axis in AXES}
    details: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    active_masks: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    gate_names = tuple(gates["full_2022"])
    for gate_name in gate_names:
        for mode in MODES:
            name = f"{gate_name}_{mode}"
            details[name], candidates[name], active_masks[name] = {}, {}, {}
            row: dict[str, Any] = {
                "candidate": name,
                "gate": gate_name,
                "mode": mode,
            }
            for axis_name in AXES:
                candidate, active = apply_role_transport(
                    parents[axis_name], proposals["low"][axis_name],
                    proposals["high"][axis_name], axes[axis_name],
                    axis_frames[axis_name], gates[axis_name][gate_name], mode,
                )
                candidates[name][axis_name] = candidate
                active_masks[name][axis_name] = active
                result = metrics(axes[axis_name], parents[axis_name], candidate)
                details[name][axis_name] = result
                row[f"{axis_name}_gain"] = result["gain"]
                row[f"{axis_name}_month_fraction"] = result[
                    "positive_month_fraction"
                ]
                row[f"{axis_name}_worst_month"] = result["worst_month_gain"]
                row[f"{axis_name}_active_fraction"] = result["active_fraction"]
            row["source_gate_passed"] = all(
                source_gate(details[name][axis]) for axis in SOURCE_AXES
            )
            row["source_min_gain"] = min(
                details[name][axis]["gain"] for axis in SOURCE_AXES
            )
            row["source_mean_gain"] = float(np.mean([
                details[name][axis]["gain"] for axis in SOURCE_AXES
            ]))
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain", "source_mean_gain"],
        ascending=[False, False, False], kind="stable",
    )
    table.to_csv(output_dir / "source_screen.csv", index=False, encoding="utf-8-sig")
    passing = table.loc[table["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "screen": table.to_dict(orient="records"),
            "eligible_for_packaging": False,
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected = str(passing.iloc[0]["candidate"])
        locked_result = details[selected]["full_2024"]
        locked_candidate = candidates[selected]["full_2024"]
        active = active_masks[selected]["full_2024"]
        family = [
            candidates[str(name)]["full_2024"]
            for name in passing["candidate"].tolist()
        ]
        family.append(parents["full_2024"].copy())
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], locked_candidate,
            active, family,
        )
        point_pass = bool(
            locked_result["gain"] > 0.0
            and locked_result["positive_month_fraction"] >= 0.625
            and locked_result["worst_month_gain"] > -3.0
            and locked_result["minimum_domain_gain"] >= 0.0
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"], candidate=locked_candidate,
            active=active,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "confirmed_for_full_fit" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "selected_candidate": selected,
            "source": {axis: details[selected][axis] for axis in SOURCE_AXES},
            "locked_2024": locked_result,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "screen": table.to_dict(orient="records"),
            "eligible_for_full_fit": bool(point_pass and robust_pass),
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
