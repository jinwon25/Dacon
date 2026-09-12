"""Audit whether the workload-augmented H1 deserves more top-level weight.

The candidate family is deliberately small and baseball-defined.  It replaces
the frozen H1 with the paired workload-augmented H1 at the deployed weight or
at one of three absolute weight lifts, either for all R_CORE rows, the deployed
runners/high-leverage scope, or its complement.  Full-2022 and late-2023 select
one recipe before the full-2024 locked view is evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import (
    BRIDGE_SCALE,
    H1_ACTIVE_WEIGHT,
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
    rcore_mask,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.core.contract import _load_contract_axis


PROTOCOL = "V206_AUGMENTED_H1_WEIGHT_SCOPE_AUDIT_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
WEIGHT_LIFTS = (0.0, 0.02, 0.05, 0.10)
SCOPES = ("all_rcore", "pressure_gate", "non_pressure_gate")


def scope_mask(
    axis: dict[str, np.ndarray],
    frame: pd.DataFrame,
    scope: str,
) -> np.ndarray:
    rcore = rcore_mask(axis)
    pressure_gate = gate_library()["runners_or_high_li"](frame)
    if scope == "all_rcore":
        return rcore
    if scope == "pressure_gate":
        return rcore & pressure_gate
    if scope == "non_pressure_gate":
        return rcore & ~pressure_gate
    raise ValueError(f"unknown scope: {scope}")


def scoped_replacement(
    parent: np.ndarray,
    proposal: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    output[active] = np.asarray(proposal, dtype=np.float64)[active]
    return np.clip(output, 0.001, 0.999)


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.50
        and result["worst_month_gain"] > -3.0
        and result["minimum_domain_gain"] >= 0.0
    )


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "paired_augmented_h1": True,
        "fixed_weight_lifts_and_baseball_scopes": True,
        "source_selection_before_locked_2024": True,
        "three_seed_confirmation_required": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def run(
    train_csv: Path,
    augmented_dir: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _context, frames, correction = _load_year_context(train_csv)
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

    augmented_raw = {
        year: np.load(
            augmented_dir / f"augmented_h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64)
        for year in (2022, 2023, 2024)
    }
    augmented_h1 = {
        "full_2022": affine(augmented_raw[2022] + correction[2022]),
        "late_2023": affine(
            (augmented_raw[2023] + correction[2023])[late23]
        ),
        "full_2024": affine(augmented_raw[2024] + correction[2024]),
    }

    source_parts: dict[str, dict[str, np.ndarray]] = {}
    for axis_name in SOURCE_AXES:
        values = sources[axis_name]
        source_parts[axis_name] = {
            "component": values["component"],
            "c3_base": c3_mix(values["sign"], values["recent"], 0.15),
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
    locked_parts = {
        "component": locked["component"],
        "bridge_component": locked["component"] + BRIDGE_SCALE * component_delta,
        "c3_base": locked["c3_base"],
        "c3_active": locked["c3_active"],
    }

    details: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    active_masks: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for lift in WEIGHT_LIFTS:
        base_weight = H1_BASE_WEIGHT + lift
        active_weight = H1_ACTIVE_WEIGHT + lift
        for scope in SCOPES:
            name = f"lift{lift:.2f}_{scope}"
            details[name], candidates[name], active_masks[name] = {}, {}, {}
            row: dict[str, Any] = {
                "candidate": name,
                "weight_lift": lift,
                "scope": scope,
                "base_weight": base_weight,
                "active_weight": active_weight,
            }
            for axis_name in AXES:
                frame = axis_frames[axis_name]
                if axis_name in SOURCE_AXES:
                    parts = source_parts[axis_name]
                    base = compose(
                        parts["component"], augmented_h1[axis_name],
                        parts["c3_base"], axes[axis_name],
                        h1_weight=base_weight,
                    )
                    active_proposal = compose(
                        parts["component"], augmented_h1[axis_name],
                        parts["c3_active"], axes[axis_name],
                        h1_weight=active_weight,
                    )
                else:
                    base = compose(
                        locked_parts["component"], augmented_h1[axis_name],
                        locked_parts["c3_base"], axes[axis_name],
                        h1_weight=base_weight,
                    )
                    active_proposal = compose(
                        locked_parts["bridge_component"], augmented_h1[axis_name],
                        locked_parts["c3_active"], axes[axis_name],
                        h1_weight=active_weight,
                    )
                full_proposal, _ = overwrite_gate(
                    base, active_proposal, axes[axis_name], frame,
                    gate_library()["runners_or_high_li"],
                )
                active = scope_mask(axes[axis_name], frame, scope)
                candidate = scoped_replacement(
                    parents[axis_name], full_proposal, active
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
            row["source_gate_passed"] = all(
                source_gate(details[name][axis]) for axis in SOURCE_AXES
            )
            row["source_min_gain"] = min(
                details[name][axis]["gain"] for axis in SOURCE_AXES
            )
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain"],
        ascending=[False, False], kind="stable",
    )
    table.to_csv(
        output_dir / "source_screen.csv", index=False, encoding="utf-8-sig"
    )
    passing = table.loc[table["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "candidate_count": len(table),
            "screen": table.to_dict(orient="records"),
            "parity": parity,
            "eligible_for_packaging": False,
            "restrictions": restrictions(),
        }
    else:
        selected_name = str(passing.iloc[0]["candidate"])
        locked_result = details[selected_name]["full_2024"]
        locked_candidate = candidates[selected_name]["full_2024"]
        active = active_masks[selected_name]["full_2024"]
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
            parent=parents["full_2024"],
            candidate=locked_candidate,
            active=active,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "confirm_three_seed" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "candidate_count": len(table),
            "selected_candidate": selected_name,
            "source": {
                axis: details[selected_name][axis] for axis in SOURCE_AXES
            },
            "locked_2024": locked_result,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_three_seed_confirmation": bool(
                point_pass and robust_pass
            ),
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
    parser.add_argument("--augmented-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.augmented_dir, args.contract_dir, args.v104_path,
        args.h1_path, args.c3_path, args.v160_path, args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
