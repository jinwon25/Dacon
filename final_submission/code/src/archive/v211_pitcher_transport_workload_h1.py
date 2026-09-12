"""Transport workload-H1 benefit through a prior-year pitcher EB gate.

Pitch-location mechanics differ by pitcher, so a workload feature can help one
pitcher and hurt another.  This audit estimates that heterogeneity only from
earlier labelled origins.  Full-2022 uses 2020/2021 H1 OOF utility; late-2023
uses full-2022 top-level utility; full-2024 uses full-2022 plus late-2023.  No
audit-year label enters its own gate.
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
from src.archive.v209_h1_workload_multiseed_audit import (
    SEEDS,
    load_seed_predictions,
)
from src.archive.v206_augmented_h1_weight_scope_audit import source_gate
from src.champion.v130_catboost_independent_oof_blend import post4
from src.core.contract import _load_contract_axis


PROTOCOL = "V211_PITCHER_TRANSPORT_WORKLOAD_H1_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
ALPHAS = (200.0, 1000.0, 5000.0)
MODES = ("only_positive", "boost_positive")
LOW_WEIGHT = 0.18
HIGH_WEIGHT = 0.26
EARLY_H1_SCALE = 0.10


def pitcher_eb_gate(
    history_pitcher: np.ndarray,
    history_improvement: np.ndarray,
    query_pitcher: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pitcher = np.asarray(history_pitcher, dtype=np.int64)
    improvement = np.asarray(history_improvement, dtype=np.float64)
    query = np.asarray(query_pitcher, dtype=np.int64)
    if len(pitcher) != len(improvement) or len(pitcher) == 0:
        raise ValueError("pitcher utility history is empty or misaligned")
    global_mean = float(np.mean(improvement))
    table = pd.DataFrame({"pitcher_id": pitcher, "gain": improvement}).groupby(
        "pitcher_id", observed=True, sort=False
    )["gain"].agg(["sum", "count"])
    posterior = (table["sum"] + float(alpha) * global_mean) / (
        table["count"] + float(alpha)
    )
    mapped = pd.Series(query).map(posterior).fillna(global_mean).to_numpy(float)
    gate = mapped > 0.0
    return gate, {
        "alpha": float(alpha),
        "history_rows": len(pitcher),
        "history_pitchers": len(table),
        "history_global_mean_brier_improvement": global_mean,
        "positive_history_pitcher_fraction": float(np.mean(posterior > 0.0)),
        "query_positive_fraction": float(np.mean(gate)),
        "query_unknown_fraction": float(
            np.mean(~np.isin(query, table.index.to_numpy(np.int64)))
        ),
    }


def row_improvement(
    target: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    return np.square(parent[mask] - target[mask]) - np.square(
        candidate[mask] - target[mask]
    )


def apply_pitcher_transport(
    parent: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    axis: dict[str, np.ndarray],
    frame: pd.DataFrame,
    pitcher_positive: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    jy = gate_library()["runners_or_high_li"](frame)
    base_active = rcore_mask(axis) & jy
    high_active = base_active & np.asarray(pitcher_positive, dtype=bool)
    if mode == "only_positive":
        output = np.asarray(parent, dtype=np.float64).copy()
    elif mode == "boost_positive":
        output = np.asarray(parent, dtype=np.float64).copy()
        output[base_active] = np.asarray(low, dtype=np.float64)[base_active]
    else:
        raise ValueError(f"unknown transport mode: {mode}")
    output[high_active] = np.asarray(high, dtype=np.float64)[high_active]
    return np.clip(output, 0.001, 0.999), high_active


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "strictly_prior_origin_pitcher_gate": True,
        "fixed_eb_alphas_and_modes": True,
        "three_seed_workload_h1": True,
        "source_selection_before_locked_2024": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference_with_frozen_pitcher_table": True,
    }


def run(
    train_csv: Path,
    baseline_checkpoint_dir: Path,
    early_augmented_dir: Path,
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
    target_all = context["control_success"].to_numpy(np.float64)
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
    locked, _locked_parity = _locked_contract(
        axes["full_2024"], frames[2024], correction[2024],
        h1_path, c3_path, v160_path,
    )

    augmented_h1: dict[int, np.ndarray] = {}
    for year in (2022, 2023, 2024):
        augmented = []
        for seed in SEEDS:
            _base, aug = load_seed_predictions(
                year, seed, baseline_checkpoint_dir, seed42_dir, multiseed_dir
            )
            augmented.append(aug)
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
    locked_bridge_component = (
        locked["component"] + BRIDGE_SCALE * component_delta
    )

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
            component = locked_bridge_component
            c3 = locked["c3_active"]
        proposals["low"][axis_name] = compose(
            component, h1, c3, axes[axis_name], h1_weight=LOW_WEIGHT
        )
        proposals["high"][axis_name] = compose(
            component, h1, c3, axes[axis_name], h1_weight=HIGH_WEIGHT
        )

    # Early untouched H1 utility supplies only the first (2022) pitcher gate.
    early_pitcher, early_utility = [], []
    for year in (2020, 2021):
        audit = season == year
        frame = context.loc[audit].reset_index(drop=True)
        early_correction = post4(
            context.loc[season < year].reset_index(drop=True), frame
        )
        base_raw = np.load(
            baseline_checkpoint_dir / f"h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64)
        aug_raw = np.load(
            early_augmented_dir / f"augmented_h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64)
        base = affine(base_raw + early_correction)
        augmented = affine(aug_raw + early_correction)
        candidate = base + EARLY_H1_SCALE * (augmented - base)
        use = np.ones(len(frame), dtype=bool)
        early_pitcher.append(frame["pitcher_id"].to_numpy(np.int64))
        early_utility.append(
            row_improvement(target_all[audit], base, candidate, use)
        )
    history = {
        "full_2022": (
            np.concatenate(early_pitcher), np.concatenate(early_utility)
        )
    }

    ungated_high: dict[str, np.ndarray] = {}
    for axis_name in AXES:
        ungated_high[axis_name], _ = overwrite_gate(
            parents[axis_name], proposals["high"][axis_name], axes[axis_name],
            axis_frames[axis_name], gate_library()["runners_or_high_li"],
        )
    exact22 = axes["full_2022"]["exact_mask"].astype(bool) & rcore_mask(
        axes["full_2022"]
    )
    utility22 = row_improvement(
        axes["full_2022"]["target"], parents["full_2022"],
        ungated_high["full_2022"], exact22,
    )
    pitcher22 = axis_frames["full_2022"]["pitcher_id"].to_numpy(np.int64)[exact22]
    history["late_2023"] = (pitcher22, utility22)
    exact23 = axes["late_2023"]["exact_mask"].astype(bool) & rcore_mask(
        axes["late_2023"]
    )
    utility23 = row_improvement(
        axes["late_2023"]["target"], parents["late_2023"],
        ungated_high["late_2023"], exact23,
    )
    pitcher23 = axis_frames["late_2023"]["pitcher_id"].to_numpy(np.int64)[exact23]
    history["full_2024"] = (
        np.concatenate([pitcher22, pitcher23]),
        np.concatenate([utility22, utility23]),
    )

    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    active_masks: dict[str, dict[str, np.ndarray]] = {}
    gate_meta: dict[str, dict[str, Any]] = {}
    for alpha in ALPHAS:
        gate_meta[str(alpha)] = {}
        pitcher_gates: dict[str, np.ndarray] = {}
        for axis_name in AXES:
            pitcher_gates[axis_name], gate_meta[str(alpha)][axis_name] = (
                pitcher_eb_gate(
                    *history[axis_name],
                    axis_frames[axis_name]["pitcher_id"].to_numpy(np.int64),
                    alpha,
                )
            )
        for mode in MODES:
            name = f"a{alpha:g}_{mode}"
            details[name], candidates[name], active_masks[name] = {}, {}, {}
            row: dict[str, Any] = {
                "candidate": name, "alpha": alpha, "mode": mode
            }
            for axis_name in AXES:
                candidate, active = apply_pitcher_transport(
                    parents[axis_name], proposals["low"][axis_name],
                    proposals["high"][axis_name], axes[axis_name],
                    axis_frames[axis_name], pitcher_gates[axis_name], mode,
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
            "screen": table.to_dict(orient="records"),
            "gate_metadata": gate_meta,
            "eligible_for_packaging": False,
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
            "source": {
                axis: details[selected][axis] for axis in SOURCE_AXES
            },
            "locked_2024": locked_result,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "gate_metadata": gate_meta[str(float(passing.iloc[0]["alpha"]))],
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
    parser.add_argument("--early-augmented-dir", type=Path, required=True)
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
        args.train_csv, args.baseline_checkpoint_dir,
        args.early_augmented_dir, args.seed42_dir, args.multiseed_dir,
        args.contract_dir, args.v104_path, args.h1_path, args.c3_path,
        args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
