"""Audit joint-workload H1 as an orthogonal update above Public 1175."""

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
from src.core.contract import _load_contract_axis


PROTOCOL = "V218_PUBLIC1175_JOINT_H1_INCREMENTAL_AUDIT_V1"
SEEDS = (42, 43, 44)
H1_ACTIVE_WEIGHT = 0.18
XGB_WEIGHT = 0.30
XGB_THRESHOLD = 0.50
PUBLISHED_FALLBACK = {
    "full_2022": {"active_rows": 68486, "gain": 23.7791},
    "late_2023": {"active_rows": 11892, "gain": 4.2552},
    "full_2024": {"active_rows": 30798, "gain": 1.5353},
}


def load_joint_h1(
    year: int,
    correction: np.ndarray,
    seed42_dir: Path,
    multiseed_dir: Path,
) -> np.ndarray:
    predictions = []
    for seed in SEEDS:
        directory = seed42_dir if seed == 42 else multiseed_dir
        predictions.append(
            np.load(
                directory / f"joint_h1_year{year}_seed{seed}.npy",
                allow_pickle=False,
            ).astype(np.float64)
        )
    if len({value.shape for value in predictions}) != 1:
        raise ValueError(f"joint H1 seed shape mismatch: {year}")
    return affine(np.mean(predictions, axis=0) + correction)


def align_regular_prediction(frame, checkpoint: Path) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
    if len(prediction) != int(regular.sum()):
        raise ValueError(
            f"fallback XGB regular-row mismatch: {checkpoint} "
            f"expected={int(regular.sum())} actual={len(prediction)}"
        )
    aligned = np.full(len(frame), np.nan, dtype=np.float64)
    aligned[regular] = prediction
    return aligned


def apply_fallback(
    jy: np.ndarray,
    xgb_prediction: np.ndarray,
    pressure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active = (
        np.asarray(pressure, dtype=bool)
        & (np.asarray(jy, dtype=np.float64) >= XGB_THRESHOLD)
    )
    if not np.isfinite(xgb_prediction[active]).all():
        raise ValueError("fallback XGB is missing an active Regular prediction")
    output = np.asarray(jy, dtype=np.float64).copy()
    output[active] = np.clip(
        (1.0 - XGB_WEIGHT) * output[active]
        + XGB_WEIGHT * np.asarray(xgb_prediction, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output, active


def restrictions() -> dict[str, bool]:
    return {
        "public1175_formula_frozen": True,
        "fallback_xgb_weight_frozen": True,
        "fallback_xgb_threshold_frozen": True,
        "joint_h1_recipe_frozen_from_v214_v215": True,
        "h1_weight_frozen_from_v209": True,
        "strictly_prior_season_predictions": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
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
        axes,
        frames,
        correction,
        v104_path,
        h1_path,
        c3_path,
        v160_path,
        bridge_oof,
    )
    sources = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    locked, locked_parity = _locked_contract(
        axes["full_2024"],
        frames[2024],
        correction[2024],
        h1_path,
        c3_path,
        v160_path,
    )
    if parity["locked_runtime_parity"] != locked_parity:
        raise ValueError("locked row-region parity changed between reconstructions")

    joint_h1 = {
        year: load_joint_h1(
            year, correction[year], seed42_dir, multiseed_dir
        )
        for year in (2022, 2023, 2024)
    }
    source_parts = {
        name: {
            "component": sources[name]["component"],
            "c3": c3_mix(
                sources[name]["sign"], sources[name]["recent"], 0.25
            ),
        }
        for name in ("full_2022", "late_2023")
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

    xgb_full = {
        year: align_regular_prediction(
            frames[year],
            fallback_oof_dir / f"fallback_xgb_oof_{year}.npy",
        )
        for year in (2022, 2023, 2024)
    }
    xgb_axes = {
        "full_2022": xgb_full[2022],
        "late_2023": xgb_full[2023][late23],
        "full_2024": xgb_full[2024],
    }

    jy_candidate: dict[str, np.ndarray] = {}
    pressure: dict[str, np.ndarray] = {}
    public1175: dict[str, np.ndarray] = {}
    final_candidate: dict[str, np.ndarray] = {}
    parent_fallback_active: dict[str, np.ndarray] = {}
    candidate_fallback_active: dict[str, np.ndarray] = {}
    fallback_parity: dict[str, dict[str, Any]] = {}
    incremental: dict[str, dict[str, Any]] = {}
    for axis_name in ("full_2022", "late_2023", "full_2024"):
        if axis_name == "full_2022":
            h1 = joint_h1[2022]
        elif axis_name == "late_2023":
            h1 = joint_h1[2023][late23]
        else:
            h1 = joint_h1[2024]
        if axis_name in source_parts:
            component = source_parts[axis_name]["component"]
            c3 = source_parts[axis_name]["c3"]
        else:
            component = locked_component
            c3 = locked["c3_active"]
        proposal = compose(
            component, h1, c3, axes[axis_name], h1_weight=H1_ACTIVE_WEIGHT
        )
        jy_candidate[axis_name], pressure[axis_name] = overwrite_gate(
            parents[axis_name],
            proposal,
            axes[axis_name],
            axis_frames[axis_name],
            gate_library()["runners_or_high_li"],
        )
        public1175[axis_name], parent_fallback_active[axis_name] = apply_fallback(
            parents[axis_name], xgb_axes[axis_name], pressure[axis_name]
        )
        final_candidate[axis_name], candidate_fallback_active[axis_name] = apply_fallback(
            jy_candidate[axis_name], xgb_axes[axis_name], pressure[axis_name]
        )
        measured = metrics(
            axes[axis_name], parents[axis_name], public1175[axis_name]
        )
        expected = PUBLISHED_FALLBACK[axis_name]
        fallback_parity[axis_name] = {
            **measured,
            "active_rows": int(parent_fallback_active[axis_name].sum()),
            "expected_active_rows": expected["active_rows"],
            "expected_gain": expected["gain"],
            "active_row_match": bool(
                int(parent_fallback_active[axis_name].sum())
                == expected["active_rows"]
            ),
            "gain_abs_difference": float(abs(measured["gain"] - expected["gain"])),
        }
        incremental[axis_name] = metrics(
            axes[axis_name], public1175[axis_name], final_candidate[axis_name]
        )
        incremental[axis_name]["changed_rows"] = int(
            np.sum(np.abs(final_candidate[axis_name] - public1175[axis_name]) > 1e-15)
        )
        incremental[axis_name]["fallback_gate_entries"] = int(
            np.sum(
                candidate_fallback_active[axis_name]
                & ~parent_fallback_active[axis_name]
            )
        )
        incremental[axis_name]["fallback_gate_exits"] = int(
            np.sum(
                parent_fallback_active[axis_name]
                & ~candidate_fallback_active[axis_name]
            )
        )

    parity_pass = all(
        item["active_row_match"] and item["gain_abs_difference"] <= 0.10
        for item in fallback_parity.values()
    )
    source_pass = bool(
        incremental["full_2022"]["gain"] > 0.0
        and incremental["late_2023"]["gain"] > 0.0
        and incremental["full_2022"]["positive_month_fraction"] >= 0.70
        and incremental["late_2023"]["positive_month_fraction"] >= 0.70
    )
    locked = incremental["full_2024"]
    locked_point_pass = bool(
        locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -3.0
        and locked["minimum_domain_gain"] >= 0.0
    )
    changed_2024 = np.abs(
        final_candidate["full_2024"] - public1175["full_2024"]
    ) > 1e-15
    robust = _robustness(
        axes["full_2024"],
        public1175["full_2024"],
        final_candidate["full_2024"],
        changed_2024,
        [public1175["full_2024"], final_candidate["full_2024"]],
    )
    robust_pass = bool(
        robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] <= 0.10
    )
    confirm = bool(
        parity_pass and source_pass and locked_point_pass and robust_pass
    )

    np.savez_compressed(
        output_dir / "public1175_incremental_axes.npz",
        **{
            f"{kind}_{name}": values[name]
            for kind, values in (
                ("jy_parent", parents),
                ("jy_candidate", jy_candidate),
                ("public1175", public1175),
                ("candidate", final_candidate),
                ("fallback_parent_active", parent_fallback_active),
                ("fallback_candidate_active", candidate_fallback_active),
            )
            for name in ("full_2022", "late_2023", "full_2024")
        },
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_full_fit" if confirm else (
            "fallback_oof_parity_reject" if not parity_pass else (
                "robust_reject" if not robust_pass else "point_reject"
            )
        ),
        "public1175_fallback_parity": fallback_parity,
        "fallback_parity_passed": parity_pass,
        "incremental_above_public1175": incremental,
        "source_gate_passed": source_pass,
        "locked_point_passed": locked_point_pass,
        "robustness": robust,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": confirm,
        "eligible_for_packaging": False,
        "jy_parity": parity,
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
        args.train_csv,
        args.seed42_dir,
        args.multiseed_dir,
        args.fallback_oof_dir,
        args.contract_dir,
        args.v104_path,
        args.h1_path,
        args.c3_path,
        args.v160_path,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
