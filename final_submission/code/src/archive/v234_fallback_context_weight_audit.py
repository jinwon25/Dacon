"""Source-select one baseball-context weight adjustment inside Public1175."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v205_h1_workload_strict_forward_audit import load_batter_ids_by_year
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    XGB_WEIGHT,
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.archive.v224_fallback_xgb_complement_scope_audit import full_axis
from src.core.contract import _load_contract_axis


PROTOCOL = "V234_FALLBACK_CONTEXT_WEIGHT_SOURCE_AUDIT_V1"
ALTERNATE_WEIGHTS = (0.20, 0.40, 0.50)
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")


def context_gates(frame: pd.DataFrame, parent: np.ndarray) -> dict[str, np.ndarray]:
    balls = frame["balls_before"].to_numpy(np.int8)
    strikes = frame["strikes_before"].to_numpy(np.int8)
    inning = frame["inning"].to_numpy(np.int8)
    score = frame["score_diff_pitcher_team"].to_numpy(np.float64)
    runners = frame["num_runners_on"].to_numpy(np.int8)
    li = frame["li"].to_numpy(np.float64)
    pitcher_hand = frame["pitcher_hand"].astype(str).to_numpy()
    batter_hand = frame["batter_hand"].astype(str).to_numpy()
    parent = np.asarray(parent, dtype=np.float64)
    return {
        "all_active": np.ones(len(frame), dtype=bool),
        "runners": runners > 0,
        "high_li": li >= 1.5,
        "runners_and_high_li": (runners > 0) & (li >= 1.5),
        "multiple_runners": runners >= 2,
        "three_ball": balls == 3,
        "two_strike": strikes == 2,
        "batter_ahead": balls > strikes,
        "pitcher_ahead": strikes > balls,
        "even_count": balls == strikes,
        "early_inning": inning <= 3,
        "middle_inning": (inning >= 4) & (inning <= 6),
        "late_inning": inning >= 7,
        "score_close": np.abs(score) <= 1.0,
        "score_tied": score == 0.0,
        "pitcher_team_ahead": score > 0.0,
        "pitcher_team_behind": score < 0.0,
        "same_hand": pitcher_hand == batter_hand,
        "opposite_hand": pitcher_hand != batter_hand,
        "parent_050_052": (parent >= 0.50) & (parent < 0.52),
        "parent_ge_052": parent >= 0.52,
        "parent_ge_054": parent >= 0.54,
    }


def apply_context_weight(
    current: np.ndarray,
    parent: np.ndarray,
    xgb_prediction: np.ndarray,
    active: np.ndarray,
    gate: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    changed = np.asarray(active, dtype=bool) & np.asarray(gate, dtype=bool)
    output = np.asarray(current, dtype=np.float64).copy()
    direction = (
        np.asarray(xgb_prediction, dtype=np.float64)
        - np.asarray(parent, dtype=np.float64)
    )
    if not np.isfinite(direction[changed]).all():
        raise ValueError("context adjustment contains missing XGB prediction")
    output[changed] = np.clip(
        np.asarray(parent, dtype=np.float64)[changed]
        + float(weight) * direction[changed],
        0.001,
        0.999,
    )
    return output, changed


def source_gate(results: dict[str, dict[str, Any]]) -> bool:
    return bool(
        all(results[name]["gain"] > 0.0 for name in SOURCE_AXES)
        and results["full_2022"]["positive_month_fraction"] >= (5.0 / 7.0)
        and results["late_2023"]["positive_month_fraction"] >= (2.0 / 3.0)
        and results["full_2022"]["worst_month_gain"] > -5.0
        and results["late_2023"]["worst_month_gain"] > -5.0
        and results["full_2022"]["minimum_domain_gain"] >= 0.0
        and results["late_2023"]["minimum_domain_gain"] >= 0.0
    )


def select_rule(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, str]] = []
    for key, axes in results.items():
        if source_gate(axes):
            gains = [axes[name]["gain"] for name in SOURCE_AXES]
            eligible.append((min(gains), float(np.mean(gains)), key))
    return None if not eligible else max(eligible)[-1]


def restrictions() -> dict[str, bool]:
    return {
        "fallback_xgb_model_frozen": True,
        "public1175_active_support_frozen": True,
        "one_context_rule_only": True,
        "row_local_baseball_contexts_only": True,
        "source_only_rule_selection": True,
        "locked_2024_not_used_for_selection": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
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
        "full_2022": full_axis(_load_contract_axis(
            contract_dir / "v84_full_2022.npz"
        )),
        "late_2023": full_axis(_load_contract_axis(
            contract_dir / "v84_late_2023.npz"
        )),
        "full_2024": full_axis(_load_contract_axis(bridge_oof)),
    }
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
    current: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    gates: dict[str, dict[str, np.ndarray]] = {}
    for name in AXES:
        current[name], active[name] = apply_fallback(
            parent[name], xgb[name], pressure[name]
        )
        gates[name] = context_gates(axis_frames[name], parent[name])
    if any(set(gates[name]) != set(gates["full_2022"]) for name in AXES):
        raise ValueError("context gate library differs by axis")

    results: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    changed_masks: dict[str, dict[str, np.ndarray]] = {}
    for gate_name in gates["full_2022"]:
        for weight in ALTERNATE_WEIGHTS:
            key = f"{gate_name}__w{weight:.2f}"
            results[key], candidates[key], changed_masks[key] = {}, {}, {}
            for name in AXES:
                candidate, changed = apply_context_weight(
                    current[name], parent[name], xgb[name], active[name],
                    gates[name][gate_name], weight,
                )
                candidates[key][name] = candidate
                changed_masks[key][name] = changed
                result = metrics(axes[name], current[name], candidate)
                result["changed_rows"] = int(changed.sum())
                results[key][name] = result
    selected = select_rule(results)
    locked = None if selected is None else results[selected]["full_2024"]
    point_pass = bool(
        locked is not None
        and locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -5.0
        and locked["minimum_domain_gain"] >= 0.0
    )
    robustness = None
    robust_pass = False
    if selected is not None:
        passing = [key for key in results if source_gate(results[key])]
        family = [candidates[key]["full_2024"] for key in passing]
        family.append(current["full_2024"])
        robustness = _robustness(
            axes["full_2024"], current["full_2024"],
            candidates[selected]["full_2024"],
            changed_masks[selected]["full_2024"], family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] <= 0.10
        )
    confirm = bool(selected is not None and point_pass and robust_pass)
    if selected is not None:
        np.savez_compressed(
            output_dir / "selected_axes.npz",
            current_full_2024=current["full_2024"],
            candidate_full_2024=candidates[selected]["full_2024"],
            changed_full_2024=changed_masks[selected]["full_2024"],
        )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_release_build" if confirm else (
            "source_reject" if selected is None else (
                "locked_point_reject" if not point_pass else "robust_reject"
            )
        ),
        "candidate_count": len(results),
        "rule_results": results,
        "selected_rule_from_sources": selected,
        "locked_2024_incremental": locked,
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
        "selected": result["selected_rule_from_sources"],
        "source": None if result["selected_rule_from_sources"] is None else {
            name: result["rule_results"][result["selected_rule_from_sources"]][name]
            for name in SOURCE_AXES
        },
        "locked": result["locked_2024_incremental"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
