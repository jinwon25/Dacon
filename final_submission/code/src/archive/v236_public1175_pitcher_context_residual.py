"""Strict-forward pitcher-context residual audit above Public1175.

The incumbent formula is frozen.  For each pitcher this experiment estimates
only the residual contrast between the two levels of three baseball contexts:
same handedness, two strikes, and runners on.  Tables are fit on earlier rows
only, strongly empirical-Bayes shrunk, and applied solely inside the incumbent
fallback support.  Full 2024 is opened once after source-only selection.
"""

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
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.archive.v224_fallback_xgb_complement_scope_audit import full_axis
from src.core.contract import _load_contract_axis


PROTOCOL = "V236_PUBLIC1175_PITCHER_CONTEXT_RESIDUAL_V1"
CONTEXTS = ("same_hand", "two_strike", "runner_on")
FAMILIES = {
    "hand_only": ("same_hand",),
    "c3": CONTEXTS,
}
FIT_SCOPES = ("rcore", "pressure", "public_active")
ETAS = (0.25, 0.50, 1.00)
SHRINK = {"same_hand": 1000.0, "two_strike": 1000.0, "runner_on": 2000.0}
CAP = 0.05
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("late_2022", "late_2023")


def context_values(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "same_hand": (
            rows["pitcher_hand"].astype(str).to_numpy()
            == rows["batter_hand"].astype(str).to_numpy()
        ).astype(np.int8),
        "two_strike": (rows["strikes_before"].to_numpy(np.int8) == 2).astype(np.int8),
        "runner_on": (rows["num_runners_on"].to_numpy(np.float64) > 0).astype(np.int8),
    }


def fit_differential(
    pitcher: np.ndarray,
    context: np.ndarray,
    residual: np.ndarray,
    shrink: float,
) -> pd.Series:
    frame = pd.DataFrame({
        "pitcher": np.asarray(pitcher),
        "context": np.asarray(context, dtype=np.int8),
        "residual": np.asarray(residual, dtype=np.float64),
    })
    grouped = frame.groupby(["pitcher", "context"], observed=True)["residual"].agg(
        ["mean", "size"]
    ).unstack()
    for field in ("mean", "size"):
        for value in (0, 1):
            if (field, value) not in grouped:
                grouped[(field, value)] = np.nan if field == "mean" else 0.0
    n0 = grouped[("size", 0)].fillna(0.0).astype(float)
    n1 = grouped[("size", 1)].fillna(0.0).astype(float)
    effective = (n0 * n1) / (n0 + n1).replace(0.0, np.nan)
    difference = grouped[("mean", 1)] - grouped[("mean", 0)]
    return (difference * effective / (effective + float(shrink))).dropna()


def predict_differential(
    table: pd.Series,
    pitcher: np.ndarray,
    context: np.ndarray,
    cap: float = CAP,
) -> np.ndarray:
    magnitude = pd.Series(np.asarray(pitcher)).map(table).fillna(0.0).to_numpy(float)
    sign = np.where(np.asarray(context, dtype=np.int8) == 1, 0.5, -0.5)
    return np.clip(sign * magnitude, -float(cap), float(cap))


def fit_scope_masks(
    rows: pd.DataFrame, public_active: np.ndarray
) -> dict[str, np.ndarray]:
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    rcore = regular & ~(
        rows["pitcher_team_id"].eq(13).to_numpy()
        | rows["batter_team_id"].eq(13).to_numpy()
    )
    pressure = pressure_gate(rows)
    return {
        "rcore": rcore,
        "pressure": pressure,
        "public_active": np.asarray(public_active, dtype=bool),
    }


def build_direction(
    source_rows: pd.DataFrame,
    source_residual: np.ndarray,
    source_mask: np.ndarray,
    query_rows: pd.DataFrame,
    family: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    source_context = context_values(source_rows)
    query_context = context_values(query_rows)
    source_pitcher = source_rows["pitcher_id"].to_numpy()[source_mask]
    query_pitcher = query_rows["pitcher_id"].to_numpy()
    direction = np.zeros(len(query_rows), dtype=np.float64)
    audit: dict[str, Any] = {"fit_rows": int(np.sum(source_mask)), "contexts": {}}
    for name in family:
        table = fit_differential(
            source_pitcher,
            source_context[name][source_mask],
            np.asarray(source_residual, dtype=np.float64)[source_mask],
            SHRINK[name],
        )
        correction = predict_differential(
            table, query_pitcher, query_context[name]
        )
        direction += correction
        audit["contexts"][name] = {
            "pitchers": int(len(table)),
            "median_abs_contrast": float(table.abs().median()) if len(table) else 0.0,
            "query_nonzero_fraction": float(np.mean(correction != 0.0)),
        }
    return direction, audit


def apply_direction(
    base: np.ndarray,
    direction: np.ndarray,
    active: np.ndarray,
    eta: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    output[active] = np.clip(
        output[active] + float(eta) * np.asarray(direction, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output


def source_gate(results: dict[str, dict[str, Any]]) -> bool:
    return bool(
        all(results[name]["gain"] > 0.0 for name in SOURCE_AXES)
        and results["late_2022"]["positive_month_fraction"] >= (2.0 / 3.0)
        and results["late_2023"]["positive_month_fraction"] >= (2.0 / 3.0)
        and all(results[name]["worst_month_gain"] > -5.0 for name in SOURCE_AXES)
        and all(results[name]["minimum_domain_gain"] >= 0.0 for name in SOURCE_AXES)
    )


def select_candidate(results: dict[str, dict[str, Any]]) -> str | None:
    eligible = []
    for key, values in results.items():
        if source_gate(values):
            gains = [values[name]["gain"] for name in SOURCE_AXES]
            eligible.append((min(gains), float(np.mean(gains)), key))
    return None if not eligible else max(eligible)[-1]


def restrictions() -> dict[str, bool]:
    return {
        "public1175_formula_frozen": True,
        "fallback_support_frozen": True,
        "strict_forward_residual_tables": True,
        "pitcher_level_removed_by_binary_contrast": True,
        "source_only_family_scope_eta_selection": True,
        "locked_2024_not_used_for_selection": True,
        "external_code_or_weights_used": False,
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
    late23_mask = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23_mask].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = {
        "full_2022": full_axis(_load_contract_axis(contract_dir / "v84_full_2022.npz")),
        "late_2023": full_axis(_load_contract_axis(contract_dir / "v84_late_2023.npz")),
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
        "late_2023": full_xgb[2023][late23_mask],
        "full_2024": full_xgb[2024],
    }
    incumbent: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    scopes: dict[str, dict[str, np.ndarray]] = {}
    for name in AXES:
        incumbent[name], active[name] = apply_fallback(
            parent[name], xgb[name], pressure_gate(axis_frames[name])
        )
        scopes[name] = fit_scope_masks(axis_frames[name], active[name])

    target22 = axes["full_2022"]["target"].astype(np.float64)
    residual22 = target22 - incumbent["full_2022"]
    early22 = axis_frames["full_2022"]["game_month"].le(7).to_numpy()
    late22 = axis_frames["full_2022"]["game_month"].ge(8).to_numpy()
    late22_axis = {
        key: np.asarray(value)[late22] for key, value in axes["full_2022"].items()
    }
    residual23 = axes["late_2023"]["target"].astype(np.float64) - incumbent["late_2023"]

    results: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    directions: dict[str, dict[str, np.ndarray]] = {}
    table_audit: dict[str, Any] = {}
    for family_name, family in FAMILIES.items():
        for scope in FIT_SCOPES:
            key_prefix = f"{family_name}__{scope}"
            direction22, audit22 = build_direction(
                axis_frames["full_2022"].loc[early22].reset_index(drop=True),
                residual22[early22], scopes["full_2022"][scope][early22],
                axis_frames["full_2022"].loc[late22].reset_index(drop=True), family,
            )
            direction23, audit23 = build_direction(
                axis_frames["full_2022"], residual22, scopes["full_2022"][scope],
                axis_frames["late_2023"], family,
            )
            source_rows24 = pd.concat(
                [axis_frames["full_2022"], axis_frames["late_2023"]],
                ignore_index=True,
            )
            source_residual24 = np.concatenate([residual22, residual23])
            source_scope24 = np.concatenate([
                scopes["full_2022"][scope], scopes["late_2023"][scope]
            ])
            direction24, audit24 = build_direction(
                source_rows24, source_residual24, source_scope24,
                axis_frames["full_2024"], family,
            )
            table_audit[key_prefix] = {
                "early22_to_late22": audit22,
                "full22_to_late23": audit23,
                "full22_late23_to_full24": audit24,
            }
            for eta in ETAS:
                key = f"{key_prefix}__e{eta:g}"
                candidate22 = apply_direction(
                    incumbent["full_2022"][late22], direction22,
                    active["full_2022"][late22], eta,
                )
                candidate23 = apply_direction(
                    incumbent["late_2023"], direction23, active["late_2023"], eta
                )
                candidate24 = apply_direction(
                    incumbent["full_2024"], direction24, active["full_2024"], eta
                )
                candidates[key] = {
                    "late_2022": candidate22,
                    "late_2023": candidate23,
                    "full_2024": candidate24,
                }
                directions[key] = {
                    "late_2022": direction22,
                    "late_2023": direction23,
                    "full_2024": direction24,
                }
                results[key] = {
                    "late_2022": metrics(
                        late22_axis, incumbent["full_2022"][late22], candidate22
                    ),
                    "late_2023": metrics(
                        axes["late_2023"], incumbent["late_2023"], candidate23
                    ),
                    "full_2024": metrics(
                        axes["full_2024"], incumbent["full_2024"], candidate24
                    ),
                }

    selected = select_candidate(results)
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
        family.append(incumbent["full_2024"])
        robustness = _robustness(
            axes["full_2024"], incumbent["full_2024"],
            candidates[selected]["full_2024"], active["full_2024"], family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] <= 0.10
        )
        np.savez_compressed(
            output_dir / "selected_axes.npz",
            incumbent_full_2024=incumbent["full_2024"],
            candidate_full_2024=candidates[selected]["full_2024"],
            direction_full_2024=directions[selected]["full_2024"],
            active_full_2024=active["full_2024"],
        )
    confirm = bool(selected is not None and point_pass and robust_pass)
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_release_build" if confirm else (
            "source_reject" if selected is None else (
                "locked_point_reject" if not point_pass else "robust_reject"
            )
        ),
        "candidate_count": len(results),
        "candidate_results": results,
        "selected_from_sources": selected,
        "selected_source": None if selected is None else {
            name: results[selected][name] for name in SOURCE_AXES
        },
        "locked_2024_incremental": locked,
        "locked_point_passed": point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "table_audit": table_audit,
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
        "selected": result["selected_from_sources"],
        "source": result["selected_source"],
        "locked": result["locked_2024_incremental"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
