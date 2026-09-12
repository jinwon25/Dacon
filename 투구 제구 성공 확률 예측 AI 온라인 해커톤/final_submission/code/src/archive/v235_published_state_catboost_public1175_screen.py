"""Research-only screen of newer published state-CatBoost OOF above Public1175.

External predictions are alignment-checked evidence only.  They are never
eligible for packaging.  A locally reimplemented model is justified only if a
source-selected small convex direction survives locked 2024.
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
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.archive.v224_fallback_xgb_complement_scope_audit import full_axis
from src.core.contract import _load_contract_axis


PROTOCOL = "V235_PUBLISHED_STATE_CATBOOST_PUBLIC1175_RESEARCH_SCREEN_V1"
FAMILIES = ("champ_oof", "champ_oof_x")
ROUTES = ("all_rcore", "public_active", "public_inactive_rcore")
WEIGHTS = (0.01, 0.025, 0.05, 0.075, 0.10)
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")


def route_masks(frame, public_active: np.ndarray) -> dict[str, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    rcore = regular & ~(
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    public_active = np.asarray(public_active, dtype=bool)
    return {
        "all_rcore": rcore,
        "public_active": public_active,
        "public_inactive_rcore": rcore & ~public_active,
    }


def convex_direction(
    base: np.ndarray,
    independent: np.ndarray,
    active: np.ndarray,
    weight: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    independent = np.asarray(independent, dtype=np.float64)
    if not np.isfinite(independent[active]).all():
        raise ValueError("external direction contains non-finite active prediction")
    output[active] = np.clip(
        (1.0 - float(weight)) * output[active]
        + float(weight) * independent[active],
        0.001,
        0.999,
    )
    return output


def load_family(path: Path, frames, late23: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        full = {}
        for year in (2022, 2023, 2024):
            value = saved[f"p{year}"].astype(np.float64)
            if len(value) != len(frames[year]):
                raise ValueError(
                    f"external OOF row mismatch: {path.name} {year} "
                    f"expected={len(frames[year])} actual={len(value)}"
                )
            full[year] = value
    return {
        "full_2022": full[2022],
        "late_2023": full[2023][late23],
        "full_2024": full[2024],
    }


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


def select_candidate(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, str]] = []
    for key, axes in results.items():
        if source_gate(axes):
            gains = [axes[name]["gain"] for name in SOURCE_AXES]
            eligible.append((min(gains), float(np.mean(gains)), key))
    return None if not eligible else max(eligible)[-1]


def restrictions() -> dict[str, bool]:
    return {
        "external_predictions_research_evidence_only": True,
        "external_code_or_weights_in_package": False,
        "alignment_checked_against_official_train_rows": True,
        "source_only_family_route_weight_selection": True,
        "locked_2024_not_used_for_selection": True,
        "local_reimplementation_required_before_packaging": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    fallback_oof_dir: Path,
    contract_dir: Path,
    bridge_oof: Path,
    external_exp_dir: Path,
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
    public1175: dict[str, np.ndarray] = {}
    public_active: dict[str, np.ndarray] = {}
    routes: dict[str, dict[str, np.ndarray]] = {}
    for name in AXES:
        public1175[name], public_active[name] = apply_fallback(
            parent[name], xgb[name], pressure_gate(axis_frames[name])
        )
        routes[name] = route_masks(axis_frames[name], public_active[name])
    independent = {
        family: load_family(external_exp_dir / f"{family}.npz", frames, late23)
        for family in FAMILIES
    }

    results: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        for route in ROUTES:
            for weight in WEIGHTS:
                key = f"{family}__{route}__w{weight:g}"
                results[key], candidates[key] = {}, {}
                for name in AXES:
                    candidate = convex_direction(
                        public1175[name], independent[family][name],
                        routes[name][route], weight,
                    )
                    candidates[key][name] = candidate
                    result = metrics(axes[name], public1175[name], candidate)
                    result["changed_rows"] = int(routes[name][route].sum())
                    results[key][name] = result
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
        family, route, _weight = selected.split("__")
        passing = [key for key in results if source_gate(results[key])]
        family_predictions = [candidates[key]["full_2024"] for key in passing]
        family_predictions.append(public1175["full_2024"])
        robustness = _robustness(
            axes["full_2024"], public1175["full_2024"],
            candidates[selected]["full_2024"], routes["full_2024"][route],
            family_predictions,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] <= 0.10
        )
    signal = bool(selected is not None and point_pass and robust_pass)
    summary = {
        "protocol": PROTOCOL,
        "status": "local_reimplementation_justified" if signal else (
            "source_reject" if selected is None else (
                "locked_point_reject" if not point_pass else "robust_reject"
            )
        ),
        "candidate_count": len(results),
        "candidate_results": results,
        "selected_from_sources": selected,
        "locked_2024_incremental": locked,
        "locked_point_passed": point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_local_reimplementation": signal,
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
    parser.add_argument("--external-exp-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.fallback_oof_dir, args.contract_dir,
        args.bridge_oof, args.external_exp_dir, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "selected": result["selected_from_sources"],
        "locked": result["locked_2024_incremental"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
