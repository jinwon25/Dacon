"""Forward audit of pitcher-specific fallback utility weights.

For each pitcher, prior OOF rows estimate the Brier-optimal XGB blend weight.
That noisy estimate is clipped and shrunk toward the deployed 0.30 weight by
prior row count.  Shrink strength is selected on early-2022 -> late-2022 and
full-2022 -> late-2023, then evaluated once on locked full-2024.
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
    XGB_WEIGHT,
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.archive.v227_fallback_forward_calibration_audit import slice_axis
from src.archive.v224_fallback_xgb_complement_scope_audit import full_axis
from src.core.contract import _load_contract_axis


PROTOCOL = "V233_PITCHER_FALLBACK_UTILITY_FORWARD_AUDIT_V1"
SHRINK_ROWS = (50.0, 100.0, 200.0, 500.0)
MAX_WEIGHT = 0.60
AXES = ("full_2022", "late_2023", "full_2024")


def fit_pitcher_weights(
    pitcher_id: np.ndarray,
    target: np.ndarray,
    parent: np.ndarray,
    xgb_prediction: np.ndarray,
    active: np.ndarray,
    shrink_rows: float,
) -> dict[int, float]:
    active = np.asarray(active, dtype=bool)
    frame = pd.DataFrame({
        "pitcher_id": np.asarray(pitcher_id)[active].astype(np.int64),
        "direction": (
            np.asarray(xgb_prediction, dtype=np.float64)[active]
            - np.asarray(parent, dtype=np.float64)[active]
        ),
        "residual": (
            np.asarray(target, dtype=np.float64)[active]
            - np.asarray(parent, dtype=np.float64)[active]
        ),
    })
    if frame.empty or not np.isfinite(frame[["direction", "residual"]]).all().all():
        raise ValueError("pitcher utility fit support is empty or non-finite")
    frame["numerator"] = frame["direction"] * frame["residual"]
    frame["denominator"] = np.square(frame["direction"])
    grouped = frame.groupby("pitcher_id", sort=False).agg(
        numerator=("numerator", "sum"),
        denominator=("denominator", "sum"),
        rows=("direction", "size"),
    )
    raw = grouped["numerator"] / grouped["denominator"].replace(0.0, np.nan)
    raw = raw.clip(0.0, MAX_WEIGHT).fillna(XGB_WEIGHT)
    reliability = grouped["rows"] / (grouped["rows"] + float(shrink_rows))
    weight = XGB_WEIGHT + reliability * (raw - XGB_WEIGHT)
    return {int(key): float(value) for key, value in weight.items()}


def apply_pitcher_weights(
    pitcher_id: np.ndarray,
    parent: np.ndarray,
    xgb_prediction: np.ndarray,
    active: np.ndarray,
    weight_map: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(active, dtype=bool)
    weight = (
        pd.Series(np.asarray(pitcher_id, dtype=np.int64))
        .map(weight_map)
        .fillna(XGB_WEIGHT)
        .to_numpy(np.float64)
    )
    output = np.asarray(parent, dtype=np.float64).copy()
    direction = (
        np.asarray(xgb_prediction, dtype=np.float64)
        - np.asarray(parent, dtype=np.float64)
    )
    if not np.isfinite(direction[active]).all():
        raise ValueError("dynamic fallback has missing active XGB prediction")
    output[active] = np.clip(
        output[active] + weight[active] * direction[active], 0.001, 0.999
    )
    return output, weight


def source_gate(results: dict[str, dict[str, Any]]) -> bool:
    return bool(
        all(results[name]["gain"] > 0.0 for name in ("late_2022", "late_2023"))
        and results["late_2022"]["positive_month_fraction"] >= (2.0 / 3.0)
        and results["late_2023"]["positive_month_fraction"] >= (2.0 / 3.0)
        and results["late_2022"]["worst_month_gain"] > -5.0
        and results["late_2023"]["worst_month_gain"] > -5.0
        and results["late_2022"]["minimum_domain_gain"] >= 0.0
        and results["late_2023"]["minimum_domain_gain"] >= 0.0
    )


def select_shrink(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, float, str]] = []
    for key, axes in results.items():
        if source_gate(axes):
            gains = [axes[name]["gain"] for name in ("late_2022", "late_2023")]
            shrink = float(key)
            eligible.append((min(gains), float(np.mean(gains)), shrink, key))
    return None if not eligible else max(eligible)[-1]


def weight_statistics(weight: np.ndarray, active: np.ndarray) -> dict[str, float]:
    values = np.asarray(weight, dtype=np.float64)[np.asarray(active, dtype=bool)]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
        "changed_fraction": float(np.mean(np.abs(values - XGB_WEIGHT) > 1e-12)),
    }


def restrictions() -> dict[str, bool]:
    return {
        "fallback_xgb_model_frozen": True,
        "public1175_gate_and_threshold_frozen": True,
        "pitcher_utility_uses_only_prior_oof_outcomes": True,
        "unseen_pitchers_default_to_public1175_weight": True,
        "source_only_shrink_selection": True,
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
    for name in AXES:
        current[name], active[name] = apply_fallback(
            parent[name], xgb[name], pressure[name]
        )

    late22 = frames[2022]["game_month"].ge(8).to_numpy()
    early22 = ~late22
    late22_axis = slice_axis(axes["full_2022"], late22)
    source_results: dict[str, dict[str, Any]] = {}
    locked_candidates: dict[str, np.ndarray] = {}
    locked_weights: dict[str, np.ndarray] = {}
    map_sizes: dict[str, dict[str, int]] = {}
    for shrink in SHRINK_ROWS:
        key = f"{shrink:g}"
        early_map = fit_pitcher_weights(
            axes["full_2022"]["pitcher_id"], axes["full_2022"]["target"],
            parent["full_2022"], xgb["full_2022"],
            active["full_2022"] & early22, shrink,
        )
        full22_map = fit_pitcher_weights(
            axes["full_2022"]["pitcher_id"], axes["full_2022"]["target"],
            parent["full_2022"], xgb["full_2022"], active["full_2022"], shrink,
        )
        pooled_map = fit_pitcher_weights(
            np.concatenate([
                axes["full_2022"]["pitcher_id"], axes["late_2023"]["pitcher_id"]
            ]),
            np.concatenate([
                axes["full_2022"]["target"], axes["late_2023"]["target"]
            ]),
            np.concatenate([parent["full_2022"], parent["late_2023"]]),
            np.concatenate([xgb["full_2022"], xgb["late_2023"]]),
            np.concatenate([active["full_2022"], active["late_2023"]]),
            shrink,
        )
        late22_full, _ = apply_pitcher_weights(
            axes["full_2022"]["pitcher_id"], parent["full_2022"],
            xgb["full_2022"], active["full_2022"], early_map,
        )
        late23_candidate, _ = apply_pitcher_weights(
            axes["late_2023"]["pitcher_id"], parent["late_2023"],
            xgb["late_2023"], active["late_2023"], full22_map,
        )
        locked_candidate, locked_weight = apply_pitcher_weights(
            axes["full_2024"]["pitcher_id"], parent["full_2024"],
            xgb["full_2024"], active["full_2024"], pooled_map,
        )
        source_results[key] = {
            "late_2022": metrics(
                late22_axis, current["full_2022"][late22], late22_full[late22]
            ),
            "late_2023": metrics(
                axes["late_2023"], current["late_2023"], late23_candidate
            ),
        }
        locked_candidates[key] = locked_candidate
        locked_weights[key] = locked_weight
        map_sizes[key] = {
            "early_2022": len(early_map),
            "full_2022": len(full22_map),
            "pooled_2022_late2023": len(pooled_map),
        }

    selected = select_shrink(source_results)
    locked = None
    point_pass = False
    robustness = None
    robust_pass = False
    selected_weight_stats = None
    if selected is not None:
        locked = metrics(
            axes["full_2024"], current["full_2024"], locked_candidates[selected]
        )
        selected_weight_stats = weight_statistics(
            locked_weights[selected], active["full_2024"]
        )
        point_pass = bool(
            locked["gain"] >= 1.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -5.0
            and locked["minimum_domain_gain"] >= 0.0
        )
        passing = [key for key in source_results if source_gate(source_results[key])]
        family = [locked_candidates[key] for key in passing]
        family.append(current["full_2024"])
        robustness = _robustness(
            axes["full_2024"], current["full_2024"],
            locked_candidates[selected], active["full_2024"], family,
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
            candidate_full_2024=locked_candidates[selected],
            active_full_2024=active["full_2024"],
            dynamic_weight_full_2024=locked_weights[selected],
        )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_release_fit" if confirm else (
            "source_reject" if selected is None else (
                "locked_point_reject" if not point_pass else "robust_reject"
            )
        ),
        "source_results": source_results,
        "pitcher_map_sizes": map_sizes,
        "selected_shrink_rows_from_sources": selected,
        "locked_2024_incremental": locked,
        "locked_weight_statistics": selected_weight_stats,
        "locked_point_passed": point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_release_fit": confirm,
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
        "selected": result["selected_shrink_rows_from_sources"],
        "source": result["source_results"],
        "locked": result["locked_2024_incremental"],
        "weights": result["locked_weight_statistics"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
