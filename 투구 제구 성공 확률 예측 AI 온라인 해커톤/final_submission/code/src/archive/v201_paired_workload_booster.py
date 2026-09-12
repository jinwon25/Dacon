"""Measure the marginal value of reconstructed workload with paired boosters.

For each audit year, two identically configured LightGBM classifiers are fit
on strictly earlier official seasons.  The baseline sees the official numeric
row features; the augmented twin additionally sees only v199's row-local
workload reconstruction.  Their prediction difference isolates the marginal
workload direction and is applied conservatively above the exact row-region parent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v199_recent_workload_reconstruction import (
    HORIZONS,
    MAX_DENOMINATOR,
    infer_denominators,
    rate_columns,
)
from src.archive.v200_recent_workload_direction_screen import apply_direction
from src.core.contract import _load_contract_axis


PROTOCOL = "V201_PAIRED_WORKLOAD_BOOSTER_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
AUDIT_YEARS = (2022, 2023, 2024)
SCALES = (0.05, 0.10, 0.20, 0.40)
EXCLUDED = {
    "row_id",
    "control_success",
    "season",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
}
MODEL_CONFIG = {
    "n_estimators": 260,
    "learning_rate": 0.035,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 1200,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 25.0,
    "random_state": 201,
    "n_jobs": 16,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def baseline_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Use numeric official inputs without imposing ordinal structure on IDs."""

    columns = [
        column
        for column in frame.columns
        if column not in EXCLUDED and pd.api.types.is_numeric_dtype(frame[column])
    ]
    output = frame.loc[:, columns].copy()
    output["top_bottom_t"] = frame["top_bottom"].astype(str).eq("T").astype(np.int8)
    output["game_type_regular"] = (
        frame["game_type"].astype(str).str.lower().isin(["regular", "정규시즌"])
    ).astype(np.int8)
    return output.astype(np.float32)


def workload_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build denominator-only additions; each output row uses only its input row."""

    output: dict[str, np.ndarray] = {}
    career_success = (
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
        .fillna(0.5)
        .to_numpy(np.float64)
    )
    career_middle = (
        pd.to_numeric(frame["asof_pitcher_middle_rate"], errors="coerce")
        .fillna(0.2)
        .to_numpy(np.float64)
    )
    denominators: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        success_column, middle_column = rate_columns(horizon)
        success = pd.to_numeric(frame[success_column], errors="coerce").to_numpy(
            np.float64
        )
        middle = pd.to_numeric(frame[middle_column], errors="coerce").to_numpy(
            np.float64
        )
        inferred = infer_denominators(
            pd.Series(success), pd.Series(middle),
            max_denominator=MAX_DENOMINATOR[horizon],
        )
        n = pd.to_numeric(
            inferred["minimum_denominator"], errors="coerce"
        ).to_numpy(np.float64)
        n = np.nan_to_num(n, nan=0.0)
        valid = (
            inferred["rounded_rational_fit"].fillna(False).to_numpy(bool)
            & np.isfinite(success)
            & np.isfinite(middle)
            & (n > 0.0)
        )
        denominators[horizon] = n
        output[f"workload_prev{horizon}_log_n"] = np.log1p(n)
        output[f"workload_prev{horizon}_missing"] = (~valid).astype(np.float64)
        reliability = n / (n + {1: 20.0, 3: 60.0, 5: 100.0}[horizon])
        output[f"workload_prev{horizon}_reliability"] = reliability
        output[f"workload_prev{horizon}_success_reliable_delta"] = np.where(
            valid, reliability * (success - career_success), 0.0
        )
        output[f"workload_prev{horizon}_middle_reliable_delta"] = np.where(
            valid, reliability * (middle - career_middle), 0.0
        )
    n1, n3, n5 = (denominators[horizon] for horizon in HORIZONS)
    output["workload_prev3_minus_prev1_log"] = np.log1p(
        np.maximum(n3 - n1, 0.0)
    )
    output["workload_prev5_minus_prev3_log"] = np.log1p(
        np.maximum(n5 - n3, 0.0)
    )
    output["workload_prev1_over_prev3"] = np.divide(
        n1, n3, out=np.zeros(len(frame)), where=n3 > 0.0
    )
    output["workload_prev3_over_prev5"] = np.divide(
        n3, n5, out=np.zeros(len(frame)), where=n5 > 0.0
    )
    return pd.DataFrame(output, index=frame.index, dtype=np.float32)


def make_model() -> LGBMClassifier:
    return LGBMClassifier(objective="binary", **MODEL_CONFIG)


def fit_paired_fold(
    baseline: pd.DataFrame,
    workload: pd.DataFrame,
    target: np.ndarray,
    season: np.ndarray,
    audit_year: int,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = output_dir / f"paired_delta_{audit_year}.npy"
    metadata_path = output_dir / f"paired_delta_{audit_year}.json"
    if checkpoint.exists() and metadata_path.exists():
        prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
        expected = int(np.sum(season == audit_year))
        if len(prediction) != expected:
            raise ValueError(f"checkpoint length mismatch: {audit_year}")
        return prediction, json.loads(metadata_path.read_text(encoding="utf-8"))

    fit = season < audit_year
    audit = season == audit_year
    weight = np.power(
        0.75,
        np.maximum((audit_year - 1) - season[fit], 0),
    )
    augmented = pd.concat([baseline, workload], axis=1)
    base_model = make_model()
    augmented_model = make_model()
    started = time.time()
    base_model.fit(baseline.loc[fit], target[fit], sample_weight=weight)
    base_fit_seconds = time.time() - started
    started = time.time()
    augmented_model.fit(augmented.loc[fit], target[fit], sample_weight=weight)
    augmented_fit_seconds = time.time() - started
    base_prediction = base_model.predict_proba(baseline.loc[audit])[:, 1]
    augmented_prediction = augmented_model.predict_proba(augmented.loc[audit])[:, 1]
    delta = np.asarray(augmented_prediction - base_prediction, dtype=np.float64)
    importance = dict(
        sorted(
            zip(augmented.columns, augmented_model.feature_importances_),
            key=lambda item: int(item[1]),
            reverse=True,
        )
    )
    metadata = {
        "audit_year": audit_year,
        "fit_rows": int(fit.sum()),
        "audit_rows": int(audit.sum()),
        "base_fit_seconds": base_fit_seconds,
        "augmented_fit_seconds": augmented_fit_seconds,
        "delta_mean_abs": float(np.mean(np.abs(delta))),
        "delta_p99_abs": float(np.quantile(np.abs(delta), 0.99)),
        "workload_importance": {
            name: int(importance.get(name, 0)) for name in workload.columns
        },
    }
    np.save(checkpoint, delta, allow_pickle=False)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    joblib.dump(base_model, output_dir / f"baseline_model_{audit_year}.joblib")
    joblib.dump(augmented_model, output_dir / f"augmented_model_{audit_year}.joblib")
    return delta, metadata


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.5
        and result["worst_month_gain"] > -2.0
        and result["minimum_domain_gain"] >= 0.0
    )


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "strictly_prior_season_model_fits": True,
        "paired_identical_booster_configuration": True,
        "source_scale_selection_before_locked_2024": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, low_memory=False)
    target = train["control_success"].to_numpy(np.int8)
    season = train["season"].to_numpy(np.int16)
    baseline = baseline_feature_frame(train)
    workload = workload_feature_frame(train)
    delta_by_year: dict[int, np.ndarray] = {}
    fold_metadata: dict[str, Any] = {}
    for year in AUDIT_YEARS:
        print(f"[v201] fitting paired fold {year}", flush=True)
        delta_by_year[year], fold_metadata[str(year)] = fit_paired_fold(
            baseline, workload, target, season, year, output_dir
        )

    _context, raw_frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, raw_frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    directions = {
        "full_2022": delta_by_year[2022],
        "late_2023": delta_by_year[2023][late23],
        "full_2024": delta_by_year[2024],
    }
    for axis in AXES:
        if len(directions[axis]) != len(parents[axis]):
            raise ValueError(f"direction alignment mismatch: {axis}")

    rows: list[dict[str, Any]] = []
    detail: dict[str, dict[str, Any]] = {}
    candidates: dict[float, dict[str, np.ndarray]] = {}
    active_masks: dict[float, dict[str, np.ndarray]] = {}
    for scale in SCALES:
        detail[str(scale)] = {}
        candidates[scale] = {}
        active_masks[scale] = {}
        row: dict[str, Any] = {"scale": scale}
        for axis in AXES:
            candidate, active = apply_direction(
                parents[axis], directions[axis], axes[axis]["exact_mask"],
                axes[axis]["domain3"], scale,
            )
            candidates[scale][axis] = candidate
            active_masks[scale][axis] = active
            score = metrics(axes[axis], parents[axis], candidate)
            detail[str(scale)][axis] = score
            row[f"{axis}_gain"] = score["gain"]
            row[f"{axis}_month_fraction"] = score["positive_month_fraction"]
            row[f"{axis}_worst_month"] = score["worst_month_gain"]
        row["source_gate_passed"] = all(
            source_gate(detail[str(scale)][axis]) for axis in SOURCE_AXES
        )
        row["source_min_gain"] = min(
            detail[str(scale)][axis]["gain"] for axis in SOURCE_AXES
        )
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain"],
        ascending=[False, False], kind="stable",
    )
    table.to_csv(output_dir / "source_scale_screen.csv", index=False, encoding="utf-8-sig")
    passing = table.loc[table["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "model_config": MODEL_CONFIG,
            "folds": fold_metadata,
            "scale_screen": table.to_dict(orient="records"),
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected_scale = float(passing.iloc[0]["scale"])
        candidate = candidates[selected_scale]["full_2024"]
        active = active_masks[selected_scale]["full_2024"]
        family = [
            candidates[float(scale)]["full_2024"] for scale in passing["scale"]
        ]
        family.append(parents["full_2024"].copy())
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], candidate, active, family
        )
        locked = detail[str(selected_scale)]["full_2024"]
        point_pass = bool(
            locked["gain"] > 0.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -2.0
            and locked["minimum_domain_gain"] >= 0.0
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"], candidate=candidate,
            direction=directions["full_2024"], active=active,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "model_config": MODEL_CONFIG,
            "folds": fold_metadata,
            "selected_scale": selected_scale,
            "source": {
                axis: detail[str(selected_scale)][axis] for axis in SOURCE_AXES
            },
            "locked_2024": locked,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_packaging": bool(point_pass and robust_pass),
            "scale_screen": table.to_dict(orient="records"),
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
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
