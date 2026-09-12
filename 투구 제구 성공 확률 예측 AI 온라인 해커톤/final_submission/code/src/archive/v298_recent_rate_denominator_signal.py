"""Audit hidden recent-game sample size encoded in the official ASOF rates.

The official rows expose previous 1/3/5-game success and middle rates, but not
the number of pitches behind those rates.  Both rates share a denominator.  A
joint rational reconstruction therefore supplies a row-local lower bound on
recent workload and on the reliability of the recent-form rates.  No row
order, other evaluation row, external data, or current label is used to build
the features.

The correction is deliberately low capacity.  A ridge residual model is fit
on early 2022 and transferred to late 2022, then fit on full 2022 and
transferred to late 2023.  A single dose is selected from those two source
transfers.  Full 2024 is opened once after the dose is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.contract import _load_contract_axis


PROTOCOL = "V298_RECENT_RATE_DENOMINATOR_SIGNAL_V1"
TARGET = "control_success"
WINDOWS = (1, 3, 5)
MAX_DENOMINATOR = {1: 220, 3: 550, 5: 850}
RIDGE_ALPHA = 1000.0
CORRECTION_CAP = 0.03


def bss(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(y.mean() * (1.0 - y.mean()))
    return float(100000.0 * (1.0 - np.mean(np.square(y - prediction)) / denominator))


def decode_shared_denominator(
    success_rate: np.ndarray,
    middle_rate: np.ndarray,
    max_denominator: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return minimum compatible denominator, ambiguity count, and fit error.

    Official recent-game rates are rounded to six decimal places.  For a
    candidate denominator ``n``, both ``n * rate`` values must therefore lie
    within the rounding interval of an integer.  The smallest compatible
    denominator is a conservative workload lower bound.  The number of
    compatible denominators records ambiguity (for example 0.5 and 0.2).
    """

    success = np.asarray(success_rate, dtype=np.float64)
    middle = np.asarray(middle_rate, dtype=np.float64)
    if success.shape != middle.shape:
        raise ValueError("recent-rate arrays must have identical shapes")
    result = np.full(success.shape, np.nan, dtype=np.float64)
    ambiguity = np.full(success.shape, np.nan, dtype=np.float64)
    error = np.full(success.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(success)
        & np.isfinite(middle)
        & (success >= 0.0)
        & (success <= 1.0)
        & (middle >= 0.0)
        & (middle <= 1.0)
    )
    if not np.any(valid):
        return result, ambiguity, error

    pairs, inverse = np.unique(
        np.column_stack((success[valid], middle[valid])), axis=0, return_inverse=True
    )
    denominators = np.arange(1, int(max_denominator) + 1, dtype=np.float64)
    decoded = np.empty(len(pairs), dtype=np.float64)
    counts = np.empty(len(pairs), dtype=np.float64)
    errors = np.empty(len(pairs), dtype=np.float64)
    for index, (rate_success, rate_middle) in enumerate(pairs):
        success_error = np.abs(denominators * rate_success - np.rint(denominators * rate_success))
        middle_error = np.abs(denominators * rate_middle - np.rint(denominators * rate_middle))
        joint_error = np.maximum(success_error, middle_error)
        tolerance = 5.05e-7 * denominators + 1e-9
        compatible = np.flatnonzero(joint_error <= tolerance)
        if len(compatible):
            decoded[index] = denominators[compatible[0]]
            counts[index] = float(len(compatible))
            errors[index] = float(joint_error[compatible[0]])
        else:
            best = int(np.argmin(joint_error / np.sqrt(denominators)))
            decoded[index] = denominators[best]
            counts[index] = 0.0
            errors[index] = float(joint_error[best])
    result[valid] = decoded[inverse]
    ambiguity[valid] = counts[inverse]
    error[valid] = errors[inverse]
    return result, ambiguity, error


def build_denominator_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic row-local workload/reliability features."""

    output = pd.DataFrame(index=np.arange(len(frame)))
    career_success = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    career_middle = pd.to_numeric(
        frame["asof_pitcher_middle_rate"], errors="coerce"
    ).to_numpy(np.float64)
    decoded: dict[int, np.ndarray] = {}
    recent_success: dict[int, np.ndarray] = {}
    recent_middle: dict[int, np.ndarray] = {}

    for window in WINDOWS:
        success = pd.to_numeric(
            frame[f"asof_pitcher_prev{window}_game_success_rate"], errors="coerce"
        ).to_numpy(np.float64)
        middle = pd.to_numeric(
            frame[f"asof_pitcher_prev{window}_game_middle_rate"], errors="coerce"
        ).to_numpy(np.float64)
        denominator, ambiguity, error = decode_shared_denominator(
            success, middle, MAX_DENOMINATOR[window]
        )
        decoded[window] = denominator
        recent_success[window] = success
        recent_middle[window] = middle
        per_game = denominator / float(window)
        log_n = np.log1p(denominator)
        success_delta = success - career_success
        middle_delta = middle - career_middle
        output[f"den_prev{window}_log_n"] = log_n
        output[f"den_prev{window}_log_ambiguity"] = np.log1p(ambiguity)
        output[f"den_prev{window}_fit_error"] = error
        output[f"den_prev{window}_per_game"] = per_game
        output[f"den_prev{window}_success_delta"] = success_delta
        output[f"den_prev{window}_middle_delta"] = middle_delta
        output[f"den_prev{window}_success_delta_x_log_n"] = success_delta * log_n
        output[f"den_prev{window}_middle_delta_x_log_n"] = middle_delta * log_n
        for threshold in (20.0, 50.0, 80.0):
            output[f"den_prev{window}_per_game_ge{int(threshold)}"] = (
                per_game >= threshold
            ).astype(np.float64)

    output["den_n1_over_n3"] = decoded[1] / np.clip(decoded[3], 1.0, None)
    output["den_n3_over_n5"] = decoded[3] / np.clip(decoded[5], 1.0, None)
    output["den_workload_slope_1_5"] = decoded[1] - decoded[5] / 5.0
    output["den_success_1_minus_3"] = recent_success[1] - recent_success[3]
    output["den_success_3_minus_5"] = recent_success[3] - recent_success[5]
    output["den_middle_1_minus_3"] = recent_middle[1] - recent_middle[3]
    output["den_middle_3_minus_5"] = recent_middle[3] - recent_middle[5]
    output["den_success_spread_x_log_n1"] = (
        output["den_success_1_minus_3"].to_numpy(np.float64)
        * np.log1p(decoded[1])
    )
    output["den_middle_spread_x_log_n1"] = (
        output["den_middle_1_minus_3"].to_numpy(np.float64)
        * np.log1p(decoded[1])
    )

    inning = pd.to_numeric(frame["inning"], errors="coerce").to_numpy(np.float64)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").to_numpy(np.float64)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").to_numpy(np.float64)
    output["den_prev1_log_n_x_late_inning"] = np.log1p(decoded[1]) * (inning >= 7.0)
    output["den_prev1_log_n_x_two_strike"] = np.log1p(decoded[1]) * (strikes >= 2.0)
    output["den_prev1_log_n_x_three_ball"] = np.log1p(decoded[1]) * (balls >= 3.0)
    output["den_prev1_success_delta_x_two_strike"] = (
        output["den_prev1_success_delta"].to_numpy(np.float64) * (strikes >= 2.0)
    )
    return output.astype(np.float64)


def fit_residual_model(features: pd.DataFrame, residual: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)),
        ]
    )
    centered = np.asarray(residual, dtype=np.float64) - float(np.mean(residual))
    model.fit(features, centered)
    return model


def metrics(
    frame: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    months = []
    for month in sorted(pd.unique(frame["game_month"])):
        mask = frame["game_month"].eq(month).to_numpy() & active
        if mask.any():
            months.append(
                {
                    "month": int(month),
                    "rows": int(mask.sum()),
                    "gain": bss(target[mask], candidate[mask]) - bss(target[mask], parent[mask]),
                }
            )
    active_gain = (
        bss(target[active], candidate[active]) - bss(target[active], parent[active])
        if active.any()
        else 0.0
    )
    return {
        "gain": bss(target, candidate) - bss(target, parent),
        "active_gain": active_gain,
        "active_rows": int(active.sum()),
        "mean_abs_shift_active": float(np.mean(np.abs(candidate[active] - parent[active]))),
        "rms_shift_active": float(np.sqrt(np.mean(np.square(candidate[active] - parent[active])))),
        "positive_month_fraction": float(np.mean([row["gain"] > 0.0 for row in months])),
        "worst_month_gain": float(min(row["gain"] for row in months)),
        "months": months,
    }


def run(
    train_csv: Path,
    contract_dir: Path,
    exact_anchor_axes: Path,
    v290_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "inning", "balls_before", "strikes_before",
        "asof_pitcher_success_rate", "asof_pitcher_middle_rate", TARGET,
        *[
            f"asof_pitcher_prev{window}_game_{rate}_rate"
            for window in WINDOWS
            for rate in ("success", "middle")
        ],
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in frames
    }
    with np.load(exact_anchor_axes, allow_pickle=False) as saved:
        parent = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v290_axes, allow_pickle=False) as saved:
        parent.update(
            {
                "late_2023": saved["candidate_late_2023"].astype(np.float64),
                "full_2024": saved["candidate_full_2024"].astype(np.float64),
            }
        )
    features = {name: build_denominator_features(frame) for name, frame in frames.items()}
    targets = {name: frame[TARGET].to_numpy(np.float64) for name, frame in frames.items()}
    active = {
        name: axes[name]["exact_mask"].astype(bool)
        & (axes[name]["domain3"].astype(str) == "R_CORE")
        for name in frames
    }
    for name in frames:
        if len(parent[name]) != len(frames[name]) or len(axes[name]["target"]) != len(frames[name]):
            raise ValueError(f"axis length mismatch: {name}")
        if not np.array_equal(targets[name], axes[name]["target"].astype(np.float64)):
            raise ValueError(f"target order mismatch: {name}")

    frame22 = frames["full_2022"]
    early22 = frame22["game_month"].le(7).to_numpy() & active["full_2022"]
    late22 = frame22["game_month"].ge(8).to_numpy() & active["full_2022"]
    model_early22 = fit_residual_model(
        features["full_2022"].loc[early22],
        targets["full_2022"][early22] - parent["full_2022"][early22],
    )
    correction_late22 = np.clip(
        model_early22.predict(features["full_2022"].loc[late22]),
        -CORRECTION_CAP,
        CORRECTION_CAP,
    )
    model22 = fit_residual_model(
        features["full_2022"].loc[active["full_2022"]],
        targets["full_2022"][active["full_2022"]] - parent["full_2022"][active["full_2022"]],
    )
    correction23 = np.clip(
        model22.predict(features["late_2023"]), -CORRECTION_CAP, CORRECTION_CAP
    )

    residual_late22 = targets["full_2022"][late22] - parent["full_2022"][late22]
    residual23 = targets["late_2023"][active["late_2023"]] - parent["late_2023"][active["late_2023"]]
    direction23 = correction23[active["late_2023"]]
    numerator = float(np.dot(residual_late22, correction_late22) + np.dot(residual23, direction23))
    denominator = float(np.dot(correction_late22, correction_late22) + np.dot(direction23, direction23))
    dose = float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator > 0.0 else 0.0

    candidate_late22 = parent["full_2022"][late22] + dose * correction_late22
    candidate23 = parent["late_2023"].copy()
    candidate23[active["late_2023"]] = np.clip(
        candidate23[active["late_2023"]] + dose * direction23, 0.001, 0.999
    )
    source_metrics = {
        "early_2022_to_late_2022": metrics(
            frame22.loc[late22].reset_index(drop=True),
            targets["full_2022"][late22],
            parent["full_2022"][late22],
            candidate_late22,
            np.ones(int(late22.sum()), dtype=bool),
        ),
        "full_2022_to_late_2023": metrics(
            frames["late_2023"], targets["late_2023"], parent["late_2023"],
            candidate23, active["late_2023"],
        ),
    }

    combined_features = pd.concat(
        [
            features["full_2022"].loc[active["full_2022"]],
            features["late_2023"].loc[active["late_2023"]],
        ],
        ignore_index=True,
    )
    combined_residual = np.concatenate(
        [
            targets["full_2022"][active["full_2022"]] - parent["full_2022"][active["full_2022"]],
            targets["late_2023"][active["late_2023"]] - parent["late_2023"][active["late_2023"]],
        ]
    )
    model_pre24 = fit_residual_model(combined_features, combined_residual)
    correction24 = np.clip(
        model_pre24.predict(features["full_2024"]), -CORRECTION_CAP, CORRECTION_CAP
    )
    candidate24 = parent["full_2024"].copy()
    candidate24[active["full_2024"]] = np.clip(
        candidate24[active["full_2024"]]
        + dose * correction24[active["full_2024"]],
        0.001,
        0.999,
    )
    locked_metrics = metrics(
        frames["full_2024"], targets["full_2024"], parent["full_2024"],
        candidate24, active["full_2024"],
    )
    source_pass = bool(
        dose > 0.0
        and all(item["gain"] > 0.0 for item in source_metrics.values())
        and all(item["positive_month_fraction"] >= 2.0 / 3.0 for item in source_metrics.values())
    )
    locked_pass = bool(
        locked_metrics["gain"] > 0.0
        and locked_metrics["positive_month_fraction"] >= 0.75
        and locked_metrics["worst_month_gain"] > -5.0
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if source_pass and locked_pass else "reject",
        "feature_count": int(features["full_2022"].shape[1]),
        "ridge_alpha": RIDGE_ALPHA,
        "correction_cap": CORRECTION_CAP,
        "source_selected_dose": dose,
        "source_metrics": source_metrics,
        "locked_full_2024": locked_metrics,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "test_csv_read": False,
            "row_id_or_csv_order_used": False,
            "other_evaluation_rows_used": False,
            "full_2024_used_for_dose_selection": False,
            "public_score_used_for_selection": False,
            "row_local_features": True,
        },
    }
    np.savez_compressed(
        output_dir / "audit_axes.npz",
        candidate_late_2022=candidate_late22,
        candidate_late_2023=candidate23,
        candidate_full_2024=candidate24,
        correction_full_2024=correction24,
        active_full_2024=active["full_2024"],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--exact-anchor-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.train_csv,
                args.contract_dir,
                args.exact_anchor_axes,
                args.v290_axes,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
