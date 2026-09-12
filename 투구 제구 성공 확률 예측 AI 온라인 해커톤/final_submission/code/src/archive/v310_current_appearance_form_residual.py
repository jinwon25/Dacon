"""Strict-forward residual screen for reconstructed current-appearance form."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.archive.v298_recent_rate_denominator_signal import metrics
from src.archive.v309_current_appearance_command_audit import (
    attach_appearance_success,
    build_current_appearance_features,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V310_CURRENT_APPEARANCE_FORM_RESIDUAL_V1"
TARGET = "control_success"
RIDGE_ALPHA = 1500.0
CORRECTION_CAP = 0.025


def form_residual_features(
    appearance: pd.DataFrame, frame: pd.DataFrame
) -> pd.DataFrame:
    z = np.column_stack([
        appearance[f"appearance_prev{window}_excess_z"].to_numpy(np.float64)
        for window in (1, 3, 5)
    ])
    finite = np.isfinite(z)
    count = finite.sum(axis=1)
    z_sum = np.where(finite, z, 0.0).sum(axis=1)
    z_mean = np.divide(
        z_sum, count, out=np.full(len(z), np.nan, dtype=np.float64), where=count > 0
    )
    centered = np.where(finite, z - z_mean[:, None], 0.0)
    z_spread = np.sqrt(np.divide(
        np.square(centered).sum(axis=1),
        count,
        out=np.full(len(z), np.nan, dtype=np.float64),
        where=count > 0,
    ))
    age = appearance["appearance_expected_age"].to_numpy(np.float64)
    age_sd = appearance["appearance_age_sd"].to_numpy(np.float64)
    typical = appearance["appearance_typical_length"].to_numpy(np.float64)
    prior = appearance["appearance_pitcher_prior"].to_numpy(np.float64)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").to_numpy(np.float64)
    strikes = pd.to_numeric(
        frame["strikes_before"], errors="coerce"
    ).to_numpy(np.float64)
    inning = pd.to_numeric(frame["inning"], errors="coerce").to_numpy(np.float64)
    reliable = age / np.clip(age + 15.0, 1.0, None)
    long_role = np.clip((typical - 15.0) / 45.0, 0.0, 1.0)
    two_strike = (strikes >= 2.0).astype(np.float64)
    three_ball = (balls >= 3.0).astype(np.float64)
    behind = (balls > strikes).astype(np.float64)
    rate_deltas = [
        appearance[f"appearance_prev{window}_rate_k15"].to_numpy(np.float64)
        - prior
        for window in (1, 3, 5)
    ]
    return pd.DataFrame(
        {
            "form_z_prev1": z[:, 0],
            "form_z_prev3": z[:, 1],
            "form_z_prev5": z[:, 2],
            "form_z_mean": z_mean,
            "form_z_spread": z_spread,
            "form_rate_delta_prev1": rate_deltas[0],
            "form_rate_delta_prev3": rate_deltas[1],
            "form_rate_delta_prev5": rate_deltas[2],
            "form_age": age / 100.0,
            "form_age_sd": age_sd / 100.0,
            "form_reliability": reliable,
            "form_long_role": long_role,
            "form_z_x_reliability": z_mean * reliable,
            "form_z_x_long_role": z_mean * long_role,
            "form_z_x_two_strike": z_mean * reliable * two_strike,
            "form_z_x_three_ball": z_mean * reliable * three_ball,
            "form_z_x_behind": z_mean * reliable * behind,
            "form_z_x_late_inning": z_mean * reliable * (inning >= 7.0),
            "form_success_est_spread": appearance[
                "appearance_success_est_spread"
            ].to_numpy(np.float64),
            "form_relative_uncertainty": appearance[
                "appearance_age_relative_uncertainty"
            ].to_numpy(np.float64),
            "form_covered": appearance[
                "appearance_renewal_covered"
            ].to_numpy(np.float64),
        }
    )


def fit_model(features: pd.DataFrame, residual: np.ndarray) -> tuple[Pipeline, float]:
    target = np.asarray(residual, dtype=np.float64)
    target -= float(target.mean())
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)),
    ])
    model.fit(features, target)
    center = float(np.mean(model.predict(features)))
    return model, center


def predict_direction(model: Pipeline, center: float, features: pd.DataFrame) -> np.ndarray:
    return np.clip(
        np.asarray(model.predict(features), dtype=np.float64) - center,
        -CORRECTION_CAP,
        CORRECTION_CAP,
    )


def pooled_dose(parts: list[tuple[np.ndarray, np.ndarray]]) -> float:
    numerator = float(sum(np.dot(residual, direction) for residual, direction in parts))
    denominator = float(sum(np.dot(direction, direction) for _, direction in parts))
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0


def _parents(v285_axes: Path, v290_axes: Path) -> dict[str, np.ndarray]:
    with np.load(v285_axes, allow_pickle=False) as saved:
        parent = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v290_axes, allow_pickle=False) as saved:
        parent.update({
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
            "full_2024": saved["candidate_full_2024"].astype(np.float64),
        })
    return parent


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "evaluation_row_id_or_order_used": False,
        "other_evaluation_rows_required": False,
        "runtime_live_features_are_row_local": True,
        "strictly_prior_season_anchors_and_renewal_prior": True,
        "source_dose_frozen_before_full_2024": True,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    contract_dir: Path,
    v285_axes: Path,
    v290_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "row_id", "season", "game_month", "game_type", "inning", "top_bottom",
        "balls_before", "strikes_before", "pitcher_id", "asof_pitcher_n",
        "asof_pitcher_success_rate", TARGET,
        *[
            f"asof_pitcher_prev{window}_game_{rate}_rate"
            for window in (1, 3, 5) for rate in ("success", "middle")
        ],
    ]
    raw = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    rows, appearances = attach_appearance_success(raw)
    parent = _parents(v285_axes, v290_axes)
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    frames: dict[str, pd.DataFrame] = {}
    feature_bank: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    for year, name in ((2022, "full_2022"), (2023, "late_2023"), (2024, "full_2024")):
        year_rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        regular = year_rows["game_type"].astype(str).eq("R").to_numpy()
        regular_frame = year_rows.loc[regular].reset_index(drop=True)
        appearance = build_current_appearance_features(rows, appearances, year)
        features = form_residual_features(appearance, regular_frame)
        axis_frame = year_rows
        if name == "late_2023":
            keep = regular_frame["game_month"].ge(8).to_numpy()
            regular_frame = regular_frame.loc[keep].reset_index(drop=True)
            features = features.loc[keep].reset_index(drop=True)
            axis_frame = year_rows.loc[year_rows["game_month"].ge(8)].reset_index(drop=True)
        frames[name] = regular_frame
        feature_bank[name] = features
        if len(axis_frame) != len(axes[name]["target"]):
            raise ValueError(f"raw-to-contract mismatch: {name}")
        axis_regular = axis_frame["game_type"].astype(str).eq("R").to_numpy()
        targets[name] = np.asarray(axes[name]["target"], dtype=np.float64)[axis_regular]
        active[name] = (
            np.asarray(axes[name]["exact_mask"], dtype=bool)[axis_regular]
            & (np.asarray(axes[name]["domain3"]).astype(str)[axis_regular] == "R_CORE")
            & features["form_covered"].to_numpy(bool)
        )
        parent[name] = parent[name][axis_regular]
        if not (
            len(frames[name]) == len(parent[name]) == len(targets[name]) == len(active[name])
        ):
            raise ValueError(f"axis length mismatch: {name}")
        if not np.array_equal(frames[name][TARGET].to_numpy(np.float64), targets[name]):
            raise ValueError(f"target order mismatch: {name}")

    frame22 = frames["full_2022"]
    early22 = frame22["game_month"].le(7).to_numpy() & active["full_2022"]
    late22 = frame22["game_month"].ge(8).to_numpy() & active["full_2022"]
    early_model, early_center = fit_model(
        feature_bank["full_2022"].loc[early22],
        targets["full_2022"][early22] - parent["full_2022"][early22],
    )
    direction22 = predict_direction(
        early_model, early_center, feature_bank["full_2022"].loc[late22]
    )
    model22, center22 = fit_model(
        feature_bank["full_2022"].loc[active["full_2022"]],
        targets["full_2022"][active["full_2022"]]
        - parent["full_2022"][active["full_2022"]],
    )
    direction23_all = predict_direction(model22, center22, feature_bank["late_2023"])
    direction23 = direction23_all[active["late_2023"]]
    dose = pooled_dose([
        (targets["full_2022"][late22] - parent["full_2022"][late22], direction22),
        (
            targets["late_2023"][active["late_2023"]]
            - parent["late_2023"][active["late_2023"]],
            direction23,
        ),
    ])
    candidate22 = np.clip(parent["full_2022"][late22] + dose * direction22, 0.001, 0.999)
    candidate23 = parent["late_2023"].copy()
    candidate23[active["late_2023"]] = np.clip(
        candidate23[active["late_2023"]] + dose * direction23, 0.001, 0.999
    )
    source = {
        "early_2022_to_late_2022": metrics(
            frame22.loc[late22].reset_index(drop=True), targets["full_2022"][late22],
            parent["full_2022"][late22], candidate22,
            np.ones(int(late22.sum()), dtype=bool),
        ),
        "full_2022_to_late_2023": metrics(
            frames["late_2023"], targets["late_2023"], parent["late_2023"],
            candidate23, active["late_2023"],
        ),
    }
    combined_features = pd.concat([
        feature_bank["full_2022"].loc[active["full_2022"]],
        feature_bank["late_2023"].loc[active["late_2023"]],
    ], ignore_index=True)
    combined_residual = np.concatenate([
        targets["full_2022"][active["full_2022"]]
        - parent["full_2022"][active["full_2022"]],
        targets["late_2023"][active["late_2023"]]
        - parent["late_2023"][active["late_2023"]],
    ])
    final_model, final_center = fit_model(combined_features, combined_residual)
    direction24 = predict_direction(final_model, final_center, feature_bank["full_2024"])
    candidate24 = parent["full_2024"].copy()
    candidate24[active["full_2024"]] = np.clip(
        candidate24[active["full_2024"]] + dose * direction24[active["full_2024"]],
        0.001, 0.999,
    )
    locked = metrics(
        frames["full_2024"], targets["full_2024"], parent["full_2024"],
        candidate24, active["full_2024"],
    )
    source_pass = bool(
        dose > 0.0
        and all(item["gain"] > 0.0 for item in source.values())
        and all(item["positive_month_fraction"] >= 2.0 / 3.0 for item in source.values())
    )
    locked_pass = bool(
        locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -3.0
    )
    np.savez_compressed(
        output_dir / "selected_axis.npz", parent=parent["full_2024"],
        candidate=candidate24, active=active["full_2024"], direction=direction24,
    )
    joblib.dump(
        {"model": final_model, "center": final_center, "dose": dose},
        output_dir / "pre24_residual_model.joblib",
    )
    result = {
        "protocol": PROTOCOL,
        "status": "robust_point_candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "feature_count": int(feature_bank["full_2024"].shape[1]),
        "ridge_alpha": RIDGE_ALPHA,
        "correction_cap": CORRECTION_CAP,
        "source_selected_dose": dose,
        "source": source,
        "locked_full_2024": locked,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_full_fit": bool(source_pass and locked_pass),
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.contract_dir, args.v285_axes, args.v290_axes,
        args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
