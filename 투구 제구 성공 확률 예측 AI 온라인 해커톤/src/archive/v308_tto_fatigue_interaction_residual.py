"""Strict-forward residual screen for TTO x pitcher-fatigue interactions.

v307 recovers the batting team's plate-appearance cycle from legal row-local
context, while v304 recovers a probabilistic current-appearance pitch count.
This experiment combines those independently reconstructed clocks.  The
fixed, low-DOF Ridge correction is evaluated on two source transfers before a
single locked full-2024 read against the v290-equivalent parent.
"""

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
from src.archive.v304_current_appearance_renewal_audit import (
    attach_appearance_state,
    build_strict_renewal_features,
)
from src.archive.v307_batting_order_tto_audit import (
    attach_batting_state,
    build_strict_batting_features,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V308_TTO_FATIGUE_INTERACTION_RESIDUAL_V1"
TARGET = "control_success"
RIDGE_ALPHA = 1500.0
CORRECTION_CAP = 0.020


def tto_residual_features(
    batting: pd.DataFrame, renewal: pd.DataFrame, frame: pd.DataFrame
) -> pd.DataFrame:
    """Return the fixed baseball-structured interaction feature family."""

    context_pa = pd.to_numeric(
        batting["pa_expected_context"], errors="coerce"
    ).to_numpy(np.float64)
    snapped_pa = pd.to_numeric(
        batting["pa_expected_slot_snap"], errors="coerce"
    ).to_numpy(np.float64)
    slot = pd.to_numeric(
        batting["predicted_lineup_slot"], errors="coerce"
    ).to_numpy(np.float64)
    p2 = pd.to_numeric(
        batting["context_p_tto2"], errors="coerce"
    ).to_numpy(np.float64)
    p3 = pd.to_numeric(
        batting["context_p_tto3"], errors="coerce"
    ).to_numpy(np.float64)
    age = pd.to_numeric(
        renewal["renewal_expected_age"], errors="coerce"
    ).to_numpy(np.float64)
    p20 = pd.to_numeric(
        renewal["renewal_p_age_ge20"], errors="coerce"
    ).to_numpy(np.float64)
    p60 = pd.to_numeric(
        renewal["renewal_p_age_ge60"], errors="coerce"
    ).to_numpy(np.float64)
    typical = pd.to_numeric(
        renewal["renewal_history_median_length"], errors="coerce"
    ).to_numpy(np.float64)
    inning = pd.to_numeric(frame["inning"], errors="coerce").to_numpy(np.float64)
    balls = pd.to_numeric(
        frame["balls_before"], errors="coerce"
    ).to_numpy(np.float64)
    strikes = pd.to_numeric(
        frame["strikes_before"], errors="coerce"
    ).to_numpy(np.float64)

    # Smooth role proxy avoids a brittle starter/reliever threshold.  A pitcher
    # with both a long historical appearance and a substantial current age is
    # the population for which lineup-wide TTO has baseball meaning.
    long_role = np.clip((typical - 15.0) / 45.0, 0.0, 1.0)
    current_depth = np.clip((age - 5.0) / 60.0, 0.0, 1.0)
    starter_depth = long_role * current_depth
    pitcher_cycles = age / 36.0  # roughly nine batters times four pitches
    slot_angle = 2.0 * np.pi * (slot - 1.0) / 9.0
    two_strike = (strikes >= 2.0).astype(np.float64)
    three_ball = (balls >= 3.0).astype(np.float64)
    behind = (balls > strikes).astype(np.float64)
    return pd.DataFrame(
        {
            "tto_context_pa": context_pa / 45.0,
            "tto_snapped_pa": snapped_pa / 45.0,
            "tto_p_second": p2,
            "tto_p_third": p3,
            "tto_slot_sin": np.sin(slot_angle),
            "tto_slot_cos": np.cos(slot_angle),
            "tto_log_lineup_history": np.log1p(
                batting["lineup_history_games"].to_numpy(np.float64)
            ),
            "tto_context_level": batting["context_level"].to_numpy(np.float64),
            "tto_age": age / 100.0,
            "tto_p_age20": p20,
            "tto_p_age60": p60,
            "tto_long_role": long_role,
            "tto_starter_depth": starter_depth,
            "tto_pitcher_cycles": pitcher_cycles,
            "tto_second_x_depth": p2 * starter_depth,
            "tto_third_x_depth": p3 * starter_depth,
            "tto_second_x_age20": p2 * p20,
            "tto_third_x_age60": p3 * p60,
            "tto_cycle_gap": pitcher_cycles - context_pa / 9.0,
            "tto_second_x_two_strike": p2 * starter_depth * two_strike,
            "tto_third_x_two_strike": p3 * starter_depth * two_strike,
            "tto_second_x_three_ball": p2 * starter_depth * three_ball,
            "tto_third_x_three_ball": p3 * starter_depth * three_ball,
            "tto_third_x_behind": p3 * starter_depth * behind,
            "tto_late_inning_x_depth": (inning >= 6.0) * starter_depth,
            "tto_lineup_covered": batting["lineup_covered"].to_numpy(np.float64),
            "tto_renewal_covered": renewal["renewal_covered"].to_numpy(np.float64),
        }
    )


def fit_model(features: pd.DataFrame, residual: np.ndarray) -> tuple[Pipeline, float]:
    target = np.asarray(residual, dtype=np.float64)
    target -= float(target.mean())
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)),
        ]
    )
    model.fit(features, target)
    center = float(np.mean(model.predict(features)))
    return model, center


def predict_correction(
    model: Pipeline, center: float, features: pd.DataFrame
) -> np.ndarray:
    direction = np.asarray(model.predict(features), dtype=np.float64) - center
    return np.clip(direction, -CORRECTION_CAP, CORRECTION_CAP)


def pooled_dose(parts: list[tuple[np.ndarray, np.ndarray]]) -> float:
    numerator = float(sum(np.dot(residual, direction) for residual, direction in parts))
    denominator = float(sum(np.dot(direction, direction) for _, direction in parts))
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0


def _parent_arrays(v285_axes: Path, v290_axes: Path) -> dict[str, np.ndarray]:
    with np.load(v285_axes, allow_pickle=False) as saved:
        full22 = saved["candidate_full_2022"].astype(np.float64)
    with np.load(v290_axes, allow_pickle=False) as saved:
        late23 = saved["candidate_late_2023"].astype(np.float64)
        full24 = saved["candidate_full_2024"].astype(np.float64)
    return {"full_2022": full22, "late_2023": late23, "full_2024": full24}


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "evaluation_row_id_or_order_used": False,
        "other_evaluation_rows_required": False,
        "runtime_live_features_are_row_local": True,
        "strictly_prior_season_structural_lookups": True,
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
        "balls_before", "strikes_before", "outs_before", "run_top_before",
        "run_bot_before", "num_runners_on", "runner_on_1b", "runner_on_2b",
        "runner_on_3b", "pitcher_id", "batter_id", "batter_team_id",
        "asof_pitcher_n", TARGET,
    ]
    raw = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    batting_rows = attach_batting_state(raw)
    renewal_rows, appearances = attach_appearance_state(raw)
    parent = _parent_arrays(v285_axes, v290_axes)
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
        regular_mask = year_rows["game_type"].astype(str).eq("R").to_numpy()
        regular_frame = year_rows.loc[regular_mask].reset_index(drop=True)
        batting = build_strict_batting_features(batting_rows, year)
        renewal = build_strict_renewal_features(renewal_rows, appearances, year)
        if not (len(batting) == len(renewal) == len(regular_frame)):
            raise ValueError(f"structural feature alignment mismatch: {year}")
        features = tto_residual_features(batting, renewal, regular_frame)
        axis_frame = year_rows
        if name == "late_2023":
            keep = regular_frame["game_month"].ge(8).to_numpy()
            regular_frame = regular_frame.loc[keep].reset_index(drop=True)
            features = features.loc[keep].reset_index(drop=True)
            axis_frame = year_rows.loc[year_rows["game_month"].ge(8)].reset_index(drop=True)
        frames[name] = regular_frame
        feature_bank[name] = features
        if len(axis_frame) != len(axes[name]["target"]):
            raise ValueError(f"raw-to-contract length mismatch: {name}")
        axis_regular = axis_frame["game_type"].astype(str).eq("R").to_numpy()
        targets[name] = np.asarray(axes[name]["target"], dtype=np.float64)[axis_regular]
        active[name] = (
            np.asarray(axes[name]["exact_mask"], dtype=bool)[axis_regular]
            & (np.asarray(axes[name]["domain3"]).astype(str)[axis_regular] == "R_CORE")
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
    model_early, center_early = fit_model(
        feature_bank["full_2022"].loc[early22],
        targets["full_2022"][early22] - parent["full_2022"][early22],
    )
    direction_late22 = predict_correction(
        model_early, center_early, feature_bank["full_2022"].loc[late22]
    )

    model22, center22 = fit_model(
        feature_bank["full_2022"].loc[active["full_2022"]],
        targets["full_2022"][active["full_2022"]]
        - parent["full_2022"][active["full_2022"]],
    )
    direction23_all = predict_correction(
        model22, center22, feature_bank["late_2023"]
    )
    direction23 = direction23_all[active["late_2023"]]
    dose = pooled_dose(
        [
            (
                targets["full_2022"][late22] - parent["full_2022"][late22],
                direction_late22,
            ),
            (
                targets["late_2023"][active["late_2023"]]
                - parent["late_2023"][active["late_2023"]],
                direction23,
            ),
        ]
    )
    candidate_late22 = np.clip(
        parent["full_2022"][late22] + dose * direction_late22, 0.001, 0.999
    )
    candidate23 = parent["late_2023"].copy()
    candidate23[active["late_2023"]] = np.clip(
        candidate23[active["late_2023"]] + dose * direction23, 0.001, 0.999
    )
    source = {
        "early_2022_to_late_2022": metrics(
            frame22.loc[late22].reset_index(drop=True),
            targets["full_2022"][late22], parent["full_2022"][late22],
            candidate_late22, np.ones(int(late22.sum()), dtype=bool),
        ),
        "full_2022_to_late_2023": metrics(
            frames["late_2023"], targets["late_2023"], parent["late_2023"],
            candidate23, active["late_2023"],
        ),
    }

    combined_features = pd.concat(
        [
            feature_bank["full_2022"].loc[active["full_2022"]],
            feature_bank["late_2023"].loc[active["late_2023"]],
        ],
        ignore_index=True,
    )
    combined_residual = np.concatenate(
        [
            targets["full_2022"][active["full_2022"]]
            - parent["full_2022"][active["full_2022"]],
            targets["late_2023"][active["late_2023"]]
            - parent["late_2023"][active["late_2023"]],
        ]
    )
    model_pre24, center_pre24 = fit_model(combined_features, combined_residual)
    direction24 = predict_correction(
        model_pre24, center_pre24, feature_bank["full_2024"]
    )
    candidate24 = parent["full_2024"].copy()
    candidate24[active["full_2024"]] = np.clip(
        candidate24[active["full_2024"]]
        + dose * direction24[active["full_2024"]], 0.001, 0.999
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
        output_dir / "selected_axis.npz",
        parent=parent["full_2024"], candidate=candidate24,
        active=active["full_2024"], direction=direction24,
    )
    joblib.dump(
        {"model": model_pre24, "center": center_pre24, "dose": dose},
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
