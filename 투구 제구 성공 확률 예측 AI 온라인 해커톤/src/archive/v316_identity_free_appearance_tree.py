"""Identity-free nonlinear current-appearance residual model above v290.

The deployable representation comes from v304/v309's row-local renewal and
current-appearance command reconstruction.  A shallow histogram tree learns
only interactions among that state and low-cardinality baseball context; no
player or team identifier and no raw ASOF fingerprint enters the model.
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.archive.v298_recent_rate_denominator_signal import metrics
from src.archive.v310_current_appearance_form_residual import form_residual_features
from src.core.contract import _load_contract_axis


PROTOCOL = "V316_IDENTITY_FREE_APPEARANCE_TREE_V1"
TARGET = "control_success"
CORRECTION_CAP = 0.025


def build_features(appearance: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    output = form_residual_features(appearance, frame)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").to_numpy(np.float64)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").to_numpy(np.float64)
    inning = pd.to_numeric(frame["inning"], errors="coerce").to_numpy(np.float64)
    outs = pd.to_numeric(frame["outs_before"], errors="coerce").to_numpy(np.float64)
    runners = pd.to_numeric(frame["num_runners_on"], errors="coerce").to_numpy(np.float64)
    score = pd.to_numeric(
        frame["score_diff_pitcher_team"], errors="coerce"
    ).to_numpy(np.float64)
    output["context_balls"] = balls
    output["context_strikes"] = strikes
    output["context_count_code"] = balls * 3.0 + strikes
    output["context_inning"] = np.clip(inning, 1.0, 12.0)
    output["context_outs"] = outs
    output["context_runners"] = runners
    output["context_score_clip"] = np.clip(score, -5.0, 5.0)
    output["context_same_hand"] = (
        frame["pitcher_hand"].astype(str).to_numpy()
        == frame["batter_hand"].astype(str).to_numpy()
    ).astype(np.float64)
    output["context_pressure_count"] = ((balls >= 3.0) | (strikes >= 2.0)).astype(np.float64)
    output["context_late_close"] = ((inning >= 7.0) & (np.abs(score) <= 1.0)).astype(np.float64)
    return output.astype(np.float64)


def fit_model(features: pd.DataFrame, residual: np.ndarray) -> tuple[Pipeline, float]:
    target = np.asarray(residual, dtype=np.float64)
    target = target - float(target.mean())
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("tree", HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=300,
            l2_regularization=25.0,
            random_state=316,
        )),
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


def analytic_dose(parts: list[tuple[np.ndarray, np.ndarray]]) -> float:
    numerator = float(sum(np.dot(residual, direction) for residual, direction in parts))
    denominator = float(sum(np.dot(direction, direction) for _, direction in parts))
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0


def run(
    train_csv: Path,
    appearance_dir: Path,
    contract_dir: Path,
    v285_axes: Path,
    v290_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "inning", "balls_before",
        "strikes_before", "outs_before", "num_runners_on",
        "score_diff_pitcher_team", "pitcher_hand", "batter_hand", TARGET,
    ]
    raw = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        all_parent = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v290_axes, allow_pickle=False) as saved:
        all_parent.update({
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
            "full_2024": saved["candidate_full_2024"].astype(np.float64),
        })

    frames: dict[str, pd.DataFrame] = {}
    features: dict[str, pd.DataFrame] = {}
    target: dict[str, np.ndarray] = {}
    parent: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    for year, name in ((2022, "full_2022"), (2023, "late_2023"), (2024, "full_2024")):
        year_rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        regular = year_rows["game_type"].astype(str).eq("R").to_numpy()
        regular_frame = year_rows.loc[regular].reset_index(drop=True)
        appearance = pd.read_parquet(
            appearance_dir / f"current_appearance_features_{year}.parquet"
        )
        if len(appearance) != len(regular_frame):
            raise ValueError(f"appearance feature alignment mismatch: {year}")
        local_features = build_features(appearance, regular_frame)
        axis_frame = year_rows
        if name == "late_2023":
            keep = regular_frame["game_month"].ge(8).to_numpy()
            regular_frame = regular_frame.loc[keep].reset_index(drop=True)
            local_features = local_features.loc[keep].reset_index(drop=True)
            axis_frame = year_rows.loc[year_rows["game_month"].ge(8)].reset_index(drop=True)
        axis_regular = axis_frame["game_type"].astype(str).eq("R").to_numpy()
        frames[name] = regular_frame
        features[name] = local_features
        target[name] = np.asarray(axes[name]["target"], dtype=np.float64)[axis_regular]
        parent[name] = all_parent[name][axis_regular]
        active[name] = (
            np.asarray(axes[name]["exact_mask"], dtype=bool)[axis_regular]
            & (np.asarray(axes[name]["domain3"]).astype(str)[axis_regular] == "R_CORE")
            & local_features["form_covered"].to_numpy(bool)
        )
        if not np.array_equal(regular_frame[TARGET].to_numpy(np.float64), target[name]):
            raise ValueError(f"target order mismatch: {name}")

    early22 = frames["full_2022"]["game_month"].le(7).to_numpy() & active["full_2022"]
    late22 = frames["full_2022"]["game_month"].ge(8).to_numpy() & active["full_2022"]
    early_model, early_center = fit_model(
        features["full_2022"].loc[early22],
        target["full_2022"][early22] - parent["full_2022"][early22],
    )
    direction22 = predict_direction(early_model, early_center, features["full_2022"].loc[late22])
    model22, center22 = fit_model(
        features["full_2022"].loc[active["full_2022"]],
        target["full_2022"][active["full_2022"]] - parent["full_2022"][active["full_2022"]],
    )
    direction23_all = predict_direction(model22, center22, features["late_2023"])
    direction23 = direction23_all[active["late_2023"]]
    dose = analytic_dose([
        (target["full_2022"][late22] - parent["full_2022"][late22], direction22),
        (target["late_2023"][active["late_2023"]] - parent["late_2023"][active["late_2023"]], direction23),
    ])
    candidate22 = np.clip(parent["full_2022"][late22] + dose * direction22, 0.001, 0.999)
    candidate23 = parent["late_2023"].copy()
    candidate23[active["late_2023"]] = np.clip(
        candidate23[active["late_2023"]] + dose * direction23, 0.001, 0.999
    )
    source = {
        "early_2022_to_late_2022": metrics(
            frames["full_2022"].loc[late22].reset_index(drop=True),
            target["full_2022"][late22], parent["full_2022"][late22], candidate22,
            np.ones(int(late22.sum()), dtype=bool),
        ),
        "full_2022_to_late_2023": metrics(
            frames["late_2023"], target["late_2023"], parent["late_2023"],
            candidate23, active["late_2023"],
        ),
    }
    final_features = pd.concat([
        features["full_2022"].loc[active["full_2022"]],
        features["late_2023"].loc[active["late_2023"]],
    ], ignore_index=True)
    final_residual = np.concatenate([
        target["full_2022"][active["full_2022"]] - parent["full_2022"][active["full_2022"]],
        target["late_2023"][active["late_2023"]] - parent["late_2023"][active["late_2023"]],
    ])
    final_model, final_center = fit_model(final_features, final_residual)
    direction24 = predict_direction(final_model, final_center, features["full_2024"])
    candidate24 = parent["full_2024"].copy()
    candidate24[active["full_2024"]] = np.clip(
        candidate24[active["full_2024"]]
        + dose * direction24[active["full_2024"]],
        0.001, 0.999,
    )
    locked = metrics(
        frames["full_2024"], target["full_2024"], parent["full_2024"],
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
        {"model": final_model, "center": final_center, "dose": dose},
        output_dir / "pre24_identity_free_tree.joblib",
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_point_candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "feature_count": int(features["full_2024"].shape[1]),
        "source_selected_dose": dose,
        "source": source,
        "locked_full_2024": locked,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "pitcher_batter_team_ids_used": False,
            "raw_asof_fingerprint_features_used": False,
            "evaluation_row_order_or_aggregates_used": False,
            "runtime_live_features_are_row_local": True,
            "full_2024_used_for_dose_selection": False,
            "public_score_used_for_selection": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--appearance-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.appearance_dir, args.contract_dir, args.v285_axes,
        args.v290_axes, args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
