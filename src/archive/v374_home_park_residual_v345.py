"""Strict-forward, row-local home-park residual complement above v345.

The main table does not expose a venue column, but regular-season home team is
deterministic from inning half: the pitching team is home in the top half and
the batting team is home in the bottom half.  We first partial out observable
pitcher command, batter, count, handedness, inning and calendar context with a
small ridge model.  Only the remaining shrunk home-team (park) residual is
added to the incumbent.  Hyperparameters are selected on full-2022 and
late-2023, before full-2024 is opened once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v324_player_transition_residual import axis_metrics
from src.archive.v362_pitcher_situation_lowrank_v345 import full_row_rms


PROTOCOL = "V374_HOME_PARK_RESIDUAL_V345_V1"
TARGET = "control_success"
FAMILIES = {
    "park": ("home_team",),
    "park_side": ("home_team", "top_bottom"),
    "park_count": ("home_team", "count_state"),
}
SHRINKAGES = (500.0, 1500.0)
WEIGHTS = (0.25, 0.50)

NUMERIC = [
    "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_batter_success_rate",
    "asof_pitcher_n",
    "asof_batter_n",
]
CATEGORICAL = [
    "count_state", "pitcher_hand", "batter_hand", "top_bottom",
    "inning_band", "game_month",
]


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    top = out["top_bottom"].astype(str).eq("T")
    out["home_team"] = np.where(top, out["pitcher_team_id"], out["batter_team_id"])
    out["home_team"] = pd.Series(out["home_team"], index=out.index).astype("Int64").astype(str)
    out["count_state"] = (
        out["balls_before"].astype(str) + "-" + out["strikes_before"].astype(str)
    )
    inning = pd.to_numeric(out["inning"], errors="coerce").fillna(1)
    out["inning_band"] = pd.cut(
        inning, bins=[-np.inf, 3, 6, np.inf], labels=["early", "middle", "late"]
    ).astype(str)
    return out


def nuisance_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("features", ColumnTransformer([
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ])),
        ("ridge", Ridge(alpha=100.0, fit_intercept=True, solver="lsqr")),
    ])


def fit_effects(
    fit_frame: pd.DataFrame,
    family: str,
    shrinkage: float,
) -> tuple[pd.Series, dict[str, Any]]:
    regular = fit_frame["game_type"].astype(str).eq("R")
    rows = fit_frame.loc[regular].reset_index(drop=True)
    model = nuisance_pipeline()
    target = rows[TARGET].to_numpy(np.float64)
    model.fit(rows, target)
    baseline = np.clip(model.predict(rows), 0.001, 0.999)
    residual = target - baseline
    keys = list(FAMILIES[family])
    grouped = rows[keys].copy()
    grouped["residual"] = residual
    stats = grouped.groupby(keys, observed=True)["residual"].agg(["sum", "count"])
    effect = stats["sum"] / (stats["count"] + float(shrinkage))
    weighted_mean = float(np.average(effect.to_numpy(), weights=stats["count"].to_numpy()))
    effect = effect - weighted_mean
    return effect, {
        "fit_rows": len(rows),
        "groups": len(effect),
        "effect_sd": float(effect.std(ddof=0)),
        "max_abs_effect": float(effect.abs().max()),
        "residual_mean": float(residual.mean()),
    }


def map_effect(effect: pd.Series, rows: pd.DataFrame, family: str) -> np.ndarray:
    keys = list(FAMILIES[family])
    if len(keys) == 1:
        index = pd.Index(rows[keys[0]].to_numpy())
    else:
        index = pd.MultiIndex.from_frame(rows[keys])
    return effect.reindex(index, fill_value=0.0).to_numpy(np.float64)


def route_mask(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    return regular & ~anchor


def make_candidate(
    parent: np.ndarray,
    signal: np.ndarray,
    active: np.ndarray,
    weight: float,
) -> np.ndarray:
    out = np.asarray(parent, dtype=np.float64).copy()
    use = active & np.not_equal(signal, 0.0)
    out[use] = np.clip(out[use] + float(weight) * signal[use], 0.001, 0.999)
    return out


def run(train_csv: Path, v335_axes: Path, v345_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = add_context(pd.read_csv(train_csv, low_memory=False))
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "full_2023": train.loc[train["season"].eq(2023)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"candidate_{axis}"].astype(np.float64)
            for axis in ("full_2022", "late_2023")
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )
    late23 = frames["full_2023"]["game_month"].ge(8).to_numpy()

    effects: dict[tuple[str, float], dict[str, pd.Series]] = {}
    signals: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    fit_details: dict[str, Any] = {}
    for family in FAMILIES:
        for shrinkage in SHRINKAGES:
            key = (family, shrinkage)
            effect22, detail22 = fit_effects(
                train.loc[train["season"].between(2020, 2021)].reset_index(drop=True),
                family, shrinkage,
            )
            effect23, detail23 = fit_effects(
                train.loc[train["season"].between(2021, 2022)].reset_index(drop=True),
                family, shrinkage,
            )
            effects[key] = {"full_2022": effect22, "late_2023": effect23}
            signals[key] = {
                "full_2022": map_effect(effect22, frames["full_2022"], family),
                "late_2023": map_effect(effect23, frames["full_2023"], family)[late23],
            }
            fit_details[f"{family}_s{int(shrinkage)}"] = {
                "full_2022": detail22, "late_2023": detail23,
            }

    active = {axis: route_mask(frames[axis]) for axis in ("full_2022", "late_2023")}
    trials: list[dict[str, Any]] = []
    payload: dict[tuple[str, float, float], dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        for shrinkage in SHRINKAGES:
            for weight in WEIGHTS:
                key = (family, shrinkage, weight)
                payload[key] = {}
                record: dict[str, Any] = {
                    "family": family, "shrinkage": shrinkage, "weight": weight,
                }
                for axis in ("full_2022", "late_2023"):
                    candidate = make_candidate(
                        parents[axis], signals[(family, shrinkage)][axis], active[axis], weight
                    )
                    payload[key][axis] = candidate
                    record[axis] = axis_metrics(
                        frames[axis], parents[axis], candidate, active[axis]
                    )
                record["minimum_source_gain"] = min(
                    record["full_2022"]["gain"], record["late_2023"]["gain"]
                )
                record["source_pass"] = bool(
                    record["full_2022"]["gain"] > 0.0
                    and record["late_2023"]["gain"] > 0.0
                    and record["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
                    and record["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
                    and record["full_2022"]["worst_month_gain"] > -10.0
                    and record["late_2023"]["worst_month_gain"] > -10.0
                )
                trials.append(record)

    passing = sorted(
        (row for row in trials if row["source_pass"]),
        key=lambda row: (row["minimum_source_gain"], row["full_2022"]["gain"] + row["late_2023"]["gain"]),
        reverse=True,
    )
    restrictions = {
        "official_train_only": True,
        "home_team_derived_row_locally_from_inning_half": True,
        "source_selection_full2022_and_late2023_only": True,
        "full2024_opened_only_after_source_gate": True,
        "team13_and_futures_rows_preserved": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not passing:
        summary = {
            "protocol": PROTOCOL, "status": "source_reject",
            "source_trials": trials, "fit_details": fit_details,
            "locked_origin_opened": False, "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    chosen = passing[0]
    family = str(chosen["family"])
    shrinkage = float(chosen["shrinkage"])
    weight = float(chosen["weight"])
    effect24, detail24 = fit_effects(
        train.loc[train["season"].between(2022, 2023)].reset_index(drop=True),
        family, shrinkage,
    )
    signal24 = map_effect(effect24, frames["full_2024"], family)
    active24 = route_mask(frames["full_2024"])
    candidate24 = make_candidate(parents["full_2024"], signal24, active24, weight)
    locked = axis_metrics(
        frames["full_2024"], parents["full_2024"], candidate24, active24
    )
    locked["full_row_rms_shift"] = full_row_rms(parents["full_2024"], candidate24)
    increment = candidate24 - parents["full_2024"]
    union = (np.abs(increment) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = float(np.corrcoef(increment[union], v345_increment[union])[0, 1])
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, parents["full_2024"], candidate24, active24,
        [parents["full_2024"], candidate24],
    )
    eligible = bool(
        locked["gain"] >= 1.5
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.00035
        and robustness["chronological_block"]["p05"] > 0.0
    )
    effect24.rename("effect").reset_index().to_csv(
        output_dir / "locked_effect.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2022=parents["full_2022"],
        candidate_full_2022=payload[(family, shrinkage, weight)]["full_2022"],
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=payload[(family, shrinkage, weight)]["late_2023"],
        parent_full_2024=parents["full_2024"], candidate_full_2024=candidate24,
        signal_full_2024=signal24, active_full_2024=active24,
    )
    summary = {
        "protocol": PROTOCOL, "status": "candidate" if eligible else "locked_reject",
        "source_trials": trials, "selected_source_recipe": chosen,
        "fit_details": {**fit_details, "locked_full_2024": detail24},
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "eligible_for_packaging": eligible, "restrictions": restrictions,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v335_axes, args.v345_axes, args.output_dir
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
