"""Independently train a hierarchical pitcher-state residual stack.

This experiment uses official train rows and prior-season TrackMan context
only.  A leakage-safe empirical-Bayes estimate separates career command from
the current season, then a CatBoost regressor learns the remaining residual.
The model and blend dose are selected on full-2022 and late-2023 before the
development-contaminated 2024 contract is opened once.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v178_row_region_signed_stack_rebase import (
    apply_direction,
    load_direction,
    load_weights,
    source_gate,
)
from src.champion.v131_catboost_h1_independent_oof import _prepare_features
from src.core.contract import _load_contract_axis


PROTOCOL = "V184_HIERARCHICAL_RESIDUAL_STACK_V1"
SOURCE_AXES = ("full_2022", "late_2023")
V178_SCALE = 0.25
LOCAL_WEIGHTS = (0.0, 0.0025, 0.005, 0.01, 0.015, 0.025, 0.05)
CATEGORICAL_BASE = (
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "game_dayofweek",
)
MODEL_CONFIG = {
    "iterations": 300,
    "learning_rate": 0.035,
    "depth": 8,
    "l2_leaf_reg": 20.0,
    "random_strength": 0.35,
    "subsample": 0.85,
    "decay": 0.55,
    "seed": 184,
}


def _prior_league_means(target: np.ndarray, season: np.ndarray) -> dict[int, float]:
    years = sorted(int(value) for value in np.unique(season))
    return {
        year: float(np.mean(target[season < year])) if np.any(season < year) else 0.5
        for year in years
    }


def hierarchical_pitcher_features(
    frame: pd.DataFrame,
    target: np.ndarray,
    season: np.ndarray,
) -> pd.DataFrame:
    """Build row-local pitcher priors without using the audit-season target.

    The official career count/rate are measured immediately before each pitch.
    For a season, the last row of every earlier season is advanced by its known
    label to obtain a clean opening-day snapshot.  Subtracting that snapshot
    recovers current-season attempts and successes.
    """

    n_rows = len(frame)
    if len(target) != n_rows or len(season) != n_rows:
        raise ValueError("frame, target and season must be aligned")
    pitcher = pd.to_numeric(frame["pitcher_id"], errors="raise").to_numpy(np.int64)
    career_n = (
        pd.to_numeric(frame["asof_pitcher_n"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    career_rate = (
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
        .fillna(0.5)
        .to_numpy(np.float64)
    )
    career_events = career_n * career_rate
    previous_n = np.zeros(n_rows, dtype=np.float64)
    previous_events = np.zeros(n_rows, dtype=np.float64)
    league_prior = np.zeros(n_rows, dtype=np.float64)
    prior_means = _prior_league_means(np.asarray(target), np.asarray(season))

    for year in sorted(int(value) for value in np.unique(season)):
        audit = np.asarray(season) == year
        history = np.asarray(season) < year
        league_prior[audit] = prior_means[year]
        if not np.any(history):
            continue
        hist_index = np.flatnonzero(history)
        # Official rows are chronological.  Sorting by original position and
        # taking tail(1) yields the state after the last known prior pitch.
        hist = pd.DataFrame(
            {
                "pitcher_id": pitcher[history],
                "row_position": hist_index,
                "n_after": career_n[history] + 1.0,
                "events_after": career_events[history] + np.asarray(target)[history],
            }
        )
        snapshot = hist.groupby("pitcher_id", sort=False).tail(1).set_index("pitcher_id")
        query = pd.Index(pitcher[audit])
        previous_n[audit] = snapshot["n_after"].reindex(query).fillna(0.0).to_numpy()
        previous_events[audit] = (
            snapshot["events_after"].reindex(query).fillna(0.0).to_numpy()
        )

    season_n = np.maximum(career_n - previous_n, 0.0)
    season_events = np.clip(career_events - previous_events, 0.0, season_n)
    raw_season_rate = np.divide(
        season_events,
        season_n,
        out=league_prior.copy(),
        where=season_n > 0.0,
    )

    recent_columns = [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]
    recent = np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            for column in recent_columns
        ]
    )
    fallback = np.broadcast_to(career_rate[:, None], recent.shape)
    recent = np.where(np.isfinite(recent), recent, fallback)
    recent_mean = np.mean(recent, axis=1)
    recent_std = np.std(recent, axis=1)

    career_strength = np.clip(
        55.0 + 220.0 * recent_std + 40.0 / (1.0 + np.log1p(career_n)),
        50.0,
        180.0,
    )
    career_base = (
        career_events + career_strength * league_prior
    ) / np.maximum(career_n + career_strength, 1.0)
    season_strength = np.clip(30.0 + 160.0 * recent_std, 25.0, 100.0)
    season_base = (
        season_events + season_strength * career_base
    ) / np.maximum(season_n + season_strength, 1.0)
    season_reliability = season_n / (season_n + 80.0)
    season_weight = 0.15 + 0.30 * season_reliability
    hierarchical = career_base + season_weight * (season_base - career_base)

    return pd.DataFrame(
        {
            "hier_league_prior": league_prior,
            "hier_career_base": career_base,
            "hier_season_raw": raw_season_rate,
            "hier_season_base": season_base,
            "hier_prediction": hierarchical,
            "hier_previous_n_log": np.log1p(previous_n),
            "hier_season_n_log": np.log1p(season_n),
            "hier_season_reliability": season_reliability,
            "hier_career_strength": career_strength,
            "hier_season_strength": season_strength,
            "hier_recent_mean": recent_mean,
            "hier_recent_std": recent_std,
            "hier_recent_minus_career": recent_mean - career_rate,
            "hier_season_minus_career": season_base - career_base,
        },
        index=frame.index,
        dtype=np.float32,
    )


def prepare_model_frame(
    frame: pd.DataFrame,
    h1_features: list[str],
    hierarchical: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_names = list(dict.fromkeys(h1_features))
    output = frame.loc[:, feature_names].copy()
    for column in hierarchical.columns:
        output[column] = hierarchical[column].to_numpy(np.float32)
        feature_names.append(column)

    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int)
    pitcher = pd.to_numeric(frame["pitcher_id"], errors="coerce").fillna(-1).astype(int)
    batter = pd.to_numeric(frame["batter_id"], errors="coerce").fillna(-1).astype(int)
    phand = pd.to_numeric(frame["pitcher_hand"], errors="coerce").fillna(-1).astype(int)
    bhand = pd.to_numeric(frame["batter_hand"], errors="coerce").fillna(-1).astype(int)
    base = frame["base_state"].fillna("NA").astype(str)
    interactions = {
        "cat_pitcher_count": pitcher.astype(str) + "|" + balls.astype(str) + "-" + strikes.astype(str),
        "cat_pitcher_platoon": pitcher.astype(str) + "|" + bhand.astype(str),
        "cat_batter_platoon": batter.astype(str) + "|" + phand.astype(str),
        "cat_count_base": balls.astype(str) + "-" + strikes.astype(str) + "|" + base,
        "cat_hand_count": phand.astype(str) + "-" + bhand.astype(str) + "|" + balls.astype(str) + "-" + strikes.astype(str),
    }
    for column, values in interactions.items():
        output[column] = values
        feature_names.append(column)

    categorical = [column for column in CATEGORICAL_BASE if column in feature_names]
    categorical.extend(interactions)
    categorical = list(dict.fromkeys(categorical))
    for column in categorical:
        output[column] = output[column].fillna("NA").astype(str)
    numeric = [column for column in feature_names if column not in categorical]
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return output, feature_names, categorical


def fit_residual_fold(
    features: pd.DataFrame,
    feature_names: list[str],
    categorical: list[str],
    target: np.ndarray,
    season: np.ndarray,
    hierarchical_base: np.ndarray,
    audit_year: int,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    config = dict(MODEL_CONFIG if config is None else config)
    checkpoint = output_dir / f"hierarchical_residual_{audit_year}.npy"
    if checkpoint.exists():
        saved = np.load(checkpoint, allow_pickle=False).astype(np.float64)
        if len(saved) != int(np.sum(season == audit_year)):
            raise ValueError(f"invalid checkpoint length for {audit_year}")
        print(f"[v184] loaded checkpoint fold={audit_year}", flush=True)
        return saved

    fit = season < audit_year
    audit = season == audit_year
    weights = np.power(
        float(config["decay"]),
        np.maximum((audit_year - 1) - season[fit], 0),
    )
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        hierarchical_base, dtype=np.float64
    )
    train_pool = Pool(
        features.loc[fit, feature_names],
        label=residual[fit],
        weight=weights,
        cat_features=categorical,
    )
    audit_pool = Pool(
        features.loc[audit, feature_names],
        cat_features=categorical,
    )
    model = CatBoostRegressor(
        iterations=int(config["iterations"]),
        learning_rate=float(config["learning_rate"]),
        depth=int(config["depth"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        random_strength=float(config["random_strength"]),
        bootstrap_type="Bernoulli",
        subsample=float(config["subsample"]),
        loss_function="RMSE",
        random_seed=int(config["seed"]),
        thread_count=16,
        verbose=50,
        allow_writing_files=False,
    )
    started = time.time()
    model.fit(train_pool)
    prediction = np.clip(
        hierarchical_base[audit] + model.predict(audit_pool), 0.001, 0.999
    ).astype(np.float64)
    np.save(checkpoint, prediction, allow_pickle=False)
    print(
        f"[v184] fold={audit_year} fit={int(fit.sum()):,} "
        f"audit={int(audit.sum()):,} elapsed={time.time()-started:.1f}s",
        flush=True,
    )
    del train_pool, audit_pool, model
    gc.collect()
    return prediction


def additive_stack(
    parent: np.ndarray,
    v178_direction: np.ndarray,
    local_prediction: np.ndarray,
    active: np.ndarray,
    local_weight: float,
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    output = parent + V178_SCALE * np.asarray(v178_direction, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    output[active] += float(local_weight) * (
        np.asarray(local_prediction, dtype=np.float64)[active] - parent[active]
    )
    return np.clip(output, 0.001, 0.999)


def _load_stack_context(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v158_path: Path,
    v165_summary: Path,
    library_root: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[int, pd.DataFrame]]:
    _context, raw_frames, correction = _load_year_context(train_csv)
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, raw_frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    weights = load_weights(v165_summary)
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {
            name: saved[name].astype(np.float64)
            for name in (*SOURCE_AXES, "full_2024")
        }
    directions = {
        name: load_direction(name, v158_base[name], weights, library_root)
        for name in (*SOURCE_AXES, "full_2024")
    }
    return axes, parents, directions, parity, raw_frames


def run(
    train_csv: Path,
    trackman_csv: Path,
    component_root: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v158_path: Path,
    v165_summary: Path,
    library_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, component_root
    )
    hierarchical = hierarchical_pitcher_features(train, target, season)
    model_frame, feature_names, categorical = prepare_model_frame(
        train, h1_features, hierarchical
    )
    prediction_by_year = {
        year: fit_residual_fold(
            model_frame, feature_names, categorical, target, season,
            hierarchical["hier_prediction"].to_numpy(np.float64), year,
            output_dir,
        )
        for year in (2022, 2023)
    }
    axes, parents, directions, parity, raw_frames = _load_stack_context(
        train_csv, contract_dir, v104_path, h1_path, c3_path, v160_path,
        bridge_oof, v158_path, v165_summary, library_root,
    )
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    local = {
        "full_2022": prediction_by_year[2022],
        "late_2023": prediction_by_year[2023][late23],
    }
    for name in SOURCE_AXES:
        if len(local[name]) != len(parents[name]):
            raise ValueError(f"OOF alignment mismatch: {name}")

    rows: list[dict[str, Any]] = []
    source_details: dict[str, dict[str, Any]] = {}
    for local_weight in LOCAL_WEIGHTS:
        details: dict[str, Any] = {}
        for name in SOURCE_AXES:
            active = np.asarray(axes[name]["exact_mask"], dtype=bool)
            candidate = additive_stack(
                parents[name], directions[name], local[name], active, local_weight
            )
            details[name] = metrics(axes[name], parents[name], candidate)
        passed = all(source_gate(details[name]) for name in SOURCE_AXES)
        source_details[str(local_weight)] = details
        rows.append(
            {
                "local_weight": local_weight,
                "full_2022_gain": details["full_2022"]["gain"],
                "late_2023_gain": details["late_2023"]["gain"],
                "minimum_gain": min(details[name]["gain"] for name in SOURCE_AXES),
                "mean_gain": float(np.mean([details[name]["gain"] for name in SOURCE_AXES])),
                "worst_month_gain": min(
                    details[name]["worst_month_gain"] for name in SOURCE_AXES
                ),
                "source_gate_passed": passed,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=[False, False, False, False],
        kind="stable",
    )
    ranking.to_csv(output_dir / "source_weight_screen.csv", index=False, encoding="utf-8-sig")
    passing = ranking.loc[ranking["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "model_config": MODEL_CONFIG,
            "source": source_details,
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected_weight = float(passing.iloc[0]["local_weight"])
        prediction24 = fit_residual_fold(
            model_frame, feature_names, categorical, target, season,
            hierarchical["hier_prediction"].to_numpy(np.float64), 2024,
            output_dir,
        )
        active24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
        candidate24 = additive_stack(
            parents["full_2024"], directions["full_2024"], prediction24,
            active24, selected_weight,
        )
        locked = metrics(axes["full_2024"], parents["full_2024"], candidate24)
        family = [
            additive_stack(
                parents["full_2024"], directions["full_2024"], prediction24,
                active24, float(weight),
            )
            for weight in passing["local_weight"].tolist()
        ]
        if len(family) == 1:
            family.append(parents["full_2024"].copy())
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], candidate24,
            active24, family,
        )
        point_pass = bool(
            locked["gain"] > 0.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -5.0
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
            parent=parents["full_2024"],
            candidate=candidate24,
            v178_direction=directions["full_2024"],
            local_prediction=prediction24,
            active=active24,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "model_config": MODEL_CONFIG,
            "frozen_v178_scale": V178_SCALE,
            "selected_local_weight_on_sources_only": selected_weight,
            "source": source_details[str(selected_weight)],
            "locked_2024": locked,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_packaging": bool(point_pass and robust_pass),
            "parity": parity,
            "restrictions": restrictions(),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def restrictions() -> dict[str, bool]:
    return {
        "official_train_labels_only": True,
        "prior_season_trackman_only": True,
        "strict_forward_model_fits": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_prediction_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
        "locked_2024_development_contaminated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v158-path", type=Path, required=True)
    parser.add_argument("--v165-summary", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.train_csv, args.trackman_csv, args.component_root,
        args.contract_dir, args.v104_path, args.h1_path, args.c3_path,
        args.v160_path, args.bridge_oof, args.v158_path, args.v165_summary,
        args.library_root, args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
