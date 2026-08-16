"""Strict temporal audit of a recent-season shared exact-ASOF base.

The model uses only one labelled source season, while row-local cumulative
ASOF counters are decomposed into current-season evidence using labelled
history strictly before the row's season.  It is evaluated against the exact
v10 OOF prediction on the next season.  No audit-season target is used in
feature construction, fitting, calibration, or prediction.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.temporal_stable_conditional import (
    Paths,
    _add_domain_and_pressure,
    _bss,
    _load_base,
)


TARGET = "control_success"
NUMERIC_COLUMNS = [
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
CATEGORICAL_COLUMNS = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "domain3",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]
PITCHER_COMPONENTS = [
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
TRANSITIONS = ((2021, 2022), (2022, 2023), (2023, 2024))


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _rounded_count(n: np.ndarray, rate: np.ndarray) -> np.ndarray:
    output = np.zeros(len(n), dtype=np.float64)
    valid = (n > 0) & np.isfinite(rate)
    output[valid] = np.rint(n[valid] * rate[valid])
    return output


def _last_pre_pitch(frame: pd.DataFrame, entity: str, n_column: str) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[:0]
    index = frame.groupby(entity, observed=True, sort=False)[n_column].idxmax()
    return frame.loc[index].copy()


def _make_season_bank(train: pd.DataFrame, cutoff: int) -> dict[str, object]:
    history = train.loc[train["season"].lt(cutoff)].copy()
    if history.empty:
        return {
            "prior": 0.5,
            "pitcher_n": {},
            "pitcher_s": {},
            "batter_n": {},
            "batter_s": {},
            "component_prior": {column: 0.5 for column in PITCHER_COMPONENTS},
            "component_base_n": {},
            "component_base": {column: {} for column in PITCHER_COMPONENTS},
        }

    latest = int(history["season"].max())
    prior = float(history.loc[history["season"].eq(latest), TARGET].mean())
    pitcher = history.groupby("pitcher_id", observed=True, sort=False)[TARGET].agg(
        n="size", success="sum"
    )
    batter = history.groupby("batter_id", observed=True, sort=False)[TARGET].agg(
        n="size", success="sum"
    )
    latest_rows = history.loc[history["season"].eq(latest)]
    component_prior = {
        column: float(pd.to_numeric(latest_rows[column], errors="coerce").mean())
        for column in PITCHER_COMPONENTS
    }
    last = _last_pre_pitch(history, "pitcher_id", "asof_pitcher_n").set_index(
        "pitcher_id"
    )
    base_n = pd.to_numeric(last["asof_pitcher_n"], errors="coerce").fillna(0.0)
    component_base = {}
    for column in PITCHER_COMPONENTS:
        rate = pd.to_numeric(last[column], errors="coerce")
        component_base[column] = np.rint(base_n * rate).to_dict()
    return {
        "prior": prior,
        "pitcher_n": pitcher["n"].to_dict(),
        "pitcher_s": pitcher["success"].to_dict(),
        "batter_n": batter["n"].to_dict(),
        "batter_s": batter["success"].to_dict(),
        "component_prior": component_prior,
        "component_base_n": base_n.to_dict(),
        "component_base": component_base,
    }


def _current_season_state(rows: pd.DataFrame, bank: dict[str, object]) -> pd.DataFrame:
    rows = rows.reset_index(drop=True)
    output = pd.DataFrame(index=np.arange(len(rows)))
    prior = float(bank["prior"])

    def add_entity(
        entity: str,
        n_column: str,
        rate_column: str,
        prefix: str,
        reliability_strength: float,
    ) -> None:
        ids = rows[entity]
        history_n = ids.map(bank[f"{prefix}_n"]).fillna(0.0).to_numpy(np.float64)
        history_s = ids.map(bank[f"{prefix}_s"]).fillna(0.0).to_numpy(np.float64)
        cumulative_n = np.maximum(np.nan_to_num(_numeric(rows, n_column, 0.0)), 0.0)
        cumulative_rate = _numeric(rows, rate_column, prior)
        cumulative_rate = np.where(np.isfinite(cumulative_rate), cumulative_rate, prior)
        cumulative_s = _rounded_count(cumulative_n, cumulative_rate)
        season_n = np.maximum(cumulative_n - history_n, 0.0)
        season_s = np.clip(cumulative_s - history_s, 0.0, season_n)
        raw = np.divide(
            season_s,
            season_n,
            out=np.full(len(rows), prior, dtype=np.float64),
            where=season_n > 0,
        )
        for strength in (40.0, 80.0, 160.0):
            output[f"season__{prefix}_rate_k{int(strength)}"] = (
                season_s + strength * prior
            ) / (season_n + strength)
        output[f"season__{prefix}_raw"] = raw
        output[f"season__{prefix}_log_n"] = np.log1p(season_n)
        output[f"season__{prefix}_reliability"] = season_n / (
            season_n + reliability_strength
        )
        output[f"season__{prefix}_minus_career"] = raw - cumulative_rate

    add_entity(
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "pitcher",
        80.0,
    )
    add_entity(
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
        "batter",
        100.0,
    )
    output["season__pitcher_minus_batter_k80"] = (
        output["season__pitcher_rate_k80"] - output["season__batter_rate_k80"]
    )

    ids = rows["pitcher_id"]
    cumulative_n = np.maximum(
        np.nan_to_num(_numeric(rows, "asof_pitcher_n", 0.0)), 0.0
    )
    base_n = ids.map(bank["component_base_n"]).fillna(0.0).to_numpy(np.float64)
    window_n = np.maximum(cumulative_n - base_n, 0.0)
    components: dict[str, np.ndarray] = {}
    for column in PITCHER_COMPONENTS:
        component_prior = float(bank["component_prior"].get(column, 0.5))
        rate = _numeric(rows, column, component_prior)
        rate = np.where(np.isfinite(rate), rate, component_prior)
        cumulative = _rounded_count(cumulative_n, rate)
        baseline = (
            ids.map(bank["component_base"].get(column, {}))
            .fillna(0.0)
            .to_numpy(np.float64)
        )
        window_count = np.clip(cumulative - baseline, 0.0, window_n)
        components[column] = (window_count + 60.0 * component_prior) / (
            window_n + 60.0
        )
    output["season__component_log_n"] = np.log1p(window_n)
    output["season__component_reliability"] = window_n / (window_n + 60.0)
    output["season__strike_rate"] = components["asof_pitcher_strike_rate"]
    output["season__ball_rate"] = components["asof_pitcher_ball_rate"]
    output["season__strike_ball_margin"] = (
        output["season__strike_rate"] - output["season__ball_rate"]
    )
    output["season__middle_rate"] = components["asof_pitcher_middle_rate"]
    output["season__reverse_rate"] = components["asof_pitcher_reverse_rate"]
    mix = np.column_stack(
        [
            components["asof_pitcher_fastball_rate"],
            components["asof_pitcher_breaking_rate"],
            components["asof_pitcher_offspeed_rate"],
        ]
    )
    mix = np.clip(mix, 1e-6, 1.0)
    mix /= mix.sum(axis=1, keepdims=True)
    output["season__fastball_rate"] = mix[:, 0]
    output["season__breaking_rate"] = mix[:, 1]
    output["season__offspeed_rate"] = mix[:, 2]
    output["season__pitchmix_entropy"] = -np.sum(mix * np.log(mix), axis=1)
    return output.astype(np.float32)


def _row_state(rows: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(rows)))
    pitcher_n = np.maximum(np.nan_to_num(_numeric(rows, "asof_pitcher_n", 0.0)), 0.0)
    batter_n = np.maximum(np.nan_to_num(_numeric(rows, "asof_batter_n", 0.0)), 0.0)
    career = _numeric(rows, "asof_pitcher_success_rate", 0.5)
    prev1 = _numeric(rows, "asof_pitcher_prev1_game_success_rate", 0.5)
    prev3 = _numeric(rows, "asof_pitcher_prev3_game_success_rate", 0.5)
    prev5 = _numeric(rows, "asof_pitcher_prev5_game_success_rate", 0.5)
    recent = 0.50 * prev1 + 0.30 * prev3 + 0.20 * prev5
    strike = _numeric(rows, "asof_pitcher_strike_rate", 0.33)
    ball = _numeric(rows, "asof_pitcher_ball_rate", 0.33)
    output["self__pitcher_log_n"] = np.log1p(pitcher_n)
    output["self__batter_log_n"] = np.log1p(batter_n)
    output["self__pitcher_reliability"] = pitcher_n / (pitcher_n + 160.0)
    output["self__batter_reliability"] = batter_n / (batter_n + 160.0)
    output["self__recent_blend"] = recent
    output["self__recent_minus_career"] = recent - career
    output["self__prev1_minus_prev5"] = prev1 - prev5
    output["self__strike_ball_margin"] = strike - ball
    output["self__count_pressure"] = (
        (rows["balls_before"].to_numpy() == 3)
        | (rows["strikes_before"].to_numpy() == 2)
    ).astype(np.float32)
    output["self__same_hand"] = (
        rows["pitcher_hand"].astype(str).to_numpy()
        == rows["batter_hand"].astype(str).to_numpy()
    ).astype(np.float32)
    return output.astype(np.float32)


def _engineered(rows: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(rows)))
    prev1 = _numeric(rows, "asof_pitcher_prev1_game_success_rate", 0.5)
    prev3 = _numeric(rows, "asof_pitcher_prev3_game_success_rate", 0.5)
    prev5 = _numeric(rows, "asof_pitcher_prev5_game_success_rate", 0.5)
    recent = 0.50 * prev1 + 0.30 * prev3 + 0.20 * prev5
    output["eng__recent"] = recent
    output["eng__recent_delta"] = recent - _numeric(
        rows, "asof_pitcher_success_rate", 0.5
    )
    output["eng__strike_ball"] = _numeric(
        rows, "asof_pitcher_strike_rate", 0.33
    ) - _numeric(rows, "asof_pitcher_ball_rate", 0.33)
    output["eng__same_hand"] = (
        rows["pitcher_hand"].astype(str).to_numpy()
        == rows["batter_hand"].astype(str).to_numpy()
    ).astype(np.float32)
    output["eng__close"] = (
        np.abs(_numeric(rows, "score_diff_pitcher_team", 0.0)) <= 1
    ).astype(np.float32)
    output["eng__late"] = (_numeric(rows, "inning", 1.0) >= 7).astype(np.float32)
    return output.astype(np.float32)


def _category_pair(
    source: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = pd.DataFrame(index=np.arange(len(source)))
    right = pd.DataFrame(index=np.arange(len(audit)))
    for column in CATEGORICAL_COLUMNS:
        values = source[column].astype("string").fillna("<NA>").astype(str)
        mapping = {value: index for index, value in enumerate(values.unique())}
        left[f"cat__{column}"] = values.map(mapping).astype(np.float32)
        right[f"cat__{column}"] = (
            audit[column]
            .astype("string")
            .fillna("<NA>")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(np.float32)
        )
    return left, right


def _feature_pair(
    train: pd.DataFrame,
    source: pd.DataFrame,
    audit: pd.DataFrame,
    source_year: int,
    audit_year: int,
    *,
    include_categories: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [column for column in NUMERIC_COLUMNS if column in train]
    left = source[numeric].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    right = audit[numeric].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    left = pd.concat(
        [
            left,
            _row_state(source),
            _current_season_state(source, _make_season_bank(train, source_year)),
        ],
        axis=1,
    )
    right = pd.concat(
        [
            right,
            _row_state(audit),
            _current_season_state(audit, _make_season_bank(train, audit_year)),
        ],
        axis=1,
    )
    if include_categories:
        cat_left, cat_right = _category_pair(source, audit)
        left = pd.concat([left, cat_left], axis=1)
        right = pd.concat([right, cat_right], axis=1)
    if list(left.columns) != list(right.columns):
        raise ValueError("source/audit feature columns differ")
    return left.astype(np.float32), right.astype(np.float32)


def _trend_pair(
    source: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [column for column in NUMERIC_COLUMNS if column in source]
    left = source[numeric].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    right = audit[numeric].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    left = pd.concat([left, _engineered(source)], axis=1)
    right = pd.concat([right, _engineered(audit)], axis=1)
    cat_left, cat_right = _category_pair(source, audit)
    return (
        pd.concat([left, cat_left], axis=1).astype(np.float32),
        pd.concat([right, cat_right], axis=1).astype(np.float32),
    )


def _core_slope(train: pd.DataFrame, source_year: int) -> float:
    means = (
        train.loc[train["season"].le(source_year) & train["domain3"].eq("R_CORE")]
        .groupby("season", observed=True)[TARGET]
        .mean()
        .tail(4)
    )
    return float(np.polyfit(means.index.to_numpy(float), means.to_numpy(float), 1)[0])


def fit_predict(
    train: pd.DataFrame, source_year: int, audit_year: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    source = train.loc[train["season"].eq(source_year)].reset_index(drop=True)
    audit = train.loc[train["season"].eq(audit_year)].reset_index(drop=True)
    target = source[TARGET].to_numpy(np.float64)

    left, right = _feature_pair(
        train,
        source,
        audit,
        source_year,
        audit_year,
        include_categories=True,
    )
    classifier = lgb.LGBMClassifier(
        objective="binary",
        verbosity=-1,
        n_jobs=6,
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=7,
        min_child_samples=350,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=6.0,
        max_bin=127,
        random_state=918,
    )
    classifier.fit(left, target)
    exact_lgb = classifier.predict_proba(right)[:, 1]
    del classifier, left, right
    gc.collect()

    left, right = _feature_pair(
        train,
        source,
        audit,
        source_year,
        audit_year,
        include_categories=False,
    )
    median = left.median().fillna(0.0)
    left_values = left.fillna(median).to_numpy(np.float64)
    right_values = right.fillna(median).to_numpy(np.float64)
    scaler = StandardScaler()
    left_values = np.clip(scaler.fit_transform(left_values), -8.0, 8.0)
    right_values = np.clip(scaler.transform(right_values), -8.0, 8.0)
    ridge = Ridge(alpha=10_000.0, solver="lsqr", max_iter=100, tol=1e-3)
    ridge.fit(left_values, target)
    exact_ridge = np.clip(ridge.predict(right_values), 0.001, 0.999)
    del ridge, left, right, left_values, right_values, scaler
    gc.collect()

    left, right = _trend_pair(source, audit)
    source_means = source.groupby("domain3", observed=True)[TARGET].mean().to_dict()
    slope = _core_slope(train, source_year)
    source_prior = source["domain3"].map(source_means).to_numpy(np.float64)
    audit_prior = audit["domain3"].map(
        {domain: value + slope for domain, value in source_means.items()}
    ).to_numpy(np.float64)
    trend = lgb.LGBMRegressor(
        objective="regression_l2",
        verbosity=-1,
        n_jobs=6,
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=7,
        min_child_samples=300,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=5.0,
        max_bin=127,
        random_state=77,
    )
    trend.fit(left, target - source_prior)
    trend_lgb = np.clip(audit_prior + trend.predict(right), 0.001, 0.999)
    del trend, left, right
    gc.collect()

    parts = {
        "exact_lgb": np.asarray(exact_lgb, dtype=np.float64),
        "exact_ridge": np.asarray(exact_ridge, dtype=np.float64),
        "trend_lgb": np.asarray(trend_lgb, dtype=np.float64),
    }
    prediction = np.mean(np.column_stack(list(parts.values())), axis=1)
    return np.clip(prediction, 1e-6, 1.0 - 1e-6), parts


def run(project: Path, research_project: Path, output_dir: Path) -> pd.DataFrame:
    project = project.resolve()
    research_project = research_project.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    artifact_root = research_project / "artifacts" / "top10_20260814"
    paths = Paths(
        corrected_cb=artifact_root / "corrected_cb_oof_20260814_01",
        advanced=artifact_root / "advanced_domain_residual_20260814_01",
        legacy_root=artifact_root,
    )
    rows: list[dict[str, object]] = []
    for source_year, audit_year in TRANSITIONS:
        candidate, parts = fit_predict(train, source_year, audit_year)
        fold = _load_base(train, paths, audit_year)
        audit_index = np.flatnonzero(train["season"].eq(audit_year).to_numpy())
        if not np.array_equal(fold["train_index"].to_numpy(np.int64), audit_index):
            raise ValueError(f"OOF/audit ordering differs for {audit_year}")
        target = fold["target"].to_numpy(np.float64)
        base = fold["base"].to_numpy(np.float64)
        audit = train.iloc[audit_index].reset_index(drop=True)
        np.savez_compressed(
            output_dir / f"recent_shared_o{audit_year}.npz",
            train_index=audit_index,
            target=target,
            base=base,
            prediction=candidate,
            **parts,
        )
        for domain in ("ALL", "R_CORE", "R_ANCHOR", "F"):
            mask = (
                np.ones(len(audit), dtype=bool)
                if domain == "ALL"
                else audit["domain3"].eq(domain).to_numpy()
            )
            for name, prediction in {"shared": candidate, **parts}.items():
                rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "domain": domain,
                        "candidate": name,
                        "weight": 1.0,
                        "n_rows": int(mask.sum()),
                        "base_bss": _bss(target[mask], base[mask]),
                        "candidate_bss": _bss(target[mask], prediction[mask]),
                        "gain": _bss(target[mask], prediction[mask])
                        - _bss(target[mask], base[mask]),
                        "prediction_mean": float(prediction[mask].mean()),
                    }
                )
            for weight in np.arange(0.05, 0.801, 0.05):
                prediction = np.clip(
                    base + weight * (candidate - base), 1e-6, 1.0 - 1e-6
                )
                rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "domain": domain,
                        "candidate": "shared_blend",
                        "weight": float(weight),
                        "n_rows": int(mask.sum()),
                        "base_bss": _bss(target[mask], base[mask]),
                        "candidate_bss": _bss(target[mask], prediction[mask]),
                        "gain": _bss(target[mask], prediction[mask])
                        - _bss(target[mask], base[mask]),
                        "prediction_mean": float(prediction[mask].mean()),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "temporal_results.csv", index=False)
    blend = result.loc[result["candidate"].eq("shared_blend")]
    robust = (
        blend.groupby(["domain", "weight"], observed=True)["gain"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["domain", "min_gain", "mean_gain"], ascending=[True, False, False])
    )
    robust.to_csv(output_dir / "robust_blends.csv", index=False)
    summary = {
        "protocol": "RECENT_SHARED_EXACT_ASOF_STRICT_TEMPORAL_V1",
        "transitions": [f"{source}->{audit}" for source, audit in TRANSITIONS],
        "best_by_domain": robust.groupby("domain", observed=True).head(1).to_dict(
            orient="records"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--research-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.research_project, args.output_dir)


if __name__ == "__main__":
    main()
