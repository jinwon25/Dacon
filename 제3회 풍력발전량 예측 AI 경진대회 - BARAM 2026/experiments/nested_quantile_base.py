from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.blocked_rolling_validation import assign_issue_blocks, load_issue_times
from src.feature_cache import load_or_build_features
from src.features import TIME_COL
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import select_feature_columns


OUTER_SEASONS = ("2024-DJF", "2024-MAM", "2024-JJA", "2024-SON")
DEFAULT_ALPHAS = (0.55, 0.60, 0.65, 0.70, 0.75)
DEFAULT_SEEDS = (42, 202, 2026)
SCADA_GROUPS = {
    "kpx_group_1": ("vestas", tuple(range(1, 7)), 50),
    "kpx_group_2": ("vestas", tuple(range(7, 13)), 50),
    "kpx_group_3": ("unison", tuple(range(1, 6)), 60),
}


@dataclass(frozen=True)
class Fold:
    name: str
    train: np.ndarray
    valid: np.ndarray
    inner_name: str
    inner_train: np.ndarray
    inner_valid: np.ndarray


@dataclass(frozen=True)
class Selection:
    objective: str
    alpha: float | None
    train_variant: str
    iterations: int
    score: float
    one_minus_nmae: float
    ficr: float


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise ValueError("alphas must contain values strictly between zero and one")
    return values


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("seeds must contain at least one integer")
    return values


def _ordered_issue_seasons(
    index: pd.DatetimeIndex, issue_times: pd.DatetimeIndex
) -> tuple[np.ndarray, list[str]]:
    _, seasons = assign_issue_blocks(index, issue_times)
    frame = pd.DataFrame({"issue": issue_times, "season": seasons})
    order = (
        frame.drop_duplicates("issue")
        .groupby("season", sort=False)["issue"]
        .median()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    return seasons, order


def make_nested_folds(
    index: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
    label_available: np.ndarray,
    *,
    purge_hours: int = 24,
    outer_seasons: Iterable[str] = OUTER_SEASONS,
    minimum_train_rows: int = 2_000,
    minimum_valid_rows: int = 500,
) -> list[Fold]:
    """Build one untouched outer season and one prior-season inner holdout.

    Complete NWP issue cycles remain together. The outer fold is never used for
    alpha, train-row policy, early stopping, or iteration selection.
    """
    label_available = np.asarray(label_available, dtype=bool)
    if len(index) != len(issue_times) or len(index) != len(label_available):
        raise ValueError("index, issue_times, and label_available must align")
    seasons, order = _ordered_issue_seasons(index, issue_times)
    purge = pd.Timedelta(hours=purge_hours)
    folds: list[Fold] = []
    for outer_name in outer_seasons:
        outer_valid = label_available & (seasons == outer_name)
        if int(outer_valid.sum()) < minimum_valid_rows:
            continue
        outer_start = pd.Timestamp(np.min(np.asarray(issue_times)[outer_valid]))
        outer_train = label_available & (np.asarray(issue_times) < outer_start - purge)
        if int(outer_train.sum()) < minimum_train_rows:
            continue
        candidates = [
            name
            for name in order
            if name != outer_name
            and np.any(label_available & (seasons == name))
            and pd.Timestamp(
                np.max(np.asarray(issue_times)[label_available & (seasons == name)])
            )
            < outer_start - purge
        ]
        if not candidates:
            continue
        inner_name = candidates[-1]
        inner_valid = outer_train & (seasons == inner_name)
        if int(inner_valid.sum()) < minimum_valid_rows:
            continue
        inner_start = pd.Timestamp(np.min(np.asarray(issue_times)[inner_valid]))
        inner_train = outer_train & (
            np.asarray(issue_times) < inner_start - purge
        )
        if int(inner_train.sum()) < minimum_train_rows:
            continue
        folds.append(
            Fold(
                name=outer_name,
                train=outer_train,
                valid=outer_valid,
                inner_name=inner_name,
                inner_train=inner_train,
                inner_valid=inner_valid,
            )
        )
    if not folds:
        raise ValueError("No nested folds satisfy the row and issue-block guards")
    return folds


def _lgb_model(
    objective: str,
    *,
    alpha: float | None,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> lgb.LGBMRegressor:
    params: dict[str, Any] = {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.035,
        "num_leaves": 40,
        "max_depth": -1,
        "min_child_samples": 55,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.72,
        "reg_alpha": 0.10,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": -1,
        "force_col_wise": True,
    }
    if objective == "quantile":
        params["alpha"] = float(alpha)
    return lgb.LGBMRegressor(**params)


def _cat_model(
    *,
    alpha: float,
    seed: int,
    iterations: int,
    thread_count: int,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        eval_metric=f"Quantile:alpha={alpha}",
        iterations=iterations,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=10.0,
        random_seed=seed,
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
        thread_count=thread_count,
    )


def _training_mask(
    base: np.ndarray,
    y: np.ndarray,
    capacity: float,
    variant: str,
    curtailment: np.ndarray | None,
) -> np.ndarray:
    mask = np.asarray(base, dtype=bool).copy()
    if variant == "eligible_only":
        mask &= y >= 0.10 * capacity
    elif variant != "all":
        raise ValueError(f"Unknown train variant: {variant}")
    if curtailment is not None:
        mask &= ~np.asarray(curtailment, dtype=bool)
    return mask


def _fit_inner_lgb(
    X: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    capacity: float,
    *,
    objective: str,
    alpha: float | None,
    variant: str,
    curtailment: np.ndarray | None,
    seed: int,
    maximum_iterations: int,
    early_stopping_rounds: int,
    n_jobs: int,
) -> tuple[Selection, np.ndarray]:
    fit_rows = _training_mask(train, y, capacity, variant, curtailment)
    model = _lgb_model(
        objective,
        alpha=alpha,
        seed=seed,
        n_estimators=maximum_iterations,
        n_jobs=n_jobs,
    )
    eval_metric = "quantile" if objective == "quantile" else "l1"
    model.fit(
        X.loc[fit_rows],
        y[fit_rows],
        eval_set=[(X.loc[valid], y[valid])],
        eval_metric=eval_metric,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    prediction = np.clip(model.predict(X.loc[valid]), 0.0, capacity)
    metric = evaluate_group(y[valid], prediction, capacity)
    selection = Selection(
        objective=objective,
        alpha=alpha,
        train_variant=variant,
        iterations=max(100, int(model.best_iteration_ or maximum_iterations)),
        score=metric.score,
        one_minus_nmae=metric.one_minus_nmae,
        ficr=metric.ficr,
    )
    return selection, prediction


def choose_selection(records: list[Selection]) -> Selection:
    if not records:
        raise ValueError("No model-selection records")
    return max(
        records,
        key=lambda item: (
            item.score,
            item.ficr,
            item.one_minus_nmae,
            -abs((item.alpha if item.alpha is not None else 0.5) - 0.5),
            item.train_variant == "all",
        ),
    )


def _fit_fixed_lgb_ensemble(
    X: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    query: np.ndarray | pd.DataFrame,
    capacity: float,
    *,
    selection: Selection,
    seeds: tuple[int, ...],
    curtailment: np.ndarray | None,
    n_jobs: int,
) -> np.ndarray:
    fit_rows = _training_mask(
        train, y, capacity, selection.train_variant, curtailment
    )
    X_query = X.loc[query] if isinstance(query, np.ndarray) else query
    predictions = []
    for seed in seeds:
        model = _lgb_model(
            selection.objective,
            alpha=selection.alpha,
            seed=seed,
            n_estimators=selection.iterations,
            n_jobs=n_jobs,
        )
        model.fit(
            X.loc[fit_rows],
            y[fit_rows],
            callbacks=[lgb.log_evaluation(0)],
        )
        predictions.append(model.predict(X_query))
    return np.clip(np.mean(predictions, axis=0), 0.0, capacity)


def _fit_inner_cat_iterations(
    X: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    capacity: float,
    *,
    selection: Selection,
    curtailment: np.ndarray | None,
    seed: int,
    maximum_iterations: int,
    thread_count: int,
) -> int:
    fit_rows = _training_mask(
        train, y, capacity, selection.train_variant, curtailment
    )
    model = _cat_model(
        alpha=float(selection.alpha),
        seed=seed,
        iterations=maximum_iterations,
        thread_count=thread_count,
    )
    model.fit(
        X.loc[fit_rows],
        y[fit_rows],
        eval_set=(X.loc[valid], y[valid]),
        use_best_model=True,
    )
    return max(100, int(model.get_best_iteration() + 1))


def _fit_fixed_cat_ensemble(
    X: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    query: np.ndarray | pd.DataFrame,
    capacity: float,
    *,
    selection: Selection,
    iterations: int,
    seeds: tuple[int, ...],
    curtailment: np.ndarray | None,
    thread_count: int,
) -> np.ndarray:
    fit_rows = _training_mask(
        train, y, capacity, selection.train_variant, curtailment
    )
    X_query = X.loc[query] if isinstance(query, np.ndarray) else query
    predictions = []
    for seed in seeds:
        model = _cat_model(
            alpha=float(selection.alpha),
            seed=seed,
            iterations=iterations,
            thread_count=thread_count,
        )
        model.fit(X.loc[fit_rows], y[fit_rows])
        predictions.append(model.predict(X_query))
    return np.clip(np.mean(predictions, axis=0), 0.0, capacity)


def load_measured_wind(
    data_dir: Path, index: pd.DatetimeIndex
) -> dict[str, np.ndarray]:
    raw: dict[str, pd.DataFrame] = {}
    for prefix in ("vestas", "unison"):
        frame = pd.read_csv(
            data_dir / "train" / f"scada_{prefix}_train.csv",
            encoding="utf-8-sig",
        )
        frame["kst_dtm"] = pd.to_datetime(frame["kst_dtm"])
        raw[prefix] = frame
    output: dict[str, np.ndarray] = {}
    for group, (prefix, turbine_ids, shift_minutes) in SCADA_GROUPS.items():
        frame = raw[prefix]
        columns = [f"{prefix}_wtg{item:02d}_ws" for item in turbine_ids]
        values = frame[columns].where(
            (frame[columns] >= 0.0) & (frame[columns] <= 60.0)
        )
        timestamp = (
            frame["kst_dtm"] + pd.Timedelta(minutes=shift_minutes)
        ).dt.floor("h")
        hourly = values.groupby(timestamp).mean().mean(axis=1)
        output[group] = hourly.reindex(index).to_numpy(dtype=float)
    return output


def _wind_columns(X: pd.DataFrame, target: str) -> list[str]:
    selected = select_feature_columns(X, target, "own_idw")
    tokens = (
        "ws",
        "_u",
        "_v",
        "10u",
        "10v",
        "MU",
        "MV",
        "gust",
        "surface_0_sp",
        "prmsl",
        "_2_t",
        "_2_2t",
        "blh",
    )
    calendar = {
        "hour",
        "month",
        "dayofweek",
        "lead_hour",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    }
    return [column for column in selected if column in calendar or any(t in column for t in tokens)]


def build_rolling_wind_calibration(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    issue_times: pd.DatetimeIndex,
    measured: dict[str, np.ndarray],
    *,
    purge_hours: int,
    n_jobs: int,
    seed: int = 31_000,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Create past-only seasonal OOF wind estimates and full-fit test estimates."""
    seasons, order = _ordered_issue_seasons(X_train.index, issue_times)
    purge = pd.Timedelta(hours=purge_hours)
    train_features: dict[str, np.ndarray] = {}
    test_features: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    for group_i, group in enumerate(CAPACITY_KWH, start=1):
        columns = _wind_columns(X_train, group)
        Xg = X_train[columns]
        Xtg = X_test.reindex(columns=columns)
        truth = measured[group]
        oof = np.full(len(X_train), np.nan, dtype=float)
        fold_report: dict[str, Any] = {}
        for season_i, season in enumerate(order, start=1):
            valid = seasons == season
            if not np.any(valid):
                continue
            start = pd.Timestamp(np.min(np.asarray(issue_times)[valid]))
            train = np.isfinite(truth) & (
                np.asarray(issue_times) < start - purge
            )
            if int(train.sum()) < 2_000:
                continue
            model = _lgb_model(
                "l1",
                alpha=None,
                seed=seed + 100 * group_i + season_i,
                n_estimators=350,
                n_jobs=n_jobs,
            )
            model.fit(Xg.loc[train], truth[train], callbacks=[lgb.log_evaluation(0)])
            oof[valid] = model.predict(Xg.loc[valid])
            comparable = valid & np.isfinite(truth)
            fold_report[season] = {
                "train_rows": int(train.sum()),
                "valid_rows": int(comparable.sum()),
                "mae": float(np.mean(np.abs(truth[comparable] - oof[comparable])))
                if comparable.any()
                else None,
                "correlation": float(np.corrcoef(truth[comparable], oof[comparable])[0, 1])
                if int(comparable.sum()) > 2
                else None,
            }
        full = np.isfinite(truth)
        if int(full.sum()) < 2_000:
            raise ValueError(f"Measured SCADA wind coverage is insufficient for {group}")
        model = _lgb_model(
            "l1",
            alpha=None,
            seed=seed + 100 * group_i + 99,
            n_estimators=450,
            n_jobs=n_jobs,
        )
        model.fit(Xg.loc[full], truth[full], callbacks=[lgb.log_evaluation(0)])
        train_features[group] = oof.astype("float32")
        test_features[group] = model.predict(Xtg).astype("float32")
        report[group] = {
            "feature_columns": len(columns),
            "measured_rows": int(full.sum()),
            "oof_rows": int(np.isfinite(oof).sum()),
            "folds": fold_report,
        }
    return train_features, test_features, report


def curtailment_mask(
    y: np.ndarray,
    measured_wind: np.ndarray,
    training_rows: np.ndarray,
    capacity: float,
    *,
    bin_width: float = 0.5,
    minimum_bin_rows: int = 20,
    wind_floor: float = 6.0,
    minimum_expected_cf: float = 0.25,
    actual_to_expected_ratio: float = 0.5,
) -> np.ndarray:
    """Flag output-limited rows using only the supplied training partition."""
    y = np.asarray(y, dtype=float)
    wind = np.asarray(measured_wind, dtype=float)
    training = (
        np.asarray(training_rows, dtype=bool)
        & np.isfinite(y)
        & np.isfinite(wind)
    )
    result = np.zeros(len(y), dtype=bool)
    if int(training.sum()) < 200:
        return result
    edges = np.arange(0.0, 30.0 + bin_width, bin_width)
    positions = np.clip(np.digitize(wind, edges) - 1, 0, len(edges) - 2)
    curve = np.full(len(edges) - 1, np.nan, dtype=float)
    cf = y / capacity
    for bin_i in range(len(curve)):
        rows = training & (positions == bin_i)
        if int(rows.sum()) >= minimum_bin_rows:
            curve[bin_i] = float(np.median(cf[rows]))
    finite = np.isfinite(curve)
    if int(finite.sum()) < 2:
        return result
    curve = np.interp(np.arange(len(curve)), np.flatnonzero(finite), curve[finite])
    expected = curve[positions]
    result = (
        training
        & (wind >= wind_floor)
        & (expected >= minimum_expected_cf)
        & (cf < actual_to_expected_ratio * expected)
    )
    return result


def _consensus_selection(selections: list[Selection]) -> Selection:
    if not selections:
        raise ValueError("Cannot make a final selection without outer selections")
    alpha_values = [float(item.alpha) for item in selections if item.alpha is not None]
    variant_values = [item.train_variant for item in selections]
    alpha_counts = Counter(alpha_values)
    variant_counts = Counter(variant_values)
    alpha = max(alpha_counts, key=lambda value: (alpha_counts[value], -abs(value - 0.5)))
    variant = max(
        variant_counts,
        key=lambda value: (variant_counts[value], value == "all"),
    )
    iterations = max(100, int(np.median([item.iterations for item in selections])))
    return Selection(
        objective="quantile",
        alpha=alpha,
        train_variant=variant,
        iterations=iterations,
        score=float(np.mean([item.score for item in selections])),
        one_minus_nmae=float(np.mean([item.one_minus_nmae for item in selections])),
        ficr=float(np.mean([item.ficr for item in selections])),
    )


def _competition_delta(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    base = evaluate_competition(truth, reference)
    new = evaluate_competition(truth, candidate)
    return {
        "reference": base,
        "candidate": new,
        "delta": {
            "score": float(new["score"] - base["score"]),
            "one_minus_nmae": float(
                new["one_minus_nmae"] - base["one_minus_nmae"]
            ),
            "ficr": float(new["ficr"] - base["ficr"]),
        },
    }


def _issue_bootstrap(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    issue_times: pd.DatetimeIndex,
    seasons: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    issue_values = np.asarray(issue_times)
    stratum_issues = {
        season: np.unique(issue_values[seasons == season])
        for season in dict.fromkeys(seasons)
    }
    positions = {
        (season, issue): np.flatnonzero(
            (seasons == season) & (issue_values == issue)
        )
        for season, issues in stratum_issues.items()
        for issue in issues
    }
    values = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled: list[np.ndarray] = []
        for season, issues in stratum_issues.items():
            drawn = rng.choice(issues, size=len(issues), replace=True)
            sampled.extend(positions[(season, issue)] for issue in drawn)
        rows = np.concatenate(sampled)
        base = evaluate_competition(
            {group: values_[rows] for group, values_ in truth.items()},
            {group: values_[rows] for group, values_ in reference.items()},
        )
        new = evaluate_competition(
            {group: values_[rows] for group, values_ in truth.items()},
            {group: values_[rows] for group, values_ in candidate.items()},
        )
        values[iteration] = new["score"] - base["score"]
    return {
        "n_bootstrap": n_bootstrap,
        "positive_fraction": float(np.mean(values > 0.0)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="artifacts_final/feature_cache")
    parser.add_argument("--artifact-dir", default="artifacts_final/base_v2/nested_quantile")
    parser.add_argument("--feature-set", default="own_idw", choices=("base", "own_idw", "own_idw_nohub", "full"))
    parser.add_argument("--alphas", default=",".join(str(item) for item in DEFAULT_ALPHAS))
    parser.add_argument("--seeds", default=",".join(str(item) for item in DEFAULT_SEEDS))
    parser.add_argument("--train-variants", default="all,eligible_only")
    parser.add_argument("--purge-hours", type=int, default=24)
    parser.add_argument("--maximum-iterations", type=int, default=1_400)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--include-catboost", action="store_true")
    parser.add_argument("--catboost-maximum-iterations", type=int, default=1_800)
    parser.add_argument("--ws-cal", action="store_true")
    parser.add_argument("--group3-curtailment", action="store_true")
    parser.add_argument("--write-submission-if-qualified", action="store_true")
    parser.add_argument(
        "--output-submission",
        default="submissions/base_v2_nested_quantile.csv",
    )
    args = parser.parse_args()

    alphas = _parse_floats(args.alphas)
    seeds = _parse_ints(args.seeds)
    variants = tuple(
        item.strip() for item in args.train_variants.split(",") if item.strip()
    )
    if not variants or any(item not in {"all", "eligible_only"} for item in variants):
        raise ValueError("train variants must be all and/or eligible_only")

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    X_all = load_or_build_features(data_dir, "train", args.cache_dir)
    X_test_all = load_or_build_features(data_dir, "test", args.cache_dir)
    labels = pd.read_csv(
        data_dir / "train" / "train_labels.csv", encoding="utf-8-sig"
    )
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X_all.index)
    issue_times = load_issue_times(
        data_dir / "train" / "gfs_train.csv", X_all.index
    )
    issue_seasons, _ = _ordered_issue_seasons(X_all.index, issue_times)

    measured_wind = (
        load_measured_wind(data_dir, X_all.index)
        if args.ws_cal or args.group3_curtailment
        else {}
    )
    ws_train: dict[str, np.ndarray] = {}
    ws_test: dict[str, np.ndarray] = {}
    ws_report: dict[str, Any] = {}
    if args.ws_cal:
        print("Building past-only SCADA wind-calibration features...", flush=True)
        ws_train, ws_test, ws_report = build_rolling_wind_calibration(
            X_all,
            X_test_all,
            issue_times,
            measured_wind,
            purge_hours=args.purge_hours,
            n_jobs=args.n_jobs,
        )

    group_reports: dict[str, Any] = {}
    oof_truth: dict[str, np.ndarray] = {}
    oof_l1: dict[str, np.ndarray] = {}
    oof_candidate: dict[str, np.ndarray] = {}
    oof_index: pd.DatetimeIndex | None = None
    final_test: dict[str, np.ndarray] = {}

    for group_i, (group, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        print(f"[{group}] preparing nested folds...", flush=True)
        columns = select_feature_columns(X_all, group, args.feature_set)
        X = X_all[columns].copy()
        X_test = X_test_all.reindex(columns=columns).copy()
        if args.ws_cal:
            X["scada_ws_cal_oof"] = ws_train[group]
            X_test["scada_ws_cal_oof"] = ws_test[group]
        y = labels[group].to_numpy(dtype=float)
        available = np.isfinite(y)
        folds = make_nested_folds(
            X.index,
            issue_times,
            available,
            purge_hours=args.purge_hours,
        )
        candidate_oof = np.full(len(X), np.nan, dtype=float)
        l1_oof = np.full(len(X), np.nan, dtype=float)
        fold_reports: list[dict[str, Any]] = []
        outer_selections: list[Selection] = []
        cat_iterations: list[int] = []

        for fold_i, fold in enumerate(folds, start=1):
            inner_curtailment = None
            outer_curtailment = None
            if args.group3_curtailment and group == "kpx_group_3":
                inner_curtailment = curtailment_mask(
                    y, measured_wind[group], fold.inner_train, capacity
                )
                outer_curtailment = curtailment_mask(
                    y, measured_wind[group], fold.train, capacity
                )

            quantile_records: list[Selection] = []
            quantile_diagnostics: list[dict[str, Any]] = []
            l1_records: list[Selection] = []
            for variant in variants:
                l1_record, _ = _fit_inner_lgb(
                    X,
                    y,
                    fold.inner_train,
                    fold.inner_valid,
                    capacity,
                    objective="l1",
                    alpha=None,
                    variant=variant,
                    curtailment=inner_curtailment,
                    seed=10_000 + 100 * group_i + fold_i,
                    maximum_iterations=args.maximum_iterations,
                    early_stopping_rounds=args.early_stopping_rounds,
                    n_jobs=args.n_jobs,
                )
                l1_records.append(l1_record)
                for alpha_i, alpha in enumerate(alphas, start=1):
                    record, _ = _fit_inner_lgb(
                        X,
                        y,
                        fold.inner_train,
                        fold.inner_valid,
                        capacity,
                        objective="quantile",
                        alpha=alpha,
                        variant=variant,
                        curtailment=inner_curtailment,
                        seed=20_000 + 1_000 * group_i + 100 * fold_i + alpha_i,
                        maximum_iterations=args.maximum_iterations,
                        early_stopping_rounds=args.early_stopping_rounds,
                        n_jobs=args.n_jobs,
                    )
                    quantile_records.append(record)
                    quantile_diagnostics.append(asdict(record))
            selected_l1 = choose_selection(l1_records)
            selected_quantile = choose_selection(quantile_records)
            selected_quantile = Selection(
                **{
                    **asdict(selected_quantile),
                    "iterations": max(100, int(round(selected_quantile.iterations * 1.10))),
                }
            )
            selected_l1 = Selection(
                **{
                    **asdict(selected_l1),
                    "iterations": max(100, int(round(selected_l1.iterations * 1.10))),
                }
            )
            outer_selections.append(selected_quantile)

            l1_prediction = _fit_fixed_lgb_ensemble(
                X,
                y,
                fold.train,
                fold.valid,
                capacity,
                selection=selected_l1,
                seeds=seeds,
                curtailment=outer_curtailment,
                n_jobs=args.n_jobs,
            )
            lgb_prediction = _fit_fixed_lgb_ensemble(
                X,
                y,
                fold.train,
                fold.valid,
                capacity,
                selection=selected_quantile,
                seeds=seeds,
                curtailment=outer_curtailment,
                n_jobs=args.n_jobs,
            )
            cat_iteration = None
            candidate_prediction = lgb_prediction
            if args.include_catboost:
                cat_iteration = _fit_inner_cat_iterations(
                    X,
                    y,
                    fold.inner_train,
                    fold.inner_valid,
                    capacity,
                    selection=selected_quantile,
                    curtailment=inner_curtailment,
                    seed=30_000 + 100 * group_i + fold_i,
                    maximum_iterations=args.catboost_maximum_iterations,
                    thread_count=args.n_jobs,
                )
                cat_iteration = max(100, int(round(cat_iteration * 1.10)))
                cat_iterations.append(cat_iteration)
                cat_prediction = _fit_fixed_cat_ensemble(
                    X,
                    y,
                    fold.train,
                    fold.valid,
                    capacity,
                    selection=selected_quantile,
                    iterations=cat_iteration,
                    seeds=seeds,
                    curtailment=outer_curtailment,
                    thread_count=args.n_jobs,
                )
                candidate_prediction = np.clip(
                    0.5 * lgb_prediction + 0.5 * cat_prediction,
                    0.0,
                    capacity,
                )

            l1_oof[fold.valid] = l1_prediction
            candidate_oof[fold.valid] = candidate_prediction
            l1_metric = evaluate_group(y[fold.valid], l1_prediction, capacity)
            candidate_metric = evaluate_group(
                y[fold.valid], candidate_prediction, capacity
            )
            fold_report = {
                "outer_season": fold.name,
                "inner_selection_season": fold.inner_name,
                "train_rows": int(fold.train.sum()),
                "valid_rows": int(fold.valid.sum()),
                "inner_train_rows": int(fold.inner_train.sum()),
                "inner_valid_rows": int(fold.inner_valid.sum()),
                "selected_l1": asdict(selected_l1),
                "selected_quantile": asdict(selected_quantile),
                "catboost_iterations": cat_iteration,
                "inner_quantile_candidates": quantile_diagnostics,
                "outer_l1": l1_metric.to_dict(),
                "outer_candidate": candidate_metric.to_dict(),
                "outer_delta": {
                    "score": candidate_metric.score - l1_metric.score,
                    "one_minus_nmae": candidate_metric.one_minus_nmae
                    - l1_metric.one_minus_nmae,
                    "ficr": candidate_metric.ficr - l1_metric.ficr,
                },
                "inner_curtailment_rows": int(inner_curtailment.sum())
                if inner_curtailment is not None
                else 0,
                "outer_curtailment_rows": int(outer_curtailment.sum())
                if outer_curtailment is not None
                else 0,
            }
            fold_reports.append(fold_report)
            print(
                f"[{group}] {fold.name}: alpha={selected_quantile.alpha:.2f} "
                f"variant={selected_quantile.train_variant} "
                f"score {l1_metric.score:.6f}->{candidate_metric.score:.6f} "
                f"delta={candidate_metric.score - l1_metric.score:+.6f}",
                flush=True,
            )

        evaluated = (
            available & np.isfinite(l1_oof) & np.isfinite(candidate_oof)
        )
        if not evaluated.any():
            raise RuntimeError(f"No outer OOF predictions for {group}")
        if oof_index is None:
            oof_index = X.index[evaluated]
        elif not oof_index.equals(X.index[evaluated]):
            raise ValueError("Nested OOF timestamps differ across groups")
        oof_truth[group] = y[evaluated]
        oof_l1[group] = l1_oof[evaluated]
        oof_candidate[group] = candidate_oof[evaluated]

        final_selection = _consensus_selection(outer_selections)
        final_curtailment = None
        if args.group3_curtailment and group == "kpx_group_3":
            final_curtailment = curtailment_mask(
                y, measured_wind[group], available, capacity
            )
        test_lgb = _fit_fixed_lgb_ensemble(
            X,
            y,
            available,
            X_test,
            capacity,
            selection=final_selection,
            seeds=seeds,
            curtailment=final_curtailment,
            n_jobs=args.n_jobs,
        )
        test_prediction = test_lgb
        final_cat_iteration = None
        if args.include_catboost:
            final_cat_iteration = max(
                100, int(np.median(cat_iterations))
            )
            test_cat = _fit_fixed_cat_ensemble(
                X,
                y,
                available,
                X_test,
                capacity,
                selection=final_selection,
                iterations=final_cat_iteration,
                seeds=seeds,
                curtailment=final_curtailment,
                thread_count=args.n_jobs,
            )
            test_prediction = np.clip(
                0.5 * test_lgb + 0.5 * test_cat, 0.0, capacity
            )
        final_test[group] = test_prediction
        group_reports[group] = {
            "capacity": capacity,
            "features": int(X.shape[1]),
            "folds": fold_reports,
            "final_selection": asdict(final_selection),
            "final_catboost_iterations": final_cat_iteration,
            "final_train_rows": int(available.sum()),
            "final_curtailment_rows": int(final_curtailment.sum())
            if final_curtailment is not None
            else 0,
            "outer_l1": evaluate_group(
                y[evaluated], l1_oof[evaluated], capacity
            ).to_dict(),
            "outer_candidate": evaluate_group(
                y[evaluated], candidate_oof[evaluated], capacity
            ).to_dict(),
        }

    if oof_index is None:
        raise RuntimeError("No common OOF index")
    overall = _competition_delta(oof_truth, oof_l1, oof_candidate)
    aligned_issue = load_issue_times(
        data_dir / "train" / "gfs_train.csv", oof_index
    )
    aligned_seasons, _ = _ordered_issue_seasons(oof_index, aligned_issue)
    seasonal: dict[str, Any] = {}
    for season in dict.fromkeys(aligned_seasons):
        rows = aligned_seasons == season
        seasonal[season] = _competition_delta(
            {group: values[rows] for group, values in oof_truth.items()},
            {group: values[rows] for group, values in oof_l1.items()},
            {group: values[rows] for group, values in oof_candidate.items()},
        )
    bootstrap = _issue_bootstrap(
        oof_truth,
        oof_l1,
        oof_candidate,
        aligned_issue,
        aligned_seasons,
        n_bootstrap=args.n_bootstrap,
        seed=20260724,
    )
    gates = {
        "score_positive": overall["delta"]["score"] > 0.0,
        "one_minus_nmae_nonnegative": overall["delta"]["one_minus_nmae"] >= 0.0,
        "ficr_nonnegative": overall["delta"]["ficr"] >= 0.0,
        "worst_season_nonnegative": min(
            item["delta"]["score"] for item in seasonal.values()
        )
        >= 0.0,
        "bootstrap_q05_nonnegative": bootstrap["q05"] >= 0.0,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"] >= 0.90,
    }
    qualified = bool(all(gates.values()))

    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    if len(sample) != len(X_test_all):
        raise ValueError("Sample submission and test features have different row counts")
    output_submission = None
    if args.write_submission_if_qualified and qualified:
        output = sample.copy()
        for group in CAPACITY_KWH:
            output[group] = final_test[group]
        output_path = Path(args.output_submission)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        output_submission = {
            "path": output_path.as_posix(),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "rows": len(output),
        }

    report = {
        "family": "issue_block_nested_quantile_base_v2",
        "configuration": {
            "feature_set": args.feature_set,
            "alphas": alphas,
            "seeds": seeds,
            "train_variants": variants,
            "purge_hours": args.purge_hours,
            "outer_seasons": OUTER_SEASONS,
            "include_catboost": args.include_catboost,
            "ws_cal": args.ws_cal,
            "group3_curtailment": args.group3_curtailment,
        },
        "validation_contract": {
            "outer_fold_never_used_for_selection_or_early_stopping": True,
            "inner_holdout": "immediately preceding complete issue-season",
            "dependency_unit": "complete data_available_kst_dtm issue cycle",
            "selection": "alpha, all/eligible-only, and iteration count on inner season",
        },
        "ws_calibration": ws_report,
        "groups": group_reports,
        "outer_macro": overall,
        "seasonal_macro": seasonal,
        "issue_block_bootstrap": bootstrap,
        "promotion_gates": gates,
        "qualified": qualified,
        "submission_requested": args.write_submission_if_qualified,
        "submission": output_submission,
    }
    report_path = artifact_dir / "report.json"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    cache: dict[str, np.ndarray] = {
        "index_ns": oof_index.astype("int64").to_numpy(),
        "test_index_ns": X_test_all.index.astype("int64").to_numpy(),
    }
    for group in CAPACITY_KWH:
        cache[f"{group}__truth"] = oof_truth[group].astype("float32")
        cache[f"{group}__l1"] = oof_l1[group].astype("float32")
        cache[f"{group}__candidate"] = oof_candidate[group].astype("float32")
        cache[f"{group}__test"] = final_test[group].astype("float32")
    np.savez_compressed(artifact_dir / "predictions.npz", **cache)
    print(
        json.dumps(
            {
                "outer_macro": overall,
                "issue_block_bootstrap": bootstrap,
                "promotion_gates": gates,
                "qualified": qualified,
                "submission": output_submission,
                "report": report_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
