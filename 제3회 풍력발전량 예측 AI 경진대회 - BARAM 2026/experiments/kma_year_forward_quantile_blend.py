"""One-year-forward KMA quantile experts for groups 1 and 3.

The experiment uses only 2023 labels/KMA forecasts to construct 2024
validation experts.  Quantile level is selected on 2023 Q4, blend weight is
selected on 2024 Q1, and Q2/H2 are confirmation periods.  If the frozen
architecture passes seed, component, month, and issue-block bootstrap gates,
the production expert is refit on 2024 and predicts 2025.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from agent_service.compliance import validate_external_data_manifest
from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.blocked_rolling_validation import evaluate_blocked_rolling
from experiments.kma_base_v2_local_overlay import OverlayPolicy, apply_overlay
from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
TARGETS = ("kpx_group_1", "kpx_group_3")
COMPONENTS = ("score", "one_minus_nmae", "ficr")
ALPHAS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
WEIGHTS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
SEEDS = (42, 202, 2026)
MAXIMUM_MOVEMENT_RATIO = 0.05
PARSIMONY_SCORE_TOLERANCE = 0.00025
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
VALIDATION_END = pd.Timestamp("2025-01-01 00:00:00")
BASE_CALENDAR_COLUMNS = {
    "hour",
    "month",
    "dayofweek",
    "lead_hour",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
}
BASE_PHYSICAL_TOKENS = (
    "ws",
    "10u",
    "10v",
    "MU",
    "MV",
    "gust",
    "_u__",
    "_v__",
    "surface_0_sp",
    "prmsl",
    "blh",
)
LDAPS_WIND_TOKENS = (
    "10u",
    "10v",
    "MU",
    "MV",
    "ws",
    "hub_u",
    "hub_v",
    "dir_sin",
    "dir_cos",
)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_targets(value: str) -> tuple[str, ...]:
    """Parse a stable, duplicate-free subset of supported target groups."""
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    if not requested:
        raise ValueError("at least one target group is required")
    if len(set(requested)) != len(requested):
        raise ValueError("target groups must not be duplicated")
    unsupported = tuple(target for target in requested if target not in SUPPORTED_TARGETS)
    if unsupported:
        raise ValueError(f"unsupported target groups: {unsupported}")
    return requested


def load_context_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load forecast-time KMA features and retain their safe issue timestamp."""
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        parse_dates=["forecast_kst_dtm", "data_available_kst_dtm"],
    )
    if frame["forecast_kst_dtm"].duplicated().any():
        raise ValueError("KMA context contains duplicate forecast timestamps")
    frame = frame.set_index("forecast_kst_dtm").sort_index()
    columns = [
        column for column in frame if column.startswith("kma_um_ctx_")
    ]
    if not columns:
        raise ValueError("KMA context contains no model features")
    features = frame[columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("KMA context model features are incomplete")
    index = pd.DatetimeIndex(features.index)
    day = index.dayofyear.to_numpy(dtype=float)
    hour = index.hour.to_numpy(dtype=float)
    features = features.copy()
    features["doy_sin"] = np.sin(2.0 * np.pi * day / 365.25)
    features["doy_cos"] = np.cos(2.0 * np.pi * day / 365.25)
    features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    issue = frame["data_available_kst_dtm"].copy()
    return features.astype("float32"), issue


def load_context_feature_bundle(
    primary_path: Path,
    extra_paths: tuple[Path, ...] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    """Join causal context sources on forecast time without calendar duplication."""
    features, issue = load_context_features(primary_path)
    for position, extra_path in enumerate(extra_paths, start=1):
        extra, _ = load_context_features(extra_path)
        if not extra.index.equals(features.index):
            raise ValueError(
                f"context feature indexes differ: {primary_path} vs {extra_path}"
            )
        model_columns = [
            column for column in extra if column.startswith("kma_um_ctx_")
        ]
        renamed = extra[model_columns].rename(
            columns={
                column: f"extra{position}__{column}"
                for column in model_columns
            }
        )
        features = features.join(renamed, how="left", validate="one_to_one")
    if features.isna().any().any():
        raise ValueError("combined context model features are incomplete")
    return features.astype("float32"), issue


def select_base_physical_columns(
    features: pd.DataFrame,
    target: str,
) -> list[str]:
    """Select compact physical aggregates and target-specific IDW features."""
    columns = [
        column
        for column in features.columns
        if column in BASE_CALENDAR_COLUMNS
        or f"__{target}__" in column
        or (
            any(
                suffix in column
                for suffix in ("__mean", "__std", "__min", "__max")
            )
            and any(token in column for token in BASE_PHYSICAL_TOKENS)
        )
    ]
    if not columns:
        raise ValueError(f"no rich physical features selected for {target}")
    return columns


def select_ldaps_wind_columns(
    features: pd.DataFrame,
    target: str,
) -> list[str]:
    """Select a compact provided-LDAPS wind stencil without broad weather fields."""
    columns = [
        column
        for column in features.columns
        if column in BASE_CALENDAR_COLUMNS
        or (
            column.startswith("ldaps__")
            and f"__{target}__" in column
            and any(token in column for token in LDAPS_WIND_TOKENS)
        )
        or (
            column.startswith("ldaps__")
            and any(
                column.endswith(suffix)
                for suffix in ("__mean", "__std", "__min", "__max")
            )
            and any(token in column for token in LDAPS_WIND_TOKENS)
        )
    ]
    if not columns:
        raise ValueError(f"no LDAPS wind features selected for {target}")
    return columns


def add_base_physical_features(
    context: pd.DataFrame,
    base_features: pd.DataFrame,
    target: str,
    *,
    mode: str = "rich",
) -> pd.DataFrame:
    """Join preregistered GFS/LDAPS physical features to KMA context."""
    if mode == "rich":
        columns = select_base_physical_columns(base_features, target)
    elif mode == "ldaps_wind":
        columns = select_ldaps_wind_columns(base_features, target)
    else:
        raise ValueError(f"unsupported base physical feature mode: {mode}")
    extra = base_features[columns].reindex(context.index).add_prefix("base__")
    if extra.isna().any().any():
        raise ValueError("rich physical features are incomplete on context index")
    combined = pd.concat([context, extra], axis=1)
    if not combined.columns.is_unique:
        raise ValueError("rich physical feature names collide with context")
    return combined.astype("float32")


def make_quantile_model(alpha: float, seed: int, n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        n_estimators=int(n_estimators),
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.75,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def fit_predict(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    query_features: pd.DataFrame,
    *,
    alpha: float,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    observed = train_target.notna()
    if int(observed.sum()) < 5_000:
        raise ValueError("year-forward expert has insufficient training labels")
    model = make_quantile_model(alpha, seed, n_estimators)
    model.fit(
        train_features.loc[observed],
        train_target.loc[observed],
        callbacks=[lgb.log_evaluation(0)],
    )
    return np.asarray(model.predict(query_features), dtype=float)


def metric_delta(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    capacity: float,
    rows: np.ndarray,
) -> dict[str, float]:
    before = evaluate_group(truth[rows], reference[rows], capacity)
    after = evaluate_group(truth[rows], candidate[rows], capacity)
    return {
        "score": float(after.score - before.score),
        "one_minus_nmae": float(
            after.one_minus_nmae - before.one_minus_nmae
        ),
        "ficr": float(after.ficr - before.ficr),
    }


def apply_bounded_blend(
    reference: np.ndarray,
    expert: np.ndarray,
    *,
    weight: float,
    capacity: float,
    maximum_movement_ratio: float = MAXIMUM_MOVEMENT_RATIO,
) -> np.ndarray:
    """Blend toward an expert while bounding every row's absolute movement."""
    reference = np.asarray(reference, dtype=float)
    expert = np.asarray(expert, dtype=float)
    movement = np.clip(
        float(weight) * (expert - reference),
        -float(maximum_movement_ratio) * capacity,
        float(maximum_movement_ratio) * capacity,
    )
    return np.clip(reference + movement, 0.0, capacity)


def evaluate_weights(
    truth: np.ndarray,
    reference: np.ndarray,
    expert: np.ndarray,
    capacity: float,
    development: np.ndarray,
    weights: tuple[float, ...] = WEIGHTS,
) -> list[dict[str, Any]]:
    """Evaluate blend weights on the development period only."""
    records: list[dict[str, Any]] = []
    for weight in weights:
        candidate = apply_bounded_blend(
            reference, expert, weight=weight, capacity=capacity
        )
        delta = metric_delta(
            truth, reference, candidate, capacity, development
        )
        records.append(
            {
                "weight": float(weight),
                "delta": delta,
                "eligible": bool(min(delta.values()) > 0.0),
            }
        )
    return records


def select_weight(
    truth: np.ndarray,
    reference: np.ndarray,
    expert: np.ndarray,
    capacity: float,
    development: np.ndarray,
    weights: tuple[float, ...] = WEIGHTS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select weight on the development period only, never Q2/H2."""
    records = evaluate_weights(
        truth,
        reference,
        expert,
        capacity,
        development,
        weights,
    )
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        raise RuntimeError("no year-forward blend weight improved every Q1 component")
    best_score = max(record["delta"]["score"] for record in eligible)
    near_best = [
        record
        for record in eligible
        if record["delta"]["score"]
        >= best_score - PARSIMONY_SCORE_TOLERANCE
    ]
    # Prefer the smallest statistically indistinguishable movement footprint.
    # This is fixed before Q2/H2 confirmation and prevents a negligible Q1
    # score difference from selecting a materially broader blend.
    selected = min(
        near_best,
        key=lambda record: (
            record["weight"],
            -min(record["delta"].values()),
        ),
    )
    return selected, records


def select_alpha(
    features: pd.DataFrame,
    target: pd.Series,
    capacity: float,
    *,
    n_estimators: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Select the direct expert's quantile using 2023 Q4 only."""
    train = (features.index < pd.Timestamp("2023-10-01")) & target.notna()
    validation = (features.index >= pd.Timestamp("2023-10-01")) & target.notna()
    rows: list[dict[str, Any]] = []
    for position, alpha in enumerate(ALPHAS):
        model = make_quantile_model(alpha, 26_000 + position, n_estimators)
        model.fit(
            features.loc[train],
            target.loc[train],
            callbacks=[lgb.log_evaluation(0)],
        )
        prediction = np.clip(
            model.predict(features.loc[validation]), 0.0, capacity
        )
        metric = evaluate_group(
            target.loc[validation].to_numpy(dtype=float),
            prediction,
            capacity,
        )
        rows.append({"alpha": float(alpha), **metric.to_dict()})
    selected = max(rows, key=lambda record: record["score"])
    return float(selected["alpha"]), rows


def select_alpha_year_forward(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    query_features: pd.DataFrame,
    query_target: pd.Series,
    capacity: float,
    *,
    n_estimators: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Select the quantile on a complete prior-year forward prediction."""
    if list(train_features.columns) != list(query_features.columns):
        raise ValueError("year-forward alpha-selection feature columns differ")
    observed_train = train_target.notna()
    observed_query = query_target.notna()
    if int(observed_train.sum()) < 5_000 or int(observed_query.sum()) < 5_000:
        raise ValueError("year-forward alpha selection has insufficient labels")
    rows: list[dict[str, Any]] = []
    for position, alpha in enumerate(ALPHAS):
        model = make_quantile_model(alpha, 22_000 + position, n_estimators)
        model.fit(
            train_features.loc[observed_train],
            train_target.loc[observed_train],
            callbacks=[lgb.log_evaluation(0)],
        )
        prediction = np.clip(
            model.predict(query_features.loc[observed_query]),
            0.0,
            capacity,
        )
        metric = evaluate_group(
            query_target.loc[observed_query].to_numpy(dtype=float),
            prediction,
            capacity,
        )
        rows.append({"alpha": float(alpha), **metric.to_dict()})
    selected = max(rows, key=lambda record: record["score"])
    return float(selected["alpha"]), rows


def _periods(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    in_year = index < VALIDATION_END
    return {
        "q1": np.asarray(index < Q2_START),
        "q2": np.asarray((index >= Q2_START) & (index < H2_START)),
        "h2": np.asarray((index >= H2_START) & in_year),
        "full": np.asarray(in_year),
    }


def _align_prediction(
    prediction: np.ndarray,
    prediction_index: pd.DatetimeIndex,
    target_index: pd.DatetimeIndex,
    fallback: np.ndarray,
) -> np.ndarray:
    series = pd.Series(np.asarray(prediction, dtype=float), index=prediction_index)
    aligned = series.reindex(target_index)
    missing = aligned.isna()
    if int(missing.sum()) > 1:
        raise ValueError("year-forward expert has more than one alignment gap")
    return aligned.fillna(pd.Series(fallback, index=target_index)).to_numpy(dtype=float)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validation_surfaces(
    driver: np.lib.npyio.NpzFile,
    kma_oof: np.lib.npyio.NpzFile,
    target: str,
    *,
    group2_oof: np.lib.npyio.NpzFile | None = None,
    group2_policy: OverlayPolicy | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{target}__valid_index_ns"])
    )
    if target == "kpx_group_1":
        return (
            driver_index,
            driver[f"{target}__valid_truth"].astype(float),
            driver[f"{target}__exact_base"].astype(float),
        )
    if target == "kpx_group_2":
        if group2_oof is None or group2_policy is None:
            raise ValueError("group-2 validation requires its complete OOF and policy")
        group2_index = pd.DatetimeIndex(
            pd.to_datetime(group2_oof["index_ns"])
        )
        if not group2_index.equals(driver_index):
            raise ValueError("group-2 and exact driver OOF indexes differ")
        reference = driver[f"{target}__exact_base"].astype(float)
        member = group2_oof[f"{target}__candidate"].astype(float)
        incumbent, _ = apply_overlay(reference, member, group2_policy)
        return (
            group2_index,
            group2_oof[f"{target}__truth"].astype(float),
            incumbent,
        )
    kma_index = pd.DatetimeIndex(pd.to_datetime(kma_oof["index_ns"]))
    if not kma_index.equals(driver_index):
        raise ValueError("KMA and exact driver OOF indexes differ")
    return (
        kma_index,
        kma_oof["truth"].astype(float),
        kma_oof["rolling_candidate"].astype(float),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    targets = parse_targets(getattr(args, "targets", ",".join(TARGETS)))
    validation_only = bool(getattr(args, "validation_only", False))
    diagnostic_redownloadable_extra = bool(
        getattr(args, "diagnostic_redownloadable_extra", False)
    )
    if diagnostic_redownloadable_extra and not validation_only:
        raise ValueError(
            "redownloadable, locally pruned extra data is diagnostic-only"
        )
    extra_contexts = {
        year: tuple(
            _rooted(value)
            for value in getattr(args, f"extra_context_{year}", ())
        )
        for year in ("2023", "2024", "2025")
    }
    extra_manifests = {
        year: tuple(
            _rooted(value)
            for value in getattr(args, f"extra_manifest_{year}", ())
        )
        for year in ("2023", "2024", "2025")
    }
    for year in ("2023", "2024", "2025"):
        if len(extra_contexts[year]) != len(extra_manifests[year]):
            raise ValueError(
                f"{year} extra context and manifest counts must match"
            )
    production_extra_contexts = tuple(
        _rooted(value)
        for value in getattr(args, "production_extra_context", ())
    )
    production_extra_manifests = tuple(
        _rooted(value)
        for value in getattr(args, "production_extra_manifest", ())
    )
    if len(production_extra_contexts) != len(production_extra_manifests):
        raise ValueError(
            "production extra context and manifest counts must match"
        )
    paths = {
        "context_2023": _rooted(args.context_2023),
        "context_2024": _rooted(args.context_2024),
        "manifest_2023": _rooted(args.manifest_2023),
        "manifest_2024": _rooted(args.manifest_2024),
    }
    if not validation_only:
        paths.update(
            {
                "context_2025": _rooted(args.context_2025),
                "manifest_2025": _rooted(args.manifest_2025),
            }
        )
    years = ("2023", "2024") if validation_only else ("2023", "2024", "2025")
    for year in years:
        for position, path in enumerate(extra_contexts[year], start=1):
            paths[f"extra_context_{year}_{position}"] = path
        for position, path in enumerate(extra_manifests[year], start=1):
            paths[f"extra_manifest_{year}_{position}"] = path
    manifest_validation = {
        year: validate_external_data_manifest(paths[f"manifest_{year}"], ROOT)
        for year in years
    }
    for year in years:
        for position, path in enumerate(extra_manifests[year], start=1):
            manifest_validation[f"{year}_extra_{position}"] = (
                validate_external_data_manifest(
                    path,
                    ROOT,
                    verify_files=not diagnostic_redownloadable_extra,
                    require_competition_eligible=(
                        not diagnostic_redownloadable_extra
                    ),
                )
            )
    for position, path in enumerate(production_extra_manifests, start=1):
        manifest_validation[f"production_extra_{position}"] = (
            validate_external_data_manifest(path, ROOT)
        )
        paths[f"production_extra_context_{position}"] = (
            production_extra_contexts[position - 1]
        )
        paths[f"production_extra_manifest_{position}"] = path
    features_2023, _ = load_context_feature_bundle(
        paths["context_2023"],
        extra_contexts["2023"],
    )
    features_2024, issue_2024 = load_context_feature_bundle(
        paths["context_2024"],
        extra_contexts["2024"],
    )
    features_2025 = None
    production_features_2024 = None
    if not validation_only:
        features_2025, _ = load_context_feature_bundle(
            paths["context_2025"],
            extra_contexts["2025"],
        )
        production_features_2024, _ = load_context_feature_bundle(
            paths["context_2024"],
            production_extra_contexts or extra_contexts["2024"],
        )
    rich_train_features = None
    rich_test_features = None
    if getattr(args, "rich_train_features", None):
        rich_train_features = pd.read_pickle(
            _rooted(args.rich_train_features)
        )
        if not validation_only:
            if not getattr(args, "rich_test_features", None):
                raise ValueError(
                    "production requires --rich-test-features with rich training features"
                )
            rich_test_features = pd.read_pickle(
                _rooted(args.rich_test_features)
            )
    alpha_selection_train = None
    alpha_selection_query = None
    if bool(getattr(args, "alpha_selection_train_context", None)) != bool(
        getattr(args, "alpha_selection_query_context", None)
    ):
        raise ValueError(
            "year-forward alpha selection requires both train and query context"
        )
    if getattr(args, "alpha_selection_train_context", None):
        required_alpha_inputs = (
            getattr(args, "alpha_selection_query_context", None),
            getattr(args, "alpha_selection_train_manifest", None),
            getattr(args, "alpha_selection_query_manifest", None),
        )
        if not all(required_alpha_inputs):
            raise ValueError(
                "year-forward alpha selection requires both contexts and manifests"
            )
        alpha_train_manifest = _rooted(args.alpha_selection_train_manifest)
        alpha_query_manifest = _rooted(args.alpha_selection_query_manifest)
        manifest_validation["alpha_selection_train"] = (
            validate_external_data_manifest(alpha_train_manifest, ROOT)
        )
        manifest_validation["alpha_selection_query"] = (
            validate_external_data_manifest(alpha_query_manifest, ROOT)
        )
        alpha_selection_train, _ = load_context_features(
            _rooted(args.alpha_selection_train_context)
        )
        alpha_selection_query, _ = load_context_features(
            _rooted(args.alpha_selection_query_context)
        )
        paths["alpha_selection_train_context"] = _rooted(
            args.alpha_selection_train_context
        )
        paths["alpha_selection_query_context"] = _rooted(
            args.alpha_selection_query_context
        )
        paths["alpha_selection_train_manifest"] = alpha_train_manifest
        paths["alpha_selection_query_manifest"] = alpha_query_manifest
    prepend_train_features = None
    if getattr(args, "prepend_train_context", None):
        if not getattr(args, "prepend_train_manifest", None):
            raise ValueError("prepended training context requires its manifest")
        prepend_manifest = _rooted(args.prepend_train_manifest)
        manifest_validation["prepend_train"] = validate_external_data_manifest(
            prepend_manifest,
            ROOT,
        )
        prepend_train_features, _ = load_context_features(
            _rooted(args.prepend_train_context)
        )
        paths["prepend_train_context"] = _rooted(args.prepend_train_context)
        paths["prepend_train_manifest"] = prepend_manifest
    labels = pd.read_csv(
        _rooted(args.labels),
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    driver = np.load(_rooted(args.driver), allow_pickle=False)
    kma_oof = np.load(_rooted(args.kma_oof), allow_pickle=False)
    group2_oof = None
    group2_policy = None
    if "kpx_group_2" in targets:
        group2_oof_path = _rooted(args.group2_oof)
        group2_report_path = _rooted(args.group2_overlay_report)
        group2_oof = np.load(group2_oof_path, allow_pickle=False)
        report = json.loads(group2_report_path.read_text(encoding="utf-8"))
        raw_policy = report["search"]["exploratory_by_group"]["kpx_group_2"][
            "policy"
        ]
        group2_policy = OverlayPolicy(
            **{**raw_policy, "alpha": float(args.group2_alpha)}
        )
        paths["group2_oof"] = group2_oof_path
        paths["group2_overlay_report"] = group2_report_path

    validation: dict[str, Any] = {}
    selected_specs: dict[str, dict[str, float]] = {}
    validation_candidates: dict[str, np.ndarray] = {}
    validation_references: dict[str, np.ndarray] = {}
    validation_cache_arrays: dict[str, np.ndarray] = {}
    full_score_deltas: list[float] = []
    stable = True

    for target in targets:
        capacity = CAPACITY_KWH[target]
        train_features = features_2023
        query_features = features_2024
        if prepend_train_features is not None:
            if list(prepend_train_features.columns) != list(train_features.columns):
                raise ValueError("prepended and primary training feature columns differ")
            train_features = pd.concat(
                [prepend_train_features, train_features],
                axis=0,
            ).sort_index()
            if train_features.index.duplicated().any():
                raise ValueError("prepended training context overlaps primary year")
        if rich_train_features is not None:
            train_features = add_base_physical_features(
                train_features,
                rich_train_features,
                target,
                mode=args.base_feature_mode,
            )
            query_features = add_base_physical_features(
                query_features,
                rich_train_features,
                target,
                mode=args.base_feature_mode,
            )
        target_2023 = labels[target].reindex(train_features.index)
        if alpha_selection_train is None or alpha_selection_query is None:
            alpha, alpha_rows = select_alpha(
                train_features,
                target_2023,
                capacity,
                n_estimators=args.n_estimators,
            )
            alpha_selection_contract = "2023 Jan-Sep -> 2023 Q4"
        else:
            alpha, alpha_rows = select_alpha_year_forward(
                alpha_selection_train,
                labels[target].reindex(alpha_selection_train.index),
                alpha_selection_query,
                labels[target].reindex(alpha_selection_query.index),
                capacity,
                n_estimators=args.n_estimators,
            )
            alpha_selection_contract = "complete 2022 -> complete 2023"
        seed_experts = [
            np.clip(
                fit_predict(
                    train_features,
                    target_2023,
                    query_features,
                    alpha=alpha,
                    seed=seed,
                    n_estimators=args.n_estimators,
                ),
                0.0,
                capacity,
            )
            for seed in SEEDS
        ]
        index, truth, reference = _validation_surfaces(
            driver,
            kma_oof,
            target,
            group2_oof=group2_oof,
            group2_policy=group2_policy,
        )
        aligned_seed_experts = [
            _align_prediction(
                expert, query_features.index, index, reference
            )
            for expert in seed_experts
        ]
        ensemble_expert = np.mean(aligned_seed_experts, axis=0)
        periods = _periods(index)
        weight_rows = evaluate_weights(
            truth,
            reference,
            ensemble_expert,
            capacity,
            periods["q1"],
        )
        eligible_weights = [record for record in weight_rows if record["eligible"]]
        if eligible_weights:
            selected, _ = select_weight(
                truth,
                reference,
                ensemble_expert,
                capacity,
                periods["q1"],
            )
            q1_selection_eligible = True
        else:
            # Keep the least-bad Q1 score footprint for complete diagnostics,
            # while explicitly failing promotion before any confirmation split
            # can influence selection.
            selected = max(
                weight_rows,
                key=lambda record: (
                    record["delta"]["score"],
                    min(record["delta"].values()),
                    -record["weight"],
                ),
            )
            q1_selection_eligible = False
        weight = float(selected["weight"])
        candidate = apply_bounded_blend(
            reference, ensemble_expert, weight=weight, capacity=capacity
        )
        period_deltas = {
            name: metric_delta(
                truth, reference, candidate, capacity, rows
            )
            for name, rows in periods.items()
        }
        seed_rows: list[dict[str, Any]] = []
        for seed, expert in zip(SEEDS, aligned_seed_experts):
            seed_candidate = apply_bounded_blend(
                reference, expert, weight=weight, capacity=capacity
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "period_deltas": {
                        name: metric_delta(
                            truth, reference, seed_candidate, capacity, rows
                        )
                        for name, rows in periods.items()
                    },
                }
            )
        monthly = {
            str(month): metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                periods["full"] & np.asarray(index.month == month),
            )
            for month in range(1, 13)
        }
        issue = issue_2024.reindex(index)
        if int(issue.loc[index < VALIDATION_END].isna().sum()):
            raise ValueError("KMA issue times are missing inside the validation year")
        issue = issue.fillna(pd.Timestamp("2024-12-31 13:00")).to_numpy()
        bootstrap = evaluate_blocked_rolling(
            truth,
            reference,
            candidate,
            index,
            issue,
            periods["full"] & (truth >= 0.10 * capacity),
            n_bootstrap=args.n_bootstrap,
            seed=20260726,
        )
        seed_components_positive = all(
            min(record["period_deltas"][period].values()) > 0.0
            for record in seed_rows
            for period in ("q1", "q2", "h2")
        )
        positive_score_months = int(
            sum(row["score"] > 0.0 for row in monthly.values())
        )
        gates = {
            "q1_selection_all_components_positive": q1_selection_eligible,
            "q2_components_positive": min(period_deltas["q2"].values()) > 0.0,
            "h2_components_positive": min(period_deltas["h2"].values()) > 0.0,
            "full_components_positive": min(period_deltas["full"].values()) > 0.0,
            "all_seed_split_components_positive": seed_components_positive,
            "positive_score_month_fraction_at_least_80pct": (
                positive_score_months / 12.0 >= 0.80
            ),
            "issue_bootstrap_q05_positive": (
                bootstrap["issue_block_bootstrap"]["q05"] > 0.0
            ),
            "issue_bootstrap_positive_fraction_at_least_98pct": (
                bootstrap["issue_block_bootstrap"]["positive_fraction"] >= 0.98
            ),
        }
        stable &= all(gates.values())
        full_score_deltas.append(period_deltas["full"]["score"])
        selected_specs[target] = {"alpha": alpha, "weight": weight}
        validation_candidates[target] = candidate
        validation_references[target] = reference
        validation_cache_arrays.update(
            {
                f"{target}__index_ns": index.asi8,
                f"{target}__issue_ns": pd.DatetimeIndex(issue).asi8,
                f"{target}__truth": truth.astype("float32"),
                f"{target}__reference": reference.astype("float32"),
                f"{target}__expert": ensemble_expert.astype("float32"),
                f"{target}__candidate": candidate.astype("float32"),
            }
        )
        movement = np.abs(candidate[periods["full"]] - reference[periods["full"]])
        validation[target] = {
            "feature_count": int(train_features.shape[1]),
            "training_rows": int(train_features.shape[0]),
            "prepended_training_year": prepend_train_features is not None,
            "rich_base_physical_features": rich_train_features is not None,
            "base_physical_feature_mode": (
                args.base_feature_mode
                if rich_train_features is not None
                else None
            ),
            "alpha_selection_2023_q4": {
                "selected": alpha,
                "records": alpha_rows,
                "contract": alpha_selection_contract,
            },
            "weight_selection_2024_q1": {
                "selected": selected,
                "records": weight_rows,
                "eligible": q1_selection_eligible,
            },
            "period_deltas": period_deltas,
            "seed_stability": seed_rows,
            "monthly_deltas": monthly,
            "positive_score_months": positive_score_months,
            "issue_block_validation": bootstrap,
            "movement": {
                "mean_ratio": float(movement.mean() / capacity),
                "p95_ratio": float(np.quantile(movement / capacity, 0.95)),
                "maximum_ratio": float(movement.max() / capacity),
            },
            "gates": gates,
        }

    expected_macro_score_delta = float(np.sum(full_score_deltas) / 3.0)
    candidate_record: dict[str, Any] | None = None
    lineage: dict[str, float] | None = None
    if stable and not validation_only:
        if features_2025 is None:
            raise RuntimeError("production features were not loaded")
        if production_features_2024 is None:
            raise RuntimeError("production training features were not loaded")
        active = pd.read_csv(
            _rooted(args.active_candidate), encoding="utf-8-sig"
        )
        sample = pd.read_csv(
            _rooted(args.sample_submission), encoding="utf-8-sig"
        )
        if not active[["forecast_id", "forecast_kst_dtm"]].equals(
            sample[["forecast_id", "forecast_kst_dtm"]]
        ):
            raise ValueError("active candidate and sample submission IDs differ")
        test_index = pd.DatetimeIndex(
            pd.to_datetime(active["forecast_kst_dtm"])
        )
        if not test_index.equals(features_2025.index):
            raise ValueError("2025 KMA features and submission timestamps differ")
        kma_submission = pd.read_csv(
            _rooted(args.kma_submission), encoding="utf-8-sig"
        )
        lineage = {
            "group1_vs_exact_driver_max_abs_kwh": float(
                np.max(
                    np.abs(
                        active["kpx_group_1"].to_numpy(dtype=float)
                        - driver["kpx_group_1__test_exact_base"].astype(float)
                    )
                )
            ),
            "group3_vs_kma_submission_max_abs_kwh": float(
                np.max(
                    np.abs(
                        active["kpx_group_3"].to_numpy(dtype=float)
                        - kma_submission["kpx_group_3"].to_numpy(dtype=float)
                    )
                )
            ),
        }
        if max(lineage.values()) > 1e-3:
            raise ValueError("active candidate lineage differs from validation baselines")
        output = active.copy()
        for target in targets:
            capacity = CAPACITY_KWH[target]
            production_train_features = production_features_2024
            production_query_features = features_2025
            if rich_train_features is not None:
                if rich_test_features is None:
                    raise RuntimeError("rich production test features were not loaded")
                production_train_features = add_base_physical_features(
                    production_train_features,
                    rich_train_features,
                    target,
                    mode=args.base_feature_mode,
                )
                production_query_features = add_base_physical_features(
                    production_query_features,
                    rich_test_features,
                    target,
                    mode=args.base_feature_mode,
                )
            target_2024 = labels[target].reindex(
                production_train_features.index
            )
            spec = selected_specs[target]
            experts = [
                np.clip(
                    fit_predict(
                        production_train_features,
                        target_2024,
                        production_query_features,
                        alpha=spec["alpha"],
                        seed=seed,
                        n_estimators=args.n_estimators,
                    ),
                    0.0,
                    capacity,
                )
                for seed in SEEDS
            ]
            expert = np.mean(experts, axis=0)
            output[target] = apply_bounded_blend(
                active[target].to_numpy(dtype=float),
                expert,
                weight=spec["weight"],
                capacity=capacity,
            )
        output_path = _rooted(args.output_submission)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        audit = CandidateValidator(load_config(ROOT)).audit(output_path)
        if not audit.valid:
            raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")
        movement_by_group: dict[str, Any] = {}
        normalized: list[np.ndarray] = []
        for target, capacity in CAPACITY_KWH.items():
            movement = np.abs(
                output[target].to_numpy(dtype=float)
                - active[target].to_numpy(dtype=float)
            )
            normalized.append(movement / capacity)
            movement_by_group[target] = {
                "changed_rows": int((movement > 1e-6).sum()),
                "mean_kwh": float(movement.mean()),
                "p95_kwh": float(np.quantile(movement, 0.95)),
                "maximum_kwh": float(movement.max()),
            }
        normalized_vector = np.concatenate(normalized)
        candidate_record = {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "candidate_validator": audit.to_dict(),
            "movement": {
                "changed_target_cell_ratio": float(
                    np.mean(normalized_vector > (1e-6 / 21_600.0))
                ),
                "p95_ratio": float(np.quantile(normalized_vector, 0.95)),
                "maximum_ratio": float(normalized_vector.max()),
                "by_group": movement_by_group,
            },
        }

    cache_record: dict[str, Any] | None = None
    if getattr(args, "output_cache", None):
        cache_path = _rooted(args.output_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **validation_cache_arrays)
        cache_record = {
            "path": cache_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(cache_path),
            "arrays": sorted(validation_cache_arrays),
        }

    report = {
        "family": "kma_one_year_forward_quantile_blend",
        "method": (
            "KMA multilevel/run-revision quantile expert; train previous calendar "
            "year, select weight on 2024 Q1, confirm on Q2/H2"
        ),
        "sources": {
            key: path.relative_to(ROOT).as_posix()
            for key, path in paths.items()
        },
        "manifest_validation": manifest_validation,
        "contract": {
            "targets": list(targets),
            "validation_only": validation_only,
            "diagnostic_redownloadable_extra": (
                diagnostic_redownloadable_extra
            ),
            "alpha_selection": (
                "complete 2022 -> complete 2023"
                if alpha_selection_train is not None
                else "2023 Jan-Sep train -> 2023 Q4"
            ),
            "validation_expert": "all 2023 -> 2024",
            "weight_selection": "2024 Q1 only",
            "confirmation": "2024 Q2 and historical H2",
            "production_expert": "all 2024 -> 2025",
            "production_context_override": bool(production_extra_contexts),
            "validation_training_window": (
                "prepended prior year + 2023"
                if prepend_train_features is not None
                else "2023 only"
            ),
            "maximum_row_movement_ratio": MAXIMUM_MOVEMENT_RATIO,
            "weight_parsimony_rule": (
                "smallest all-component-positive Q1 weight within "
                f"{PARSIMONY_SCORE_TOLERANCE} score of the Q1 best"
            ),
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
        },
        "selected_specs": selected_specs,
        "validation": validation,
        "expected_macro_score_delta_if_2024_transfers": expected_macro_score_delta,
        "projected_public_score_if_local_delta_transfers": float(
            args.active_public_score + expected_macro_score_delta
        ),
        "promotion": {
            "stable": bool(stable),
            "strict_monthly_all_positive": False,
            "tier": (
                "validation_pass"
                if stable and validation_only
                else ("exploratory" if stable else "rejected")
            ),
            "reason": (
                "all preregistered split/seed/bootstrap gates passed; production "
                "generation was intentionally deferred"
                if stable and validation_only
                else (
                    "all preregistered split/seed/bootstrap gates passed; isolated "
                    "negative months prevent a strict promotion"
                    if stable
                    else "at least one split, seed, month-fraction, or bootstrap gate failed"
                )
            ),
        },
        "active_lineage": lineage,
        "candidate": candidate_record,
        "validation_cache": cache_record,
    }
    output_report = _rooted(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--rich-train-features",
        help="optional pickle of preregistered GFS/LDAPS physical features",
    )
    parser.add_argument(
        "--base-feature-mode",
        choices=("rich", "ldaps_wind"),
        default="rich",
        help="preregistered subset selected from --rich-train-features",
    )
    parser.add_argument(
        "--rich-test-features",
        help="matching production feature pickle when not validation-only",
    )
    parser.add_argument(
        "--alpha-selection-train-context",
        help="optional complete prior-year context used to select quantile alpha",
    )
    parser.add_argument(
        "--alpha-selection-query-context",
        help="complete next-year context paired with alpha-selection training",
    )
    parser.add_argument(
        "--alpha-selection-train-manifest",
        help="manifest for --alpha-selection-train-context",
    )
    parser.add_argument(
        "--alpha-selection-query-manifest",
        help="manifest for --alpha-selection-query-context",
    )
    parser.add_argument(
        "--prepend-train-context",
        help="optional earlier context with identical columns for expanding-window validation",
    )
    parser.add_argument(
        "--prepend-train-manifest",
        help="manifest for --prepend-train-context",
    )
    parser.add_argument(
        "--targets",
        default=",".join(TARGETS),
        help="comma-separated subset of kpx_group_1,kpx_group_2,kpx_group_3",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="validate 2023 -> 2024 without loading 2025 context or writing a candidate",
    )
    parser.add_argument(
        "--diagnostic-redownloadable-extra",
        action="store_true",
        help=(
            "validation-only audit of extra data whose immutable source URLs and "
            "hashes are retained but whose local raw files were pruned"
        ),
    )
    parser.add_argument(
        "--driver", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--kma-oof",
        default=(
            "artifacts_final/external_weather/kma_um_regional_context_2024/"
            "power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--group2-oof",
        default="artifacts_final/lineage/base_v2_group2_complete_oof.npz",
    )
    parser.add_argument(
        "--group2-overlay-report",
        default="artifacts_final/base_v2/kma_incumbent_local_overlay_20260725.json",
    )
    parser.add_argument("--group2-alpha", type=float, default=0.2375)
    for year in ("2023", "2024", "2025"):
        parser.add_argument(
            f"--context-{year}",
            default=(
                f"artifacts_final/external_weather/"
                f"kma_um_regional_context_{year}/features.csv"
            ),
        )
        parser.add_argument(
            f"--manifest-{year}",
            default=(
                f"artifacts_final/external_weather/"
                f"kma_um_regional_context_{year}/manifest.json"
            ),
        )
        parser.add_argument(
            f"--extra-context-{year}",
            action="append",
            default=[],
            help="optional additional causal context CSV; may be repeated",
        )
        parser.add_argument(
            f"--extra-manifest-{year}",
            action="append",
            default=[],
            help="manifest paired with each additional context CSV",
        )
    parser.add_argument(
        "--production-extra-context",
        action="append",
        default=[],
        help="optional 2024 production-training context override; may be repeated",
    )
    parser.add_argument(
        "--production-extra-manifest",
        action="append",
        default=[],
        help="manifest paired with each production context override",
    )
    parser.add_argument(
        "--active-candidate",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
    )
    parser.add_argument(
        "--kma-submission",
        default="submissions/blend_best_kma_um_power_curve_gate.csv",
    )
    parser.add_argument(
        "--sample-submission", default="data/sample_submission.csv"
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--active-public-score", type=float, default=0.6440998116)
    parser.add_argument(
        "--output-submission",
        default=(
            "artifacts_final/candidates/"
            "kma_year_forward_g1w10_g3w075_20260726.csv"
        ),
    )
    parser.add_argument(
        "--output-report",
        default=(
            "artifacts_final/diagnostics/"
            "kma_year_forward_quantile_blend_20260726.json"
        ),
    )
    parser.add_argument(
        "--output-cache",
        help="optional NPZ containing aligned validation surfaces",
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "selected_specs": report["selected_specs"],
                "expected_macro_score_delta": report[
                    "expected_macro_score_delta_if_2024_transfers"
                ],
                "projected_public_score": report[
                    "projected_public_score_if_local_delta_transfers"
                ],
                "promotion": report["promotion"],
                "candidate": report["candidate"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
