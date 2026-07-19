from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from experiments.cross_group_trajectory_smoothing import (
    shift_within_issue_cycle,
    triangular_smooth,
)
from experiments.exact_oof_meta_gate import _fit_transfer_member, _current_prediction
from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
H2_START = pd.Timestamp("2024-07-01 01:00:00")
SEEDS = (26_071, 26_072, 26_073)


@dataclass(frozen=True)
class InjectionPolicy:
    component_weights: tuple[float, float, float]
    alpha: float
    max_disagreement: float
    require_driver_agreement: bool


def lead_phase(index: pd.DatetimeIndex) -> np.ndarray:
    """Return four six-hour phases for lead hours 12 through 35."""
    lead = ((pd.DatetimeIndex(index).hour.to_numpy() - 1) % 24) + 12
    return ((lead - 12) // 6).astype(np.int8)


def _curated_weather_columns(frame: pd.DataFrame) -> list[str]:
    own_tokens = (
        "hub_ws117",
        "hub_u117",
        "hub_v117",
        "hub_dir_sin",
        "hub_dir_cos",
        "ws10",
        "ws50",
        "ws80",
        "ws100",
        "surface_0_gust",
    )
    summary_tokens = (
        "hub_ws117",
        "hub_u117",
        "hub_v117",
        "ws10",
        "ws50",
        "ws80",
        "ws100",
        "surface_0_gust",
    )
    columns: list[str] = []
    for column in frame.columns:
        if "__kpx_group_" in column and column.endswith("__idw"):
            if any(token in column for token in own_tokens):
                columns.append(column)
        elif column.endswith(("__mean", "__std", "__min", "__max")):
            if any(token in column for token in summary_tokens):
                columns.append(column)
        elif "__hub_ws117__grid_" in column:
            columns.append(column)
    return sorted(set(columns))


def _dominant_signal_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "ldaps__kpx_group_3__hub_ws117__idw",
        "ldaps__kpx_group_3__hub_u117__idw",
        "ldaps__kpx_group_3__hub_v117__idw",
        "ldaps__hub_ws117__std",
        "gfs__kpx_group_3__hub_ws117__idw",
        "gfs__kpx_group_3__hub_u117__idw",
        "gfs__kpx_group_3__hub_v117__idw",
        "gfs__kpx_group_3__surface_0_gust__idw",
    ]
    return [column for column in preferred if column in frame.columns]


def build_cross_section_features(
    weather: pd.DataFrame,
    group_1: np.ndarray,
    group_2: np.ndarray,
) -> pd.DataFrame:
    """Build compact cross-farm, time-phase, and bidirectional NWP features."""
    index = pd.DatetimeIndex(weather.index)
    group_1 = np.asarray(group_1, dtype=float) / CAPACITY_KWH["kpx_group_1"]
    group_2 = np.asarray(group_2, dtype=float) / CAPACITY_KWH["kpx_group_2"]
    if len(weather) != len(group_1) or len(weather) != len(group_2):
        raise ValueError("Weather and driver predictions must be aligned")

    mean_driver = (group_1 + group_2) / 2.0
    data: dict[str, np.ndarray] = {
        "driver_1": group_1,
        "driver_2": group_2,
        "driver_mean": mean_driver,
        "driver_diff": group_2 - group_1,
        "driver_absdiff": np.abs(group_2 - group_1),
        "driver_min": np.minimum(group_1, group_2),
        "driver_max": np.maximum(group_1, group_2),
        "driver_product": group_1 * group_2,
        "driver_mean_sq": mean_driver**2,
        "driver_mean_cu": mean_driver**3,
        "driver_1_smooth": triangular_smooth(group_1, index),
        "driver_2_smooth": triangular_smooth(group_2, index),
        "driver_mean_smooth": triangular_smooth(mean_driver, index),
        "driver_mean_slope": (
            shift_within_issue_cycle(mean_driver, index, 1)
            - shift_within_issue_cycle(mean_driver, index, -1)
        )
        / 2.0,
        "driver_mean_curvature": (
            shift_within_issue_cycle(mean_driver, index, -1)
            - 2.0 * mean_driver
            + shift_within_issue_cycle(mean_driver, index, 1)
        ),
    }

    for column in ("hour", "month", "dayofweek", "lead_hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos"):
        if column in weather:
            data[column] = weather[column].to_numpy(dtype=float)

    result = pd.DataFrame(data, index=index)
    weather_columns = _curated_weather_columns(weather)
    result = result.join(weather[weather_columns].astype("float32"))

    for column in _dominant_signal_columns(weather):
        values = weather[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            finite = values[np.isfinite(values)]
            fill_value = float(np.median(finite)) if len(finite) else 0.0
            values = np.where(np.isfinite(values), values, fill_value)
        previous = shift_within_issue_cycle(values, index, -1)
        following = shift_within_issue_cycle(values, index, 1)
        previous_2 = shift_within_issue_cycle(values, index, -2)
        following_2 = shift_within_issue_cycle(values, index, 2)
        result[f"{column}__smooth3"] = (previous + 2.0 * values + following) / 4.0
        result[f"{column}__smooth5"] = (
            previous_2 + 2.0 * previous + 3.0 * values + 2.0 * following + following_2
        ) / 9.0
        result[f"{column}__slope"] = (following - previous) / 2.0
        result[f"{column}__curvature"] = previous - 2.0 * values + following

    return result.replace([np.inf, -np.inf], np.nan).astype("float32")


def regime_inputs(features: pd.DataFrame) -> pd.DataFrame:
    tokens = (
        "hub_ws117__idw",
        "hub_u117__idw",
        "hub_v117__idw",
        "hub_ws117__std",
        "surface_0_gust__idw",
        "driver_mean",
        "driver_diff",
    )
    columns = [column for column in features if any(token in column for token in tokens)]
    if not columns:
        raise ValueError("No regime input columns were found")
    return features[columns]


def fit_regimes(
    train_features: pd.DataFrame,
    query_features: pd.DataFrame,
    n_clusters: int = 4,
    seed: int = 20_226,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    train = regime_inputs(train_features)
    query = regime_inputs(query_features).reindex(columns=train.columns)
    medians = train.median().fillna(0.0)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.fillna(medians))
    query_scaled = scaler.transform(query.fillna(medians))
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    train_regime = model.fit_predict(train_scaled).astype(np.int8)
    query_regime = model.predict(query_scaled).astype(np.int8)
    counts = {str(regime): int((train_regime == regime).sum()) for regime in range(n_clusters)}
    return train_regime, query_regime, counts


def _make_model(seed: int, min_samples_leaf: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=min_samples_leaf,
        max_features=0.75,
        n_jobs=-1,
        random_state=seed,
    )


def _fit_predict_component(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_query: pd.DataFrame,
    train_partition: np.ndarray | None,
    query_partition: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    medians = X_train.median().fillna(0.0)
    train = X_train.fillna(medians)
    query = X_query.reindex(columns=train.columns).fillna(medians)
    sample_weight = 0.25 + 0.75 * (y_train >= 0.10) + np.clip(y_train, 0.0, 1.0)

    if train_partition is None or query_partition is None:
        model = _make_model(seed, min_samples_leaf=25)
        model.fit(train, y_train, sample_weight=sample_weight)
        return model.predict(query)

    prediction = np.full(len(query), np.nan, dtype=float)
    for partition in np.unique(query_partition):
        train_mask = train_partition == partition
        query_mask = query_partition == partition
        if train_mask.sum() < 200:
            continue
        model = _make_model(seed + int(partition), min_samples_leaf=18)
        model.fit(train.loc[train_mask], y_train[train_mask], sample_weight=sample_weight[train_mask])
        prediction[query_mask] = model.predict(query.loc[query_mask])
    return prediction


def fit_component_matrix(
    X_train: pd.DataFrame,
    y_train_kwh: np.ndarray,
    X_query: pd.DataFrame,
    train_regime: np.ndarray,
    query_regime: np.ndarray,
    seeds: tuple[int, ...] = SEEDS,
) -> np.ndarray:
    y_ratio = np.asarray(y_train_kwh, dtype=float) / CAPACITY
    train_phase = lead_phase(X_train.index)
    query_phase = lead_phase(X_query.index)
    seed_components = []
    for seed in seeds:
        global_prediction = _fit_predict_component(
            X_train, y_ratio, X_query, None, None, seed
        )
        phase_prediction = _fit_predict_component(
            X_train, y_ratio, X_query, train_phase, query_phase, seed + 1_000
        )
        regime_prediction = _fit_predict_component(
            X_train, y_ratio, X_query, train_regime, query_regime, seed + 2_000
        )
        for prediction in (phase_prediction, regime_prediction):
            missing = ~np.isfinite(prediction)
            prediction[missing] = global_prediction[missing]
        seed_components.append(
            np.column_stack([global_prediction, phase_prediction, regime_prediction])
        )
    return np.clip(np.mean(seed_components, axis=0) * CAPACITY, 0.0, CAPACITY)


def component_weight_grid() -> list[tuple[float, float, float]]:
    values = (0.0, 0.25, 0.5, 0.75, 1.0)
    weights = []
    for global_weight, phase_weight in product(values, repeat=2):
        regime_weight = 1.0 - global_weight - phase_weight
        if regime_weight >= 0.0 and regime_weight in values:
            weights.append((global_weight, phase_weight, regime_weight))
    return weights


def blend_components(matrix: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(matrix, dtype=float) @ np.asarray(weights, dtype=float)


def inject_member(
    base: np.ndarray,
    member: np.ndarray,
    group_1: np.ndarray,
    group_2: np.ndarray,
    alpha: float,
    max_disagreement: float,
    require_driver_agreement: bool,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base, dtype=float)
    member = np.asarray(member, dtype=float)
    group_1_ratio = np.asarray(group_1, dtype=float) / CAPACITY_KWH["kpx_group_1"]
    group_2_ratio = np.asarray(group_2, dtype=float) / CAPACITY_KWH["kpx_group_2"]
    mask = (
        (base >= 0.10 * CAPACITY)
        & (np.abs(member - base) / CAPACITY <= max_disagreement)
    )
    if require_driver_agreement:
        mask &= np.abs(group_1_ratio - group_2_ratio) <= 0.08
    output = base.copy()
    output[mask] += alpha * (member[mask] - base[mask])
    return np.clip(output, 0.0, CAPACITY), mask


def _metric_delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def evaluate_delta(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    period: np.ndarray,
) -> dict[str, object]:
    before = evaluate_group(truth[period], base[period], CAPACITY)
    after = evaluate_group(truth[period], candidate[period], CAPACITY)
    return {
        "base": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": _metric_delta(before, after),
    }


def _policy_key(item: tuple[InjectionPolicy, dict[str, object]]) -> tuple[float, float, float]:
    policy, report = item
    delta = report["delta"]
    # Prefer score, then FICR, then the more conservative movement.
    return (float(delta["score"]), float(delta["ficr"]), -policy.alpha)


def select_policy(
    truth: np.ndarray,
    base: np.ndarray,
    components: np.ndarray,
    group_1: np.ndarray,
    group_2: np.ndarray,
    period: np.ndarray,
) -> tuple[InjectionPolicy, list[dict[str, object]]]:
    trials: list[tuple[InjectionPolicy, dict[str, object]]] = []
    for weights in component_weight_grid():
        member = blend_components(components, weights)
        for alpha, disagreement, require_agreement in product(
            (0.05, 0.10, 0.15, 0.20, 0.25),
            (0.03, 0.04, 0.05, 0.06, 0.08),
            (True, False),
        ):
            policy = InjectionPolicy(weights, alpha, disagreement, require_agreement)
            candidate, gate = inject_member(
                base, member, group_1, group_2, alpha, disagreement, require_agreement
            )
            metrics = evaluate_delta(truth, base, candidate, period)
            metrics["changed_rows"] = int((gate & period).sum())
            trials.append((policy, metrics))
    selected, selected_metrics = max(trials, key=_policy_key)
    leaderboard = []
    for policy, metrics in sorted(trials, key=_policy_key, reverse=True)[:20]:
        leaderboard.append(
            {
                "policy": {
                    "component_weights": list(policy.component_weights),
                    "alpha": policy.alpha,
                    "max_disagreement": policy.max_disagreement,
                    "require_driver_agreement": policy.require_driver_agreement,
                },
                **metrics,
            }
        )
    if selected_metrics["delta"]["score"] <= 0.0:
        raise RuntimeError("No positive selection-period policy was found")
    return selected, leaderboard


def _monthly_deltas(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    period: np.ndarray,
) -> dict[str, dict[str, float]]:
    output = {}
    # The 00:00 target belongs to the forecast cycle issued on the previous day.
    cycle_month = (index - pd.Timedelta(hours=1)).month
    for month in sorted(np.unique(cycle_month[period])):
        mask = period & (cycle_month == month)
        output[str(month)] = evaluate_delta(truth, base, candidate, mask)["delta"]
    return output


def _bootstrap_days(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    period: np.ndarray,
    n_bootstrap: int = 2_000,
    seed: int = 20_260_717,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    days = index[period].normalize().unique()
    positions = {
        day: np.flatnonzero(period & (index.normalize() == day)) for day in days
    }
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([positions[day] for day in sampled])
        deltas.append(
            evaluate_group(truth[rows], candidate[rows], CAPACITY).score
            - evaluate_group(truth[rows], base[rows], CAPACITY).score
        )
    values = np.asarray(deltas)
    return {
        "n_bootstrap": n_bootstrap,
        "positive_fraction": float((values > 0.0).mean()),
        "q025": float(np.quantile(values, 0.025)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q975": float(np.quantile(values, 0.975)),
    }


def _locked_policy_stress(
    leaderboard: list[dict[str, object]],
    truth: np.ndarray,
    base: np.ndarray,
    components: np.ndarray,
    group_1: np.ndarray,
    group_2: np.ndarray,
    period: np.ndarray,
) -> dict[str, object]:
    rows = []
    for trial in leaderboard:
        policy = trial["policy"]
        member = blend_components(
            components, tuple(float(value) for value in policy["component_weights"])
        )
        candidate, gate = inject_member(
            base,
            member,
            group_1,
            group_2,
            float(policy["alpha"]),
            float(policy["max_disagreement"]),
            bool(policy["require_driver_agreement"]),
        )
        metrics = evaluate_delta(truth, base, candidate, period)
        rows.append(
            {
                "policy": policy,
                "changed_rows": int((gate & period).sum()),
                "delta": metrics["delta"],
            }
        )
    return {
        "tested_top_selection_policies": len(rows),
        "positive_score_policies": int(sum(row["delta"]["score"] > 0.0 for row in rows)),
        "positive_all_metric_policies": int(
            sum(
                row["delta"]["score"] > 0.0
                and row["delta"]["one_minus_nmae"] >= 0.0
                and row["delta"]["ficr"] >= 0.0
                for row in rows
            )
        ),
        "best_locked_rows": sorted(
            rows, key=lambda row: row["delta"]["score"], reverse=True
        )[:5],
    }


def run_experiment(
    data_dir: Path,
    feature_cache_dir: Path,
    driver_cache_path: Path,
    current_submission_path: Path,
    artifact_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    train_weather = pd.read_pickle(feature_cache_dir / "features_train.pkl")
    test_weather = pd.read_pickle(feature_cache_dir / "features_test.pkl")
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")
    driver_cache = np.load(driver_cache_path, allow_pickle=True)
    index = pd.DatetimeIndex(pd.to_datetime(driver_cache[f"{TARGET}__valid_index_ns"]))
    truth = driver_cache[f"{TARGET}__valid_truth"].astype(float)
    group_1 = driver_cache["kpx_group_1__exact_base"].astype(float)
    group_2 = driver_cache["kpx_group_2__exact_base"].astype(float)
    base = driver_cache[f"{TARGET}__exact_base"].astype(float)

    train_rows = labels.index.year == 2023
    transfer_train = labels.loc[train_rows].dropna(
        subset=["kpx_group_1", "kpx_group_2", TARGET]
    )
    X_train = build_cross_section_features(
        train_weather.reindex(transfer_train.index),
        transfer_train["kpx_group_1"].to_numpy(dtype=float),
        transfer_train["kpx_group_2"].to_numpy(dtype=float),
    )
    X_valid = build_cross_section_features(
        train_weather.reindex(index), group_1, group_2
    )
    train_regime, valid_regime, regime_counts = fit_regimes(X_train, X_valid)
    components = fit_component_matrix(
        X_train,
        transfer_train[TARGET].to_numpy(dtype=float),
        X_valid,
        train_regime,
        valid_regime,
    )

    old_member = _fit_transfer_member(labels, group_1, group_2, index, (2023,), 51_000)
    pre_meta = _current_prediction(group_1, group_2, base, old_member, index)
    meta_cache = np.load(artifact_dir.parent / "meta_gate" / "meta_gate_cache.npz")
    if not np.array_equal(meta_cache["valid_index_ns"], index.astype("int64").to_numpy()):
        raise ValueError("Meta-gate and exact-driver validation indexes differ")
    selected_base = meta_cache["valid_candidate"].astype(float)

    h1 = index < H2_START
    h2 = index >= H2_START
    policy, leaderboard = select_policy(
        truth, pre_meta, components, group_1, group_2, h1
    )
    selected_member = blend_components(components, policy.component_weights)
    candidate, gate = inject_member(
        selected_base,
        selected_member,
        group_1,
        group_2,
        policy.alpha,
        policy.max_disagreement,
        policy.require_driver_agreement,
    )
    locked = evaluate_delta(truth, selected_base, candidate, h2)
    component_metrics = {
        name: {
            "h1": evaluate_group(truth[h1], components[h1, i], CAPACITY).to_dict(),
            "h2": evaluate_group(truth[h2], components[h2, i], CAPACITY).to_dict(),
        }
        for i, name in enumerate(("global", "lead_phase", "weather_regime"))
    }

    monthly = _monthly_deltas(truth, selected_base, candidate, index, h2)
    bootstrap = _bootstrap_days(truth, selected_base, candidate, index, h2)
    policy_stress = _locked_policy_stress(
        leaderboard, truth, selected_base, components, group_1, group_2, h2
    )

    # A candidate must improve the untouched half on both the official score and NMAE.
    promote = bool(
        locked["delta"]["score"] > 0.0
        and locked["delta"]["one_minus_nmae"] >= 0.0
        and sum(delta["score"] > 0.0 for delta in monthly.values()) >= 4
        and bootstrap["positive_fraction"] >= 0.60
    )

    final_summary: dict[str, object] = {"promoted": promote, "output": None}
    test_components = None
    if promote:
        current_submission = pd.read_csv(current_submission_path, encoding="utf-8-sig")
        test_index = pd.DatetimeIndex(pd.to_datetime(current_submission["forecast_kst_dtm"]))
        full_train = labels.dropna(subset=["kpx_group_1", "kpx_group_2", TARGET])
        X_full = build_cross_section_features(
            train_weather.reindex(full_train.index),
            full_train["kpx_group_1"].to_numpy(dtype=float),
            full_train["kpx_group_2"].to_numpy(dtype=float),
        )
        test_group_1 = current_submission["kpx_group_1"].to_numpy(dtype=float)
        test_group_2 = current_submission["kpx_group_2"].to_numpy(dtype=float)
        X_test = build_cross_section_features(test_weather.reindex(test_index), test_group_1, test_group_2)
        full_regime, test_regime, final_regime_counts = fit_regimes(X_full, X_test)
        test_components = fit_component_matrix(
            X_full,
            full_train[TARGET].to_numpy(dtype=float),
            X_test,
            full_regime,
            test_regime,
        )
        test_member = blend_components(test_components, policy.component_weights)
        test_candidate, test_gate = inject_member(
            current_submission[TARGET].to_numpy(dtype=float),
            test_member,
            test_group_1,
            test_group_2,
            policy.alpha,
            policy.max_disagreement,
            policy.require_driver_agreement,
        )
        output = current_submission.copy()
        output[TARGET] = test_candidate
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        movement = test_candidate - current_submission[TARGET].to_numpy(dtype=float)
        final_summary = {
            "promoted": True,
            "output": str(output_path),
            "changed_rows": int(test_gate.sum()),
            "changed_ratio": float(test_gate.mean()),
            "mean_absolute_movement_kwh": float(np.abs(movement).mean()),
            "p95_absolute_movement_kwh": float(np.quantile(np.abs(movement), 0.95)),
            "regime_counts": final_regime_counts,
        }

    report: dict[str, object] = {
        "method": "KDD/GEFCom-inspired lead-phase and weather-regime cross-sectional ensemble",
        "feature_count": int(X_train.shape[1]),
        "training_rows_2023": int(len(X_train)),
        "regime_counts_2023": regime_counts,
        "seeds": list(SEEDS),
        "components": component_metrics,
        "selection": {
            "period": "2024-H1 on the pre-meta exact OOF surface",
            "policy": {
                "component_weights": list(policy.component_weights),
                "alpha": policy.alpha,
                "max_disagreement": policy.max_disagreement,
                "require_driver_agreement": policy.require_driver_agreement,
            },
            "top_trials": leaderboard,
        },
        "locked_validation": {
            "period": "2024-H2 on the selected meta-gate OOF surface",
            "changed_rows": int((gate & h2).sum()),
            "metrics": locked,
            "expected_competition_macro_score_delta": locked["delta"]["score"] / 3.0,
            "monthly_deltas": monthly,
            "positive_months": int(sum(row["score"] > 0.0 for row in monthly.values())),
            "day_bootstrap": bootstrap,
            "selection_policy_stress": policy_stress,
        },
        "final": final_summary,
        "decision": (
            "Promoted only when locked H2 score and NMAE improve, at least four of six months improve, "
            "and the day-bootstrap positive fraction is at least 60%."
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "phase_regime_cache.npz",
        valid_index_ns=index.astype("int64").to_numpy(),
        valid_components=components.astype("float32"),
        valid_member=selected_member.astype("float32"),
        valid_candidate=candidate.astype("float32"),
        valid_gate=gate,
        **(
            {"test_components": test_components.astype("float32")}
            if test_components is not None
            else {}
        ),
    )
    (artifact_dir / "phase_regime_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--feature-cache-dir", default="artifacts_final/feature_cache")
    parser.add_argument("--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz")
    parser.add_argument(
        "--current-submission",
        default="submissions/blend_best_crossg3_traj_meta25_p55.csv",
    )
    parser.add_argument("--artifact-dir", default="artifacts_final/phase_regime")
    parser.add_argument(
        "--output",
        default="submissions/blend_best_phase_regime_crossg3.csv",
    )
    parser.add_argument(
        "--agent-evaluation-output",
        help="Optional standardized agent-service Evaluation JSON output.",
    )
    args = parser.parse_args()
    report = run_experiment(
        Path(args.data_dir),
        Path(args.feature_cache_dir),
        Path(args.driver_cache),
        Path(args.current_submission),
        Path(args.artifact_dir),
        Path(args.output),
    )
    if args.agent_evaluation_output:
        from agent_service.adapters import phase_regime_evaluation

        evaluation_path = Path(args.agent_evaluation_output)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.write_text(
            json.dumps(
                phase_regime_evaluation(report).to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
