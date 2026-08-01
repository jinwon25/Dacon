"""Test the incremental value of causal NOAA observations for groups 1/2.

Two otherwise identical shallow residual models are fit: one with only the
frozen active/NWP context and one augmented by issue-time NOAA observations.
Only their prediction difference is allowed to move the incumbent.  This
isolates the external observation signal from the generic residual learner.
Q1 trains the models, Q2 selects one bounded policy, and H2 is opened once.
Group 3 stays frozen after repeated public replacement failures.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.incumbent_residual_noncrossing import (
    COMPONENTS,
    interval_month,
    issue_block_bootstrap_all,
    metric_delta,
    movement_summary,
)
from experiments.kma_observation_block_router import load_observation_features
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
MODELED_TARGETS = ("kpx_group_1", "kpx_group_2")
FROZEN_TARGET = "kpx_group_3"
SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class Policy:
    alpha: float
    cap_ratio: float

    @property
    def name(self) -> str:
        return f"a{int(round(self.alpha * 100)):03d}_cap{int(round(self.cap_ratio * 10000)):03d}bp"


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def policies() -> tuple[Policy, ...]:
    return tuple(
        Policy(alpha, cap)
        for cap in (0.0025, 0.0050, 0.0100)
        for alpha in (0.25, 0.50, 1.00)
    )


def compose_increment(
    active: dict[str, np.ndarray],
    increment_ratio: dict[str, np.ndarray],
    policy: Policy,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for target in TARGETS:
        base = np.asarray(active[target], dtype=float)
        if target == FROZEN_TARGET:
            output[target] = base.copy()
            continue
        bounded = np.clip(
            np.asarray(increment_ratio[target], dtype=float),
            -policy.cap_ratio,
            policy.cap_ratio,
        )
        output[target] = np.clip(
            base + policy.alpha * bounded * CAPACITY_KWH[target],
            0.0,
            CAPACITY_KWH[target],
        )
    return output


def _base_columns(target: str) -> dict[str, str]:
    columns: dict[str, str] = {}
    for source in ("ldaps", "gfs"):
        for variable in ("hub_ws117", "hub_u117", "hub_v117"):
            source_name = f"{source}__{target}__{variable}__idw"
            columns[source_name] = f"{source}__{variable}"
    gust = f"gfs__{target}__surface_0_gust__idw"
    columns[gust] = "gfs__gust"
    return columns


def make_design(
    feature_frame: pd.DataFrame,
    observation_frame: pd.DataFrame,
    index: pd.DatetimeIndex,
    issues: pd.Series,
    active: dict[str, np.ndarray],
    *,
    include_observations: bool,
) -> tuple[pd.DataFrame, dict[str, slice]]:
    weather = feature_frame.reindex(index)
    if weather.empty or weather.isna().all(axis=None):
        raise ValueError("official feature alignment failed")
    issue_index = pd.DatetimeIndex(pd.to_datetime(issues).to_numpy())
    observations = observation_frame.reindex(issue_index)
    observations.index = index
    obs_columns = [column for column in observations if column.startswith("asos_")]
    if include_observations and not obs_columns:
        raise ValueError("NOAA observation design has no numeric features")

    lead = (index - issue_index).total_seconds().to_numpy(dtype=float) / 3600.0
    day = index.dayofyear.to_numpy(dtype=float)
    hour = index.hour.to_numpy(dtype=float)
    blocks: list[pd.DataFrame] = []
    slices: dict[str, slice] = {}
    start = 0
    for position, target in enumerate(MODELED_TARGETS):
        mapping = _base_columns(target)
        missing = [column for column in mapping if column not in weather]
        if missing:
            raise ValueError(f"official features missing for {target}: {missing}")
        block = weather[list(mapping)].rename(columns=mapping).reset_index(drop=True)
        block["active_ratio"] = (
            np.asarray(active[target], dtype=float) / CAPACITY_KWH[target]
        )
        block["lead_hours"] = lead
        block["lead_phase"] = np.floor(np.clip(lead, 0.0, 47.0) / 6.0)
        block["doy_sin"] = np.sin(2.0 * np.pi * day / 365.25)
        block["doy_cos"] = np.cos(2.0 * np.pi * day / 365.25)
        block["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        block["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        block["group_1"] = float(position == 0)
        block["group_2"] = float(position == 1)
        if include_observations:
            obs = observations[obs_columns].reset_index(drop=True)
            obs = obs.apply(pd.to_numeric, errors="coerce")
            block = pd.concat([block, obs.add_prefix("obs__")], axis=1)
        stop = start + len(block)
        slices[target] = slice(start, stop)
        blocks.append(block)
        start = stop
    design = pd.concat(blocks, axis=0, ignore_index=True).astype("float32")
    finite_columns = [
        column
        for column in design
        if np.isfinite(design[column].to_numpy(dtype=float)).any()
    ]
    return design[finite_columns], slices


def _stack_target(
    truth: dict[str, np.ndarray],
    active: dict[str, np.ndarray],
) -> np.ndarray:
    return np.concatenate(
        [
            (
                np.asarray(truth[target], dtype=float)
                - np.asarray(active[target], dtype=float)
            )
            / CAPACITY_KWH[target]
            for target in MODELED_TARGETS
        ]
    )


def _stack_rows(rows: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(rows, dtype=bool), len(MODELED_TARGETS))


def _model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber",
        n_estimators=160,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.20,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def fit_increment_predictions(
    control_design: pd.DataFrame,
    observation_design: pd.DataFrame,
    slices: dict[str, slice],
    target: np.ndarray,
    train_rows: np.ndarray,
    query_rows: np.ndarray,
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[int, dict[str, np.ndarray]]:
    train = _stack_rows(train_rows)
    query = _stack_rows(query_rows)
    output: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        control = _model(seed)
        observation = _model(seed)
        control.fit(control_design.loc[train], target[train], callbacks=[lgb.log_evaluation(0)])
        observation.fit(
            observation_design.loc[train],
            target[train],
            callbacks=[lgb.log_evaluation(0)],
        )
        control_prediction = np.zeros(len(control_design), dtype=float)
        observation_prediction = np.zeros(len(observation_design), dtype=float)
        control_prediction[query] = control.predict(control_design.loc[query])
        observation_prediction[query] = observation.predict(observation_design.loc[query])
        output[seed] = {
            name: (
                observation_prediction[target_slice]
                - control_prediction[target_slice]
            )
            for name, target_slice in slices.items()
        }
    return output


def _mean_increment(
    predictions: dict[int, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        target: np.mean(
            [predictions[seed][target] for seed in predictions], axis=0
        )
        for target in MODELED_TARGETS
    }


def _periods(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    months = interval_month(index)
    return {
        "q1": months <= 3,
        "q2": (months >= 4) & (months <= 6),
        "h1": months <= 6,
        "h2": months >= 7,
        "full": np.ones(len(index), dtype=bool),
    }


def _policy_record(
    policy: Policy,
    truth: dict[str, np.ndarray],
    active: dict[str, np.ndarray],
    predictions: dict[int, dict[str, np.ndarray]],
    rows: np.ndarray,
    month_values: np.ndarray,
) -> dict[str, Any]:
    ensemble = compose_increment(active, _mean_increment(predictions), policy)
    seed_candidates = {
        seed: compose_increment(active, prediction, policy)
        for seed, prediction in predictions.items()
    }
    delta = metric_delta(truth, active, ensemble, rows)
    seed_deltas = {
        str(seed): metric_delta(truth, active, candidate, rows)
        for seed, candidate in seed_candidates.items()
    }
    months = {
        str(month): metric_delta(
            truth,
            active,
            ensemble,
            rows & (month_values == month),
        )
        for month in sorted(set(month_values[rows]))
    }
    eligible = bool(
        all(delta[component] > 0.0 for component in COMPONENTS)
        and all(
            value[component] >= 0.0
            for value in seed_deltas.values()
            for component in COMPONENTS
        )
        and all(value["score"] >= 0.0 for value in months.values())
    )
    return {
        "policy": asdict(policy) | {"name": policy.name},
        "delta": delta,
        "seed_deltas": seed_deltas,
        "monthly_deltas": months,
        "movement": movement_summary(active, ensemble),
        "eligible": eligible,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    observation_frame, manifest_validation = load_observation_features(
        _rooted(args.observation_features), _rooted(args.manifest)
    )
    baselines, truths, index, issues = load_frozen_validation_baselines(
        _rooted(args.primary_cache),
        _rooted(args.residual_cache),
        _rooted(args.group3_cache),
    )
    active = {target: baselines[target].to_numpy(dtype=float) for target in TARGETS}
    truth = {target: truths[target].to_numpy(dtype=float) for target in TARGETS}
    feature_frame = pd.read_pickle(_rooted(args.features_train))
    control_design, slices = make_design(
        feature_frame,
        observation_frame,
        index,
        issues,
        active,
        include_observations=False,
    )
    observation_design, observation_slices = make_design(
        feature_frame,
        observation_frame,
        index,
        issues,
        active,
        include_observations=True,
    )
    if slices != observation_slices:
        raise ValueError("control and observation target slices differ")
    target = _stack_target(truth, active)
    periods = _periods(index)
    months = interval_month(index)

    development_predictions = fit_increment_predictions(
        control_design,
        observation_design,
        slices,
        target,
        periods["q1"],
        periods["q2"],
    )
    development = [
        _policy_record(
            policy,
            truth,
            active,
            development_predictions,
            periods["q2"],
            months,
        )
        for policy in policies()
    ]
    eligible = [record for record in development if record["eligible"]]
    selected = max(
        eligible,
        key=lambda record: (
            min(value["score"] for value in record["monthly_deltas"].values()),
            record["delta"]["score"],
            -record["policy"]["alpha"],
            -record["policy"]["cap_ratio"],
        ),
        default=None,
    )
    locked = None
    promoted = False
    if selected is not None:
        policy = Policy(
            alpha=float(selected["policy"]["alpha"]),
            cap_ratio=float(selected["policy"]["cap_ratio"]),
        )
        locked_predictions = fit_increment_predictions(
            control_design,
            observation_design,
            slices,
            target,
            periods["h1"],
            periods["h2"],
        )
        locked = _policy_record(
            policy,
            truth,
            active,
            locked_predictions,
            periods["h2"],
            months,
        )
        candidate = compose_increment(active, _mean_increment(locked_predictions), policy)
        bootstrap = issue_block_bootstrap_all(
            truth,
            active,
            candidate,
            index,
            issues,
            periods["h2"],
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
        bootstrap_pass = all(
            bootstrap["summary"][component]["q05"] >= 0.0
            for component in COMPONENTS
        )
        promoted = bool(locked["eligible"] and bootstrap_pass)
        locked["issue_block_bootstrap"] = bootstrap
        locked["promotion_eligible"] = promoted

    report = {
        "family": "noaa_observation_incremental_residual",
        "method": (
            "difference of paired shallow residual models with and without "
            "causal multi-station NOAA observation features"
        ),
        "contract": {
            "development": "Q1 fit -> Q2 bounded policy selection",
            "confirmation": "H1 refit -> H2 opened once",
            "increment_only": "observation model prediction minus identical control prediction",
            "group3_frozen": True,
            "public_score_used_for_selection": False,
            "submission_writer": False,
        },
        "manifest_validation": manifest_validation,
        "design": {
            "rows": int(len(control_design)),
            "control_columns": int(control_design.shape[1]),
            "observation_columns": int(observation_design.shape[1]),
            "seeds": list(SEEDS),
            "policies": len(development),
        },
        "development": {
            "eligible_policies": len(eligible),
            "selected": selected,
            "records": development,
        },
        "locked": locked,
        "verdict": "promoted" if promoted else "rejected_fail_closed",
    }
    output = _rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation-features",
        default="artifacts_final/external_weather/noaa_isd/issue_features_2024.csv",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts_final/external_weather/noaa_isd/manifest_2024.json",
    )
    parser.add_argument(
        "--features-train", default="artifacts_final/feature_cache/features_train.pkl"
    )
    parser.add_argument(
        "--primary-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--residual-cache",
        default="artifacts_final/lineage/kma_jma_msm_stencil_production_20260726.npz",
    )
    parser.add_argument(
        "--group3-cache",
        default=(
            "artifacts_final/external_weather/kma_um_regional_context_2024/"
            "power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "noaa_observation_incremental_residual_20260802.json"
        ),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "selected": report["development"]["selected"],
                "locked": report["locked"],
                "verdict": report["verdict"],
                "output": _rooted(args.output).relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
