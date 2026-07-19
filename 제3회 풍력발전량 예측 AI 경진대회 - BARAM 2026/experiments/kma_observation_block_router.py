"""Forward-validated group-3 router using causal KMA ASOS state features.

This experiment is deliberately diagnostic-only.  Q1 fits utility models that
route Q2; only a policy positive for both seeds, both official components and
every Q2 month may open locked H2.  An otherwise identical no-observation
control measures whether ASOS adds information rather than merely adding a new
router implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from agent_service.compliance import (
    audit_observation_availability,
    validate_external_data_manifest,
)
from experiments.blocked_rolling_validation import load_issue_times
from experiments.dynamic_router_oracle import load_experts
from experiments.weather_similarity_router import (
    H2_START,
    Q2_START,
    ROOT,
    build_blocks,
    evaluate_period,
    period_blocks,
    utility_table,
)
from src.metrics import CAPACITY_KWH, evaluate_group


CAPACITY = CAPACITY_KWH["kpx_group_3"]
END = pd.Timestamp("2025-01-02")
DEFAULT_FEATURES = (
    ROOT / "artifacts_final" / "external_weather" / "kma_asos_2024" / "issue_features.csv"
)
DEFAULT_MANIFEST = DEFAULT_FEATURES.parent / "manifest.json"
DEFAULT_OUTPUT = (
    ROOT / "artifacts_final" / "diagnostics" / "kma_observation_block_router.json"
)
SEEDS = (17, 29)
RETAINED_EXPERTS = (
    "incumbent_finesweep",
    "lineage_exact_base",
    "lineage_stack5",
    "lineage_blend_v1",
    "lineage_weighted_member",
    "trajectory_residual",
    "spatiotemporal_seed_mean",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float


@dataclass(frozen=True)
class Policy:
    model: ModelSpec
    alpha: float
    coverage: float

    @property
    def name(self) -> str:
        return (
            f"{self.model.name}_a{int(round(self.alpha * 100)):02d}_"
            f"c{int(round(self.coverage * 100)):02d}"
        )


def model_specs() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec("shallow", 140, 3, 0.03, 30.0),
        ModelSpec("moderate", 180, 4, 0.025, 50.0),
    )


def policies() -> tuple[Policy, ...]:
    return tuple(
        Policy(model, alpha, coverage)
        for model, alpha, coverage in itertools.product(
            model_specs(), (0.05, 0.10, 0.15), (0.05, 0.10, 0.15)
        )
    )


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, ModelSpec):
        return value.__dict__
    if isinstance(value, Policy):
        return {"name": value.name, **value.__dict__, "model": value.model.__dict__}
    return value


def load_observation_features(
    feature_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation = validate_external_data_manifest(manifest_path, ROOT)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_record = manifest.get("feature_file", {})
    expected_hash = str(feature_record.get("sha256", ""))
    actual_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("KMA ASOS feature-file checksum does not match its manifest")

    frame = pd.read_csv(feature_path, encoding="utf-8-sig")
    required = {
        "data_available_kst_dtm",
        "safe_observation_cutoff_kst",
        "latest_observation_kst",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"KMA ASOS feature file is missing columns: {sorted(missing)}")
    for column in required:
        frame[column] = pd.to_datetime(frame[column])
    if frame["data_available_kst_dtm"].duplicated().any():
        raise ValueError("KMA ASOS feature file contains duplicate issue rows")
    delay = int(manifest["availability_evidence"]["conservative_delay_minutes"])
    audit_observation_availability(
        frame.rename(
            columns={
                "data_available_kst_dtm": "prediction_reference_kst",
                "latest_observation_kst": "observation_kst",
            }
        ),
        conservative_publication_delay=timedelta(minutes=delay),
    )
    if (
        frame["latest_observation_kst"] > frame["safe_observation_cutoff_kst"]
    ).any():
        raise ValueError("KMA ASOS features include an observation after the safe cutoff")
    numeric = [column for column in frame.columns if column.startswith("asos_")]
    if not numeric:
        raise ValueError("KMA ASOS feature file contains no numeric station features")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if np.isinf(frame[numeric].to_numpy(dtype=float)).any():
        raise ValueError("KMA ASOS station features contain infinite values")
    return frame.set_index("data_available_kst_dtm").sort_index(), validation


def attach_observation_features(
    frame: pd.DataFrame,
    rows_by_block: dict[str, np.ndarray],
    issue_times: pd.DatetimeIndex,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()
    issue_by_block: list[pd.Timestamp] = []
    for block in output.index:
        rows = rows_by_block[str(block)]
        block_issues = pd.DatetimeIndex(issue_times[rows]).unique()
        if len(block_issues) != 1:
            raise ValueError("a six-hour block spans more than one issue cycle")
        issue_by_block.append(pd.Timestamp(block_issues[0]))
    aligned = observations.reindex(pd.DatetimeIndex(issue_by_block))
    if aligned.index.to_series().isna().any() or aligned.empty:
        raise ValueError("KMA ASOS issue-feature alignment failed")
    missing_issues = aligned.filter(like="asos_").isna().all(axis=1)
    if missing_issues.any():
        raise ValueError(
            f"KMA ASOS features are unavailable for {int(missing_issues.sum())} router blocks"
        )
    additions: dict[str, np.ndarray] = {
        "issue_ns": pd.DatetimeIndex(issue_by_block).asi8
    }
    for column in aligned.columns:
        if not column.startswith("asos_"):
            continue
        additions[f"obs__{column}"] = aligned[column].to_numpy(dtype=float)
    return pd.concat(
        [output, pd.DataFrame(additions, index=output.index)], axis=1
    )


def _feature_columns(frame: pd.DataFrame, include_observations: bool) -> list[str]:
    metadata = {"representative_ns", "issue_ns", "phase", "rows"}
    columns = [column for column in frame.columns if column not in metadata]
    if not include_observations:
        columns = [column for column in columns if not column.startswith("obs__")]
    retained = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            retained.append(column)
    if not retained:
        raise ValueError("router has no usable feature columns")
    return retained


def fit_predict_utility(
    frame: pd.DataFrame,
    train_blocks: np.ndarray,
    query_blocks: np.ndarray,
    train_utility: pd.DataFrame,
    model_spec: ModelSpec,
    seed: int,
    *,
    include_observations: bool,
) -> pd.DataFrame:
    columns = _feature_columns(frame, include_observations)
    train_x = frame.loc[train_blocks, columns].apply(pd.to_numeric, errors="coerce")
    query_x = frame.loc[query_blocks, columns].apply(pd.to_numeric, errors="coerce")
    output: dict[str, np.ndarray] = {}
    for expert in train_utility.columns:
        target = train_utility.loc[train_blocks, expert].to_numpy(dtype=float) * 10_000.0
        model = CatBoostRegressor(
            iterations=model_spec.iterations,
            depth=model_spec.depth,
            learning_rate=model_spec.learning_rate,
            l2_leaf_reg=model_spec.l2_leaf_reg,
            loss_function="RMSE",
            random_seed=seed,
            random_strength=0.5,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
        )
        model.fit(train_x, target)
        output[str(expert)] = model.predict(query_x) / 10_000.0
    return pd.DataFrame(output, index=query_blocks)


def route_predictions(
    experts: dict[str, np.ndarray],
    rows_by_block: dict[str, np.ndarray],
    query_blocks: np.ndarray,
    predicted_utility: pd.DataFrame,
    *,
    alpha: float,
    coverage: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 < alpha <= 1.0 or not 0.0 < coverage <= 1.0:
        raise ValueError("router alpha and coverage must be in (0, 1]")
    if not predicted_utility.index.equals(pd.Index(query_blocks)):
        predicted_utility = predicted_utility.reindex(query_blocks)
    if predicted_utility.isna().any().any():
        raise ValueError("router utility predictions are incomplete")
    choices = []
    for block, values in predicted_utility.iterrows():
        position = int(np.argmax(values.to_numpy(dtype=float)))
        choices.append(
            {
                "block": str(block),
                "expert": str(values.index[position]),
                "predicted_utility": float(values.iloc[position]),
            }
        )
    eligible = [choice for choice in choices if choice["predicted_utility"] > 0.0]
    keep = min(int(np.floor(coverage * len(query_blocks))), len(eligible))
    selected = {
        choice["block"]
        for choice in sorted(
            eligible, key=lambda item: item["predicted_utility"], reverse=True
        )[:keep]
    }
    base = experts["incumbent_finesweep"]
    output = base.copy()
    selected_experts: dict[str, int] = {}
    for choice in choices:
        if choice["block"] not in selected:
            continue
        rows = rows_by_block[choice["block"]]
        expert = choice["expert"]
        output[rows] = np.clip(
            base[rows] + alpha * (experts[expert][rows] - base[rows]),
            0.0,
            CAPACITY,
        )
        selected_experts[expert] = selected_experts.get(expert, 0) + 1
    return output, {
        "available_blocks": int(len(query_blocks)),
        "positive_utility_blocks": int(len(eligible)),
        "selected_blocks": int(len(selected)),
        "coverage": float(len(selected) / max(len(query_blocks), 1)),
        "selected_experts": selected_experts,
    }


def _incremental_evaluation(
    index: pd.DatetimeIndex,
    truth: np.ndarray,
    control: np.ndarray,
    observation: np.ndarray,
    rows_by_block: dict[str, np.ndarray],
    block_names: np.ndarray,
) -> dict[str, float]:
    rows = np.zeros(len(index), dtype=bool)
    for block in block_names:
        rows[rows_by_block[str(block)]] = True
    before = evaluate_group(truth[rows], control[rows], CAPACITY)
    after = evaluate_group(truth[rows], observation[rows], CAPACITY)
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def _components_positive(evaluation: dict[str, Any]) -> bool:
    delta = evaluation["delta"]
    return bool(
        delta["score"] > 0.0
        and delta["one_minus_nmae"] >= 0.0
        and delta["ficr"] >= 0.0
        and evaluation["worst_month_score_delta"] >= 0.0
    )


def qualifies(
    ensemble: dict[str, Any],
    seeds: dict[int, dict[str, Any]],
    incremental: dict[str, float],
) -> bool:
    return bool(
        _components_positive(ensemble)
        and all(_components_positive(value) for value in seeds.values())
        and incremental["score"] > 0.0
        and incremental["one_minus_nmae"] >= 0.0
        and incremental["ficr"] >= 0.0
    )


def issue_bootstrap(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    issue_times: pd.DatetimeIndex,
    rows: np.ndarray,
    *,
    samples: int,
    seed: int = 20260719,
) -> dict[str, float | int]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    issues = pd.unique(np.asarray(issue_times)[rows])
    positions = {
        issue: np.flatnonzero(rows & (np.asarray(issue_times) == issue)) for issue in issues
    }
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for iteration in range(samples):
        sampled = rng.choice(issues, size=len(issues), replace=True)
        selected = np.concatenate([positions[issue] for issue in sampled])
        deltas[iteration] = (
            evaluate_group(truth[selected], candidate[selected], CAPACITY).score
            - evaluate_group(truth[selected], base[selected], CAPACITY).score
        )
    return {
        "samples": samples,
        "positive_fraction": float(np.mean(deltas > 0.0)),
        "q05": float(np.quantile(deltas, 0.05)),
        "median": float(np.quantile(deltas, 0.50)),
        "q95": float(np.quantile(deltas, 0.95)),
    }


def _prediction_surfaces(
    frame: pd.DataFrame,
    train_blocks: np.ndarray,
    query_blocks: np.ndarray,
    train_utility: pd.DataFrame,
    model: ModelSpec,
) -> dict[str, dict[int, pd.DataFrame]]:
    surfaces: dict[str, dict[int, pd.DataFrame]] = {"observation": {}, "control": {}}
    for seed in SEEDS:
        surfaces["observation"][seed] = fit_predict_utility(
            frame,
            train_blocks,
            query_blocks,
            train_utility,
            model,
            seed,
            include_observations=True,
        )
        surfaces["control"][seed] = fit_predict_utility(
            frame,
            train_blocks,
            query_blocks,
            train_utility,
            model,
            seed,
            include_observations=False,
        )
    return surfaces


def _evaluate_policy(
    policy: Policy,
    surfaces: dict[str, dict[int, pd.DataFrame]],
    experts: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    truth: np.ndarray,
    rows_by_block: dict[str, np.ndarray],
    query_blocks: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions: dict[str, dict[int, np.ndarray]] = {"observation": {}, "control": {}}
    routing: dict[str, dict[int, dict[str, Any]]] = {"observation": {}, "control": {}}
    for branch in ("observation", "control"):
        for seed in SEEDS:
            predictions[branch][seed], routing[branch][seed] = route_predictions(
                experts,
                rows_by_block,
                query_blocks,
                surfaces[branch][seed],
                alpha=policy.alpha,
                coverage=policy.coverage,
            )
    observation_mean = np.mean(
        [predictions["observation"][seed] for seed in SEEDS], axis=0
    )
    control_mean = np.mean([predictions["control"][seed] for seed in SEEDS], axis=0)
    seed_evaluations = {
        seed: evaluate_period(
            index,
            truth,
            experts["incumbent_finesweep"],
            predictions["observation"][seed],
            rows_by_block,
            query_blocks,
        )
        for seed in SEEDS
    }
    ensemble = evaluate_period(
        index,
        truth,
        experts["incumbent_finesweep"],
        observation_mean,
        rows_by_block,
        query_blocks,
    )
    control = evaluate_period(
        index,
        truth,
        experts["incumbent_finesweep"],
        control_mean,
        rows_by_block,
        query_blocks,
    )
    incremental = _incremental_evaluation(
        index,
        truth,
        control_mean,
        observation_mean,
        rows_by_block,
        query_blocks,
    )
    result = {
        "policy": policy,
        "routing": routing,
        "ensemble": ensemble,
        "seeds": seed_evaluations,
        "control": control,
        "incremental_vs_no_observation_control": incremental,
    }
    result["qualifies"] = qualifies(ensemble, seed_evaluations, incremental)
    return result, observation_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    args = parser.parse_args()

    observations, manifest_validation = load_observation_features(
        args.observation_features, args.manifest
    )
    index, truth, all_experts = load_experts()
    experts = {name: all_experts[name] for name in RETAINED_EXPERTS}
    issue = load_issue_times(ROOT / "data" / "train" / "gfs_train.csv", index)
    base_frame, rows_by_block = build_blocks(index, issue, experts)
    frame = attach_observation_features(base_frame, rows_by_block, issue, observations)
    q1 = period_blocks(frame, None, Q2_START)
    q2 = period_blocks(frame, Q2_START, H2_START)
    h1 = period_blocks(frame, None, H2_START)
    h2 = period_blocks(frame, H2_START, END)

    q1_utility = utility_table(truth, experts, rows_by_block, q1)
    development: list[dict[str, Any]] = []
    surfaces_by_model = {
        model.name: _prediction_surfaces(frame, q1, q2, q1_utility, model)
        for model in model_specs()
    }
    for policy in policies():
        record, _ = _evaluate_policy(
            policy,
            surfaces_by_model[policy.model.name],
            experts,
            index,
            truth,
            rows_by_block,
            q2,
        )
        development.append(record)

    eligible = [record for record in development if record["qualifies"]]
    selected = max(
        eligible,
        key=lambda record: (
            min(value["delta"]["score"] for value in record["seeds"].values()),
            record["ensemble"]["delta"]["score"],
        ),
        default=None,
    )
    locked = None
    promoted = False
    if selected is not None:
        policy = selected["policy"]
        h1_utility = utility_table(truth, experts, rows_by_block, h1)
        surfaces = _prediction_surfaces(frame, h1, h2, h1_utility, policy.model)
        locked_record, locked_prediction = _evaluate_policy(
            policy,
            surfaces,
            experts,
            index,
            truth,
            rows_by_block,
            h2,
        )
        h2_rows = np.zeros(len(index), dtype=bool)
        for block in h2:
            h2_rows[rows_by_block[str(block)]] = True
        bootstrap = issue_bootstrap(
            truth,
            experts["incumbent_finesweep"],
            locked_prediction,
            issue,
            h2_rows,
            samples=args.bootstrap_samples,
        )
        promoted = bool(
            locked_record["qualifies"]
            and bootstrap["q05"] >= 0.0
            and bootstrap["positive_fraction"] >= 0.90
        )
        locked = {
            **locked_record,
            "issue_bootstrap": bootstrap,
            "promoted": promoted,
        }

    report = {
        "method": "causal KMA ASOS group-3 six-hour utility router",
        "contract": {
            "dependency_unit": "one issue cycle and one six-hour lead phase",
            "observation_boundary": "latest observation plus conservative lag <= issue time",
            "development": "Q1 fit -> Q2 policy selection",
            "locked": "selected policy only; H1 fit -> H2 opened once",
            "seed_gate": "seeds 17 and 29 must each improve both official components",
            "incremental_gate": "ASOS branch must beat identical no-observation control",
            "bootstrap_gate": "locked issue-cycle q05 >= 0 and positive fraction >= 0.90",
            "submission_created": False,
        },
        "manifest_validation": manifest_validation,
        "features": {
            "observation_columns": int(
                sum(column.startswith("obs__") for column in frame.columns)
            ),
            "control_columns": int(len(_feature_columns(frame, False))),
            "observation_augmented_columns": int(len(_feature_columns(frame, True))),
        },
        "experts": list(experts),
        "block_counts": {
            "q1": int(len(q1)),
            "q2": int(len(q2)),
            "h1": int(len(h1)),
            "h2": int(len(h2)),
        },
        "policy_count": int(len(development)),
        "qualified_q2_policy_count": int(len(eligible)),
        "selected_policy": None if selected is None else selected["policy"],
        "selected_development": selected,
        "locked": locked,
        "decision": (
            "promotion gate passed; deployment construction requires a separate audit"
            if promoted
            else "rejected; no submission created"
        ),
        "top_q2": sorted(
            development,
            key=lambda record: record["ensemble"]["delta"]["score"],
            reverse=True,
        )[:5],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_to_builtin(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            _to_builtin(
                {
                    "qualified_q2_policy_count": len(eligible),
                    "selected_policy": report["selected_policy"],
                    "locked": locked,
                    "decision": report["decision"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
