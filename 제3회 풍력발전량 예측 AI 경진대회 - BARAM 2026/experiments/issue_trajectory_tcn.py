"""Leakage-safe 24-hour issue-trajectory TCN expert.

The provided NWP files expose one complete 24-hour target trajectory at each
forecast issue.  This experiment treats that issue as the independent sample
and learns the full trajectory jointly for all three KPX groups.  It is
deliberately different from the incumbent row-wise tree and monotone
power-curve experts.

Validation contract
-------------------
* train only on 2022-2023 issues and predict 2024;
* select a quantile/blend weight on 2024 Q1;
* confirm the frozen choice on Q2 and locked H2;
* fit 2022-2024 and predict 2025 only for targets passing every gate.

No test generation labels or public leaderboard scores are used for model or
policy selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.blocked_rolling_validation import evaluate_blocked_rolling
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from experiments.kma_year_forward_quantile_blend import (
    apply_bounded_blend,
    metric_delta,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
QUANTILES = (0.50, 0.65, 0.80)
WEIGHTS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
SEEDS = (17, 29, 41)
ID_COLUMNS = ("forecast_id", "forecast_kst_dtm")


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_trajectory_columns(frame: pd.DataFrame) -> list[str]:
    """Retain compact wind physics, group IDW stencils, and calendar phase."""
    calendar = {
        "lead_hour",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    }
    wind_tokens = (
        "__ws",
        "hub_ws117",
        "hub_u117",
        "hub_v117",
        "surface_0_gust",
    )
    columns = [
        column
        for column in frame.columns
        if column in calendar
        or (
            any(token in column for token in wind_tokens)
            and (
                "__idw" in column
                or column.endswith(("__mean", "__std", "__min", "__max"))
            )
        )
    ]
    if not columns:
        raise ValueError("no trajectory features selected")
    return columns


def load_issue_series(path: Path) -> pd.Series:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["forecast_kst_dtm", "data_available_kst_dtm"],
        parse_dates=["forecast_kst_dtm", "data_available_kst_dtm"],
    ).drop_duplicates("forecast_kst_dtm")
    if frame["forecast_kst_dtm"].duplicated().any():
        raise ValueError(f"duplicate forecast timestamps in {path}")
    return frame.set_index("forecast_kst_dtm")[
        "data_available_kst_dtm"
    ].sort_index()


@dataclass(frozen=True)
class SequenceBundle:
    features: np.ndarray
    targets: np.ndarray | None
    target_mask: np.ndarray | None
    timestamps: np.ndarray
    issues: np.ndarray


def make_sequence_bundle(
    features: pd.DataFrame,
    issue: pd.Series,
    labels: pd.DataFrame | None,
) -> SequenceBundle:
    """Convert aligned hourly rows into complete, ordered 24-hour issues."""
    issue = pd.to_datetime(issue.reindex(features.index))
    if issue.isna().any():
        raise ValueError("issue timestamps do not cover trajectory features")
    groups = issue.groupby(issue, sort=True).groups
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    mask_blocks: list[np.ndarray] = []
    timestamp_blocks: list[np.ndarray] = []
    retained_issues: list[np.datetime64] = []
    for issue_time, positions in groups.items():
        block_index = pd.DatetimeIndex(positions).sort_values()
        if len(block_index) != 24:
            continue
        expected = pd.date_range(block_index[0], periods=24, freq="h")
        if not block_index.equals(expected):
            continue
        values = features.reindex(block_index).to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            continue
        feature_blocks.append(values)
        timestamp_blocks.append(block_index.to_numpy(dtype="datetime64[ns]"))
        retained_issues.append(np.datetime64(issue_time, "ns"))
        if labels is not None:
            raw = labels.reindex(block_index)[list(TARGETS)].to_numpy(
                dtype=np.float32
            )
            observed = np.isfinite(raw)
            normalized = raw.copy()
            for position, target in enumerate(TARGETS):
                normalized[:, position] /= CAPACITY_KWH[target]
            target_blocks.append(np.nan_to_num(normalized, nan=0.0))
            mask_blocks.append(observed)
    if not feature_blocks:
        raise ValueError("no complete 24-hour issue trajectories")
    return SequenceBundle(
        features=np.stack(feature_blocks),
        targets=np.stack(target_blocks) if target_blocks else None,
        target_mask=np.stack(mask_blocks) if mask_blocks else None,
        timestamps=np.stack(timestamp_blocks),
        issues=np.asarray(retained_issues, dtype="datetime64[ns]"),
    )


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.norm(values)
        values = self.conv(values.transpose(1, 2)).transpose(1, 2)
        return residual + self.dropout(self.activation(values))


class IssueTrajectoryTCN(nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(
            hidden,
            len(TARGETS) * len(QUANTILES),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input(values)
        for block in self.blocks:
            values = block(values)
        values = self.output(self.output_norm(values))
        shape = (*values.shape[:2], len(TARGETS), len(QUANTILES))
        return torch.sigmoid(values.reshape(shape))


def masked_pinball_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    error = truth.unsqueeze(-1) - prediction
    quantiles = torch.as_tensor(
        QUANTILES,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    loss = torch.maximum(
        quantiles * error,
        (quantiles - 1.0) * error,
    )
    weights = observed.unsqueeze(-1).to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def standardize(
    train: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = train.reshape(-1, train.shape[-1]).mean(axis=0)
    scale = train.reshape(-1, train.shape[-1]).std(axis=0)
    scale[scale < 1e-5] = 1.0
    return (
        ((train - center) / scale).astype(np.float32),
        ((query - center) / scale).astype(np.float32),
        center.astype(np.float32),
        scale.astype(np.float32),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_predict(
    train: SequenceBundle,
    query: SequenceBundle,
    *,
    seed: int,
    hidden: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if train.targets is None or train.target_mask is None:
        raise ValueError("training bundle has no labels")
    _seed_everything(seed)
    train_x, query_x, _, _ = standardize(train.features, query.features)
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train.targets.astype(np.float32)),
        torch.from_numpy(train.target_mask),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = IssueTrajectoryTCN(
        train_x.shape[-1],
        hidden=hidden,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    epoch_losses: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        count = 0
        for x_batch, y_batch, mask_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = masked_pinball_loss(prediction, y_batch, mask_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(x_batch)
            count += len(x_batch)
        epoch_losses.append(total / max(count, 1))
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(query_x), batch_size):
            batch = torch.from_numpy(query_x[start : start + batch_size])
            predictions.append(model(batch).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    return prediction, {
        "seed": int(seed),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "initial_loss": float(epoch_losses[0]),
        "final_loss": float(epoch_losses[-1]),
    }


def flatten_prediction(
    prediction: np.ndarray,
    timestamps: np.ndarray,
    target_position: int,
    quantile_position: int,
    target_index: pd.DatetimeIndex,
) -> np.ndarray:
    series = pd.Series(
        prediction[:, :, target_position, quantile_position].reshape(-1),
        index=pd.DatetimeIndex(timestamps.reshape(-1)),
    )
    aligned = series.reindex(target_index)
    if aligned.isna().any():
        raise ValueError("trajectory prediction does not cover target index")
    return aligned.to_numpy(dtype=float)


def select_policy(
    truth: np.ndarray,
    reference: np.ndarray,
    predictions: dict[float, np.ndarray],
    capacity: float,
    rows: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for quantile, expert in predictions.items():
        for weight in WEIGHTS:
            candidate = apply_bounded_blend(
                reference,
                expert,
                weight=weight,
                capacity=capacity,
            )
            delta = metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                rows,
            )
            records.append(
                {
                    "quantile": float(quantile),
                    "weight": float(weight),
                    "delta": delta,
                    "eligible": bool(min(delta.values()) > 0.0),
                }
            )
    eligible = [record for record in records if record["eligible"]]
    pool = eligible or records
    selected = max(
        pool,
        key=lambda record: (
            record["delta"]["score"],
            min(record["delta"].values()),
            -record["weight"],
        ),
    )
    return {
        "quantile": float(selected["quantile"]),
        "weight": float(selected["weight"]),
        "selection_eligible": bool(selected["eligible"]),
    }, records


def _periods(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "q1": np.asarray(index < pd.Timestamp("2024-04-01")),
        "q2": np.asarray(
            (index >= pd.Timestamp("2024-04-01"))
            & (index < pd.Timestamp("2024-07-01"))
        ),
        "h2": np.asarray(index >= pd.Timestamp("2024-07-01")),
        "full": np.ones(len(index), dtype=bool),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    feature_train = pd.read_pickle(_rooted(args.features_train))
    feature_test = pd.read_pickle(_rooted(args.features_test))
    columns = select_trajectory_columns(feature_train)
    feature_train = feature_train[columns]
    feature_test = feature_test.reindex(columns=columns)
    labels = pd.read_csv(
        _rooted(args.labels),
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    issue_train = load_issue_series(_rooted(args.gfs_train))
    issue_test = load_issue_series(_rooted(args.gfs_test))

    all_train = make_sequence_bundle(feature_train, issue_train, labels)
    validation_train_rows = all_train.timestamps[:, 0] < np.datetime64(
        "2024-01-01"
    )
    validation_query_rows = (
        (all_train.timestamps[:, 0] >= np.datetime64("2024-01-01"))
        & (all_train.timestamps[:, 0] < np.datetime64("2025-01-01"))
    )

    def subset(bundle: SequenceBundle, rows: np.ndarray) -> SequenceBundle:
        return SequenceBundle(
            features=bundle.features[rows],
            targets=(
                bundle.targets[rows] if bundle.targets is not None else None
            ),
            target_mask=(
                bundle.target_mask[rows]
                if bundle.target_mask is not None
                else None
            ),
            timestamps=bundle.timestamps[rows],
            issues=bundle.issues[rows],
        )

    fit_bundle = subset(all_train, validation_train_rows)
    validation_bundle = subset(all_train, validation_query_rows)
    raw_validation_cache = _rooted(args.raw_validation_cache)
    if raw_validation_cache.exists():
        retained = np.load(raw_validation_cache, allow_pickle=False)
        validation_prediction = retained["prediction"].astype(float)
        if validation_prediction.shape[:2] != (
            len(validation_bundle.features),
            24,
        ):
            raise ValueError("raw TCN validation cache shape differs")
        training_records = json.loads(
            str(retained["training_records_json"].item())
        )
    else:
        seed_predictions: list[np.ndarray] = []
        training_records = []
        for seed in SEEDS:
            prediction, record = fit_predict(
                fit_bundle,
                validation_bundle,
                seed=seed,
                hidden=args.hidden,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            seed_predictions.append(prediction)
            training_records.append(record)
        validation_prediction = np.mean(seed_predictions, axis=0)
        raw_validation_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_validation_cache,
            prediction=validation_prediction.astype("float32"),
            training_records_json=np.asarray(
                json.dumps(training_records, allow_nan=False)
            ),
        )

    baselines, truths, baseline_index, baseline_issues = (
        load_frozen_validation_baselines(
            _rooted(args.primary_cache),
            _rooted(args.residual_cache),
            _rooted(args.group3_cache),
        )
    )
    common = baseline_index.intersection(
        pd.DatetimeIndex(validation_bundle.timestamps.reshape(-1))
    )
    periods = _periods(common)
    validation: dict[str, Any] = {}
    promoted: list[str] = []
    selected_specs: dict[str, dict[str, float]] = {}
    cache: dict[str, np.ndarray] = {}
    for target_position, target in enumerate(TARGETS):
        capacity = CAPACITY_KWH[target]
        truth = truths[target].reindex(common).to_numpy(dtype=float)
        reference = baselines[target].reindex(common).to_numpy(dtype=float)
        predictions = {
            quantile: flatten_prediction(
                validation_prediction,
                validation_bundle.timestamps,
                target_position,
                quantile_position,
                common,
            )
            * capacity
            for quantile_position, quantile in enumerate(QUANTILES)
        }
        spec, selection_records = select_policy(
            truth,
            reference,
            predictions,
            capacity,
            periods["q1"],
        )
        selected_specs[target] = spec
        candidate = apply_bounded_blend(
            reference,
            predictions[spec["quantile"]],
            weight=spec["weight"],
            capacity=capacity,
        )
        deltas = {
            name: metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                rows,
            )
            for name, rows in periods.items()
        }
        monthly_h2 = {
            str(month): metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                periods["h2"] & np.asarray(common.month == month),
            )
            for month in range(7, 13)
        }
        positive_months = sum(
            delta["score"] > 0.0 for delta in monthly_h2.values()
        )
        bootstrap = evaluate_blocked_rolling(
            truth,
            reference,
            candidate,
            common,
            pd.DatetimeIndex(pd.to_datetime(issue_train.reindex(common))),
            periods["h2"] & (truth >= 0.10 * capacity),
            n_bootstrap=args.n_bootstrap,
            seed=20260727 + target_position,
        )
        gates = {
            "q1_all_components_positive": bool(
                spec["selection_eligible"]
            ),
            "q2_all_components_positive": bool(
                min(deltas["q2"].values()) > 0.0
            ),
            "h2_all_components_positive": bool(
                min(deltas["h2"].values()) > 0.0
            ),
            "positive_h2_score_months_at_least_five": bool(
                positive_months >= 5
            ),
            "issue_bootstrap_q05_positive": bool(
                bootstrap["issue_block_bootstrap"]["q05"] > 0.0
            ),
            "issue_bootstrap_positive_fraction_at_least_95pct": bool(
                bootstrap["issue_block_bootstrap"]["positive_fraction"]
                >= 0.95
            ),
        }
        promotion = bool(all(gates.values()))
        if promotion:
            promoted.append(target)
        movement = np.abs(candidate - reference)
        validation[target] = {
            "selected_spec": spec,
            "selection_records": selection_records,
            "period_deltas": deltas,
            "monthly_h2_deltas": monthly_h2,
            "positive_h2_score_months": int(positive_months),
            "issue_block_validation": bootstrap,
            "movement": {
                "changed_rows": int(np.sum(movement > 1e-6)),
                "mean_capacity_ratio": float(movement.mean() / capacity),
                "p95_capacity_ratio": float(
                    np.quantile(movement, 0.95) / capacity
                ),
                "maximum_capacity_ratio": float(
                    movement.max() / capacity
                ),
            },
            "gates": gates,
            "promotion": "promoted" if promotion else "rejected",
        }
        prefix = f"{target}__"
        cache[f"{prefix}index_ns"] = common.view("int64")
        cache[f"{prefix}truth"] = truth.astype("float32")
        cache[f"{prefix}reference"] = reference.astype("float32")
        cache[f"{prefix}candidate"] = candidate.astype("float32")

    candidate_record = None
    if promoted:
        production_bundle = make_sequence_bundle(
            feature_test,
            issue_test,
            None,
        )
        production_predictions: list[np.ndarray] = []
        production_training: list[dict[str, Any]] = []
        for seed in SEEDS:
            prediction, record = fit_predict(
                all_train,
                production_bundle,
                seed=seed,
                hidden=args.hidden,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            production_predictions.append(prediction)
            production_training.append(record)
        production_prediction = np.mean(production_predictions, axis=0)
        base = pd.read_csv(
            _rooted(args.active_candidate),
            encoding="utf-8-sig",
            parse_dates=["forecast_kst_dtm"],
        )
        output = base.copy()
        output_index = pd.DatetimeIndex(output["forecast_kst_dtm"])
        for target_position, target in enumerate(TARGETS):
            if target not in promoted:
                continue
            quantile_position = QUANTILES.index(
                selected_specs[target]["quantile"]
            )
            expert = flatten_prediction(
                production_prediction,
                production_bundle.timestamps,
                target_position,
                quantile_position,
                output_index,
            ) * CAPACITY_KWH[target]
            output[target] = apply_bounded_blend(
                output[target].to_numpy(dtype=float),
                expert,
                weight=selected_specs[target]["weight"],
                capacity=CAPACITY_KWH[target],
            )
        output["forecast_kst_dtm"] = output[
            "forecast_kst_dtm"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")
        output_path = _rooted(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        audit = CandidateValidator(load_config(ROOT)).audit(output_path)
        if not audit.valid:
            raise RuntimeError(
                f"CandidateValidator rejected TCN output: {audit.errors}"
            )
        candidate_record = {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "promoted_targets": promoted,
            "production_training": production_training,
            "candidate_validator": audit.to_dict(),
        }

    cache_path = _rooted(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    report = {
        "family": "issue_trajectory_multigroup_quantile_tcn",
        "method": (
            "24-hour issue-level temporal convolution; multi-group masked "
            "quantile loss; blend selection on Q1 and frozen Q2/H2 confirmation"
        ),
        "contract": {
            "validation_fit_years": [2022, 2023],
            "validation_query_year": 2024,
            "selection_period": "2024-Q1",
            "confirmation_periods": ["2024-Q2", "2024-H2"],
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
            "maximum_per_row_movement_ratio": 0.05,
        },
        "architecture": {
            "features": int(len(columns)),
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "quantiles": list(QUANTILES),
            "seeds": list(SEEDS),
        },
        "coverage": {
            "validation_train_issues": int(
                validation_train_rows.sum()
            ),
            "validation_query_issues": int(
                validation_query_rows.sum()
            ),
            "validation_rows": int(len(common)),
        },
        "validation_training": training_records,
        "raw_validation_cache": {
            "path": raw_validation_cache.relative_to(ROOT).as_posix(),
            "sha256": _sha256(raw_validation_cache),
        },
        "selected_specs": selected_specs,
        "validation": validation,
        "promoted_targets": promoted,
        "candidate": candidate_record,
        "cache": {
            "path": cache_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(cache_path),
        },
        "decision": (
            "promote passing targets"
            if promoted
            else "reject trajectory TCN; no target passed locked gates"
        ),
    }
    report_path = _rooted(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-train",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument(
        "--features-test",
        default="artifacts_final/feature_cache/features_test.pkl",
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--gfs-train", default="data/train/gfs_train.csv")
    parser.add_argument("--gfs-test", default="data/test/gfs_test.csv")
    parser.add_argument(
        "--primary-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--residual-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_msm_stencil_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--group3-cache",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--active-candidate",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/candidates/"
            "issue_trajectory_tcn_promoted_20260727.csv"
        ),
    )
    parser.add_argument(
        "--cache",
        default=(
            "artifacts_final/lineage/"
            "issue_trajectory_tcn_validation_20260727.npz"
        ),
    )
    parser.add_argument(
        "--raw-validation-cache",
        default=(
            "artifacts_final/lineage/"
            "issue_trajectory_tcn_raw_validation_20260727.npz"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "issue_trajectory_tcn_20260727.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "selected_specs": report["selected_specs"],
                "promoted_targets": report["promoted_targets"],
                "candidate": report["candidate"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
