"""Dual-year non-crossing probabilistic core screen for groups 1 and 2.

The existing residual-distribution experiments fit each conditional quantile
independently.  This experiment instead learns all quantiles jointly with a
simplex head, so forecasts are bounded and cannot cross.  A settlement-aware
Bayes action is derived from the resulting conditional distribution.

Validation is deliberately fail-closed:

* fit on calendar 2022 and select one correction weight on calendar 2023;
* refit the same architecture on calendar 2023 and confirm the same
  median-to-action comparison on 2024;
* separately require the completed distributional forecast to beat the exact
  active 2024 OOF surface as a full replacement;
* require full/H2/month and exact 40:60 complementary-subset stability;
* never write a submission from this diagnostic.

An action delta selected around the distributional median is never overlaid on
an unrelated incumbent.  FiCR is threshold-sensitive, so that asymmetric
comparison is not a valid forward-validation contract.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.global_capacity_model import make_group_frame
from experiments.group12_difference_reconciliation import GROUPS, pair_delta
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from experiments.public_private_subset_stress import (
    complementary_subset_stress,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
QUANTILE_LEVELS = np.arange(0.05, 1.0, 0.10, dtype=np.float32)
WEIGHTS = (0.025, 0.05, 0.10, 0.20, 0.30, 0.50)
COMPONENTS = ("score", "one_minus_nmae", "ficr")


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.nanmean(values, axis=0).astype(np.float32)
        scale = np.nanstd(values, axis=0).astype(np.float32)
        scale = np.where(scale >= 1e-5, scale, 1.0).astype(np.float32)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        return np.nan_to_num(output, nan=0.0, posinf=8.0, neginf=-8.0).clip(
            -8.0, 8.0
        )


class NonCrossingQuantileMLP(nn.Module):
    """Bounded quantiles obtained as a cumulative simplex."""

    def __init__(
        self,
        n_features: int,
        *,
        hidden: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Q+1 non-negative gaps partition [0, 1].  The first Q cumulative
        # values are strictly ordered bounded quantiles.
        self.gaps = nn.Linear(hidden // 2, len(QUANTILE_LEVELS) + 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gaps = torch.softmax(self.gaps(self.backbone(values)), dim=-1)
        return torch.cumsum(gaps, dim=-1)[..., :-1]


def pinball_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    levels = torch.as_tensor(
        QUANTILE_LEVELS,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    error = truth[:, None] - prediction
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()


def _loader(
    features: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(features, dtype=np.float32)),
        torch.from_numpy(np.asarray(target, dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def _train_epochs(
    model: NonCrossingQuantileMLP,
    features: np.ndarray,
    target: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    model.train()
    for _ in range(epochs):
        for batch_features, batch_target in _loader(
            features,
            target,
            batch_size=batch_size,
            shuffle=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(batch_features), batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()


def _validation_loss(
    model: NonCrossingQuantileMLP,
    features: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    rows = 0
    with torch.no_grad():
        for batch_features, batch_target in _loader(
            features,
            target,
            batch_size=batch_size,
            shuffle=False,
        ):
            loss = pinball_loss(model(batch_features), batch_target)
            total += float(loss) * len(batch_target)
            rows += len(batch_target)
    return total / rows


def fit_one_seed(
    train_features: np.ndarray,
    train_target: np.ndarray,
    inner_train: np.ndarray,
    inner_valid: np.ndarray,
    query_features: np.ndarray,
    *,
    seed: int,
    hidden: int,
    dropout: float,
    maximum_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select epochs inside the training year, then refit on its full data."""
    _seed_everything(seed)
    standardizer = Standardizer.fit(train_features[inner_train])
    inner_x = standardizer.transform(train_features)
    model = NonCrossingQuantileMLP(
        train_features.shape[1],
        hidden=hidden,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    train_loader = _loader(
        inner_x[inner_train],
        train_target[inner_train],
        batch_size=batch_size,
        shuffle=True,
    )
    best_epoch = 1
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(batch_features), batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        valid_loss = _validation_loss(
            model,
            inner_x[inner_valid],
            train_target[inner_valid],
            batch_size=batch_size,
        )
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("non-crossing model did not produce an inner state")

    # Refit without carrying inner-validation weights into the full-year fit.
    _seed_everything(seed + 100_000)
    full_standardizer = Standardizer.fit(train_features)
    full_x = full_standardizer.transform(train_features)
    final_model = NonCrossingQuantileMLP(
        train_features.shape[1],
        hidden=hidden,
        dropout=dropout,
    )
    _train_epochs(
        final_model,
        full_x,
        train_target,
        epochs=best_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    query_x = full_standardizer.transform(query_features)
    final_model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(query_x), batch_size):
            values = torch.from_numpy(query_x[start : start + batch_size])
            predictions.append(final_model(values).numpy())
    quantiles = np.vstack(predictions).astype(np.float32)
    return quantiles, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_inner_pinball": best_loss,
        "inner_train_rows": int(inner_train.sum()),
        "inner_valid_rows": int(inner_valid.sum()),
        "full_train_rows": int(len(train_features)),
    }


def _calendar_rows(index: pd.DatetimeIndex, year: int) -> np.ndarray:
    return np.asarray(index.year == year)


def stack_training_year(
    frames: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    months: list[np.ndarray] = []
    for target in GROUPS:
        rows = _calendar_rows(frames[target].index, year)
        raw_target = labels[target].reindex(frames[target].index).to_numpy(
            dtype=float
        )
        rows &= np.isfinite(raw_target)
        features.append(frames[target].loc[rows].to_numpy(dtype=np.float32))
        targets.append(
            (raw_target[rows] / CAPACITY_KWH[target]).astype(np.float32)
        )
        months.append(frames[target].index[rows].month.to_numpy())
    return (
        np.vstack(features),
        np.concatenate(targets),
        np.concatenate(months),
    )


def stack_query(
    frames: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray, dict[str, slice]]:
    blocks: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    start = 0
    for target in GROUPS:
        block = frames[target].reindex(index).to_numpy(dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"query features are incomplete for {target}")
        blocks.append(block)
        slices[target] = slice(start, start + len(block))
        start += len(block)
    return np.vstack(blocks), slices


def fit_forward_distribution(
    frames: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    *,
    train_year: int,
    query_index: pd.DatetimeIndex,
    seeds: tuple[int, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    train_features, train_target, months = stack_training_year(
        frames,
        labels,
        train_year,
    )
    inner_train = months <= 9
    inner_valid = months >= 10
    query_features, query_slices = stack_query(frames, query_index)
    members: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        prediction, record = fit_one_seed(
            train_features,
            train_target,
            inner_train,
            inner_valid,
            query_features,
            seed=seed + train_year,
            hidden=args.hidden,
            dropout=args.dropout,
            maximum_epochs=args.maximum_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        members.append(prediction)
        records.append(record)
        print(
            f"train={train_year} seed={seed} epoch={record['best_epoch']} "
            f"inner_pinball={record['best_inner_pinball']:.6f}",
            flush=True,
        )
    ensemble = np.mean(members, axis=0)
    if np.any(np.diff(ensemble, axis=1) < -1e-7):
        raise AssertionError("ensemble quantiles crossed")
    return {
        target: ensemble[query_slices[target]]
        for target in GROUPS
    }, records


def distribution_point_forecasts(
    quantiles: np.ndarray,
    capacity: float,
    mean_generation: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return conditional median and official-utility Bayes action."""
    values = np.asarray(quantiles, dtype=float) * capacity
    median = 0.5 * (values[:, 4] + values[:, 5])
    offsets = np.arange(-0.04, 0.0401, 0.01) * capacity
    # The predictive distribution may be wide, but the point-action search is
    # intentionally bounded.  Treating tail quantiles themselves as actions
    # allowed 15%+ production movement in the smoke run and confounded model
    # quality with an unsafe policy.
    candidates = np.clip(
        median[:, None] + offsets[None, :],
        0.0,
        capacity,
    )
    scenarios = values
    eligible = scenarios >= 0.10 * capacity
    error = (
        np.abs(candidates[:, :, None] - scenarios[:, None, :]) / capacity
    )
    units = np.where(error <= 0.06, 1.0, np.where(error <= 0.08, 0.75, 0.0))
    generation_weight = (
        scenarios[:, None, :] / max(float(mean_generation), 1.0)
    )
    utility = np.where(
        eligible[:, None, :],
        -0.5 * error + 0.5 * generation_weight * units,
        0.0,
    ).mean(axis=2)
    best = closest_utility_maximizer(utility, candidates, median)
    action = candidates[np.arange(len(candidates)), best]
    return median, action, {
        "mean_absolute_action_from_median_ratio": float(
            np.mean(np.abs(action - median)) / capacity
        ),
        "p95_action_from_median_ratio": float(
            np.quantile(np.abs(action - median), 0.95) / capacity
        ),
        "maximum_action_from_median_ratio": float(
            np.max(np.abs(action - median)) / capacity
        ),
        "nonzero_action_fraction": float(np.mean(np.abs(action - median) > 1e-8)),
    }


def closest_utility_maximizer(
    utility: np.ndarray,
    candidates: np.ndarray,
    reference: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Choose the action nearest the reference on a utility plateau."""
    utility = np.asarray(utility, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if utility.shape != candidates.shape or utility.ndim != 2:
        raise ValueError("utility and candidates must be aligned 2-D arrays")
    if reference.shape != (len(utility),):
        raise ValueError("reference must contain one value per utility row")
    best_utility = np.max(utility, axis=1, keepdims=True)
    admissible = utility >= best_utility - float(tolerance)
    distance = np.abs(candidates - reference[:, None])
    return np.argmin(np.where(admissible, distance, np.inf), axis=1)


def _truth_for_index(
    labels: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    return {
        target: labels[target].reindex(index).to_numpy(dtype=float)
        for target in GROUPS
    }


def _point_surfaces(
    quantiles: dict[str, np.ndarray],
    labels: pd.DataFrame,
    training_year: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    median: dict[str, np.ndarray] = {}
    action: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for target in GROUPS:
        capacity = CAPACITY_KWH[target]
        training_truth = labels.loc[
            labels.index.year == training_year, target
        ].to_numpy(dtype=float)
        mean_generation = float(
            np.nanmean(training_truth[training_truth >= 0.10 * capacity])
        )
        median[target], action[target], diagnostics[target] = (
            distribution_point_forecasts(
                quantiles[target],
                capacity,
                mean_generation,
            )
        )
    return median, action, diagnostics


def apply_action_delta(
    base: dict[str, np.ndarray],
    median: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    weight: float,
) -> dict[str, np.ndarray]:
    return {
        target: np.clip(
            np.asarray(base[target], dtype=float)
            + weight
            * (
                np.asarray(action[target], dtype=float)
                - np.asarray(median[target], dtype=float)
            ),
            0.0,
            CAPACITY_KWH[target],
        )
        for target in GROUPS
    }


def interpolate_distribution_action(
    median: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    weight: float,
) -> dict[str, np.ndarray]:
    """Interpolate inside one distributional forecast family.

    Keeping the median as the anchor is essential: a delta whose utility was
    selected around this median cannot be assumed to retain its utility around
    an unrelated incumbent forecast.
    """
    return apply_action_delta(median, median, action, weight)


def _subset_pass(result: dict[str, Any]) -> bool:
    return bool(
        all(
            result[split][component]["q05"] >= 0.0
            for split in ("public", "private")
            for component in COMPONENTS
        )
    )


def _weight_record(
    truth: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    median: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    weight: float,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    candidate = apply_action_delta(base, median, action, weight)
    full = pair_delta(
        truth,
        base,
        candidate,
        np.ones(len(index), dtype=bool),
    )
    iid = complementary_subset_stress(
        truth,
        base,
        candidate,
        index,
        public_fraction=0.40,
        repetitions=repetitions,
        seed=seed,
        stratify_month=False,
    )
    stratified = complementary_subset_stress(
        truth,
        base,
        candidate,
        index,
        public_fraction=0.40,
        repetitions=repetitions,
        seed=seed + 1,
        stratify_month=True,
    )
    return {
        "weight": weight,
        "full_delta": full,
        "iid": iid,
        "month_stratified": stratified,
        "eligible": bool(
            all(full[component] > 0.0 for component in COMPONENTS)
            and _subset_pass(iid)
            and _subset_pass(stratified)
        ),
    }


def _confirm_candidate(
    truth: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    *,
    repetitions: int,
) -> dict[str, Any]:
    full = pair_delta(
        truth,
        base,
        candidate,
        np.ones(len(index), dtype=bool),
    )
    h2_rows = index >= pd.Timestamp("2024-07-01")
    h2 = pair_delta(truth, base, candidate, h2_rows)
    monthly = {
        str(month): pair_delta(
            truth,
            base,
            candidate,
            np.asarray(index.month == month),
        )
        for month in sorted(set(index.month))
    }
    iid = complementary_subset_stress(
        truth,
        base,
        candidate,
        index,
        public_fraction=0.40,
        repetitions=repetitions,
        seed=20260731,
        stratify_month=False,
    )
    stratified = complementary_subset_stress(
        truth,
        base,
        candidate,
        index,
        public_fraction=0.40,
        repetitions=repetitions,
        seed=20260801,
        stratify_month=True,
    )
    movement = np.concatenate(
        [
            np.abs(candidate[target] - base[target])
            / CAPACITY_KWH[target]
            for target in GROUPS
        ]
    )
    gates = {
        "full_all_components_positive": bool(
            all(full[component] > 0.0 for component in COMPONENTS)
        ),
        "h2_all_components_nonnegative": bool(
            all(h2[component] >= 0.0 for component in COMPONENTS)
        ),
        "every_month_score_nonnegative": bool(
            all(row["score"] >= 0.0 for row in monthly.values())
        ),
        "iid_40_60_component_q05_nonnegative": _subset_pass(iid),
        "stratified_40_60_component_q05_nonnegative": _subset_pass(
            stratified
        ),
        "p95_movement_at_most_4pct": bool(
            np.quantile(movement, 0.95) <= 0.04
        ),
    }
    return {
        "full_delta": full,
        "h2_delta": h2,
        "monthly_deltas": monthly,
        "iid_subset_stress": iid,
        "month_stratified_subset_stress": stratified,
        "movement": {
            "changed_fraction": float(np.mean(movement > 1e-10)),
            "mean_ratio": float(np.mean(movement)),
            "p95_ratio": float(np.quantile(movement, 0.95)),
            "maximum_ratio": float(np.max(movement)),
        },
        "gates": gates,
        "qualified": bool(all(gates.values())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.torch_threads)
    train_features = pd.read_pickle(_rooted(args.feature_train))
    labels = pd.read_csv(_rooted(args.labels), encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").sort_index()
    frames = {
        target: make_group_frame(train_features, target)
        for target in GROUPS
    }
    columns = list(frames[GROUPS[0]].columns)
    if any(list(frames[target].columns) != columns for target in GROUPS):
        raise ValueError("pooled group frames do not share an exact schema")
    seeds = tuple(
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    )

    selection_index = pd.DatetimeIndex(
        train_features.index[train_features.index.year == 2023]
    )
    quantiles_2023, fit_2022 = fit_forward_distribution(
        frames,
        labels,
        train_year=2022,
        query_index=selection_index,
        seeds=seeds,
        args=args,
    )
    median_2023, action_2023, action_diag_2023 = _point_surfaces(
        quantiles_2023,
        labels,
        2022,
    )
    truth_2023 = _truth_for_index(labels, selection_index)
    selection_records = [
        _weight_record(
            truth_2023,
            median_2023,
            median_2023,
            action_2023,
            selection_index,
            weight,
            repetitions=args.selection_repetitions,
            seed=20260729 + position * 10,
        )
        for position, weight in enumerate(WEIGHTS)
    ]
    eligible = [record for record in selection_records if record["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda record: (
                record["full_delta"]["score"],
                record["full_delta"]["ficr"],
            ),
        )
        if eligible
        else None
    )

    confirmation: dict[str, Any] | None = None
    fit_2023: list[dict[str, Any]] = []
    action_diag_2024: dict[str, Any] | None = None
    if selected is not None:
        baselines, truth_series, active_index, _issues = (
            load_frozen_validation_baselines(
                _rooted(args.primary_cache),
                _rooted(args.residual_cache),
                _rooted(args.group3_cache),
            )
        )
        quantiles_2024, fit_2023 = fit_forward_distribution(
            frames,
            labels,
            train_year=2023,
            query_index=active_index,
            seeds=seeds,
            args=args,
        )
        median_2024, action_2024, action_diag_2024 = _point_surfaces(
            quantiles_2024,
            labels,
            2023,
        )
        active_base = {
            target: baselines[target].to_numpy(dtype=float)
            for target in GROUPS
        }
        active_truth = {
            target: truth_series[target].to_numpy(dtype=float)
            for target in GROUPS
        }
        distribution_candidate = interpolate_distribution_action(
            median_2024,
            action_2024,
            float(selected["weight"]),
        )
        family_internal = _confirm_candidate(
            active_truth,
            median_2024,
            distribution_candidate,
            active_index,
            repetitions=args.confirmation_repetitions,
        )
        incumbent_replacement = _confirm_candidate(
            active_truth,
            active_base,
            distribution_candidate,
            active_index,
            repetitions=args.confirmation_repetitions,
        )
        confirmation = {
            "baseline_contract": {
                "selection_base": "distributional_core_median_2023",
                "internal_confirmation_base": (
                    "distributional_core_median_2024"
                ),
                "deployment_comparison": (
                    "complete distributional candidate versus exact active "
                    "2024 incumbent OOF"
                ),
                "incumbent_delta_overlay_allowed": False,
                "reason": (
                    "FiCR utility depends on the absolute baseline position "
                    "relative to the 6% and 8% error cliffs"
                ),
            },
            "family_internal": family_internal,
            "incumbent_replacement": incumbent_replacement,
            "qualified": bool(
                family_internal["qualified"]
                and incumbent_replacement["qualified"]
            ),
        }

    report = {
        "schema_version": "v2_baseline_symmetric",
        "family": "dual_year_noncrossing_distributional_core",
        "method": (
            "pooled capacity-normalized non-crossing quantile MLP; "
            "conditional official-utility Bayes action"
        ),
        "references": {
            "continuous_conditional_distribution": (
                "Wen et al. 2022, IEEE TSTE, DOI 10.1109/TSTE.2022.3191330"
            ),
            "decision_scenarios": (
                "Cramer et al. 2022, arXiv:2204.02242"
            ),
            "regime_calibration": (
                "Gneiting et al. 2006, JASA, DOI 10.1198/016214506000000456"
            ),
        },
        "configuration": {
            "targets": GROUPS,
            "feature_count": len(columns),
            "quantile_levels": QUANTILE_LEVELS.tolist(),
            "weights": WEIGHTS,
            "seeds": seeds,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "maximum_epochs": args.maximum_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "validation_contract": {
            "selection": "calendar 2022 fit -> complete calendar 2023",
            "confirmation": (
                "calendar 2023 fit -> complete calendar 2024; first compare "
                "within the distributional family, then compare the complete "
                "forecast against the exact active 2024 OOF surface"
            ),
            "baseline_symmetry_required": True,
            "incumbent_delta_overlay_allowed": False,
            "epoch_selection": (
                "Jan-Sep train -> Oct-Dec validation inside each fit year"
            ),
            "public_score_used": False,
            "test_labels_used": False,
            "submission_side_effect": False,
        },
        "fit_2022": fit_2022,
        "selection_2023": {
            "action_diagnostics": action_diag_2023,
            "records": selection_records,
            "selected": selected,
        },
        "fit_2023": fit_2023,
        "confirmation_2024": confirmation,
        "action_diagnostics_2024": action_diag_2024,
        "promotion": {
            "selection_eligible": selected is not None,
            "confirmation_opened": confirmation is not None,
            "qualified": bool(
                confirmation is not None
                and confirmation["qualified"]
            ),
            "candidate_written": False,
            "decision": (
                "research_pass_no_candidate_writer"
                if confirmation is not None and confirmation["qualified"]
                else "rejected_fail_closed"
            ),
        },
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
        "--feature-train",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
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
    parser.add_argument("--seeds", default="17,41")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--maximum-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--selection-repetitions", type=int, default=1_000)
    parser.add_argument(
        "--confirmation-repetitions",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "noncrossing_distributional_core_symmetric_20260729.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "selected": report["selection_2023"]["selected"],
                "confirmation_2024": report["confirmation_2024"],
                "promotion": report["promotion"],
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
