from __future__ import annotations

"""Small, pre-declared FICR loss study for the spatial-temporal model.

This experiment deliberately reuses the audited train tensor and meta-gate
prediction caches.  Two loss configurations are fixed below before evaluation;
the variant/blend is selected on 2024 Q1/Q2 and H2 is evaluated exactly once.
"""

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.spatiotemporal_multitask import (
    CAPACITIES,
    TARGETS,
    DayDataset,
    SpatialTemporalMultiTask,
    calendar_tensor,
    feature_statistics,
    graph_adjacency,
    group_pooling_weights,
    metric_from_days,
    predict_loader,
    set_seed,
)
from src.metrics import CAPACITY_KWH, evaluate_group


TARGET = "kpx_group_3"
TARGET_POSITION = TARGETS.index(TARGET)
CAPACITY = CAPACITY_KWH[TARGET]
Q1_START = pd.Timestamp("2024-01-01 01:00:00")
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")


@dataclass(frozen=True)
class LossVariant:
    name: str
    reward_strength: float
    temperature: float


# Pre-declared before the locked run.  ``sharp`` concentrates gradients close
# to the exact 6%/8% settlement cliffs.  ``balanced`` keeps a slightly wider
# transition and a modestly stronger settlement term.
LOSS_VARIANTS = (
    LossVariant("sharp", reward_strength=0.020, temperature=0.004),
    LossVariant("balanced", reward_strength=0.040, temperature=0.006),
)
BLEND_ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def official_boundary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reward_strength: float,
    temperature: float,
) -> torch.Tensor:
    """Group-macro NMAE plus a smooth surrogate at official FICR cliffs."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    valid = torch.isfinite(target)
    eligible = valid & (target >= 0.10)
    safe_target = torch.nan_to_num(target, nan=0.0)
    error = torch.abs(prediction - safe_target)
    eligible_weight = eligible.float()

    group_count = eligible_weight.sum(dim=(0, 1))
    group_mae = (eligible_weight * error).sum(dim=(0, 1)) / group_count.clamp_min(1.0)
    soft_reward = 0.75 * torch.sigmoid((0.08 - error) / temperature)
    soft_reward += 0.25 * torch.sigmoid((0.06 - error) / temperature)
    generation_weight = eligible_weight * safe_target
    group_reward = (generation_weight * soft_reward).sum(dim=(0, 1))
    group_reward /= generation_weight.sum(dim=(0, 1)).clamp_min(1.0)

    available = group_count > 0
    return group_mae[available].mean() - reward_strength * group_reward[available].mean()


def _make_model(arrays: np.lib.npyio.NpzFile, train_days: np.ndarray, hidden: int) -> nn.Module:
    ldaps_mean, ldaps_std = feature_statistics(arrays["train_ldaps"], train_days)
    gfs_mean, gfs_std = feature_statistics(arrays["train_gfs"], train_days)
    return SpatialTemporalMultiTask(
        ldaps_shape=arrays["train_ldaps"].shape[2:4],
        gfs_shape=arrays["train_gfs"].shape[2:4],
        hidden=hidden,
        ldaps_adjacency=graph_adjacency(arrays["ldaps_coordinates"]),
        gfs_adjacency=graph_adjacency(arrays["gfs_coordinates"]),
        ldaps_pooling=group_pooling_weights(arrays["ldaps_coordinates"]),
        gfs_pooling=group_pooling_weights(arrays["gfs_coordinates"]),
        ldaps_mean=ldaps_mean,
        ldaps_std=ldaps_std,
        gfs_mean=gfs_mean,
        gfs_std=gfs_std,
    )


def train_variant(
    arrays: np.lib.npyio.NpzFile,
    train_days: np.ndarray,
    selection_days: np.ndarray,
    validation_days: np.ndarray,
    variant: LossVariant,
    seed: int,
    hidden: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    set_seed(seed)
    model = _make_model(arrays, train_days, hidden)
    calendar = calendar_tensor(arrays["train_timestamps_ns"])
    train_loader = DataLoader(
        DayDataset(
            arrays["train_ldaps"], arrays["train_gfs"], calendar,
            arrays["train_targets"], train_days,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    selection_loader = DataLoader(
        DayDataset(
            arrays["train_ldaps"], arrays["train_gfs"], calendar,
            arrays["train_targets"], selection_days,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = -np.inf
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for ldaps, gfs, day_calendar, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(ldaps, gfs, day_calendar)
            loss = official_boundary_loss(
                prediction, target, variant.reward_strength, variant.temperature
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        selection_prediction = predict_loader(model, selection_loader)
        selection_score = float(
            metric_from_days(
                arrays["train_targets"][selection_days], selection_prediction
            )["score"]
        )
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "selection_score": selection_score}
        )
        if selection_score > best_score + 1e-5:
            best_score = selection_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(variant.name, seed, history[-1], flush=True)
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    validation_loader = DataLoader(
        DayDataset(
            arrays["train_ldaps"], arrays["train_gfs"], calendar, None, validation_days
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    prediction = predict_loader(model, validation_loader)
    return prediction, {
        "variant": variant.name,
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_best_score": best_score,
        "history": history,
    }


def metric_delta(
    truth: np.ndarray, base: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    before = evaluate_group(truth[mask], base[mask], CAPACITY)
    after = evaluate_group(truth[mask], candidate[mask], CAPACITY)
    return {
        "score": float(after.score - before.score),
        "one_minus_nmae": float(after.one_minus_nmae - before.one_minus_nmae),
        "ficr": float(after.ficr - before.ficr),
    }


def day_bootstrap_delta(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    timestamps: pd.DatetimeIndex,
    mask: np.ndarray,
    n_bootstrap: int = 2_000,
    seed: int = 20_260_717,
) -> dict[str, float]:
    """Paired day-block bootstrap of the official group score delta."""
    eligible_positions = np.flatnonzero(mask)
    eligible_days = timestamps[eligible_positions].normalize()
    unique_days = pd.DatetimeIndex(eligible_days.unique())
    positions_by_day = [eligible_positions[eligible_days == day] for day in unique_days]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled = rng.integers(0, len(positions_by_day), size=len(positions_by_day))
        positions = np.concatenate([positions_by_day[index] for index in sampled])
        before = evaluate_group(truth[positions], base[positions], CAPACITY)
        after = evaluate_group(truth[positions], candidate[positions], CAPACITY)
        deltas[iteration] = after.score - before.score
    return {
        "n_bootstrap": int(n_bootstrap),
        "positive_fraction": float(np.mean(deltas > 0.0)),
        "q05": float(np.quantile(deltas, 0.05)),
        "median": float(np.median(deltas)),
        "q95": float(np.quantile(deltas, 0.95)),
    }


def choose_development_policy(records: list[dict[str, object]]) -> dict[str, object]:
    """Select on Q1/Q2 only, requiring both components and seeds to transfer."""
    eligible = []
    for record in records:
        q1 = record["q1"]
        q2 = record["q2"]
        seed_deltas = record["seed_score_deltas"]
        if (
            q1["one_minus_nmae"] > 0.0
            and q1["ficr"] > 0.0
            and q2["one_minus_nmae"] > 0.0
            and q2["ficr"] > 0.0
            and min(seed_deltas["q1"]) > 0.0
            and min(seed_deltas["q2"]) > 0.0
        ):
            eligible.append(record)
    if not eligible:
        raise RuntimeError("No loss/blend policy passed the Q1/Q2 development contract")
    return max(
        eligible,
        key=lambda record: (
            min(record["q1"]["score"], record["q2"]["score"]),
            min(record["seed_score_deltas"]["q1"] + record["seed_score_deltas"]["q2"]),
            -float(record["alpha"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor-cache", default="artifacts_final/spatiotemporal/spatiotemporal_train_tensors.npz"
    )
    parser.add_argument("--base-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz")
    parser.add_argument("--artifact-dir", default="artifacts_final/spatiotemporal_ficr_loss")
    parser.add_argument("--seeds", default="17,29")
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(Path(args.tensor_cache), allow_pickle=True)
    day_start = pd.to_datetime(arrays["train_timestamps_ns"][:, 0])
    day_end = pd.to_datetime(arrays["train_timestamps_ns"][:, -1])
    train_days = np.flatnonzero(day_end < pd.Timestamp("2024-01-01"))
    selection_days = np.flatnonzero(
        (day_start >= pd.Timestamp("2024-01-01"))
        & (day_start < pd.Timestamp("2024-07-01"))
    )
    validation_days = np.flatnonzero(
        (day_start >= pd.Timestamp("2024-01-01")) & (day_start < END)
    )
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(arrays["train_timestamps_ns"][validation_days].reshape(-1))
    )
    truth = (
        arrays["train_targets"][validation_days, :, TARGET_POSITION].reshape(-1)
        * CAPACITIES[TARGET_POSITION]
    )
    base_cache = np.load(Path(args.base_cache), allow_pickle=True)
    base_index = pd.DatetimeIndex(pd.to_datetime(base_cache["valid_index_ns"]))
    base_values = base_cache["valid_candidate"].astype(float)
    positions = base_index.get_indexer(timestamps)
    # Six 2024 timestamps have no exact OOF group-3 prediction because their
    # labels are unavailable.  Keep the physical day tensor intact for model
    # inference, but exclude those rows from paired policy comparisons.
    base = np.full(len(timestamps), np.nan, dtype=float)
    covered = positions >= 0
    base[covered] = base_values[positions[covered]]
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())

    predictions: dict[str, list[np.ndarray]] = {}
    train_reports = []
    for variant in LOSS_VARIANTS:
        predictions[variant.name] = []
        prediction_path = output / f"{variant.name}_validation_predictions.npz"
        if args.reuse_predictions:
            retained = np.load(prediction_path, allow_pickle=False)
            if not np.array_equal(retained["timestamps_ns"], timestamps.astype("int64")):
                raise ValueError(f"Stale timestamps in {prediction_path}")
            if not np.array_equal(retained["seeds"], np.asarray(seeds, dtype=np.int64)):
                raise ValueError(f"Stale seeds in {prediction_path}")
            predictions[variant.name] = [item.astype(float) for item in retained["predictions"]]
        else:
            for seed in seeds:
                prediction, train_report = train_variant(
                    arrays, train_days, selection_days, validation_days, variant, seed,
                    args.hidden, args.epochs, args.patience, args.batch_size,
                )
                group3 = prediction[:, :, TARGET_POSITION].reshape(-1) * CAPACITY
                predictions[variant.name].append(group3)
                train_reports.append(train_report)
            np.savez_compressed(
                prediction_path,
                timestamps_ns=timestamps.astype("int64").to_numpy(),
                truth=truth.astype(np.float32),
                predictions=np.stack(predictions[variant.name]).astype(np.float32),
                seeds=np.asarray(seeds, dtype=np.int64),
            )

    paired = np.isfinite(truth) & np.isfinite(base)
    q1 = (timestamps >= Q1_START) & (timestamps < Q2_START) & paired
    q2 = (timestamps >= Q2_START) & (timestamps < H2_START) & paired
    h2 = (timestamps >= H2_START) & (timestamps < END) & paired
    development_records = []
    for variant in LOSS_VARIANTS:
        members = predictions[variant.name]
        ensemble = np.mean(members, axis=0)
        for alpha in BLEND_ALPHAS:
            candidate = np.clip((1.0 - alpha) * base + alpha * ensemble, 0.0, CAPACITY)
            seed_candidates = [
                np.clip((1.0 - alpha) * base + alpha * member, 0.0, CAPACITY)
                for member in members
            ]
            development_records.append(
                {
                    "variant": variant.name,
                    "alpha": alpha,
                    "q1": metric_delta(truth, base, candidate, q1),
                    "q2": metric_delta(truth, base, candidate, q2),
                    "seed_score_deltas": {
                        "q1": [metric_delta(truth, base, item, q1)["score"] for item in seed_candidates],
                        "q2": [metric_delta(truth, base, item, q2)["score"] for item in seed_candidates],
                    },
                }
            )
    selected = choose_development_policy(development_records)

    # H2 is opened once, only for the Q1/Q2-selected policy.
    selected_members = predictions[str(selected["variant"])]
    selected_ensemble = np.mean(selected_members, axis=0)
    selected_candidate = np.clip(
        (1.0 - float(selected["alpha"])) * base
        + float(selected["alpha"]) * selected_ensemble,
        0.0,
        CAPACITY,
    )
    locked = metric_delta(truth, base, selected_candidate, h2)
    locked_seed_deltas = [
        metric_delta(
            truth,
            base,
            np.clip(
                (1.0 - float(selected["alpha"])) * base
                + float(selected["alpha"]) * member,
                0.0,
                CAPACITY,
            ),
            h2,
        )["score"]
        for member in selected_members
    ]
    months = {}
    for month in range(7, 13):
        mask = h2 & (timestamps.month == month)
        months[str(month)] = metric_delta(truth, base, selected_candidate, mask)
    bootstrap = day_bootstrap_delta(truth, base, selected_candidate, timestamps, h2)

    report = {
        "method": "pre-declared official-boundary/group-macro spatial loss tune",
        "lineage": {
            "tensor_cache": str(Path(args.tensor_cache)),
            "base_cache": str(Path(args.base_cache)),
        },
        "variants": [variant.__dict__ for variant in LOSS_VARIANTS],
        "seeds": list(seeds),
        "development_contract": {
            "selection": "maximize worst Q1/Q2 score after both components and every seed are positive",
            "locked": "evaluate the selected variant/alpha once on 2024-H2",
        },
        "training": train_reports,
        "development_records": development_records,
        "selected": selected,
        "locked_h2": {
            "delta": locked,
            "seed_score_deltas": locked_seed_deltas,
            "monthly_deltas": months,
            "day_bootstrap": bootstrap,
            "passes_basic_gate": bool(
                locked["score"] >= 0.00015
                and locked["one_minus_nmae"] >= 0.0
                and locked["ficr"] >= 0.0
                and min(locked_seed_deltas) > 0.0
                and sum(value["score"] > 0.0 for value in months.values()) >= 4
            ),
        },
    }
    (output / "ficr_loss_tune_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "locked_h2": report["locked_h2"]}, indent=2))


if __name__ == "__main__":
    main()
