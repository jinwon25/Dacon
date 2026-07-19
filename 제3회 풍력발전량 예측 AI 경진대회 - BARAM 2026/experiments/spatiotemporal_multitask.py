from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.features import META_COLS, TIME_COL, TURBINES_BY_GROUP, _add_wind_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group


TARGETS = list(CAPACITY_KWH)
CAPACITIES = np.asarray([CAPACITY_KWH[target] for target in TARGETS], dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _source_tensor(path: Path, source: str) -> tuple[np.ndarray, pd.DatetimeIndex, list[str], np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL])
    frame = _add_wind_features(frame, source)
    value_columns = [column for column in frame.columns if column not in META_COLS]
    frame[value_columns] = frame[value_columns].astype("float32")

    timestamps = pd.DatetimeIndex(sorted(frame[TIME_COL].unique()))
    nodes = np.asarray(sorted(frame["grid_id"].unique()), dtype=int)
    coordinates = (
        frame[["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id")
        .set_index("grid_id")
        .reindex(nodes)[["latitude", "longitude"]]
        .to_numpy(dtype=np.float32)
    )
    ordered = frame.set_index([TIME_COL, "grid_id"]).reindex(
        pd.MultiIndex.from_product([timestamps, nodes], names=[TIME_COL, "grid_id"])
    )
    values = ordered[value_columns].to_numpy(dtype=np.float32)
    tensor = values.reshape(len(timestamps), len(nodes), len(value_columns))
    return tensor, timestamps, value_columns, coordinates


def build_split_tensor_cache(
    data_dir: Path,
    cache_dir: Path,
    split: str,
    rebuild: bool = False,
) -> Path:
    """Build one physical cache without accessing the other data split."""
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    output = cache_dir / f"spatiotemporal_{split}_tensors.npz"
    if output.exists() and not rebuild:
        return output
    cache_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {}
    split_timestamps: pd.DatetimeIndex | None = None
    for source in ("ldaps", "gfs"):
        tensor, timestamps, columns, coordinates = _source_tensor(
            data_dir / split / f"{source}_{split}.csv", source
        )
        if split_timestamps is None:
            split_timestamps = timestamps
        elif not split_timestamps.equals(timestamps):
            raise ValueError(f"{split} timestamps differ between NWP sources")
        if len(timestamps) % 24 != 0:
            raise ValueError(f"{split} timestamps do not form complete 24-hour cycles")
        arrays[f"{split}_{source}"] = tensor.reshape(
            len(timestamps) // 24, 24, tensor.shape[1], tensor.shape[2]
        )
        arrays[f"{source}_coordinates"] = coordinates
        metadata[f"{source}_columns"] = columns
    assert split_timestamps is not None
    arrays[f"{split}_timestamps_ns"] = split_timestamps.astype("int64").to_numpy().reshape(-1, 24)

    if split == "train":
        labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
        labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
        if labels["kst_dtm"].duplicated().any():
            raise ValueError("Training labels contain duplicate timestamps")
        labels = labels.set_index("kst_dtm")
        train_index = pd.to_datetime(arrays["train_timestamps_ns"].reshape(-1))
        if not train_index.difference(labels.index).empty or not labels.index.difference(
            train_index
        ).empty:
            raise ValueError("Training label and NWP timestamps do not match exactly")
        target_values = labels.reindex(train_index)[TARGETS].to_numpy(dtype=np.float32) / CAPACITIES
        arrays["train_targets"] = target_values.reshape(-1, 24, len(TARGETS))
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    return output


def _distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]
    mean_latitude = np.deg2rad((latitude[:, None] + latitude[None, :]) / 2.0)
    dy = (latitude[:, None] - latitude[None, :]) * 111.32
    dx = (longitude[:, None] - longitude[None, :]) * 111.32 * np.cos(mean_latitude)
    return np.sqrt(dx * dx + dy * dy)


def graph_adjacency(coordinates: np.ndarray) -> np.ndarray:
    distance = _distance_matrix(coordinates)
    positive = distance[distance > 0]
    scale = float(np.median(np.sort(distance, axis=1)[:, 1])) if len(positive) else 1.0
    adjacency = np.exp(-(distance**2) / (2.0 * max(scale, 1e-3) ** 2))
    nearest = np.argsort(distance, axis=1)[:, :5]
    mask = np.zeros_like(adjacency, dtype=bool)
    np.put_along_axis(mask, nearest, True, axis=1)
    adjacency = np.where(mask | mask.T, adjacency, 0.0)
    adjacency += np.eye(len(coordinates), dtype=float)
    return (adjacency / adjacency.sum(axis=1, keepdims=True)).astype(np.float32)


def group_pooling_weights(coordinates: np.ndarray) -> np.ndarray:
    weights = []
    for target in TARGETS:
        raw = np.zeros(len(coordinates), dtype=float)
        for turbine_latitude, turbine_longitude in TURBINES_BY_GROUP[target]:
            mean_latitude = np.deg2rad((coordinates[:, 0] + turbine_latitude) / 2.0)
            dy = (coordinates[:, 0] - turbine_latitude) * 111.32
            dx = (coordinates[:, 1] - turbine_longitude) * 111.32 * np.cos(mean_latitude)
            distance = np.sqrt(dx * dx + dy * dy)
            raw += 1.0 / (distance + 0.20) ** 2
        weights.append(raw / raw.sum())
    return np.asarray(weights, dtype=np.float32)


class DayDataset(Dataset):
    def __init__(
        self,
        ldaps: np.ndarray,
        gfs: np.ndarray,
        calendar: np.ndarray,
        targets: np.ndarray | None,
        indices: np.ndarray,
    ) -> None:
        self.ldaps = torch.from_numpy(ldaps)
        self.gfs = torch.from_numpy(gfs)
        self.calendar = torch.from_numpy(calendar)
        self.targets = None if targets is None else torch.from_numpy(targets)
        self.indices = torch.from_numpy(indices.astype(np.int64))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        index = int(self.indices[item])
        values: tuple[torch.Tensor, ...] = (
            self.ldaps[index],
            self.gfs[index],
            self.calendar[index],
        )
        if self.targets is not None:
            values += (self.targets[index],)
        return values


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_nodes: int,
        hidden: int,
        adjacency: np.ndarray,
        pooling: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", torch.from_numpy(adjacency))
        self.register_buffer("pooling", torch.from_numpy(pooling))
        self.register_buffer("mean", torch.from_numpy(mean.reshape(1, 1, 1, -1)))
        self.register_buffer("std", torch.from_numpy(std.reshape(1, 1, 1, -1)))
        self.input_projection = nn.Linear(n_features, hidden)
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, hidden) * 0.02)
        self.self_projection = nn.Linear(hidden, hidden)
        self.neighbor_projection = nn.Linear(hidden, hidden)
        self.normalization = nn.LayerNorm(hidden)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.nan_to_num((values - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        hidden = torch.nn.functional.gelu(
            self.input_projection(values) + self.node_embedding.view(1, 1, *self.node_embedding.shape)
        )
        neighbors = torch.einsum("nm,btmh->btnh", self.adjacency, hidden)
        hidden = self.normalization(
            hidden
            + torch.nn.functional.gelu(
                self.self_projection(hidden) + self.neighbor_projection(neighbors)
            )
        )
        return torch.einsum("gn,btnh->btgh", self.pooling, hidden)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.normalization = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.dropout(
            torch.nn.functional.gelu(self.normalization(self.convolution(values)))
        )
        return values + update


class SpatialTemporalMultiTask(nn.Module):
    def __init__(
        self,
        ldaps_shape: tuple[int, int],
        gfs_shape: tuple[int, int],
        hidden: int,
        ldaps_adjacency: np.ndarray,
        gfs_adjacency: np.ndarray,
        ldaps_pooling: np.ndarray,
        gfs_pooling: np.ndarray,
        ldaps_mean: np.ndarray,
        ldaps_std: np.ndarray,
        gfs_mean: np.ndarray,
        gfs_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.ldaps_encoder = SpatialEncoder(
            ldaps_shape[1], ldaps_shape[0], hidden, ldaps_adjacency, ldaps_pooling, ldaps_mean, ldaps_std
        )
        self.gfs_encoder = SpatialEncoder(
            gfs_shape[1], gfs_shape[0], hidden, gfs_adjacency, gfs_pooling, gfs_mean, gfs_std
        )
        self.group_embedding = nn.Parameter(torch.randn(len(TARGETS), hidden) * 0.02)
        input_channels = hidden * 5 + 5
        self.temporal_input = nn.Linear(input_channels, hidden * 2)
        self.temporal_blocks = nn.ModuleList(
            [TemporalBlock(hidden * 2, dilation, dropout=0.10) for dilation in (1, 2, 4)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )

    def forward(self, ldaps: torch.Tensor, gfs: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        ldaps_group = self.ldaps_encoder(ldaps)
        gfs_group = self.gfs_encoder(gfs)
        ldaps_context = ldaps_group.mean(dim=2, keepdim=True).expand_as(ldaps_group)
        gfs_context = gfs_group.mean(dim=2, keepdim=True).expand_as(gfs_group)
        group_embedding = self.group_embedding.view(1, 1, len(TARGETS), -1).expand(
            ldaps.shape[0], ldaps.shape[1], -1, -1
        )
        calendar_group = calendar.unsqueeze(2).expand(-1, -1, len(TARGETS), -1)
        combined = torch.cat(
            [ldaps_group, gfs_group, ldaps_context, gfs_context, group_embedding, calendar_group], dim=-1
        )
        temporal = self.temporal_input(combined)
        batch, steps, groups, channels = temporal.shape
        temporal = temporal.permute(0, 2, 3, 1).reshape(batch * groups, channels, steps)
        for block in self.temporal_blocks:
            temporal = block(temporal)
        temporal = temporal.reshape(batch, groups, channels, steps).permute(0, 3, 1, 2)
        return 1.05 * torch.sigmoid(self.output(temporal).squeeze(-1))


def calendar_tensor(timestamps_ns: np.ndarray) -> np.ndarray:
    timestamps = pd.to_datetime(timestamps_ns.reshape(-1))
    hour = timestamps.hour.to_numpy()
    day = timestamps.dayofyear.to_numpy()
    lead = (((hour - 1) % 24) + 12) / 35.0
    values = np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 365.25),
            np.cos(2 * np.pi * day / 365.25),
            lead,
        ]
    ).astype(np.float32)
    return values.reshape(timestamps_ns.shape[0], 24, -1)


def feature_statistics(values: np.ndarray, day_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = values[day_indices]
    mean = np.nanmean(selected, axis=(0, 1, 2)).astype(np.float32)
    std = np.nanstd(selected, axis=(0, 1, 2)).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
    return mean, std


def competition_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reward_strength: float,
) -> torch.Tensor:
    """Smooth surrogate of the official group-macro competition metric.

    The official score first evaluates each KPX group independently and then
    gives the three group scores equal weight.  This distinction matters here
    because group 3 has substantially more missing training labels than groups
    1 and 2.  Pooling every eligible row into one denominator would therefore
    underweight group 3.
    """
    valid = torch.isfinite(target)
    eligible = valid & (target >= 0.10)
    safe_target = torch.nan_to_num(target, nan=0.0)
    error = torch.abs(prediction - safe_target)
    eligible_weight = eligible.float()

    # 1-NMAE is an unweighted mean within each group, followed by a macro
    # average over groups.  Targets and predictions are already capacity ratios.
    group_count = eligible_weight.sum(dim=(0, 1))
    group_mae = (eligible_weight * error).sum(dim=(0, 1)) / group_count.clamp_min(1.0)

    # The official FICR settlement is generation weighted within each group.
    # A normalized reward is 1 inside 6%, 0.75 inside 8%, and 0 outside;
    # sigmoids provide the differentiable training approximation.
    soft_reward = 0.75 * torch.sigmoid((0.08 - error) / 0.008) + 0.25 * torch.sigmoid(
        (0.06 - error) / 0.008
    )
    reward_weight = eligible_weight * safe_target
    group_reward = (reward_weight * soft_reward).sum(dim=(0, 1)) / reward_weight.sum(
        dim=(0, 1)
    ).clamp_min(1.0)

    available_groups = group_count > 0
    mae = group_mae[available_groups].mean()
    reward = group_reward[available_groups].mean()
    return mae - reward_strength * reward


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    predictions = []
    for batch in loader:
        ldaps, gfs, calendar = batch[:3]
        predictions.append(model(ldaps, gfs, calendar).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def metric_from_days(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    truth = {name: target[:, :, i].reshape(-1) * CAPACITIES[i] for i, name in enumerate(TARGETS)}
    pred = {name: prediction[:, :, i].reshape(-1) * CAPACITIES[i] for i, name in enumerate(TARGETS)}
    return evaluate_competition(truth, pred)


def train_validation_model(
    arrays: np.lib.npyio.NpzFile,
    train_days: np.ndarray,
    selection_days: np.ndarray,
    evaluation_days: np.ndarray,
    seed: int,
    hidden: int,
    epochs: int,
    patience: int,
    batch_size: int,
    reward_strength: float,
) -> tuple[np.ndarray, dict[str, object], int]:
    set_seed(seed)
    ldaps_mean, ldaps_std = feature_statistics(arrays["train_ldaps"], train_days)
    gfs_mean, gfs_std = feature_statistics(arrays["train_gfs"], train_days)
    model = SpatialTemporalMultiTask(
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
    calendar = calendar_tensor(arrays["train_timestamps_ns"])
    train_loader = DataLoader(
        DayDataset(
            arrays["train_ldaps"], arrays["train_gfs"], calendar, arrays["train_targets"], train_days
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    selection_loader = DataLoader(
        DayDataset(
            arrays["train_ldaps"], arrays["train_gfs"], calendar, arrays["train_targets"], selection_days
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for ldaps, gfs, day_calendar, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(ldaps, gfs, day_calendar)
            loss = competition_loss(prediction, target, reward_strength)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        selection_prediction = predict_loader(model, selection_loader)
        selection_target = arrays["train_targets"][selection_days]
        selection_metric = metric_from_days(selection_target, selection_prediction)
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "selection_score": selection_metric["score"]}
        )
        if selection_metric["score"] > best_score + 1e-5:
            best_score = selection_metric["score"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(seed, epoch, history[-1], flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    all_valid_days = np.concatenate([selection_days, evaluation_days])
    valid_loader = DataLoader(
        DayDataset(
            arrays["train_ldaps"], arrays["train_gfs"], calendar, None, all_valid_days
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    prediction = predict_loader(model, valid_loader)
    report = {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_best_score": best_score,
        "full_validation_metric": metric_from_days(arrays["train_targets"][all_valid_days], prediction),
        "evaluation_metric": metric_from_days(
            arrays["train_targets"][evaluation_days], prediction[len(selection_days) :]
        ),
        "history": history,
    }
    return prediction, report, best_epoch


def proxy_blend_diagnostics(
    validation_timestamps: pd.DatetimeIndex,
    validation_truth: np.ndarray,
    neural_prediction: np.ndarray,
    proxy_paths: dict[str, Path],
) -> dict[str, object]:
    lookup = pd.DataFrame(
        neural_prediction.reshape(-1, len(TARGETS)), index=validation_timestamps, columns=TARGETS
    )
    truth_lookup = pd.DataFrame(
        validation_truth.reshape(-1, len(TARGETS)), index=validation_timestamps, columns=TARGETS
    )
    report: dict[str, object] = {}
    for proxy_name, proxy_path in proxy_paths.items():
        cache = np.load(proxy_path, allow_pickle=True)
        proxy_report = {}
        for target_i, target in enumerate(TARGETS):
            index = pd.to_datetime(cache[f"{target}__valid_index_ns"])
            common = index.intersection(lookup.index)
            positions = index.get_indexer(common)
            base = (
                cache[f"{target}__valid_matrix"].astype(float)
                @ cache[f"{target}__selected_weights"].astype(float)
            )[positions]
            truth = truth_lookup.loc[common, target].to_numpy(dtype=float) * CAPACITIES[target_i]
            member = lookup.loc[common, target].to_numpy(dtype=float) * CAPACITIES[target_i]
            # Preserve complete 24-hour NWP issue cycles: the June 30 issue
            # contains the 2024-07-01 00:00 forecast, while the H2 evaluation
            # cycles begin at 2024-07-01 01:00.
            second_half = common >= pd.Timestamp("2024-07-01 01:00:00")
            first_half = ~second_half
            base_full = evaluate_group(truth, base, CAPACITIES[target_i])
            base_h1 = evaluate_group(truth[first_half], base[first_half], CAPACITIES[target_i])
            base_h2 = evaluate_group(truth[second_half], base[second_half], CAPACITIES[target_i])
            candidates = []
            for alpha in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
                blended = np.clip((1.0 - alpha) * base + alpha * member, 0.0, CAPACITIES[target_i])
                full = evaluate_group(truth, blended, CAPACITIES[target_i])
                h1 = evaluate_group(
                    truth[first_half], blended[first_half], CAPACITIES[target_i]
                )
                h2 = evaluate_group(
                    truth[second_half], blended[second_half], CAPACITIES[target_i]
                )
                candidates.append(
                    {
                        "alpha": alpha,
                        "full_score_delta": full.score - base_full.score,
                        "full_nmae_delta": full.one_minus_nmae - base_full.one_minus_nmae,
                        "full_ficr_delta": full.ficr - base_full.ficr,
                        "h1_score_delta": h1.score - base_h1.score,
                        "h1_nmae_delta": h1.one_minus_nmae - base_h1.one_minus_nmae,
                        "h1_ficr_delta": h1.ficr - base_h1.ficr,
                        "h2_score_delta": h2.score - base_h2.score,
                        "h2_nmae_delta": h2.one_minus_nmae - base_h2.one_minus_nmae,
                        "h2_ficr_delta": h2.ficr - base_h2.ficr,
                    }
                )
            proxy_report[target] = candidates
        report[proxy_name] = proxy_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_spatiotemporal")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reward-strength", type=float, default=0.03)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Validation/training never opens the evaluation data.  The test-only
    # cache is created later by the dedicated inference entry point.
    cache_path = build_split_tensor_cache(
        Path(args.data_dir), artifact_dir, "train", args.rebuild_cache
    )
    arrays = np.load(cache_path, allow_pickle=True)
    day_timestamps = pd.to_datetime(arrays["train_timestamps_ns"][:, 0])
    day_end_timestamps = pd.to_datetime(arrays["train_timestamps_ns"][:, -1])
    train_days = np.flatnonzero(day_end_timestamps < pd.Timestamp("2024-01-01"))
    selection_days = np.flatnonzero(
        (day_timestamps >= pd.Timestamp("2024-01-01"))
        & (day_timestamps < pd.Timestamp("2024-07-01"))
    )
    evaluation_days = np.flatnonzero(
        (day_timestamps >= pd.Timestamp("2024-07-01"))
        & (day_timestamps < pd.Timestamp("2025-01-01"))
    )
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    seed_predictions = []
    seed_reports = []
    best_epochs = []
    for seed in seeds:
        prediction, report, best_epoch = train_validation_model(
            arrays,
            train_days,
            selection_days,
            evaluation_days,
            seed,
            args.hidden,
            args.epochs,
            args.patience,
            args.batch_size,
            args.reward_strength,
        )
        seed_predictions.append(prediction)
        seed_reports.append(report)
        best_epochs.append(best_epoch)
    ensemble_prediction = np.mean(seed_predictions, axis=0)
    validation_days = np.concatenate([selection_days, evaluation_days])
    validation_timestamps = pd.to_datetime(arrays["train_timestamps_ns"][validation_days].reshape(-1))
    validation_truth = arrays["train_targets"][validation_days]
    report = {
        "architecture": {
            "hidden": args.hidden,
            "reward_strength": args.reward_strength,
            "seeds": seeds,
            "best_epochs": best_epochs,
        },
        "seed_reports": seed_reports,
        "ensemble_full_metric": metric_from_days(validation_truth, ensemble_prediction),
        "ensemble_h2_metric": metric_from_days(
            arrays["train_targets"][evaluation_days], ensemble_prediction[len(selection_days) :]
        ),
        "proxy_blends": proxy_blend_diagnostics(
            validation_timestamps,
            validation_truth,
            ensemble_prediction,
            {
                # The old per-experiment trees were consolidated after their
                # lineage was audited.  Keep validation tied to the retained
                # exact cache instead of silently rebuilding stale proxies.
                "weighted": Path("artifacts_final/lineage_inputs/weighted_prediction_cache.npz"),
            },
        ),
    }
    (artifact_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        artifact_dir / "validation_predictions.npz",
        timestamps_ns=validation_timestamps.astype("int64").to_numpy(),
        truth=validation_truth.reshape(-1, len(TARGETS)).astype(np.float32),
        prediction=ensemble_prediction.reshape(-1, len(TARGETS)).astype(np.float32),
    )
    print(json.dumps({"ensemble_full_metric": report["ensemble_full_metric"], "ensemble_h2_metric": report["ensemble_h2_metric"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
