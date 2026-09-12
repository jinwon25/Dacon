"""Separated train/inference lifecycle for the optional neural diversity member."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.spatiotemporal_final import train_final_model
from experiments.spatiotemporal_multitask import (
    CAPACITIES,
    TARGETS,
    DayDataset,
    SpatialTemporalMultiTask,
    build_split_tensor_cache,
    calendar_tensor,
    feature_statistics,
    graph_adjacency,
    group_pooling_weights,
    predict_loader,
)
from src.features import TIME_COL
from src.metrics import CAPACITY_KWH


def train_spatiotemporal_member(
    data_dir: str | Path,
    artifact_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (17, 29),
    epochs_per_seed: tuple[int, ...] = (16, 10),
    hidden: int = 24,
    batch_size: int = 32,
    reward_strength: float = 0.03,
) -> dict:
    """Fit and serialize the member without opening test data."""

    if not seeds or len(seeds) != len(epochs_per_seed):
        raise ValueError("seeds and epochs_per_seed must have equal non-zero length")
    started = time.perf_counter()
    data_dir, artifact_dir = Path(data_dir), Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = build_split_tensor_cache(data_dir, artifact_dir, "train", rebuild=False)
    arrays = np.load(cache_path, allow_pickle=True)
    days = np.arange(len(arrays["train_ldaps"]), dtype=int)
    ldaps_mean, ldaps_std = feature_statistics(arrays["train_ldaps"], days)
    gfs_mean, gfs_std = feature_statistics(arrays["train_gfs"], days)
    np.savez_compressed(
        artifact_dir / "model_context.npz",
        ldaps_coordinates=arrays["ldaps_coordinates"],
        gfs_coordinates=arrays["gfs_coordinates"],
        ldaps_mean=ldaps_mean,
        ldaps_std=ldaps_std,
        gfs_mean=gfs_mean,
        gfs_std=gfs_std,
    )
    model_paths = []
    for seed, epochs in zip(seeds, epochs_per_seed, strict=True):
        model = train_final_model(
            arrays, seed, hidden, epochs, batch_size, reward_strength,
        )
        path = artifact_dir / f"model_seed{seed}.pt"
        torch.save(model.state_dict(), path)
        model_paths.append(path.name)
    manifest = {
        "seeds": list(seeds),
        "epochs_per_seed": list(epochs_per_seed),
        "hidden": hidden,
        "batch_size": batch_size,
        "reward_strength": reward_strength,
        "models": model_paths,
        "train_metadata_json": str(arrays["metadata_json"]),
        "train_cache": cache_path.name,
        "test_data_opened": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def _restore_model(
    context: np.lib.npyio.NpzFile,
    arrays: np.lib.npyio.NpzFile,
    hidden: int,
    state_path: Path,
) -> SpatialTemporalMultiTask:
    model = SpatialTemporalMultiTask(
        ldaps_shape=arrays["test_ldaps"].shape[2:4],
        gfs_shape=arrays["test_gfs"].shape[2:4],
        hidden=hidden,
        ldaps_adjacency=graph_adjacency(context["ldaps_coordinates"]),
        gfs_adjacency=graph_adjacency(context["gfs_coordinates"]),
        ldaps_pooling=group_pooling_weights(context["ldaps_coordinates"]),
        gfs_pooling=group_pooling_weights(context["gfs_coordinates"]),
        ldaps_mean=context["ldaps_mean"],
        ldaps_std=context["ldaps_std"],
        gfs_mean=context["gfs_mean"],
        gfs_std=context["gfs_std"],
    )
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    return model


def infer_spatiotemporal_member(
    data_dir: str | Path,
    artifact_dir: str | Path,
    output: str | Path,
) -> Path:
    """Load serialized models and predict test NWP without train labels."""

    data_dir, artifact_dir, output = Path(data_dir), Path(artifact_dir), Path(output)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    test_cache = build_split_tensor_cache(data_dir, artifact_dir, "test", rebuild=False)
    arrays = np.load(test_cache, allow_pickle=True)
    if str(arrays["metadata_json"]) != manifest["train_metadata_json"]:
        raise ValueError("Train/test spatiotemporal feature columns differ")
    context = np.load(artifact_dir / "model_context.npz")
    for source in ("ldaps", "gfs"):
        if not np.array_equal(context[f"{source}_coordinates"], arrays[f"{source}_coordinates"]):
            raise ValueError(f"Train/test {source} coordinates differ")
    loader = DataLoader(
        DayDataset(
            arrays["test_ldaps"],
            arrays["test_gfs"],
            calendar_tensor(arrays["test_timestamps_ns"], arrays["test_availability_ns"]),
            None,
            np.arange(len(arrays["test_ldaps"]), dtype=int),
        ),
        batch_size=int(manifest["batch_size"]),
        shuffle=False,
    )
    predictions = []
    for model_path in manifest["models"]:
        model = _restore_model(context, arrays, int(manifest["hidden"]), artifact_dir / model_path)
        predictions.append(predict_loader(model, loader))
    ratio = np.mean(predictions, axis=0)
    values = ratio.reshape(-1, len(TARGETS)) * CAPACITIES
    timestamps = pd.DatetimeIndex(pd.to_datetime(arrays["test_timestamps_ns"].reshape(-1)))
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    if not pd.DatetimeIndex(pd.to_datetime(sample[TIME_COL])).equals(timestamps):
        raise ValueError("Spatiotemporal timestamps do not match sample_submission")
    member = sample.copy()
    for target_i, target in enumerate(TARGETS):
        member[target] = np.clip(values[:, target_i], 0.0, CAPACITY_KWH[target])
    output.parent.mkdir(parents=True, exist_ok=True)
    member.to_csv(output, index=False, encoding="utf-8-sig")
    return output
