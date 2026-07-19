from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.spatiotemporal_multitask import (
    CAPACITIES,
    TARGETS,
    DayDataset,
    SpatialTemporalMultiTask,
    build_split_tensor_cache,
    calendar_tensor,
    competition_loss,
    feature_statistics,
    graph_adjacency,
    group_pooling_weights,
    predict_loader,
    set_seed,
)
from src.metrics import CAPACITY_KWH


KEY_COLUMNS = ["forecast_id", "forecast_kst_dtm"]


def hybrid_group3_prediction(
    base: np.ndarray,
    cross_member: np.ndarray,
    spatial_member: np.ndarray,
    group_1_ratio: np.ndarray,
    group_2_ratio: np.ndarray,
    spatial_weight: float = 0.20,
    correction_weight: float = 0.25,
    max_group_disagreement: float = 0.08,
    max_member_disagreement: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    capacity = CAPACITY_KWH["kpx_group_3"]
    member = (1.0 - spatial_weight) * cross_member + spatial_weight * spatial_member
    mask = (
        (np.abs(group_1_ratio - group_2_ratio) <= max_group_disagreement)
        & (np.abs(member - base) / capacity <= max_member_disagreement)
        & (base >= 0.10 * capacity)
    )
    prediction = base.copy()
    prediction[mask] = (
        (1.0 - correction_weight) * base[mask] + correction_weight * member[mask]
    )
    return np.clip(prediction, 0.0, capacity), member, mask


def train_final_model(
    arrays: np.lib.npyio.NpzFile,
    seed: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    reward_strength: float,
) -> SpatialTemporalMultiTask:
    set_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    train_days = np.arange(len(arrays["train_ldaps"]), dtype=int)
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
    train_loader = DataLoader(
        DayDataset(
            arrays["train_ldaps"],
            arrays["train_gfs"],
            calendar_tensor(arrays["train_timestamps_ns"]),
            arrays["train_targets"],
            train_days,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for ldaps, gfs, calendar, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(ldaps, gfs, calendar)
            loss = competition_loss(prediction, target, reward_strength)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 4 == 0 or epoch == epochs:
            print("final", seed, epoch, float(np.mean(losses)), flush=True)
    return model


def assert_compatible_caches(
    train_arrays: np.lib.npyio.NpzFile,
    test_arrays: np.lib.npyio.NpzFile,
) -> None:
    train_metadata = json.loads(str(train_arrays["metadata_json"]))
    test_metadata = json.loads(str(test_arrays["metadata_json"]))
    if train_metadata != test_metadata:
        raise ValueError("Train/test NWP feature columns differ")
    for source in ("ldaps", "gfs"):
        if not np.array_equal(
            train_arrays[f"{source}_coordinates"], test_arrays[f"{source}_coordinates"]
        ):
            raise ValueError(f"Train/test {source} grid coordinates differ")


def predict_test(
    model: SpatialTemporalMultiTask,
    arrays: np.lib.npyio.NpzFile,
    batch_size: int,
) -> np.ndarray:
    test_days = np.arange(len(arrays["test_ldaps"]), dtype=int)
    test_loader = DataLoader(
        DayDataset(
            arrays["test_ldaps"],
            arrays["test_gfs"],
            calendar_tensor(arrays["test_timestamps_ns"]),
            None,
            test_days,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return predict_loader(model, test_loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--train-tensor-cache",
        default="artifacts_spatiotemporal/spatiotemporal_train_tensors.npz",
    )
    parser.add_argument(
        "--test-tensor-cache",
        default="artifacts_spatiotemporal/spatiotemporal_test_tensors.npz",
    )
    parser.add_argument("--base", default="artifacts_cross_group/base_pre_cross.csv")
    parser.add_argument("--cross-member", default="artifacts_cross_group/cross_group_member.csv")
    parser.add_argument("--artifact-dir", default="artifacts_spatiotemporal")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="17,29")
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--epochs-per-seed", default="16,10")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reward-strength", type=float, default=0.03)
    parser.add_argument("--spatial-weight", type=float, default=0.20)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_cache_path = build_split_tensor_cache(
        Path(args.data_dir), artifact_dir, "train", rebuild=False
    )
    if Path(args.train_tensor_cache) != train_cache_path:
        train_cache_path = Path(args.train_tensor_cache)
    train_arrays = np.load(train_cache_path, allow_pickle=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    epochs_per_seed = [
        int(value.strip()) for value in args.epochs_per_seed.split(",") if value.strip()
    ]
    if not seeds or len(seeds) != len(epochs_per_seed):
        raise ValueError("--seeds and --epochs-per-seed must have the same non-zero length")
    models = [
        train_final_model(
            train_arrays,
            seed=seed,
            hidden=args.hidden,
            epochs=epochs,
            batch_size=args.batch_size,
            reward_strength=args.reward_strength,
        )
        for seed, epochs in zip(seeds, epochs_per_seed, strict=True)
    ]

    # Evaluation data is opened only after every model has finished fitting.
    test_cache_path = build_split_tensor_cache(
        Path(args.data_dir), artifact_dir, "test", rebuild=False
    )
    if Path(args.test_tensor_cache) != test_cache_path:
        test_cache_path = Path(args.test_tensor_cache)
    test_arrays = np.load(test_cache_path, allow_pickle=True)
    assert_compatible_caches(train_arrays, test_arrays)
    neural_ratio = np.mean(
        [predict_test(model, test_arrays, args.batch_size) for model in models], axis=0
    )
    test_timestamps = pd.to_datetime(test_arrays["test_timestamps_ns"].reshape(-1))
    neural_values = neural_ratio.reshape(-1, len(TARGETS)) * CAPACITIES

    sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv", encoding="utf-8-sig")
    if not pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"])).equals(test_timestamps):
        raise ValueError("Tensor test timestamps do not match sample submission")
    neural_member = sample[KEY_COLUMNS].copy()
    for target_i, target in enumerate(TARGETS):
        neural_member[target] = np.clip(
            neural_values[:, target_i], 0.0, CAPACITY_KWH[target]
        )
    neural_member.to_csv(
        artifact_dir / "spatiotemporal_member.csv", index=False, encoding="utf-8-sig"
    )

    base = pd.read_csv(args.base, encoding="utf-8-sig")
    cross = pd.read_csv(args.cross_member, encoding="utf-8-sig")
    if not base[KEY_COLUMNS].equals(cross[KEY_COLUMNS]) or not base[KEY_COLUMNS].equals(
        neural_member[KEY_COLUMNS]
    ):
        raise ValueError("Submission keys do not match")
    group_3, hybrid_member, mask = hybrid_group3_prediction(
        base=base["kpx_group_3"].to_numpy(dtype=float),
        cross_member=cross["kpx_group_3"].to_numpy(dtype=float),
        spatial_member=neural_member["kpx_group_3"].to_numpy(dtype=float),
        group_1_ratio=base["kpx_group_1"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_1"],
        group_2_ratio=base["kpx_group_2"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_2"],
        spatial_weight=args.spatial_weight,
    )
    output = base.copy()
    output["kpx_group_3"] = group_3
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    delta = group_3 - base["kpx_group_3"].to_numpy(dtype=float)
    report = {
        "seeds": seeds,
        "epochs_per_seed": epochs_per_seed,
        "hidden": args.hidden,
        "reward_strength": args.reward_strength,
        "spatial_weight": args.spatial_weight,
        "cross_weight": 1.0 - args.spatial_weight,
        "changed_rows": int(mask.sum()),
        "changed_ratio": float(mask.mean()),
        "mean_delta": float(delta.mean()),
        "mean_absolute_delta": float(np.abs(delta).mean()),
        "p95_absolute_delta": float(np.quantile(np.abs(delta), 0.95)),
        "member_correlation": float(
            np.corrcoef(cross["kpx_group_3"].to_numpy(dtype=float), neural_member["kpx_group_3"])[0, 1]
        ),
    }
    (artifact_dir / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        artifact_dir / "test_predictions.npz",
        timestamps_ns=test_timestamps.astype("int64").to_numpy(),
        neural_prediction=neural_values.astype(np.float32),
        hybrid_group3=hybrid_member.astype(np.float32),
        output_group3=group_3.astype(np.float32),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
