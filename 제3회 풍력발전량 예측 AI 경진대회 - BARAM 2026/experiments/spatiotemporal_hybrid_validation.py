from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.cross_group_transfer import (
    fit_predict_models,
    selected_prediction,
    transfer_features,
)
from experiments.spatiotemporal_final import hybrid_group3_prediction
from src.metrics import CAPACITY_KWH, evaluate_group


TARGET = "kpx_group_3"
H2_START = pd.Timestamp("2024-07-01 01:00:00")


def metric_deltas(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    capacity = CAPACITY_KWH[TARGET]
    base_metric = evaluate_group(truth, baseline, capacity)
    candidate_metric = evaluate_group(truth, candidate, capacity)
    return {
        "score": candidate_metric.score - base_metric.score,
        "one_minus_nmae": candidate_metric.one_minus_nmae - base_metric.one_minus_nmae,
        "ficr": candidate_metric.ficr - base_metric.ficr,
    }


def validate_hybrid(
    labels: pd.DataFrame,
    neural_path: Path,
    weighted_cache_path: Path,
    proxy_paths: dict[str, Path],
    spatial_weights: list[float],
) -> dict[str, object]:
    neural = np.load(neural_path, allow_pickle=True)
    neural_index = pd.to_datetime(neural["timestamps_ns"])
    neural_prediction = neural["prediction"].astype(float)
    if neural_prediction.shape != (len(neural_index), 3):
        raise ValueError("Unexpected spatial-temporal validation prediction shape")

    weighted = np.load(weighted_cache_path, allow_pickle=True)
    weighted_indices = {
        target: pd.to_datetime(weighted[f"{target}__valid_index_ns"])
        for target in CAPACITY_KWH
    }
    if not all(weighted_indices[TARGET].equals(index) for index in weighted_indices.values()):
        raise ValueError("Weighted-cache validation indices differ between groups")
    common = neural_index.intersection(weighted_indices[TARGET])
    neural_positions = neural_index.get_indexer(common)
    weighted_positions = weighted_indices[TARGET].get_indexer(common)
    if (neural_positions < 0).any() or (weighted_positions < 0).any():
        raise ValueError("Failed to align neural and weighted validation timestamps")

    train = labels[labels.index.year == 2023].dropna(
        subset=["kpx_group_1", "kpx_group_2", TARGET]
    )
    cross_train_features = transfer_features(
        train["kpx_group_1"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_1"],
        train["kpx_group_2"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_2"],
        train.index,
    )
    group_1_ratio = (
        selected_prediction(weighted, "kpx_group_1")[weighted_positions]
        / CAPACITY_KWH["kpx_group_1"]
    )
    group_2_ratio = (
        selected_prediction(weighted, "kpx_group_2")[weighted_positions]
        / CAPACITY_KWH["kpx_group_2"]
    )
    cross_member = np.clip(
        fit_predict_models(
            cross_train_features,
            train[TARGET].to_numpy(dtype=float),
            transfer_features(group_1_ratio, group_2_ratio, common),
            seed=51_000,
            mode="base",
        ),
        0.0,
        CAPACITY_KWH[TARGET],
    )
    spatial_member = (
        neural_prediction[neural_positions, 2] * CAPACITY_KWH[TARGET]
    )
    truth = labels.reindex(common)[TARGET].to_numpy(dtype=float)
    split_masks = {
        "h1": common < H2_START,
        "h2": common >= H2_START,
        "full": np.ones(len(common), dtype=bool),
    }

    report: dict[str, object] = {
        "rows": int(len(common)),
        "first_timestamp": str(common.min()),
        "last_timestamp": str(common.max()),
        "h2_start": str(H2_START),
        "cross_train_rows": int(len(train)),
        "spatial_weights": spatial_weights,
        "proxies": {},
    }
    for proxy_name, proxy_path in proxy_paths.items():
        proxy = np.load(proxy_path, allow_pickle=True)
        proxy_index = pd.to_datetime(proxy[f"{TARGET}__valid_index_ns"])
        proxy_positions = proxy_index.get_indexer(common)
        if (proxy_positions < 0).any():
            raise ValueError(f"{proxy_name} cache does not cover all validation timestamps")
        base = selected_prediction(proxy, TARGET)[proxy_positions]
        cross_only, _, _ = hybrid_group3_prediction(
            base,
            cross_member,
            cross_member,
            group_1_ratio,
            group_2_ratio,
            spatial_weight=0.0,
        )
        candidates = []
        for spatial_weight in spatial_weights:
            candidate, _, gate = hybrid_group3_prediction(
                base,
                cross_member,
                spatial_member,
                group_1_ratio,
                group_2_ratio,
                spatial_weight=spatial_weight,
            )
            item: dict[str, object] = {
                "spatial_weight": spatial_weight,
                "changed_rows": int(np.count_nonzero(candidate != cross_only)),
                "gate_rows": int(gate.sum()),
            }
            for split_name, split_mask in split_masks.items():
                item[split_name] = metric_deltas(
                    truth[split_mask], cross_only[split_mask], candidate[split_mask]
                )
            candidates.append(item)
        report["proxies"][proxy_name] = candidates

    # Hyperparameter selection sees H1 only.  H2 remains a confirmation set.
    robust_weights = []
    for spatial_weight in spatial_weights:
        if spatial_weight == 0.0:
            continue
        items = [
            next(
                item
                for item in report["proxies"][proxy_name]
                if item["spatial_weight"] == spatial_weight
            )
            for proxy_name in proxy_paths
        ]
        h1_components = [
            float(item["h1"][component])
            for item in items
            for component in ("score", "one_minus_nmae", "ficr")
        ]
        if min(h1_components) > 0.0:
            robust_weights.append(
                {
                    "spatial_weight": spatial_weight,
                    "minimum_h1_component_delta": min(h1_components),
                    "mean_h1_score_delta": float(
                        np.mean([item["h1"]["score"] for item in items])
                    ),
                }
            )
    selected = (
        max(robust_weights, key=lambda item: item["mean_h1_score_delta"])
        if robust_weights
        else None
    )
    if selected is not None:
        weight = selected["spatial_weight"]
        confirmation = [
            next(
                item
                for item in report["proxies"][proxy_name]
                if item["spatial_weight"] == weight
            )["h2"]
            for proxy_name in proxy_paths
        ]
        selected["h2_all_components_positive"] = bool(
            min(
                float(metric[component])
                for metric in confirmation
                for component in ("score", "one_minus_nmae", "ficr")
            )
            > 0.0
        )
        selected["h2_metrics"] = confirmation
    report["robust_h1_candidates"] = robust_weights
    report["selected_from_h1"] = selected
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--neural-predictions",
        default="artifacts_spatiotemporal/validation_predictions.npz",
    )
    parser.add_argument(
        "--weighted-cache", default="artifacts_weighted_metric/prediction_cache.npz"
    )
    parser.add_argument("--global-cache", default="artifacts_global/prediction_cache.npz")
    parser.add_argument("--pool-cache", default="artifacts_final_pool/prediction_cache.npz")
    parser.add_argument(
        "--output", default="artifacts_spatiotemporal/hybrid_validation_report.json"
    )
    parser.add_argument("--spatial-weights", default="0,0.05,0.10,0.20,0.30,0.50")
    args = parser.parse_args()

    labels = pd.read_csv(
        Path(args.data_dir) / "train" / "train_labels.csv", encoding="utf-8-sig"
    )
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")
    weights = [
        float(value.strip()) for value in args.spatial_weights.split(",") if value.strip()
    ]
    report = validate_hybrid(
        labels,
        Path(args.neural_predictions),
        Path(args.weighted_cache),
        {
            "weighted": Path(args.weighted_cache),
            "global": Path(args.global_cache),
            "pool": Path(args.pool_cache),
        },
        weights,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["selected_from_h1"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
