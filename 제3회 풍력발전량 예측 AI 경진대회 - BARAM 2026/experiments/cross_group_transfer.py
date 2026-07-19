from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from src.metrics import CAPACITY_KWH, evaluate_group


KEY_COLUMNS = ["forecast_id", "forecast_kst_dtm"]


def transfer_features(group_1: np.ndarray, group_2: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
    hour = timestamps.hour.to_numpy()
    dayofyear = timestamps.dayofyear.to_numpy()
    return np.column_stack(
        [
            group_1,
            group_2,
            (group_1 + group_2) / 2.0,
            group_2 - group_1,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dayofyear / 365.25),
            np.cos(2 * np.pi * dayofyear / 365.25),
        ]
    )


def make_model(
    seed: int,
    min_samples_leaf: int = 30,
    max_features: float = 0.9,
    n_estimators: int = 700,
) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=seed,
    )


def model_specs(mode: str) -> list[tuple[int, float]]:
    if mode == "base":
        return [(30, 0.9)]
    if mode == "smooth":
        return [(120, 1.0)]
    if mode == "ensemble":
        return [(30, 0.9), (120, 0.6)]
    raise ValueError(f"Unknown model mode: {mode}")


def fit_predict_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    seed: int,
    mode: str,
) -> np.ndarray:
    predictions = []
    for model_i, (min_samples_leaf, max_features) in enumerate(model_specs(mode)):
        model = make_model(
            seed=seed + model_i,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_predict))
    return np.mean(predictions, axis=0)


def selected_prediction(cache: np.lib.npyio.NpzFile, target: str) -> np.ndarray:
    return (
        cache[f"{target}__valid_matrix"].astype(float)
        @ cache[f"{target}__selected_weights"].astype(float)
    )


def validate_transfer(
    labels: pd.DataFrame,
    weighted_cache_path: Path,
    proxy_cache_paths: dict[str, Path],
    alpha: float,
    max_group_disagreement: float,
    max_member_disagreement: float,
    model_mode: str = "base",
) -> dict[str, object]:
    capacity_1 = CAPACITY_KWH["kpx_group_1"]
    capacity_2 = CAPACITY_KWH["kpx_group_2"]
    capacity_3 = CAPACITY_KWH["kpx_group_3"]
    weighted = np.load(weighted_cache_path, allow_pickle=True)
    all_index = pd.to_datetime(weighted["kpx_group_3__valid_index_ns"])
    keep = all_index.year == 2024
    valid_index = all_index[keep]

    train = labels[(labels.index.year == 2023)].dropna(
        subset=["kpx_group_1", "kpx_group_2", "kpx_group_3"]
    )
    train_features = transfer_features(
        train["kpx_group_1"].to_numpy(dtype=float) / capacity_1,
        train["kpx_group_2"].to_numpy(dtype=float) / capacity_2,
        train.index,
    )
    valid_group_1 = selected_prediction(weighted, "kpx_group_1")[keep] / capacity_1
    valid_group_2 = selected_prediction(weighted, "kpx_group_2")[keep] / capacity_2
    member = np.clip(
        fit_predict_models(
            train_features,
            train["kpx_group_3"].to_numpy(dtype=float),
            transfer_features(valid_group_1, valid_group_2, valid_index),
            seed=51_000,
            mode=model_mode,
        ),
        0.0,
        capacity_3,
    )
    truth = labels.reindex(valid_index)["kpx_group_3"].to_numpy(dtype=float)
    second_half = valid_index >= pd.Timestamp("2024-07-01")
    report: dict[str, object] = {"model_mode": model_mode, "proxies": {}}

    for name, cache_path in proxy_cache_paths.items():
        proxy = np.load(cache_path, allow_pickle=True)
        if not np.array_equal(
            proxy["kpx_group_3__valid_index_ns"], weighted["kpx_group_3__valid_index_ns"]
        ):
            raise ValueError(f"Validation index mismatch for {name}")
        base = selected_prediction(proxy, "kpx_group_3")[keep]
        mask = (
            (np.abs(valid_group_1 - valid_group_2) <= max_group_disagreement)
            & (np.abs(member - base) / capacity_3 <= max_member_disagreement)
            & (base >= 0.10 * capacity_3)
        )
        candidate = base.copy()
        candidate[mask] = (1.0 - alpha) * base[mask] + alpha * member[mask]
        base_full = evaluate_group(truth, base, capacity_3)
        candidate_full = evaluate_group(truth, candidate, capacity_3)
        base_h2 = evaluate_group(truth[second_half], base[second_half], capacity_3)
        candidate_h2 = evaluate_group(
            truth[second_half], candidate[second_half], capacity_3
        )
        months_improved = 0
        for month in range(1, 13):
            month_mask = valid_index.month == month
            if evaluate_group(
                truth[month_mask], candidate[month_mask], capacity_3
            ).score > evaluate_group(truth[month_mask], base[month_mask], capacity_3).score:
                months_improved += 1
        report["proxies"][name] = {
            "eligible_ratio": float(mask.mean()),
            "full_score_delta": candidate_full.score - base_full.score,
            "full_one_minus_nmae_delta": (
                candidate_full.one_minus_nmae - base_full.one_minus_nmae
            ),
            "full_ficr_delta": candidate_full.ficr - base_full.ficr,
            "h2_score_delta": candidate_h2.score - base_h2.score,
            "h2_one_minus_nmae_delta": (
                candidate_h2.one_minus_nmae - base_h2.one_minus_nmae
            ),
            "h2_ficr_delta": candidate_h2.ficr - base_h2.ficr,
            "months_improved": months_improved,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base", default="artifacts_cross_group/base_pre_cross.csv")
    parser.add_argument("--weighted-cache", default="artifacts_weighted_metric/prediction_cache.npz")
    parser.add_argument("--global-cache", default="artifacts_global/prediction_cache.npz")
    parser.add_argument("--pool-cache", default="artifacts_final_pool/prediction_cache.npz")
    parser.add_argument("--artifact-dir", default="artifacts_cross_group")
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max-group-disagreement", type=float, default=0.08)
    parser.add_argument("--max-member-disagreement", type=float, default=0.06)
    parser.add_argument("--model-mode", choices=["base", "smooth", "ensemble"], default="base")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")
    report = validate_transfer(
        labels=labels,
        weighted_cache_path=Path(args.weighted_cache),
        proxy_cache_paths={
            "weighted": Path(args.weighted_cache),
            "global": Path(args.global_cache),
            "pool": Path(args.pool_cache),
        },
        alpha=args.alpha,
        max_group_disagreement=args.max_group_disagreement,
        max_member_disagreement=args.max_member_disagreement,
        model_mode=args.model_mode,
    )

    base = pd.read_csv(args.base, encoding="utf-8-sig")
    timestamps = pd.to_datetime(base["forecast_kst_dtm"])
    train = labels[
        (labels.index >= pd.Timestamp("2023-01-01"))
        & (labels.index < pd.Timestamp("2025-01-01"))
    ].dropna(subset=["kpx_group_1", "kpx_group_2", "kpx_group_3"])
    train_features = transfer_features(
        train["kpx_group_1"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_1"],
        train["kpx_group_2"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_2"],
        train.index,
    )
    test_group_1 = base["kpx_group_1"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_1"]
    test_group_2 = base["kpx_group_2"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_2"]
    member = np.clip(
        fit_predict_models(
            train_features,
            train["kpx_group_3"].to_numpy(dtype=float),
            transfer_features(test_group_1, test_group_2, pd.DatetimeIndex(timestamps)),
            seed=52_000,
            mode=args.model_mode,
        ),
        0.0,
        CAPACITY_KWH["kpx_group_3"],
    )
    base_group_3 = base["kpx_group_3"].to_numpy(dtype=float)
    mask = (
        (np.abs(test_group_1 - test_group_2) <= args.max_group_disagreement)
        & (
            np.abs(member - base_group_3) / CAPACITY_KWH["kpx_group_3"]
            <= args.max_member_disagreement
        )
        & (base_group_3 >= 0.10 * CAPACITY_KWH["kpx_group_3"])
    )
    output = base.copy()
    output.loc[mask, "kpx_group_3"] = (
        (1.0 - args.alpha) * base_group_3[mask] + args.alpha * member[mask]
    )
    output["kpx_group_3"] = output["kpx_group_3"].clip(
        0.0, CAPACITY_KWH["kpx_group_3"]
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    member_output = base[KEY_COLUMNS].copy()
    member_output["kpx_group_3"] = member
    member_output.to_csv(artifact_dir / "cross_group_member.csv", index=False, encoding="utf-8-sig")
    delta = output["kpx_group_3"].to_numpy(dtype=float) - base_group_3
    report["final"] = {
        "train_rows": int(len(train)),
        "alpha": args.alpha,
        "max_group_disagreement": args.max_group_disagreement,
        "max_member_disagreement": args.max_member_disagreement,
        "model_mode": args.model_mode,
        "eligible_rows": int(mask.sum()),
        "eligible_ratio": float(mask.mean()),
        "mean_delta": float(delta.mean()),
        "mean_absolute_delta": float(np.abs(delta).mean()),
        "p95_absolute_delta": float(np.quantile(np.abs(delta), 0.95)),
    }
    (artifact_dir / "cross_group_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
