"""Screen east/centre/west KMA UMRG context as a new spatial signal.

The existing one-year-forward candidate uses one KMA UMRG point.  This screen
keeps its alpha, blend weight, seed set, and movement bound frozen, replacing
only the centre-point expert with an expert that receives east/centre/west
mean, spread, and zonal-gradient features.  It uses 2023 labels to predict
2024 and never downloads or predicts 2025 unless this independent spatial
increment survives the forward gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_service.compliance import validate_external_data_manifest
from experiments.blocked_rolling_validation import evaluate_blocked_rolling
from experiments.kma_year_forward_quantile_blend import (
    H2_START,
    Q2_START,
    SEEDS,
    VALIDATION_END,
    _align_prediction,
    apply_bounded_blend,
    fit_predict,
    load_context_features,
    metric_delta,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("kpx_group_1", "kpx_group_3")
FROZEN_SPECS = {
    "kpx_group_1": {"alpha": 0.80, "weight": 0.10},
    "kpx_group_3": {"alpha": 0.75, "weight": 0.075},
}
REPLACEMENT_FACTORS = (0.50, 1.00)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_spatial_features(
    centre: pd.DataFrame,
    east: pd.DataFrame,
    west: pd.DataFrame,
) -> pd.DataFrame:
    """Add three-point mean, standard deviation, and east-west gradient."""
    if not centre.index.equals(east.index) or not centre.index.equals(west.index):
        raise ValueError("KMA spatial feature indexes differ")
    common = [
        column
        for column in centre.columns
        if column.startswith("kma_um_ctx_")
        and column in east.columns
        and column in west.columns
    ]
    if not common:
        raise ValueError("KMA spatial inputs have no common model features")
    stack = np.stack(
        [
            west[common].to_numpy(dtype=float),
            centre[common].to_numpy(dtype=float),
            east[common].to_numpy(dtype=float),
        ],
        axis=2,
    )
    derived = pd.DataFrame(
        np.concatenate(
            [
                np.mean(stack, axis=2),
                np.std(stack, axis=2),
                east[common].to_numpy(dtype=float)
                - west[common].to_numpy(dtype=float),
            ],
            axis=1,
        ),
        index=centre.index,
        columns=[
            *[f"kma_spatial_mean__{column}" for column in common],
            *[f"kma_spatial_std__{column}" for column in common],
            *[
                f"kma_spatial_east_minus_west__{column}"
                for column in common
            ],
        ],
    )
    output = pd.concat([centre, derived], axis=1)
    if output.isna().any().any() or not np.isfinite(output.to_numpy()).all():
        raise ValueError("KMA spatial features are incomplete")
    return output.astype("float32")


def interpolate_expert(
    centre: np.ndarray,
    spatial: np.ndarray,
    factor: float,
) -> np.ndarray:
    if not 0.0 <= factor <= 1.0:
        raise ValueError("replacement factor must lie in [0, 1]")
    centre = np.asarray(centre, dtype=float)
    spatial = np.asarray(spatial, dtype=float)
    if centre.shape != spatial.shape:
        raise ValueError("centre and spatial experts must align")
    return centre + float(factor) * (spatial - centre)


def _validation_surface(
    driver: np.lib.npyio.NpzFile,
    kma_oof: np.lib.npyio.NpzFile,
    target: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{target}__valid_index_ns"])
    )
    if target == "kpx_group_1":
        return (
            driver_index,
            driver[f"{target}__valid_truth"].astype(float),
            driver[f"{target}__exact_base"].astype(float),
        )
    kma_index = pd.DatetimeIndex(pd.to_datetime(kma_oof["index_ns"]))
    if not kma_index.equals(driver_index):
        raise ValueError("KMA and driver validation indexes differ")
    return (
        kma_index,
        kma_oof["truth"].astype(float),
        kma_oof["rolling_candidate"].astype(float),
    )


def _periods(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "q1": np.asarray(index < Q2_START),
        "q2": np.asarray((index >= Q2_START) & (index < H2_START)),
        "h1": np.asarray(index < H2_START),
        "h2": np.asarray(
            (index >= H2_START) & (index < VALIDATION_END)
        ),
        "full": np.asarray(index < VALIDATION_END),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "centre_2023": _rooted(args.centre_2023),
        "centre_2024": _rooted(args.centre_2024),
        "east_2023": _rooted(args.east_2023),
        "east_2024": _rooted(args.east_2024),
        "west_2023": _rooted(args.west_2023),
        "west_2024": _rooted(args.west_2024),
    }
    manifests = {
        key: path.parent / "manifest.json" for key, path in paths.items()
    }
    manifest_validation = {
        key: validate_external_data_manifest(path, ROOT)
        for key, path in manifests.items()
    }
    loaded = {
        key: load_context_features(path) for key, path in paths.items()
    }
    centre_2023, _ = loaded["centre_2023"]
    centre_2024, issue_2024 = loaded["centre_2024"]
    spatial_2023 = build_spatial_features(
        centre_2023, loaded["east_2023"][0], loaded["west_2023"][0]
    )
    spatial_2024 = build_spatial_features(
        centre_2024, loaded["east_2024"][0], loaded["west_2024"][0]
    )
    labels = pd.read_csv(
        _rooted(args.labels),
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    driver = np.load(_rooted(args.driver), allow_pickle=False)
    kma_oof = np.load(_rooted(args.kma_oof), allow_pickle=False)

    validation: dict[str, Any] = {}
    cache: dict[str, np.ndarray] = {}
    promoted_targets: list[str] = []
    for target in TARGETS:
        spec = FROZEN_SPECS[target]
        capacity = CAPACITY_KWH[target]
        target_2023 = labels[target].reindex(centre_2023.index)
        centre_seed = [
            np.clip(
                fit_predict(
                    centre_2023,
                    target_2023,
                    centre_2024,
                    alpha=spec["alpha"],
                    seed=seed,
                    n_estimators=args.n_estimators,
                ),
                0.0,
                capacity,
            )
            for seed in SEEDS
        ]
        spatial_seed = [
            np.clip(
                fit_predict(
                    spatial_2023,
                    target_2023,
                    spatial_2024,
                    alpha=spec["alpha"],
                    seed=seed,
                    n_estimators=args.n_estimators,
                ),
                0.0,
                capacity,
            )
            for seed in SEEDS
        ]
        index, truth, reference = _validation_surface(
            driver, kma_oof, target
        )
        aligned_centre = [
            _align_prediction(item, centre_2024.index, index, reference)
            for item in centre_seed
        ]
        aligned_spatial = [
            _align_prediction(item, spatial_2024.index, index, reference)
            for item in spatial_seed
        ]
        centre_expert = np.mean(aligned_centre, axis=0)
        spatial_expert = np.mean(aligned_spatial, axis=0)
        incumbent = apply_bounded_blend(
            reference,
            centre_expert,
            weight=spec["weight"],
            capacity=capacity,
        )
        periods = _periods(index)
        records: list[dict[str, Any]] = []
        candidates: dict[float, np.ndarray] = {}
        for factor in REPLACEMENT_FACTORS:
            expert = interpolate_expert(
                centre_expert, spatial_expert, factor
            )
            candidate = apply_bounded_blend(
                reference,
                expert,
                weight=spec["weight"],
                capacity=capacity,
            )
            candidates[factor] = candidate
            deltas = {
                name: metric_delta(
                    truth, incumbent, candidate, capacity, rows
                )
                for name, rows in periods.items()
            }
            records.append(
                {
                    "factor": factor,
                    "period_deltas": deltas,
                    "h1_eligible": bool(
                        min(deltas["q1"].values()) >= 0.0
                        and min(deltas["q2"].values()) >= 0.0
                        and deltas["h1"]["score"] > 0.0
                    ),
                }
            )
        eligible = [row for row in records if row["h1_eligible"]]
        selected = (
            max(
                eligible,
                key=lambda row: (
                    row["period_deltas"]["h1"]["score"],
                    min(row["period_deltas"]["q1"].values()),
                    min(row["period_deltas"]["q2"].values()),
                    -row["factor"],
                ),
            )
            if eligible
            else None
        )
        # Preserve the exact frozen single-point incumbent even when the
        # spatial replacement is rejected; downstream incremental experiments
        # must not have to reconstruct this expensive validation surface.
        cache[f"{target}__index_ns"] = index.astype("int64").to_numpy()
        cache[f"{target}__incumbent"] = incumbent.astype("float32")
        cache[f"{target}__centre_expert"] = centre_expert.astype("float32")
        cache[f"{target}__spatial_expert"] = spatial_expert.astype("float32")
        if selected is None:
            validation[target] = {
                "records": records,
                "selected": None,
                "promotion": "rejected_on_h1",
            }
            continue

        factor = float(selected["factor"])
        candidate = candidates[factor]
        seed_deltas: list[dict[str, Any]] = []
        for seed, centre_item, spatial_item in zip(
            SEEDS, aligned_centre, aligned_spatial
        ):
            seed_incumbent = apply_bounded_blend(
                reference,
                centre_item,
                weight=spec["weight"],
                capacity=capacity,
            )
            seed_candidate = apply_bounded_blend(
                reference,
                interpolate_expert(centre_item, spatial_item, factor),
                weight=spec["weight"],
                capacity=capacity,
            )
            seed_deltas.append(
                {
                    "seed": seed,
                    "period_deltas": {
                        name: metric_delta(
                            truth,
                            seed_incumbent,
                            seed_candidate,
                            capacity,
                            rows,
                        )
                        for name, rows in periods.items()
                    },
                }
            )
        monthly = {
            str(month): metric_delta(
                truth,
                incumbent,
                candidate,
                capacity,
                periods["full"] & np.asarray(index.month == month),
            )
            for month in range(1, 13)
        }
        issue = issue_2024.reindex(index)
        if int(issue.loc[index < VALIDATION_END].isna().sum()):
            raise ValueError("KMA issue times are missing in the validation year")
        issue = issue.fillna(pd.Timestamp("2024-12-31 13:00")).to_numpy()
        bootstrap = evaluate_blocked_rolling(
            truth,
            incumbent,
            candidate,
            index,
            issue,
            periods["h2"] & (truth >= 0.10 * capacity),
            n_bootstrap=args.n_bootstrap,
            seed=20260726,
        )
        positive_months = int(
            sum(value["score"] > 0.0 for value in monthly.values())
        )
        h2 = selected["period_deltas"]["h2"]
        gates = {
            "h1_q1_q2_components_nonnegative": bool(
                selected["h1_eligible"]
            ),
            "h2_components_positive": min(h2.values()) > 0.0,
            "every_seed_h2_components_positive": all(
                min(row["period_deltas"]["h2"].values()) > 0.0
                for row in seed_deltas
            ),
            "positive_month_fraction_at_least_75pct": (
                positive_months / 12.0 >= 0.75
            ),
            "h2_issue_bootstrap_q05_positive": (
                bootstrap["issue_block_bootstrap"]["q05"] > 0.0
            ),
            "h2_issue_bootstrap_positive_fraction_at_least_90pct": (
                bootstrap["issue_block_bootstrap"]["positive_fraction"] >= 0.90
            ),
        }
        promoted = bool(all(gates.values()))
        if promoted:
            promoted_targets.append(target)
        cache[f"{target}__candidate"] = candidate.astype("float32")
        validation[target] = {
            "records": records,
            "selected": selected,
            "seed_stability": seed_deltas,
            "monthly_deltas": monthly,
            "positive_months": positive_months,
            "h2_issue_block_validation": bootstrap,
            "gates": gates,
            "promotion": "promoted_for_2025_collection"
            if promoted
            else "rejected",
        }

    output_cache = _rooted(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, **cache)
    report = {
        "family": "kma_umrg_east_centre_west_year_forward_screen",
        "sources": {
            key: path.relative_to(ROOT).as_posix()
            for key, path in paths.items()
        },
        "manifest_validation": manifest_validation,
        "contract": {
            "validation_expert": "all 2023 labels -> 2024",
            "frozen_alpha_weight_seed_and_movement_bound": True,
            "selection": (
                "factor 0.5 or 1.0 on H1; Q1 and Q2 must each improve "
                "score, 1-NMAE, and FICR"
            ),
            "confirmation": "H2, seeds, months, and complete issue cycles",
            "public_score_used": False,
            "test_actual_generation_used": False,
            "2025_spatial_data_collected": False,
        },
        "feature_counts": {
            "centre": int(centre_2023.shape[1]),
            "spatial": int(spatial_2023.shape[1]),
        },
        "validation": validation,
        "promoted_targets": promoted_targets,
        "production_collection_eligible": bool(promoted_targets),
        "cache": {
            "path": output_cache.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_cache),
        },
    }
    output_report = _rooted(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--kma-oof",
        default=(
            "artifacts_final/external_weather/kma_um_regional_context_2024/"
            "power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--centre-2023",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2023/features.csv"
        ),
    )
    parser.add_argument(
        "--centre-2024",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/features.csv"
        ),
    )
    for direction in ("east", "west"):
        for year in ("2023", "2024"):
            parser.add_argument(
                f"--{direction}-{year}",
                default=(
                    "artifacts_final/external_weather/"
                    f"kma_um_regional_spatial_{direction}_{year}/features.csv"
                ),
            )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--output-cache",
        default="artifacts_final/lineage/kma_spatial_context_screen_20260726.npz",
    )
    parser.add_argument(
        "--output-report",
        default=(
            "artifacts_final/diagnostics/"
            "kma_spatial_context_screen_20260726.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted_targets": report["promoted_targets"],
                "production_collection_eligible": report[
                    "production_collection_eligible"
                ],
                "validation": {
                    target: {
                        "selected": value.get("selected"),
                        "positive_months": value.get("positive_months"),
                        "gates": value.get("gates"),
                        "promotion": value["promotion"],
                    }
                    for target, value in report["validation"].items()
                },
                "report": args.output_report,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
