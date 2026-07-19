from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent_service.compliance import validate_external_data_manifest
from experiments.build_noaa_gefs_spread_features import decode_file
from src.metrics import CAPACITY_KWH, evaluate_group


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
Q1_END = pd.Timestamp("2024-04-01")
H2_START = pd.Timestamp("2024-07-01")


def decode_snapshot(
    manifest_path: Path,
    gfs_path: Path,
    project_root: Path,
    workers: int,
) -> pd.DataFrame:
    validate_external_data_manifest(manifest_path, project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("coverage", {}).get("product") != "geavg":
        raise ValueError("snapshot screen requires a GEFS geavg manifest")
    gfs = pd.read_csv(
        gfs_path,
        encoding="utf-8-sig",
        usecols=[
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "grid_id",
            "latitude",
            "longitude",
            "heightAboveGround_10_10u",
            "heightAboveGround_10_10v",
        ],
    )
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    locations = gfs[["grid_id", "latitude", "longitude"]].drop_duplicates()
    source_paths = [project_root / item["path"] for item in manifest["raw_files"]]
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(decode_file, path, locations, "mean"): path
            for path in source_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            frames.append(future.result())
            if completed % 100 == 0 or completed == len(source_paths):
                print(f"decoded_mean={completed}/{len(source_paths)}", flush=True)
    decoded = pd.concat(frames, ignore_index=True)
    decoded["forecast_kst_dtm"] = (
        decoded["initialization_utc"]
        + pd.to_timedelta(decoded["lead_hour"], unit="h")
    ).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    snapshot = decoded.merge(
        gfs,
        on=["forecast_kst_dtm", "grid_id", "latitude", "longitude"],
        how="inner",
        validate="one_to_one",
    )
    if len(snapshot) != len(decoded):
        raise ValueError("supplied GFS does not cover every GEFS snapshot grid")
    snapshot = snapshot.rename(
        columns={
            "heightAboveGround_10_10u": "gfs_u10",
            "heightAboveGround_10_10v": "gfs_v10",
        }
    )
    snapshot["delta_u10"] = snapshot["gefs_u10_mean"] - snapshot["gfs_u10"]
    snapshot["delta_v10"] = snapshot["gefs_v10_mean"] - snapshot["gfs_v10"]
    snapshot["vector_disagreement"] = np.hypot(
        snapshot["delta_u10"], snapshot["delta_v10"]
    )
    snapshot["gefs_speed10"] = np.hypot(
        snapshot["gefs_u10_mean"], snapshot["gefs_v10_mean"]
    )
    snapshot["gfs_speed10"] = np.hypot(snapshot["gfs_u10"], snapshot["gfs_v10"])
    snapshot["speed_disagreement"] = (
        snapshot["gefs_speed10"] - snapshot["gfs_speed10"]
    )
    return snapshot.sort_values(["forecast_kst_dtm", "grid_id"]).reset_index(drop=True)


def daily_oof(driver_path: Path, meta_path: Path) -> pd.DataFrame:
    driver = np.load(driver_path)
    meta = np.load(meta_path)
    index = pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    if not index.equals(driver_index):
        raise ValueError("driver and meta OOF indexes differ")
    truth = driver[f"{TARGET}__valid_truth"].astype(float)
    base = meta["valid_candidate"].astype(float)
    frame = pd.DataFrame({"truth": truth, "base": base}, index=index)
    frame = frame[(frame.index >= "2024-01-01") & (frame.index < "2025-01-01")]
    rows = []
    for day, block in frame.groupby(frame.index.normalize()):
        eligible = block["truth"].to_numpy() >= 0.10 * CAPACITY
        truth_day = block["truth"].to_numpy()[eligible]
        base_day = block["base"].to_numpy()[eligible]
        if not len(truth_day):
            continue
        metric = evaluate_group(truth_day, base_day, CAPACITY)
        rows.append(
            {
                "date": day,
                "eligible_rows": int(len(truth_day)),
                "mean_abs_error_ratio": float(
                    np.mean(np.abs(truth_day - base_day) / CAPACITY)
                ),
                "weighted_signed_residual_ratio": float(
                    np.average((truth_day - base_day) / CAPACITY, weights=truth_day)
                ),
                "ficr_loss": 1.0 - metric.ficr,
                "score": metric.score,
            }
        )
    return pd.DataFrame(rows).set_index("date").sort_index()


def aggregate_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    snapshot = snapshot.copy()
    snapshot["date"] = snapshot["forecast_kst_dtm"].dt.normalize()
    columns = [
        "vector_disagreement",
        "speed_disagreement",
        "delta_u10",
        "delta_v10",
        "gefs_speed10",
        "gfs_speed10",
    ]
    result = snapshot.groupby("date")[columns].agg(["mean", "max", "min", "std"])
    result.columns = [f"{column}_{stat}" for column, stat in result.columns]
    result["speed_disagreement_abs_mean"] = snapshot.groupby("date")[
        "speed_disagreement"
    ].apply(lambda value: float(np.abs(value).mean()))
    return result


def correlations(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    joined = features.join(outcomes, how="inner")
    selected = joined.loc[mask]
    targets = ["mean_abs_error_ratio", "weighted_signed_residual_ratio", "ficr_loss"]
    return {
        feature: {
            target: float(selected[feature].corr(selected[target], method="spearman"))
            for target in targets
        }
        for feature in features.columns
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts_final/external_weather/noaa_gefs_mean_f33_2024/manifest.json",
    )
    parser.add_argument("--gfs", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument(
        "--snapshot-output",
        default="artifacts_final/external_weather/noaa_gefs_mean_f33_2024/snapshot.csv",
    )
    parser.add_argument(
        "--report-output",
        default="artifacts_final/external_weather/noaa_gefs_mean_f33_2024/screen.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")

    root = Path.cwd().resolve()
    snapshot = decode_snapshot(Path(args.manifest), Path(args.gfs), root, args.workers)
    snapshot_output = Path(args.snapshot_output)
    snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(snapshot_output, index=False, encoding="utf-8-sig")
    features = aggregate_snapshot(snapshot)
    outcomes = daily_oof(Path(args.driver_cache), Path(args.meta_cache))
    common = features.index.intersection(outcomes.index)
    features = features.reindex(common)
    outcomes = outcomes.reindex(common)
    q1 = np.asarray(common < Q1_END)
    q2 = np.asarray((common >= Q1_END) & (common < H2_START))
    q1_corr = correlations(features, outcomes, q1)
    q2_corr = correlations(features, outcomes, q2)

    stable = []
    for feature in features.columns:
        for target in (
            "mean_abs_error_ratio",
            "weighted_signed_residual_ratio",
            "ficr_loss",
        ):
            first = q1_corr[feature][target]
            second = q2_corr[feature][target]
            if (
                np.isfinite(first)
                and np.isfinite(second)
                and first * second > 0.0
                and abs(first) >= 0.12
                and abs(second) >= 0.08
            ):
                stable.append(
                    {
                        "feature": feature,
                        "target": target,
                        "q1_spearman": first,
                        "q2_spearman": second,
                        "minimum_abs_spearman": min(abs(first), abs(second)),
                    }
                )
    stable.sort(key=lambda item: item["minimum_abs_spearman"], reverse=True)
    report = {
        "family": "gefs_mean_gfs_cycle_disagreement_f33_screen",
        "source_manifest": args.manifest,
        "snapshot_rows": int(len(snapshot)),
        "days": int(len(common)),
        "snapshot_sha256": hashlib.sha256(snapshot_output.read_bytes()).hexdigest(),
        "q1_correlations": q1_corr,
        "q2_correlations": q2_corr,
        "stable_signals": stable,
        "decision": {
            "expand_to_full_leads": bool(stable),
            "reason": (
                "at least one disagreement signal is directionally stable"
                if stable
                else "no disagreement signal met the preregistered Q1/Q2 stability gate"
            ),
        },
    }
    report_path = Path(args.report_output)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
