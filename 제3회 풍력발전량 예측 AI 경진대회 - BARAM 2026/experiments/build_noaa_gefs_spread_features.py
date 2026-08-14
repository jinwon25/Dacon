from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from eccodes import (
    codes_get,
    codes_grib_find_nearest,
    codes_grib_new_from_file,
    codes_release,
)

from agent_service.compliance import validate_external_data_manifest


def _idw(neighbours: tuple[dict[str, Any], ...]) -> float:
    distances = np.asarray([item["distance"] for item in neighbours], dtype=float)
    values = np.asarray([item["value"] for item in neighbours], dtype=float)
    exact = distances <= 1e-9
    if exact.any():
        return float(values[exact].mean())
    weights = 1.0 / np.square(distances)
    return float(np.dot(values, weights) / weights.sum())


def decode_file(
    path: Path,
    locations: pd.DataFrame,
    statistic: str = "spread",
) -> pd.DataFrame:
    if statistic not in {"mean", "spread"}:
        raise ValueError("statistic must be mean or spread")
    records: dict[tuple[pd.Timestamp, int, int], dict[str, Any]] = {}
    with path.open("rb") as stream:
        while True:
            handle = codes_grib_new_from_file(stream)
            if handle is None:
                break
            try:
                short_name = str(codes_get(handle, "shortName"))
                if short_name not in {"10u", "10v"}:
                    raise ValueError(f"unexpected GEFS field in {path}: {short_name}")
                initialization = pd.Timestamp(
                    str(codes_get(handle, "dataDate"))
                    + f" {int(codes_get(handle, 'dataTime')):04d}",
                    tz="UTC",
                )
                lead = int(codes_get(handle, "step"))
                units = str(codes_get(handle, "units"))
                if units != "m s**-1":
                    raise ValueError(f"unexpected GEFS wind unit: {units}")
                component = "u" if short_name == "10u" else "v"
                column = f"gefs_{component}10_{statistic}"
                for row in locations.itertuples(index=False):
                    neighbours = codes_grib_find_nearest(
                        handle,
                        float(row.latitude),
                        float(row.longitude),
                        npoints=4,
                    )
                    key = (initialization, lead, int(row.grid_id))
                    record = records.setdefault(
                        key,
                        {
                            "initialization_utc": initialization,
                            "lead_hour": lead,
                            "grid_id": int(row.grid_id),
                            "latitude": float(row.latitude),
                            "longitude": float(row.longitude),
                        },
                    )
                    record[column] = _idw(neighbours)
            finally:
                codes_release(handle)
    decoded = pd.DataFrame(records.values())
    required = {f"gefs_u10_{statistic}", f"gefs_v10_{statistic}"}
    if decoded.empty or not required.issubset(decoded.columns):
        raise ValueError(f"decoded GEFS file is incomplete: {path}")
    return decoded


def interpolate_to_targets(
    metadata: pd.DataFrame,
    join_plan: pd.DataFrame,
    decoded: pd.DataFrame,
    statistic: str = "spread",
) -> pd.DataFrame:
    if statistic not in {"mean", "spread"}:
        raise ValueError("statistic must be mean or spread")
    keys = ["forecast_kst_dtm", "data_available_kst_dtm"]
    metadata = metadata[
        keys + ["grid_id", "latitude", "longitude"]
    ].drop_duplicates()
    for column in keys:
        metadata[column] = pd.to_datetime(metadata[column])
        join_plan[column] = pd.to_datetime(join_plan[column])
    join_plan["initialization_utc"] = pd.to_datetime(
        join_plan["initialization_utc"], utc=True
    )
    decoded["initialization_utc"] = pd.to_datetime(
        decoded["initialization_utc"], utc=True
    )
    plan_keys = join_plan[keys].drop_duplicates()
    covered_metadata = metadata.merge(plan_keys, on=keys, how="inner")
    target = covered_metadata.merge(
        join_plan, on=keys, how="left", validate="many_to_one"
    )
    expected_rows = len(plan_keys) * metadata["grid_id"].nunique()
    if len(target) != expected_rows:
        raise ValueError("GEFS join plan has missing or duplicate target/grid rows")

    value_columns = [f"gefs_u10_{statistic}", f"gefs_v10_{statistic}"]
    for bound in ("lower", "upper"):
        source = decoded[
            ["initialization_utc", "lead_hour", "grid_id"] + value_columns
        ].rename(
            columns={
                "lead_hour": f"{bound}_lead_hour",
                **{column: f"{bound}_{column}" for column in value_columns},
            }
        )
        target = target.merge(
            source,
            on=["initialization_utc", f"{bound}_lead_hour", "grid_id"],
            how="left",
            validate="many_to_one",
        )
    if target.filter(regex=r"^(lower|upper)_gefs_").isna().any().any():
        raise ValueError("decoded GEFS spread is missing a target/grid bracket")
    weight = target["upper_weight"].to_numpy(float)
    output = target[keys + ["grid_id", "latitude", "longitude"]].copy()
    for column in value_columns:
        lower = target[f"lower_{column}"].to_numpy(float)
        upper = target[f"upper_{column}"].to_numpy(float)
        output[column] = lower * (1.0 - weight) + upper * weight
    output[f"gefs_uv10_{statistic}_norm"] = np.hypot(
        output[f"gefs_u10_{statistic}"], output[f"gefs_v10_{statistic}"]
    )
    return output.sort_values(["forecast_kst_dtm", "grid_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts_final/external_weather/noaa_gefs_pilot/manifest.json",
    )
    parser.add_argument(
        "--join-plan",
        default="artifacts_final/external_weather/noaa_gefs_pilot/join_plan.csv",
    )
    parser.add_argument("--forecast-metadata", default="data/test/gfs_test.csv")
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/noaa_gefs_pilot/features.csv",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--statistic", choices=("mean", "spread"), default="spread")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")

    project_root = Path.cwd().resolve()
    manifest_path = Path(args.manifest)
    validate_external_data_manifest(manifest_path, project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(
        args.forecast_metadata,
        encoding="utf-8-sig",
        usecols=[
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "grid_id",
            "latitude",
            "longitude",
        ],
    )
    locations = metadata[["grid_id", "latitude", "longitude"]].drop_duplicates()
    source_paths = [project_root / item["path"] for item in manifest["raw_files"]]
    frames = []
    # ecCodes owns process-global definition state and is not thread-safe during
    # concurrent initialization on Windows. Independent worker processes avoid
    # that race while decoding separate GRIB messages.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(decode_file, source_path, locations, args.statistic): source_path
            for source_path in source_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            frames.append(future.result())
            if completed % 500 == 0 or completed == len(source_paths):
                print(f"decoded={completed}/{len(source_paths)}", flush=True)
    decoded = pd.concat(frames, ignore_index=True)
    join_plan = pd.read_csv(args.join_plan, encoding="utf-8-sig")
    features = interpolate_to_targets(
        metadata, join_plan, decoded, statistic=args.statistic
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    features.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(output)
    report = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "forecast_metadata": args.forecast_metadata,
        "join_plan": args.join_plan,
        "rows": int(len(features)),
        "targets": int(features["forecast_kst_dtm"].nunique()),
        "grids": int(features["grid_id"].nunique()),
        "columns": list(features.columns),
        "missing_values": int(features.isna().sum().sum()),
        "ranges": {
            column: {
                "min": float(features[column].min()),
                "max": float(features[column].max()),
            }
            for column in (
                f"gefs_u10_{args.statistic}",
                f"gefs_v10_{args.statistic}",
                f"gefs_uv10_{args.statistic}_norm",
            )
        },
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
