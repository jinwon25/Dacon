from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import pandas as pd
from eccodes import (
    codes_get,
    codes_grib_find_nearest,
    codes_grib_new_from_file,
    codes_release,
)

from agent_service.compliance import validate_external_data_manifest
from experiments.build_noaa_gefs_spread_features import _idw


def decode_file(path: Path, locations: pd.DataFrame) -> pd.DataFrame:
    records: dict[tuple[pd.Timestamp, int, int], dict[str, Any]] = {}
    with path.open("rb") as stream:
        while True:
            handle = codes_grib_new_from_file(stream)
            if handle is None:
                break
            try:
                short_name = str(codes_get(handle, "shortName"))
                if short_name not in {"10u", "10v"}:
                    raise ValueError(f"unexpected CFSv2 field in {path}: {short_name}")
                initialization = pd.Timestamp(
                    str(codes_get(handle, "dataDate"))
                    + f" {int(codes_get(handle, 'dataTime')):04d}",
                    tz="UTC",
                )
                lead = int(codes_get(handle, "step"))
                units = str(codes_get(handle, "units"))
                if units != "m s**-1":
                    raise ValueError(f"unexpected CFSv2 wind unit: {units}")
                column = "cfsv2_u10" if short_name == "10u" else "cfsv2_v10"
                for location in locations.itertuples(index=False):
                    neighbours = codes_grib_find_nearest(
                        handle,
                        float(location.latitude),
                        float(location.longitude),
                        npoints=4,
                    )
                    key = (initialization, lead, int(location.grid_id))
                    record = records.setdefault(
                        key,
                        {
                            "initialization_utc": initialization,
                            "lead_hour": lead,
                            "grid_id": int(location.grid_id),
                            "latitude": float(location.latitude),
                            "longitude": float(location.longitude),
                        },
                    )
                    record[column] = _idw(neighbours)
            finally:
                codes_release(handle)
    decoded = pd.DataFrame(records.values())
    if decoded.empty or not {"cfsv2_u10", "cfsv2_v10"}.issubset(decoded.columns):
        raise ValueError(f"decoded CFSv2 file is incomplete: {path}")
    if decoded[["cfsv2_u10", "cfsv2_v10"]].isna().any().any():
        raise ValueError(f"decoded CFSv2 file contains missing wind: {path}")
    return decoded


def interpolate_to_targets(
    metadata: pd.DataFrame, join_plan: pd.DataFrame, decoded: pd.DataFrame
) -> pd.DataFrame:
    keys = ["forecast_kst_dtm", "data_available_kst_dtm"]
    metadata = metadata[keys + ["grid_id", "latitude", "longitude"]].drop_duplicates()
    for column in keys:
        metadata[column] = pd.to_datetime(metadata[column])
        join_plan[column] = pd.to_datetime(join_plan[column])
    join_plan["initialization_utc"] = pd.to_datetime(
        join_plan["initialization_utc"], utc=True
    )
    decoded["initialization_utc"] = pd.to_datetime(
        decoded["initialization_utc"], utc=True
    )
    decoded_keys = ["initialization_utc", "lead_hour", "grid_id"]
    duplicated = decoded.duplicated(decoded_keys, keep=False)
    if duplicated.any():
        spreads = decoded.loc[duplicated].groupby(decoded_keys)[
            ["cfsv2_u10", "cfsv2_v10"]
        ].agg(lambda values: float(values.max() - values.min()))
        if (spreads.to_numpy(dtype=float) > 1e-9).any():
            raise ValueError("duplicate CFSv2 messages disagree on decoded wind")
        decoded = decoded.drop_duplicates(decoded_keys, keep="first")
    plan_keys = join_plan[keys].drop_duplicates()
    target = metadata.merge(plan_keys, on=keys, how="inner").merge(
        join_plan, on=keys, how="left", validate="many_to_one"
    )
    expected = len(plan_keys) * metadata["grid_id"].nunique()
    if len(target) != expected:
        raise ValueError("CFSv2 join plan has missing or duplicate target/grid rows")
    values = ["cfsv2_u10", "cfsv2_v10"]
    for bound in ("lower", "upper"):
        source = decoded[
            ["initialization_utc", "lead_hour", "grid_id"] + values
        ].rename(
            columns={
                "lead_hour": f"{bound}_lead_hour",
                **{value: f"{bound}_{value}" for value in values},
            }
        )
        target = target.merge(
            source,
            on=["initialization_utc", f"{bound}_lead_hour", "grid_id"],
            how="left",
            validate="many_to_one",
        )
    if target.filter(regex=r"^(lower|upper)_cfsv2_").isna().any().any():
        raise ValueError("decoded CFSv2 wind is missing a target/grid lead bracket")
    output = target[keys + ["grid_id", "latitude", "longitude"]].copy()
    weight = target["upper_weight"].to_numpy(dtype=float)
    for value in values:
        lower = target[f"lower_{value}"].to_numpy(dtype=float)
        upper = target[f"upper_{value}"].to_numpy(dtype=float)
        output[value] = lower * (1.0 - weight) + upper * weight
    output["cfsv2_speed10"] = np.hypot(output["cfsv2_u10"], output["cfsv2_v10"])
    return output.sort_values(["forecast_kst_dtm", "grid_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts_final/external_weather/noaa_cfsv2_2024/manifest.json",
    )
    parser.add_argument(
        "--join-plan",
        default="artifacts_final/external_weather/noaa_cfsv2_2024/join_plan.csv",
    )
    parser.add_argument("--forecast-metadata", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/noaa_cfsv2_2024/features.csv",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    root = Path.cwd().resolve()
    validate_external_data_manifest(Path(args.manifest), root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
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
    paths = [root / item["path"] for item in manifest["raw_files"]]
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(decode_file, path, locations): path for path in paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            frames.append(future.result())
            if completed % 50 == 0 or completed == len(paths):
                print(f"decoded={completed}/{len(paths)}", flush=True)
    decoded = pd.concat(frames, ignore_index=True)
    join_plan = pd.read_csv(args.join_plan, encoding="utf-8-sig")
    features = interpolate_to_targets(metadata, join_plan, decoded)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    features.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(output)
    report = {
        "source_manifest": args.manifest,
        "source_manifest_sha256": hashlib.sha256(
            Path(args.manifest).read_bytes()
        ).hexdigest(),
        "forecast_metadata": args.forecast_metadata,
        "join_plan": args.join_plan,
        "rows": int(len(features)),
        "targets": int(features["forecast_kst_dtm"].nunique()),
        "grids": int(features["grid_id"].nunique()),
        "missing_values": int(features.isna().sum().sum()),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
