from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH


FORECAST_COL = "forecast_kst_dtm"
AVAILABLE_COL = "data_available_kst_dtm"
TIMEZONE = "Asia/Seoul"

SCADA_GROUPS = {
    "kpx_group_1": {"file": "scada_vestas_train.csv", "prefix": "vestas", "ids": range(1, 7), "rated_kw": 3_600.0},
    "kpx_group_2": {"file": "scada_vestas_train.csv", "prefix": "vestas", "ids": range(7, 13), "rated_kw": 3_600.0},
    "kpx_group_3": {"file": "scada_unison_train.csv", "prefix": "unison", "ids": range(1, 6), "rated_kw": 4_200.0},
}


def prediction_day(forecast: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the competition's 01:00--next-day 00:00 forecast day."""
    index = pd.DatetimeIndex(forecast)
    day = index.normalize()
    return day - pd.to_timedelta((index.hour == 0).astype(int), unit="D")


def day_ahead_cutoff(forecast: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Previous forecast-day 14:00 KST, represented as timezone-naive KST."""
    return prediction_day(forecast) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)


def _iso(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(_iso(k)): int(v) for k, v in values.value_counts(dropna=False).sort_index().items()}


def audit_weather(path: Path, source: str, split: str) -> dict[str, Any]:
    cols = [FORECAST_COL, AVAILABLE_COL, "grid_id", "latitude", "longitude"]
    frame = pd.read_csv(path, usecols=cols, encoding="utf-8-sig")
    for col in (FORECAST_COL, AVAILABLE_COL):
        frame[col] = pd.to_datetime(frame[col], errors="raise")

    forecast = pd.DatetimeIndex(frame[FORECAST_COL])
    available = pd.DatetimeIndex(frame[AVAILABLE_COL])
    cutoff = day_ahead_cutoff(forecast)
    lead = (forecast - available).total_seconds() / 3_600.0
    margin = (cutoff - available).total_seconds() / 3_600.0
    legal = available <= cutoff

    pair_cols = [FORECAST_COL, "grid_id"]
    exact_cols = [FORECAST_COL, "grid_id", AVAILABLE_COL]
    cycles_per_pair = frame.groupby(pair_cols, sort=False)[AVAILABLE_COL].nunique()
    legal_frame = frame.loc[legal].copy()
    latest_legal = legal_frame.groupby(pair_cols, sort=False)[AVAILABLE_COL].transform("max")
    stale_legal = legal_frame[AVAILABLE_COL] < latest_legal
    expected_grid_count = int(frame["grid_id"].nunique())
    grids_per_forecast = frame.groupby(FORECAST_COL, sort=False)["grid_id"].nunique()

    return {
        "source": source,
        "split": split,
        "path": path.as_posix(),
        "rows": int(len(frame)),
        "forecast_range": [_iso(frame[FORECAST_COL].min()), _iso(frame[FORECAST_COL].max())],
        "data_available_range": [_iso(frame[AVAILABLE_COL].min()), _iso(frame[AVAILABLE_COL].max())],
        "timestamp_storage": "timezone-naive",
        "documented_semantic_timezone": TIMEZONE,
        "grid_count": expected_grid_count,
        "grid_ids": sorted(int(v) for v in frame["grid_id"].unique()),
        "grid_rows_per_forecast": _counts(grids_per_forecast),
        "duplicate_forecast_grid_available_rows": int(frame.duplicated(exact_cols, keep=False).sum()),
        "duplicate_forecast_grid_rows": int(frame.duplicated(pair_cols, keep=False).sum()),
        "forecast_grid_pairs_with_multiple_cycles": int((cycles_per_pair > 1).sum()),
        "post_cutoff_rows": int((~legal).sum()),
        "forecast_grid_pairs_without_legal_cycle": int(len(cycles_per_pair) - legal_frame.groupby(pair_cols).ngroups),
        "stale_legal_rows_if_latest_cycle_selected": int(stale_legal.sum()),
        "availability_margin_hours": {
            "min": float(np.min(margin)),
            "median": float(np.median(margin)),
            "max": float(np.max(margin)),
        },
        "available_cycle_hour_counts": _counts(frame[AVAILABLE_COL].dt.hour),
        "lead_hour_counts": _counts(pd.Series(lead).round(6)),
        "lead_hour_range": [float(np.min(lead)), float(np.max(lead))],
        "forecast_hour_counts": _counts(frame[FORECAST_COL].dt.hour),
        "all_rows_cutoff_legal": bool(legal.all()),
        "all_forecasts_have_expected_grid_count": bool((grids_per_forecast == expected_grid_count).all()),
    }


def audit_labels(path: Path) -> dict[str, Any]:
    labels = pd.read_csv(path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"], errors="raise")
    result: dict[str, Any] = {
        "path": path.as_posix(),
        "rows": int(len(labels)),
        "timestamp_range": [_iso(labels["kst_dtm"].min()), _iso(labels["kst_dtm"].max())],
        "timestamp_storage": "timezone-naive",
        "documented_semantic_timezone": TIMEZONE,
        "duplicate_timestamps": int(labels["kst_dtm"].duplicated(keep=False).sum()),
        "groups": {},
    }
    for target, capacity in CAPACITY_KWH.items():
        values = labels[target]
        provided = values.notna()
        eligible = provided & (values >= 0.10 * capacity)
        result["groups"][target] = {
            "capacity_kwh": capacity,
            "provided_rows": int(provided.sum()),
            "provided_range": [
                _iso(labels.loc[provided, "kst_dtm"].min()),
                _iso(labels.loc[provided, "kst_dtm"].max()),
            ],
            "missing_rows": int(values.isna().sum()),
            "missing_rate_all_timestamps": float(values.isna().mean()),
            "eligible_rows": int(eligible.sum()),
            "eligible_rate_of_provided": float(eligible.sum() / provided.sum()),
            "negative_rows": int((values < 0).sum()),
            "capacity_exceed_rows": int((values > capacity).sum()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def _dms_to_decimal(value: str) -> tuple[float, float]:
    match = re.fullmatch(
        r"\s*(\d+)[°º](\d+)'([0-9.]+)\"([NS])\s+(\d+)[°º](\d+)'([0-9.]+)\"([EW])\s*",
        str(value),
    )
    if match is None:
        raise ValueError(f"Cannot parse DMS coordinate: {value!r}")
    lat_d, lat_m, lat_s, lat_h, lon_d, lon_m, lon_s, lon_h = match.groups()
    lat = float(lat_d) + float(lat_m) / 60.0 + float(lat_s) / 3_600.0
    lon = float(lon_d) + float(lon_m) / 60.0 + float(lon_s) / 3_600.0
    return (-lat if lat_h == "S" else lat, -lon if lon_h == "W" else lon)


def audit_info(path: Path) -> dict[str, Any]:
    raw = pd.read_excel(path, sheet_name="info", header=None)
    header_matches = raw.index[raw.apply(lambda row: row.astype(str).eq("KPX그룹").any(), axis=1)]
    if len(header_matches) != 1:
        raise ValueError(f"Expected one info.xlsx header row containing KPX그룹, found {len(header_matches)}")
    info = pd.read_excel(path, sheet_name="info", header=int(header_matches[0]))
    info = info.dropna(how="all").dropna(axis=1, how="all")
    info["KPX그룹"] = info["KPX그룹"].ffill().astype(int)
    info["그룹설비용량(MW)"] = info["그룹설비용량(MW)"].ffill()
    records = []
    for row in info.itertuples(index=False):
        record = dict(zip(info.columns, row))
        lat, lon = _dms_to_decimal(str(record["좌표(Google)"]))
        records.append(
            {
                "group": int(record["KPX그룹"]),
                "manufacturer": str(record["제작사"]),
                "model": str(record["모델명"]),
                "turbine_id": int(record["호기"]),
                "latitude": lat,
                "longitude": lon,
                "turbine_capacity_mw": float(record["설비용량(MW)"]),
                "group_capacity_mw": float(record["그룹설비용량(MW)"]),
            }
        )
    mapping = pd.DataFrame(records)
    group_summary = []
    for group, part in mapping.groupby("group"):
        group_summary.append(
            {
                "group": int(group),
                "manufacturer": sorted(part["manufacturer"].unique().tolist()),
                "models": sorted(part["model"].unique().tolist()),
                "turbines": int(len(part)),
                "turbine_capacity_sum_mw": float(part["turbine_capacity_mw"].sum()),
                "declared_group_capacity_mw": float(part["group_capacity_mw"].iloc[0]),
                "latitude_range": [float(part["latitude"].min()), float(part["latitude"].max())],
                "longitude_range": [float(part["longitude"].min()), float(part["longitude"].max())],
            }
        )
    return {"path": path.as_posix(), "rows": len(records), "groups": group_summary, "turbines": records}


def _clean_scada_power(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where((numeric >= 0) & (numeric < 10_000))


def _hourly_power_candidates(
    frame: pd.DataFrame,
    power_cols: list[str],
) -> dict[str, pd.Series]:
    total = pd.concat([_clean_scada_power(frame[col]) for col in power_cols], axis=1).sum(axis=1, min_count=len(power_cols))
    timestamp = frame["kst_dtm"]
    conventions = {
        "floor": timestamp.dt.floor("h"),
        "ceil": timestamp.dt.ceil("h"),
        "floor_plus_1h": timestamp.dt.floor("h") + pd.Timedelta(hours=1),
    }
    result: dict[str, pd.Series] = {}
    for convention, bucket in conventions.items():
        tmp = pd.DataFrame({"bucket": bucket, "value": total, "source_time": timestamp})
        grouped = tmp.groupby("bucket", sort=True)
        count = grouped["source_time"].nunique()
        sum_value = grouped["value"].sum(min_count=6).where(count == 6)
        mean_value = grouped["value"].mean().where(count == 6)
        result[f"{convention}:sum"] = sum_value
        result[f"{convention}:mean"] = mean_value
        result[f"{convention}:kw_to_kwh"] = sum_value / 6.0
    return result


def _alignment_metrics(series: pd.Series, truth: pd.Series, capacity: float) -> list[dict[str, Any]]:
    rows = []
    for lag in range(-6, 7):
        shifted = series.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=lag)
        aligned = pd.concat([truth.rename("truth"), shifted.rename("scada")], axis=1).dropna()
        if len(aligned) < 2:
            continue
        pred = aligned["scada"].to_numpy(dtype=float)
        actual = aligned["truth"].to_numpy(dtype=float)
        nonzero = np.abs(pred) > 1e-9
        rows.append(
            {
                "lag_hours": lag,
                "rows": int(len(aligned)),
                "correlation": float(np.corrcoef(actual, pred)[0, 1]),
                "normalized_mae": float(np.mean(np.abs(actual - pred)) / capacity),
                "bias_kwh": float(np.mean(pred - actual)),
                "median_actual_to_scada_ratio": float(np.median(actual[nonzero] / pred[nonzero])) if nonzero.any() else None,
            }
        )
    return rows


def audit_scada(data_dir: Path, labels_path: Path) -> dict[str, Any]:
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"], errors="raise")
    labels = labels.set_index("kst_dtm")
    frames: dict[str, pd.DataFrame] = {}
    files: dict[str, Any] = {}

    for file_name in sorted({str(spec["file"]) for spec in SCADA_GROUPS.values()}):
        frame = pd.read_csv(data_dir / "train" / file_name, encoding="utf-8-sig")
        frame["kst_dtm"] = pd.to_datetime(frame["kst_dtm"], errors="raise")
        frames[file_name] = frame
        numeric_cols = [c for c in frame.columns if c != "kst_dtm"]
        power_cols = [c for c in numeric_cols if "_power_" in c]
        wind_cols = [c for c in numeric_cols if c.endswith("_ws")]
        direction_cols = [c for c in numeric_cols if c.endswith("_wd")]
        diffs = frame["kst_dtm"].sort_values().diff().dropna().dt.total_seconds().div(60)
        files[file_name] = {
            "rows": int(len(frame)),
            "timestamp_range": [_iso(frame["kst_dtm"].min()), _iso(frame["kst_dtm"].max())],
            "timestamp_storage": "timezone-naive",
            "documented_semantic_timezone": TIMEZONE,
            "duplicate_timestamps": int(frame["kst_dtm"].duplicated(keep=False).sum()),
            "interval_minutes_counts": _counts(diffs),
            "power_columns": len(power_cols),
            "wind_speed_columns": len(wind_cols),
            "wind_direction_columns": len(direction_cols),
            "missing_values": int(frame[numeric_cols].isna().sum().sum()),
            "sentinel_le_minus_900": int((frame[numeric_cols] <= -900).sum().sum()),
            "negative_power_values": int((frame[power_cols] < 0).sum().sum()),
            "negative_wind_speed_values": int((frame[wind_cols] < 0).sum().sum()),
            "direction_outside_minus360_360": int(((frame[direction_cols] < -360) | (frame[direction_cols] > 360)).sum().sum()),
        }

    groups: dict[str, Any] = {}
    for target, spec in SCADA_GROUPS.items():
        frame = frames[str(spec["file"])]
        prefix = str(spec["prefix"])
        turbine_ids = list(spec["ids"])
        rated_kw = float(spec["rated_kw"])
        power_cols = [f"{prefix}_wtg{i:02d}_power_kw10m" for i in turbine_ids]
        ws_cols = [f"{prefix}_wtg{i:02d}_ws" for i in turbine_ids]
        power = frame[power_cols].apply(pd.to_numeric, errors="coerce")
        wind = frame[ws_cols].apply(pd.to_numeric, errors="coerce")
        plausible_interval_max = rated_kw / 6.0
        power_values = power.to_numpy(dtype=float)
        wind_values = wind.to_numpy(dtype=float)
        finite_pair = np.isfinite(power_values) & np.isfinite(wind_values)
        non_corrupt_power = np.abs(power_values) < 10_000
        stopped = finite_pair & non_corrupt_power & (power_values <= 0.01 * plausible_interval_max) & (wind_values >= 5.0)
        alignment: dict[str, Any] = {}
        for name, series in _hourly_power_candidates(frame, power_cols).items():
            rows = _alignment_metrics(series, labels[target], CAPACITY_KWH[target])
            alignment[name] = rows
        all_rows = [(name, row) for name, values in alignment.items() for row in values]
        # Correlation is scale invariant. Round away multiplication noise, then
        # prefer the candidate whose absolute scale is also closest to KPX.
        best_corr_name, best_corr = max(
            all_rows,
            key=lambda item: (round(item[1]["correlation"], 12), -item[1]["normalized_mae"]),
        )
        best_mae_name, best_mae = min(all_rows, key=lambda item: item[1]["normalized_mae"])
        groups[target] = {
            "source_file": str(spec["file"]),
            "manufacturer": prefix,
            "turbine_ids": turbine_ids,
            "rated_kw_each": rated_kw,
            "power_missing_values": int(power.isna().sum().sum()),
            "power_sentinel_le_minus_900": int((power <= -900).sum().sum()),
            "negative_power_values": int((power < 0).sum().sum()),
            "small_negative_power_values": int(((power < 0) & (power > -100)).sum().sum()),
            "corrupt_abs_power_ge_10000": int((power.abs() >= 10_000).sum().sum()),
            "above_rated_kw_values": int((power > rated_kw).sum().sum()),
            "above_10min_energy_equivalent_105pct": int((power > 1.05 * plausible_interval_max).sum().sum()),
            "stopped_or_curtailed_candidates_ws_ge_5": int(stopped.sum()),
            "stopped_or_curtailed_rate": float(stopped.sum() / finite_pair.sum()),
            "alignment_candidates": alignment,
            "best_cross_correlation": {"candidate": best_corr_name, **best_corr},
            "best_absolute_scale_match": {"candidate": best_mae_name, **best_mae},
        }
    return {"files": files, "groups": groups}


def run_data_audit(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    weather = []
    for split in ("train", "test"):
        for source in ("ldaps", "gfs"):
            weather.append(audit_weather(data_dir / split / f"{source}_{split}.csv", source, split))
    labels_path = data_dir / "train" / "train_labels.csv"
    return {
        "metric_contract": {
            "eligibility": "actual >= 0.10 * group capacity",
            "ficr_weighting": "actual generation weighted",
            "settlement_unit_price": {"error_le_0.06": 4.0, "error_le_0.08": 3.0, "otherwise": 0.0},
        },
        "cutoff_contract": {
            "timezone": TIMEZONE,
            "forecast_day": "01:00 through next-day 00:00",
            "cutoff": "previous forecast-day 14:00 KST",
            "selection": "latest data_available_kst_dtm at or before cutoff for each forecast/grid",
        },
        "weather": weather,
        "labels": audit_labels(labels_path),
        "info": audit_info(data_dir / "info.xlsx"),
        "scada": audit_scada(data_dir, labels_path),
    }


def write_audit(report: dict[str, Any], output_json: Path, output_markdown: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# P0 Leakage and data-availability audit",
        "",
        "Timestamps are stored without timezone offsets but are documented and interpreted as Asia/Seoul (KST).",
        "The forecast day is the 24-hour block from 01:00 through next-day 00:00; its legal cutoff is the prior day at 14:00 KST.",
        "",
        "## Weather availability",
        "",
        "| Source | Split | Rows | Forecast range | Availability range | Grids | Exact duplicates | Multi-cycle pairs | Post-cutoff rows | Lead hours | Margin h |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in report["weather"]:
        lines.append(
            f"| {row['source']} | {row['split']} | {row['rows']:,} | {row['forecast_range'][0]} – {row['forecast_range'][1]} "
            f"| {row['data_available_range'][0]} – {row['data_available_range'][1]} | {row['grid_count']} "
            f"| {row['duplicate_forecast_grid_available_rows']} | {row['forecast_grid_pairs_with_multiple_cycles']} "
            f"| {row['post_cutoff_rows']} | {row['lead_hour_range'][0]:g}–{row['lead_hour_range'][1]:g} "
            f"| {row['availability_margin_hours']['min']:g}–{row['availability_margin_hours']['max']:g} |"
        )

    lines += ["", "## Labels", "", "| Group | Provided range | Provided | Missing | Eligible >=10% | Eligible rate | Negative | Above capacity |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for target, row in report["labels"]["groups"].items():
        lines.append(
            f"| {target} | {row['provided_range'][0]} – {row['provided_range'][1]} | {row['provided_rows']:,} "
            f"| {row['missing_rows']:,} | {row['eligible_rows']:,} | {row['eligible_rate_of_provided']:.2%} "
            f"| {row['negative_rows']} | {row['capacity_exceed_rows']} |"
        )

    lines += ["", "## Turbine mapping", "", "| Group | Manufacturer/model | Turbines | Capacity sum MW | Declared MW |", "|---:|---|---:|---:|---:|"]
    for row in report["info"]["groups"]:
        lines.append(
            f"| {row['group']} | {', '.join(row['manufacturer'])} / {', '.join(row['models'])} | {row['turbines']} "
            f"| {row['turbine_capacity_sum_mw']:g} | {row['declared_group_capacity_mw']:g} |"
        )

    lines += ["", "## SCADA alignment", "", "`sum` means summing six 10-minute per-turbine values into hourly energy. `kw_to_kwh` is the competing divide-by-six interpretation.", "", "| Group | Best correlation candidate | Lag h | Corr | Best absolute-scale candidate | Lag h | NMAE | Bias kWh | Stop/curtail candidates |", "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for target, row in report["scada"]["groups"].items():
        corr = row["best_cross_correlation"]
        mae = row["best_absolute_scale_match"]
        lines.append(
            f"| {target} | {corr['candidate']} | {corr['lag_hours']} | {corr['correlation']:.6f} "
            f"| {mae['candidate']} | {mae['lag_hours']} | {mae['normalized_mae']:.6f} | {mae['bias_kwh']:.1f} "
            f"| {row['stopped_or_curtailed_candidates_ws_ge_5']:,} ({row['stopped_or_curtailed_rate']:.2%}) |"
        )

    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
