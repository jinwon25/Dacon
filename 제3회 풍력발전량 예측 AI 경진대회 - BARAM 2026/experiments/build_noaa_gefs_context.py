"""Build compact, causal GEFS-spread context for year-forward power models.

The NOAA feature archive contains a 3x3, 0.5-degree stencil for each target
hour. This module reduces that stencil to fixed spatial summaries and never
reads generation labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRID_IDS = tuple(range(1, 10))
SOURCE_COLUMNS = {
    "gefs_u10_spread": "u10",
    "gefs_v10_spread": "v10",
    "gefs_uv10_spread_norm": "uv10_norm",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def aggregate_gefs_spread(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Reduce each complete 3x3 GEFS stencil to 21 fixed physical features."""

    required = {
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        *SOURCE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"GEFS spread features are missing columns: {missing}")
    data = frame[list(required)].copy()
    data["forecast_kst_dtm"] = pd.to_datetime(
        data["forecast_kst_dtm"], errors="raise"
    )
    data["data_available_kst_dtm"] = pd.to_datetime(
        data["data_available_kst_dtm"], errors="raise"
    )
    data["grid_id"] = pd.to_numeric(data["grid_id"], errors="raise").astype(int)
    if data.duplicated(["forecast_kst_dtm", "grid_id"]).any():
        raise ValueError("GEFS spread context has duplicate time/grid rows")
    counts = data.groupby("forecast_kst_dtm")["grid_id"].nunique()
    if not counts.eq(len(GRID_IDS)).all():
        raise ValueError("every GEFS target hour must contain all nine grids")
    observed_grids = tuple(sorted(data["grid_id"].unique()))
    if observed_grids != GRID_IDS:
        raise ValueError(f"unexpected GEFS grid IDs: {observed_grids}")
    issue_counts = data.groupby("forecast_kst_dtm")[
        "data_available_kst_dtm"
    ].nunique()
    if not issue_counts.eq(1).all():
        raise ValueError("GEFS grids disagree on their availability timestamp")

    grouped = data.groupby("forecast_kst_dtm", sort=True)
    pieces: list[pd.DataFrame] = []
    for source, short_name in SOURCE_COLUMNS.items():
        values = pd.to_numeric(data[source], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"GEFS source feature is incomplete: {source}")
        data[source] = values
        prefix = f"kma_um_ctx_gefs_spread_{short_name}"
        summaries = grouped[source].agg(["mean", "std", "min", "max"]).rename(
            columns={
                statistic: f"{prefix}__{statistic}"
                for statistic in ("mean", "std", "min", "max")
            }
        )
        pivot = data.pivot(
            index="forecast_kst_dtm",
            columns="grid_id",
            values=source,
        ).sort_index()
        contrasts = pd.DataFrame(index=pivot.index)
        contrasts[f"{prefix}__centre"] = pivot[5]
        contrasts[f"{prefix}__east_west"] = (
            pivot[[3, 6, 9]].mean(axis=1)
            - pivot[[1, 4, 7]].mean(axis=1)
        )
        contrasts[f"{prefix}__north_south"] = (
            pivot[[1, 2, 3]].mean(axis=1)
            - pivot[[7, 8, 9]].mean(axis=1)
        )
        pieces.extend((summaries, contrasts))

    features = pd.concat(pieces, axis=1).sort_index().astype("float32")
    if features.shape[1] != 21 or not features.columns.is_unique:
        raise RuntimeError("GEFS compact feature contract changed unexpectedly")
    if features.isna().any().any():
        raise ValueError("GEFS compact features contain missing values")
    issue = grouped["data_available_kst_dtm"].first().reindex(features.index)
    return features, issue


def align_to_primary_context(
    features: pd.DataFrame,
    source_issue: pd.Series,
    primary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Align to the KMA calendar and fill at most one causal leading boundary."""

    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = sorted(required.difference(primary.columns))
    if missing:
        raise ValueError(f"primary context is missing columns: {missing}")
    calendar = primary[list(required)].copy()
    calendar["forecast_kst_dtm"] = pd.to_datetime(
        calendar["forecast_kst_dtm"], errors="raise"
    )
    calendar["data_available_kst_dtm"] = pd.to_datetime(
        calendar["data_available_kst_dtm"], errors="raise"
    )
    if calendar["forecast_kst_dtm"].duplicated().any():
        raise ValueError("primary context has duplicate forecast timestamps")
    calendar = calendar.set_index("forecast_kst_dtm").sort_index()

    aligned = features.reindex(calendar.index)
    aligned_issue = source_issue.reindex(calendar.index)
    missing_rows = aligned.isna().all(axis=1)
    partial_rows = aligned.isna().any(axis=1) & ~missing_rows
    if partial_rows.any():
        raise ValueError("GEFS alignment produced partially missing rows")
    positions = np.flatnonzero(missing_rows.to_numpy())
    if len(positions) > 1 or (len(positions) == 1 and positions[0] != 0):
        raise ValueError("GEFS alignment permits only one leading boundary gap")
    imputed: list[str] = []
    if len(positions) == 1:
        # The 2023 KMA calendar contains one 00:00 boundary whose preceding
        # GEFS target was not collected. Never backfill it from the later
        # 01:00 forecast: that forecast belonged to a later issue. A fixed
        # all-zero spread sentinel is label-free and preserves causality.
        aligned.iloc[0] = 0.0
        aligned_issue.iloc[0] = calendar["data_available_kst_dtm"].iloc[0]
        imputed.append(calendar.index[0].isoformat())
    if aligned.isna().any().any() or aligned_issue.isna().any():
        raise ValueError("GEFS alignment remains incomplete")

    causal = aligned_issue <= calendar["data_available_kst_dtm"]
    if not causal.all():
        raise ValueError(
            f"GEFS context has {int((~causal).sum())} post-reference rows"
        )
    return aligned.astype("float32"), aligned_issue, imputed


def build_context(
    source_features: Path,
    source_manifest: Path,
    primary_context: Path,
    output: Path,
    provenance_output: Path,
) -> dict[str, Any]:
    source = pd.read_csv(source_features, encoding="utf-8-sig")
    primary = pd.read_csv(primary_context, encoding="utf-8-sig")
    compact, source_issue = aggregate_gefs_spread(source)
    aligned, aligned_issue, imputed = align_to_primary_context(
        compact, source_issue, primary
    )
    result = aligned.copy()
    result.insert(0, "data_available_kst_dtm", aligned_issue.to_numpy())
    result.insert(0, "forecast_kst_dtm", aligned.index)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    primary_issue = pd.to_datetime(primary["data_available_kst_dtm"])
    margins = (
        primary_issue.to_numpy(dtype="datetime64[ns]")
        - aligned_issue.to_numpy(dtype="datetime64[ns]")
    ) / np.timedelta64(1, "m")
    report = {
        "family": "noaa_gefs_spread_compact_context",
        "source_features": source_features.relative_to(ROOT).as_posix(),
        "source_features_sha256": _sha256(source_features),
        "source_manifest": source_manifest.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": _sha256(source_manifest),
        "primary_context": primary_context.relative_to(ROOT).as_posix(),
        "primary_context_sha256": _sha256(primary_context),
        "rows": int(len(result)),
        "feature_count": int(aligned.shape[1]),
        "imputed_boundary_timestamps": imputed,
        "causality": {
            "violations": 0,
            "minimum_margin_minutes": float(np.min(margins)),
            "maximum_margin_minutes": float(np.max(margins)),
        },
        "output": output.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(output),
    }
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--primary-context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance-output", required=True)
    args = parser.parse_args()
    report = build_context(
        _rooted(args.source_features),
        _rooted(args.source_manifest),
        _rooted(args.primary_context),
        _rooted(args.output),
        _rooted(args.provenance_output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
