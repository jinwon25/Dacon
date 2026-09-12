"""Locally emulate UMKR context from causally available UMRG forecasts.

The fixed linear downscaler is selected without power-generation labels.  Its
quality gate is measured on 2023 Q4 after fitting 2023 Jan-Sep.  If the gate
passes, a final emulator is fit on all 2023 weather pairs and applied unchanged
to UMRG 2023, 2024, and 2025.  This keeps power-model train/validation/test
features on one transformation and never calls a remote inference service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.kma_year_forward_quantile_blend import load_context_features


ROOT = Path(__file__).resolve().parents[1]
RIDGE_ALPHA = 10.0
MINIMUM_MEDIAN_CORRELATION = 0.90
MAXIMUM_MEDIAN_NORMALIZED_RMSE = 0.50


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip_emulated_features(
    prediction: np.ndarray,
    columns: list[str],
) -> np.ndarray:
    output = np.asarray(prediction, dtype=float).copy()
    for position, column in enumerate(columns):
        if "_speed" in column or "_shear" in column:
            output[:, position] = np.maximum(output[:, position], 0.0)
        if "_cos" in column:
            output[:, position] = np.clip(output[:, position], -1.0, 1.0)
    return output


def feature_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, column in enumerate(columns):
        actual = truth[:, position]
        estimate = prediction[:, position]
        rmse = float(np.sqrt(np.mean((estimate - actual) ** 2)))
        scale = float(np.std(actual))
        correlation = float(np.corrcoef(actual, estimate)[0, 1])
        rows.append(
            {
                "feature": column,
                "correlation": correlation,
                "rmse": rmse,
                "normalized_rmse": rmse / max(scale, 1e-6),
            }
        )
    return rows


def make_emulator() -> Any:
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def _raw_context(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        parse_dates=["forecast_kst_dtm", "data_available_kst_dtm"],
    )
    if frame["forecast_kst_dtm"].duplicated().any():
        raise ValueError("context contains duplicate forecast timestamps")
    return frame.set_index("forecast_kst_dtm").sort_index()


def run(args: argparse.Namespace) -> dict[str, Any]:
    regional_paths = {
        year: _rooted(getattr(args, f"regional_{year}"))
        for year in (2023, 2024, 2025)
    }
    local_2023_path = _rooted(args.local_2023)
    regional = {
        year: load_context_features(path)[0]
        for year, path in regional_paths.items()
    }
    local_2023, _ = load_context_features(local_2023_path)
    x_2023 = regional[2023]
    if not x_2023.index.equals(local_2023.index):
        raise ValueError("2023 UMRG and UMKR indexes differ")
    target_columns = [
        column for column in local_2023 if column.startswith("kma_um_ctx_")
    ]
    y_2023 = local_2023[target_columns]

    development = x_2023.index < pd.Timestamp("2023-10-01")
    confirmation = ~development
    diagnostic = make_emulator()
    diagnostic.fit(x_2023.loc[development], y_2023.loc[development])
    holdout_prediction = clip_emulated_features(
        diagnostic.predict(x_2023.loc[confirmation]),
        target_columns,
    )
    metrics = feature_metrics(
        y_2023.loc[confirmation].to_numpy(dtype=float),
        holdout_prediction,
        target_columns,
    )
    median_correlation = float(
        np.median([row["correlation"] for row in metrics])
    )
    median_normalized_rmse = float(
        np.median([row["normalized_rmse"] for row in metrics])
    )
    quality_gates = {
        "median_correlation_at_least_0_90": (
            median_correlation >= MINIMUM_MEDIAN_CORRELATION
        ),
        "median_normalized_rmse_at_most_0_50": (
            median_normalized_rmse <= MAXIMUM_MEDIAN_NORMALIZED_RMSE
        ),
    }
    promoted = bool(all(quality_gates.values()))

    outputs: dict[str, Any] = {}
    if promoted:
        emulator = make_emulator()
        emulator.fit(x_2023, y_2023)
        output_root = _rooted(args.output_root)
        for year in (2023, 2024, 2025):
            prediction = clip_emulated_features(
                emulator.predict(regional[year]),
                target_columns,
            )
            source = _raw_context(regional_paths[year])
            source = source.reindex(regional[year].index)
            if source.isna().any().any():
                raise ValueError(f"{year} regional source alignment is incomplete")
            frame = pd.DataFrame(
                prediction,
                index=regional[year].index,
                columns=target_columns,
            )
            frame.insert(
                0,
                "data_available_kst_dtm",
                source["data_available_kst_dtm"].to_numpy(),
            )
            frame.insert(0, "forecast_kst_dtm", frame.index)
            year_dir = output_root / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            feature_path = year_dir / "features.csv"
            frame.to_csv(feature_path, index=False, encoding="utf-8-sig")

            parent_manifest_path = regional_paths[year].with_name("manifest.json")
            parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
            manifest = {
                "schema_version": 1,
                "competition_eligible": True,
                "provider": "Korea Meteorological Administration / local Ridge emulator",
                "dataset": f"Locally emulated UMKR context from KMA UMRG ({args.point_name}, {year})",
                "source_type": "operational_forecast_archive",
                "documentation_url": parent["documentation_url"],
                "license": parent["license"],
                "license_url": parent["license_url"],
                "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "coverage": {
                    "targets": int(len(frame)),
                    "model": "fixed local Ridge UMRG-to-UMKR emulator",
                    "training_weather_year": 2023,
                    "power_labels_used": False,
                },
                "availability_evidence": parent["availability_evidence"],
                "causality_audit": parent["causality_audit"],
                "parent_manifests": [
                    parent_manifest_path.relative_to(ROOT).as_posix(),
                    local_2023_path.with_name("manifest.json")
                    .relative_to(ROOT)
                    .as_posix(),
                ],
                "feature_file": {
                    "path": feature_path.relative_to(ROOT).as_posix(),
                    "rows": int(len(frame)),
                    "sha256": _sha256(feature_path),
                },
                "raw_files": [
                    {
                        "path": feature_path.relative_to(ROOT).as_posix(),
                        "source_url": (
                            "local-ridge-emulator://KMA-UMRG-to-UMKR/"
                            f"{args.point_name}/{year}"
                        ),
                        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "sha256": _sha256(feature_path),
                    }
                ],
            }
            manifest_path = year_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs[str(year)] = {
                "features": feature_path.relative_to(ROOT).as_posix(),
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(feature_path),
                "rows": int(len(frame)),
            }

    report = {
        "family": "fixed_local_umrg_to_umkr_ridge_emulator",
        "point_name": args.point_name,
        "contract": {
            "diagnostic_fit": "2023 Jan-Sep weather pairs",
            "diagnostic_confirmation": "2023 Q4 weather pairs",
            "final_fit": "all 2023 weather pairs only",
            "application": "same frozen emulator to 2023, 2024, and 2025 UMRG",
            "power_labels_used": False,
            "remote_model_inference_used": False,
            "ridge_alpha": RIDGE_ALPHA,
        },
        "sources": {
            **{
                f"regional_{year}": path.relative_to(ROOT).as_posix()
                for year, path in regional_paths.items()
            },
            "local_2023": local_2023_path.relative_to(ROOT).as_posix(),
        },
        "holdout": {
            "rows": int(confirmation.sum()),
            "feature_metrics": metrics,
            "median_correlation": median_correlation,
            "median_normalized_rmse": median_normalized_rmse,
        },
        "quality_gates": quality_gates,
        "promotion": "promoted" if promoted else "rejected",
        "outputs": outputs,
    }
    report_path = _rooted(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for year in (2023, 2024, 2025):
        parser.add_argument(
            f"--regional-{year}",
            default=(
                "artifacts_final/external_weather/"
                f"kma_um_regional_context_{year}/features.csv"
            ),
        )
    parser.add_argument("--local-2023", required=True)
    parser.add_argument("--point-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "point_name": report["point_name"],
                "holdout": {
                    "median_correlation": report["holdout"][
                        "median_correlation"
                    ],
                    "median_normalized_rmse": report["holdout"][
                        "median_normalized_rmse"
                    ],
                },
                "quality_gates": report["quality_gates"],
                "promotion": report["promotion"],
                "outputs": report["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
