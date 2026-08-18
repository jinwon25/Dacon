"""Train the frozen v14 refinement on top of the submitted v13 overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.recent_shared_exact_asof import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    _core_slope,
    _current_season_state,
    _engineered,
    _make_season_bank,
    _row_state,
)
from src.temporal_stable_conditional import _add_domain_and_pressure


ANCHOR_TEAM = 13


def _numeric_exact(frame: pd.DataFrame, train: pd.DataFrame, year: int) -> pd.DataFrame:
    numeric = [column for column in NUMERIC_COLUMNS if column in frame]
    return pd.concat(
        [
            frame[numeric]
            .apply(pd.to_numeric, errors="coerce")
            .reset_index(drop=True),
            _row_state(frame),
            _current_season_state(frame, _make_season_bank(train, year)),
        ],
        axis=1,
    ).astype(np.float32)


def _category_maps(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_COLUMNS:
        values = frame[column].astype("string").fillna("<NA>").astype(str)
        output[column] = {
            value: index for index, value in enumerate(values.unique().tolist())
        }
    return output


def _trend_features(
    frame: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    numeric = [column for column in NUMERIC_COLUMNS if column in frame]
    output = pd.concat(
        [
            frame[numeric]
            .apply(pd.to_numeric, errors="coerce")
            .reset_index(drop=True),
            _engineered(frame),
        ],
        axis=1,
    )
    categories = pd.DataFrame(index=np.arange(len(frame)))
    for column, mapping in category_maps.items():
        categories[f"cat__{column}"] = (
            frame[column]
            .astype("string")
            .fillna("<NA>")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(np.float32)
            .to_numpy()
        )
    return pd.concat([output, categories], axis=1).astype(np.float32)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )

    anchor_frames: list[pd.DataFrame] = []
    anchor_targets: list[np.ndarray] = []
    for year in (2023, 2024):
        season = train.loc[train["season"].eq(year)].reset_index(drop=True)
        anchor = season["domain3"].eq("R_ANCHOR").to_numpy()
        anchor_rows = season.loc[anchor].reset_index(drop=True)
        anchor_frames.append(_numeric_exact(anchor_rows, train, year))
        anchor_targets.append(
            anchor_rows["control_success"].to_numpy(np.float64)
        )
    anchor_x = pd.concat(anchor_frames, ignore_index=True)
    anchor_y = np.concatenate(anchor_targets)
    anchor_median = anchor_x.median().fillna(0.0)
    anchor_scaler = StandardScaler()
    anchor_scaled = anchor_scaler.fit_transform(anchor_x.fillna(anchor_median))
    anchor_model = Ridge(
        alpha=500_000.0,
        solver="lsqr",
        max_iter=200,
        tol=1e-4,
    )
    anchor_model.fit(anchor_scaled, anchor_y)
    joblib.dump(
        anchor_model, output_dir / "v14_anchor_ridge.joblib", compress=3
    )

    source = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    category_maps = _category_maps(source)
    trend_x = _trend_features(source, category_maps)
    source_means = (
        source.groupby("domain3", observed=True)["control_success"].mean().to_dict()
    )
    source_prior = source["domain3"].map(source_means).to_numpy(np.float64)
    trend_model = lgb.LGBMRegressor(
        objective="regression_l2",
        verbosity=-1,
        n_jobs=6,
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=7,
        min_child_samples=300,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=5.0,
        max_bin=127,
        random_state=77,
    )
    trend_model.fit(
        trend_x,
        source["control_success"].to_numpy(np.float64) - source_prior,
    )
    (output_dir / "v14_f_trend_lgb.txt").write_text(
        trend_model.booster_.model_to_string(num_iteration=180), encoding="utf-8"
    )

    preprocess = {
        "anchor_columns": list(anchor_x.columns),
        "anchor_median": anchor_median,
        "anchor_mean": anchor_scaler.mean_,
        "anchor_scale": anchor_scaler.scale_,
        "trend_columns": list(trend_x.columns),
        "trend_category_maps": category_maps,
        "trend_source_means": source_means,
        "trend_slope": _core_slope(train, 2024),
    }
    joblib.dump(
        preprocess,
        output_dir / "v14_refinement_preprocess.joblib",
        compress=3,
    )
    spec = {
        "candidate": "v14_reliability_anchor_and_f_trend_v1",
        "source_seasons": [2023, 2024],
        "forecast_season": 2025,
        "anchor_team_id": ANCHOR_TEAM,
        "anchor_alpha": 500_000.0,
        "anchor_reliability_threshold": 0.90,
        "anchor_weight": 0.20,
        "f_trend_alpha": 0.15,
        "parent_f_exact_weight": 0.75,
        "f_trend_num_iterations": 180,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "selection_rule": (
            "anchor gate positive on 2021->2022, 2022->2023, 2023->2024; "
            "F alpha=.15 maximises three-transition minimum"
        ),
    }
    (output_dir / "v14_refinement_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        **spec,
        "n_anchor_train": int(len(anchor_y)),
        "anchor_feature_count": int(anchor_x.shape[1]),
        "n_f_trend_train": int(len(source)),
        "trend_feature_count": int(trend_x.shape[1]),
        "artifacts": {
            path.name: path.stat().st_size
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
