"""Train the frozen 2024 -> 2025 exact-ASOF overlay artifacts."""

from __future__ import annotations

import argparse
import hashlib
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
    _current_season_state,
    _make_season_bank,
    _row_state,
)
from src.temporal_stable_conditional import (
    Paths,
    _add_domain_and_pressure,
    _find_legacy_file,
    _load_base,
    build_bank,
    build_features,
)


def _category_maps(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_COLUMNS:
        values = frame[column].astype("string").fillna("<NA>").astype(str)
        maps[column] = {
            value: index for index, value in enumerate(values.unique().tolist())
        }
    return maps


def _exact_features(
    frame: pd.DataFrame,
    bank: dict[str, object],
    category_maps: dict[str, dict[str, int]],
    *,
    include_categories: bool,
) -> pd.DataFrame:
    numeric = [column for column in NUMERIC_COLUMNS if column in frame]
    output = pd.concat(
        [
            frame[numeric]
            .apply(pd.to_numeric, errors="coerce")
            .reset_index(drop=True),
            _row_state(frame),
            _current_season_state(frame, bank),
        ],
        axis=1,
    )
    if include_categories:
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
        output = pd.concat([output, categories], axis=1)
    return output.astype(np.float32)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(
    project: Path,
    research_project: Path,
    output_dir: Path,
    source_artifact_root: Path | None = None,
) -> dict[str, object]:
    project = project.resolve()
    research_project = research_project.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    source = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    target = source["control_success"].to_numpy(np.float64)
    category_maps = _category_maps(source)
    source_bank = _make_season_bank(train, 2024)
    deployment_bank = _make_season_bank(train, 2025)

    exact_lgb_x = _exact_features(
        source, source_bank, category_maps, include_categories=True
    )
    classifier = lgb.LGBMClassifier(
        objective="binary",
        verbosity=-1,
        n_jobs=6,
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=7,
        min_child_samples=350,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=6.0,
        max_bin=127,
        random_state=918,
    )
    classifier.fit(exact_lgb_x, target)
    # The LightGBM Windows C API cannot write directly to a Unicode path.
    # Python's UTF-8 path handling is safe, so persist the exact model string.
    (output_dir / "recent_exact_lgb.txt").write_text(
        classifier.booster_.model_to_string(num_iteration=120), encoding="utf-8"
    )

    exact_ridge_x = _exact_features(
        source, source_bank, category_maps, include_categories=False
    )
    ridge_median = exact_ridge_x.median().fillna(0.0)
    ridge_raw = exact_ridge_x.fillna(ridge_median).to_numpy(np.float64)
    exact_scaler = StandardScaler()
    ridge_x = np.clip(exact_scaler.fit_transform(ridge_raw), -8.0, 8.0)
    exact_ridge = Ridge(alpha=10_000.0, solver="lsqr", max_iter=100, tol=1e-3)
    exact_ridge.fit(ridge_x, target)
    joblib.dump(exact_ridge, output_dir / "recent_exact_ridge.joblib", compress=3)

    artifact_root = (
        source_artifact_root.resolve()
        if source_artifact_root is not None
        else research_project / "artifacts" / "top10_20260814"
    )
    paths = Paths(
        corrected_cb=artifact_root / "corrected_cb_oof_20260814_01",
        advanced=artifact_root / "advanced_domain_residual_20260814_01",
        legacy_root=artifact_root,
    )
    fold = _load_base(train, paths, 2024)
    source_indices = np.flatnonzero(train["season"].eq(2024).to_numpy())
    if not np.array_equal(fold["train_index"].to_numpy(np.int64), source_indices):
        raise ValueError("2024 source and v10 OOF order differ")
    core = source["domain3"].eq("R_CORE").to_numpy()
    core_rows = source.loc[core].reset_index(drop=True)
    core_target = fold["target"].to_numpy(np.float64)[core]
    core_base = fold["base"].to_numpy(np.float64)[core]
    stable_source_bank = build_bank(train.loc[train["season"].eq(2023)])
    stable_deployment_bank = build_bank(train.loc[train["season"].eq(2024)])
    stable_frame = build_features(core_rows, core_base, stable_source_bank)
    stable_scaler = StandardScaler()
    stable_x = stable_scaler.fit_transform(stable_frame)
    stable_ridge = Ridge(alpha=10_000.0, fit_intercept=True, solver="cholesky")
    stable_ridge.fit(stable_x, core_target - core_base)
    joblib.dump(stable_ridge, output_dir / "recent_stable_ridge.joblib", compress=3)

    core_bias = float(np.mean(core_target - core_base))
    preprocess = {
        "exact_bank": deployment_bank,
        "category_maps": category_maps,
        "exact_lgb_columns": list(exact_lgb_x.columns),
        "exact_ridge_columns": list(exact_ridge_x.columns),
        "exact_ridge_median": ridge_median,
        "exact_ridge_mean": exact_scaler.mean_,
        "exact_ridge_scale": exact_scaler.scale_,
        "stable_bank": stable_deployment_bank,
        "stable_columns": list(stable_frame.columns),
        "stable_mean": stable_scaler.mean_,
        "stable_scale": stable_scaler.scale_,
    }
    joblib.dump(preprocess, output_dir / "recent_exact_preprocess.joblib", compress=3)
    spec = {
        "candidate": "v13_recent_exact_asof_overlay_v1",
        "source_season": 2024,
        "forecast_season": 2025,
        "anchor_team_id": 13,
        "core_bias": core_bias,
        "core_exact_lgb_weight": 0.125,
        "core_exact_ridge_weight": 0.20,
        "core_stable_gamma": 0.20,
        "f_exact_lgb_weight": 0.75,
        "exact_lgb_num_iterations": 120,
        "selection_rule": (
            "core weights selected on 2022->2023 and confirmed on 2021->2022/"
            "2023->2024; F blend confirmed on post-break 2023->2024"
        ),
        "test_aggregate_used": False,
        "row_local_inference": True,
    }
    (output_dir / "recent_exact_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        **spec,
        "n_source_rows": int(len(source)),
        "n_core_rows": int(core.sum()),
        "target_rate": float(target.mean()),
        "exact_feature_count": int(exact_lgb_x.shape[1]),
        "stable_feature_count": int(stable_frame.shape[1]),
        "source_oof_sha256": {
            "corrected_cb_o2024.npz": _sha256(
                paths.corrected_cb / "corrected_cb_o2024.npz"
            ),
            "correction_compact_l15_o2024.npz": _sha256(
                paths.advanced / "correction_compact_l15_o2024.npz"
            ),
            "legacy_baseline_s42_o2024.npz": _sha256(
                _find_legacy_file(paths.legacy_root, 2024)
            ),
        },
        "source_oof_note": "Frozen official-train OOF inputs; this run refits final models but does not refit those OOFs.",
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
    parser.add_argument("--research-project", type=Path, default=Path("."))
    parser.add_argument("--source-artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    run(
        args.project,
        args.research_project,
        args.output_dir,
        source_artifact_root=args.source_artifact_root,
    )


if __name__ == "__main__":
    main()
