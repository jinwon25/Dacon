"""Full-fit the frozen v209 three-seed workload H1 recipe for 2025.

This is a deployment-only fit.  Feature family, CatBoost parameters, and seeds
come directly from the forward-OOF v209 audit.  The fitted estimators replace
only the model list in a copy of the deployed H1 bundle, preserving its frozen
TrackMan context, current-season priors, affine map, and row-local corrections.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any
import zipfile

import joblib
import numpy as np

from src.archive.v201_paired_workload_booster import workload_feature_frame
from src.archive.v203_h1_workload_feature_screen import MODEL_CONFIG
from src.champion.v131_catboost_h1_independent_oof import _pipeline, _prepare_features


PROTOCOL = "V342_FINALIZE_WORKLOAD_H1_V1"
SEEDS = (42, 43, 44)
H1_MEMBER = "model/h1/model/rf.pkl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def seeded_config(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"unregistered seed: {seed}")
    return {**MODEL_CONFIG, "seed": int(seed)}


def replace_models(
    deployed_bundle: dict[str, Any],
    models: list[Any],
    features: list[str],
) -> dict[str, Any]:
    if len(deployed_bundle.get("models", [])) != 3:
        raise ValueError("deployed H1 bundle must contain three models")
    if len(models) != len(SEEDS):
        raise ValueError("workload H1 requires exactly three fitted models")
    if len(features) != len(set(features)):
        raise ValueError("augmented H1 features are not unique")
    output = dict(deployed_bundle)
    output["models"] = list(models)
    output["features"] = list(features)
    output["note"] = (
        str(output.get("note", ""))
        + " | v342 frozen v209 workload-H1 full fit seeds=42,43,44"
    )
    output["v342"] = {
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "base_feature_count": len(deployed_bundle.get("features", [])),
        "augmented_feature_count": len(features),
    }
    return output


def run(
    train_csv: Path,
    trackman_csv: Path,
    component_root: Path,
    deployed_zip: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, _season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, component_root
    )
    workload = workload_feature_frame(train)
    workload_columns = list(workload.columns)
    collisions = sorted(set(h1_features) & set(workload_columns))
    if collisions:
        raise ValueError(f"workload feature collision: {collisions}")
    for column in workload_columns:
        train[column] = workload[column].to_numpy(np.float32)
    augmented_features = list(h1_features) + workload_columns
    del workload
    gc.collect()

    with zipfile.ZipFile(deployed_zip, "r") as archive:
        if archive.namelist().count(H1_MEMBER) != 1:
            raise ValueError("deployed ZIP has no unique H1 bundle")
        deployed_bundle = joblib.load(io.BytesIO(archive.read(H1_MEMBER)))
    if not isinstance(deployed_bundle, dict):
        raise ValueError("deployed H1 bundle is not a dictionary")
    if list(deployed_bundle.get("features", [])) != list(h1_features):
        raise ValueError("prepared H1 feature definition differs from deployed bundle")

    models: list[Any] = []
    fits: list[dict[str, Any]] = []
    for seed in SEEDS:
        started = time.time()
        model = _pipeline(augmented_features, seeded_config(seed))
        model.fit(train.loc[:, augmented_features], target.astype(np.int8))
        elapsed = time.time() - started
        probe = model.predict_proba(train.loc[:999, augmented_features])[:, 1]
        if not np.isfinite(probe).all() or not np.logical_and(probe > 0, probe < 1).all():
            raise ValueError(f"invalid workload H1 probe prediction for seed {seed}")
        models.append(model)
        fits.append(
            {
                "seed": int(seed),
                "rows": len(train),
                "features": len(augmented_features),
                "fit_seconds": elapsed,
                "probe_mean": float(np.mean(probe)),
                "probe_std": float(np.std(probe)),
            }
        )
        gc.collect()

    bundle = replace_models(deployed_bundle, models, augmented_features)
    bundle_path = output_dir / "workload_h1_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    summary = {
        "protocol": PROTOCOL,
        "status": "full_fit_complete",
        "output_bundle": str(bundle_path),
        "output_sha256": sha256(bundle_path),
        "output_bytes": bundle_path.stat().st_size,
        "source_zip": str(deployed_zip),
        "source_zip_sha256": sha256(deployed_zip),
        "source_member": H1_MEMBER,
        "base_feature_count": len(h1_features),
        "workload_feature_count": len(workload_columns),
        "augmented_feature_count": len(augmented_features),
        "workload_features": workload_columns,
        "model_config": MODEL_CONFIG,
        "fits": fits,
        "preserved_bundle_keys": sorted(
            key for key in deployed_bundle if key not in {"models", "features", "note"}
        ),
        "restrictions": {
            "official_train_only": True,
            "v209_feature_family_frozen": True,
            "v209_model_config_frozen": True,
            "v209_seeds_frozen": True,
            "all_available_train_rows_used_for_2025_fit": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--deployed-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.trackman_csv,
        args.component_root,
        args.deployed_zip,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
