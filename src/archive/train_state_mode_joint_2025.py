"""Fit the 2025 deployment artifacts for the selected joint state/mode route."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.failure_mode_privileged_distillation import MODE_NAMES, reconstruct_failure_mode
from src.archive.latent_failure_mode_state_model import (
    _mode_classifier,
    _typed_features,
)
from src.multi_year_state_model import _add_categories, _model, _state_features
from src.recent_shared_exact_asof import _make_season_bank
from src.temporal_stable_conditional import _add_domain_and_pressure


FORECAST_YEAR = 2025
STATE_ORDER = (
    "selected_h05_l15_b075",
    "global_h4_l15_b075",
    "global_h05_l31_b075",
    "global_h05_l15_b100",
    "global_h05_l15_b050",
    "domain_h05_l15_b075",
)
STATE_RECIPES = {
    "selected_h05_l15_b075": (0.5, 15, 0.75),
    "global_h4_l15_b075": (4.0, 15, 0.75),
    "global_h05_l31_b075": (0.5, 31, 0.75),
    "global_h05_l15_b100": (0.5, 15, 1.00),
    "global_h05_l15_b050": (0.5, 15, 0.50),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _save_booster(model: object, path: Path) -> None:
    """Write through Python because LightGBM's Windows C API rejects Unicode paths."""
    path.write_text(model.booster_.model_to_string(), encoding="utf-8")


def _weights(season: np.ndarray, half_life: float) -> np.ndarray:
    output = np.exp2(-(FORECAST_YEAR - 1.0 - season.astype(np.float64)) / half_life)
    return output / output.mean()


def _baseline(numeric_state: pd.DataFrame, pitcher_weight: float) -> np.ndarray:
    pitcher = numeric_state["season__pitcher_rate_k80"].to_numpy(np.float64)
    batter = numeric_state["season__batter_rate_k80"].to_numpy(np.float64)
    return pitcher_weight * pitcher + (1.0 - pitcher_weight) * batter


def run(project: Path, joint_summary_path: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    joint_summary_path = (project / joint_summary_path).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = project / "data" / "train.csv"
    train = _add_domain_and_pressure(pd.read_csv(train_path, low_memory=False))
    target = train["control_success"].to_numpy(np.float64)
    season = train["season"].to_numpy(np.float64)
    numeric_state, _ = _state_features(train)
    features = _add_categories(train, numeric_state)
    categorical = [column for column in features if column.startswith("cat__")]
    categories = {
        column: [str(value) for value in features[column].cat.categories]
        for column in categorical
    }

    state_files: dict[str, object] = {}
    for name, (half_life, leaves, pitcher_weight) in STATE_RECIPES.items():
        print(f"[joint-final] state={name}", flush=True)
        baseline = _baseline(numeric_state, pitcher_weight)
        model = _model(leaves=leaves, seed=9100 + leaves + int(10 * half_life))
        model.fit(
            features,
            target - baseline,
            sample_weight=_weights(season, half_life),
            categorical_feature=categorical,
        )
        filename = f"joint_{name}.txt"
        _save_booster(model, output_dir / filename)
        state_files[name] = {
            "file": filename,
            "half_life": half_life,
            "leaves": leaves,
            "pitcher_weight": pitcher_weight,
        }
        del model, baseline
        gc.collect()

    domain_files = {}
    domain_baseline = _baseline(numeric_state, 0.75)
    domain_weight = _weights(season, 0.5)
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        print(f"[joint-final] state=domain_h05_l15_b075 domain={domain}", flush=True)
        mask = train["domain3"].eq(domain).to_numpy()
        model = _model(leaves=15, seed=9200 + len(domain))
        model.fit(
            features.loc[mask],
            target[mask] - domain_baseline[mask],
            sample_weight=domain_weight[mask],
            categorical_feature=categorical,
        )
        filename = f"joint_domain_h05_l15_b075_{domain}.txt"
        _save_booster(model, output_dir / filename)
        domain_files[domain] = filename
        del model
        gc.collect()
    state_files["domain_h05_l15_b075"] = {
        "files": domain_files,
        "half_life": 0.5,
        "leaves": 15,
        "pitcher_weight": 0.75,
    }

    mode = reconstruct_failure_mode(train)
    valid = mode >= 0
    classifier_files = {}
    for half_life in (0.5, 2.0, 4.0):
        print(f"[joint-final] mode_classifier h={half_life:g}", flush=True)
        classifier = _mode_classifier(
            seed=9300 + int(round(10 * half_life))
        )
        classifier.fit(
            features.loc[valid],
            mode[valid].astype(np.int64),
            sample_weight=_weights(season[valid], half_life),
            categorical_feature=categorical,
        )
        name = f"h{half_life:g}"
        filename = f"joint_mode_classifier_{name}.txt"
        _save_booster(classifier, output_dir / filename)
        classifier_files[name] = filename
        del classifier
        gc.collect()

    print("[joint-final] mode_outcome", flush=True)
    typed = _typed_features(features, mode)
    typed_categorical = [*categorical, "cat__latent_failure_mode"]
    outcome = _model(leaves=15, seed=9400)
    outcome.fit(
        typed.loc[valid],
        target[valid] - domain_baseline[valid],
        sample_weight=_weights(season[valid], 0.5),
        categorical_feature=typed_categorical,
    )
    outcome_filename = "joint_mode_outcome.txt"
    _save_booster(outcome, output_dir / outcome_filename)
    del outcome, typed
    gc.collect()

    conditional_weight = _weights(season[valid], 0.5)
    conditional_success = np.asarray(
        [
            np.average(
                target[valid][mode[valid] == klass],
                weights=conditional_weight[mode[valid] == klass],
            )
            for klass in range(4)
        ],
        dtype=np.float64,
    )
    preprocess = {
        "forecast_year": FORECAST_YEAR,
        "bank": _make_season_bank(train, FORECAST_YEAR),
        "feature_columns": list(features.columns),
        "categorical_columns": categorical,
        "categories": categories,
        "mode_categories": ["__MISSING__", *MODE_NAMES],
    }
    preprocess_filename = "joint_state_mode_preprocess.joblib"
    joblib.dump(preprocess, output_dir / preprocess_filename, compress=3)

    joint_summary = json.loads(joint_summary_path.read_text(encoding="utf-8"))
    spec = {
        "candidate": "joint_state_latent_failure_mode_2025_v1",
        "forecast_year": FORECAST_YEAR,
        "state_order": list(STATE_ORDER),
        "state_models": state_files,
        "mode_classifiers": classifier_files,
        "mode_outcome": outcome_filename,
        "preprocess": preprocess_filename,
        "conditional_success": conditional_success.tolist(),
        "mode_names": list(MODE_NAMES),
        "choices": joint_summary["choices"],
        "local_evidence": joint_summary["aggregate"],
        "row_local_inference": True,
        "test_aggregate_used": False,
        "selection_caveat": joint_summary["selection_caveat"],
    }
    spec_path = output_dir / "joint_state_mode_spec.json"
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files = sorted(output_dir.glob("joint_*"))
    manifest = {
        "protocol": "JOINT_STATE_MODE_FINAL_TRAIN_2025_V1",
        "train_sha256": _sha256(train_path),
        "training_rows": int(len(train)),
        "mode_label_coverage": float(valid.mean()),
        "artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
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
    parser.add_argument(
        "--joint-summary-path",
        type=Path,
        default=Path("artifacts/state_mode_joint_20260816_02/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/state_mode_joint_final_20260816"),
    )
    args = parser.parse_args()
    run(args.project, args.joint_summary_path, args.output_dir)


if __name__ == "__main__":
    main()
