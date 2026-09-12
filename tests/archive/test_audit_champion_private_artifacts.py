from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.audit_champion_private_artifacts import (
    audit_private_release,
    domain3,
    sha256_array,
    sha256_file,
    sha256_strings,
)


def test_audit_private_release_checks_model_and_oof_alignment(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    model_dir = artifacts / "standalone_champion_1161"
    oof_dir = artifacts / "oof_champion_1161"
    config_dir = tmp_path / "configs"
    model_dir.mkdir(parents=True)
    oof_dir.mkdir(parents=True)
    config_dir.mkdir()

    model_zip = model_dir / "standalone_champion_1161.zip"
    with zipfile.ZipFile(model_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("script.py", "def predict_dataframe(frame):\n    return []\n")
        archive.writestr("requirements.txt", "numpy\npandas\n")
        archive.writestr("model/dummy.txt", "model")
    model_sha = sha256_file(model_zip)
    model_manifest = model_dir / "standalone_manifest.json"
    model_manifest.write_text(
        json.dumps(
            {
                "canonical_release": {
                    "sha256": model_sha,
                    "bytes": model_zip.stat().st_size,
                },
                "promoted_to_champion": True,
                "public_result": {
                    "public_score": 1161.2020600422,
                    "submission_id": 59988,
                    "official_runtime_seconds": 41,
                },
                "standalone_no_parent_zip_dependency": True,
                "row_local_inference": True,
                "test_aggregate_used": False,
            }
        ),
        encoding="utf-8",
    )

    config = config_dir / "v84_fixed_v56_fm.json"
    config.write_text('{"protocol":"test"}\n', encoding="utf-8")
    train = pd.DataFrame(
        {
            "row_id": ["a", "b", "c"],
            "control_success": [1.0, 0.0, 1.0],
            "season": [2024, 2024, 2024],
            "game_month": [4, 5, 6],
            "game_type": ["R", "R", "F"],
            "pitcher_team_id": [1, 13, 2],
            "batter_team_id": [2, 2, 3],
            "pitcher_id": [10, 11, 12],
            "batter_id": [20, 21, 22],
        }
    )
    train_csv = tmp_path / "train.csv"
    train.to_csv(train_csv, index=False)

    raw_index = np.arange(3, dtype=np.int64)
    target = train["control_success"].to_numpy(np.float64)
    prediction = np.asarray([0.7, 0.3, 0.6], dtype=np.float64)
    exact = np.ones(3, dtype=bool)
    arrays = {
        "raw_index": raw_index,
        "target": target,
        "parent": prediction,
        "exact_mask": exact,
        "season": train["season"].to_numpy(np.int16),
        "game_month": train["game_month"].to_numpy(np.int16),
        "domain3": domain3(train).astype(str),
        "pitcher_id": train["pitcher_id"].to_numpy(),
        "batter_id": train["batter_id"].to_numpy(),
        "common_parent": np.asarray([0.5, 0.5, 0.5], dtype=np.float64),
    }
    axis_file = oof_dir / "v84_full_2024.npz"
    np.savez_compressed(axis_file, **arrays)
    private_manifest = {
        "protocol": "CHAMPION_1161_PRIVATE_OOF_EVIDENCE_V1",
        "visibility": "PRIVATE_OFFICIAL_DACON_TEAM_ONLY",
        "model": {
            "sha256": model_sha,
            "config": "../../configs/v84_fixed_v56_fm.json",
            "config_sha256": sha256_file(config),
        },
        "train_csv_sha256": sha256_file(train_csv),
        "train_rows": 3,
        "row_local": True,
        "test_aggregate_used": False,
        "contains_competition_targets_and_player_ids": True,
        "caveat": "test evidence",
        "axes": [
            {
                "axis": "v84_full_2024",
                "file": axis_file.name,
                "file_sha256": sha256_file(axis_file),
                "bytes": axis_file.stat().st_size,
                "role": "development_contaminated_family_inspected",
                "fidelity": "exact_v84_all_domains",
                "evaluation_season": 2024,
                "training_max_season": 2023,
                "rows": 3,
                "exact_rows": 3,
                "exact_domains": ["F", "R_ANCHOR", "R_CORE"],
                "raw_index_sha256": sha256_array(raw_index),
                "row_id_sha256": sha256_strings(train["row_id"]),
                "target_sha256": sha256_array(target),
                "prediction_sha256": sha256_array(prediction),
            }
        ],
    }
    (oof_dir / "manifest.json").write_text(
        json.dumps(private_manifest), encoding="utf-8"
    )

    result = audit_private_release(
        model_zip,
        model_manifest,
        oof_dir,
        train_csv=train_csv,
    )

    assert result["status"] == "pass"
    assert result["model"]["sha256"] == model_sha
    assert result["axes"][0]["train_alignment"] == "pass"
