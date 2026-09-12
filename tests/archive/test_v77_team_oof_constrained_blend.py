from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.archive.v77_team_oof_constrained_blend import (
    align_bundles,
    fit_robust_blend,
    load_bundle,
    single_candidate_headroom,
)


SHA_A = "A" * 64
SHA_B = "B" * 64
SHA_C = "C" * 64


def _manifest(name: str, parent: str = SHA_A) -> dict[str, object]:
    return {
        "protocol": "TEAM_OOF_BUNDLE_V1",
        "model_name": name,
        "model_sha256": SHA_A if name == "incumbent" else SHA_B,
        "config_sha256": SHA_B,
        "training_data_sha256": SHA_C,
        "comparison_parent_sha256": parent,
        "row_local": True,
        "test_aggregate_used": False,
        "public_score_used_for_weight": False,
        "axes": [
            {
                "axis": "outer_2022",
                "role": "nested_outer",
                "evaluation_season": 2022,
                "training_max_season": 2021,
                "recipe_frozen_before_axis": True,
            },
            {
                "axis": "outer_2023",
                "role": "nested_outer",
                "evaluation_season": 2023,
                "training_max_season": 2022,
                "recipe_frozen_before_axis": True,
            },
        ],
    }


def _frame(prediction: np.ndarray) -> pd.DataFrame:
    target = np.tile(np.array([0.0, 1.0]), 600)
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(len(target))],
            "axis": np.where(np.arange(len(target)) < 600, "outer_2022", "outer_2023"),
            "evaluation_season": np.where(np.arange(len(target)) < 600, 2022, 2023),
            "target": target,
            "prediction": prediction,
            "domain3": np.where(np.arange(len(target)) % 3 == 0, "F", "R_CORE"),
            "month": np.where(np.arange(len(target)) % 2 == 0, 4, 5),
            "pitcher_id": np.arange(len(target)) % 30,
            "batter_id": np.arange(len(target)) % 40,
        }
    )


def _write_bundle(
    tmp_path: Path, name: str, prediction: np.ndarray, parent: str = SHA_A
):
    csv = tmp_path / f"{name}.csv"
    manifest = tmp_path / f"{name}.json"
    _frame(prediction).to_csv(csv, index=False)
    manifest.write_text(json.dumps(_manifest(name, parent)), encoding="utf-8")
    return load_bundle(csv, manifest)


def test_bundle_alignment_and_robust_blend_finds_safe_candidate(tmp_path: Path) -> None:
    target = np.tile(np.array([0.0, 1.0]), 600)
    incumbent = np.full(len(target), 0.5)
    candidate = 0.5 + 0.10 * (target - 0.5)
    base = _write_bundle(tmp_path, "incumbent", incumbent)
    model = _write_bundle(tmp_path, "candidate", candidate)
    aligned, names = align_bundles(base, [model])
    result = fit_robust_blend(
        aligned,
        names,
        ["outer_2022"],
        min_group_rows=50,
    )
    assert result["weights"]["candidate"] > 0.99
    assert result["source_unclipped_bss_gain"] > 0.0
    assert result["maximum_group_brier_increase"] <= 1e-10


def test_bundle_rejects_wrong_comparison_parent(tmp_path: Path) -> None:
    target = np.tile(np.array([0.0, 1.0]), 600)
    base = _write_bundle(tmp_path, "incumbent", np.full(len(target), 0.5))
    model = _write_bundle(tmp_path, "candidate", np.full(len(target), 0.5), SHA_C)
    with pytest.raises(ValueError, match="comparison parent mismatch"):
        align_bundles(base, [model])


def test_manifest_rejects_non_forward_axis(tmp_path: Path) -> None:
    target = np.tile(np.array([0.0, 1.0]), 600)
    csv = tmp_path / "bad.csv"
    manifest = _manifest("candidate")
    manifest["axes"][0]["training_max_season"] = 2022  # type: ignore[index]
    path = tmp_path / "bad.json"
    _frame(np.full(len(target), 0.5)).to_csv(csv, index=False)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not strict-forward"):
        load_bundle(csv, path)


def test_single_candidate_headroom_matches_known_optimum() -> None:
    target = np.array([0.0, 1.0, 0.0, 1.0])
    incumbent = np.full(4, 0.5)
    candidate = np.array([0.4, 0.6, 0.4, 0.6])
    result = single_candidate_headroom(target, incumbent, candidate)
    assert result["optimal_candidate_weight"] == pytest.approx(1.0)
    assert result["brier_reduction"] == pytest.approx(0.09)
