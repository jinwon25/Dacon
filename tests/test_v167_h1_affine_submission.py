from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.archive.v167_build_h1_affine_submission_package import (
    PROTOCOL,
    _affine,
    _deterministic_zip_directory,
)
from src.core.packaging import _sha256


def test_v167_frozen_affine_and_parent_contract() -> None:
    config = json.loads(
        Path("research/configs/v167_h1_affine_submission_package.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["protocol"] == PROTOCOL
    assert config["candidate_affine"] == {
        "alpha": 1.09,
        "center": 0.5854452601930041,
    }
    assert (
        config["parent_package_sha256"]
        == "7A27BE5878A79934544C741F283C139D40FB20484D52DB494928BCBE27E1E337"
    )
    assert config["h1_model_member"] == "model/h1/model/rf.pkl"


def test_v167_affine_matches_formula_and_clips() -> None:
    values = np.array([-1.0, 0.2, 0.5, 0.9, 2.0], dtype=np.float64)
    alpha = 1.09
    center = 0.5854452601930041
    expected = np.clip(center + alpha * (values - center), 0.001, 0.999)
    np.testing.assert_array_equal(_affine(values, alpha, center), expected)


def test_v167_zip_writer_is_byte_reproducible(tmp_path: Path) -> None:
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    for source in (source_a, source_b):
        (source / "model").mkdir(parents=True)
        (source / "script.py").write_text("print('ok')\n", encoding="utf-8")
        (source / "requirements.txt").write_text("numpy==2.2.6\n", encoding="utf-8")
        (source / "model" / "value.bin").write_bytes(b"same-model-bytes")
    timestamp = [2026, 8, 25, 0, 0, 0]
    zip_a = tmp_path / "a.zip"
    zip_b = tmp_path / "b.zip"
    _deterministic_zip_directory(source_a, zip_a, timestamp)
    _deterministic_zip_directory(source_b, zip_b, timestamp)
    assert _sha256(zip_a) == _sha256(zip_b)
