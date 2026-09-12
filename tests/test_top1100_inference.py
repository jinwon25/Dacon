from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.archive.features import FeatureBuilder


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _inference_module():
    spec = importlib.util.spec_from_file_location("submission_inference", PROJECT_DIR / "src" / "archive" / "v2_offline_entrypoint_script.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trackman_weights_are_row_local_and_support_affine_extension():
    module = _inference_module()
    frame = pd.DataFrame({"game_type": ["F", "R", "X", None]})
    hybrid = {
        "trackman_default_weight": 0.0,
        "trackman_weight_by_game_type": {"F": 1.5, "R": 0.05},
    }
    actual = module.resolve_trackman_weights(frame, hybrid)
    np.testing.assert_allclose(actual, [1.5, 0.05, 0.0, 0.0])


def test_hierarchical_catboost_features_match_training_builder():
    if not (PROJECT_DIR / "data" / "train.csv").is_file():
        pytest.skip("requires Git-ignored DACON train.csv")
    module = _inference_module()
    frame = pd.read_csv(PROJECT_DIR / "data" / "train.csv", nrows=64)
    target = frame["control_success"].to_numpy(dtype=np.int8)
    builder = FeatureBuilder(
        feature_set="hierarchical_v2",
        drop_columns=["season"],
    ).fit(frame, target)
    expected = builder.raw_transform(frame)
    for column in builder.categorical_columns:
        expected[column] = (
            expected[column].astype("string").fillna("__MISSING__").astype(str)
        )
    for column in expected.columns:
        if column not in builder.categorical_columns:
            expected[column] = pd.to_numeric(expected[column], errors="coerce").astype(
                "float32"
            )

    actual = module.build_features(
        frame.drop(columns="control_success"),
        builder.to_spec(),
        encode_categories=False,
    )
    assert actual.columns.tolist() == expected.columns.tolist()
    for column in expected.columns:
        if column in builder.categorical_columns:
            assert actual[column].tolist() == expected[column].tolist()
        else:
            np.testing.assert_allclose(
                actual[column].to_numpy(),
                expected[column].to_numpy(),
                equal_nan=True,
            )
