from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.archive.train_v14_refinement import _trend_features
from src.archive.v10_overlay_script import _v14_trend_matrix


def test_v14_trend_training_and_inference_features_match(project_dir=None):
    # The artifact maps were fit on all 2024 rows.  A small source-season slice
    # is sufficient to prove that training and packaged inference implement the
    # same row-local transformation and column order.
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    train_path = project / "data" / "train.csv"
    preprocess_path = (
        project
        / "artifacts"
        / "v14_refinement_final_20260815"
        / "v14_refinement_preprocess.joblib"
    )
    if not train_path.is_file() or not preprocess_path.is_file():
        pytest.skip("requires DACON train.csv and Git-ignored v14 artifact")
    frame = _add_domain_and_pressure(
        pd.read_csv(train_path, nrows=128, low_memory=False)
    ).drop(columns="control_success")
    preprocess = joblib.load(preprocess_path)
    expected = _trend_features(frame, preprocess["trend_category_maps"])
    actual = _v14_trend_matrix(frame, preprocess)
    assert actual.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(
        actual.to_numpy(np.float64),
        expected.to_numpy(np.float64),
        equal_nan=True,
    )
