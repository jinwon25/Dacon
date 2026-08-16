from pathlib import Path

import numpy as np
import pandas as pd

import src.v10_overlay_script as inference
from src.train_v20_target1160 import _fit_eb, _read_indexed_rows


def test_v20_overlay_is_identity_without_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "MODEL_DIR", tmp_path)
    probability = np.asarray([0.4, 0.6], dtype=np.float64)
    frame = pd.DataFrame({"row_id": ["a", "b"]})
    actual = inference.apply_v20_target1160_overlay(probability, frame)
    np.testing.assert_array_equal(actual, probability)


def test_fit_eb_respects_anchor_domain_and_shrinkage():
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 1, 2, 2],
            "batter_hand": [1, 1, 2, 2],
            "pressure": ["normal"] * 4,
            "pitcher_team_id": [10, 10, 11, 11],
            "balls_before": [0, 0, 3, 3],
            "strikes_before": [0, 0, 1, 1],
            "pitcher_hand": [1, 1, 2, 2],
            "domain3": ["R_CORE", "R_CORE", "R_ANCHOR", "R_ANCHOR"],
        }
    )
    recipes = _fit_eb(frame, np.asarray([0.2, 0.2, 0.4, 0.4]))
    anchor = next(recipe for recipe in recipes if recipe["name"] == "count_hands_anchor")
    assert anchor["n_groups"] == 1
    effect = next(iter(anchor["effects"].values()))
    assert np.isclose(effect, 0.8 / 52.0)


def test_read_indexed_rows_restores_requested_order(tmp_path: Path):
    path = tmp_path / "rows.csv"
    pd.DataFrame({"a": np.arange(8), "b": np.arange(8) * 10}).to_csv(
        path, index=False
    )
    actual = _read_indexed_rows(path, np.asarray([6, 1, 4]), ["a", "b"])
    assert actual["a"].tolist() == [6, 1, 4]
    assert actual["b"].tolist() == [60, 10, 40]
