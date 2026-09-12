from __future__ import annotations

import json

import numpy as np
import pandas as pd

import src.archive.v10_overlay_script as inference


def test_v16_residual_is_row_local_and_core_only(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = {
        "method": "smoothed_oof_residual_mean",
        "group_columns": ["pitcher_id", "batter_hand"],
        "anchor_team_id": 13,
        "unseen_effect": 0.0,
        "effects": {"10\u001fR": 0.0125, "20\u001fL": -0.02},
    }
    (model_dir / "v16_residual_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    monkeypatch.setattr(inference, "MODEL_DIR", model_dir)
    frame = pd.DataFrame(
        {
            "pitcher_id": [10, 20, 10, 999],
            "batter_hand": ["R", "L", "R", "R"],
            "game_type": ["R", "R", "F", "R"],
            "pitcher_team_id": [1, 13, 1, 1],
            "batter_team_id": [2, 2, 2, 2],
        }
    )
    base = np.full(4, 0.5)
    actual = inference.apply_v16_residual(base, frame)
    np.testing.assert_allclose(actual, [0.5125, 0.5, 0.5, 0.5])

    split = np.concatenate(
        [
            inference.apply_v16_residual(base[::2], frame.iloc[::2]),
            inference.apply_v16_residual(base[1::2], frame.iloc[1::2]),
        ]
    )
    expected = np.concatenate([actual[::2], actual[1::2]])
    np.testing.assert_allclose(split, expected)


def test_v16_pressure_group_uses_count_state(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = {
        "method": "smoothed_oof_residual_mean",
        "group_columns": ["pitcher_id", "batter_hand", "pressure"],
        "anchor_team_id": 13,
        "unseen_effect": 0.0,
        "effects": {
            "10\u001fR\u001fthreeball": 0.01,
            "10\u001fR\u001ftwostrike": -0.02,
            "10\u001fR\u001fnormal": 0.003,
        },
    }
    (model_dir / "v16_residual_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    monkeypatch.setattr(inference, "MODEL_DIR", model_dir)
    frame = pd.DataFrame(
        {
            "pitcher_id": [10, 10, 10],
            "batter_hand": ["R", "R", "R"],
            "balls_before": [3, 1, 1],
            "strikes_before": [1, 2, 1],
            "game_type": ["R", "R", "R"],
            "pitcher_team_id": [1, 1, 1],
            "batter_team_id": [2, 2, 2],
        }
    )
    actual = inference.apply_v16_residual(np.full(3, 0.5), frame)
    np.testing.assert_allclose(actual, [0.51, 0.48, 0.503])
