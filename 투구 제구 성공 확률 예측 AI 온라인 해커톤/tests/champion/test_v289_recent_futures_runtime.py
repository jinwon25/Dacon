import json

import numpy as np
import pandas as pd

from src.champion import v289_recent_futures_runtime as runtime


def test_recent_futures_runtime_only_scores_f_rows(tmp_path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "row_id": ["A", "B"],
            "season": [2025, 2025],
            "game_type": ["R", "F"],
            "top_bottom": ["T", "B"],
            "base_state": ["___", "1__"],
            "pitcher_hand": [1, 2],
            "batter_hand": [2, 2],
            "pitcher_team_id": [1, 2],
            "batter_team_id": [2, 1],
            "pitcher_id": [10, 11],
            "batter_id": [20, 21],
            "balls_before": [0, 1],
            "strikes_before": [0, 2],
            "num_runners_on": [0, 1],
            "li": [1.0, 2.0],
        }
    )
    columns = [
        "game_type",
        "pitcher_id",
        "balls_before",
        "count_code",
        "same_hand",
        "pressure_code",
    ]
    (tmp_path / "feature_columns.json").write_text(
        json.dumps(columns), encoding="utf-8"
    )
    for seed in runtime.SEEDS:
        (tmp_path / f"futures_expert_seed{seed}.cbm").write_text("x")

    class FakeModel:
        def load_model(self, _path):
            return None

        def predict_proba(self, features):
            assert len(features) == 1
            return np.asarray([[0.4, 0.6]])

    monkeypatch.setattr(runtime, "CatBoostClassifier", FakeModel)
    prediction = runtime.predict(frame, tmp_path)
    assert np.isnan(prediction[0])
    np.testing.assert_allclose(prediction[1], 0.6)
