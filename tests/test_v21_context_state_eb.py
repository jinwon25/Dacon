from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.train_v21_context_state_eb import add_v21_features


def test_add_v21_features_has_fixed_row_local_bins() -> None:
    frame = pd.DataFrame(
        {
            "asof_pitcher_prev3_game_success_rate": [0.0, 0.549, 1.0, np.nan],
            "asof_pitcher_prev5_game_success_rate": [0.19, 0.81, 0.5, np.nan],
            "asof_pitcher_reverse_rate": [0.09, 0.51, 0.999, np.nan],
            "inning": [1, 5, 8, 10],
        }
    )
    result = add_v21_features(frame)
    assert result["prev3_b10"].tolist() == [0, 5, 10, 5]
    assert result["prev3_b20"].tolist() == [0, 10, 20, 10]
    assert result["prev5_b10"].tolist() == [1, 8, 5, 5]
    assert result["reverse_b10"].tolist() == [0, 5, 9, 5]
    assert result["inning_band"].astype(str).tolist() == [
        "early",
        "middle",
        "late",
        "extra",
    ]
