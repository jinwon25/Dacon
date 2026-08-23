from __future__ import annotations

import numpy as np
import pandas as pd

from src.v113_fine_pitch_failure_prior_v104 import reconstruct_failure_components


def test_reconstruct_failure_components_uses_next_asof_snapshot() -> None:
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 2, 1, 2, 1, 2],
            "asof_pitcher_n": [0, 0, 1, 1, 2, 2],
            "asof_pitcher_reverse_rate": [0.0, 0.0, 1.0, 0.0, 0.5, 0.0],
            "asof_pitcher_middle_rate": [0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
            "control_success": [0, 0, 0, 1, 1, 1],
        }
    )
    result = reconstruct_failure_components(frame)
    assert np.array_equal(result["reverse"].iloc[:2].to_numpy(), [1.0, 0.0])
    assert np.array_equal(result["middle"].iloc[:2].to_numpy(), [0.0, 1.0])
    assert np.array_equal(result["wayoff"].iloc[:2].to_numpy(), [0.0, 0.0])
    assert result.iloc[-2:].isna().all().all()


def test_wayoff_is_failure_without_reverse_or_middle_increment() -> None:
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 1],
            "asof_pitcher_n": [4, 5],
            "asof_pitcher_reverse_rate": [0.2, 0.16],
            "asof_pitcher_middle_rate": [0.2, 0.16],
            "control_success": [0, 1],
        }
    )
    result = reconstruct_failure_components(frame)
    assert result.loc[0, "reverse"] == 0.0
    assert result.loc[0, "middle"] == 0.0
    assert result.loc[0, "wayoff"] == 1.0
