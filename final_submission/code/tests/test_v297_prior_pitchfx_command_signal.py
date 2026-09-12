from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v297_prior_pitchfx_command_signal import (
    make_pitchfx_profiles,
    optimal_dose,
)


def test_make_pitchfx_profiles_is_pitcher_local() -> None:
    frame = pd.DataFrame(
        {
            "game_pk": ["a", "a", "b", "b"],
            "pitcher": [10, 10, 20, 20],
            "pitch_type": ["FF", "SL", "FF", "FF"],
            "balls": [0, 1, 2, 3],
            "strikes": [0, 1, 2, 2],
            "stand": ["L", "R", "L", "R"],
            "plate_x": [0.0, 0.8, -0.2, 0.2],
            "plate_z": [2.5, 3.0, 2.0, 3.0],
            "sz_top": [3.5] * 4,
            "sz_bot": [1.5] * 4,
        }
    )
    profile = make_pitchfx_profiles(frame).set_index("pitcher_code")
    assert set(profile.index) == {10, 20}
    assert profile.loc[10, "pfx_n"] == 2
    assert profile.loc[20, "pfx_zone_rate"] == 1.0


def test_optimal_dose_is_clipped_and_improving_direction() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    parent = np.full(4, 0.5)
    signal = np.array([-0.1, 0.1, -0.1, 0.1])
    assert optimal_dose(y, parent, signal) == 1.0
    assert optimal_dose(y, parent, -signal) == 0.0
