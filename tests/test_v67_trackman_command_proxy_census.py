from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v67_trackman_command_proxy_census import (
    build_trackman_physical_profile,
    pitcher_residual_table,
    weighted_correlation,
)


def test_weighted_correlation_has_expected_extremes() -> None:
    x = pd.Series([1.0, 2.0, 3.0])
    w = pd.Series([1.0, 2.0, 3.0])
    assert np.isclose(weighted_correlation(x, x, w), 1.0)
    assert np.isclose(weighted_correlation(x, -x, w), -1.0)


def test_within_pitch_sd_is_not_inflated_by_pitch_family_means() -> None:
    frame = pd.DataFrame(
        {
            "season": [2022] * 6,
            "pitcher_trackman_id": [7] * 6,
            "pitch_type_group": ["fastball"] * 3 + ["breaking"] * 3,
            "rel_speed": [9.0, 10.0, 11.0, 89.0, 90.0, 91.0],
            "spin_rate": [1.0] * 6,
            "induced_vert_break": [1.0] * 6,
            "horz_break": [1.0] * 6,
            "extension": [1.0] * 6,
            "rel_height": [1.0] * 6,
            "rel_side": [1.0] * 6,
            "zone_speed": [1.0] * 6,
        }
    )
    profile = build_trackman_physical_profile(frame)
    assert profile.loc[0, "tm_rel_speed_std"] > 40.0
    assert np.isclose(profile.loc[0, "tm_within_pitch_rel_speed_std"], 1.0)


def test_pitcher_residual_eb_uses_only_selected_domain() -> None:
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 1, 2],
            "target": [1.0, 0.0, 1.0],
            "domain3": ["R_ANCHOR", "F", "R_ANCHOR"],
        }
    )
    output = pitcher_residual_table(
        frame, np.array([0.5, 0.5, 0.25]), "R_ANCHOR", residual_prior=1.0
    ).set_index("pitcher_id")
    assert output.loc[1, "rows"] == 1
    assert np.isclose(output.loc[1, "residual_eb"], 0.25)
    assert np.isclose(output.loc[2, "residual_eb"], 0.375)
