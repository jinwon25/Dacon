import numpy as np
import pandas as pd

from src.archive.v24_semantic_eda import career_counter_audit, level_mobility_summary


def test_career_counter_audit_recognizes_exact_pre_pitch_counts():
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 2, 1, 1],
            "batter_id": [9, 9, 8, 9],
            "control_success": [1, 0, 0, 1],
            "asof_pitcher_n": [0, 0, 1, 2],
            "asof_pitcher_success_rate": [np.nan, np.nan, 1.0, 0.5],
            "asof_batter_n": [0, 1, 0, 2],
            "asof_batter_success_rate": [np.nan, 1.0, np.nan, 0.5],
        }
    )
    result = career_counter_audit(frame).set_index("entity")
    assert result.loc["pitcher", "n_exact_fraction"] == 1.0
    assert result.loc["batter", "n_exact_fraction"] == 1.0
    assert result.loc["pitcher", "success_count_exact_fraction"] == 1.0


def test_level_mobility_reports_paired_pitchers():
    frame = pd.DataFrame(
        {
            "season": [2024] * 80,
            "pitcher_id": [1] * 40 + [2] * 40,
            "game_type": (["R"] * 20 + ["F"] * 20) * 2,
            "control_success": ([0] * 20 + [1] * 20) + ([1] * 40),
        }
    )
    result = level_mobility_summary(frame).iloc[0]
    assert result["both_levels"] == 2
    assert result["paired_n_ge20"] == 2
    assert result["paired_f_minus_r_mean"] == 0.5
