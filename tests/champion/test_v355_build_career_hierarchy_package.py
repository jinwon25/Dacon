import pandas as pd

from src.champion.v355_build_career_hierarchy_package import build_career_bundle


def test_career_bundle_half_scales_domain_sufficient_statistics():
    train = pd.DataFrame(
        {
            "season": [2022, 2023, 2023],
            "game_type": ["R", "R", "R"],
            "pitcher_team_id": [13, 13, 13],
            "batter_team_id": [16, 16, 16],
            "pitcher_id": [1, 1, 1],
            "batter_id": [2, 2, 2],
            "control_success": [1, 0, 1],
        }
    )
    bundle = build_career_bundle(train, prediction_year=2024)
    assert bundle["pitcher_id_latest_n"][(1, "R_ANCHOR")] == 1.5
    assert bundle["pitcher_id_latest_s"][(1, "R_ANCHOR")] == 1.0
    assert bundle["signal"] == "hier::domain_career_k80_p75"
