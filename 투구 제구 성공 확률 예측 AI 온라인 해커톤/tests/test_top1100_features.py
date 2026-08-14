import numpy as np
import pandas as pd

from src.top1100_features import build_features
from tests.test_features import _main_rows


def test_top1100_features_are_target_free_and_finite():
    frame = _main_rows().assign(control_success=[1, 0])
    features = build_features(frame, frame.iloc[:1], include_ids=True)
    assert "control_success" not in features.columns
    assert features.select_dtypes(include="object").empty
    numeric = features.select_dtypes(exclude="category")
    assert np.isfinite(numeric.to_numpy(dtype="float64")).all()
    assert (features["pitcher_season_n"] >= 0).all()


def test_top1100_features_can_exclude_entity_ids():
    frame = _main_rows()
    features = build_features(frame, frame.iloc[:1], include_ids=False)
    assert "pitcher_id" not in features.columns
    assert "batter_id" not in features.columns
    assert "count_platoon" in features.columns


def test_season_state_uses_immediately_previous_season_endpoint():
    rows = []
    for season, n, rate in [(2020, 100, 0.40), (2021, 250, 0.60), (2022, 300, 0.55)]:
        row = _main_rows().iloc[0].copy()
        row["season"] = season
        row["pitcher_id"] = "P1"
        row["batter_id"] = "B1"
        row["asof_pitcher_n"] = n
        row["asof_pitcher_success_rate"] = rate
        row["asof_batter_n"] = n
        row["asof_batter_success_rate"] = rate
        row["asof_pitcher_pitchmix_n"] = n
        rows.append(row)
    history = pd.DataFrame(rows)
    features = build_features(history.iloc[[2]].reset_index(drop=True), history, include_ids=False)

    # 2022 must subtract the 2021 endpoint (250), not the 2020 endpoint (100).
    assert features.loc[0, "pitcher_season_n"] == 50
    assert features.loc[0, "batter_season_n"] == 50
    assert np.isclose(features.loc[0, "pitcher_prior_rate"], (150 + 20 * 0.52) / 270)
