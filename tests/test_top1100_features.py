import numpy as np

from src.archive.top1100_features import build_features
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
