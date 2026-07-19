from pathlib import Path

import pandas as pd

from src.feature_cache import load_or_build_features


def test_load_existing_feature_cache_without_rebuild(tmp_path: Path) -> None:
    expected = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="forecast_kst_dtm"),
    )
    expected.to_pickle(tmp_path / "features_train.pkl")

    actual = load_or_build_features("unused", "train", tmp_path)

    pd.testing.assert_frame_equal(actual, expected)
