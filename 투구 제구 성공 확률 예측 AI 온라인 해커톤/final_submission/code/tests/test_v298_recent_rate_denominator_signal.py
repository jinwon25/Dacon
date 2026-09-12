import numpy as np
import pandas as pd

from src.archive.v298_recent_rate_denominator_signal import (
    build_denominator_features,
    decode_shared_denominator,
)


def test_joint_denominator_recovers_coprime_counts() -> None:
    success = np.array([37 / 83, 51 / 101, np.nan])
    middle = np.array([14 / 83, 17 / 101, 0.2])
    denominator, ambiguity, error = decode_shared_denominator(success, middle, 150)
    assert denominator[:2].tolist() == [83.0, 101.0]
    assert np.all(ambiguity[:2] >= 1.0)
    assert np.all(error[:2] < 1e-4)
    assert np.isnan(denominator[2])


def test_feature_builder_is_row_local_and_finite_for_valid_rates() -> None:
    frame = pd.DataFrame(
        {
            "asof_pitcher_success_rate": [0.52, 0.48],
            "asof_pitcher_middle_rate": [0.16, 0.20],
            "inning": [1, 8],
            "balls_before": [0, 3],
            "strikes_before": [0, 2],
            **{
                f"asof_pitcher_prev{window}_game_success_rate": [37 / (83 * window), 0.5]
                for window in (1, 3, 5)
            },
            **{
                f"asof_pitcher_prev{window}_game_middle_rate": [14 / (83 * window), 0.2]
                for window in (1, 3, 5)
            },
        }
    )
    full = build_denominator_features(frame)
    first = build_denominator_features(frame.iloc[[0]].reset_index(drop=True))
    assert list(full.columns) == list(first.columns)
    assert np.allclose(full.iloc[0].to_numpy(), first.iloc[0].to_numpy(), equal_nan=True)
    assert np.isfinite(full.to_numpy()).all()
