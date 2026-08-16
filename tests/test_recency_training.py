import numpy as np
import pytest

from src.recency_training import season_decay_weights


def test_season_decay_weights_are_mean_one_and_half_by_age():
    weights = season_decay_weights(np.array([2022, 2023, 2024]), half_life=1.0)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[1] / weights[2] == pytest.approx(0.5)
    assert weights[0] / weights[2] == pytest.approx(0.25)


def test_season_decay_weights_reject_invalid_inputs():
    with pytest.raises(ValueError):
        season_decay_weights(np.array([]), half_life=1.0)
    with pytest.raises(ValueError):
        season_decay_weights(np.array([2024]), half_life=0.0)
