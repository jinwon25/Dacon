import numpy as np
import pandas as pd

from src.archive.trackman_latent_integration import _pitch_weights


def test_pitch_weights_are_row_local_and_sum_to_one():
    rows = pd.DataFrame(
        {
            "pitcher_id": [1, 2],
            "asof_pitcher_fastball_rate": [0.6, np.nan],
            "asof_pitcher_breaking_rate": [0.3, np.nan],
            "asof_pitcher_offspeed_rate": [0.1, np.nan],
        }
    )
    counts = pd.DataFrame(
        {"fastball": [60], "breaking": [30], "offspeed": [5], "other": [5]},
        index=pd.Index([1], name="pitcher_id"),
    )
    weights, reliability = _pitch_weights(rows, counts)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(weights[0], [0.57, 0.285, 0.095, 0.05])
    assert reliability[0] > 0 and reliability[1] == 0
