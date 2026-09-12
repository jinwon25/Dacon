from __future__ import annotations

import numpy as np
import pandas as pd

from src.champion.v22_failure_mode_profiles import _group_probability


def test_group_probability_uses_hierarchical_prior_for_unknown_group() -> None:
    history = pd.DataFrame({"pitcher_id": [1, 1, 2]})
    label = np.array([0, 0, 1], dtype=np.int8)
    query = pd.DataFrame({"pitcher_id": [1, 3]})
    prior = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.10, 0.20, 0.30, 0.40],
        ]
    )
    probability, n = _group_probability(
        history, label, query, ("pitcher_id",), prior, alpha=2.0
    )
    assert np.allclose(n, [2.0, 0.0])
    assert np.allclose(probability[0], [2.5 / 4.0, 0.5 / 4.0, 0.5 / 4.0, 0.5 / 4.0])
    assert np.allclose(probability[1], prior[1])
