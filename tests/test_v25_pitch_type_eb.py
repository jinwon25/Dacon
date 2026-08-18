import numpy as np
import pandas as pd

from src.v25_pitch_type_eb_student import _group_probability, _normalise


def test_group_probability_shrinks_to_row_prior() -> None:
    history = pd.DataFrame({"pitcher_id": [1, 1, 2]})
    label = np.asarray([0, 0, 1])
    query = pd.DataFrame({"pitcher_id": [1, 3]})
    prior = np.asarray([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
    probability, n = _group_probability(
        history, label, query, ("pitcher_id",), prior, alpha=2.0
    )
    np.testing.assert_allclose(n, [2.0, 0.0])
    np.testing.assert_allclose(probability[1], prior[1])
    assert probability[0, 0] > prior[0, 0]
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_normalise_handles_small_values() -> None:
    output = _normalise(np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]))
    np.testing.assert_allclose(output.sum(axis=1), 1.0)
