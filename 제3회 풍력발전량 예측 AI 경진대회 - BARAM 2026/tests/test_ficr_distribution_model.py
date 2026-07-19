import numpy as np

from experiments.ficr_distribution_model import ficr_decision


def test_ficr_decision_returns_bounded_point_per_row() -> None:
    quantiles = np.asarray(
        [
            [3000, 5000, 7000, 9000, 11000],
            [8000, 9000, 10000, 11000, 12000],
        ],
        dtype=float,
    )
    result = ficr_decision(quantiles, capacity=21000.0, mean_generation=8000.0)

    assert result.shape == (2,)
    assert np.all(result >= 0.0)
    assert np.all(result <= 21000.0)
