import numpy as np
import pandas as pd

from experiments.cross_group_transfer import transfer_features


def test_transfer_features_shape_and_group_columns() -> None:
    group_1 = np.asarray([0.2, 0.4])
    group_2 = np.asarray([0.3, 0.5])
    timestamps = pd.DatetimeIndex(["2025-01-01 01:00", "2025-07-01 13:00"])

    result = transfer_features(group_1, group_2, timestamps)

    assert result.shape == (2, 8)
    np.testing.assert_allclose(result[:, 0], group_1)
    np.testing.assert_allclose(result[:, 1], group_2)
    np.testing.assert_allclose(result[:, 3], group_2 - group_1)
