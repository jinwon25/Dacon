import numpy as np
import pandas as pd

from src.archive.v301_multidomain_futures_expert import compose


def test_compose_replaces_only_futures_at_fixed_dose():
    frame = pd.DataFrame({"game_type": ["R", "F", "F"]})
    parent = np.array([0.4, 0.5, 0.6])
    incumbent = np.array([0.41, 0.52, 0.58])
    expert = np.array([0.7, 0.2])
    result = compose(parent, incumbent, frame, expert)
    np.testing.assert_allclose(result, [0.41, 0.52, 0.56])
