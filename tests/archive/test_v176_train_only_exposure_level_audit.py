from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v176_train_only_exposure_level_audit import (
    apply_direction,
    entity_exposure_direction,
)


def test_exposure_direction_uses_history_only_and_unknown_is_neutral() -> None:
    history = pd.DataFrame({"batter_id": [1, 1, 1, 2]})
    query = pd.DataFrame({"batter_id": [1, 2, 3]})
    direction = entity_exposure_direction(
        history, query, "batter_id", log_transform=False
    )
    assert direction[0] > 0.0
    assert direction[1] < 0.0
    assert direction[2] == 0.0


def test_apply_direction_is_probability_additive_and_clipped() -> None:
    base = np.array([0.2, 0.999])
    direction = np.array([2.0, 2.0])
    np.testing.assert_allclose(
        apply_direction(base, direction, 0.1), [0.4, 0.999]
    )
