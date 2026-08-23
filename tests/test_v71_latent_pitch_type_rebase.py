from __future__ import annotations

import numpy as np

from src.archive.v71_latent_pitch_type_rebase import _evaluate


def test_evaluate_adds_frozen_shift_to_new_parent() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "target": [1.0, 0.0, 1.0],
            "game_month": [4, 4, 4],
            "domain3": ["R_CORE", "R_ANCHOR", "F"],
        }
    )
    parent = np.array([0.5, 0.5, 0.5])
    result, candidate = _evaluate(frame, parent, np.array([0.1, -0.1, 0.1]))
    assert np.allclose(candidate, [0.6, 0.4, 0.6])
    assert result["gain"] > 0
