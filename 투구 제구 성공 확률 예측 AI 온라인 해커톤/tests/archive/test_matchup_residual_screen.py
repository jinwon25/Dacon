import numpy as np
import pandas as pd

from src.archive.matchup_residual_screen import _map_effect


def test_map_effect_unseen_key_has_zero_correction():
    stats = pd.DataFrame(
        {"weighted_residual": [2.0], "effective_count": [8.0]},
        index=pd.Index(["1\x1f10"], name="key"),
    )
    rows = pd.DataFrame({"pitcher_id": [1, 2], "batter_id": [10, 20]})
    effect, seen = _map_effect(
        stats, rows, ("pitcher_id", "batter_id"), alpha=2.0
    )
    assert np.allclose(effect, [0.2, 0.0])
    assert seen.tolist() == [True, False]
