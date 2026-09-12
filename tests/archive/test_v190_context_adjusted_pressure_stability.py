import numpy as np
import pandas as pd

from src.archive.v190_context_adjusted_pressure_stability import (
    context_adjusted_residual,
    stable_pressure_features,
)


def _history() -> pd.DataFrame:
    rows = 40
    return pd.DataFrame(
        {
            "season": np.repeat([2019, 2020], rows // 2),
            "pitcher_id": np.tile(np.repeat([1, 2], 10), 2),
            "control_success": np.tile([0, 1], rows // 2),
            "balls_before": np.arange(rows) % 4,
            "strikes_before": np.arange(rows) % 3,
            "outs_before": np.arange(rows) % 3,
            "inning": np.arange(rows) % 9 + 1,
            "li": np.where(np.arange(rows) % 2, 2.0, 0.5),
            "score_diff_pitcher_team": np.arange(rows) % 5 - 2,
            "num_runners_on": np.arange(rows) % 3,
            "runner_on_2b": (np.arange(rows) % 3 == 1).astype(int),
            "runner_on_3b": (np.arange(rows) % 3 == 2).astype(int),
            "base_state": (np.arange(rows) % 4).astype(str),
            "pitcher_hand": np.arange(rows) % 2 + 1,
            "batter_hand": (np.arange(rows) + 1) % 2 + 1,
            "top_bottom": np.where(np.arange(rows) % 2, "T", "B"),
        }
    )


def test_context_adjusted_residual_is_finite() -> None:
    residual = context_adjusted_residual(_history())
    assert residual.shape == (40,)
    assert np.isfinite(residual).all()


def test_query_labels_do_not_affect_pressure_features() -> None:
    history = _history()
    query = history.iloc[:5].drop(columns="control_success")
    first = stable_pressure_features(query, history, 300.0)
    changed = query.copy()
    second = stable_pressure_features(changed, history, 300.0)
    np.testing.assert_allclose(first, second)
