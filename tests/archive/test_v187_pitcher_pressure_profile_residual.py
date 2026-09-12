import numpy as np
import pandas as pd

from src.archive.v187_pitcher_pressure_profile_residual import (
    apply_correction,
    attach_profile,
)


def _frame(target=(1, 0, 1, 0)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": [1, 1, 2, 2], "control_success": target,
            "li": [2.0, 0.5, 4.0, 0.2], "inning": [8, 2, 9, 3],
            "score_diff_pitcher_team": [0, 4, -1, -5],
            "balls_before": [3, 0, 3, 1], "strikes_before": [2, 0, 1, 2],
            "num_runners_on": [2, 0, 1, 0], "runner_on_2b": [1, 0, 0, 0],
            "runner_on_3b": [0, 0, 1, 0],
        }
    )


def test_attach_profile_uses_only_supplied_history_labels() -> None:
    history = _frame()
    query = _frame(target=(0, 0, 0, 0)).drop(columns="control_success")
    first = attach_profile(query, history, 100.0)
    changed_query = query.copy()
    changed_query["pitcher_id"] = [1, 1, 2, 2]
    second = attach_profile(changed_query, history, 100.0)
    np.testing.assert_allclose(first, second)


def test_apply_correction_preserves_inactive_rows() -> None:
    base = np.array([0.4, 0.5, 0.6])
    correction = np.array([0.01, -0.02, 0.03])
    active = np.array([True, False, True])
    np.testing.assert_allclose(
        apply_correction(base, correction, active, 0.5), [0.405, 0.5, 0.615]
    )
