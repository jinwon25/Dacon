import numpy as np
import pandas as pd

from src.archive.v110_v104_cross_architecture_rebase import apply_correction, fit_alpha
from src.champion.v104_source_stability_mask import context_labels, policy_mask


def test_fit_alpha_recovers_bounded_probability_dose():
    parent = np.array([0.3, 0.5, 0.7])
    delta = np.array([0.1, -0.1, 0.05])
    target = parent + 0.4 * delta
    assert np.isclose(fit_alpha([target], [parent], [delta], 1.0), 0.4)
    assert fit_alpha([target], [parent], [delta], 0.25) == 0.25


def test_apply_correction_is_bounded():
    result = apply_correction(
        np.array([0.002, 0.998]), np.array([-1.0, 1.0]), alpha=1.0
    )
    assert np.array_equal(result, np.array([0.001, 0.999]))


def test_existing_majority_mask_can_define_disjoint_inactive_route():
    frame = pd.DataFrame(
        {
            "balls_before": [0, 0],
            "strikes_before": [0, 1],
            "asof_pitcher_n": [500, 5],
            "pitcher_hand": [2, 1],
            "batter_hand": [2, 2],
        }
    )
    safe = {
        "count": ["0-0"],
        "history": ["200-999"],
        "platoon": ["2-2"],
    }
    active = policy_mask(context_labels(frame), safe, "majority_two")
    assert np.array_equal(active, np.array([True, False]))
    assert np.array_equal(~active, np.array([False, True]))
