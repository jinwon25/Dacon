import numpy as np
import pandas as pd

from src.v104_source_stability_mask import context_labels, policy_mask


def test_context_labels_are_row_local():
    frame = pd.DataFrame(
        {
            "balls_before": [2, 3],
            "strikes_before": [0, 2],
            "pitcher_hand": [1, 2],
            "batter_hand": [2, 1],
            "asof_pitcher_n": [100, 1500],
        }
    )
    labels = context_labels(frame)
    assert labels["count"].tolist() == ["2-0", "3-2"]
    assert labels["history"].tolist() == ["30-199", "1000+"]
    assert labels["platoon"].tolist() == ["1-2", "2-1"]


def test_policy_masks_combine_frozen_safe_sets():
    labels = {
        "count": np.array(["a", "a", "b"]),
        "history": np.array(["h", "x", "h"]),
        "platoon": np.array(["p", "x", "p"]),
    }
    safe = {"count": ["a"], "history": ["h"], "platoon": ["p"]}
    assert policy_mask(labels, safe, "all_three").tolist() == [True, False, False]
    assert policy_mask(labels, safe, "majority_two").tolist() == [True, False, True]
