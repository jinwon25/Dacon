from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v203_h1_workload_feature_screen import (
    apply_delta,
    h1_top_delta,
    restrictions,
)


def test_h1_top_delta_uses_affine_and_route_weights() -> None:
    frame = pd.DataFrame({"num_runners_on": [0, 1], "li": [1.0, 1.0]})
    delta = h1_top_delta(
        np.array([0.55, 0.55]),
        np.array([0.56, 0.56]),
        np.zeros(2),
        frame,
    )
    assert delta[1] > delta[0] > 0.0


def test_apply_delta_protects_noncore_and_inexact_rows() -> None:
    candidate, active = apply_delta(
        np.array([0.5, 0.5, 0.5]),
        np.array([0.01, 0.01, 0.01]),
        np.array([True, False, True]),
        np.array(["R_CORE", "R_CORE", "F"]),
        0.5,
    )
    assert active.tolist() == [True, False, False]
    assert np.allclose(candidate, [0.505, 0.5, 0.5])


def test_v203_requires_three_seed_confirmation() -> None:
    audit = restrictions()
    assert audit["paired_same_h1_recipe_and_seed"]
    assert audit["three_seed_confirmation_required"]
    assert audit["strictly_prior_season_fits"]
    assert audit["row_local_inference"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
