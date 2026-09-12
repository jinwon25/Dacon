from __future__ import annotations

import numpy as np

from experiments.public_factor_response_audit import derive_factor_response


def test_single_group_delta_is_three_times_macro_delta() -> None:
    control = {
        "score": 0.60,
        "one_minus_nmae": 0.80,
        "ficr": 0.40,
    }
    treatment = {
        "score": 0.61,
        "one_minus_nmae": 0.802,
        "ficr": 0.418,
    }
    result = derive_factor_response(
        treatment,
        control,
        targets=("kpx_group_1",),
    )
    assert np.isclose(result["affected_group_mean_delta"]["score"], 0.03)
    assert result["score_identity_passed"]


def test_two_group_delta_uses_three_halves_multiplier() -> None:
    control = {
        "score": 0.60,
        "one_minus_nmae": 0.80,
        "ficr": 0.40,
    }
    treatment = {
        "score": 0.59,
        "one_minus_nmae": 0.794,
        "ficr": 0.386,
    }
    result = derive_factor_response(
        treatment,
        control,
        targets=("kpx_group_2", "kpx_group_3"),
    )
    assert np.isclose(result["affected_group_mean_delta"]["score"], -0.015)
    assert result["score_identity_passed"]
