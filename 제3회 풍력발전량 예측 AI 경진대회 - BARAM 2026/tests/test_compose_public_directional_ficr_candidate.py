from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.compose_public_directional_ficr_candidate import (
    DirectionalStepPolicy,
    apply_directional_step,
    complementary_factor_frames,
    exact_public_composition,
)


def test_directional_step_gates_and_caps() -> None:
    baseline = np.array([1.0, 4.0, 9.5])
    signal = np.array([-2.0, 2.0, 2.0])
    candidate, gate = apply_directional_step(
        baseline,
        signal,
        capacity=10.0,
        policy=DirectionalStepPolicy(
            direction="positive",
            step_ratio=0.05,
            minimum_prediction_ratio=0.30,
            minimum_signal_ratio=0.10,
        ),
    )

    assert gate.tolist() == [False, True, True]
    assert np.allclose(candidate, [1.0, 4.5, 10.0])


def test_complementary_factor_frames_recompose_target() -> None:
    incumbent = pd.DataFrame(
        {
            "forecast_id": ["a", "b"],
            "forecast_kst_dtm": ["t1", "t2"],
            "kpx_group_1": [1.0, 2.0],
            "kpx_group_2": [3.0, 4.0],
            "kpx_group_3": [5.0, 6.0],
        }
    )
    target = incumbent.copy()
    target[["kpx_group_1", "kpx_group_2", "kpx_group_3"]] += 10.0

    group1, group23 = complementary_factor_frames(incumbent, target)

    assert group1["kpx_group_1"].equals(target["kpx_group_1"])
    assert group1["kpx_group_2"].equals(incumbent["kpx_group_2"])
    assert group23["kpx_group_1"].equals(incumbent["kpx_group_1"])
    assert group23["kpx_group_2"].equals(target["kpx_group_2"])
    assert group23["kpx_group_3"].equals(target["kpx_group_3"])


def test_exact_public_composition_is_componentwise_additive() -> None:
    incumbent = {"score": 0.64, "one_minus_nmae": 0.87, "ficr": 0.41}
    group1 = {"score": 0.642, "one_minus_nmae": 0.869, "ficr": 0.415}
    group23 = {"score": 0.645, "one_minus_nmae": 0.872, "ficr": 0.418}

    result = exact_public_composition(incumbent, group1, group23)

    assert np.isclose(result["score"], 0.647)
    assert np.isclose(result["one_minus_nmae"], 0.871)
    assert np.isclose(result["ficr"], 0.423)
