from __future__ import annotations

import numpy as np

from src.v122_anchor_public_quadratic import fit_public_quadratic


OBSERVATIONS = [
    {"eta": 0.075, "public_score": 1155.8293405409},
    {"eta": 0.10, "public_score": 1156.6153781694},
    {"eta": 0.15, "public_score": 1157.9736407889},
]


def test_exact_public_quadratic_recovers_observations_and_vertex() -> None:
    result = fit_public_quadratic(OBSERVATIONS, (0.15, 0.50))
    assert result["fit_max_abs"] < 1e-9
    assert np.isclose(result["raw_vertex_eta"], 0.3632218788686338)
    assert np.isclose(result["selected_curve_score"], 1160.5658236508777)


def test_public_quadratic_rejects_convex_curve() -> None:
    observations = [
        {"eta": 0.0, "public_score": 0.0},
        {"eta": 0.5, "public_score": 0.25},
        {"eta": 1.0, "public_score": 1.0},
    ]
    try:
        fit_public_quadratic(observations, (0.0, 1.0))
    except ValueError as exc:
        assert "concave" in str(exc)
    else:
        raise AssertionError("convex Public curve was accepted")
