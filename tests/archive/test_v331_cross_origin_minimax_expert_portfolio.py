import numpy as np

from src.archive.v331_cross_origin_minimax_expert_portfolio import (
    EXPERT_ORDER,
    ORIGINS,
    PROTOCOL,
    apply_portfolio,
    form_gain,
    quadratic_gain_form,
)


def test_v331_protocol_and_dimensions() -> None:
    assert PROTOCOL == "V331_CROSS_ORIGIN_MINIMAX_EXPERT_PORTFOLIO_V1"
    assert ORIGINS == ("full_2022", "late_2023", "full_2024")
    assert len(EXPERT_ORDER) == 8


def test_quadratic_form_matches_direct_bss_gain() -> None:
    target = np.asarray([0.0, 1.0, 0.0, 1.0])
    parent = np.asarray([0.4, 0.6, 0.4, 0.6])
    directions = np.column_stack([
        np.asarray([-0.02, 0.02, -0.02, 0.02]),
        np.asarray([0.01, -0.01, 0.01, -0.01]),
    ])
    weight = np.asarray([0.5, 0.25])
    form = quadratic_gain_form(target, parent, directions, np.ones(4, dtype=bool))
    candidate = parent + directions @ weight
    rate = target.mean()
    direct = 100000.0 * (
        np.mean(np.square(target - parent)) - np.mean(np.square(target - candidate))
    ) / (rate * (1.0 - rate))
    assert np.isclose(form_gain(weight, form), direct)


def test_apply_portfolio_protects_nonregular_rows() -> None:
    parent = np.asarray([0.4, 0.5, 0.6])
    directions = np.ones((3, 2)) * 0.01
    candidate, active = apply_portfolio(
        parent, directions, np.asarray([0.5, 0.5]), np.asarray([True, False, True])
    )
    assert candidate[1] == parent[1]
    assert not active[1]
    assert np.all(candidate[[0, 2]] > parent[[0, 2]])
