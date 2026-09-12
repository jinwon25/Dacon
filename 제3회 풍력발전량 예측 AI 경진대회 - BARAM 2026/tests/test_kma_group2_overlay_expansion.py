from __future__ import annotations

from experiments.kma_group2_overlay_expansion import (
    select_largest_bounded_alpha,
)


def _record(alpha: float, score: float, nmae: float, ficr: float) -> dict:
    return {
        "alpha": alpha,
        "total_delta": {
            "score": score,
            "one_minus_nmae": nmae,
            "ficr": ficr,
        },
    }


def test_selects_largest_alpha_inside_component_floor() -> None:
    records = [
        _record(0.125, 0.001, -0.0001, 0.002),
        _record(0.20, 0.001, -0.00034, 0.002),
        _record(0.25, 0.002, -0.00036, 0.004),
    ]

    selected = select_largest_bounded_alpha(records)

    assert selected is not None
    assert selected["alpha"] == 0.20


def test_rejects_expansion_without_minimum_score() -> None:
    records = [_record(0.20, 0.0001, 0.0, 0.0002)]

    assert select_largest_bounded_alpha(records) is None


def test_can_select_against_current_public_incumbent() -> None:
    records = [
        {
            **_record(0.225, 0.002, -0.0005, 0.004),
            "incremental": {
                "score": 0.0002,
                "one_minus_nmae": -0.0001,
                "ficr": 0.0005,
            },
        },
        {
            **_record(0.2375, 0.002, -0.0006, 0.004),
            "incremental": {
                "score": 0.0003,
                "one_minus_nmae": -0.0002,
                "ficr": 0.0008,
            },
        },
    ]

    selected = select_largest_bounded_alpha(
        records,
        delta_key="incremental",
    )

    assert selected is not None
    assert selected["alpha"] == 0.2375
