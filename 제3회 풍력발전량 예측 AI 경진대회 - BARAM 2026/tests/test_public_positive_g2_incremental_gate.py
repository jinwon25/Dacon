from __future__ import annotations

import argparse

import pytest

from experiments.public_positive_g2_incremental_gate import parse_weights


def test_parse_weights_accepts_strict_grid_above_anchor() -> None:
    assert parse_weights("0.185, 0.20,0.235") == (0.185, 0.20, 0.235)


@pytest.mark.parametrize(
    "value",
    ("", "0.1825", "0.20,0.19", "0.20,0.20"),
)
def test_parse_weights_rejects_invalid_grid(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_weights(value)
