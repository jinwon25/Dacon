from __future__ import annotations

import numpy as np
import pytest

from src.champion.v56_shared_horizon_fm import (
    equal_period_domain_centres,
    source_domain_weights,
)


def test_source_domain_equal_weights_have_equal_group_totals() -> None:
    periods = np.array([2020, 2020, 2020, 2021, 2021, 2021, 2021])
    domains = np.array(["R", "R", "F", "R", "F", "F", "F"])
    weight = source_domain_weights(periods, domains, "source_domain_equal")
    totals = []
    for period in (2020, 2021):
        for domain in ("R", "F"):
            mask = (periods == period) & (domains == domain)
            totals.append(weight[mask].sum())
    assert np.allclose(totals, totals[0])
    assert np.isclose(weight.mean(), 1.0)


def test_equal_period_domain_centres_ignore_period_row_counts() -> None:
    values = np.array([1.0, 1.0, 1.0, 3.0, -2.0, 2.0, 2.0])
    periods = np.array([2020, 2020, 2020, 2021, 2020, 2021, 2021])
    domains = np.array(["R", "R", "R", "R", "F", "F", "F"])
    centres, groups = equal_period_domain_centres(values, periods, domains)
    assert centres["R"] == 2.0
    assert centres["F"] == 0.0
    assert groups["2020:R"] == 1.0
    assert groups["2021:R"] == 3.0


def test_shared_risk_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        source_domain_weights(np.zeros(2), np.zeros(3), "uniform")
    with pytest.raises(ValueError):
        source_domain_weights(np.zeros(2), np.zeros(2), "future")
