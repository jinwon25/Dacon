import numpy as np
import pandas as pd
import pytest

from src.v33_logodds_asof_prior import (
    apply_direction,
    logodds_matchup,
    posterior_rate,
)


def test_logodds_matchup_preserves_domain_fixed_point():
    prior = np.array([0.35, 0.50, 0.70])
    combined = logodds_matchup(prior, prior, prior, 1.0, 0.5)
    assert np.allclose(combined, prior)


def test_posterior_rate_uses_prior_for_cold_start_and_rate_with_support():
    prior = np.array([0.4, 0.4])
    result = posterior_rate(
        np.array([np.nan, 0.8]), np.array([0.0, 1_000_000.0]), prior, 100.0
    )
    assert result[0] == pytest.approx(0.4)
    assert result[1] == pytest.approx(0.8, abs=1e-4)


def test_apply_direction_changes_only_selected_domain():
    frame = pd.DataFrame(
        {
            "v22": [0.4, 0.4],
            "v25": [0.4, 0.4],
            "domain3": ["R_CORE", "F"],
        }
    )
    candidate, mask = apply_direction(
        frame, np.array([0.2, 0.2]), eta=0.1, domain="R_CORE"
    )
    assert mask.tolist() == [True, False]
    assert candidate.tolist() == pytest.approx([0.42, 0.4])
