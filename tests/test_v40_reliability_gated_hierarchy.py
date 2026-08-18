import numpy as np

from src.v40_reliability_gated_hierarchy import (
    formula_family,
    gated_name,
    reliability_raw,
    split_gated_name,
)


def test_formula_family_preserves_scope_and_prior_kind():
    assert formula_family("hier::domain_latest_k80_p75") == "domain_latest"
    assert formula_family("hier::global_blend_k40_p90") == "global_blend"


def test_reliability_gate_is_parent_identity_below_threshold():
    raw = np.array([0.2, 0.8, 0.6])
    parent = np.array([0.4, 0.5, 0.7])
    season_n = np.array([0.0, 75.0, 149.0])
    output = reliability_raw(raw, parent, season_n, 75.0)
    np.testing.assert_allclose(output, [0.4, 0.8, 0.6])


def test_gated_name_round_trip():
    name = gated_name("hier::domain_latest_k80_p75", 150.0)
    assert split_gated_name(name) == ("hier::domain_latest_k80_p75", 150.0)
