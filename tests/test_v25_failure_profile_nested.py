import numpy as np

from src.v25_failure_profile_nested_screen import compose, diagnostics


def test_compose_directions_preserve_v22_at_zero_weight() -> None:
    v21 = np.asarray([0.2, 0.8])
    v22 = np.asarray([0.3, 0.7])
    raw = np.asarray([0.5, 0.4])
    for direction in ("blend_to_raw", "add_v21_residual"):
        np.testing.assert_allclose(compose(v21, v22, raw, 0.0, direction), v22)
    np.testing.assert_allclose(
        compose(v21, v22, raw, 0.5, "blend_to_raw"), [0.4, 0.55]
    )
    np.testing.assert_allclose(
        compose(v21, v22, raw, 0.5, "add_v21_residual"), [0.45, 0.5]
    )


def test_diagnostics_reports_all_groups() -> None:
    target = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    parent = np.full(6, 0.5)
    candidate = np.asarray([0.4, 0.6, 0.4, 0.6, 0.4, 0.6])
    month = np.asarray([3, 3, 4, 4, 5, 5])
    domain = np.asarray(["R_CORE", "R_CORE", "R_ANCHOR", "R_ANCHOR", "F", "F"])
    result = diagnostics(target, parent, candidate, month, domain)
    assert result["gain"] > 0.0
    assert result["positive_month_fraction"] == 1.0
    assert set(result["domain_gains"]) == {"R_CORE", "R_ANCHOR", "F"}
