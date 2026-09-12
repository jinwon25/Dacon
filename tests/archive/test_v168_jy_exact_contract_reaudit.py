from __future__ import annotations

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import affine, c3_mix


def test_affine_matches_runtime_order() -> None:
    prediction = np.array([0.40, 0.55, 0.70])
    expected = 0.5854452601930041 + 1.09 * (prediction - 0.5854452601930041)
    np.testing.assert_allclose(affine(prediction), expected, atol=0.0, rtol=0.0)


def test_c3_mix_endpoints_and_deployed_weight() -> None:
    sign = np.array([-0.02, 0.03])
    recent = np.array([0.04, -0.01])
    np.testing.assert_array_equal(c3_mix(sign, recent, 0.0), sign)
    np.testing.assert_array_equal(c3_mix(sign, recent, 1.0), recent)
    np.testing.assert_allclose(c3_mix(sign, recent, 0.15), 0.85 * sign + 0.15 * recent)
