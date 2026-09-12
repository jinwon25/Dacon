from __future__ import annotations

import numpy as np
import pytest

from src.archive.v171_independent_domain_residual_audit import domain_mask


def test_domain_mask_is_exact_and_boolean() -> None:
    axis = {"domain3": np.array(["R_CORE", "F", "R_ANCHOR", "R_CORE"])}
    result = domain_mask(axis, "R_CORE")
    np.testing.assert_array_equal(result, np.array([True, False, False, True]))
    assert result.dtype == np.bool_


def test_domain_mask_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        domain_mask({"domain3": np.array(["R_CORE"])}, "ALL")
