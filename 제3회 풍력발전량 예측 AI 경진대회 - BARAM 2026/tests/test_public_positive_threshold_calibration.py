from __future__ import annotations

import numpy as np
import pytest

from experiments.public_positive_threshold_calibration import apply_policy


def test_apply_policy_supports_affine_threshold_and_piecewise() -> None:
    base = np.asarray([1_000.0, 5_000.0, 9_000.0])
    affine = apply_policy(
        base,
        {"kind": "affine", "scale": 1.05, "offset": -100.0},
    )
    assert affine.tolist() == pytest.approx([950.0, 5_150.0, 9_350.0])

    threshold = apply_policy(
        base,
        {"kind": "threshold_offset", "minimum_ratio": 0.2, "offset": 200.0},
    )
    assert threshold.tolist() == pytest.approx([1_000.0, 5_200.0, 9_200.0])

    piecewise = apply_policy(
        base,
        {
            "kind": "piecewise_offset",
            "low_offset": 100.0,
            "high_offset": -200.0,
            "breakpoint": 0.5,
        },
    )
    assert piecewise.tolist() == pytest.approx([1_100.0, 5_100.0, 9_100.0])


def test_apply_policy_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        apply_policy(
            np.asarray([1.0]),
            {"kind": "unknown"},
        )
