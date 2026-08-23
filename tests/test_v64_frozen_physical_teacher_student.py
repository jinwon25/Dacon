from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.archive.v64_frozen_physical_teacher_student import (
    FROZEN_DOMAIN,
    FROZEN_VARIANT,
    FROZEN_WEIGHT,
    apply_frozen_correction,
    verify_frozen_selection,
)


def test_frozen_correction_is_aligned_and_clipped() -> None:
    parent = np.array([0.2, 0.9])
    correction = np.array([0.1, 1.0])
    actual = apply_frozen_correction(parent, correction)
    assert np.allclose(actual, np.array([0.24, 0.999]))
    with pytest.raises(ValueError):
        apply_frozen_correction(parent, np.array([0.1]))


def test_selection_evidence_must_match_exact_frozen_recipe(tmp_path: Path) -> None:
    path = tmp_path / "selection.csv"
    pd.DataFrame(
        [
            {
                "variant": FROZEN_VARIANT,
                "domain": FROZEN_DOMAIN,
                "weight": FROZEN_WEIGHT,
                "gain": 12.0,
                "positive_month_fraction": 1.0,
                "worst_month_gain": 3.0,
                "minimum_domain_gain": 2.0,
            }
        ]
    ).to_csv(path, index=False)
    evidence = verify_frozen_selection(path)
    assert evidence["gain_late_2023"] == 12.0
