from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
    apply_fallback,
    restrictions,
)


def test_apply_fallback_uses_frozen_threshold_and_weight() -> None:
    jy = np.array([0.49, 0.50, 0.70])
    xgb = np.array([0.9, 0.6, 0.4])
    final, active = apply_fallback(jy, xgb, np.array([True, True, False]))
    assert active.tolist() == [False, True, False]
    np.testing.assert_allclose(final, [0.49, 0.53, 0.70])


def test_align_regular_prediction_rejects_wrong_length(tmp_path) -> None:
    frame = pd.DataFrame({"game_type": ["R", "F", "R"]})
    path = tmp_path / "xgb.npy"
    np.save(path, np.array([0.5]))
    with pytest.raises(ValueError, match="regular-row mismatch"):
        align_regular_prediction(frame, path)


def test_v218_freezes_public1175_axis() -> None:
    audit = restrictions()
    assert audit["public1175_formula_frozen"]
    assert audit["fallback_xgb_weight_frozen"]
    assert audit["fallback_xgb_threshold_frozen"]
    assert audit["joint_h1_recipe_frozen_from_v214_v215"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
