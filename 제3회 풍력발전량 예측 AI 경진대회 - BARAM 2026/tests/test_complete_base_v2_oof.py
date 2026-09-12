from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.complete_base_v2_oof import stitch_prediction


def test_stitch_prediction_completes_non_overlapping_blocks() -> None:
    full = pd.date_range("2024-01-01", periods=6, freq="h")
    retained = full[:4]
    extension = full[4:]
    values, report = stitch_prediction(
        full,
        retained,
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        extension,
        np.asarray([5.0, 6.0]),
    )
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert report["missing_rows"] == 0
    assert report["overlap_rows"] == 0
    assert report["extension_rows"] == 2


def test_stitch_prediction_rejects_overlap() -> None:
    full = pd.date_range("2024-01-01", periods=4, freq="h")
    with pytest.raises(ValueError, match="overlap"):
        stitch_prediction(
            full,
            full[:3],
            np.asarray([1.0, 2.0, 3.0]),
            full[2:],
            np.asarray([4.0, 5.0]),
        )


def test_stitch_prediction_rejects_gap() -> None:
    full = pd.date_range("2024-01-01", periods=4, freq="h")
    with pytest.raises(ValueError, match="missing timestamps"):
        stitch_prediction(
            full,
            full[:2],
            np.asarray([1.0, 2.0]),
            full[3:],
            np.asarray([4.0]),
        )
