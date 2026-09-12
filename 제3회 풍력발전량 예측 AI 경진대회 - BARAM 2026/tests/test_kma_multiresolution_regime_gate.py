from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.kma_multiresolution_regime_gate import (
    build_regime_features,
    make_gate,
    route_candidate,
)


def _context(index: pd.DatetimeIndex, offset: float) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for level in ("10", "850", "700"):
        data[f"kma_um_ctx_u{level}_r0"] = np.arange(len(index)) + offset
        data[f"kma_um_ctx_v{level}_r0"] = np.arange(len(index)) - offset
    for level in ("850", "700"):
        data[f"kma_um_ctx_shear10_{level}_r0"] = (
            np.arange(len(index)) + offset
        )
    return pd.DataFrame(data, index=index)


def test_regime_features_are_aligned_and_finite() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h")
    features = build_regime_features(
        _context(index, 0.0),
        _context(index, 1.0),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([2.0, 1.0, 4.0, 3.0]),
        10.0,
    )
    assert features.index.equals(index)
    assert "vector_disagreement_10" in features
    assert "signed_candidate_movement" in features
    assert np.isfinite(features.to_numpy()).all()


def test_gate_threshold_uses_only_development_rows() -> None:
    values = np.asarray([1.0, 2.0, 100.0, 200.0])
    development = np.asarray([True, True, False, False])
    gate, threshold = make_gate(
        values,
        development,
        quantile=0.5,
        direction="high",
    )
    assert threshold == 1.5
    assert gate.tolist() == [False, True, True, True]


def test_router_can_only_retain_or_remove_current_movement() -> None:
    reference = np.asarray([10.0, 20.0, 30.0])
    current = np.asarray([12.0, 18.0, 35.0])
    routed = route_candidate(
        reference,
        current,
        np.asarray([True, False, True]),
    )
    assert routed.tolist() == [12.0, 20.0, 35.0]
