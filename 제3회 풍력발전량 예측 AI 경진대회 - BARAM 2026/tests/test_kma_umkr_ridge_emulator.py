from __future__ import annotations

import numpy as np

from experiments.kma_umkr_ridge_emulator import (
    clip_emulated_features,
    feature_metrics,
)


def test_emulator_clips_physical_feature_domains() -> None:
    values = np.asarray([[-2.0, -3.0, 1.5, 4.0]])
    columns = [
        "kma_um_ctx_speed10_r0",
        "kma_um_ctx_shear10_850_r0",
        "kma_um_ctx_cos10_850_r0",
        "kma_um_ctx_u10_r0",
    ]
    clipped = clip_emulated_features(values, columns)
    assert clipped.tolist() == [[0.0, 0.0, 1.0, 4.0]]


def test_feature_metrics_identify_exact_reconstruction() -> None:
    truth = np.asarray([[1.0], [2.0], [3.0]])
    metrics = feature_metrics(truth, truth.copy(), ["wind"])
    assert metrics[0]["correlation"] == 1.0
    assert metrics[0]["rmse"] == 0.0
    assert metrics[0]["normalized_rmse"] == 0.0
