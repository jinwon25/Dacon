from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from experiments.issue_trajectory_tcn import (
    IssueTrajectoryTCN,
    make_sequence_bundle,
    masked_pinball_loss,
    select_trajectory_columns,
)


def test_make_sequence_bundle_keeps_only_complete_issues() -> None:
    first = pd.date_range("2024-01-01 01:00", periods=24, freq="h")
    second = pd.date_range("2024-01-02 01:00", periods=23, freq="h")
    index = first.append(second)
    features = pd.DataFrame(
        {"gfs__ws100__mean": np.arange(len(index), dtype=float)},
        index=index,
    )
    issue = pd.Series(
        [pd.Timestamp("2023-12-31 13:00")] * 24
        + [pd.Timestamp("2024-01-01 13:00")] * 23,
        index=index,
    )
    labels = pd.DataFrame(
        {
            "kpx_group_1": 1_000.0,
            "kpx_group_2": 2_000.0,
            "kpx_group_3": np.nan,
        },
        index=index,
    )
    bundle = make_sequence_bundle(features, issue, labels)
    assert bundle.features.shape == (1, 24, 1)
    assert bundle.targets is not None
    assert bundle.target_mask is not None
    assert bundle.targets.shape == (1, 24, 3)
    assert not bundle.target_mask[:, :, 2].any()


def test_tcn_output_and_masked_loss_are_finite() -> None:
    model = IssueTrajectoryTCN(7, hidden=8, dropout=0.0)
    features = torch.randn(2, 24, 7)
    prediction = model(features)
    assert prediction.shape == (2, 24, 3, 3)
    assert torch.all((prediction >= 0.0) & (prediction <= 1.0))
    truth = torch.rand(2, 24, 3)
    observed = torch.ones_like(truth, dtype=torch.bool)
    observed[:, :, 2] = False
    loss = masked_pinball_loss(prediction, truth, observed)
    assert torch.isfinite(loss)
    loss.backward()


def test_trajectory_column_selection_is_compact_and_physical() -> None:
    frame = pd.DataFrame(
        columns=[
            "lead_hour",
            "doy_sin",
            "gfs__ws100__mean",
            "gfs__kpx_group_1__hub_ws117__idw",
            "ldaps__surface_0_sp__mean",
            "irrelevant",
        ]
    )
    selected = select_trajectory_columns(frame)
    assert selected == [
        "lead_hour",
        "doy_sin",
        "gfs__ws100__mean",
        "gfs__kpx_group_1__hub_ws117__idw",
    ]
