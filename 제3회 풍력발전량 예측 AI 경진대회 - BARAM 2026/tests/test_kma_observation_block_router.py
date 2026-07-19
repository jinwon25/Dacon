from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.kma_observation_block_router import (
    attach_observation_features,
    qualifies,
    route_predictions,
)


def test_attach_observations_uses_one_issue_for_each_block() -> None:
    frame = pd.DataFrame(
        {
            "representative_ns": [1, 2],
            "phase": [0, 1],
            "rows": [2, 2],
            "weather": [0.1, 0.2],
        },
        index=["b1", "b2"],
    )
    rows_by_block = {"b1": np.array([0, 1]), "b2": np.array([2, 3])}
    issue = pd.DatetimeIndex(
        ["2024-01-01 13:00", "2024-01-01 13:00", "2024-01-02 13:00", "2024-01-02 13:00"]
    )
    observations = pd.DataFrame(
        {"asos_stn216__h03__ws_mean": [4.0, 7.0]},
        index=pd.DatetimeIndex(["2024-01-01 13:00", "2024-01-02 13:00"]),
    )
    result = attach_observation_features(frame, rows_by_block, issue, observations)
    assert result["obs__asos_stn216__h03__ws_mean"].tolist() == [4.0, 7.0]
    assert result["issue_ns"].nunique() == 2


def test_route_predictions_moves_only_highest_positive_utility_block() -> None:
    experts = {
        "incumbent_finesweep": np.zeros(4),
        "expert": np.full(4, 10.0),
    }
    rows_by_block = {"b1": np.array([0, 1]), "b2": np.array([2, 3])}
    utility = pd.DataFrame({"expert": [0.2, 0.1]}, index=["b1", "b2"])
    prediction, report = route_predictions(
        experts,
        rows_by_block,
        np.array(["b1", "b2"]),
        utility,
        alpha=0.1,
        coverage=0.5,
    )
    assert prediction.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert report["selected_blocks"] == 1
    assert report["selected_experts"] == {"expert": 1}


def _evaluation(score: float = 0.001) -> dict:
    return {
        "delta": {
            "score": score,
            "one_minus_nmae": 0.001,
            "ficr": 0.001,
        },
        "worst_month_score_delta": 0.0,
    }


def test_router_qualification_requires_seed_and_incremental_improvement() -> None:
    incremental = {"score": 0.001, "one_minus_nmae": 0.0, "ficr": 0.001}
    assert qualifies(_evaluation(), {17: _evaluation(), 29: _evaluation()}, incremental)
    assert not qualifies(
        _evaluation(),
        {17: _evaluation(), 29: _evaluation(-0.001)},
        incremental,
    )
    assert not qualifies(
        _evaluation(),
        {17: _evaluation(), 29: _evaluation()},
        {**incremental, "score": -0.001},
    )
