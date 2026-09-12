from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.compare_nested_base_runs import compare_runs
from src.metrics import CAPACITY_KWH


def _cache(offset: float) -> dict[str, np.ndarray]:
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    cache: dict[str, np.ndarray] = {
        "index_ns": index.astype("int64").to_numpy(),
        "test_index_ns": index[:24].astype("int64").to_numpy(),
    }
    for group, capacity in CAPACITY_KWH.items():
        truth = np.full(len(index), 0.5 * capacity, dtype=np.float32)
        cache[f"{group}__truth"] = truth
        cache[f"{group}__candidate"] = truth + offset
        cache[f"{group}__test"] = np.full(24, 0.5 * capacity + offset)
    return cache


def test_compare_runs_detects_candidate_improvement() -> None:
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    issues = pd.DatetimeIndex(index.floor("D"))
    result = compare_runs(
        _cache(400.0),
        _cache(100.0),
        issues,
        n_bootstrap=20,
        seed=1,
    )
    assert result["outer_macro"]["delta"]["score"] > 0.0
    assert result["outer_macro"]["delta"]["one_minus_nmae"] > 0.0
    assert result["qualified"]


def test_compare_runs_rejects_timestamp_mismatch() -> None:
    reference = _cache(400.0)
    candidate = _cache(100.0)
    candidate["index_ns"] = candidate["index_ns"] + 1
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    issues = pd.DatetimeIndex(index.floor("D"))
    try:
        compare_runs(
            reference,
            candidate,
            issues,
            n_bootstrap=5,
            seed=1,
        )
    except ValueError as error:
        assert "OOF timestamps differ" in str(error)
    else:
        raise AssertionError("timestamp mismatch was not rejected")


def test_compare_runs_retains_small_season_loss_as_exploratory(
    monkeypatch,
) -> None:
    reference = _cache(400.0)
    candidate = _cache(100.0)
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    issues = pd.DatetimeIndex(index.floor("D"))

    def fake_seasons(*_args):
        return np.array(["stable"] * 24 + ["weak"] * 24), {}

    call_count = 0

    def fake_delta(*_args):
        nonlocal call_count
        call_count += 1
        delta = {
            "score": 0.001 if call_count != 3 else -0.0002,
            "one_minus_nmae": 0.0001,
            "ficr": 0.0019,
        }
        return {"delta": delta}

    monkeypatch.setattr(
        "experiments.compare_nested_base_runs._ordered_issue_seasons",
        fake_seasons,
    )
    monkeypatch.setattr(
        "experiments.compare_nested_base_runs._competition_delta",
        fake_delta,
    )
    monkeypatch.setattr(
        "experiments.compare_nested_base_runs._issue_bootstrap",
        lambda *_args, **_kwargs: {
            "q05": 0.0001,
            "positive_fraction": 0.95,
        },
    )

    result = compare_runs(
        reference,
        candidate,
        issues,
        n_bootstrap=20,
        seed=1,
    )

    assert not result["qualified"]
    assert result["decision_tier"] == "exploratory"
