from __future__ import annotations

import json

import pandas as pd

from experiments.compose_group_candidate import (
    _stable_full_delta,
    _validated_full_delta,
    compose_frames,
)


def _frame(g1: float, g2: float, g3: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_id": ["a", "b"],
            "forecast_kst_dtm": ["2025-01-01 00:00", "2025-01-01 01:00"],
            "kpx_group_1": [g1, g1],
            "kpx_group_2": [g2, g2],
            "kpx_group_3": [g3, g3],
        }
    )


def test_compose_frames_replaces_only_selected_group_columns() -> None:
    base = _frame(1.0, 2.0, 3.0)
    output = compose_frames(
        base,
        _frame(10.0, 20.0, 30.0),
        _frame(100.0, 200.0, 300.0),
    )
    assert output["kpx_group_1"].tolist() == [10.0, 10.0]
    assert output["kpx_group_2"].tolist() == [2.0, 2.0]
    assert output["kpx_group_3"].tolist() == [300.0, 300.0]


def test_compose_frames_rejects_different_ids() -> None:
    base = _frame(1.0, 2.0, 3.0)
    group1 = _frame(10.0, 20.0, 30.0)
    group1.loc[1, "forecast_id"] = "different"
    try:
        compose_frames(base, group1, _frame(100.0, 200.0, 300.0))
    except ValueError as exc:
        assert "IDs differ" in str(exc)
    else:
        raise AssertionError("mismatched IDs must be rejected")


def test_stable_full_delta_accepts_target_level_promotion(tmp_path) -> None:
    report = {
        "validation": {
            "kpx_group_1": {
                "promotion": "promoted",
                "period_deltas": {"full": {"score": 0.0123}},
            }
        }
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert _stable_full_delta(path, "kpx_group_1") == 0.0123


def test_validated_full_delta_requires_opt_in_for_near_stable(tmp_path) -> None:
    report = {
        "validation": {
            "kpx_group_2": {
                "promotion": "rejected",
                "promotion_tier": "near_stable",
                "period_deltas": {"full": {"score": 0.0042}},
            }
        }
    }
    path = tmp_path / "near_stable.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    try:
        _validated_full_delta(path, "kpx_group_2")
    except ValueError as exc:
        assert "not stable" in str(exc)
    else:
        raise AssertionError("near-stable target requires explicit opt-in")
    assert (
        _validated_full_delta(
            path,
            "kpx_group_2",
            allow_near_stable=True,
        )
        == 0.0042
    )


def test_controlled_exploratory_delta_requires_explicit_opt_in(
    tmp_path,
) -> None:
    report = {
        "promotion_tier": "controlled_exploratory",
        "validation": {
            "kpx_group_1": {
                "period_deltas": {"full": {"score": 0.0123}},
                "gates": {"q1": True, "q2": True},
            }
        },
    }
    path = tmp_path / "controlled.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    try:
        _validated_full_delta(path, "kpx_group_1")
    except KeyError:
        pass
    else:
        raise AssertionError("controlled exploratory target requires opt-in")
    assert (
        _validated_full_delta(
            path,
            "kpx_group_1",
            allow_controlled_exploratory=True,
        )
        == 0.0123
    )
