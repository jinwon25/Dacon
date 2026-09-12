import pandas as pd
import pytest

from src.archive.local_scorecard import build_scorecard


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {"year": [2022, 2023, 2024], "gain_vs_incumbent": [2.0, 4.0, 8.0]}
    )


def _bootstrap(p05_2022: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2022, 2023, 2024],
            "bootstrap_mean": [2.0, 4.0, 8.0],
            "p05": [p05_2022, 1.0, 2.0],
            "prob_positive": [0.96, 0.98, 0.99],
        }
    )


def test_scorecard_uses_half_life_weights_and_transfer_scenarios():
    transfer = pd.DataFrame(
        {"local_gain": [10.0, 20.0, 30.0], "public_gain": [5.0, 10.0, 15.0]}
    )
    card = build_scorecard(
        "candidate",
        _metrics(),
        _bootstrap(),
        incumbent_public=1000.0,
        transfer_history=transfer,
    )

    assert card["strict_gate"] is True
    assert card["post_break_gate"] is True
    assert card["local_estimators"]["half_life_1y_gain"] == pytest.approx(6.0)
    assert card["local_estimators"]["post_break_half_life_1y_gain"] == pytest.approx(
        20.0 / 3.0
    )
    assert card["public_scenarios"]["latest_year"] == pytest.approx(1004.0)
    assert card["public_scenarios"]["strict_bootstrap_lower"] == pytest.approx(
        1000.25
    )


def test_pre_break_failure_can_fail_strict_but_pass_post_break():
    card = build_scorecard(
        "candidate",
        _metrics(),
        _bootstrap(p05_2022=-1.0),
        incumbent_public=1000.0,
        calibration_local_gain=20.0,
        calibration_public_gain=10.0,
    )

    assert card["strict_gate"] is False
    assert card["post_break_gate"] is True


def test_single_transfer_observation_suppresses_public_projection():
    card = build_scorecard(
        "candidate",
        _metrics(),
        _bootstrap(),
        incumbent_public=1000.0,
        calibration_local_gain=20.0,
        calibration_public_gain=10.0,
    )
    assert card["public_scenarios"]["status"] == "insufficient_clean_transfer_history"
    assert "latest_year" not in card["public_scenarios"]


def test_scorecard_rejects_mismatched_years():
    bootstrap = _bootstrap().iloc[1:].reset_index(drop=True)
    with pytest.raises(ValueError, match="years differ"):
        build_scorecard(
            "candidate",
            _metrics(),
            bootstrap,
            incumbent_public=1000.0,
            calibration_local_gain=20.0,
            calibration_public_gain=10.0,
        )
