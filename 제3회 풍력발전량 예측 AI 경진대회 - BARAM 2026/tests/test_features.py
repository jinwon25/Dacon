import pandas as pd

from src.features import _calendar_features, _select_latest_legal_cycle


def test_latest_legal_cycle_is_selected_before_feature_pivot() -> None:
    frame = pd.DataFrame({
        "forecast_kst_dtm": ["2025-01-01 01:00:00"] * 3,
        "data_available_kst_dtm": ["2024-12-31 12:00:00", "2024-12-31 13:00:00", "2024-12-31 15:00:00"],
        "grid_id": [1, 1, 1],
        "value": [1.0, 2.0, 3.0],
    })

    selected = _select_latest_legal_cycle(frame)

    assert len(selected) == 1
    assert selected.iloc[0]["value"] == 2.0


def test_calendar_lead_is_derived_from_availability() -> None:
    index = pd.to_datetime(["2025-01-01 01:00:00", "2025-01-02 00:00:00"])
    availability = pd.Series(pd.to_datetime(["2024-12-31 13:00:00"] * 2), index=index)

    features = _calendar_features(index, availability)

    assert features["lead_hour"].tolist() == [12.0, 35.0]
