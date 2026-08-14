from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.build_noaa_isd_issue_features import (
    build_issue_features,
    parse_wind,
)


def test_parse_wind_decodes_meteorological_direction_and_missing_codes() -> None:
    values = pd.Series(["270,1,N,0040,1", "999,9,9,9999,9"])

    parsed = parse_wind(values)

    assert parsed.loc[0, "wind_speed_ms"] == 4.0
    assert parsed.loc[0, "wind_u_ms"] == 4.0
    assert abs(parsed.loc[0, "wind_v_ms"]) < 1e-10
    assert parsed.loc[1].isna().all()


def test_issue_features_never_use_observations_after_safe_cutoff() -> None:
    index = pd.date_range("2024-01-01 08:00", periods=7, freq="h")
    station = pd.DataFrame(
        {
            "wind_direction_deg": np.full(len(index), 270.0),
            "wind_speed_ms": np.arange(1.0, len(index) + 1.0),
            "wind_u_ms": np.arange(1.0, len(index) + 1.0),
            "wind_v_ms": np.zeros(len(index)),
        },
        index=index,
    )
    issues = pd.DatetimeIndex(["2024-01-01 13:00", "2024-01-01 14:00"])

    features, audit = build_issue_features({"00000199999": station}, issues)

    assert features.loc[0, "safe_observation_cutoff_kst"] == pd.Timestamp(
        "2024-01-01 11:00"
    )
    assert features.loc[0, "latest_observation_kst"] == pd.Timestamp(
        "2024-01-01 11:00"
    )
    assert (
        features.loc[0, "asos_noaa_stn00000199999__h01__ws_latest"]
        == 4.0
    )
    assert audit["violations"] == 0
    assert audit["minimum_availability_margin_minutes"] == 0.0
