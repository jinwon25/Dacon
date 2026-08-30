import pandas as pd

from src.audit_v343_transition_workload import pressure_balanced_frame


def test_pressure_balanced_frame_requires_core_rows(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["F"],
            "pitcher_team_id": [1],
            "batter_team_id": [2],
            "num_runners_on": [0],
            "li": [1.0],
            "control_success": [1],
        }
    )
    path = tmp_path / "fixture.csv"
    frame.to_csv(path, index=False)
    # balanced_frame creates a synthetic R_CORE branch from this valid row.
    # One source row is still insufficient for pressure/non-pressure coverage.
    try:
        pressure_balanced_frame(path, 1)
    except ValueError as error:
        assert "at least two R_CORE rows" in str(error)
    else:
        raise AssertionError("expected insufficient core fixture to fail")
