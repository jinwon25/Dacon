import numpy as np
import pandas as pd

from src.v38_latest_trackman_pfd import (
    _bootstrap_audit,
    _prepare_pair,
    student_columns,
)


def test_without_ids_drops_only_identifier_columns():
    safe = ["game_month", "pitcher_id", "batter_id", "pitcher_hand"]
    assert student_columns(safe, "with_ids") == safe
    assert student_columns(safe, "without_ids") == ["game_month", "pitcher_hand"]


def test_audit_physical_columns_are_missing_and_not_needed_by_student():
    source = pd.DataFrame(
        {
            "game_month": [3],
            "pitcher_id": [1],
            **{f"tm_{column}": [1.0] for column in (
                "pitch_type_group",
                "rel_speed",
                "spin_rate",
                "induced_vert_break",
                "horz_break",
                "extension",
                "rel_height",
                "rel_side",
                "zone_speed",
            )},
        }
    )
    source["tm_pitch_type_group"] = "fastball"
    audit = pd.DataFrame({"game_month": [8], "pitcher_id": [2]})
    source_x, audit_x, full = _prepare_pair(
        source, audit, ["game_month", "pitcher_id"]
    )
    assert "tm_rel_speed" in full
    assert np.isnan(audit_x.loc[0, "tm_rel_speed"])
    assert source_x.loc[0, "tm_pitch_type_group"] == "fastball"


def test_bootstrap_audit_covers_independent_cluster_views():
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "batter_id": [1, 2, 1, 2, 3, 4, 3, 4],
        }
    )
    target = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=float)
    parent = np.full(8, 0.55)
    candidate = np.full(8, 0.50)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = _bootstrap_audit(
            frame, target, candidate, parent, n_resamples=20, block_size=2
        )
    assert set(result) == {
        "pitcher",
        "batter",
        "pitcher_x_batter",
        "block_2",
    }
    assert all(
        0 < item["n_valid_resamples"] <= 20 for item in result.values()
    )
