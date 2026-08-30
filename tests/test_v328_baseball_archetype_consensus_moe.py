from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v328_baseball_archetype_consensus_moe import (
    apply_router,
    build_archetypes,
    fit_router,
)


def _frame(n: int = 800) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_type": ["R"] * n,
            "pitcher_team_id": [1] * n,
            "batter_team_id": [2] * n,
            "asof_pitcher_n": [900] * n,
            "asof_pitcher_reverse_rate": [0.10] * n,
            "asof_pitcher_middle_rate": [0.25] * n,
            "asof_pitcher_ball_rate": [0.08] * n,
            "asof_pitcher_strike_rate": [0.06] * n,
            "asof_pitcher_pitchmix_n": [500] * n,
            "asof_pitcher_fastball_rate": [0.60] * n,
            "asof_pitcher_breaking_rate": [0.30] * n,
            "asof_pitcher_offspeed_rate": [0.10] * n,
            "balls_before": [3] * n,
            "strikes_before": [2] * n,
            "asof_pitcher_prev1_game_success_rate": [0.60] * n,
            "asof_pitcher_prev5_game_success_rate": [0.50] * n,
            "pitcher_hand": ["R"] * n,
            "batter_hand": ["L"] * n,
            "li": [2.0] * n,
            "game_month": np.tile(np.arange(4, 8), n // 4),
            "control_success": np.tile([1.0, 0.0], n // 2),
        }
    )


def test_build_archetypes_uses_fixed_baseball_buckets() -> None:
    output = build_archetypes(_frame())
    assert output.loc[0].to_dict() == {
        "route": "R_CORE",
        "support": "ESTABLISHED",
        "failure": "MIDDLE",
        "pitchmix": "FASTBALL",
        "count": "FULL",
        "form": "UP",
        "platoon_state": "OPPOSITE",
        "leverage": "HIGH",
    }


def test_router_selects_and_applies_only_positive_expert() -> None:
    frame = _frame()
    parent = np.full(len(frame), 0.60)
    target = frame["control_success"].to_numpy()
    helpful = target * 0.05 - (1.0 - target) * 0.05
    harmful = -helpful
    keys = np.full(len(frame), "R_CORE|ESTABLISHED")
    directions = {
        "season_innovation": helpful,
        "completed_anchor": harmful,
        "platoon": np.zeros(len(frame)),
        "trackman_physical": np.zeros(len(frame)),
        "lowrank_interaction": np.zeros(len(frame)),
        "beta_binomial": np.zeros(len(frame)),
        "player_transition": np.zeros(len(frame)),
        "joint_role_h1": np.zeros(len(frame)),
    }
    table, _ = fit_router(frame, parent, directions, keys)
    assert table == {"R_CORE|ESTABLISHED": "season_innovation"}
    candidate, active, assignment = apply_router(
        parent, directions, keys, table, np.ones(len(frame), dtype=bool)
    )
    assert active.all()
    assert np.all(assignment == "season_innovation")
    assert np.allclose(candidate, parent + helpful)


def test_apply_router_protects_nonregular_rows() -> None:
    parent = np.asarray([0.4, 0.4])
    directions = {name: np.asarray([0.1, 0.1]) for name in (
        "season_innovation", "completed_anchor", "platoon", "trackman_physical",
        "lowrank_interaction", "beta_binomial", "player_transition", "joint_role_h1",
    )}
    candidate, active, _ = apply_router(
        parent, directions, np.asarray(["A", "A"]), {"A": "platoon"},
        np.asarray([True, False]),
    )
    assert np.allclose(candidate, [0.5, 0.4])
    assert active.tolist() == [True, False]
