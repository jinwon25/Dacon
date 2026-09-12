from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v328_baseball_archetype_consensus_moe import EXPERT_ORDER
from src.archive.v329_two_origin_stable_expert_cells import stable_router


def test_stable_router_requires_same_expert_to_help_both_origins() -> None:
    frames = {}
    parents = {}
    directions = {}
    keys = {}
    for source in ("full_2022", "late_2023"):
        n = 800
        target = np.tile([1.0, 0.0], n // 2)
        frames[source] = pd.DataFrame(
            {
                "control_success": target,
                "game_month": np.tile([8, 9], n // 2),
                "game_type": ["R"] * n,
            }
        )
        parents[source] = np.full(n, 0.5)
        helpful = np.where(target == 1.0, 0.05, -0.05)
        directions[source] = {
            expert: (helpful if expert == "beta_binomial" else np.zeros(n))
            for expert in EXPERT_ORDER
        }
        keys[source] = np.full(n, "R_CORE|DEVELOPING")
    router, evidence = stable_router(frames, parents, directions, keys)
    assert router == {"R_CORE|DEVELOPING": "beta_binomial"}
    assert evidence.query("passes")["expert"].tolist() == ["beta_binomial"]


def test_stable_router_rejects_one_origin_reversal() -> None:
    frames = {}
    parents = {}
    directions = {}
    keys = {}
    for source, sign in (("full_2022", 1.0), ("late_2023", -1.0)):
        n = 800
        target = np.tile([1.0, 0.0], n // 2)
        frames[source] = pd.DataFrame(
            {
                "control_success": target,
                "game_month": np.tile([8, 9], n // 2),
                "game_type": ["R"] * n,
            }
        )
        parents[source] = np.full(n, 0.5)
        direction = sign * np.where(target == 1.0, 0.05, -0.05)
        directions[source] = {
            expert: (direction if expert == "platoon" else np.zeros(n))
            for expert in EXPERT_ORDER
        }
        keys[source] = np.full(n, "R_CORE|SAME")
    router, _ = stable_router(frames, parents, directions, keys)
    assert router == {}
