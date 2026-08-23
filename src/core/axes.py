"""Champion OOF axis loading and derived residual features.

Moved verbatim from ``src/v23_structural_residual_screen.py`` during the core extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from src.core.v25_recipe import ETA as V25_SOURCE_ETA


CHAMPION_DIR = Path("artifacts/champion_oof_20260817_01")


def apply_v22_recipe(
    v21: np.ndarray,
    domain: np.ndarray,
    pitcher_rate: np.ndarray,
    batter_rate: np.ndarray,
) -> np.ndarray:
    """Apply the exact balanced v22 correction used in the submitted ZIP."""
    parent = np.asarray(v21, dtype=np.float64)
    domain = np.asarray(domain).astype(str)
    pitcher = np.nan_to_num(np.asarray(pitcher_rate, dtype=np.float64), nan=0.5)
    batter = np.nan_to_num(np.asarray(batter_rate, dtype=np.float64), nan=0.5)
    correction = np.zeros(len(parent), dtype=np.float64)
    parameters = {
        "R_CORE": (0.44, 0.035),
        "R_ANCHOR": (0.48, 0.020),
        "F": (0.52, 0.020),
    }
    for name, (anchor, weight) in parameters.items():
        selected = domain == name
        correction[selected] += weight * (anchor - parent[selected])
    prior = 0.75 * pitcher + 0.25 * batter
    correction += 0.050 * (prior - parent)
    return np.clip(parent + correction, 0.001, 0.999)


def _derived(frame: pd.DataFrame, domain: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["domain3"] = np.asarray(domain).astype(str)
    output["count_state"] = (
        output["balls_before"].astype("Int64").astype(str)
        + "-"
        + output["strikes_before"].astype("Int64").astype(str)
    )
    output["hand_matchup"] = (
        output["pitcher_hand"].astype("string").fillna("__MISSING__")
        + "-"
        + output["batter_hand"].astype("string").fillna("__MISSING__")
    )
    output["inning_bucket"] = pd.cut(
        pd.to_numeric(output["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, np.inf),
        labels=("early", "middle", "late"),
    ).astype("string")
    for column in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        output[f"log1p_{column}"] = np.log1p(
            pd.to_numeric(output[column], errors="coerce").clip(lower=0.0)
        )
    output["recent_success_delta_1_5"] = (
        pd.to_numeric(output["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(output["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    )
    output["recent_middle_delta_1_5"] = (
        pd.to_numeric(output["asof_pitcher_prev1_game_middle_rate"], errors="coerce")
        - pd.to_numeric(output["asof_pitcher_prev5_game_middle_rate"], errors="coerce")
    )
    output["pitcher_batter_rate_gap"] = (
        pd.to_numeric(output["asof_pitcher_success_rate"], errors="coerce")
        - pd.to_numeric(output["asof_batter_success_rate"], errors="coerce")
    )
    return output


def _load_axis(project: Path, axis: str, raw: pd.DataFrame) -> pd.DataFrame:
    with np.load(project / CHAMPION_DIR / f"{axis}.npz", allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        v21 = saved["v21"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    if axis == "y2023_early_to_late":
        frame = raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True)
    elif axis == "y2023_to_y2024":
        frame = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    else:
        raise ValueError(axis)
    if not np.array_equal(target, frame["control_success"].to_numpy(np.float64)):
        raise ValueError(f"target order mismatch for {axis}")
    frame = _derived(frame, domain)
    frame["target"] = target
    frame["v21"] = v21
    frame["v22"] = apply_v22_recipe(
        v21,
        domain,
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce").to_numpy(),
        pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce").to_numpy(),
    )
    return frame


load_axis = _load_axis


def _joint_domain(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype("string").fillna("__MISSING__").eq("R")
    anchor = frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    return np.where(
        ~regular.to_numpy(), "F", np.where(anchor.to_numpy(), "R_ANCHOR", "R_CORE")
    )


def _early_to_late_2024(project: Path, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    year = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    derived = _derived(year, _joint_domain(year))
    fit = derived.loc[derived["game_month"].le(7)].reset_index(drop=True)
    audit = derived.loc[derived["game_month"].ge(8)].reset_index(drop=True)
    with np.load(
        project / "artifacts" / "champion_oof_20260817_01" / "y2024_early_to_late.npz",
        allow_pickle=True,
    ) as saved:
        target = saved["target"].astype(np.float64)
        v21 = saved["v21"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    if not np.array_equal(target, audit["control_success"].to_numpy(np.float64)):
        raise ValueError("2024 early-to-late target order mismatch")
    audit["target"] = target
    audit["v21"] = v21
    audit["domain3"] = domain
    audit["v22"] = apply_v22_recipe(
        v21,
        domain,
        pd.to_numeric(audit["asof_pitcher_success_rate"], errors="coerce").to_numpy(),
        pd.to_numeric(audit["asof_batter_success_rate"], errors="coerce").to_numpy(),
    )
    return fit, audit


def _cached_v25_axes(project: Path, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the v25 audit frames from frozen direct-prediction caches."""
    selection = _load_axis(project, "y2023_early_to_late", raw)
    outer = _load_axis(project, "y2023_to_y2024", raw)
    _, replication = _early_to_late_2024(project, raw)
    frames = {
        "selection_late_2023": selection,
        "outer_full_2024": outer,
        "replication_late_2024": replication,
    }
    cache_root = project / "artifacts" / "v29_anchor_route_20260817_01"
    output: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        direct = np.load(cache_root / f"{name}_direct.npy").astype(np.float64)
        if len(direct) != len(frame):
            raise ValueError(f"v25 direct cache row mismatch: {name}")
        local = frame.copy()
        v22 = local["v22"].to_numpy(np.float64)
        anchor = local["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
        v25 = v22.copy()
        v25[anchor] = np.clip(
            v22[anchor] + V25_SOURCE_ETA * (direct[anchor] - v22[anchor]),
            0.001,
            0.999,
        )
        local["v25"] = v25
        output[name] = local
    return output


def _quadratic_gain(
    target: np.ndarray,
    parent: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Exact Brier-skill gain for an unclipped linear probability direction."""
    local_target = target[mask]
    local_direction = direction[mask]
    local_error = local_target - parent[mask]
    reference = float(np.mean(local_target) * (1.0 - np.mean(local_target)))
    scale = 1_000_000.0 if reference <= 0.0 else 100_000.0 / reference
    linear = 2.0 * float(np.mean(local_error * local_direction))
    quadratic = float(np.mean(np.square(local_direction)))
    return scale * (weights * linear - np.square(weights) * quadratic)
