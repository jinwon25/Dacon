"""Frozen OOF signal banks and stage-one gain grids.

Moved verbatim from ``src/v35_three_stage_multibank.py`` during the core extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN = ("oracle", "mode_label", "raw_by_mode")


EXACT_NAMES = (
    "exact_lgb",
    "exact_ridge",
    "trend_lgb",
    "binary_seed_ensemble",
    "l2_leaves7",
    "l2_leaves15",
)


RECENT_NAMES = ("prediction", "exact_lgb", "exact_ridge", "trend_lgb")


def signal_family(name: str) -> str:
    return str(name).split("::", 1)[0]


def _mode_dir(year: int) -> str:
    return (
        "latent_failure_mode_state_20260817_02"
        if year == 2022
        else "latent_failure_mode_state_20260816_01"
    )


def _load_bank(
    project: Path, year: int, names: set[str] | None = None
) -> dict[str, np.ndarray]:
    """Load only legal prediction columns shared by all three audit years."""

    def wanted(name: str) -> bool:
        return names is None or name in names

    output: dict[str, np.ndarray] = {}
    mode_path = (
        project
        / "artifacts"
        / _mode_dir(year)
        / f"latent_failure_mode_o{year}.npz"
    )
    with np.load(mode_path, allow_pickle=True) as saved:
        mode_names = [str(value) for value in saved["names"].tolist()]
        for index, raw_name in enumerate(mode_names):
            name = f"mode::{raw_name}"
            if wanted(name) and not any(token in raw_name.lower() for token in FORBIDDEN):
                output[name] = saved["raw"][:, index].astype(np.float64)

    variants_path = (
        project
        / "artifacts"
        / "multi_year_state_variants_20260816_01"
        / f"state_variants_o{year}.npz"
    )
    with np.load(variants_path, allow_pickle=True) as saved:
        for raw_name in saved.files:
            name = f"state::{raw_name}"
            if raw_name.startswith(("global_", "domain_")) and wanted(name):
                output[name] = saved[raw_name].astype(np.float64)
    selected_path = (
        project
        / "artifacts"
        / "multi_year_state_selected_20260816_01"
        / f"selected_state_o{year}.npz"
    )
    if wanted("state::selected"):
        with np.load(selected_path, allow_pickle=True) as saved:
            output["state::selected"] = saved["raw"].astype(np.float64)

    exact_path = (
        project
        / "artifacts"
        / "v14_exact_model_screen_20260815_01"
        / f"exact_model_screen_o{year}.npz"
    )
    with np.load(exact_path, allow_pickle=True) as saved:
        for raw_name in EXACT_NAMES:
            name = f"exact::{raw_name}"
            if wanted(name):
                output[name] = saved[raw_name].astype(np.float64)

    recent_path = (
        project
        / "artifacts"
        / "recent_shared_exact_asof_20260815_02"
        / f"recent_shared_o{year}.npz"
    )
    with np.load(recent_path, allow_pickle=True) as saved:
        for raw_name in RECENT_NAMES:
            name = f"recent::{raw_name}"
            if wanted(name):
                output[name] = saved[raw_name].astype(np.float64)
    if names is not None and set(output) != set(names):
        missing = sorted(set(names) - set(output))
        raise ValueError(f"bank signals missing for {year}: {missing}")
    return output


def _metadata(project: Path, year: int) -> dict[str, np.ndarray]:
    path = (
        project
        / "artifacts"
        / "multi_year_state_selected_20260816_01"
        / f"selected_state_o{year}.npz"
    )
    with np.load(path, allow_pickle=True) as saved:
        return {
            "target": saved["target"].astype(np.float64),
            "parent": saved["incumbent"].astype(np.float64),
            "month": saved["game_month"].astype(np.int16),
            "domain": saved["domain3"].astype(str),
        }


def _mask(domain: np.ndarray, route: str) -> np.ndarray:
    if route == "ALL":
        return np.ones(len(domain), dtype=bool)
    return np.asarray(domain, dtype=str) == route


def _gain_grid(
    target: np.ndarray,
    parent: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    local_target = np.asarray(target, dtype=np.float64)[mask]
    local_parent = np.asarray(parent, dtype=np.float64)[mask]
    local_direction = np.asarray(direction, dtype=np.float64)[mask]
    residual = local_target - local_parent
    reference = float(local_target.mean() * (1.0 - local_target.mean()))
    scale = 1_000_000.0 if reference <= 0.0 else 100_000.0 / reference
    linear = 2.0 * float(np.mean(residual * local_direction))
    quadratic = float(np.mean(np.square(local_direction)))
    return scale * (weights * linear - np.square(weights) * quadratic)


def grid_rows(
    frame: pd.DataFrame,
    parent: np.ndarray,
    v21: np.ndarray,
    raw: np.ndarray,
    *,
    signal: str,
    direction_mode: str,
    route: str,
) -> list[dict[str, object]]:
    if direction_mode == "toward_parent":
        direction = np.asarray(raw, dtype=np.float64) - parent
    elif direction_mode == "delta_v21":
        direction = np.asarray(raw, dtype=np.float64) - v21
    else:
        raise ValueError(f"unknown direction: {direction_mode}")
    route_mask = _mask(frame["domain3"].astype(str).to_numpy(), route)
    direction = np.where(route_mask, direction, 0.0)
    weights = np.asarray(WEIGHTS, dtype=np.float64)
    endpoints = parent[:, None] + direction[:, None] * weights[None, :]
    if float(endpoints.min()) < 0.001 or float(endpoints.max()) > 0.999:
        raise ValueError(f"candidate clips: {signal} {direction_mode} {route}")
    target = frame["target"].to_numpy(np.float64)
    all_mask = np.ones(len(frame), dtype=bool)
    gains = _gain_grid(target, parent, direction, all_mask, weights)
    month_values = []
    for month in sorted(frame["game_month"].unique()):
        month_mask = frame["game_month"].eq(month).to_numpy()
        if np.any(month_mask & route_mask):
            month_values.append(
                _gain_grid(target, parent, direction, month_mask, weights)
            )
    month_matrix = np.vstack(month_values)
    domain_values = {}
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        domain_mask = frame["domain3"].astype(str).eq(domain).to_numpy()
        domain_values[domain] = _gain_grid(
            target, parent, direction, domain_mask, weights
        )
    domain_matrix = np.vstack(list(domain_values.values()))
    applied = (
        domain_matrix.min(axis=0)
        if route == "ALL"
        else domain_values[route]
    )
    output = []
    for index, weight in enumerate(weights):
        selection_score = min(
            float(gains[index]),
            float(month_matrix[:, index].min()),
            float(applied[index]),
        )
        output.append(
            {
                "signal": signal,
                "family": signal_family(signal),
                "direction": direction_mode,
                "domain": route,
                "weight": float(weight),
                "gain": float(gains[index]),
                "positive_month_fraction": float(
                    np.mean(month_matrix[:, index] > 0.0)
                ),
                "worst_month_gain": float(month_matrix[:, index].min()),
                "minimum_domain_gain": float(domain_matrix[:, index].min()),
                "applied_domain_gain": float(applied[index]),
                "selection_score": selection_score,
                "mean_abs_shift": float(
                    weight * np.mean(np.abs(direction))
                ),
            }
        )
    return output


load_bank = _load_bank
metadata = _metadata
