"""Frozen OOF contract loading and route diagnostics.

Moved verbatim from ``src/archive/v97_conditional_direct_forward.py`` during the core extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _bss(target: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(np.mean(target))
    return 100_000.0 * (
        1.0 - float(np.mean(np.square(prediction - target))) / (rate * (1.0 - rate))
    )


def _gain(target: np.ndarray, parent: np.ndarray, candidate: np.ndarray) -> float:
    return _bss(target, candidate) - _bss(target, parent)


def _apply(
    parent: np.ndarray,
    direct: np.ndarray,
    domain: np.ndarray,
    route: tuple[str, ...],
    eta: float,
) -> np.ndarray:
    active = np.isin(domain.astype(str), route)
    output = parent.copy()
    output[active] = np.clip(
        parent[active] + eta * (direct[active] - parent[active]), 0.001, 0.999
    )
    return output


def _diagnostics(axis: dict[str, np.ndarray], route: tuple[str, ...], eta: float) -> dict[str, Any]:
    candidate = _apply(axis["parent"], axis["direct"], axis["domain3"], route, eta)
    target = axis["target"]
    parent = axis["parent"]
    active = np.isin(axis["domain3"].astype(str), route)
    months = []
    for month in sorted(np.unique(axis["game_month"])):
        mask = axis["game_month"] == month
        months.append({"month": int(month), "gain": _gain(target[mask], parent[mask], candidate[mask])})
    domains = {}
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        mask = axis["domain3"].astype(str) == domain
        if mask.any():
            domains[domain] = _gain(target[mask], parent[mask], candidate[mask])
    row_gain = np.square(parent - target) - np.square(candidate - target)
    direction = direct_minus_parent = axis["direct"] - parent
    return {
        "gain": _gain(target, parent, candidate),
        "positive_month_fraction": float(np.mean([item["gain"] > 0.0 for item in months])),
        "worst_month_gain": float(min(item["gain"] for item in months)),
        "minimum_domain_gain": float(min(domains.values())),
        "active_fraction": float(active.mean()),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
        "error_direction_correlation": float(np.corrcoef(parent - target, direct_minus_parent)[0, 1]),
        "row_gain_mean": float(row_gain.mean()),
        "months": months,
        "domains": domains,
    }


def _load_contract_axis(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as saved:
        return {name: saved[name] for name in saved.files}


load_contract_axis = _load_contract_axis
route_diagnostics = _diagnostics
