"""Month/domain gain diagnostics and frozen-direction composition.

Moved verbatim from ``src/archive/v30_diverse_covariance_screen.py`` during the core extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


V25_ETA = 0.075


V27_ETA = 0.10


def v27_parent(frame: pd.DataFrame) -> np.ndarray:
    """Recover the exact v27 blend from the v22 and v25 OOF columns."""
    v22 = frame["v22"].to_numpy(np.float64)
    v25 = frame["v25"].to_numpy(np.float64)
    return np.clip(v22 + (V27_ETA / V25_ETA) * (v25 - v22), 0.001, 0.999)


def compose(
    parent: np.ndarray,
    v21: np.ndarray,
    signal: np.ndarray,
    mask: np.ndarray,
    *,
    kind: str,
    mode: str,
    weight: float,
) -> np.ndarray:
    """Apply a frozen OOF direction to a disjoint deployment domain."""
    output = np.asarray(parent, dtype=np.float64).copy()
    if kind == "delta":
        direction = np.asarray(signal, dtype=np.float64)
    elif kind == "prediction" and mode == "toward_parent":
        direction = np.asarray(signal, dtype=np.float64) - parent
    elif kind == "prediction" and mode == "delta_v21":
        direction = np.asarray(signal, dtype=np.float64) - np.asarray(
            v21, dtype=np.float64
        )
    else:
        raise ValueError(f"invalid composition: kind={kind!r}, mode={mode!r}")
    output[mask] = np.clip(
        output[mask] + float(weight) * direction[mask], 0.001, 0.999
    )
    return output


def diagnostics(
    frame: pd.DataFrame,
    parent: np.ndarray,
    candidate: np.ndarray,
    apply_mask: np.ndarray,
) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)

    def gain(mask: np.ndarray) -> float:
        local_target = target[mask]
        mse_gain = float(
            np.mean(np.square(local_target - parent[mask]))
            - np.mean(np.square(local_target - candidate[mask]))
        )
        reference = float(np.mean(local_target) * (1.0 - np.mean(local_target)))
        return 1_000_000.0 * mse_gain if reference <= 0.0 else 100_000.0 * mse_gain / reference

    months = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        if np.any(mask & apply_mask):
            months.append({"month": int(month), "gain": gain(mask)})
    domain_gains = {}
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        mask = frame["domain3"].astype(str).eq(domain).to_numpy()
        domain_gains[domain] = gain(mask)
    return {
        "gain": gain(np.ones(len(frame), dtype=bool)),
        "positive_month_fraction": float(
            np.mean([item["gain"] > 0.0 for item in months])
        ),
        "worst_month_gain": float(min(item["gain"] for item in months)),
        "minimum_domain_gain": float(min(domain_gains.values())),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
        "months": months,
        "domain_gains": domain_gains,
    }
