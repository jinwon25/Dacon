"""Frozen numerical contracts for the deployed v148 package."""

from __future__ import annotations

import numpy as np


DEPLOYED_BRIDGE_WEIGHT = 0.15


def reconstruct_deployed_v148(v142: np.ndarray, v138: np.ndarray) -> np.ndarray:
    """Reconstruct deployed v148, not v147's separately selected 5% audit point."""

    parent = np.asarray(v142, dtype=np.float64)
    endpoint = np.asarray(v138, dtype=np.float64)
    if parent.shape != endpoint.shape:
        raise ValueError("v142 and v138 arrays must be aligned")
    return np.clip(
        parent + DEPLOYED_BRIDGE_WEIGHT * (endpoint - parent), 0.001, 0.999
    )
