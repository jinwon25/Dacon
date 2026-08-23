"""Residual overlay feature frame and Brier skill helper.

Moved verbatim from ``src/v20_residual_overlay_screen.py`` during the core extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


WEIGHTS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.25, 0.50, 1.00)


def _bss(target: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(np.mean(target))
    reference = rate * (1.0 - rate)
    return float(100000.0 * (1.0 - np.mean((target - prediction) ** 2) / reference))


def _feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    balls = frame["balls_before"].to_numpy()
    strikes = frame["strikes_before"].to_numpy()
    frame["pressure"] = np.where(
        balls == 3, "threeball", np.where(strikes == 2, "twostrike", "normal")
    )
    frame["inning_band"] = pd.cut(
        frame["inning"],
        bins=(-np.inf, 3, 6, 9, np.inf),
        labels=("early", "middle", "late", "extra"),
    ).astype("string")
    frame["li_band"] = pd.cut(
        frame["li"],
        bins=(-np.inf, 0.7, 1.2, 2.0, np.inf),
        labels=("low", "medium", "high", "extreme"),
    ).astype("string")
    return frame
