"""Season-level training features. Do not aggregate evaluation rows with these functions."""
import numpy as np
import pandas as pd


def multi_scale_success(frame):
    """Eight smoothed pitcher/batter season rates from the supplied training frame."""
    result = {}
    for player, count, rate, prefix in (
        ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", "p_succ"),
        ("batter_id", "asof_batter_n", "asof_batter_success_rate", "b_succ"),
    ):
        values = frame[[player, "season", count, rate]].copy()
        values["success"] = values[count] * values[rate].fillna(0)
        anchor = values.loc[values.groupby([player, "season"])[count].idxmin()]
        anchor = anchor.set_index([player, "season"])[[count, "success"]]
        aligned = frame[[player, "season"]].join(anchor, on=[player, "season"])
        n = np.maximum(frame[count].to_numpy() - aligned[count].fillna(0).to_numpy(), 0)
        successes = np.maximum(
            np.nan_to_num(frame[count].to_numpy() * frame[rate].to_numpy())
            - aligned["success"].fillna(0).to_numpy(), 0)
        prior = np.nanmean(frame[rate])
        for strength in (25, 75, 400, 1000):
            result[f"{prefix}_k{strength}"] = (successes + strength * prior) / (n + strength)
    return pd.DataFrame(result, index=frame.index).astype(np.float32)
