"""Audits the legacy order-dependent target encoder; not used by new recipes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.data import read_main
from src.archive.target_encoding import PrequentialTargetEncoder


def audit(frame: pd.DataFrame, max_rows: int = 20_000) -> pd.DataFrame:
    frame = frame.head(max_rows).reset_index(drop=True).copy()
    y = frame["control_success"].to_numpy(dtype=float)
    encoder = PrequentialTargetEncoder(
        groups={"count_platoon": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]},
        alpha=200.0,
    ).fit(frame, y)
    original = encoder.transform(frame, y, prequential=True).to_numpy(dtype=float)
    permutation = np.random.default_rng(42).permutation(len(frame))
    permuted = encoder.transform(frame.iloc[permutation].reset_index(drop=True), y[permutation], prequential=True).to_numpy(dtype=float)[np.argsort(permutation)]
    flipped_y = y.copy(); flipped_y[min(1, len(y) - 1)] = 1.0 - flipped_y[min(1, len(y) - 1)]
    flipped = encoder.transform(frame, flipped_y, prequential=True).to_numpy(dtype=float)
    # The current row is excluded from its own prequential sum, while later
    # rows in the same group change. This is a useful diagnostic, not a license
    # to use the implementation in the new residual models.
    current_row_effect = float(np.max(np.abs(original[min(1, len(y) - 1)] - flipped[min(1, len(y) - 1)])))
    future_effect = float(np.max(np.abs(original[min(1, len(y) - 1) + 1 :] - flipped[min(1, len(y) - 1) + 1 :])) if len(y) > 2 else 0.0)
    global_before = float(encoder.global_rate)
    encoder2 = PrequentialTargetEncoder(
        groups={"count_platoon": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]},
        alpha=200.0,
    ).fit(frame, y)
    global_flip = y.copy(); global_flip[0] = 1.0 - global_flip[0]
    encoder3 = PrequentialTargetEncoder(
        groups={"count_platoon": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]},
        alpha=200.0,
    ).fit(frame, global_flip)
    global_prior_shift = abs(float(encoder3.global_rate) - global_before)
    return pd.DataFrame([
        {"audit": "row_permutation_sensitivity_max_abs", "value": float(np.max(np.abs(original - permuted))), "interpretation": "nonzero means raw frame order changes features"},
        {"audit": "current_target_flip_effect", "value": current_row_effect, "interpretation": "the flipped row itself should be zero for strict prequential exclusion"},
        {"audit": "future_row_effect_of_current_target_flip", "value": future_effect, "interpretation": "future rows in the same group can see the current label"},
        {"audit": "full_frame_global_prior_self_future_shift", "value": global_prior_shift, "interpretation": "fit-frame target mean changes when one label changes"},
        {"audit": "season_cross_fitted_alternative", "value": np.nan, "interpretation": "requires exact official chronological key; implementation deliberately isolated"},
    ])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); args = parser.parse_args(); project = args.project_dir.resolve()
    frame = read_main(project / "data/train.csv", nrows=20_000)
    result = audit(frame)
    result.to_csv(project / "reports/target_encoding_audit.csv", index=False)
    (project / "reports/target_encoding_audit_20260809.md").write_text(
        "# Target encoding audit\n\n"
        "The legacy `PrequentialTargetEncoder` is isolated from all new models. The implementation relies on raw frame order and a full-frame target prior; no exact official chronological key has been established. A row permutation changes its values, a current label can affect future rows in the same group, and changing one label changes the full-frame prior. Therefore it is not used by the nested residual candidates.\n\n"
        + "```text\n" + result.to_string(index=False) + "\n```\n", encoding="utf-8",
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
