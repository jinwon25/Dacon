"""Audit the deployed H1 pre-season ASOF bank against official train state.

The H1 runtime subtracts a frozen pre-season count/event tuple from each query
row's cumulative ASOF values.  A count mismatch would change the meaning of all
current-season rates.  This audit checks the packaged v290 H1 bank against the
latest observable official-train state without reading the evaluation file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SPECS = {
    "pitch": ("pitcher_id", "asof_pitcher_n"),
    "mix": ("pitcher_id", "asof_pitcher_pitchmix_n"),
    "bat": ("batter_id", "asof_batter_n"),
}


def run(train_csv: Path, bundle_path: Path, output_json: Path) -> dict:
    columns = sorted({column for spec in SPECS.values() for column in spec})
    train = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    bundle = joblib.load(bundle_path)
    prior = bundle["asof_prior"]
    result: dict[str, object] = {
        "protocol": "AUDIT_H1_ASOF_PRIOR_V1",
        "train_csv": str(train_csv),
        "bundle_path": str(bundle_path),
        "kinds": {},
    }
    all_exact = True
    for kind, (id_column, n_column) in SPECS.items():
        grouped = train.groupby(id_column, observed=True)[n_column].agg(
            first="min", latest_pre_pitch="max", row_count="count"
        )
        packaged = pd.Series(
            {int(entity): float(values[0]) for entity, values in prior[kind].items()},
            name="packaged_prior_n",
        )
        joined = grouped.join(packaged, how="outer")
        joined["latest_observable_post_pitch"] = joined["latest_pre_pitch"] + 1.0
        count_error = (
            joined["packaged_prior_n"] - joined["latest_observable_post_pitch"]
        )
        ids_match = bool(joined["packaged_prior_n"].notna().all())
        exact = bool(ids_match and np.all(np.abs(count_error.to_numpy(float)) <= 1e-12))
        all_exact &= exact
        result["kinds"][kind] = {
            "entities": int(len(joined)),
            "ids_match": ids_match,
            "all_first_counts_zero": bool((joined["first"] == 0).all()),
            "packaged_equals_train_row_count": bool(
                np.allclose(joined["packaged_prior_n"], joined["row_count"])
            ),
            "packaged_equals_latest_pre_pitch_plus_one": exact,
            "max_abs_count_error": float(np.nanmax(np.abs(count_error))),
        }
    result["count_anchor_exact"] = bool(all_exact)
    result["remaining_terminal_event_uncertainty"] = (
        "At most one terminal pitch event per entity for rate numerators; "
        "the deployed bank uses the latest observable cumulative event total."
    )
    result["decision"] = (
        "no_material_h1_count_anchor_bug" if all_exact else "retrain_h1_with_exact_anchor"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.bundle, args.output_json), indent=2))


if __name__ == "__main__":
    main()
