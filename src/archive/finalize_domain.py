"""Append executed domain experiments to the immutable experiment ledger."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.archive.data import TARGET_COL, read_main


def _skill(target: np.ndarray, brier: float) -> float:
    rate = float(target.mean())
    reference = rate * (1.0 - rate)
    return max(0.0, 100000.0 * (1.0 - float(brier) / reference))


def run(project_dir: Path) -> int:
    ledger_path = project_dir / "reports" / "experiments.csv"
    ledger = pd.read_csv(ledger_path)
    columns = ledger.columns.tolist()
    existing = set(ledger["experiment_id"].astype(str))
    now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    train = read_main(project_dir / "data" / "train.csv")
    targets = {
        year: train.loc[train["season"] == year, TARGET_COL].to_numpy()
        for year in (2021, 2022, 2023, 2024)
    }
    rows: list[dict[str, object]] = []

    drift = pd.read_csv(project_dir / "reports" / "domain_drift_results.csv")
    for record in drift.to_dict("records"):
        year = int(record["outer_validation_season"])
        rows.append(
            {
                "experiment_id": f"D7_game_type_regime_residual_v1_{year}",
                "run_date": now,
                "validation_split": f"train_before_{year}_validate_{year}",
                "feature_groups": "frozen incumbent plus train-only game_type regime residual",
                "model_params": json.dumps(
                    {
                        "minimum_rate_residual_gap": 0.05,
                        "minimum_group_rows": 5000,
                        "offsets": json.loads(record["offsets"]),
                    },
                    sort_keys=True,
                ),
                "seed": 42,
                "raw_brier_score": record["brier"],
                "calibrated_brier_score": "",
                "local_brier_skill_score": _skill(targets[year], record["brier"]),
                "calibration": "frozen game_type logit offset",
                "training_seconds": 0.0,
                "inference_seconds": "",
                "model_size_mb": 0.001,
                "peak_memory_mb": "",
                "leakage_risk_notes": (
                    "offset fit on outer-train annual target rates only; row-local game_type apply; "
                    "outer years are now development data"
                ),
            }
        )

    rf = pd.read_csv(project_dir / "reports" / "rf_seed_results.csv")
    for record in rf.to_dict("records"):
        year = int(record["outer_validation_season"])
        rows.append(
            {
                "experiment_id": f"D8_{record['candidate']}_{year}",
                "run_date": now,
                "validation_split": f"train_before_{year}_validate_{year}",
                "feature_groups": "official features; fixed RF probability seed average",
                "model_params": json.dumps(
                    {"seeds": [42, 202, 777], "n_estimators_each": 100},
                    sort_keys=True,
                ),
                "seed": 42,
                "raw_brier_score": record["brier"],
                "calibrated_brier_score": "",
                "local_brier_skill_score": _skill(targets[year], record["brier"]),
                "calibration": (
                    "incumbent trend plus regime"
                    if "plus_regime" in record["candidate"]
                    else "incumbent trend"
                ),
                "training_seconds": "",
                "inference_seconds": "",
                "model_size_mb": "",
                "peak_memory_mb": "",
                "leakage_risk_notes": (
                    "fixed seeds with no seed selection; stopped after 2023/2024 pilot; "
                    "2023 exceeded worst-fold threshold"
                ),
            }
        )

    features = pd.read_csv(project_dir / "reports" / "domain_feature_results.csv")
    for record in features.to_dict("records"):
        year = int(record["outer_validation_season"])
        rows.append(
            {
                "experiment_id": f"D9_{record['candidate']}_{year}",
                "run_date": now,
                "validation_split": f"train_before_{year}_validate_{year}",
                "feature_groups": record["feature_set"],
                "model_params": json.dumps(
                    {
                        "base": "lgb_engineered_l31",
                        "best_iteration": int(record["best_iteration"]),
                    },
                    sort_keys=True,
                ),
                "seed": 42,
                "raw_brier_score": record["brier"],
                "calibrated_brier_score": "",
                "local_brier_skill_score": _skill(targets[year], record["brier"]),
                "calibration": (
                    "incumbent trend plus regime"
                    if "plus_regime" in record["candidate"]
                    else "incumbent trend"
                ),
                "training_seconds": record["fit_seconds"],
                "inference_seconds": record["inference_seconds"],
                "model_size_mb": "",
                "peak_memory_mb": record["peak_memory_mb"],
                "leakage_risk_notes": (
                    "row-local official as-of features; category maps fit on outer train; "
                    "regime boundary developed on repeated outer folds"
                ),
            }
        )

    new_rows = [row for row in rows if row["experiment_id"] not in existing]
    if new_rows:
        frame = pd.DataFrame(new_rows)
        frame = frame.reindex(columns=columns)
        frame.to_csv(ledger_path, mode="a", header=False, index=False)
    print(f"appended {len(new_rows)} domain experiment rows")
    return len(new_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.project_dir.resolve())


if __name__ == "__main__":
    main()
