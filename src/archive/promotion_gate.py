"""Apply the pre-registered promotion gate and write a candidate manifest.

No archive is created unless every condition is true.  A failed recipe gets a
durable manifest so its result cannot be accidentally resubmitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.data import read_main
from src.metrics import brier_score


def _bootstrap_fraction(project: Path, recipe: str, repeats: int = 500) -> tuple[float, float]:
    preds = np.load(project / "artifacts/followup/residual_predictions.npz", allow_pickle=False)
    train = read_main(project / "data/train.csv", usecols=["pitcher_id"])
    pieces = []
    for year in (2022, 2023, 2024):
        key = f"{year}_{recipe}"
        if key not in preds:
            continue
        p = preds[key].astype(float)
        base = preds[f"{year}_base"].astype(float)
        idx = np.load(project / "artifacts/followup/v2_nested_predictions.npz", allow_pickle=False)[f"{year}_valid_idx"].astype(int)
        y = np.load(project / "artifacts/followup/v2_nested_predictions.npz", allow_pickle=False)[f"{year}_target"].astype(float)
        pieces.append(pd.DataFrame({"year": year, "pitcher": train.iloc[idx]["pitcher_id"].to_numpy(), "delta_loss": (p - y) ** 2 - (base - y) ** 2}))
    if not pieces:
        return 0.0, 0.0
    data = pd.concat(pieces, ignore_index=True)
    pitchers = data["pitcher"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(42)
    negatives = 0; negatives_2024 = 0
    weights = {2022: 1.0, 2023: 2.0, 2024: 3.0}
    for _ in range(repeats):
        sample = rng.choice(pitchers, size=len(pitchers), replace=True)
        sampled = data[data["pitcher"].isin(sample)]
        fold = sampled.groupby("year", observed=True)["delta_loss"].mean()
        score = sum(weights[y] * float(fold.get(y, 0.0)) for y in weights) / sum(weights[y] for y in weights if y in fold)
        negatives += int(score < 0)
        if 2024 in fold: negatives_2024 += int(float(fold[2024]) < 0)
    return negatives / repeats, negatives_2024 / repeats


def evaluate(project: Path) -> pd.DataFrame:
    results = pd.read_csv(project / "reports/residual_recipe_results.csv")
    config = json.loads((project / "configs/next_cycle_20260809.json").read_text(encoding="utf-8"))
    rows = []
    for recipe, group in results.groupby("recipe", observed=True):
        if not (project / "artifacts/followup/residual_predictions.npz").exists() or group["brier"].isna().all():
            rows.append({"recipe": recipe, "delta_2022": np.nan, "delta_2023": np.nan, "delta_2024": np.nan, "recency_weighted_delta": np.nan, "worst_fold_delta": np.nan, "strict_fold_count": 0, "bootstrap_fraction_negative": 0.0, "bootstrap_2024_fraction_negative": 0.0, "checks": json.dumps({"nested_oof_available": False}), "pass": False})
            continue
        folds = group.set_index("outer_validation_season")
        deltas = {int(year): float(row["delta_v2"]) for year, row in group.set_index("outer_validation_season").iterrows()}
        locked = [deltas[y] for y in (2022, 2023, 2024) if y in deltas]
        rw = sum((i + 1) * value for i, value in enumerate(locked)) / sum(range(1, len(locked) + 1)) if locked else 0.0
        worst = max(locked) if locked else 0.0
        strict_count = sum(value < 0 for value in locked)
        frac, frac24 = _bootstrap_fraction(project, recipe)
        safety = group.get("correction_abs_p995", pd.Series([0.0])).max() <= config["promotion_gate"]["correction_abs_p995_max"] and group.get("correction_abs_max", pd.Series([0.0])).max() <= config["promotion_gate"]["correction_abs_max"] and group.get("correction_mean", pd.Series([0.0])).abs().max() <= config["promotion_gate"]["correction_abs_mean_max"]
        checks = {
            "recency_weighted_delta": rw <= config["promotion_gate"]["recency_weighted_delta_max"],
            "locked_2024_delta": deltas.get(2024, 0.0) <= config["promotion_gate"]["locked_2024_delta_max"],
            "worst_fold": worst <= config["promotion_gate"]["worst_fold_delta_max"],
            "minimum_strict_folds": strict_count >= config["promotion_gate"]["minimum_strictly_improved_folds"],
            "bootstrap_combined": frac >= config["promotion_gate"]["bootstrap_fraction_negative_min"],
            "bootstrap_2024": frac24 >= config["promotion_gate"]["bootstrap_fraction_negative_min"],
            "correction_safety": bool(safety),
        }
        rows.append({"recipe": recipe, "delta_2022": deltas.get(2022, np.nan), "delta_2023": deltas.get(2023, np.nan), "delta_2024": deltas.get(2024, np.nan), "recency_weighted_delta": rw, "worst_fold_delta": worst, "strict_fold_count": strict_count, "bootstrap_fraction_negative": frac, "bootstrap_2024_fraction_negative": frac24, "checks": json.dumps(checks, sort_keys=True), "pass": all(checks.values())})
    out = pd.DataFrame(rows)
    out.to_csv(project / "reports/next_cycle_bootstrap.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); args = parser.parse_args(); project = args.project_dir.resolve()
    out = evaluate(project)
    candidate = project / "artifacts/candidates/main_residual_nested_v1"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "manifest.json").write_text(json.dumps({"candidate": "main_residual_nested_v1", "parent": "submit_v2.zip", "status": "promoted" if out["pass"].any() else "not_promoted", "reason": "all preregistered gates required", "gate_rows": out.to_dict(orient="records")}, indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
