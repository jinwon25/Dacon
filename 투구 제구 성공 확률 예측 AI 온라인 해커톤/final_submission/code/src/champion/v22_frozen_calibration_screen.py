"""Forward-audit frozen residual calibration maps above v21.

Calibration maps are fitted on one labelled source window and applied without
refitting to a later audit window.  Fixed-width probability bins are used so
the transform remains row-local and can be serialized for 2025 inference.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.overlay import _bss


BIN_COUNTS = (1, 4, 8, 16)
ALPHAS = (100.0, 500.0, 1000.0, 5000.0, 20000.0)
WEIGHTS = (0.25, 0.50, 0.75, 1.00)
DOMAINS = ("R_CORE", "R_ANCHOR", "F")


def _bin(probability: np.ndarray, count: int) -> np.ndarray:
    if count == 1:
        return np.zeros(len(probability), dtype=np.int16)
    scaled = (np.clip(probability, 0.30, 0.70) - 0.30) / 0.40
    return np.minimum(np.floor(scaled * count).astype(np.int16), count - 1)


def _fit_map(
    target: np.ndarray,
    prediction: np.ndarray,
    domain: np.ndarray,
    bins: int,
    alpha: float,
) -> dict[str, float]:
    keys = pd.Series(domain, dtype="string") + "\x1f" + pd.Series(
        _bin(prediction, bins), dtype="string"
    )
    table = pd.DataFrame({"key": keys, "residual": target - prediction}).groupby(
        "key", observed=True
    )["residual"].agg(["sum", "count"])
    return (table["sum"] / (table["count"] + alpha)).to_dict()


def _apply_map(
    prediction: np.ndarray,
    domain: np.ndarray,
    bins: int,
    effect: dict[str, float],
    weight: float,
) -> np.ndarray:
    keys = pd.Series(domain, dtype="string") + "\x1f" + pd.Series(
        _bin(prediction, bins), dtype="string"
    )
    correction = keys.map(effect).fillna(0.0).to_numpy(np.float64)
    return np.clip(prediction + weight * correction, 0.001, 0.999)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as saved:
        return {
            "target": saved["target"].astype(np.float64),
            "v21": saved["v21"].astype(np.float64),
            "domain": saved["domain3"].astype(str),
            "month": saved["game_month"].astype(np.int16),
        }


def _diagnostics(fold: dict[str, np.ndarray], candidate: np.ndarray) -> dict[str, object]:
    target = fold["target"]
    parent = fold["v21"]
    months = []
    for month in sorted(np.unique(fold["month"])):
        mask = fold["month"] == month
        months.append(
            _bss(target[mask], candidate[mask]) - _bss(target[mask], parent[mask])
        )
    domains = []
    for domain in DOMAINS:
        mask = fold["domain"] == domain
        domains.append(
            _bss(target[mask], candidate[mask]) - _bss(target[mask], parent[mask])
        )
    gain = _bss(target, candidate) - _bss(target, parent)
    loss_improvement = np.square(target - parent) - np.square(target - candidate)
    scale = 100000.0 / (target.mean() * (1.0 - target.mean()))
    se = scale * float(np.std(loss_improvement, ddof=1)) / np.sqrt(len(target))
    return {
        "gain": gain,
        "row_normal_ci95": [gain - 1.96 * se, gain + 1.96 * se],
        "positive_month_fraction": float(np.mean(np.asarray(months) > 0.0)),
        "worst_month_gain": float(min(months)),
        "minimum_domain_gain": float(min(domains)),
    }


def run(champion_dir: Path, output_dir: Path) -> dict[str, object]:
    champion_dir = champion_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    y23_late = _load(champion_dir / "y2023_early_to_late.npz")
    y24_full = _load(champion_dir / "y2023_to_y2024.npz")
    y24_late = _load(champion_dir / "y2024_early_to_late.npz")
    early_mask = y24_full["month"] <= 7
    y24_early = {key: value[early_mask] for key, value in y24_full.items()}
    late_mask = y24_full["month"] >= 8
    if not np.array_equal(y24_full["target"][late_mask], y24_late["target"]):
        raise ValueError("2024 late target alignment failed")
    pairs = {
        "y2023_late_to_y2024_full": (y23_late, y24_full),
        "y2024_early_to_late": (y24_early, y24_late),
    }

    rows = []
    for bins, alpha, weight in itertools.product(BIN_COUNTS, ALPHAS, WEIGHTS):
        gains = {}
        diagnostics = {}
        for name, (source, audit) in pairs.items():
            effect = _fit_map(
                source["target"], source["v21"], source["domain"], bins, alpha
            )
            candidate = _apply_map(
                audit["v21"], audit["domain"], bins, effect, weight
            )
            diagnostics[name] = _diagnostics(audit, candidate)
            gains[name] = diagnostics[name]["gain"]
        rows.append(
            {
                "bins": bins,
                "alpha": alpha,
                "weight": weight,
                **gains,
                "min_gain": min(gains.values()),
                "mean_gain": float(np.mean(list(gains.values()))),
                **{
                    f"{name}__positive_month_fraction": value["positive_month_fraction"]
                    for name, value in diagnostics.items()
                },
                **{
                    f"{name}__minimum_domain_gain": value["minimum_domain_gain"]
                    for name, value in diagnostics.items()
                },
            }
        )
    metrics = pd.DataFrame(rows).sort_values(
        ["min_gain", "mean_gain"], ascending=False
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    best = metrics.iloc[0]
    best_diagnostics = {}
    learned_maps = {}
    for name, (source, audit) in pairs.items():
        effect = _fit_map(
            source["target"],
            source["v21"],
            source["domain"],
            int(best["bins"]),
            float(best["alpha"]),
        )
        learned_maps[name] = effect
        candidate = _apply_map(
            audit["v21"], audit["domain"], int(best["bins"]), effect, float(best["weight"])
        )
        best_diagnostics[name] = _diagnostics(audit, candidate)
    summary = {
        "protocol": "V22_FROZEN_DOMAIN_PROBABILITY_CALIBRATION_V1",
        "candidate_count": int(len(metrics)),
        "strictly_positive": int((metrics["min_gain"] > 0.0).sum()),
        "best": metrics.head(30).to_dict("records"),
        "best_diagnostics": best_diagnostics,
        "best_learned_maps": learned_maps,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--champion-dir", type=Path, default=Path("artifacts/champion_oof_20260817_01")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v22_frozen_calibration_20260817_01"),
    )
    args = parser.parse_args()
    run(args.champion_dir, args.output_dir)


if __name__ == "__main__":
    main()
