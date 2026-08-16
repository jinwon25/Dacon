"""Forward-only batter and pitcher-batter residual screen above the v17 analogue.

The v17 correction models pitcher x batter-hand x pressure, but it never uses
the actual batter identity.  This experiment asks whether persistent batter or
pitcher-batter residuals add resolution after v17.  Every audit season is
predicted from earlier labelled seasons only; unseen keys receive zero change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v16_residual_calibration_screen import EPSILON, _bss_gain, _keys, load_v14_folds


V17_NAME = "multi_pitcher_batter_hand_pressure_d1_a3200_w1.5"
GROUPS: dict[str, tuple[str, ...]] = {
    "batter": ("batter_id",),
    "batter_pitcher_hand": ("batter_id", "pitcher_hand"),
    "batter_count": ("batter_id", "balls_before", "strikes_before"),
    "batter_pressure": ("batter_id", "pressure"),
    "batter_pitcher_hand_pressure": ("batter_id", "pitcher_hand", "pressure"),
    "pitcher_batter": ("pitcher_id", "batter_id"),
    "pitcher_batter_count": (
        "pitcher_id",
        "batter_id",
        "balls_before",
        "strikes_before",
    ),
    "pitcher_batter_pressure": ("pitcher_id", "batter_id", "pressure"),
    "pitcher_batter_base": ("pitcher_id", "batter_id", "base_state"),
}


def _v17_fold(
    project: Path,
    year: int,
    fallback: np.ndarray,
) -> np.ndarray:
    path = (
        project
        / "artifacts"
        / "v16_multiseason_20260815_02"
        / f"{V17_NAME}_o{year}.npz"
    )
    if not path.exists():
        return fallback.copy()
    with np.load(path) as saved:
        candidate = saved["candidate"].astype(np.float64)
        target = saved["target"].astype(np.float64)
    if len(candidate) != len(fallback) or len(target) != len(fallback):
        raise ValueError(f"v17 fold length mismatch for {year}")
    return candidate


def _history_stats(
    folds: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    incumbents: dict[int, np.ndarray],
    audit_year: int,
    columns: tuple[str, ...],
    decay: float,
    source_domain: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in sorted(year for year in folds if year < audit_year):
        rows, target, _ = folds[year]
        prediction = incumbents[year]
        mask = (
            rows["domain3"].eq("R_CORE").to_numpy()
            if source_domain == "R_CORE"
            else np.ones(len(rows), dtype=bool)
        )
        weight = float(decay) ** float(audit_year - 1 - year)
        parts.append(
            pd.DataFrame(
                {
                    "key": _keys(rows.loc[mask].reset_index(drop=True), columns),
                    "weighted_residual": weight * (target[mask] - prediction[mask]),
                    "weight": weight,
                }
            )
        )
    history = pd.concat(parts, ignore_index=True)
    return history.groupby("key", observed=True).agg(
        weighted_residual=("weighted_residual", "sum"),
        effective_count=("weight", "sum"),
    )


def _map_effect(
    stats: pd.DataFrame,
    audit_rows: pd.DataFrame,
    columns: tuple[str, ...],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    effect = stats["weighted_residual"] / (stats["effective_count"] + float(alpha))
    keys = _keys(audit_rows, columns)
    mapped = keys.map(effect)
    return mapped.fillna(0.0).to_numpy(np.float64), mapped.notna().to_numpy(bool)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    folds = load_v14_folds(project, train)
    incumbents = {
        year: _v17_fold(project, year, fold[2]) for year, fold in folds.items()
    }
    rows_out: list[dict[str, object]] = []
    for audit_year in (2023, 2024):
        audit_rows, target, _ = folds[audit_year]
        incumbent = incumbents[audit_year]
        for group_name, columns in GROUPS.items():
            pair_group = group_name.startswith("pitcher_batter")
            alphas = (
                (10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0)
                if pair_group
                else (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0)
            )
            source_domain = "R_CORE"
            apply_domain = "R_CORE"
            apply_mask = audit_rows["domain3"].eq("R_CORE").to_numpy()
            for decay in (0.5, 0.75, 1.0):
                stats = _history_stats(
                    folds,
                    incumbents,
                    audit_year,
                    columns,
                    decay,
                    source_domain,
                )
                for alpha in alphas:
                    correction, seen = _map_effect(stats, audit_rows, columns, alpha)
                    changed = apply_mask & seen
                    for weight in (0.25, 0.5, 1.0, 1.5):
                        name = (
                            f"{group_name}_src{source_domain}_apply{apply_domain}"
                            f"_d{decay:g}_a{alpha:g}_w{weight:g}"
                        )
                        candidate = incumbent.copy()
                        candidate[apply_mask] += weight * correction[apply_mask]
                        candidate = np.clip(candidate, EPSILON, 1.0 - EPSILON)
                        rows_out.append(
                            {
                                "candidate": name,
                                "audit_year": audit_year,
                                "group": group_name,
                                "source_domain": source_domain,
                                "apply_domain": apply_domain,
                                "decay": decay,
                                "alpha": alpha,
                                "weight": weight,
                                "coverage": float(changed.mean()),
                                "gain_vs_v17": _bss_gain(
                                    target, candidate, incumbent
                                ),
                            }
                        )
    metrics = pd.DataFrame(rows_out)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust = (
        metrics.groupby("candidate", observed=True)["gain_vs_v17"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False)
    )
    robust["positive_folds"] = (
        metrics.assign(positive=metrics["gain_vs_v17"].gt(0.0))
        .groupby("candidate", observed=True)["positive"]
        .sum()
        .reindex(robust["candidate"])
        .to_numpy()
    )
    robust.to_csv(output_dir / "robust.csv", index=False)
    passing = robust.loc[robust["min_gain"].gt(0.0)]
    summary = {
        "protocol": "ACTUAL_BATTER_MATCHUP_RESIDUAL_FORWARD_V1",
        "audit_years": [2023, 2024],
        "source_prediction": {
            "2022": "v14 fallback (v17 cache unavailable)",
            "2023": "v17 analogue",
        },
        "n_recipes": int(metrics["candidate"].nunique()),
        "n_strict_improvements": int(len(passing)),
        "best": robust.head(30).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/matchup_residual_20260816_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
