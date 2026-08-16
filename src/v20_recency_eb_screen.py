"""Recency-weighted forward empirical-Bayes screen above v19."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v20_residual_overlay_screen import (
    ALPHAS,
    DOMAINS,
    GROUPS,
    WEIGHTS,
    _load_folds,
    _quadratic_gain_grid,
    _splits,
)


HALF_LIVES = (0.5, 1.0, 2.0, 4.0, 8.0)
RECENCY_GROUP_NAMES = {
    "pitcher_batter_hand",
    "pitcher_hand_pressure",
    "pitcher_pressure",
    "pitcher_count",
    "pitcher_hand_team",
    "count_hands",
}
RECENCY_GROUPS = tuple(group for group in GROUPS if group.name in RECENCY_GROUP_NAMES)


def _mapped_stats_bank(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    columns: tuple[str, ...],
    domain: str,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    fit_mask = np.ones(len(fit), dtype=bool)
    audit_mask = np.ones(len(audit), dtype=bool)
    if domain != "ALL":
        fit_mask &= fit["domain3"].eq(domain).to_numpy()
        audit_mask &= audit["domain3"].eq(domain).to_numpy()
    source = fit.loc[fit_mask]
    source_keys = pd.MultiIndex.from_frame(source.loc[:, list(columns)])
    codes, uniques = pd.factorize(source_keys, sort=False)
    destination = audit.loc[audit_mask, list(columns)]
    destination_codes = uniques.get_indexer(pd.MultiIndex.from_frame(destination))
    known = destination_codes >= 0
    positions = np.flatnonzero(audit_mask)[known]
    residual = source["residual"].to_numpy(np.float64)
    month = source["game_month"].to_numpy(np.float64)
    cutoff = float(np.max(month))
    output: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for half_life in HALF_LIVES:
        row_weight = np.exp2(-(cutoff - month) / half_life)
        residual_sum = np.bincount(
            codes, weights=row_weight * residual, minlength=len(uniques)
        )
        effective_n = np.bincount(
            codes, weights=row_weight, minlength=len(uniques)
        )
        mapped_sum = np.zeros(len(audit), dtype=np.float64)
        mapped_n = np.zeros(len(audit), dtype=np.float64)
        mapped_sum[positions] = residual_sum[destination_codes[known]]
        mapped_n[positions] = effective_n[destination_codes[known]]
        output[half_life] = mapped_sum, mapped_n
    return output


def _correction(stats: tuple[np.ndarray, np.ndarray], alpha: float) -> np.ndarray:
    residual_sum, n = stats
    output = np.zeros_like(residual_sum)
    known = n > 0.0
    output[known] = residual_sum[known] / (n[known] + alpha)
    return output


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = _splits(_load_folds(project))
    rows: list[dict[str, object]] = []
    for group in RECENCY_GROUPS:
        for domain in DOMAINS:
            banks = {
                split_name: _mapped_stats_bank(fit, audit, group.columns, domain)
                for split_name, (fit, audit) in splits.items()
            }
            for half_life in HALF_LIVES:
                for alpha in ALPHAS:
                    for split_name, (_, audit) in splits.items():
                        raw = _correction(banks[split_name][half_life], alpha)
                        gains, worst_months, month_fractions, worst_domains = (
                            _quadratic_gain_grid(audit, raw)
                        )
                        for index, weight in enumerate(WEIGHTS):
                            rows.append(
                                {
                                    "group": group.name,
                                    "columns": "+".join(group.columns),
                                    "domain": domain,
                                    "half_life": half_life,
                                    "alpha": alpha,
                                    "weight": weight,
                                    "split": split_name,
                                    "gain": float(gains[index]),
                                    "worst_month_gain": float(worst_months[index]),
                                    "month_positive_fraction": float(
                                        month_fractions[index]
                                    ),
                                    "worst_domain_gain": float(worst_domains[index]),
                                }
                            )
    metrics = pd.DataFrame(rows)
    robust = (
        metrics.groupby(
            ["group", "columns", "domain", "half_life", "alpha", "weight"],
            as_index=False,
        )
        .agg(
            min_gain=("gain", "min"),
            mean_gain=("gain", "mean"),
            max_gain=("gain", "max"),
            min_worst_month_gain=("worst_month_gain", "min"),
            min_month_positive_fraction=("month_positive_fraction", "min"),
            min_worst_domain_gain=("worst_domain_gain", "min"),
        )
        .sort_values(["min_gain", "mean_gain"], ascending=False)
        .reset_index(drop=True)
    )
    family_best = (
        robust.loc[robust["min_gain"] > 0.0]
        .drop_duplicates(["group", "domain"])
        .head(12)
        .reset_index(drop=True)
    )
    family_records = family_best.to_dict("records")
    group_by_name = {group.name: group for group in RECENCY_GROUPS}
    signal_bank: dict[str, list[np.ndarray]] = {name: [] for name in splits}
    for choice in family_records:
        group = group_by_name[str(choice["group"])]
        for split_name, (fit, audit) in splits.items():
            stats = _mapped_stats_bank(
                fit, audit, group.columns, str(choice["domain"])
            )[float(choice["half_life"])]
            signal_bank[split_name].append(_correction(stats, float(choice["alpha"])))
    quadratic: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, (_, audit) in splits.items():
        signals = np.column_stack(signal_bank[split_name])
        target = audit["target"].to_numpy(np.float64)
        incumbent = audit["v19"].to_numpy(np.float64)
        residual = target - incumbent
        rate = float(np.mean(target))
        scale = 100000.0 / (rate * (1.0 - rate))
        quadratic[split_name] = (
            scale * 2.0 * np.mean(residual[:, None] * signals, axis=0),
            scale * (signals.T @ signals) / len(signals),
        )
    ensemble_weights = (0.05, 0.10, 0.15, 0.25, 0.50, 1.00)
    ensemble_rows: list[dict[str, object]] = []
    for size in (1, 2, 3):
        for indices in itertools.combinations(range(len(family_records)), size):
            index = np.asarray(indices, dtype=np.int64)
            choices = [family_records[i] for i in indices]
            for weights in itertools.product(ensemble_weights, repeat=size):
                weight = np.asarray(weights, dtype=np.float64)
                gains = {}
                for split_name, (linear, cross) in quadratic.items():
                    gains[split_name] = float(
                        linear[index] @ weight
                        - weight @ cross[np.ix_(index, index)] @ weight
                    )
                ensemble_rows.append(
                    {
                        "members": ";".join(
                            f"{row['group']}|{row['domain']}|h{float(row['half_life']):g}"
                            f"|a{float(row['alpha']):g}|w{member_weight:g}"
                            for row, member_weight in zip(choices, weights, strict=True)
                        ),
                        **gains,
                        "min_gain": min(gains.values()),
                        "mean_gain": float(np.mean(list(gains.values()))),
                    }
                )
    ensembles = pd.DataFrame(ensemble_rows).sort_values(
        ["min_gain", "mean_gain"], ascending=False
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust.to_csv(output_dir / "robust.csv", index=False)
    family_best.to_csv(output_dir / "family_best.csv", index=False)
    ensembles.to_csv(output_dir / "ensembles.csv", index=False)
    summary = {
        "protocol": "V19_FORWARD_RECENCY_EB_V1",
        "half_lives_months": HALF_LIVES,
        "best_individual": robust.head(20).to_dict("records"),
        "best_ensemble": ensembles.head(20).to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v20_recency_eb_20260816_01"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
