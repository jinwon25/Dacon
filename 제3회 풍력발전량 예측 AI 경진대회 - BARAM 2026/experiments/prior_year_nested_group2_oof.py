"""Build a causal 2023 group-2 OOF surface for multiyear FiCR checks.

The retained Base-v2 OOF starts in 2024, so it cannot tell whether the
group-2 quantile direction is a repeatable mechanism or a one-year accident.
This diagnostic rebuilds the same nested selection protocol on the four 2023
issue seasons.  Every outer season is untouched by alpha, train-row policy,
early-stopping, and iteration selection.  No test prediction or submission is
created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.nested_quantile_base import (
    Selection,
    _fit_fixed_lgb_ensemble,
    _fit_inner_lgb,
    _ordered_issue_seasons,
    _parse_floats,
    _parse_ints,
    choose_selection,
    make_nested_folds,
)
from src.feature_cache import load_or_build_features
from src.metrics import CAPACITY_KWH, evaluate_group
from train import select_feature_columns


ROOT = Path(__file__).resolve().parents[1]
TARGET = "kpx_group_2"
OUTER_SEASONS = ("2023-DJF", "2023-MAM", "2023-JJA", "2023-SON")


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _metric_delta(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float]:
    base = evaluate_group(
        truth[rows], reference[rows], CAPACITY_KWH[TARGET]
    )
    new = evaluate_group(
        truth[rows], candidate[rows], CAPACITY_KWH[TARGET]
    )
    return {
        "score": float(new.score - base.score),
        "one_minus_nmae": float(new.one_minus_nmae - base.one_minus_nmae),
        "ficr": float(new.ficr - base.ficr),
    }


def _period_rows(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    month = index.month.to_numpy()
    return {
        "q1": month <= 3,
        "q2": (month >= 4) & (month <= 6),
        "h2": month >= 7,
        "full": np.ones(len(index), dtype=bool),
    }


def _issue_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    issues: pd.DatetimeIndex,
    seasons: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    issue_values = np.asarray(issues)
    strata = {
        season: np.unique(issue_values[seasons == season])
        for season in dict.fromkeys(seasons)
    }
    positions = {
        (season, issue): np.flatnonzero(
            (seasons == season) & (issue_values == issue)
        )
        for season, values in strata.items()
        for issue in values
    }
    values = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sampled = np.concatenate(
            [
                positions[(season, issue)]
                for season, available in strata.items()
                for issue in rng.choice(
                    available, size=len(available), replace=True
                )
            ]
        )
        values[repetition] = _metric_delta(
            truth,
            reference,
            candidate,
            sampled,
        )["score"]
    return {
        "repetitions": int(repetitions),
        "positive_fraction": float(np.mean(values > 0.0)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _dose_record(
    truth: np.ndarray,
    reference: np.ndarray,
    expert: np.ndarray,
    index: pd.DatetimeIndex,
    weight: float,
) -> dict[str, Any]:
    candidate = np.clip(
        reference + weight * (expert - reference),
        0.0,
        CAPACITY_KWH[TARGET],
    )
    return {
        "period_deltas_vs_l1": {
            name: _metric_delta(truth, reference, candidate, rows)
            for name, rows in _period_rows(index).items()
        },
        "positive_score_months": int(
            sum(
                _metric_delta(
                    truth,
                    reference,
                    candidate,
                    index.month.to_numpy() == month,
                )["score"]
                > 0.0
                for month in range(1, 13)
            )
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    alphas = _parse_floats(args.alphas)
    seeds = _parse_ints(args.seeds)
    variants = tuple(
        item.strip() for item in args.train_variants.split(",") if item.strip()
    )
    if not variants or any(
        variant not in {"all", "eligible_only"} for variant in variants
    ):
        raise ValueError("train variants must be all and/or eligible_only")

    data_dir = _rooted(args.data_dir)
    features = load_or_build_features(
        data_dir, "train", _rooted(args.cache_dir)
    )
    labels = pd.read_csv(
        data_dir / "train" / "train_labels.csv",
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm").reindex(features.index)
    issues = load_issue_times(
        data_dir / "train" / "gfs_train.csv", features.index
    )
    columns = select_feature_columns(features, TARGET, args.feature_set)
    x = features[columns]
    y = labels[TARGET].to_numpy(dtype=float)
    available = np.isfinite(y)
    folds = make_nested_folds(
        x.index,
        issues,
        available,
        purge_hours=args.purge_hours,
        outer_seasons=OUTER_SEASONS,
        minimum_train_rows=args.minimum_train_rows,
        minimum_valid_rows=args.minimum_valid_rows,
    )
    if tuple(fold.name for fold in folds) != OUTER_SEASONS:
        raise ValueError(
            "not all requested prior-year outer seasons passed the row guards"
        )

    l1_oof = np.full(len(x), np.nan, dtype=float)
    candidate_oof = np.full(len(x), np.nan, dtype=float)
    reports: list[dict[str, Any]] = []
    capacity = CAPACITY_KWH[TARGET]
    for fold_i, fold in enumerate(folds, start=1):
        l1_records: list[Selection] = []
        quantile_records: list[Selection] = []
        for variant in variants:
            l1_record, _ = _fit_inner_lgb(
                x,
                y,
                fold.inner_train,
                fold.inner_valid,
                capacity,
                objective="l1",
                alpha=None,
                variant=variant,
                curtailment=None,
                seed=11_000 + fold_i,
                maximum_iterations=args.maximum_iterations,
                early_stopping_rounds=args.early_stopping_rounds,
                n_jobs=args.n_jobs,
            )
            l1_records.append(l1_record)
            for alpha_i, alpha in enumerate(alphas, start=1):
                quantile_record, _ = _fit_inner_lgb(
                    x,
                    y,
                    fold.inner_train,
                    fold.inner_valid,
                    capacity,
                    objective="quantile",
                    alpha=alpha,
                    variant=variant,
                    curtailment=None,
                    seed=21_000 + 100 * fold_i + alpha_i,
                    maximum_iterations=args.maximum_iterations,
                    early_stopping_rounds=args.early_stopping_rounds,
                    n_jobs=args.n_jobs,
                )
                quantile_records.append(quantile_record)
        selected_l1 = choose_selection(l1_records)
        selected_quantile = choose_selection(quantile_records)
        selected_l1 = Selection(
            **{
                **asdict(selected_l1),
                "iterations": max(
                    100, int(round(selected_l1.iterations * 1.10))
                ),
            }
        )
        selected_quantile = Selection(
            **{
                **asdict(selected_quantile),
                "iterations": max(
                    100, int(round(selected_quantile.iterations * 1.10))
                ),
            }
        )
        l1_prediction = _fit_fixed_lgb_ensemble(
            x,
            y,
            fold.train,
            fold.valid,
            capacity,
            selection=selected_l1,
            seeds=seeds,
            curtailment=None,
            n_jobs=args.n_jobs,
        )
        quantile_prediction = _fit_fixed_lgb_ensemble(
            x,
            y,
            fold.train,
            fold.valid,
            capacity,
            selection=selected_quantile,
            seeds=seeds,
            curtailment=None,
            n_jobs=args.n_jobs,
        )
        l1_oof[fold.valid] = l1_prediction
        candidate_oof[fold.valid] = quantile_prediction
        fold_truth = y[fold.valid]
        base_metric = evaluate_group(fold_truth, l1_prediction, capacity)
        new_metric = evaluate_group(fold_truth, quantile_prediction, capacity)
        reports.append(
            {
                "outer_season": fold.name,
                "inner_selection_season": fold.inner_name,
                "train_rows": int(fold.train.sum()),
                "valid_rows": int(fold.valid.sum()),
                "selected_l1": asdict(selected_l1),
                "selected_quantile": asdict(selected_quantile),
                "outer_l1": base_metric.to_dict(),
                "outer_quantile": new_metric.to_dict(),
                "outer_delta": {
                    "score": float(new_metric.score - base_metric.score),
                    "one_minus_nmae": float(
                        new_metric.one_minus_nmae - base_metric.one_minus_nmae
                    ),
                    "ficr": float(new_metric.ficr - base_metric.ficr),
                },
            }
        )
        print(
            f"[{fold.name}] alpha={selected_quantile.alpha} "
            f"variant={selected_quantile.train_variant} "
            f"delta={new_metric.score - base_metric.score:+.6f}",
            flush=True,
        )

    evaluated = available & np.isfinite(l1_oof) & np.isfinite(candidate_oof)
    index = x.index[evaluated]
    truth = y[evaluated]
    reference = l1_oof[evaluated]
    expert = candidate_oof[evaluated]
    aligned_issues = pd.DatetimeIndex(np.asarray(issues)[evaluated])
    aligned_seasons, _ = _ordered_issue_seasons(index, aligned_issues)
    full_rows = np.ones(len(index), dtype=bool)
    overall_delta = _metric_delta(
        truth, reference, expert, full_rows
    )
    seasonal_delta = {
        season: _metric_delta(
            truth,
            reference,
            expert,
            aligned_seasons == season,
        )
        for season in OUTER_SEASONS
    }
    bootstrap = _issue_bootstrap(
        truth,
        reference,
        expert,
        aligned_issues,
        aligned_seasons,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    weights = tuple(
        float(value)
        for value in np.arange(
            args.minimum_dose,
            args.maximum_dose + 0.5 * args.dose_step,
            args.dose_step,
        )
    )
    dose_response = {
        f"{weight:.4f}": _dose_record(
            truth, reference, expert, index, weight
        )
        for weight in weights
    }
    stable_weights = [
        weight
        for weight in weights
        if all(
            dose_response[f"{weight:.4f}"]["period_deltas_vs_l1"][period][
                "score"
            ]
            >= 0.0
            for period in ("q1", "q2", "h2", "full")
        )
        and dose_response[f"{weight:.4f}"]["positive_score_months"] >= 8
    ]
    best_stable_weight = (
        max(
            stable_weights,
            key=lambda weight: dose_response[f"{weight:.4f}"][
                "period_deltas_vs_l1"
            ]["full"]["score"],
        )
        if stable_weights
        else None
    )

    output_cache = _rooted(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_cache,
        index_ns=index.astype("int64").to_numpy(),
        issue_ns=aligned_issues.astype("int64").to_numpy(),
        truth=truth.astype("float32"),
        l1=reference.astype("float32"),
        quantile=expert.astype("float32"),
    )
    report = {
        "schema_version": "prior_year_nested_group2_oof.v1",
        "contract": {
            "target": TARGET,
            "outer_seasons": OUTER_SEASONS,
            "outer_fold_used_for_selection": False,
            "public_score_used": False,
            "test_prediction_created": False,
            "submission_created": False,
        },
        "configuration": {
            "feature_set": args.feature_set,
            "alphas": alphas,
            "seeds": seeds,
            "train_variants": variants,
            "purge_hours": args.purge_hours,
            "maximum_iterations": args.maximum_iterations,
            "early_stopping_rounds": args.early_stopping_rounds,
            "diagnostic_compute_profile": (
                "reduced alpha grid and fixed seed; temporal nesting unchanged"
            ),
        },
        "folds": reports,
        "overall_quantile_minus_l1": overall_delta,
        "seasonal_quantile_minus_l1": seasonal_delta,
        "issue_block_bootstrap_score": bootstrap,
        "dose_response_vs_l1": dose_response,
        "stable_dose_weights": [float(weight) for weight in stable_weights],
        "best_stable_dose_weight": (
            float(best_stable_weight)
            if best_stable_weight is not None
            else None
        ),
        "coverage": {
            "rows": int(len(index)),
            "start": index.min().isoformat(),
            "end": index.max().isoformat(),
        },
        "output": {
            "path": output_cache.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(output_cache.read_bytes()).hexdigest(),
        },
    }
    output_report = _rooted(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--cache-dir", default="artifacts_final/feature_cache"
    )
    parser.add_argument(
        "--feature-set",
        default="base",
        choices=("base", "own_idw", "own_idw_nohub", "full"),
    )
    parser.add_argument("--alphas", default="0.65,0.70")
    parser.add_argument(
        "--seeds", default="42"
    )
    parser.add_argument("--train-variants", default="all,eligible_only")
    parser.add_argument("--purge-hours", type=int, default=24)
    parser.add_argument("--minimum-train-rows", type=int, default=2_000)
    parser.add_argument("--minimum-valid-rows", type=int, default=500)
    parser.add_argument("--maximum-iterations", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--minimum-dose", type=float, default=0.05)
    parser.add_argument("--maximum-dose", type=float, default=0.35)
    parser.add_argument("--dose-step", type=float, default=0.0125)
    parser.add_argument(
        "--output-cache",
        default="artifacts_final/lineage/prior_year_nested_group2_oof.npz",
    )
    parser.add_argument(
        "--output-report",
        default=(
            "artifacts_final/diagnostics/"
            "prior_year_nested_group2_oof_20260802.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "overall_quantile_minus_l1": report[
                    "overall_quantile_minus_l1"
                ],
                "seasonal_quantile_minus_l1": report[
                    "seasonal_quantile_minus_l1"
                ],
                "bootstrap": report["issue_block_bootstrap_score"],
                "best_stable_dose_weight": report[
                    "best_stable_dose_weight"
                ],
                "coverage": report["coverage"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
