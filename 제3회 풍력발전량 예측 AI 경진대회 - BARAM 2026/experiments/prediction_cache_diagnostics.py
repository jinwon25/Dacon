from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, evaluate_group


def _metric(y_true: np.ndarray, y_pred: np.ndarray, capacity: float) -> dict[str, float | int]:
    return evaluate_group(y_true, np.clip(y_pred, 0, capacity), capacity).to_dict()


def analyze_cache(
    cache_path: Path,
    baseline_name: str | None = None,
    min_month_samples: int = 24,
) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as cache:
        names = [str(value) for value in cache["candidate_names"].tolist()]
        if not names:
            raise ValueError("Prediction cache contains no candidates.")
        baseline_name = baseline_name or names[0]
        if baseline_name not in names:
            raise ValueError(f"Unknown baseline {baseline_name!r}; choose one of {names}.")
        baseline_i = names.index(baseline_name)

        report: dict[str, object] = {
            "cache": str(cache_path),
            "candidate_names": names,
            "baseline": baseline_name,
            "min_month_samples": min_month_samples,
            "targets": {},
        }
        for target, capacity in CAPACITY_KWH.items():
            prefix = f"{target}__"
            timestamps = pd.to_datetime(cache[prefix + "valid_index_ns"], unit="ns")
            truth = cache[prefix + "valid_truth"].astype(float)
            matrix = cache[prefix + "valid_matrix"].astype(float)
            selected_weights = cache[prefix + "selected_weights"].astype(float)
            if matrix.shape != (len(truth), len(names)):
                raise ValueError(f"Invalid validation matrix shape for {target}: {matrix.shape}")
            if len(selected_weights) != len(names):
                raise ValueError(f"Invalid selected weight length for {target}.")

            predictions = {
                name: matrix[:, i]
                for i, name in enumerate(names)
            }
            predictions["selected_blend"] = matrix @ selected_weights
            full_metrics = {
                name: _metric(truth, pred, capacity)
                for name, pred in predictions.items()
            }

            months = pd.PeriodIndex(timestamps, freq="M")
            monthly: dict[str, dict[str, dict[str, float | int]]] = {}
            for month in months.unique().sort_values():
                mask = np.asarray(months == month)
                monthly[str(month)] = {
                    name: _metric(truth[mask], pred[mask], capacity)
                    for name, pred in predictions.items()
                }

            stable_months = {
                month: month_metrics
                for month, month_metrics in monthly.items()
                if month_metrics[baseline_name]["n_samples"] >= min_month_samples
            }
            if not stable_months:
                raise ValueError(
                    f"No months for {target} contain at least {min_month_samples} eligible samples."
                )
            baseline_month_scores = np.asarray(
                [month_metrics[baseline_name]["score"] for month_metrics in stable_months.values()],
                dtype=float,
            )
            stability = {}
            for name in predictions:
                candidate_month_scores = np.asarray(
                    [month_metrics[name]["score"] for month_metrics in stable_months.values()],
                    dtype=float,
                )
                deltas = candidate_month_scores - baseline_month_scores
                stability[name] = {
                    "months_improved": int(np.sum(deltas > 0)),
                    "months_tied": int(np.sum(np.isclose(deltas, 0))),
                    "months_total": int(len(deltas)),
                    "months_excluded": int(len(monthly) - len(stable_months)),
                    "mean_monthly_score_delta": float(deltas.mean()),
                    "median_monthly_score_delta": float(np.median(deltas)),
                    "worst_monthly_score_delta": float(deltas.min()),
                    "best_monthly_score_delta": float(deltas.max()),
                }

            report["targets"][target] = {
                "selected_weights": {
                    name: float(weight)
                    for name, weight in zip(names, selected_weights)
                    if weight > 1e-8
                },
                "full_metrics": full_metrics,
                "stability_vs_baseline": stability,
                "monthly_metrics": monthly,
            }
    return report


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Prediction Cache Diagnostics",
        "",
        f"Baseline: `{report['baseline']}`",
        "",
        "| target | candidate | score | 1-NMAE | FICR | improved months | mean monthly delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for target, target_report in report["targets"].items():
        metrics = target_report["full_metrics"]
        stability = target_report["stability_vs_baseline"]
        for name, values in metrics.items():
            stable = stability[name]
            lines.append(
                f"| {target} | {name} | {values['score']:.6f} | "
                f"{values['one_minus_nmae']:.6f} | {values['ficr']:.6f} | "
                f"{stable['months_improved']}/{stable['months_total']} | "
                f"{stable['mean_monthly_score_delta']:+.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--baseline", default="")
    parser.add_argument("--min-month-samples", type=int, default=24)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", default="")
    args = parser.parse_args()

    report = analyze_cache(
        Path(args.cache),
        args.baseline or None,
        min_month_samples=args.min_month_samples,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_markdown:
        output_markdown = Path(args.output_markdown)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))


if __name__ == "__main__":
    main()
