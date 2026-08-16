"""Standardized local scorecard for candidate-over-incumbent BSS gains.

The official test labels are unavailable, so this module deliberately reports
several historical-forward estimators instead of pretending that one local
number is a leaderboard score.  Public projections are optional scenarios
calibrated from an explicitly supplied historical transfer observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _validate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "gain_vs_v13"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"metrics missing columns: {sorted(missing)}")
    out = frame.loc[:, ["year", "gain_vs_v13"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out["gain_vs_v13"] = pd.to_numeric(
        out["gain_vs_v13"], errors="raise"
    ).astype(float)
    if out.empty or out["year"].duplicated().any():
        raise ValueError("metrics must contain one or more unique years")
    if not np.isfinite(out["gain_vs_v13"]).all():
        raise ValueError("metrics contain non-finite gains")
    return out.sort_values("year").reset_index(drop=True)


def _validate_bootstrap(frame: pd.DataFrame, years: np.ndarray) -> pd.DataFrame:
    required = {"year", "bootstrap_mean", "p05", "prob_positive"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"bootstrap missing columns: {sorted(missing)}")
    out = frame.loc[:, sorted(required)].copy()
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    for column in ("bootstrap_mean", "p05", "prob_positive"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
    if out["year"].duplicated().any():
        raise ValueError("bootstrap years must be unique")
    if set(out["year"]) != set(int(year) for year in years):
        raise ValueError("metrics and bootstrap years differ")
    if not np.isfinite(out[["bootstrap_mean", "p05", "prob_positive"]]).all().all():
        raise ValueError("bootstrap contains non-finite values")
    if not out["prob_positive"].between(0.0, 1.0).all():
        raise ValueError("prob_positive must be in [0, 1]")
    return out.sort_values("year").reset_index(drop=True)


def _decay_weights(years: np.ndarray, half_life: float) -> np.ndarray:
    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    age = int(years.max()) - years
    raw = np.power(0.5, age / half_life)
    return raw / raw.sum()


def build_scorecard(
    name: str,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    incumbent_public: float,
    calibration_local_gain: float,
    calibration_public_gain: float,
    half_life: float = 1.0,
    post_break_year: int = 2023,
    gate_probability: float = 0.95,
) -> dict[str, Any]:
    """Summarize strict-forward evidence and optional Public score scenarios."""
    metric = _validate_metrics(metrics)
    boot = _validate_bootstrap(bootstrap, metric["year"].to_numpy())
    joined = metric.merge(boot, on="year", validate="one_to_one")

    years = joined["year"].to_numpy(dtype=int)
    gains = joined["gain_vs_v13"].to_numpy(dtype=float)
    weights = _decay_weights(years, half_life)
    post = joined["year"] >= int(post_break_year)
    if not post.any():
        raise ValueError("no fold is on or after post_break_year")
    post_years = joined.loc[post, "year"].to_numpy(dtype=int)
    post_weights = _decay_weights(post_years, half_life)
    post_gains = joined.loc[post, "gain_vs_v13"].to_numpy(dtype=float)

    strict_gate = bool(
        (joined["gain_vs_v13"] > 0.0).all()
        and (joined["p05"] > 0.0).all()
        and (joined["prob_positive"] >= gate_probability).all()
    )
    post_break_gate = bool(
        (joined.loc[post, "gain_vs_v13"] > 0.0).all()
        and (joined.loc[post, "p05"] > 0.0).all()
        and (joined.loc[post, "prob_positive"] >= gate_probability).all()
    )

    if calibration_local_gain <= 0.0:
        raise ValueError("calibration_local_gain must be positive")
    transfer = float(calibration_public_gain / calibration_local_gain)
    latest_gain = float(gains[-1])
    recency_gain = float(np.dot(weights, gains))
    post_break_gain = float(np.dot(post_weights, post_gains))
    strict_lower_gain = float(joined["p05"].min())
    post_break_lower_gain = float(joined.loc[post, "p05"].min())

    def project(local_gain: float) -> float:
        return float(incumbent_public + transfer * local_gain)

    return {
        "candidate": name,
        "years": years.tolist(),
        "fold_gain_bss": {str(row.year): float(row.gain_vs_v13) for row in joined.itertuples()},
        "strict_gate": strict_gate,
        "post_break_gate": post_break_gate,
        "local_estimators": {
            "latest_year_gain": latest_gain,
            "half_life_1y_gain": recency_gain,
            "post_break_half_life_1y_gain": post_break_gain,
            "strict_min_bootstrap_p05": strict_lower_gain,
            "post_break_min_bootstrap_p05": post_break_lower_gain,
        },
        "public_scenarios": {
            "incumbent": float(incumbent_public),
            "transfer_factor": transfer,
            "latest_year": project(latest_gain),
            "half_life_1y": project(recency_gain),
            "post_break_half_life_1y": project(post_break_gain),
            "strict_bootstrap_lower": project(strict_lower_gain),
            "post_break_bootstrap_lower": project(post_break_lower_gain),
        },
        "caveat": (
            "Local gains are unclipped BSS-equivalents. Public scenarios use one "
            "historical local-to-Public transfer observation; "
            "they are heuristic scenarios, not confidence intervals or score guarantees."
        ),
    }


def _markdown(cards: list[dict[str, Any]]) -> str:
    lines = [
        "| candidate | strict | post-break | latest unclipped BSS-eq | decay BSS-eq | post-break BSS-eq | "
        "Public latest | Public decay | Public post-break | strict lower |",
        "|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for card in cards:
        local = card["local_estimators"]
        public = card["public_scenarios"]
        lines.append(
            f"| {card['candidate']} | {card['strict_gate']} | {card['post_break_gate']} | "
            f"{local['latest_year_gain']:+.4f} | {local['half_life_1y_gain']:+.4f} | "
            f"{local['post_break_half_life_1y_gain']:+.4f} | {public['latest_year']:.3f} | "
            f"{public['half_life_1y']:.3f} | {public['post_break_half_life_1y']:.3f} | "
            f"{public['strict_bootstrap_lower']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Local columns are unclipped BSS-equivalent gains. Public columns are heuristic scenarios based on one transfer observation; they are not confidence intervals.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("NAME", "METRICS_CSV", "BOOTSTRAP_CSV"),
        required=True,
    )
    parser.add_argument("--incumbent-public", type=float, required=True)
    parser.add_argument("--calibration-local-gain", type=float, required=True)
    parser.add_argument("--calibration-public-gain", type=float, required=True)
    parser.add_argument("--half-life", type=float, default=1.0)
    parser.add_argument("--post-break-year", type=int, default=2023)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cards = []
    for name, metrics_path, bootstrap_path in args.candidate:
        cards.append(
            build_scorecard(
                name,
                pd.read_csv(Path(metrics_path)),
                pd.read_csv(Path(bootstrap_path)),
                incumbent_public=args.incumbent_public,
                calibration_local_gain=args.calibration_local_gain,
                calibration_public_gain=args.calibration_public_gain,
                half_life=args.half_life,
                post_break_year=args.post_break_year,
            )
        )
    if args.format == "json":
        print(json.dumps(cards, ensure_ascii=False, indent=2))
    else:
        print(_markdown(cards))


if __name__ == "__main__":
    main()
