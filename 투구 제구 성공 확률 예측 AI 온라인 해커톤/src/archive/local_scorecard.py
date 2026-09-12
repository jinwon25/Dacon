"""Standardized local scorecard for candidate-over-incumbent BSS gains.

The official test labels are unavailable, so this module deliberately reports
several historical-forward estimators instead of pretending that one local
number is a leaderboard score.  Public projections are optional scenarios
and remain suppressed until at least three historical transfer observations
are explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _validate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if "year" not in frame:
        raise ValueError("metrics missing columns: ['year']")
    gain_columns = [
        column
        for column in ("gain_vs_incumbent", "gain_vs_v13")
        if column in frame
    ]
    if not gain_columns:
        raise ValueError(
            "metrics must contain gain_vs_incumbent "
            "(legacy gain_vs_v13 is accepted for historical reports)"
        )
    if len(gain_columns) > 1:
        raise ValueError("metrics must contain only one gain column")
    out = frame.loc[:, ["year", gain_columns[0]]].copy().rename(
        columns={gain_columns[0]: "gain_vs_incumbent"}
    )
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out["gain_vs_incumbent"] = pd.to_numeric(
        out["gain_vs_incumbent"], errors="raise"
    ).astype(float)
    if out.empty or out["year"].duplicated().any():
        raise ValueError("metrics must contain one or more unique years")
    if not np.isfinite(out["gain_vs_incumbent"]).all():
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
    calibration_local_gain: float | None = None,
    calibration_public_gain: float | None = None,
    transfer_history: pd.DataFrame | None = None,
    half_life: float = 1.0,
    post_break_year: int = 2023,
    gate_probability: float = 0.95,
) -> dict[str, Any]:
    """Summarize strict-forward evidence and optional Public score scenarios."""
    metric = _validate_metrics(metrics)
    boot = _validate_bootstrap(bootstrap, metric["year"].to_numpy())
    joined = metric.merge(boot, on="year", validate="one_to_one")

    years = joined["year"].to_numpy(dtype=int)
    gains = joined["gain_vs_incumbent"].to_numpy(dtype=float)
    weights = _decay_weights(years, half_life)
    post = joined["year"] >= int(post_break_year)
    if not post.any():
        raise ValueError("no fold is on or after post_break_year")
    post_years = joined.loc[post, "year"].to_numpy(dtype=int)
    post_weights = _decay_weights(post_years, half_life)
    post_gains = joined.loc[post, "gain_vs_incumbent"].to_numpy(dtype=float)

    strict_gate = bool(
        (joined["gain_vs_incumbent"] > 0.0).all()
        and (joined["p05"] > 0.0).all()
        and (joined["prob_positive"] >= gate_probability).all()
    )
    post_break_gate = bool(
        (joined.loc[post, "gain_vs_incumbent"] > 0.0).all()
        and (joined.loc[post, "p05"] > 0.0).all()
        and (joined.loc[post, "prob_positive"] >= gate_probability).all()
    )

    latest_gain = float(gains[-1])
    recency_gain = float(np.dot(weights, gains))
    post_break_gain = float(np.dot(post_weights, post_gains))
    strict_lower_gain = float(joined["p05"].min())
    post_break_lower_gain = float(joined.loc[post, "p05"].min())

    observations: list[tuple[float, float]] = []
    if transfer_history is not None:
        required = {"local_gain", "public_gain"}
        missing = required.difference(transfer_history.columns)
        if missing:
            raise ValueError(f"transfer_history missing columns: {sorted(missing)}")
        for row in transfer_history.loc[:, ["local_gain", "public_gain"]].itertuples(
            index=False
        ):
            observations.append((float(row.local_gain), float(row.public_gain)))
    if calibration_local_gain is not None or calibration_public_gain is not None:
        if calibration_local_gain is None or calibration_public_gain is None:
            raise ValueError("both legacy calibration gains must be supplied together")
        observations.append((float(calibration_local_gain), float(calibration_public_gain)))
    if any(not np.isfinite(item).all() for item in map(np.asarray, observations)):
        raise ValueError("transfer observations must be finite")
    if any(local <= 0.0 for local, _ in observations):
        raise ValueError("transfer local gains must be positive")

    if len(observations) >= 3:
        ratios = np.asarray([public / local for local, public in observations])
        transfer = float(np.median(ratios))

        def project(local_gain: float) -> float:
            return float(incumbent_public + transfer * local_gain)

        public_scenarios = {
            "status": "heuristic_scenario_only",
            "incumbent": float(incumbent_public),
            "n_transfer_observations": int(len(observations)),
            "median_transfer_factor": transfer,
            "transfer_factor_min": float(ratios.min()),
            "transfer_factor_max": float(ratios.max()),
            "latest_year": project(latest_gain),
            "half_life_1y": project(recency_gain),
            "post_break_half_life_1y": project(post_break_gain),
            "strict_bootstrap_lower": project(strict_lower_gain),
            "post_break_bootstrap_lower": project(post_break_lower_gain),
        }
    else:
        public_scenarios = {
            "status": "insufficient_clean_transfer_history",
            "incumbent": float(incumbent_public),
            "n_transfer_observations": int(len(observations)),
            "minimum_required": 3,
        }

    return {
        "candidate": name,
        "years": years.tolist(),
        "fold_gain_vs_incumbent_bss": {
            str(row.year): float(row.gain_vs_incumbent) for row in joined.itertuples()
        },
        "strict_gate": strict_gate,
        "post_break_gate": post_break_gate,
        "local_estimators": {
            "latest_year_gain": latest_gain,
            "half_life_1y_gain": recency_gain,
            "post_break_half_life_1y_gain": post_break_gain,
            "strict_min_bootstrap_p05": strict_lower_gain,
            "post_break_min_bootstrap_p05": post_break_lower_gain,
        },
        "public_scenarios": public_scenarios,
        "caveat": (
            "Local gains are paired unclipped BSS-equivalents. Public projection is "
            "suppressed until at least three clean local-to-Public observations exist; "
            "even then it is a heuristic scenario, never a promotion gate."
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
        public_available = public["status"] == "heuristic_scenario_only"
        lines.append(
            f"| {card['candidate']} | {card['strict_gate']} | {card['post_break_gate']} | "
            f"{local['latest_year_gain']:+.4f} | {local['half_life_1y_gain']:+.4f} | "
            f"{local['post_break_half_life_1y_gain']:+.4f} | "
            f"{public['latest_year']:.3f} | {public['half_life_1y']:.3f} | "
            f"{public['post_break_half_life_1y']:.3f} | "
            f"{public['strict_bootstrap_lower']:.3f} |"
            if public_available
            else f"| {card['candidate']} | {card['strict_gate']} | {card['post_break_gate']} | "
            f"{local['latest_year_gain']:+.4f} | {local['half_life_1y_gain']:+.4f} | "
            f"{local['post_break_half_life_1y_gain']:+.4f} | n/a | n/a | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "Local columns are unclipped BSS-equivalent gains. Public columns are suppressed with fewer than three clean transfer observations; when shown, they are heuristic scenarios rather than confidence intervals or promotion gates.",
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
    parser.add_argument("--calibration-local-gain", type=float)
    parser.add_argument("--calibration-public-gain", type=float)
    parser.add_argument("--transfer-history", type=Path)
    parser.add_argument("--half-life", type=float, default=1.0)
    parser.add_argument("--post-break-year", type=int, default=2023)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cards = []
    transfer_history = (
        pd.read_csv(args.transfer_history) if args.transfer_history is not None else None
    )
    for name, metrics_path, bootstrap_path in args.candidate:
        cards.append(
            build_scorecard(
                name,
                pd.read_csv(Path(metrics_path)),
                pd.read_csv(Path(bootstrap_path)),
                incumbent_public=args.incumbent_public,
                calibration_local_gain=args.calibration_local_gain,
                calibration_public_gain=args.calibration_public_gain,
                transfer_history=transfer_history,
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
