"""Leakage-safe game-type regime offset on top of the frozen incumbent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import apply_logit_offset
from src.data import TARGET_COL, read_main
from src.followup import _load_all_caches
from src.metrics import brier_score, cluster_bootstrap_delta, probability_logit


CANDIDATE_NAME = "game_type_regime_residual_v1"
MIN_RATE_RESIDUAL_GAP = 0.05
MIN_GROUP_ROWS = 5_000


def fit_game_type_regime_offsets(
    history: pd.DataFrame,
    forecast_year: int,
    *,
    min_rate_residual_gap: float = MIN_RATE_RESIDUAL_GAP,
    min_group_rows: int = MIN_GROUP_ROWS,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Detect a large sign-changing group residual using target history only.

    A residual is a game-type annual rate minus the same-season global rate.
    The fixed five-percentage-point gap prevents small, statistically significant
    but operationally negligible changes from triggering.  No validation/test
    prediction mean or target is inspected.
    """
    if history.empty or int(history["season"].max()) >= forecast_year:
        raise ValueError("history must be non-empty and strictly before forecast_year")
    annual = (
        history.groupby(["season", "game_type"], observed=True)[TARGET_COL]
        .agg(["size", "mean"])
        .rename(columns={"size": "n_rows", "mean": "group_rate"})
        .reset_index()
    )
    global_rate = history.groupby("season", observed=True)[TARGET_COL].mean()
    annual["global_rate"] = annual["season"].map(global_rate)
    annual["rate_residual"] = annual["group_rate"] - annual["global_rate"]
    annual["logit_residual"] = probability_logit(
        annual["group_rate"].to_numpy(dtype=np.float64)
    ) - probability_logit(annual["global_rate"].to_numpy(dtype=np.float64))

    offsets: dict[str, float] = {}
    diagnostics: list[dict[str, object]] = []
    for game_type, group in annual.groupby("game_type", observed=True):
        group = group.sort_values("season").reset_index(drop=True)
        offset = 0.0
        window = 0
        earlier_rate_residual = np.nan
        recent_rate_residual = np.nan
        detected = False
        reason = "insufficient_history_or_no_large_sign_change"
        if len(group) >= 4 and int(group.iloc[-1]["n_rows"]) >= min_group_rows:
            latest = group.iloc[-1]
            earlier = group.iloc[:-1]
            latest_residual = float(latest["rate_residual"])
            earlier_mean = float(earlier["rate_residual"].mean())
            gap = abs(latest_residual - earlier_mean)
            if (
                latest_residual * earlier_mean < 0.0
                and gap >= min_rate_residual_gap
            ):
                detected = True
                window = 1
                earlier_rate_residual = earlier_mean
                recent_rate_residual = latest_residual
                offset = float(latest["logit_residual"])
                reason = "latest_year_large_sign_change"

        if len(group) >= 5:
            recent_two = group.iloc[-2:]
            earlier = group.iloc[:-2]
            recent_mean = float(recent_two["rate_residual"].mean())
            earlier_mean = float(earlier["rate_residual"].mean())
            same_recent_sign = bool(
                float(recent_two.iloc[0]["rate_residual"])
                * float(recent_two.iloc[1]["rate_residual"])
                > 0.0
            )
            gap = abs(recent_mean - earlier_mean)
            if (
                same_recent_sign
                and recent_mean * earlier_mean < 0.0
                and gap >= min_rate_residual_gap
                and int(recent_two["n_rows"].min()) >= min_group_rows
            ):
                detected = True
                window = 2
                earlier_rate_residual = earlier_mean
                recent_rate_residual = recent_mean
                offset = float(recent_two["logit_residual"].mean())
                reason = "two_year_persistent_large_sign_change"

        offsets[str(game_type)] = offset
        diagnostics.append(
            {
                "forecast_year": forecast_year,
                "game_type": str(game_type),
                "detected": detected,
                "recent_window": window,
                "earlier_rate_residual": earlier_rate_residual,
                "recent_rate_residual": recent_rate_residual,
                "logit_offset": offset,
                "threshold": min_rate_residual_gap,
                "reason": reason,
                "history_seasons": ";".join(group["season"].astype(str)),
            }
        )
    return offsets, pd.DataFrame(diagnostics)


def apply_game_type_offsets(
    probability: np.ndarray,
    game_type: pd.Series,
    offsets: dict[str, float],
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    output = probability.copy()
    types = game_type.astype("string").fillna("__MISSING__")
    for value, offset in offsets.items():
        mask = types.eq(value).to_numpy()
        if mask.any() and offset != 0.0:
            output[mask] = apply_logit_offset(output[mask], offset)
    if not np.isfinite(output).all():
        raise ValueError("game-type offset produced non-finite predictions")
    return np.clip(output, 0.0, 1.0)


def _bootstrap_rows(
    frame: pd.DataFrame,
    target: np.ndarray,
    candidate: np.ndarray,
    incumbent: np.ndarray,
    scope: str,
    n_resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    cluster_specs = {
        "pitcher_season": (
            frame["pitcher_id"].astype("string")
            + "-"
            + frame["season"].astype("string")
        ),
        "pitcher": frame["pitcher_id"].astype("string"),
    }
    rows = []
    for cluster_type, clusters in cluster_specs.items():
        rows.append(
            {
                "scope": scope,
                "candidate": CANDIDATE_NAME,
                "reference": "incumbent",
                "cluster_type": cluster_type,
                **cluster_bootstrap_delta(
                    target,
                    candidate,
                    incumbent,
                    clusters,
                    n_resamples=n_resamples,
                    seed=seed,
                ),
            }
        )
    return rows


def run(project_dir: Path, n_resamples: int = 10_000) -> pd.DataFrame:
    config = json.loads(
        (project_dir / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    train = read_main(project_dir / "data" / "train.csv")
    folds, caches = _load_all_caches(project_dir, config, train)
    years = [int(value) for value in config["outer_validation_seasons"]]
    rows: list[dict[str, object]] = []
    offset_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    predictions: dict[int, np.ndarray] = {}

    for year in years:
        fold = folds[year]
        history = train.iloc[fold.train_idx]
        valid = train.iloc[fold.valid_idx]
        offsets, diagnostics = fit_game_type_regime_offsets(history, year)
        offset_rows.append(diagnostics)
        incumbent = caches[year]["incumbent"]
        target = caches[year]["target"]
        candidate = apply_game_type_offsets(incumbent, valid["game_type"], offsets)
        predictions[year] = candidate
        delta = brier_score(target, candidate) - brier_score(target, incumbent)
        rows.append(
            {
                "candidate": CANDIDATE_NAME,
                "outer_validation_season": year,
                "brier": brier_score(target, candidate),
                "incumbent_brier": brier_score(target, incumbent),
                "delta_brier": delta,
                "prediction_mean": float(candidate.mean()),
                "target_rate": float(target.mean()),
                "offsets": json.dumps(offsets, sort_keys=True),
            }
        )
        bootstrap_rows.extend(
            _bootstrap_rows(
                valid,
                target,
                candidate,
                incumbent,
                str(year),
                n_resamples,
                int(config["seed"]) + 20_000 + year,
            )
        )
        for month, index in valid.groupby("game_month", observed=True).groups.items():
            positions = valid.index.get_indexer(index)
            monthly_rows.append(
                {
                    "season": year,
                    "game_month": int(month),
                    "n_rows": len(positions),
                    "delta_brier": brier_score(target[positions], candidate[positions])
                    - brier_score(target[positions], incumbent[positions]),
                }
            )

    all_valid = pd.concat(
        [train.iloc[folds[year].valid_idx] for year in years], ignore_index=True
    )
    target_all = np.concatenate([caches[year]["target"] for year in years])
    incumbent_all = np.concatenate([caches[year]["incumbent"] for year in years])
    candidate_all = np.concatenate([predictions[year] for year in years])
    bootstrap_rows.extend(
        _bootstrap_rows(
            all_valid,
            target_all,
            candidate_all,
            incumbent_all,
            "2021-2024",
            n_resamples,
            int(config["seed"]) + 30_000,
        )
    )

    result = pd.DataFrame(rows)
    deltas = result.set_index("outer_validation_season")["delta_brier"]
    weights = np.array(
        [float(config["recency_weights"][str(year)]) for year in years]
    )
    recency_delta = float(np.average([deltas[year] for year in years], weights=weights))
    gate = config["submission_gate"]
    combined_ps = next(
        row
        for row in bootstrap_rows
        if row["scope"] == "2021-2024" and row["cluster_type"] == "pitcher_season"
    )
    checks = {
        "recency": recency_delta
        <= -float(gate["minimum_recency_weighted_improvement"]),
        "latest": float(deltas[2024])
        <= -float(gate["minimum_2024_improvement"]),
        "worst": float(deltas.max())
        <= float(gate["maximum_single_fold_worsening"]),
        "bootstrap": float(combined_ps["improvement_probability"])
        >= float(gate["minimum_cluster_improvement_probability"]),
    }
    result["recency_weighted_delta"] = recency_delta
    result["worst_fold_delta"] = float(deltas.max())
    result["combined_pitcher_season_improvement_probability"] = float(
        combined_ps["improvement_probability"]
    )
    result["passes_statistical_gate"] = all(checks.values())

    reports = project_dir / "reports"
    result.to_csv(reports / "domain_drift_results.csv", index=False)
    pd.concat(offset_rows, ignore_index=True).to_csv(
        reports / "domain_drift_offsets.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        reports / "domain_drift_bootstrap.csv", index=False
    )
    pd.DataFrame(monthly_rows).to_csv(
        reports / "domain_drift_monthly.csv", index=False
    )
    print(result.to_string(index=False))
    print("gate", checks)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    run(args.project_dir.resolve(), args.resamples)


if __name__ == "__main__":
    main()
