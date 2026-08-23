"""Sparse linear probability challenger above the frozen v27 parent.

The incumbent family is dominated by trees, empirical-Bayes lookups, and a
small neural ensemble.  This experiment deliberately builds a different error
surface: a regularised logistic model over sparse baseball interactions plus
row-local numeric state.  It uses exactly the two seasons preceding each
forecast origin.

The same route and blend weight must improve the older 2022 origin and late
2023.  That recipe is frozen before full and late 2024 are opened.  All
current-season state is reconstructed from the current row's official ASOF
counters and labelled seasons strictly before that row's season; no audit-row
aggregate is used.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder

from src.recent_shared_exact_asof import (
    _current_season_state,
    _make_season_bank,
    _row_state,
)
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata, grid_rows
from src.v36_two_origin_consensus import select_consensus


TARGET = "control_success"
SIGNAL = "sparse_logit_v1"
RECENT_SEASON_WEIGHT = 1.0
OLDER_SEASON_WEIGHT = 0.60

DROP_NUMERIC = {
    "row_id",
    "season",
    TARGET,
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
}


def _safe_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__").astype(str)


def interaction_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Create fixed, inference-safe sparse categorical interactions."""

    pitcher = _safe_string(rows["pitcher_id"])
    batter = _safe_string(rows["batter_id"])
    pitcher_team = _safe_string(rows["pitcher_team_id"])
    batter_team = _safe_string(rows["batter_team_id"])
    pitcher_hand = _safe_string(rows["pitcher_hand"])
    batter_hand = _safe_string(rows["batter_hand"])
    game_type = _safe_string(rows["game_type"])
    domain = _safe_string(rows["domain3"])
    count = (
        _safe_string(rows["balls_before"])
        + "-"
        + _safe_string(rows["strikes_before"])
    )
    base = _safe_string(rows["base_state"])
    inning = pd.cut(
        pd.to_numeric(rows["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, np.inf),
        labels=("early", "middle", "late"),
    ).astype("string").fillna("__MISSING__").astype(str)
    score = pd.cut(
        pd.to_numeric(rows["score_diff_pitcher_team"], errors="coerce"),
        bins=(-np.inf, -3, -1, 0, 1, 3, np.inf),
        labels=("behind4", "behind2", "behind1", "ahead1", "ahead3", "ahead4"),
    ).astype("string").fillna("__MISSING__").astype(str)
    leverage = pd.cut(
        pd.to_numeric(rows["li"], errors="coerce"),
        bins=(-np.inf, 0.75, 1.5, 3.0, np.inf),
        labels=("low", "medium", "high", "very_high"),
    ).astype("string").fillna("__MISSING__").astype(str)

    return pd.DataFrame(
        {
            "pitcher": pitcher,
            "batter": batter,
            "pitcher_team": pitcher_team,
            "batter_team": batter_team,
            "game_type": game_type,
            "domain3": domain,
            "count": count,
            "base_state": base,
            "hand_matchup": pitcher_hand + "-" + batter_hand,
            "inning_bucket": inning,
            "score_bucket": score,
            "leverage_bucket": leverage,
            "pitcher_game_type": pitcher + "|" + game_type,
            "batter_game_type": batter + "|" + game_type,
            "pitcher_opponent_hand": pitcher + "|" + batter_hand,
            "batter_pitcher_hand": batter + "|" + pitcher_hand,
            "pitcher_count": pitcher + "|" + count,
            "batter_count": batter + "|" + count,
            "pitcher_domain_count": pitcher + "|" + domain + "|" + count,
            "team_matchup": pitcher_team + "|" + batter_team,
            "domain_count": domain + "|" + count,
            "count_base": count + "|" + base,
            "inning_base": inning + "|" + base,
            "domain_leverage": domain + "|" + leverage,
            "domain_score": domain + "|" + score,
        },
        index=np.arange(len(rows)),
    )


def row_local_numeric(train: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Build official and exact-ASOF numeric features for one or more seasons."""

    numeric_columns = [
        column
        for column in rows.columns
        if column not in DROP_NUMERIC
        and not pd.api.types.is_object_dtype(rows[column])
        and not isinstance(rows[column].dtype, pd.CategoricalDtype)
    ]
    output = rows[numeric_columns].apply(pd.to_numeric, errors="coerce").reset_index(
        drop=True
    )
    output = pd.concat([output, _row_state(rows)], axis=1)
    states = []
    for year, positions in rows.groupby("season", sort=False).indices.items():
        local = rows.iloc[np.asarray(positions)].reset_index(drop=True)
        state = _current_season_state(
            local, _make_season_bank(train, int(year))
        )
        state.index = np.asarray(positions)
        states.append(state)
    season_state = pd.concat(states).sort_index(kind="stable").reset_index(drop=True)
    if len(season_state) != len(rows):
        raise ValueError("current-season state row mismatch")
    output = pd.concat([output, season_state], axis=1)
    if output.columns.duplicated().any():
        duplicates = output.columns[output.columns.duplicated()].tolist()
        raise ValueError(f"duplicate numeric features: {duplicates}")
    return output.astype(np.float32)


def matrix_pair(
    fit_numeric: pd.DataFrame,
    audit_numeric: pd.DataFrame,
    fit_category: pd.DataFrame,
    audit_category: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, int]]:
    """Fit all sparse preprocessing on the source rows only."""

    if list(fit_numeric.columns) != list(audit_numeric.columns):
        raise ValueError("numeric source/audit schema mismatch")
    if list(fit_category.columns) != list(audit_category.columns):
        raise ValueError("categorical source/audit schema mismatch")
    median = fit_numeric.median(axis=0).fillna(0.0)
    fit_value = fit_numeric.fillna(median)
    audit_value = audit_numeric.fillna(median)
    mean = fit_value.mean(axis=0)
    scale = fit_value.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    fit_scaled = np.clip(
        ((fit_value - mean) / scale).to_numpy(np.float32), -8.0, 8.0
    )
    audit_scaled = np.clip(
        ((audit_value - mean) / scale).to_numpy(np.float32), -8.0, 8.0
    )
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=25,
        dtype=np.float32,
    )
    fit_sparse = encoder.fit_transform(fit_category)
    audit_sparse = encoder.transform(audit_category)
    fit_matrix = sparse.hstack(
        [sparse.csr_matrix(fit_scaled), fit_sparse], format="csr", dtype=np.float32
    )
    audit_matrix = sparse.hstack(
        [sparse.csr_matrix(audit_scaled), audit_sparse],
        format="csr",
        dtype=np.float32,
    )
    return fit_matrix, audit_matrix, {
        "numeric": int(fit_scaled.shape[1]),
        "categorical_encoded": int(fit_sparse.shape[1]),
        "total": int(fit_matrix.shape[1]),
    }


def fit_predict_origin(
    train: pd.DataFrame, audit_year: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit on the two immediately previous seasons and predict one full year."""

    fit = train.loc[
        train["season"].between(audit_year - 2, audit_year - 1)
    ].reset_index(drop=True)
    audit = train.loc[train["season"].eq(audit_year)].reset_index(drop=True)
    if fit.empty or audit.empty:
        raise ValueError(f"missing fit/audit rows for origin {audit_year}")
    fit_numeric = row_local_numeric(train, fit)
    audit_numeric = row_local_numeric(train, audit)
    fit_matrix, audit_matrix, shape = matrix_pair(
        fit_numeric,
        audit_numeric,
        interaction_frame(fit),
        interaction_frame(audit),
    )
    target = fit[TARGET].to_numpy(np.int8)
    sample_weight = np.where(
        fit["season"].eq(audit_year - 1).to_numpy(),
        RECENT_SEASON_WEIGHT,
        OLDER_SEASON_WEIGHT,
    ).astype(np.float64)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=2.0e-5,
        max_iter=15,
        tol=1.0e-4,
        shuffle=True,
        random_state=4600 + audit_year,
        average=True,
        n_jobs=6,
    )
    model.fit(fit_matrix, target, sample_weight=sample_weight)
    prediction = model.predict_proba(audit_matrix)[:, 1].astype(np.float64)
    summary = {
        "audit_year": int(audit_year),
        "fit_years": [int(audit_year - 2), int(audit_year - 1)],
        "fit_rows": int(len(fit)),
        "audit_rows": int(len(audit)),
        "feature_shape": shape,
        "iterations": int(model.n_iter_),
        "prediction_mean": float(prediction.mean()),
        "prediction_sd": float(prediction.std()),
    }
    del fit_numeric, audit_numeric, fit_matrix, audit_matrix, model
    gc.collect()
    return np.clip(prediction, 0.001, 0.999), summary


def _selection_frame(
    target: np.ndarray, month: np.ndarray, domain: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": np.asarray(target, dtype=np.float64),
            "game_month": np.asarray(month, dtype=np.int16),
            "domain3": np.asarray(domain).astype(str),
        }
    )


def _screen(
    frame: pd.DataFrame, parent: np.ndarray, raw: np.ndarray
) -> pd.DataFrame:
    rows = []
    for route in ("ALL", "R_CORE", "R_ANCHOR", "F"):
        rows.extend(
            grid_rows(
                frame,
                parent,
                parent,
                raw,
                signal=SIGNAL,
                direction_mode="toward_parent",
                route=route,
            )
        )
    return pd.DataFrame(rows)


def _candidate(
    frame: pd.DataFrame, raw: np.ndarray, route: str, weight: float
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    active = (
        np.ones(len(frame), dtype=bool)
        if route == "ALL"
        else frame["domain3"].astype(str).eq(route).to_numpy()
    )
    output = parent.copy()
    output[active] = np.clip(
        parent[active]
        + float(weight) * (np.asarray(raw, dtype=np.float64)[active] - parent[active]),
        0.001,
        0.999,
    )
    return output, active


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    predictions: dict[int, np.ndarray] = {}
    fit_summaries = []
    for year in (2022, 2023, 2024):
        print(f"[v46] fit/predict origin={year}", flush=True)
        predictions[year], fit_summary = fit_predict_origin(train, year)
        fit_summaries.append(fit_summary)
        rows = train.loc[train["season"].eq(year)].reset_index(drop=True)
        if len(rows) != len(predictions[year]):
            raise ValueError(f"prediction row mismatch for {year}")
        np.savez_compressed(
            output_dir / f"sparse_logit_o{year}.npz",
            target=rows[TARGET].to_numpy(np.int8),
            prediction=predictions[year],
            game_month=rows["game_month"].to_numpy(np.int16),
            domain3=rows["domain3"].astype(str).to_numpy(),
        )

    meta22 = _metadata(project, 2022)
    if not np.array_equal(
        meta22["target"],
        train.loc[train["season"].eq(2022), TARGET].to_numpy(np.float64),
    ):
        raise ValueError("2022 metadata target mismatch")
    frame22 = _selection_frame(
        meta22["target"], meta22["month"], meta22["domain"]
    )
    stage1 = _screen(frame22, meta22["parent"], predictions[2022])
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)

    axes = _cached_v25_axes(project, train)
    frame23 = axes["selection_late_2023"]
    late23 = train.loc[train["season"].eq(2023), "game_month"].ge(8).to_numpy()
    if not np.array_equal(
        frame23["target"].to_numpy(np.float64),
        train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8), TARGET
        ].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 target mismatch")
    stage2 = _screen(frame23, v27_parent(frame23), predictions[2023][late23])
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": SIGNAL,
        "domain": str(chosen["domain"]),
        "weight": float(chosen["weight"]),
    }

    results: dict[str, dict[str, object]] = {}
    late24 = train.loc[train["season"].eq(2024), "game_month"].ge(8).to_numpy()
    for axis, raw in (
        ("outer_full_2024", predictions[2024]),
        ("replication_late_2024", predictions[2024][late24]),
    ):
        frame = axes[axis]
        if len(raw) != len(frame):
            raise ValueError(f"2024 prediction row mismatch: {axis}")
        candidate, active = _candidate(
            frame, raw, recipe["domain"], recipe["weight"]
        )
        results[axis] = diagnostics(
            frame, v27_parent(frame), candidate, active
        )
        np.savez_compressed(
            output_dir / f"{axis}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=v27_parent(frame),
            sparse_logit=raw,
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    gates = {
        "two_origin_consensus": bool(chosen["passes_consensus_gate"]),
        "outer_gain_at_least_5": results["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": results["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": results["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "outer_minimum_domain_nonnegative": results["outer_full_2024"][
            "minimum_domain_gain"
        ]
        >= 0.0,
        "replication_gain_positive": results["replication_late_2024"]["gain"]
        > 0.0,
        "replication_month_fraction_at_least_two_thirds": results[
            "replication_late_2024"
        ]["positive_month_fraction"]
        >= 2.0 / 3.0,
    }
    summary = {
        "protocol": "V46_TWO_ORIGIN_SPARSE_LOGIT_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "model": {
            "family": "averaged SGD logistic on sparse baseball interactions",
            "fit_window": "two immediately preceding seasons",
            "alpha": 2.0e-5,
            "max_iter": 15,
            "min_category_frequency": 25,
            "older_season_weight": OLDER_SEASON_WEIGHT,
        },
        "fit_summaries": fit_summaries,
        "chosen": {
            **recipe,
            "gain_2022": float(chosen["gain_2022"]),
            "gain_late_2023": float(chosen["gain_2023"]),
            "worst_month_2022": float(chosen["worst_month_gain_2022"]),
            "worst_month_late_2023": float(chosen["worst_month_gain_2023"]),
            "consensus_score": float(chosen["consensus_score"]),
        },
        "consensus_candidate_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "audits": results,
        "gates": {key: bool(value) for key, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v46_sparse_logit_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
