"""Group-balanced ExtraTrees diversity screen above the final 1158 parent.

The incumbent already contains boosted trees, random forests, neural models,
empirical-Bayes lookups, and a tiny TrackMan gate.  This candidate deliberately
uses a low-variance, highly randomised squared-error forest without player IDs.
Only row-local game state and official ASOF counters are used.  Two fixed risks
are compared: ordinary row risk and equal total risk for every source-season x
deployment-domain group.

One exact model/risk/route/weight recipe must improve both the older 2022
origin and late 2023.  That recipe is frozen before the final-gate full-2024
and late-2024 audits are opened.  Query-row aggregates are never computed.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v30_diverse_covariance_screen import diagnostics
from src.v35_three_stage_multibank import _metadata


TARGET = "control_success"
FIT_WINDOW = 3
RISKS = ("uniform", "season_domain_equal")
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.02, 0.05, 0.10, 0.15, 0.20)
CONSENSUS_KEYS = ("risk", "route", "weight")
MIN_CATEGORY_FREQUENCY = 25

IDENTITY_OR_LABEL_COLUMNS = {
    "row_id",
    "season",
    TARGET,
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
}
CATEGORICAL_COLUMNS = (
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "count_state",
    "hand_matchup",
    "domain3",
)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fixed baseball-state features without aggregating input rows."""

    out = _add_domain_and_pressure(frame)
    balls = _numeric(out, "balls_before", -1.0)
    strikes = _numeric(out, "strikes_before", -1.0)
    pitcher_hand = out["pitcher_hand"].astype("string").fillna("__MISSING__")
    batter_hand = out["batter_hand"].astype("string").fillna("__MISSING__")
    out["count_state"] = (
        pd.Series(balls, index=out.index).fillna(-1).astype(int).astype(str)
        + "-"
        + pd.Series(strikes, index=out.index).fillna(-1).astype(int).astype(str)
    )
    out["hand_matchup"] = pitcher_hand.astype(str) + "-" + batter_hand.astype(str)

    pitcher_n = np.maximum(_numeric(out, "asof_pitcher_n"), 0.0)
    batter_n = np.maximum(_numeric(out, "asof_batter_n"), 0.0)
    pitchmix_n = np.maximum(_numeric(out, "asof_pitcher_pitchmix_n"), 0.0)
    career = _numeric(out, "asof_pitcher_success_rate", 0.5)
    prev1 = _numeric(out, "asof_pitcher_prev1_game_success_rate", 0.5)
    prev3 = _numeric(out, "asof_pitcher_prev3_game_success_rate", 0.5)
    prev5 = _numeric(out, "asof_pitcher_prev5_game_success_rate", 0.5)
    recent = 0.50 * prev1 + 0.30 * prev3 + 0.20 * prev5
    strike_rate = _numeric(out, "asof_pitcher_strike_rate", 0.0)
    ball_rate = _numeric(out, "asof_pitcher_ball_rate", 0.0)
    middle_rate = _numeric(out, "asof_pitcher_middle_rate", 0.0)
    reverse_rate = _numeric(out, "asof_pitcher_reverse_rate", 0.0)
    pressure = ((balls == 3) | (strikes == 2)).astype(np.float64)

    out["eng_pitcher_log_n"] = np.log1p(pitcher_n)
    out["eng_batter_log_n"] = np.log1p(batter_n)
    out["eng_pitchmix_log_n"] = np.log1p(pitchmix_n)
    out["eng_pitcher_reliability"] = pitcher_n / (pitcher_n + 160.0)
    out["eng_batter_reliability"] = batter_n / (batter_n + 160.0)
    out["eng_pitchmix_reliability"] = pitchmix_n / (pitchmix_n + 160.0)
    out["eng_recent_blend"] = recent
    out["eng_recent_minus_career"] = recent - career
    out["eng_prev1_minus_prev5"] = prev1 - prev5
    out["eng_strike_ball_margin"] = strike_rate - ball_rate
    out["eng_middle_reverse_margin"] = middle_rate - reverse_rate
    out["eng_pressure"] = pressure
    out["eng_pressure_recent_delta"] = pressure * (recent - career)
    out["eng_pressure_middle_reverse"] = pressure * (middle_rate - reverse_rate)
    out["eng_same_hand"] = (pitcher_hand.to_numpy() == batter_hand.to_numpy()).astype(
        np.float64
    )
    out["eng_abs_score_diff"] = np.abs(_numeric(out, "score_diff_pitcher_team"))

    mix = np.column_stack(
        [
            _numeric(out, "asof_pitcher_fastball_rate", 1.0 / 3.0),
            _numeric(out, "asof_pitcher_breaking_rate", 1.0 / 3.0),
            _numeric(out, "asof_pitcher_offspeed_rate", 1.0 / 3.0),
        ]
    )
    mix = np.where(np.isfinite(mix), mix, 1.0 / 3.0)
    mix = np.clip(mix, 1e-6, None)
    mix /= mix.sum(axis=1, keepdims=True)
    out["eng_pitchmix_entropy"] = -np.sum(mix * np.log(mix), axis=1)
    return out


def _safe_category(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string").fillna("__MISSING__").astype(str)


def matrix_pair(
    source: pd.DataFrame, query: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build dense float matrices; all preprocessing is fitted on source only."""

    numeric_columns = [
        column
        for column in source.columns
        if column not in IDENTITY_OR_LABEL_COLUMNS
        and column not in CATEGORICAL_COLUMNS
        and pd.api.types.is_numeric_dtype(source[column])
    ]
    left_numeric = source[numeric_columns].apply(pd.to_numeric, errors="coerce")
    right_numeric = query[numeric_columns].apply(pd.to_numeric, errors="coerce")
    medians = left_numeric.median(axis=0).fillna(0.0)
    left_blocks = [left_numeric.fillna(medians).to_numpy(np.float32)]
    right_blocks = [right_numeric.fillna(medians).to_numpy(np.float32)]
    category_names: list[str] = []
    for column in CATEGORICAL_COLUMNS:
        left = _safe_category(source, column)
        right = _safe_category(query, column)
        counts = left.value_counts(sort=False)
        values = sorted(counts.index[counts.ge(MIN_CATEGORY_FREQUENCY)].tolist())
        for value in values:
            left_blocks.append(left.eq(value).to_numpy(np.float32)[:, None])
            right_blocks.append(right.eq(value).to_numpy(np.float32)[:, None])
            category_names.append(f"{column}={value}")
    left_matrix = np.ascontiguousarray(np.hstack(left_blocks), dtype=np.float32)
    right_matrix = np.ascontiguousarray(np.hstack(right_blocks), dtype=np.float32)
    if left_matrix.shape[1] != right_matrix.shape[1]:
        raise ValueError("source/query feature width mismatch")
    if not np.isfinite(left_matrix).all() or not np.isfinite(right_matrix).all():
        raise ValueError("non-finite ExtraTrees feature")
    return left_matrix, right_matrix, {
        "numeric_columns": numeric_columns,
        "numeric_count": int(len(numeric_columns)),
        "categorical_encoded_count": int(len(category_names)),
        "total_count": int(left_matrix.shape[1]),
        "categorical_features": category_names,
    }


def group_weights(source: pd.DataFrame, risk: str) -> np.ndarray:
    """Return mean-one weights with equal total mass per season/domain group."""

    if risk == "uniform":
        return np.ones(len(source), dtype=np.float64)
    if risk != "season_domain_equal":
        raise ValueError(f"unknown risk: {risk}")
    keys = source["season"].astype(str) + "\x1f" + source["domain3"].astype(str)
    counts = keys.value_counts(sort=False)
    weight = keys.map(1.0 / counts).to_numpy(np.float64)
    weight /= weight.mean()
    return weight


def fit_predict_origin(
    train: pd.DataFrame, origin: int
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Fit both preregistered risks on the three seasons before one origin."""

    source = train.loc[
        train["season"].between(origin - FIT_WINDOW, origin - 1)
    ].reset_index(drop=True)
    query = train.loc[train["season"].eq(origin)].reset_index(drop=True)
    if source.empty or query.empty:
        raise ValueError(f"empty source/query for origin {origin}")
    left, right, schema = matrix_pair(source, query)
    target = source[TARGET].to_numpy(np.float64)
    predictions: dict[str, np.ndarray] = {}
    fit_audits: dict[str, object] = {}
    for risk_index, risk in enumerate(RISKS):
        weight = group_weights(source, risk)
        model = ExtraTreesRegressor(
            n_estimators=64,
            criterion="squared_error",
            max_depth=18,
            min_samples_leaf=300,
            max_features=0.70,
            bootstrap=True,
            max_samples=0.65,
            n_jobs=6,
            random_state=6200 + 10 * origin + risk_index,
        )
        model.fit(left, target, sample_weight=weight)
        prediction = np.clip(model.predict(right), 0.01, 0.99).astype(np.float64)
        predictions[risk] = prediction
        keys = source["season"].astype(str) + ":" + source["domain3"].astype(str)
        fit_audits[risk] = {
            "weight_min": float(weight.min()),
            "weight_max": float(weight.max()),
            "weighted_mass_by_group": {
                str(key): float(weight[keys.eq(key).to_numpy()].sum())
                for key in sorted(keys.unique())
            },
            "prediction_mean": float(prediction.mean()),
            "prediction_sd": float(prediction.std()),
        }
        del model
        gc.collect()
    audit = {
        "origin": int(origin),
        "source_years": list(range(origin - FIT_WINDOW, origin)),
        "source_rows": int(len(source)),
        "query_rows": int(len(query)),
        "schema": schema,
        "risks": fit_audits,
    }
    del left, right
    gc.collect()
    return predictions, audit


def _frame(target: np.ndarray, month: np.ndarray, domain: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": np.asarray(target, dtype=np.float64),
            "game_month": np.asarray(month, dtype=np.int16),
            "domain3": np.asarray(domain).astype(str),
        }
    )


def blend_candidate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    challenger: np.ndarray,
    route: str,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    challenger = np.asarray(challenger, dtype=np.float64)
    if parent.shape != challenger.shape or parent.shape != (len(frame),):
        raise ValueError("blend input length mismatch")
    active = (
        np.ones(len(frame), dtype=bool)
        if route == "ALL"
        else frame["domain3"].astype(str).eq(route).to_numpy()
    )
    output = parent.copy()
    output[active] = np.clip(
        parent[active]
        + float(weight) * (challenger[active] - parent[active]),
        0.001,
        0.999,
    )
    return output, active


def _screen(
    frame: pd.DataFrame,
    parent: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for risk, challenger in predictions.items():
        for route in ROUTES:
            for weight in WEIGHTS:
                candidate, active = blend_candidate(
                    frame, parent, challenger, route, weight
                )
                result = diagnostics(frame, parent, candidate, active)
                applied = (
                    min(result["domain_gains"].values())
                    if route == "ALL"
                    else result["domain_gains"][route]
                )
                rows.append(
                    {
                        "risk": risk,
                        "route": route,
                        "weight": float(weight),
                        "gain": float(result["gain"]),
                        "positive_month_fraction": float(
                            result["positive_month_fraction"]
                        ),
                        "worst_month_gain": float(result["worst_month_gain"]),
                        "applied_domain_gain": float(applied),
                        "mean_abs_shift": float(result["mean_abs_shift"]),
                    }
                )
    return pd.DataFrame(rows)


def select_consensus(
    older: pd.DataFrame, recent: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    merged = older.merge(
        recent, on=list(CONSENSUS_KEYS), suffixes=("_2022", "_2023")
    )
    if merged.empty:
        raise ValueError("empty exact-recipe consensus")
    merged["minimum_gain"] = merged[["gain_2022", "gain_2023"]].min(axis=1)
    merged["minimum_worst_month"] = merged[
        ["worst_month_gain_2022", "worst_month_gain_2023"]
    ].min(axis=1)
    merged["minimum_applied_domain"] = merged[
        ["applied_domain_gain_2022", "applied_domain_gain_2023"]
    ].min(axis=1)
    merged["consensus_score"] = merged[
        ["minimum_gain", "minimum_worst_month", "minimum_applied_domain"]
    ].min(axis=1)
    merged["passes_consensus_gate"] = (
        merged["gain_2022"].gt(0.0)
        & merged["gain_2023"].gt(0.0)
        & merged["positive_month_fraction_2022"].ge(0.75)
        & merged["positive_month_fraction_2023"].eq(1.0)
        & merged["minimum_worst_month"].gt(0.0)
        & merged["minimum_applied_domain"].gt(0.0)
    )
    merged = merged.sort_values(
        ["passes_consensus_gate", "consensus_score", "minimum_gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    passing = merged.loc[merged["passes_consensus_gate"]]
    chosen = passing.iloc[0] if len(passing) else merged.iloc[0]
    return merged, chosen


def _load_final_axis(final_oof_dir: Path, name: str) -> tuple[pd.DataFrame, np.ndarray]:
    # ``domain3`` is a NumPy object-string array in the frozen v61 cache.
    with np.load(final_oof_dir / f"{name}.npz", allow_pickle=True) as saved:
        frame = _frame(saved["target"], saved["game_month"], saved["domain3"])
        parent = saved["final_gate_parent"].astype(np.float64)
    return frame, parent


def _load_prediction_cache(
    path: Path, expected_target: np.ndarray
) -> dict[str, np.ndarray] | None:
    """Resume an interrupted audit without repeating expensive forest fits."""

    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as saved:
        if not np.array_equal(saved["target"].astype(np.float64), expected_target):
            raise ValueError(f"cached target/order mismatch: {path}")
        missing = set(RISKS) - set(saved.files)
        if missing:
            raise ValueError(f"cached risks missing from {path}: {sorted(missing)}")
        return {risk: saved[risk].astype(np.float64) for risk in RISKS}


def run(project: Path, final_oof_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    final_oof_dir = final_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = prepare_rows(pd.read_csv(project / "data" / "train.csv", low_memory=False))
    predictions: dict[int, dict[str, np.ndarray]] = {}
    fit_audits: dict[str, object] = {}
    for origin in (2022, 2023, 2024):
        rows = train.loc[train["season"].eq(origin)].reset_index(drop=True)
        target = rows[TARGET].to_numpy(np.float64)
        cache_path = output_dir / f"extra_trees_o{origin}.npz"
        cached = _load_prediction_cache(cache_path, target)
        if cached is not None:
            print(f"[v62] reuse ExtraTrees origin={origin}", flush=True)
            predictions[origin] = cached
            fit_audits[str(origin)] = {
                "origin": int(origin),
                "source_years": list(range(origin - FIT_WINDOW, origin)),
                "query_rows": int(len(rows)),
                "reused_prediction_cache": True,
            }
        else:
            print(f"[v62] fit ExtraTrees origin={origin}", flush=True)
            predictions[origin], fit_audits[str(origin)] = fit_predict_origin(
                train, origin
            )
            np.savez_compressed(
                cache_path,
                target=target,
                game_month=rows["game_month"].to_numpy(np.int16),
                domain3=rows["domain3"].astype(str).to_numpy(),
                **predictions[origin],
            )

    meta22 = _metadata(project, 2022)
    expected22 = train.loc[train["season"].eq(2022), TARGET].to_numpy(np.float64)
    if not np.array_equal(meta22["target"], expected22):
        raise ValueError("2022 metadata target/order mismatch")
    frame22 = _frame(meta22["target"], meta22["month"], meta22["domain"])
    stage1 = _screen(frame22, meta22["parent"], predictions[2022])
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)

    frame23, parent23 = _load_final_axis(final_oof_dir, "selection_late_2023")
    late23 = train.loc[train["season"].eq(2023), "game_month"].ge(8).to_numpy()
    expected23 = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8), TARGET
    ].to_numpy(np.float64)
    if not np.array_equal(frame23["target"].to_numpy(np.float64), expected23):
        raise ValueError("late-2023 final-parent target/order mismatch")
    stage2_predictions = {
        risk: value[late23] for risk, value in predictions[2023].items()
    }
    stage2 = _screen(frame23, parent23, stage2_predictions)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "risk": str(chosen["risk"]),
        "route": str(chosen["route"]),
        "weight": float(chosen["weight"]),
    }

    audits: dict[str, object] = {}
    for name in ("outer_full_2024", "replication_late_2024"):
        frame, parent = _load_final_axis(final_oof_dir, name)
        mask = (
            np.ones(len(predictions[2024][recipe["risk"]]), dtype=bool)
            if name == "outer_full_2024"
            else train.loc[train["season"].eq(2024), "game_month"].ge(8).to_numpy()
        )
        challenger = predictions[2024][recipe["risk"]][mask]
        if not np.array_equal(
            frame["target"].to_numpy(np.float64),
            train.loc[train["season"].eq(2024), TARGET].to_numpy(np.float64)[mask],
        ):
            raise ValueError(f"2024 target/order mismatch: {name}")
        candidate, active = blend_candidate(
            frame, parent, challenger, recipe["route"], recipe["weight"]
        )
        audits[name] = diagnostics(frame, parent, candidate, active)
        np.savez_compressed(
            output_dir / f"{name}.npz",
            target=frame["target"].to_numpy(np.float64),
            final_gate_parent=parent,
            extra_trees=challenger,
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    outer = audits["outer_full_2024"]
    late = audits["replication_late_2024"]
    gates = {
        "two_origin_exact_consensus": bool(chosen["passes_consensus_gate"]),
        "outer_gain_at_least_5": bool(outer["gain"] >= 5.0),
        "outer_positive_month_fraction_at_least_075": bool(
            outer["positive_month_fraction"] >= 0.75
        ),
        "outer_worst_month_above_minus_5": bool(outer["worst_month_gain"] > -5.0),
        "outer_all_domains_nonnegative": bool(outer["minimum_domain_gain"] >= 0.0),
        "late_gain_positive": bool(late["gain"] > 0.0),
        "late_all_months_positive": bool(late["positive_month_fraction"] == 1.0),
    }
    summary = {
        "protocol": "V62_GROUP_BALANCED_EXTRA_TREES_FINAL_PARENT_V1",
        "parent": "standalone champion 1158.0745556751 exact temporal OOF",
        "model": {
            "family": "ExtraTrees squared-error direct probability",
            "player_ids_used": False,
            "fit_window": FIT_WINDOW,
            "risks": list(RISKS),
            "n_estimators": 64,
            "max_depth": 18,
            "min_samples_leaf": 300,
            "max_features": 0.70,
            "bootstrap_max_samples": 0.65,
        },
        "fit_audits": fit_audits,
        "selection": "exact risk/route/weight consensus on 2022 and late-2023 only",
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
        "audits": audits,
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "test_distribution_used": False,
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
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--final-oof-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v62_group_balanced_extra_trees_20260822_01"),
    )
    args = parser.parse_args()
    output = args.output_dir
    if not output.is_absolute():
        output = Path.cwd() / output
    run(args.project, args.final_oof_dir, output)


if __name__ == "__main__":
    main()
