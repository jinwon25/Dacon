"""Forward-only residual experiment with corrected within-season state.

The base prediction for every outer season is the stored V2-NESTED-R1 OOF
prediction.  A small LightGBM learns ``target - V2`` only from earlier outer
OOF seasons.  It changes regular-season rows only, matching the 2025 sample
regime and isolating the unstable ``game_type=F`` block.

The important ablation is ``context`` versus ``corrected_state``.  The latter
uses the immediately preceding season endpoint; the earlier implementation
accidentally shifted that endpoint twice and therefore subtracted the state
from two seasons ago.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta
from src.top1100_features import build_features


YEARS = (2021, 2022, 2023, 2024)
ETAS = (0.0, 0.05, 0.10, 0.20, 0.25, 0.35, 0.50)
STATE_COLUMNS = {
    "pitcher_season_n",
    "pitcher_season_success_count",
    "batter_season_n",
    "batter_season_success_count",
    "pitcher_prior_rate",
    "pitcher_season_rate",
    "pitcher_season_logit_delta",
    "batter_prior_rate",
    "batter_season_rate",
    "pitcher_state_sd",
    "pitcher_newcomer",
    "pitcher_long_gap",
}


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest().upper()


def _find_oof(research_project: Path, year: int) -> Path:
    matches = sorted(
        (research_project / "artifacts" / "top1100" / "v2_r1_oof").rglob(
            f"v2_r1_o{year}.npz"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one V2 OOF artifact for {year}, found {matches}")
    return matches[0]


def _load_folds(research_project: Path, train: pd.DataFrame) -> dict[int, pd.DataFrame]:
    train_ids = pd.Index(train["row_id"].astype(str))
    if train_ids.has_duplicates:
        raise ValueError("train row_id is not unique")
    folds: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        path = _find_oof(research_project, year)
        with np.load(path, allow_pickle=True) as bundle:
            row_id = bundle["row_id"].astype(str)
            index = train_ids.get_indexer(row_id)
            if (index < 0).any():
                raise ValueError(f"{year} OOF contains row_id absent from train")
            target = bundle["target"].astype(np.float64)
            v2 = bundle["p_v2_nested"].astype(np.float64)
        frame = pd.DataFrame(
            {
                "row_id": row_id,
                "train_index": index,
                "target": target,
                "v2": v2,
                "game_type": train.iloc[index]["game_type"].astype(str).to_numpy(),
                "pitcher_id": train.iloc[index]["pitcher_id"].astype(str).to_numpy(),
            }
        )
        expected = train.iloc[index]["control_success"].to_numpy(np.float64)
        if not np.array_equal(target, expected):
            raise ValueError(f"{year} OOF target does not match train")
        folds[year] = frame
    return folds


def _make_features(train: pd.DataFrame) -> pd.DataFrame:
    features = build_features(train, train, include_ids=False)
    # Low-cardinality row-local categories omitted by include_ids=False.
    for column in ("game_type", "top_bottom", "base_state", "pitcher_hand", "batter_hand"):
        features[column] = (
            train[column].astype("string").fillna("__MISSING__").astype("category")
        )
    return features


def _align_categories(
    fit: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fit = fit.copy()
    audit = audit.copy()
    categorical = list(fit.select_dtypes(include=["category", "object", "string"]).columns)
    for column in categorical:
        vocabulary = pd.Index(fit[column].astype("string").fillna("__MISSING__").unique())
        fit[column] = pd.Categorical(
            fit[column].astype("string").fillna("__MISSING__"), categories=vocabulary
        )
        audit[column] = pd.Categorical(
            audit[column].astype("string").fillna("__MISSING__"), categories=vocabulary
        )
    return fit, audit, categorical


def _fit_correction(
    fit_features: pd.DataFrame,
    fit_residual: np.ndarray,
    audit_features: pd.DataFrame,
) -> np.ndarray:
    fit_features, audit_features, categorical = _align_categories(
        fit_features, audit_features
    )
    dataset = lgb.Dataset(
        fit_features,
        label=fit_residual,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 5,
            "min_data_in_leaf": 2000,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 50.0,
            "verbosity": -1,
            "seed": 20260814,
            "num_threads": 6,
            "deterministic": True,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=160,
        callbacks=[lgb.log_evaluation(0)],
    )
    return np.clip(np.asarray(model.predict(audit_features), dtype=np.float64), -0.04, 0.04)


def _candidate_prediction(
    fold: pd.DataFrame, correction: np.ndarray, eta: float
) -> np.ndarray:
    prediction = fold["v2"].to_numpy(np.float64, copy=True)
    regular = fold["game_type"].eq("R").to_numpy()
    prediction[regular] = np.clip(
        prediction[regular] + eta * correction,
        1e-6,
        1.0 - 1e-6,
    )
    return prediction


def _choose_recipe(metrics: pd.DataFrame, years: tuple[int, ...]) -> pd.Series:
    eligible = metrics.loc[metrics["season"].isin(years)].copy()
    eligible["selection_weight"] = eligible["season"].map(
        {year: i + 1 for i, year in enumerate(years)}
    )
    summary = (
        eligible.groupby(["feature_recipe", "fit_mode", "eta"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "selection_delta": np.average(
                        group["delta"], weights=group["selection_weight"]
                    )
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    # Conservative deterministic tie-break: smaller correction, state recipe,
    # then R-only fitting.  eta=0 remains available as a fail-closed choice.
    summary["state_preference"] = summary["feature_recipe"].ne("corrected_state")
    summary["mode_preference"] = summary["fit_mode"].ne("R_ONLY")
    return summary.sort_values(
        ["selection_delta", "eta", "state_preference", "mode_preference"],
        kind="mergesort",
    ).iloc[0]


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate extra."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if np.isnan(value) else f"{float(value):.12g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def finalize_existing(out_dir: Path) -> None:
    """Finish the human-readable report from already persisted run outputs."""
    out_dir = out_dir.resolve()
    selection = pd.read_csv(out_dir / "nested_selection.csv")
    bootstrap = json.loads((out_dir / "bootstrap.json").read_text(encoding="utf-8"))
    weighted_delta = float(
        np.average(selection["audit_delta"], weights=selection["season"] - 2020)
    )
    decision = (
        "# Corrected season-state residual OOF\n\n"
        "V2-NESTED-R1 is corrected only on regular-season rows. The O22 recipe "
        "is fixed in code; O23 and O24 select from completed earlier outer folds only.\n\n"
        + _markdown_table(selection)
        + f"\n\nRecency-weighted selected delta: `{weighted_delta:.12f}`.\n\n"
        + "Paired season:pitcher bootstrap:\n\n```json\n"
        + json.dumps(bootstrap, indent=2)
        + "\n```\n"
    )
    (out_dir / "decision.md").write_text(decision, encoding="utf-8")


def run(
    data_project: Path,
    research_project: Path,
    out_dir: Path,
    n_resamples: int = 5000,
) -> pd.DataFrame:
    data_project = data_project.resolve()
    research_project = research_project.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    train = read_main(data_project / "data" / "train.csv")
    folds = _load_folds(research_project, train)
    features = _make_features(train)
    recipes = {
        "context": [column for column in features.columns if column not in STATE_COLUMNS],
        "corrected_state": list(features.columns),
    }

    metric_rows: list[dict[str, object]] = []
    corrections: dict[tuple[int, str, str], np.ndarray] = {}
    for year in (2022, 2023, 2024):
        history = pd.concat([folds[past] for past in YEARS if past < year], ignore_index=True)
        audit = folds[year]
        regular_audit = audit["game_type"].eq("R").to_numpy()
        for feature_recipe, columns in recipes.items():
            for fit_mode in ("ALL", "R_ONLY"):
                fit_rows = history
                if fit_mode == "R_ONLY":
                    fit_rows = history.loc[history["game_type"].eq("R")]
                fit_index = fit_rows["train_index"].to_numpy(np.int64)
                audit_index = audit.loc[regular_audit, "train_index"].to_numpy(np.int64)
                correction = _fit_correction(
                    features.iloc[fit_index][columns].reset_index(drop=True),
                    (
                        fit_rows["target"].to_numpy(np.float64)
                        - fit_rows["v2"].to_numpy(np.float64)
                    ),
                    features.iloc[audit_index][columns].reset_index(drop=True),
                )
                corrections[(year, feature_recipe, fit_mode)] = correction
                np.savez_compressed(
                    out_dir / f"correction_{feature_recipe}_{fit_mode.lower()}_o{year}.npz",
                    row_id=audit.loc[regular_audit, "row_id"].astype(str).to_numpy(),
                    correction=correction,
                )
                baseline_brier = brier_score(audit["target"], audit["v2"])
                r_baseline = brier_score(
                    audit.loc[regular_audit, "target"], audit.loc[regular_audit, "v2"]
                )
                for eta in ETAS:
                    prediction = _candidate_prediction(audit, correction, eta)
                    candidate_brier = brier_score(audit["target"], prediction)
                    r_brier = brier_score(
                        audit.loc[regular_audit, "target"], prediction[regular_audit]
                    )
                    metric_rows.append(
                        {
                            "season": year,
                            "feature_recipe": feature_recipe,
                            "fit_mode": fit_mode,
                            "eta": eta,
                            "n_fit": len(fit_rows),
                            "n_audit": len(audit),
                            "n_audit_r": int(regular_audit.sum()),
                            "brier": candidate_brier,
                            "v2_brier": baseline_brier,
                            "delta": candidate_brier - baseline_brier,
                            "r_delta": r_brier - r_baseline,
                            "outer_target_used_for_fit": False,
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "grid_metrics.csv", index=False)

    selections: list[dict[str, object]] = []
    selected_predictions: dict[int, np.ndarray] = {}
    # O22 is the preregistered starting point.  Later decisions only use
    # already completed outer folds.
    fixed_o22 = {
        "feature_recipe": "corrected_state",
        "fit_mode": "R_ONLY",
        "eta": 0.25,
        "selection_delta": np.nan,
    }
    for year, history_years in ((2022, ()), (2023, (2022,)), (2024, (2022, 2023))):
        choice = pd.Series(fixed_o22) if not history_years else _choose_recipe(metrics, history_years)
        feature_recipe = str(choice["feature_recipe"])
        fit_mode = str(choice["fit_mode"])
        eta = float(choice["eta"])
        audit = folds[year]
        prediction = _candidate_prediction(
            audit, corrections[(year, feature_recipe, fit_mode)], eta
        )
        selected_predictions[year] = prediction
        delta = brier_score(audit["target"], prediction) - brier_score(
            audit["target"], audit["v2"]
        )
        selections.append(
            {
                "season": year,
                "selection_years": ",".join(map(str, history_years)) or "preregistered",
                "feature_recipe": feature_recipe,
                "fit_mode": fit_mode,
                "eta": eta,
                "selection_delta": float(choice["selection_delta"]),
                "audit_delta": delta,
                "outer_target_used_for_selection": False,
            }
        )
    selection = pd.DataFrame(selections)
    selection.to_csv(out_dir / "nested_selection.csv", index=False)

    combined_target = np.concatenate([folds[year]["target"].to_numpy() for year in (2022, 2023, 2024)])
    combined_v2 = np.concatenate([folds[year]["v2"].to_numpy() for year in (2022, 2023, 2024)])
    combined_candidate = np.concatenate([selected_predictions[year] for year in (2022, 2023, 2024)])
    combined_cluster = np.concatenate(
        [
            (str(year) + ":" + folds[year]["pitcher_id"].astype(str)).to_numpy()
            for year in (2022, 2023, 2024)
        ]
    )
    bootstrap = cluster_bootstrap_delta(
        combined_target,
        combined_candidate,
        combined_v2,
        combined_cluster,
        n_resamples=n_resamples,
        seed=20260814,
    )
    (out_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    manifest = {
        "recipe_id": "CORRECTED-STATE-R-RESID-OFF-V2-R1",
        "years": YEARS,
        "eta_grid": ETAS,
        "outer_target_used_for_fit": False,
        "outer_target_used_for_selection": False,
        "v2_hashes": {
            str(year): _sha256_array(folds[year]["v2"].to_numpy()) for year in YEARS
        },
        "state_columns": sorted(STATE_COLUMNS),
        "feature_counts": {name: len(columns) for name, columns in recipes.items()},
        "bootstrap": bootstrap,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    weighted_delta = float(
        np.average(selection["audit_delta"], weights=selection["season"] - 2020)
    )
    finalize_existing(out_dir)
    print(selection.to_string(index=False), flush=True)
    print(f"recency_weighted_delta={weighted_delta:.12f}", flush=True)
    print(json.dumps(bootstrap, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, required=True)
    parser.add_argument("--research-project", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    run(args.data_project, args.research_project, args.out_dir, args.n_resamples)


if __name__ == "__main__":
    main()
