"""Multi-season residual model around row-local current-season posteriors.

Unlike the recent exact model, this learner uses all seasons before an audit
year with fixed exponential decay.  Absolute annual drift is absorbed by an
inference-safe baseline reconstructed from the row's cumulative ASOF counters;
the shallow LightGBM learns only repeatable contextual residual structure.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.recent_shared_exact_asof import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    _current_season_state,
    _make_season_bank,
    _row_state,
)
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.trackman_privileged_distillation import V17_NAME, _diagnostics


def _state_features(train: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    parts: list[pd.DataFrame] = []
    baselines: list[np.ndarray] = []
    for season in sorted(train["season"].unique()):
        rows = train.loc[train["season"].eq(season)].reset_index(drop=True)
        numeric = rows[[column for column in NUMERIC_COLUMNS if column in rows]].apply(
            pd.to_numeric, errors="coerce"
        ).reset_index(drop=True)
        state = _current_season_state(rows, _make_season_bank(train, int(season)))
        local = _row_state(rows)
        features = pd.concat([numeric, state, local], axis=1)
        features["source_season"] = int(season)
        features["original_index"] = rows.index.to_numpy()
        parts.append(features)
        pitcher = state["season__pitcher_rate_k80"].to_numpy(np.float64)
        batter = state["season__batter_rate_k80"].to_numpy(np.float64)
        baselines.append(0.75 * pitcher + 0.25 * batter)
    # train.csv is season-ordered.  Assert rather than silently relying on it.
    output = pd.concat(parts, ignore_index=True)
    if not np.array_equal(
        output["source_season"].to_numpy(), train["season"].to_numpy()
    ):
        raise ValueError("train rows are not season ordered")
    output = output.drop(columns=["source_season", "original_index"])
    return output.astype(np.float32), np.concatenate(baselines)


def _add_categories(
    train: pd.DataFrame,
    numeric_state: pd.DataFrame,
) -> pd.DataFrame:
    output = numeric_state.copy()
    for column in CATEGORICAL_COLUMNS:
        output[f"cat__{column}"] = (
            train[column].astype("string").fillna("__MISSING__").astype("category")
        )
    return output


def _model(*, leaves: int, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        verbosity=-1,
        n_jobs=6,
        n_estimators=180 if leaves == 7 else 150,
        learning_rate=0.025,
        num_leaves=leaves,
        max_depth=3 if leaves == 7 else 4,
        min_child_samples=800,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=3.0,
        reg_lambda=20.0,
        max_bin=127,
        random_state=seed,
    )


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    numeric_state, baseline = _state_features(train)
    features = _add_categories(train, numeric_state)
    categorical = [column for column in features if column.startswith("cat__")]
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for audit_year in (2023, 2024):
        print(f"[state] audit_year={audit_year}", flush=True)
        fit_mask = train["season"].lt(audit_year).to_numpy()
        audit_mask = train["season"].eq(audit_year).to_numpy()
        fit_season = train.loc[fit_mask, "season"].to_numpy(np.float64)
        target = train.loc[fit_mask, "control_success"].to_numpy(np.float64)
        audit = train.loc[audit_mask].reset_index(drop=True)
        audit_target = audit["control_success"].to_numpy(np.float64)
        artifact = np.load(
            project
            / "artifacts"
            / "v16_multiseason_20260815_02"
            / f"{V17_NAME}_o{audit_year}.npz"
        )
        incumbent = artifact["candidate"].astype(np.float64)
        if not np.array_equal(audit_target, artifact["target"].astype(np.float64)):
            raise ValueError(f"v17 order mismatch for {audit_year}")
        for half_life in (0.5, 1.0, 2.0, 4.0):
            sample_weight = np.exp2(-(audit_year - 1.0 - fit_season) / half_life)
            sample_weight /= sample_weight.mean()
            for leaves in (7, 15):
                print(
                    f"[state] audit_year={audit_year} half_life={half_life:g} leaves={leaves}",
                    flush=True,
                )
                model = _model(leaves=leaves, seed=4100 + audit_year + leaves)
                model.fit(
                    features.loc[fit_mask],
                    target - baseline[fit_mask],
                    sample_weight=sample_weight,
                    categorical_feature=categorical,
                )
                raw = np.clip(
                    baseline[audit_mask] + model.predict(features.loc[audit_mask]),
                    0.001,
                    0.999,
                )
                del model
                gc.collect()
                domains = {
                    "ALL": np.ones(len(audit), dtype=bool),
                    "R_CORE": audit["domain3"].eq("R_CORE").to_numpy(),
                    "R_ANCHOR": audit["domain3"].eq("R_ANCHOR").to_numpy(),
                    "F": audit["domain3"].eq("F").to_numpy(),
                }
                for domain, apply_mask in domains.items():
                    for weight in (0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35):
                        candidate = incumbent.copy()
                        candidate[apply_mask] = np.clip(
                            incumbent[apply_mask]
                            + weight * (raw[apply_mask] - incumbent[apply_mask]),
                            0.001,
                            0.999,
                        )
                        diagnostic = _diagnostics(
                            audit, audit_target, incumbent, candidate
                        )
                        metric_rows.append(
                            {
                                "audit_year": audit_year,
                                "candidate": f"state_h{half_life:g}_l{leaves}_{domain}",
                                "half_life": half_life,
                                "leaves": leaves,
                                "domain": domain,
                                "weight": weight,
                                "gain_vs_v17": diagnostic["gain"],
                                "month_positive_fraction": float(
                                    np.mean([row["gain"] > 0 for row in diagnostic["months"]])
                                ),
                                "worst_month_gain": float(
                                    min(row["gain"] for row in diagnostic["months"])
                                ),
                                "minimum_domain_gain": float(
                                    min(row["gain"] for row in diagnostic["domains"])
                                ),
                            }
                        )
                fold_rows.append(
                    {
                        "audit_year": audit_year,
                        "half_life": half_life,
                        "leaves": leaves,
                        "raw_mean": float(raw.mean()),
                        "raw_sd": float(raw.std()),
                        "target_mean": float(audit_target.mean()),
                        "v17_mean": float(incumbent.mean()),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "folds.csv", index=False)
    robust = (
        metrics.groupby(["candidate", "weight"], observed=True)["gain_vs_v17"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False)
    )
    robust.to_csv(output_dir / "robust.csv", index=False)
    summary = {
        "protocol": "MULTI_YEAR_ROW_LOCAL_STATE_RESIDUAL_FORWARD_V1",
        "audit_years": [2023, 2024],
        "robust": robust.head(50).to_dict(orient="records"),
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
        default=Path("artifacts/multi_year_state_20260816_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
