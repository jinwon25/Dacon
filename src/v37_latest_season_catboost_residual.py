"""Latest-season nested CatBoost residual pilot above v27.

March-May 2024 fits two preregistered feature variants.  June-July selects the
variant, deployment domain, and conservative blend weight.  The recipe is
then refitted on March-July with three fixed seeds and evaluated once on
August-October.  Every feature is row-local or derived from history strictly
before 2024; no audit-row aggregate is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.recent_shared_exact_asof import (
    NUMERIC_COLUMNS,
    PITCHER_COMPONENTS,
    TARGET,
    _current_season_state,
    _make_season_bank,
    _row_state,
)
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v30_diverse_covariance_screen import (
    diagnostics,
    v27_parent,
)
from src.v23_structural_residual_screen import _load_axis
from src.v25_postbreak_anchor_audit import _early_to_late_2024


VARIANTS = ("context", "player_ids")
DOMAINS = ("ALL", "R_CORE", "R_ANCHOR")
ETAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
SELECTION_SEED = 37
REFIT_SEEDS = (37, 137, 237)
CORRECTION_CLIP = 0.05
BASE_CATEGORICAL = (
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "domain3",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "count_state",
    "platoon",
    "team_matchup",
)
PLAYER_CATEGORICAL = (
    "pitcher_id",
    "batter_id",
    "pitcher_count",
    "pitcher_batter_hand",
    "batter_count",
)


def categorical_columns(variant: str) -> list[str]:
    if variant == "context":
        return list(BASE_CATEGORICAL)
    if variant == "player_ids":
        return [*BASE_CATEGORICAL, *PLAYER_CATEGORICAL]
    raise ValueError(f"unknown feature variant: {variant}")


def _cat_frame(
    numeric: pd.DataFrame,
    raw: pd.DataFrame,
    base: np.ndarray,
    variant: str,
) -> pd.DataFrame:
    output = numeric.reset_index(drop=True).copy()
    count = raw["balls_before"].astype(str) + "-" + raw["strikes_before"].astype(str)
    values = {
        "game_dayofweek": raw["game_dayofweek"],
        "top_bottom": raw["top_bottom"],
        "game_type": raw["game_type"],
        "domain3": raw["domain3"],
        "base_state": raw["base_state"],
        "pitcher_hand": raw["pitcher_hand"],
        "batter_hand": raw["batter_hand"],
        "pitcher_team_id": raw["pitcher_team_id"],
        "batter_team_id": raw["batter_team_id"],
        "count_state": count,
        "platoon": raw["pitcher_hand"].astype(str)
        + "-"
        + raw["batter_hand"].astype(str),
        "team_matchup": raw["pitcher_team_id"].astype(str)
        + "-"
        + raw["batter_team_id"].astype(str),
        "pitcher_id": raw["pitcher_id"],
        "batter_id": raw["batter_id"],
        "pitcher_count": raw["pitcher_id"].astype(str) + "-" + count,
        "pitcher_batter_hand": raw["pitcher_id"].astype(str)
        + "-"
        + raw["batter_hand"].astype(str),
        "batter_count": raw["batter_id"].astype(str) + "-" + count,
    }
    for column in categorical_columns(variant):
        output[column] = (
            values[column].astype("string").fillna("__MISSING__").astype(str).to_numpy()
        )
    output["frozen_base_probability"] = np.asarray(base, dtype=np.float64)
    return output


def _model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=280,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=30.0,
        random_strength=0.5,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        random_seed=seed,
        thread_count=6,
        allow_writing_files=False,
        verbose=100,
    )


def _recency_weight(frame: pd.DataFrame, last_month: int) -> np.ndarray:
    month = frame["game_month"].to_numpy(np.float64)
    weight = np.exp2(-(float(last_month) - month) / 2.0)
    return weight / weight.mean()


def fit_predict(
    fit_features: pd.DataFrame,
    audit_features: pd.DataFrame,
    fit_target: np.ndarray,
    fit_base: np.ndarray,
    sample_weight: np.ndarray,
    *,
    variant: str,
    seed: int,
    model_path: Path | None = None,
) -> np.ndarray:
    categories = categorical_columns(variant)
    fit_pool = Pool(
        fit_features,
        label=np.asarray(fit_target, dtype=np.float64)
        - np.asarray(fit_base, dtype=np.float64),
        weight=np.asarray(sample_weight, dtype=np.float64),
        cat_features=categories,
    )
    audit_pool = Pool(audit_features, cat_features=categories)
    model = _model(seed)
    model.fit(fit_pool)
    correction = np.clip(
        model.predict(audit_pool), -CORRECTION_CLIP, CORRECTION_CLIP
    ).astype(np.float64)
    if model_path is not None:
        model.save_model(str(model_path))
    return correction


def apply_correction(
    frame: pd.DataFrame,
    correction: np.ndarray,
    *,
    eta: float,
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    mask = (
        np.ones(len(frame), dtype=bool)
        if domain == "ALL"
        else frame["domain3"].astype(str).eq(domain).to_numpy()
    )
    candidate = parent.copy()
    candidate[mask] = np.clip(
        parent[mask] + float(eta) * correction[mask], 0.001, 0.999
    )
    return candidate, mask


def _numeric_2024(raw24: pd.DataFrame, bank: dict[str, object]) -> pd.DataFrame:
    numeric = [column for column in NUMERIC_COLUMNS if column in raw24]
    return pd.concat(
        [
            raw24[numeric].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            _row_state(raw24),
            _current_season_state(raw24, bank),
        ],
        axis=1,
    ).astype(np.float32)


def _load_2024_and_bank_source(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bank_columns = [
        "season",
        TARGET,
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        *PITCHER_COMPONENTS,
    ]
    history_parts = []
    year_parts = []
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        history = chunk.loc[chunk["season"].lt(2024), bank_columns]
        if len(history):
            history_parts.append(history.copy())
        year = chunk.loc[chunk["season"].eq(2024)]
        if len(year):
            year_parts.append(year.copy())
    if not history_parts or not year_parts:
        raise ValueError("train.csv does not contain both pre-2024 history and 2024")
    return (
        pd.concat(year_parts, ignore_index=True),
        pd.concat(history_parts, ignore_index=True),
    )


def _attach_v25(
    project: Path, frame: pd.DataFrame, cache_name: str
) -> pd.DataFrame:
    direct = np.load(
        project
        / "artifacts"
        / "v29_anchor_route_20260817_01"
        / f"{cache_name}_direct.npy"
    ).astype(np.float64)
    if len(direct) != len(frame):
        raise ValueError(f"v25 direct cache row mismatch: {cache_name}")
    output = frame.copy()
    v22 = output["v22"].to_numpy(np.float64)
    anchor = output["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
    v25 = v22.copy()
    v25[anchor] = np.clip(
        v22[anchor] + 0.075 * (direct[anchor] - v22[anchor]), 0.001, 0.999
    )
    output["v25"] = v25
    return output


def _prepare_features(
    numeric24: pd.DataFrame,
    fit_mask: np.ndarray,
    audit_mask: np.ndarray,
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    fit_base: np.ndarray,
    audit_base: np.ndarray,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_fit = numeric24.loc[fit_mask].reset_index(drop=True)
    numeric_audit = numeric24.loc[audit_mask].reset_index(drop=True)
    return (
        _cat_frame(numeric_fit, fit, fit_base, variant),
        _cat_frame(numeric_audit, audit, audit_base, variant),
    )


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw24_unprepared, bank_source = _load_2024_and_bank_source(
        project / "data" / "train.csv"
    )
    raw24 = _add_domain_and_pressure(raw24_unprepared)
    del raw24_unprepared
    season_bank = _make_season_bank(bank_source, 2024)
    del bank_source
    numeric24 = _numeric_2024(raw24, season_bank)
    del season_bank
    full_axis = _attach_v25(
        project,
        _load_axis(project, "y2023_to_y2024", raw24),
        "outer_full_2024",
    )
    _, replication = _early_to_late_2024(project, raw24)
    replication = _attach_v25(project, replication, "replication_late_2024")
    if not np.array_equal(
        raw24["control_success"].to_numpy(np.float64),
        full_axis["target"].to_numpy(np.float64),
    ):
        raise ValueError("full-2024 target order mismatch")

    fit_mask = raw24["game_month"].le(5).to_numpy()
    selection_mask = raw24["game_month"].isin((6, 7)).to_numpy()
    fit = raw24.loc[fit_mask].reset_index(drop=True)
    selection_raw = raw24.loc[selection_mask].reset_index(drop=True)
    selection = full_axis.loc[selection_mask].reset_index(drop=True)
    full_parent = v27_parent(full_axis)
    fit_base = full_parent[fit_mask]
    selection_base = full_parent[selection_mask]
    selection_predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        print(f"[v37] selection fit variant={variant}", flush=True)
        fit_features, selection_features = _prepare_features(
            numeric24,
            fit_mask,
            selection_mask,
            fit,
            selection_raw,
            fit_base,
            selection_base,
            variant,
        )
        correction = fit_predict(
            fit_features,
            selection_features,
            fit["control_success"].to_numpy(np.float64),
            fit_base,
            _recency_weight(fit, 5),
            variant=variant,
            seed=SELECTION_SEED,
            model_path=output_dir / f"selection_{variant}.cbm",
        )
        selection_predictions[variant] = correction
        for domain in DOMAINS:
            for eta in ETAS:
                candidate, mask = apply_correction(
                    selection, correction, eta=eta, domain=domain
                )
                result = diagnostics(selection, selection_base, candidate, mask)
                applied_gain = (
                    min(result["domain_gains"].values())
                    if domain == "ALL"
                    else result["domain_gains"][domain]
                )
                rows.append(
                    {
                        "variant": variant,
                        "domain": domain,
                        "eta": eta,
                        "applied_domain_gain": float(applied_gain),
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"months", "domain_gains"}
                        },
                    }
                )
        del fit_features, selection_features
    metrics = pd.DataFrame(rows)
    metrics["selection_score"] = metrics[
        ["gain", "worst_month_gain", "applied_domain_gain"]
    ].min(axis=1)
    metrics["passes_selection_gate"] = (
        metrics["gain"].gt(0.0)
        & metrics["positive_month_fraction"].eq(1.0)
        & metrics["worst_month_gain"].gt(0.0)
        & metrics["applied_domain_gain"].gt(0.0)
        & metrics["minimum_domain_gain"].gt(-5.0)
    )
    metrics = metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = metrics.loc[metrics["passes_selection_gate"]]
    selected = passing.iloc[0] if len(passing) else metrics.iloc[0]
    recipe = {
        "variant": str(selected["variant"]),
        "domain": str(selected["domain"]),
        "eta": float(selected["eta"]),
    }

    refit_mask = raw24["game_month"].le(7).to_numpy()
    audit_mask = raw24["game_month"].ge(8).to_numpy()
    refit = raw24.loc[refit_mask].reset_index(drop=True)
    audit_raw = raw24.loc[audit_mask].reset_index(drop=True)
    audit = replication
    if not np.array_equal(
        audit_raw["control_success"].to_numpy(np.float64),
        audit["target"].to_numpy(np.float64),
    ):
        raise ValueError("late-2024 target order mismatch")
    refit_base = full_parent[refit_mask]
    audit_base = v27_parent(audit)
    refit_features, audit_features = _prepare_features(
        numeric24,
        refit_mask,
        audit_mask,
        refit,
        audit_raw,
        refit_base,
        audit_base,
        recipe["variant"],
    )
    seed_rows = []
    corrections = []
    for seed in REFIT_SEEDS:
        print(f"[v37] refit seed={seed}", flush=True)
        correction = fit_predict(
            refit_features,
            audit_features,
            refit["control_success"].to_numpy(np.float64),
            refit_base,
            _recency_weight(refit, 7),
            variant=recipe["variant"],
            seed=seed,
            model_path=output_dir / f"refit_{recipe['variant']}_s{seed}.cbm",
        )
        corrections.append(correction)
        candidate, mask = apply_correction(
            audit, correction, eta=recipe["eta"], domain=recipe["domain"]
        )
        result = diagnostics(audit, audit_base, candidate, mask)
        seed_rows.append(
            {
                "seed": seed,
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"months", "domain_gains"}
                },
            }
        )
    pd.DataFrame(seed_rows).to_csv(output_dir / "seed_audits.csv", index=False)
    mean_correction = np.mean(np.vstack(corrections), axis=0)
    candidate, active = apply_correction(
        audit,
        mean_correction,
        eta=recipe["eta"],
        domain=recipe["domain"],
    )
    audit_result = diagnostics(audit, audit_base, candidate, active)
    gates = {
        "selection_gate": bool(selected["passes_selection_gate"]),
        "audit_gain_at_least_3": audit_result["gain"] >= 3.0,
        "audit_all_months_positive": audit_result["positive_month_fraction"] == 1.0,
        "audit_worst_month_positive": audit_result["worst_month_gain"] > 0.0,
        "audit_minimum_domain_above_minus_5": audit_result[
            "minimum_domain_gain"
        ]
        > -5.0,
        "all_seed_gains_positive": all(row["gain"] > 0.0 for row in seed_rows),
    }
    summary = {
        "protocol": "V37_LATEST_SEASON_NESTED_CATBOOST_RESIDUAL_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "fit_axis": "2024 March-May",
        "selection_axis": "2024 June-July",
        "audit_axis": "2024 August-October",
        "selection_candidate_count": int(len(metrics)),
        "selection_gate_count": int(metrics["passes_selection_gate"].sum()),
        "chosen": recipe,
        "selection": {
            key: selected[key]
            for key in (
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "minimum_domain_gain",
                "applied_domain_gain",
            )
        },
        "seed_audits": seed_rows,
        "audit": audit_result,
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
        "latest_year_repeated_development_risk": True,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "audit_late_2024.npz",
        target=audit["target"].to_numpy(np.float64),
        v27=audit_base,
        correction=mean_correction,
        candidate=candidate,
        active=active,
        domain3=audit["domain3"].astype(str).to_numpy(),
        game_month=audit["game_month"].to_numpy(np.int16),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v37_latest_catboost_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
