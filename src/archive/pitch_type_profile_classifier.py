"""Previous-season hierarchical pitch-choice profiles and classifier screen."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.latent_pitch_type_state_model import (
    RATE_COLUMNS,
    TYPE_NAMES,
    reconstruct_current_pitch_type,
)
from src.multi_year_state_model import _add_categories, _state_features
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.trackman_privileged_distillation import _diagnostics, bss


PROFILE_SPECS = (
    ("pitcher", ("pitcher_id",), 100.0, None),
    ("count_hand", ("balls_before", "strikes_before", "batter_hand"), 1000.0, None),
    ("pitcher_count", ("pitcher_id", "balls_before", "strikes_before"), 35.0, "pitcher"),
    ("pitcher_hand", ("pitcher_id", "batter_hand"), 50.0, "pitcher"),
    (
        "pitcher_count_hand",
        ("pitcher_id", "balls_before", "strikes_before", "batter_hand"),
        20.0,
        "pitcher_count",
    ),
    (
        "pitcher_domain_count",
        ("pitcher_id", "domain3", "balls_before", "strikes_before"),
        35.0,
        "pitcher_count",
    ),
)


def _group_probability(
    history: pd.DataFrame,
    history_label: np.ndarray,
    query: pd.DataFrame,
    keys: tuple[str, ...],
    prior: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    work = history.loc[:, keys].copy()
    work["__type"] = history_label
    counts = (
        work.groupby([*keys, "__type"], observed=True)
        .size()
        .unstack("__type", fill_value=0)
        .reindex(columns=range(3), fill_value=0)
        .reset_index()
    )
    count_columns = [f"__count_{index}" for index in range(3)]
    counts.columns = [*keys, *count_columns]
    mapped = query.loc[:, keys].merge(counts, on=list(keys), how="left", sort=False)
    raw_count = mapped.loc[:, count_columns].fillna(0.0).to_numpy(np.float64)
    total = raw_count.sum(axis=1)
    probability = (raw_count + alpha * prior) / (total[:, None] + alpha)
    return probability.astype(np.float32), total.astype(np.float32)


def previous_season_type_profiles(rows: pd.DataFrame, label: np.ndarray) -> pd.DataFrame:
    """Return cutoff-safe type probabilities for every row in ``rows``."""

    output = pd.DataFrame(index=rows.index)
    seasons = sorted(int(value) for value in rows["season"].unique())
    for season in seasons:
        print(f"[type-profile] feature season={season}", flush=True)
        query_mask = rows["season"].eq(season).to_numpy()
        history_mask = rows["season"].lt(season).to_numpy() & (label >= 0)
        query = rows.loc[query_mask].reset_index(drop=True)
        if history_mask.any():
            history = rows.loc[history_mask].reset_index(drop=True)
            history_label = label[history_mask]
            global_count = np.bincount(history_label, minlength=3).astype(np.float64)
            global_probability = (global_count + 1.0) / (global_count.sum() + 3.0)
        else:
            history = rows.iloc[0:0].copy()
            history_label = np.empty(0, dtype=np.int8)
            global_probability = np.full(3, 1.0 / 3.0, dtype=np.float64)
        probability_by_name: dict[str, np.ndarray] = {}
        for name, keys, alpha, parent in PROFILE_SPECS:
            if parent is None:
                prior = np.tile(global_probability, (len(query), 1))
            else:
                prior = probability_by_name[parent]
            probability, count = _group_probability(
                history, history_label, query, keys, prior, alpha
            )
            probability_by_name[name] = probability
            positions = np.flatnonzero(query_mask)
            for index, type_name in enumerate(TYPE_NAMES):
                output.loc[positions, f"typeprof__{name}__{type_name}"] = probability[
                    :, index
                ]
            output.loc[positions, f"typeprof__{name}__log_n"] = np.log1p(count)
    return output.astype(np.float32)


def _classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        verbosity=-1,
        n_jobs=6,
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=700,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=2.0,
        reg_lambda=18.0,
        max_bin=127,
        random_state=seed,
    )


def _normalize_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability.astype(np.float64), 1e-8, None)
    return probability / probability.sum(axis=1, keepdims=True)


def _temperature(probability: np.ndarray, power: float) -> np.ndarray:
    return _normalize_probability(np.power(np.clip(probability, 1e-8, 1.0), power))


def run(project: Path, latent_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    latent_dir = (project / latent_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    label = reconstruct_current_pitch_type(train)
    numeric_state, _ = _state_features(train)
    profile = previous_season_type_profiles(train, label)
    features = _add_categories(train, pd.concat([numeric_state, profile], axis=1))
    categorical = [column for column in features if column.startswith("cat__")]

    fold_rows: list[dict[str, object]] = []
    for audit_year in (2023, 2024):
        print(f"[type-profile] classifier audit_year={audit_year}", flush=True)
        # Earlier-season profiles for source rows are themselves cutoff-safe.
        fit_mask = (
            train["season"].between(2020, audit_year - 1).to_numpy() & (label >= 0)
        )
        audit_mask = train["season"].eq(audit_year).to_numpy()
        audit_label = label[audit_mask]
        predictions: dict[str, np.ndarray] = {}
        for half_life in (0.5, 2.0, 4.0):
            fit_season = train.loc[fit_mask, "season"].to_numpy(np.float64)
            weight = np.exp2(-(audit_year - 1.0 - fit_season) / half_life)
            weight /= weight.mean()
            model = _classifier(seed=6100 + audit_year + int(10 * half_life))
            model.fit(
                features.loc[fit_mask],
                label[fit_mask].astype(np.int64),
                sample_weight=weight,
                categorical_feature=categorical,
            )
            predictions[f"lgb_h{half_life:g}"] = model.predict_proba(
                features.loc[audit_mask]
            ).astype(np.float64)
            del model
            gc.collect()

        audit_profile = profile.loc[audit_mask].reset_index(drop=True)
        for profile_name in (
            "pitcher",
            "pitcher_count",
            "pitcher_hand",
            "pitcher_count_hand",
            "pitcher_domain_count",
        ):
            predictions[f"profile_{profile_name}"] = _normalize_probability(
                audit_profile[
                    [f"typeprof__{profile_name}__{name}" for name in TYPE_NAMES]
                ].to_numpy(np.float64)
            )
        current_mix = (
            train.loc[audit_mask, RATE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        fallback = predictions["profile_pitcher"]
        missing = ~np.isfinite(current_mix).all(axis=1) | (np.nansum(current_mix, axis=1) <= 0)
        current_mix[missing] = fallback[missing]
        predictions["current_season_mix"] = _normalize_probability(current_mix)
        predictions["mean_lgb"] = np.mean(
            np.stack([predictions[name] for name in ("lgb_h0.5", "lgb_h2", "lgb_h4")]),
            axis=0,
        )
        predictions["mean_lgb_profile"] = 0.75 * predictions["mean_lgb"] + 0.25 * predictions[
            "profile_pitcher_count_hand"
        ]

        with np.load(latent_dir / f"latent_pitch_type_o{audit_year}.npz", allow_pickle=True) as z:
            target = z["target"].astype(np.float64)
            incumbent = z["incumbent"].astype(np.float64)
            raw_by_type = z["raw_by_type"].astype(np.float64)
            domain3 = z["domain3"].astype(str)
            game_month = z["game_month"].astype(np.int16)
            pitcher_id = z["pitcher_id"]
        audit = train.loc[audit_mask].reset_index(drop=True)
        known = audit_label >= 0
        candidate_rows: list[dict[str, object]] = []
        cache: dict[str, np.ndarray] = {}
        for probability_name, probability in predictions.items():
            for power in (0.5, 1.0, 1.5, 2.0):
                transformed = _temperature(probability, power)
                name = f"{probability_name}_pow{power:g}"
                raw = np.sum(transformed * raw_by_type, axis=1)
                cache[name] = raw
                accuracy = float(
                    np.mean(np.argmax(transformed[known], axis=1) == audit_label[known])
                )
                logloss = float(
                    -np.mean(
                        np.log(
                            np.clip(
                                transformed[np.flatnonzero(known), audit_label[known]],
                                1e-12,
                                1.0,
                            )
                        )
                    )
                )
                for blend_weight in (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0):
                    candidate = np.clip(
                        incumbent + blend_weight * (raw - incumbent), 0.001, 0.999
                    )
                    diagnostic = _diagnostics(audit, target, incumbent, candidate)
                    candidate_rows.append(
                        {
                            "audit_year": audit_year,
                            "probability": name,
                            "blend_weight": blend_weight,
                            "type_accuracy": accuracy,
                            "type_logloss": logloss,
                            "gain": diagnostic["gain"],
                            "month_positive_fraction": float(
                                np.mean([row["gain"] > 0 for row in diagnostic["months"]])
                            ),
                            "worst_month_gain": float(
                                min(row["gain"] for row in diagnostic["months"])
                            ),
                        }
                    )
        candidates = pd.DataFrame(candidate_rows).sort_values("gain", ascending=False)
        candidates.to_csv(output_dir / f"metrics_o{audit_year}.csv", index=False)
        cache_names = list(cache)
        np.savez_compressed(
            output_dir / f"pitch_type_profile_o{audit_year}.npz",
            target=target,
            incumbent=incumbent,
            domain3=domain3,
            game_month=game_month,
            pitcher_id=pitcher_id,
            names=np.asarray(cache_names, dtype=object),
            raw=np.column_stack([cache[name] for name in cache_names]),
        )
        fold_row = {
            "audit_year": audit_year,
            "best": candidates.head(30).to_dict(orient="records"),
        }
        fold_rows.append(fold_row)
        print(json.dumps(fold_row, ensure_ascii=False, indent=2), flush=True)

    metrics = pd.concat(
        [pd.read_csv(output_dir / f"metrics_o{year}.csv") for year in (2023, 2024)],
        ignore_index=True,
    )
    robust = (
        metrics.groupby(["probability", "blend_weight"], observed=True)["gain"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False)
    )
    robust.to_csv(output_dir / "robust.csv", index=False)
    summary = {
        "protocol": "PREVIOUS_SEASON_HIERARCHICAL_PITCH_TYPE_CLASSIFIER_V1",
        "folds": fold_rows,
        "robust": robust.head(50).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--latent-dir",
        type=Path,
        default=Path("artifacts/latent_pitch_type_state_20260816_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pitch_type_profile_20260816_01"),
    )
    args = parser.parse_args()
    run(args.project, args.latent_dir, args.output_dir)


if __name__ == "__main__":
    main()
