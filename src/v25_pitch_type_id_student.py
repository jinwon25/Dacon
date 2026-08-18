"""Pitcher-aware latent pitch-type classifier and nested v22 audit."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.latent_pitch_type_state_model import reconstruct_current_pitch_type
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v24_semantic_signal_screen import _v22
from src.v25_failure_profile_nested_screen import (
    DIRECTIONS,
    WEIGHTS,
    _champion_axis,
    compose,
    diagnostics,
)
from src.v25_pitch_type_eb_student import _normalise, _temperature


NUMERIC = (
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "li",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
)
BASE_CATEGORICAL = (
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "game_type",
    "domain3",
    "base_state",
    "top_bottom",
)
POWERS = (0.75, 1.0, 1.25)
ORIGINAL_FRACTIONS = (0.0, 0.25, 0.50, 0.75)


def feature_frame(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = rows.loc[:, list(NUMERIC)].apply(pd.to_numeric, errors="coerce").copy()
    mix = output[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].fillna(0.0).to_numpy(np.float64)
    mix = np.clip(mix, 1e-6, 1.0)
    mix /= mix.sum(axis=1, keepdims=True)
    output["mix_entropy"] = -np.sum(mix * np.log(mix), axis=1)
    output["mix_max"] = np.max(mix, axis=1)
    output["pitchmix_log_n"] = np.log1p(
        pd.to_numeric(rows["asof_pitcher_pitchmix_n"], errors="coerce").fillna(0.0)
    )

    categories: dict[str, pd.Series] = {}
    for column in BASE_CATEGORICAL:
        categories[f"cat__{column}"] = rows[column].astype("string").fillna("__NA__")
    pitcher = rows["pitcher_id"].astype("string")
    balls = rows["balls_before"].astype("string")
    strikes = rows["strikes_before"].astype("string")
    batter_hand = rows["batter_hand"].astype("string")
    categories["cat__pitcher_count"] = pitcher + "|" + balls + "|" + strikes
    categories["cat__pitcher_count_hand"] = (
        pitcher + "|" + balls + "|" + strikes + "|" + batter_hand
    )
    categories["cat__pitcher_batter_hand"] = pitcher + "|" + batter_hand
    categories["cat__team_count_hand"] = (
        rows["pitcher_team_id"].astype("string")
        + "|"
        + balls
        + "|"
        + strikes
        + "|"
        + rows["pitcher_hand"].astype("string")
        + "|"
        + batter_hand
    )
    categorical_columns = list(categories)
    for name, values in categories.items():
        output[name] = values.astype("category")
    return output, categorical_columns


def _model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        verbosity=-1,
        n_jobs=6,
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=300,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=2.0,
        reg_lambda=18.0,
        max_bin=127,
        cat_smooth=50.0,
        random_state=seed,
    )


def _load_latent(project: Path, year: int) -> dict[str, np.ndarray]:
    with np.load(
        project
        / "artifacts/latent_pitch_type_state_20260816_01"
        / f"latent_pitch_type_o{year}.npz",
        allow_pickle=True,
    ) as saved:
        return {key: saved[key] for key in saved.files}


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data/train.csv", low_memory=False)
    )
    label = reconstruct_current_pitch_type(train)
    features, categorical = feature_frame(train)
    latent = {year: _load_latent(project, year) for year in (2023, 2024)}
    id_probability: dict[int, np.ndarray] = {}
    classifier_metrics = []
    for year in (2023, 2024):
        fit = train["season"].lt(year).to_numpy() & (label >= 0)
        audit = train["season"].eq(year).to_numpy()
        fit_season = train.loc[fit, "season"].to_numpy(np.float64)
        weight = np.exp2(-(year - 1.0 - fit_season) / 0.5)
        weight /= weight.mean()
        model = _model(seed=25200 + year)
        model.fit(
            features.loc[fit],
            label[fit].astype(np.int64),
            sample_weight=weight,
            categorical_feature=categorical,
        )
        probability = _normalise(model.predict_proba(features.loc[audit]))
        id_probability[year] = probability
        known = label[audit] >= 0
        audit_label = label[audit]
        classifier_metrics.append(
            {
                "audit_year": year,
                "accuracy": float(
                    np.mean(np.argmax(probability[known], axis=1) == audit_label[known])
                ),
                "logloss": float(
                    -np.mean(
                        np.log(
                            np.clip(
                                probability[np.flatnonzero(known), audit_label[known]],
                                1e-12,
                                1.0,
                            )
                        )
                    )
                ),
                "original_accuracy": float(
                    np.mean(
                        np.argmax(latent[year]["type_probability"][known], axis=1)
                        == audit_label[known]
                    )
                ),
            }
        )
        del model
        gc.collect()

    probability_bank: dict[int, dict[str, np.ndarray]] = {2023: {}, 2024: {}}
    raw_bank: dict[int, dict[str, np.ndarray]] = {2023: {}, 2024: {}}
    for year in (2023, 2024):
        original = _normalise(latent[year]["type_probability"].astype(np.float64))
        raw_by_type = latent[year]["raw_by_type"].astype(np.float64)
        for fraction in ORIGINAL_FRACTIONS:
            mixed = _normalise(
                (1.0 - fraction) * id_probability[year] + fraction * original
            )
            for power in POWERS:
                name = f"id_lgb__orig{fraction:g}__pow{power:g}"
                probability_bank[year][name] = _temperature(mixed, power)
                raw_bank[year][name] = np.sum(
                    probability_bank[year][name] * raw_by_type, axis=1
                )
    names = sorted(set(raw_bank[2023]) & set(raw_bank[2024]))

    axis23 = _champion_axis(project, "y2023_early_to_late")
    month23 = latent[2023]["game_month"].astype(np.int16)
    late23 = month23 >= 8
    rows23 = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8)
    ].reset_index(drop=True)
    v21_23 = axis23["v21"].astype(np.float64)
    domain23 = axis23["domain3"].astype(str)
    v22_23 = _v22(rows23, v21_23, domain23)
    selection_rows = []
    for name in names:
        for direction in DIRECTIONS:
            for blend_weight in WEIGHTS:
                candidate = compose(
                    v21_23,
                    v22_23,
                    raw_bank[2023][name][late23],
                    blend_weight,
                    direction,
                )
                result = diagnostics(
                    latent[2023]["target"].astype(np.float64)[late23],
                    v22_23,
                    candidate,
                    month23[late23],
                    domain23,
                )
                selection_rows.append(
                    {
                        "candidate": name,
                        "direction": direction,
                        "weight": blend_weight,
                        "gain": result["gain"],
                        "positive_month_fraction": result["positive_month_fraction"],
                        "worst_month_gain": result["worst_month_gain"],
                        "minimum_domain_gain": result["minimum_domain_gain"],
                        "selection_score": min(
                            result["gain"],
                            result["worst_month_gain"],
                            result["minimum_domain_gain"],
                        ),
                    }
                )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    selection.to_csv(output_dir / "selection.csv", index=False)
    selected = selection.iloc[0].to_dict()
    selection_gate = bool(
        selected["selection_score"] > 0.0
        and selected["positive_month_fraction"] >= 1.0
    )
    summary: dict[str, object] = {
        "protocol": "V25_PITCHER_ID_LATENT_TYPE_STUDENT_V1",
        "classifier_metrics": classifier_metrics,
        "selected": selected,
        "selection_gate_passed": selection_gate,
        "outer_audit_run": False,
        "eligible_for_packaging": False,
    }
    if selection_gate:
        axis24 = _champion_axis(project, "y2023_to_y2024")
        rows24 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
        v21_24 = axis24["v21"].astype(np.float64)
        domain24 = axis24["domain3"].astype(str)
        v22_24 = _v22(rows24, v21_24, domain24)
        raw24 = raw_bank[2024][str(selected["candidate"])]
        candidate24 = compose(
            v21_24,
            v22_24,
            raw24,
            float(selected["weight"]),
            str(selected["direction"]),
        )
        target24 = latent[2024]["target"].astype(np.float64)
        month24 = latent[2024]["game_month"].astype(np.int16)
        outer = diagnostics(target24, v22_24, candidate24, month24, domain24)
        late24 = month24 >= 8
        late = diagnostics(
            target24[late24],
            v22_24[late24],
            candidate24[late24],
            month24[late24],
            domain24[late24],
        )
        eligible = bool(
            outer["gain"] >= 5.0
            and outer["positive_month_fraction"] >= 0.75
            and outer["minimum_domain_gain"] > 0.0
            and outer["worst_month_gain"] > -10.0
            and late["gain"] > 0.0
            and late["minimum_domain_gain"] > 0.0
        )
        summary.update(
            {
                "outer_audit_run": True,
                "outer_2024": outer,
                "late_2024": late,
                "eligible_for_packaging": eligible,
            }
        )
        np.savez_compressed(
            output_dir / "outer_prediction.npz",
            target=target24,
            v21=v21_24,
            v22=v22_24,
            raw=raw24,
            candidate=candidate24,
            game_month=month24,
            domain3=domain24,
        )
    np.savez_compressed(
        output_dir / "type_probability.npz",
        probability_2023=id_probability[2023],
        probability_2024=id_probability[2024],
    )
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
        default=Path("artifacts/v25_pitch_type_id_student_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
