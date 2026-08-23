"""Empirical-Bayes latent pitch-type student above v22.

The current pitch type is reconstructed only for labelled training rows. At
audit time, type probabilities use prior seasons plus the current row's legal
ASOF pitch mix, count, hands, pitcher and team. Conditional success by latent
type comes from the already forward-trained outcome model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.latent_pitch_type_state_model import TYPE_NAMES, reconstruct_current_pitch_type
from src.champion.v24_semantic_signal_screen import _v22
from src.champion.v25_failure_profile_nested_screen import (
    DIRECTIONS,
    WEIGHTS,
    _champion_axis,
    compose,
    diagnostics,
)


POWERS = (0.75, 1.0, 1.25)
LGB_FRACTIONS = (0.0, 0.25, 0.50, 0.75)


def _normalise(probability: np.ndarray) -> np.ndarray:
    output = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, None)
    return output / output.sum(axis=1, keepdims=True)


def _temperature(probability: np.ndarray, power: float) -> np.ndarray:
    return _normalise(np.power(np.clip(probability, 1e-8, 1.0), power))


def _group_probability(
    history: pd.DataFrame,
    label: np.ndarray,
    query: pd.DataFrame,
    keys: tuple[str, ...],
    prior: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    work = history.loc[:, list(keys)].copy()
    work["__type"] = label
    counts = (
        work.groupby([*keys, "__type"], observed=True)
        .size()
        .unstack("__type", fill_value=0)
        .reindex(columns=range(len(TYPE_NAMES)), fill_value=0)
        .reset_index()
    )
    columns = [f"__count_{index}" for index in range(len(TYPE_NAMES))]
    counts.columns = [*keys, *columns]
    mapped = query.loc[:, list(keys)].merge(counts, on=list(keys), how="left", sort=False)
    raw = mapped[columns].fillna(0.0).to_numpy(np.float64)
    n = raw.sum(axis=1)
    probability = (raw + alpha * prior) / (n[:, None] + alpha)
    return _normalise(probability), n


def _row_mix(query: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    values = query[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    values = np.nan_to_num(values, nan=0.0)
    total = values.sum(axis=1)
    output = np.divide(
        values,
        total[:, None],
        out=np.tile(fallback, (len(query), 1)),
        where=total[:, None] > 0.0,
    )
    return _normalise(output)


def probability_bank(
    train: pd.DataFrame,
    label: np.ndarray,
    audit_year: int,
    lgb_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    history_mask = train["season"].lt(audit_year).to_numpy() & (label >= 0)
    audit_mask = train["season"].eq(audit_year).to_numpy()
    history = train.loc[history_mask].reset_index(drop=True)
    query = train.loc[audit_mask].reset_index(drop=True)
    history_label = label[history_mask]
    count = np.bincount(history_label, minlength=len(TYPE_NAMES)).astype(np.float64)
    global_probability = (count + 1.0) / (count.sum() + len(TYPE_NAMES))
    global_prior = np.tile(global_probability, (len(query), 1))
    row_mix = _row_mix(query, global_probability)
    pitcher, _ = _group_probability(
        history, history_label, query, ("pitcher_id",), row_mix, 120.0
    )
    count_hands, _ = _group_probability(
        history,
        history_label,
        query,
        ("balls_before", "strikes_before", "pitcher_hand", "batter_hand"),
        global_prior,
        1200.0,
    )
    pitcher_count, _ = _group_probability(
        history,
        history_label,
        query,
        ("pitcher_id", "balls_before", "strikes_before"),
        pitcher,
        50.0,
    )
    pitcher_count_hand, _ = _group_probability(
        history,
        history_label,
        query,
        ("pitcher_id", "balls_before", "strikes_before", "batter_hand"),
        pitcher_count,
        30.0,
    )
    team_count_hands, _ = _group_probability(
        history,
        history_label,
        query,
        (
            "pitcher_team_id",
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "batter_hand",
        ),
        count_hands,
        400.0,
    )
    bases = {
        "row_mix": row_mix,
        "pitcher": pitcher,
        "count_hands": count_hands,
        "pitcher_count": pitcher_count,
        "pitcher_count_hand": pitcher_count_hand,
        "team_count_hands": team_count_hands,
    }
    output: dict[str, np.ndarray] = {}
    lgb_probability = _normalise(lgb_probability)
    for name, profile in bases.items():
        for fraction in LGB_FRACTIONS:
            mixed = _normalise(
                fraction * lgb_probability + (1.0 - fraction) * profile
            )
            for power in POWERS:
                output[
                    f"{name}__lgb{fraction:g}__pow{power:g}"
                ] = _temperature(mixed, power)
    output["lgb_original"] = lgb_probability
    return output


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
    train = pd.read_csv(project / "data/train.csv", low_memory=False)
    label = reconstruct_current_pitch_type(train)
    latent23 = _load_latent(project, 2023)
    latent24 = _load_latent(project, 2024)
    probabilities23 = probability_bank(
        train, label, 2023, latent23["type_probability"].astype(np.float64)
    )
    probabilities24 = probability_bank(
        train, label, 2024, latent24["type_probability"].astype(np.float64)
    )
    names = sorted(set(probabilities23) & set(probabilities24))
    raw23 = {
        name: np.sum(
            probabilities23[name] * latent23["raw_by_type"].astype(np.float64), axis=1
        )
        for name in names
    }
    raw24 = {
        name: np.sum(
            probabilities24[name] * latent24["raw_by_type"].astype(np.float64), axis=1
        )
        for name in names
    }

    known23 = latent23["type_label"].astype(np.int8) >= 0
    known24 = latent24["type_label"].astype(np.int8) >= 0
    type_metrics = []
    for name in names:
        for year, probability, latent, known in (
            (2023, probabilities23[name], latent23, known23),
            (2024, probabilities24[name], latent24, known24),
        ):
            target_type = latent["type_label"].astype(np.int8)
            type_metrics.append(
                {
                    "candidate": name,
                    "audit_year": year,
                    "accuracy": float(
                        np.mean(np.argmax(probability[known], axis=1) == target_type[known])
                    ),
                    "logloss": float(
                        -np.mean(
                            np.log(
                                np.clip(
                                    probability[np.flatnonzero(known), target_type[known]],
                                    1e-12,
                                    1.0,
                                )
                            )
                        )
                    ),
                }
            )
    pd.DataFrame(type_metrics).to_csv(output_dir / "type_metrics.csv", index=False)

    axis23 = _champion_axis(project, "y2023_early_to_late")
    late23 = latent23["game_month"].astype(np.int16) >= 8
    rows23 = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8)
    ].reset_index(drop=True)
    v21_23 = axis23["v21"].astype(np.float64)
    domain23 = axis23["domain3"].astype(str)
    v22_23 = _v22(rows23, v21_23, domain23)
    selection_rows = []
    for name in names:
        for direction in DIRECTIONS:
            for weight in WEIGHTS:
                candidate = compose(
                    v21_23,
                    v22_23,
                    raw23[name][late23],
                    weight,
                    direction,
                )
                result = diagnostics(
                    latent23["target"].astype(np.float64)[late23],
                    v22_23,
                    candidate,
                    latent23["game_month"].astype(np.int16)[late23],
                    domain23,
                )
                selection_rows.append(
                    {
                        "candidate": name,
                        "direction": direction,
                        "weight": weight,
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
    table = pd.DataFrame(selection_rows).sort_values(
        ["selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    table.to_csv(output_dir / "selection.csv", index=False)
    selected = table.iloc[0].to_dict()
    selection_gate = bool(
        selected["selection_score"] > 0.0
        and selected["positive_month_fraction"] >= 1.0
    )
    summary: dict[str, object] = {
        "protocol": "V25_PITCH_TYPE_EB_STUDENT_NESTED_V1",
        "type_candidate_count": len(names),
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
        raw_selected = raw24[str(selected["candidate"])]
        candidate24 = compose(
            v21_24,
            v22_24,
            raw_selected,
            float(selected["weight"]),
            str(selected["direction"]),
        )
        target24 = latent24["target"].astype(np.float64)
        month24 = latent24["game_month"].astype(np.int16)
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
            raw=raw_selected,
            candidate=candidate24,
            game_month=month24,
            domain3=domain24,
        )
    best_type = (
        pd.DataFrame(type_metrics)
        .sort_values(["audit_year", "logloss"])
        .groupby("audit_year", as_index=False)
        .first()
        .to_dict("records")
    )
    summary["best_type_metrics"] = best_type
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
        default=Path("artifacts/v25_pitch_type_eb_student_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
