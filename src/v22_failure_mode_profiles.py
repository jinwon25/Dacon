"""Hierarchical previous-season failure-mode profiles above the v21 champion.

The four training-only latent modes are reconstructed from the next cumulative
ASOF snapshot, exactly as in :mod:`src.failure_mode_privileged_distillation`.
At inference, however, this model uses only profiles frozen at the end of the
previous season plus fields from the current row.  No current-pitch mode or
other evaluation row is read.

The experiment improves the original latent-mode classifier in two ways:

* explicit empirical-Bayes backoff from global/count profiles to pitcher and
  pitcher-by-context profiles;
* exact row-local current-season ASOF state alongside those frozen profiles.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.failure_mode_privileged_distillation import MODE_NAMES, reconstruct_failure_mode
from src.multi_year_state_model import _add_categories, _state_features
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v20_residual_overlay_screen import _bss


PROFILE_SPECS = (
    ("count_hands", ("balls_before", "strikes_before", "pitcher_hand", "batter_hand"), 1200.0, None),
    ("pitcher", ("pitcher_id",), 120.0, None),
    ("pitcher_hand", ("pitcher_id", "batter_hand"), 70.0, "pitcher"),
    ("pitcher_count", ("pitcher_id", "balls_before", "strikes_before"), 45.0, "pitcher"),
    (
        "pitcher_count_hand",
        ("pitcher_id", "balls_before", "strikes_before", "batter_hand"),
        30.0,
        "pitcher_count",
    ),
    (
        "pitcher_domain_count",
        ("pitcher_id", "domain3", "balls_before", "strikes_before"),
        45.0,
        "pitcher_count",
    ),
    (
        "team_count_hands",
        ("pitcher_team_id", "balls_before", "strikes_before", "pitcher_hand", "batter_hand"),
        400.0,
        "count_hands",
    ),
)
POWERS = (0.75, 1.0, 1.25, 1.5)
BLEND_WEIGHTS = (0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20)


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
    work["__mode"] = label
    counts = (
        work.groupby([*keys, "__mode"], observed=True)
        .size()
        .unstack("__mode", fill_value=0)
        .reindex(columns=range(len(MODE_NAMES)), fill_value=0)
        .reset_index()
    )
    count_columns = [f"__count_{index}" for index in range(len(MODE_NAMES))]
    counts.columns = [*keys, *count_columns]
    mapped = query.loc[:, list(keys)].merge(counts, on=list(keys), how="left", sort=False)
    raw_count = mapped[count_columns].fillna(0.0).to_numpy(np.float64)
    n = raw_count.sum(axis=1)
    probability = (raw_count + alpha * prior) / (n[:, None] + alpha)
    return probability.astype(np.float32), n.astype(np.float32)


def previous_season_mode_profiles(rows: pd.DataFrame, label: np.ndarray) -> pd.DataFrame:
    """Create profiles whose source season is strictly earlier than each row."""
    output = pd.DataFrame(index=rows.index)
    for season in sorted(int(value) for value in rows["season"].unique()):
        query_mask = rows["season"].eq(season).to_numpy()
        history_mask = rows["season"].lt(season).to_numpy() & (label >= 0)
        query = rows.loc[query_mask].reset_index(drop=True)
        if history_mask.any():
            history = rows.loc[history_mask].reset_index(drop=True)
            history_label = label[history_mask]
            count = np.bincount(history_label, minlength=len(MODE_NAMES)).astype(np.float64)
            global_probability = (count + 1.0) / (count.sum() + len(MODE_NAMES))
        else:
            history = rows.iloc[:0].copy()
            history_label = np.empty(0, dtype=np.int8)
            global_probability = np.full(len(MODE_NAMES), 1.0 / len(MODE_NAMES))
        positions = np.flatnonzero(query_mask)
        by_name: dict[str, np.ndarray] = {}
        for name, keys, alpha, parent in PROFILE_SPECS:
            prior = (
                np.tile(global_probability, (len(query), 1))
                if parent is None
                else by_name[parent]
            )
            probability, n = _group_probability(
                history, history_label, query, keys, prior, alpha
            )
            by_name[name] = probability
            for index, mode_name in enumerate(MODE_NAMES):
                output.loc[positions, f"modeprof__{name}__{mode_name}"] = probability[:, index]
            output.loc[positions, f"modeprof__{name}__log_n"] = np.log1p(n)
    return output.astype(np.float32)


def _component_proxy(rows: pd.DataFrame) -> np.ndarray:
    reverse = pd.to_numeric(rows["asof_pitcher_reverse_rate"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    middle = pd.to_numeric(rows["asof_pitcher_middle_rate"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    ball = pd.to_numeric(rows["asof_pitcher_ball_rate"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    strike = pd.to_numeric(rows["asof_pitcher_strike_rate"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    bad = np.clip(reverse + middle, 0.0, 1.0)
    ball_only = np.clip(ball * (1.0 - bad), 0.0, 1.0)
    strike_only = np.clip(strike * (1.0 - bad) * (1.0 - ball), 0.0, 1.0)
    other = np.maximum(1.0 - bad - ball_only - strike_only, 1e-6)
    return _normalise(np.column_stack([bad, ball_only, strike_only, other]))


def _classifier(seed: int, leaves: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(MODE_NAMES),
        verbosity=-1,
        n_jobs=6,
        n_estimators=260,
        learning_rate=0.025,
        num_leaves=leaves,
        max_depth=5 if leaves == 31 else 6,
        min_child_samples=700,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=2.0,
        reg_lambda=20.0,
        max_bin=127,
        random_state=seed,
    )


def _axis_prediction(
    prediction_by_year: dict[int, dict[str, np.ndarray]],
    axis: str,
) -> dict[str, np.ndarray]:
    if axis == "y2023_to_y2024":
        return prediction_by_year[2024]
    year = 2023 if axis == "y2023_early_to_late" else 2024
    return {
        name: value[prediction_by_year[year]["__month"] >= 8]
        for name, value in prediction_by_year[year].items()
        if name != "__month"
    }


def _load_saved_year(output_dir: Path, year: int) -> dict[str, np.ndarray]:
    with np.load(
        output_dir / f"failure_mode_profiles_o{year}.npz", allow_pickle=True
    ) as saved:
        names = [str(value) for value in saved["names"].tolist()]
        raw = saved["raw"].astype(np.float64)
        month = saved["game_month"].astype(np.int16)
    return {
        "__month": month,
        **{name: raw[:, index] for index, name in enumerate(names)},
    }


def run(
    project: Path,
    champion_dir: Path,
    output_dir: Path,
    audit_years: tuple[int, ...] = (2023, 2024),
) -> dict[str, object]:
    project = project.resolve()
    champion_dir = (project / champion_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(pd.read_csv(project / "data" / "train.csv", low_memory=False))
    label = reconstruct_failure_mode(train)
    profile = previous_season_mode_profiles(train, label)
    numeric_state, _ = _state_features(train)
    component = _component_proxy(train)
    for index, name in enumerate(MODE_NAMES):
        profile[f"modeproxy__{name}"] = component[:, index].astype(np.float32)
    features = _add_categories(train, pd.concat([numeric_state, profile], axis=1))
    categorical = [column for column in features if column.startswith("cat__")]

    prediction_by_year: dict[int, dict[str, np.ndarray]] = {}
    fold_summary = []
    for audit_year in audit_years:
        print(f"[v22-mode-profile] audit_year={audit_year}", flush=True)
        fit_mask = train["season"].lt(audit_year).to_numpy() & (label >= 0)
        audit_mask = train["season"].eq(audit_year).to_numpy()
        fit_season = train.loc[fit_mask, "season"].to_numpy(np.float64)
        probability: dict[str, np.ndarray] = {}
        for half_life in (0.5, 2.0, 4.0):
            sample_weight = np.exp2(-(audit_year - 1.0 - fit_season) / half_life)
            sample_weight /= sample_weight.mean()
            model = _classifier(
                seed=8200 + audit_year + int(10 * half_life), leaves=31
            )
            model.fit(
                features.loc[fit_mask],
                label[fit_mask].astype(np.int64),
                sample_weight=sample_weight,
                categorical_feature=categorical,
            )
            probability[f"profile_lgb_h{half_life:g}"] = model.predict_proba(
                features.loc[audit_mask]
            ).astype(np.float64)
            del model
            gc.collect()

        audit_profile = profile.loc[audit_mask].reset_index(drop=True)
        for profile_name in (
            "count_hands",
            "pitcher",
            "pitcher_hand",
            "pitcher_count",
            "pitcher_count_hand",
            "pitcher_domain_count",
            "team_count_hands",
        ):
            probability[f"profile_{profile_name}"] = _normalise(
                audit_profile[
                    [f"modeprof__{profile_name}__{name}" for name in MODE_NAMES]
                ].to_numpy(np.float64)
            )
        probability["component_proxy"] = component[audit_mask]
        probability["profile_lgb_mean"] = np.mean(
            np.stack(
                [
                    probability["profile_lgb_h0.5"],
                    probability["profile_lgb_h2"],
                    probability["profile_lgb_h4"],
                ]
            ),
            axis=0,
        )
        probability["profile_lgb075_pitcher025"] = (
            0.75 * probability["profile_lgb_mean"]
            + 0.25 * probability["profile_pitcher_count_hand"]
        )

        with np.load(
            project
            / "artifacts"
            / "latent_failure_mode_state_20260816_01"
            / f"latent_failure_mode_o{audit_year}.npz",
            allow_pickle=True,
        ) as saved:
            target = saved["target"].astype(np.float64)
            raw_by_mode = saved["raw_by_mode"].astype(np.float64)
            month = saved["game_month"].astype(np.int16)
        audit_label = label[audit_mask]
        known = audit_label >= 0
        outcome_weight = np.exp2(-(audit_year - 1.0 - fit_season) / 0.5)
        conditional_success = np.array(
            [
                np.average(
                    train.loc[fit_mask, "control_success"].to_numpy(np.float64)[
                        label[fit_mask] == klass
                    ],
                    weights=outcome_weight[label[fit_mask] == klass],
                )
                for klass in range(len(MODE_NAMES))
            ]
        )
        raw: dict[str, np.ndarray] = {"__month": month}
        diagnostics = []
        for probability_name, value in probability.items():
            for power in POWERS:
                transformed = _temperature(value, power)
                base_name = f"{probability_name}_pow{power:g}"
                raw[f"modeoutcome__{base_name}"] = np.sum(
                    transformed * raw_by_mode, axis=1
                )
                raw[f"conditional__{base_name}"] = transformed @ conditional_success
                diagnostics.append(
                    {
                        "candidate": base_name,
                        "accuracy": float(
                            np.mean(
                                np.argmax(transformed[known], axis=1)
                                == audit_label[known]
                            )
                        ),
                        "logloss": float(
                            -np.mean(
                                np.log(
                                    np.clip(
                                        transformed[
                                            np.flatnonzero(known), audit_label[known]
                                        ],
                                        1e-12,
                                        1.0,
                                    )
                                )
                            )
                        ),
                    }
                )
        prediction_by_year[audit_year] = raw
        best_logloss = min(diagnostics, key=lambda row: row["logloss"])
        best_accuracy = max(diagnostics, key=lambda row: row["accuracy"])
        fold_summary.append(
            {
                "audit_year": audit_year,
                "known_fraction": float(known.mean()),
                "best_logloss": best_logloss,
                "best_accuracy": best_accuracy,
            }
        )
        legal_names = [name for name in raw if name != "__month"]
        np.savez_compressed(
            output_dir / f"failure_mode_profiles_o{audit_year}.npz",
            target=target,
            game_month=month,
            names=np.asarray(legal_names, dtype=object),
            raw=np.column_stack([raw[name] for name in legal_names]),
        )
        print(json.dumps(fold_summary[-1], ensure_ascii=False), flush=True)

    for audit_year in (2023, 2024):
        if audit_year not in prediction_by_year:
            prediction_by_year[audit_year] = _load_saved_year(output_dir, audit_year)

    metric_rows = []
    for axis in ("y2023_to_y2024", "y2023_early_to_late", "y2024_early_to_late"):
        with np.load(champion_dir / f"{axis}.npz", allow_pickle=True) as saved:
            target = saved["target"].astype(np.float64)
            v21 = saved["v21"].astype(np.float64)
        bank = _axis_prediction(prediction_by_year, axis)
        for name, prediction in bank.items():
            if len(prediction) != len(target):
                raise ValueError(f"prediction order mismatch for {axis}: {name}")
            for weight in BLEND_WEIGHTS:
                candidate = np.clip(
                    v21 + weight * (prediction - v21), 0.001, 0.999
                )
                metric_rows.append(
                    {
                        "axis": axis,
                        "candidate": name,
                        "weight": weight,
                        "gain_vs_v21": _bss(target, candidate) - _bss(target, v21),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    robust = (
        metrics.groupby(["candidate", "weight"], as_index=False)
        .agg(
            min_gain=("gain_vs_v21", "min"),
            mean_gain=("gain_vs_v21", "mean"),
            max_gain=("gain_vs_v21", "max"),
        )
        .sort_values(["min_gain", "mean_gain"], ascending=False)
        .reset_index(drop=True)
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust.to_csv(output_dir / "robust.csv", index=False)
    summary = {
        "protocol": "V22_HIERARCHICAL_FAILURE_MODE_PROFILE_V1",
        "computed_audit_years": list(audit_years),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "folds": fold_summary,
        "candidate_count": int(len(robust)),
        "strictly_positive": int((robust["min_gain"] > 0.0).sum()),
        "best": robust.head(50).to_dict("records"),
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
        "--champion-dir",
        type=Path,
        default=Path("artifacts/champion_oof_20260817_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v22_failure_mode_profiles_20260817_01"),
    )
    parser.add_argument(
        "--audit-years",
        default="2023,2024",
        help="Comma-separated years to compute; missing years are loaded from output-dir.",
    )
    args = parser.parse_args()
    audit_years = tuple(int(value) for value in args.audit_years.split(",") if value)
    if not audit_years or not set(audit_years).issubset({2023, 2024}):
        raise ValueError("audit-years must contain 2023 and/or 2024")
    run(args.project, args.champion_dir, args.output_dir, audit_years)


if __name__ == "__main__":
    main()
