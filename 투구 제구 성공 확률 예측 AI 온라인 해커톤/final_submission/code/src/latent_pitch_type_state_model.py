"""Latent current-pitch-type mixture around the row-local state model.

Training rows may recover the current pitch group from the *next* cumulative
pitch-mix snapshot for the same pitcher and season.  At audit/inference time,
the current pitch group is never read from another audit row: a classifier
trained only on earlier seasons supplies the three mixture probabilities.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.multi_year_state_model import _add_categories, _model, _state_features
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.trackman_privileged_distillation import V17_NAME, _diagnostics, bss


TYPE_NAMES = ("fastball", "breaking", "offspeed")
RATE_COLUMNS = tuple(f"asof_pitcher_{name}_rate" for name in TYPE_NAMES)


def reconstruct_current_pitch_type(rows: pd.DataFrame) -> np.ndarray:
    """Recover current pitch group from the next same-pitcher ASOF snapshot.

    The returned integer labels are 0/1/2 for ``TYPE_NAMES`` and -1 where the
    next snapshot is unavailable or the cumulative counter does not advance by
    exactly one.  This function is training-only privileged preprocessing.
    """

    required = {"season", "pitcher_id", "asof_pitcher_pitchmix_n", *RATE_COLUMNS}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"missing pitch-mix columns: {sorted(missing)}")
    n = pd.to_numeric(rows["asof_pitcher_pitchmix_n"], errors="coerce").fillna(0.0)
    rates = rows.loc[:, RATE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    counts = rates.to_numpy(np.float64) * n.to_numpy(np.float64)[:, None]
    frame = pd.DataFrame(
        {
            "season": rows["season"].to_numpy(),
            "pitcher_id": rows["pitcher_id"].to_numpy(),
            "n": n.to_numpy(np.float64),
            "position": np.arange(len(rows), dtype=np.int64),
        }
    )
    for column_index, name in enumerate(TYPE_NAMES):
        frame[f"count_{name}"] = counts[:, column_index]
    grouped = frame.groupby(["season", "pitcher_id"], sort=False, observed=True)
    next_n = grouped["n"].shift(-1).to_numpy(np.float64)
    next_counts = np.column_stack(
        [grouped[f"count_{name}"].shift(-1).to_numpy(np.float64) for name in TYPE_NAMES]
    )
    delta = next_counts - counts
    label = np.full(len(rows), -1, dtype=np.int8)
    valid = (
        np.isfinite(next_n)
        & np.isclose(next_n - n.to_numpy(np.float64), 1.0, atol=1e-6)
        & np.isfinite(delta).all(axis=1)
    )
    best = np.argmax(np.where(np.isfinite(delta), delta, -np.inf), axis=1)
    best_value = delta[np.arange(len(rows)), best]
    residual = np.sum(np.abs(delta - np.eye(3, dtype=np.float64)[best]), axis=1)
    valid &= (best_value > 0.98) & (residual < 0.05)
    label[valid] = best[valid].astype(np.int8)
    return label


def _type_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        verbosity=-1,
        n_jobs=6,
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=5,
        min_child_samples=600,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=2.0,
        reg_lambda=15.0,
        max_bin=127,
        random_state=seed,
    )


def _typed_features(features: pd.DataFrame, label: np.ndarray) -> pd.DataFrame:
    output = features.copy()
    values = np.full(len(label), "__MISSING__", dtype=object)
    valid = label >= 0
    names = np.asarray(TYPE_NAMES, dtype=object)
    values[valid] = names[label[valid]]
    output["cat__latent_pitch_type"] = pd.Categorical(
        values, categories=["__MISSING__", *TYPE_NAMES]
    )
    return output


def _predict_each_type(
    model: lgb.LGBMRegressor,
    features: pd.DataFrame,
    categorical: list[str],
) -> np.ndarray:
    predictions = []
    for name in TYPE_NAMES:
        typed = features.copy()
        typed["cat__latent_pitch_type"] = pd.Categorical(
            np.full(len(typed), name, dtype=object),
            categories=["__MISSING__", *TYPE_NAMES],
        )
        predictions.append(model.predict(typed, categorical_feature=categorical))
        del typed
    return np.column_stack(predictions).astype(np.float64)


def run(project: Path, output_dir: Path, half_life: float) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    label = reconstruct_current_pitch_type(train)
    label_coverage = float(np.mean(label >= 0))
    print(f"[latent-type] reconstructed coverage={label_coverage:.6f}", flush=True)
    numeric_state, baseline = _state_features(train)
    features = _add_categories(train, numeric_state)
    categorical = [column for column in features if column.startswith("cat__")]
    typed_features = _typed_features(features, label)
    typed_categorical = [*categorical, "cat__latent_pitch_type"]

    rows_out: list[dict[str, object]] = []
    for audit_year in (2023, 2024):
        print(f"[latent-type] audit_year={audit_year}", flush=True)
        fit_mask = train["season"].lt(audit_year).to_numpy() & (label >= 0)
        audit_mask = train["season"].eq(audit_year).to_numpy()
        fit_season = train.loc[fit_mask, "season"].to_numpy(np.float64)
        sample_weight = np.exp2(-(audit_year - 1.0 - fit_season) / half_life)
        sample_weight /= sample_weight.mean()

        classifier = _type_model(seed=5200 + audit_year)
        classifier.fit(
            features.loc[fit_mask],
            label[fit_mask].astype(np.int64),
            sample_weight=sample_weight,
            categorical_feature=categorical,
        )
        type_probability = classifier.predict_proba(features.loc[audit_mask]).astype(
            np.float64
        )
        del classifier
        gc.collect()

        outcome = _model(leaves=15, seed=5300 + audit_year)
        outcome.fit(
            typed_features.loc[fit_mask],
            train.loc[fit_mask, "control_success"].to_numpy(np.float64)
            - baseline[fit_mask],
            sample_weight=sample_weight,
            categorical_feature=typed_categorical,
        )
        residual_by_type = _predict_each_type(
            outcome, features.loc[audit_mask], typed_categorical
        )
        del outcome
        gc.collect()

        base = baseline[audit_mask]
        raw_by_type = np.clip(base[:, None] + residual_by_type, 0.001, 0.999)
        student_raw = np.sum(type_probability * raw_by_type, axis=1)
        audit_label = label[audit_mask]
        oracle_raw = student_raw.copy()
        oracle_mask = audit_label >= 0
        oracle_raw[oracle_mask] = raw_by_type[
            np.flatnonzero(oracle_mask), audit_label[oracle_mask]
        ]

        audit = train.loc[audit_mask].reset_index(drop=True)
        target = audit["control_success"].to_numpy(np.float64)
        with np.load(
            project
            / "artifacts"
            / "v16_multiseason_20260815_02"
            / f"{V17_NAME}_o{audit_year}.npz"
        ) as saved:
            incumbent = saved["candidate"].astype(np.float64)
            if not np.array_equal(target, saved["target"].astype(np.float64)):
                raise ValueError(f"v17 order mismatch for {audit_year}")

        known = audit_label >= 0
        accuracy = float(
            np.mean(np.argmax(type_probability[known], axis=1) == audit_label[known])
        )
        logloss = float(
            -np.mean(
                np.log(
                    np.clip(
                        type_probability[np.flatnonzero(known), audit_label[known]],
                        1e-12,
                        1.0,
                    )
                )
            )
        )
        np.savez_compressed(
            output_dir / f"latent_pitch_type_o{audit_year}.npz",
            target=target,
            incumbent=incumbent,
            student_raw=student_raw,
            oracle_raw=oracle_raw,
            raw_by_type=raw_by_type,
            type_probability=type_probability,
            type_label=audit_label,
            domain3=audit["domain3"].astype(str).to_numpy(),
            game_month=audit["game_month"].to_numpy(np.int16),
            pitcher_id=audit["pitcher_id"].to_numpy(),
        )
        result = {
            "audit_year": audit_year,
            "type_coverage": float(known.mean()),
            "type_accuracy": accuracy,
            "type_logloss": logloss,
            "student_raw_gain": bss(target, student_raw) - bss(target, incumbent),
            "oracle_raw_gain": bss(target, oracle_raw) - bss(target, incumbent),
            "student_blends": [],
            "oracle_blends": [],
        }
        for raw_name, raw in (("student", student_raw), ("oracle", oracle_raw)):
            destination = result[f"{raw_name}_blends"]
            for weight in (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0):
                candidate = np.clip(incumbent + weight * (raw - incumbent), 0.001, 0.999)
                diagnostic = _diagnostics(audit, target, incumbent, candidate)
                destination.append(
                    {
                        "weight": weight,
                        "gain": diagnostic["gain"],
                        "month_positive_fraction": float(
                            np.mean([row["gain"] > 0 for row in diagnostic["months"]])
                        ),
                        "worst_month_gain": float(
                            min(row["gain"] for row in diagnostic["months"])
                        ),
                    }
                )
        rows_out.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    summary = {
        "protocol": "LATENT_CURRENT_PITCH_TYPE_STATE_MIXTURE_FORWARD_V1",
        "half_life": half_life,
        "training_label_coverage": label_coverage,
        "folds": rows_out,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/latent_pitch_type_state_20260816_01"),
    )
    parser.add_argument("--half-life", type=float, default=0.5)
    args = parser.parse_args()
    run(args.project, args.output_dir, args.half_life)


if __name__ == "__main__":
    main()
