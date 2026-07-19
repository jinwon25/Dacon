from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from experiments.cfsv2_disagreement_screen import build_disagreement_features
from experiments.exact_oof_meta_gate import (
    CAPACITY,
    META_SEEDS,
    Q2_START,
    _evaluate_period,
    apply_meta_gate,
    fit_probabilities,
    settlement_benefit_labels,
)
from experiments.exact_oof_meta_gate_sweep import _prepare_validation


FINE_THRESHOLD = 0.545
FINE_ALPHA = 0.50
H2_START = pd.Timestamp("2024-07-01 01:00:00")


def _crossfit_meta_probabilities(
    features: np.ndarray,
    label: np.ndarray,
    sample_mask: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """Return Q1 out-of-fold probabilities without training on each scored row."""
    sample_indices = np.flatnonzero(sample_mask)
    if len(sample_indices) < 100 or len(np.unique(label[sample_indices])) != 2:
        raise ValueError("meta cross-fit needs at least 100 rows and both labels")
    seed_probabilities = []
    for seed in META_SEEDS:
        probability = np.zeros(len(label), dtype=float)
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + 30_000
        )
        for train_local, valid_local in splitter.split(
            sample_indices, label[sample_indices]
        ):
            train_indices = sample_indices[train_local]
            valid_indices = sample_indices[valid_local]
            model = ExtraTreesClassifier(
                n_estimators=600,
                min_samples_leaf=50,
                max_features=1.0,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(features[train_indices], label[train_indices])
            probability[valid_indices] = model.predict_proba(
                features[valid_indices]
            )[:, 1]
        if not np.isfinite(probability[sample_indices]).all():
            raise ValueError("meta cross-fit produced non-finite probabilities")
        seed_probabilities.append(probability)
    return np.mean(seed_probabilities, axis=0)


def _risk_features(
    index: pd.DatetimeIndex,
    current: np.ndarray,
    member: np.ndarray,
    meta_probability: np.ndarray,
) -> pd.DataFrame:
    hour = 2.0 * np.pi * index.hour.to_numpy() / 24.0
    day = 2.0 * np.pi * index.dayofyear.to_numpy() / 366.0
    return pd.DataFrame(
        {
            "meta_probability": meta_probability,
            "current_ratio": current / CAPACITY,
            "member_ratio": member / CAPACITY,
            "movement_ratio": (member - current) / CAPACITY,
            "movement_abs_ratio": np.abs(member - current) / CAPACITY,
            "hour_sin": np.sin(hour),
            "hour_cos": np.cos(hour),
            "day_sin": np.sin(day),
            "day_cos": np.cos(day),
        },
        index=index,
    )


def _apply_keep_policy(
    current: np.ndarray,
    fine_candidate: np.ndarray,
    fine_gate: np.ndarray,
    benefit_probability: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.asarray(fine_gate, dtype=bool) & (
        np.asarray(benefit_probability, dtype=float) >= threshold
    )
    candidate = np.asarray(current, dtype=float).copy()
    candidate[keep] = np.asarray(fine_candidate, dtype=float)[keep]
    return candidate, keep


def _fit_risk_members(
    features: pd.DataFrame,
    label: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    if train.sum() < 80 or len(np.unique(label[train])) != 2:
        raise ValueError("risk gate needs at least 80 gated Q1 rows and both labels")
    predictions = []
    leaf = max(8, int(train.sum() // 18))
    for seed in META_SEEDS:
        model = ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=leaf,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + 40_000,
        )
        model.fit(features.loc[train], label[train])
        predictions.append(model.predict_proba(features)[:, 1])
    return np.asarray(predictions)


def _select_policy(
    truth: np.ndarray,
    current: np.ndarray,
    fine_candidate: np.ndarray,
    fine_gate: np.ndarray,
    q2: np.ndarray,
    seed_probability: np.ndarray,
) -> dict[str, object] | None:
    mean_probability = seed_probability.mean(axis=0)
    fine_metrics = _evaluate_period(truth, current, fine_candidate, q2)
    rows = []
    for threshold in np.arange(0.25, 0.751, 0.025):
        candidate, keep = _apply_keep_policy(
            current, fine_candidate, fine_gate, mean_probability, float(threshold)
        )
        total = _evaluate_period(truth, current, candidate, q2)
        incremental = _evaluate_period(truth, fine_candidate, candidate, q2)
        seed_rows = []
        for seed, probability in zip(META_SEEDS, seed_probability):
            seed_candidate, seed_keep = _apply_keep_policy(
                current, fine_candidate, fine_gate, probability, float(threshold)
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "kept_rows": int((seed_keep & q2).sum()),
                    "total_delta": _evaluate_period(
                        truth, current, seed_candidate, q2
                    )["delta"],
                    "incremental_delta": _evaluate_period(
                        truth, fine_candidate, seed_candidate, q2
                    )["delta"],
                }
            )
        eligible = bool(
            min(total["delta"].values()) > 0.0
            and incremental["delta"]["score"] > 0.0
            and incremental["delta"]["ficr"] >= 0.0
            and all(
                min(row["total_delta"].values()) > 0.0
                and row["incremental_delta"]["score"] > 0.0
                for row in seed_rows
            )
        )
        rows.append(
            {
                "threshold": float(round(threshold, 3)),
                "kept_rows": int((keep & q2).sum()),
                "dropped_rows": int((fine_gate & q2).sum() - (keep & q2).sum()),
                "total_vs_current": total,
                "incremental_vs_fine": incremental,
                "seed_rows": seed_rows,
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        return None
    selected = max(
        eligible_rows,
        key=lambda row: (
            min(seed["incremental_delta"]["score"] for seed in row["seed_rows"]),
            row["incremental_vs_fine"]["delta"]["score"],
            -row["dropped_rows"],
        ),
    )
    return {
        "fine_vs_current": fine_metrics,
        "selected": selected,
        "top": sorted(
            rows,
            key=lambda row: row["incremental_vs_fine"]["delta"]["score"],
            reverse=True,
        )[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--cfsv2-features",
        default="artifacts_final/external_weather/noaa_cfsv2_h1_2024/features.csv",
    )
    parser.add_argument("--gfs", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/noaa_cfsv2_h1_2024/meta_risk_gate.json",
    )
    parser.add_argument("--minimum-incremental-vs-control", type=float, default=0.0001)
    args = parser.parse_args()

    (
        _labels,
        index,
        truth,
        _group_1,
        _group_2,
        _base,
        member,
        current,
        meta_feature_array,
        action,
    ) = _prepare_validation(Path(args.labels), Path(args.driver_cache))
    q1 = (index >= pd.Timestamp("2024-01-01")) & (index < Q2_START)
    q2 = (index >= Q2_START) & (index < H2_START)

    q1_meta_label = settlement_benefit_labels(truth, current, member, q1)
    q1_meta_probability = _crossfit_meta_probabilities(
        meta_feature_array, q1_meta_label, q1 & action
    )
    q1_fine, q1_fine_gate = apply_meta_gate(
        current,
        member,
        action,
        q1_meta_probability,
        threshold=FINE_THRESHOLD,
        extra_alpha=FINE_ALPHA,
    )
    h1_label = settlement_benefit_labels(truth, current, member, index < Q2_START)
    q2_meta_probability, _ = fit_probabilities(
        meta_feature_array, h1_label, (index < Q2_START) & action
    )
    q2_fine, q2_fine_gate = apply_meta_gate(
        current,
        member,
        action,
        q2_meta_probability,
        threshold=FINE_THRESHOLD,
        extra_alpha=FINE_ALPHA,
    )
    fine_candidate = current.copy()
    fine_gate = np.zeros(len(index), dtype=bool)
    fine_candidate[q1] = q1_fine[q1]
    fine_candidate[q2] = q2_fine[q2]
    fine_gate[q1] = q1_fine_gate[q1]
    fine_gate[q2] = q2_fine_gate[q2]
    risk_label = settlement_benefit_labels(
        truth, current, fine_candidate, q1, step=1.0
    )

    external = build_disagreement_features(
        Path(args.cfsv2_features), Path(args.gfs)
    )
    common = index.intersection(external.index)
    common = common[(common >= pd.Timestamp("2024-01-01")) & (common < H2_START)]
    positions = index.get_indexer(common)
    expected = index[q1 | q2]
    missing = expected.difference(common)
    # Forecast targets begin at 01:00; the lineage additionally contains the
    # 00:00 boundary row used by the original validation split.
    allowed_boundary = pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00")])
    if not missing.equals(allowed_boundary):
        raise ValueError("risk gate requires complete 2024 H1 CFSv2 coverage")
    combined_meta_probability = np.where(q1, q1_meta_probability, q2_meta_probability)
    control = _risk_features(
        common,
        current[positions],
        member[positions],
        combined_meta_probability[positions],
    )
    with_cfs = control.join(external.reindex(common))
    local_q1 = q1[positions]
    local_q2 = q2[positions]
    local_fine_gate = fine_gate[positions]
    train = local_q1 & local_fine_gate
    control_probability = _fit_risk_members(
        control, risk_label[positions], train
    )
    cfs_probability = _fit_risk_members(
        with_cfs, risk_label[positions], train
    )
    control_result = _select_policy(
        truth[positions],
        current[positions],
        fine_candidate[positions],
        local_fine_gate,
        local_q2,
        control_probability,
    )
    cfs_result = _select_policy(
        truth[positions],
        current[positions],
        fine_candidate[positions],
        local_fine_gate,
        local_q2,
        cfs_probability,
    )
    control_gain = (
        control_result["selected"]["incremental_vs_fine"]["delta"]["score"]
        if control_result is not None
        else -np.inf
    )
    cfs_gain = (
        cfs_result["selected"]["incremental_vs_fine"]["delta"]["score"]
        if cfs_result is not None
        else -np.inf
    )
    expand_h2 = bool(
        cfs_result is not None
        and cfs_gain >= control_gain + args.minimum_incremental_vs_control
    )
    report = {
        "family": "cfsv2_meta_gate_harm_risk_abstention",
        "split": {
            "risk_train": (
                "2024 Q1 using five-fold out-of-fold meta probabilities; "
                "each scored row is excluded from its meta model"
            ),
            "selection": "2024 Q2 using exact H1-trained fine meta probabilities",
            "locked_h2": "unopened",
            "risk_train_gated_rows": int(train.sum()),
            "risk_train_positive_fraction": float(risk_label[positions][train].mean()),
            "q2_fine_rows": int((local_fine_gate & local_q2).sum()),
            "excluded_lineage_boundary": str(missing[0]),
        },
        "fine_policy": {"threshold": FINE_THRESHOLD, "alpha": FINE_ALPHA},
        "control_risk_gate": control_result,
        "cfsv2_risk_gate": cfs_result,
        "incremental_score_vs_control": (
            float(cfs_gain - control_gain)
            if np.isfinite(cfs_gain) and np.isfinite(control_gain)
            else None
        ),
        "decision": {
            "expand_h2_collection": expand_h2,
            "minimum_incremental_vs_control": args.minimum_incremental_vs_control,
            "reason": (
                "CFSv2 risk abstention adds material Q2 gain beyond the identical control"
                if expand_h2
                else "CFSv2 risk abstention failed the preregistered incremental control gate"
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
