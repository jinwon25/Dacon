"""Train-only prototype retrieval residual above frozen v27.

Within an exact domain/count/hand bucket, source rows are compressed into at
most 32 target-free MiniBatchKMeans prototypes in the official row-state
space.  A prototype stores an empirical-Bayes-shrunk, bucket-centred source
OOF residual.  Evaluation rows retrieve only the nearest frozen prototype;
they are never reference points for each other and their distribution is not
used during fitting.

The single representation is fitted at rolling origins.  Route and damping
must pass both 2022 and late-2023 before the 2024 audits are inspected.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

from sklearn.cluster import MiniBatchKMeans

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.champion.v50_low_rank_pitcher_context import select_consensus
from src.champion.v53_factorization_offset import TARGET, _frame


PROTOTYPES_PER_BUCKET = 32
MIN_ROWS_PER_PROTOTYPE = 256
RESIDUAL_SMOOTHING = 200.0
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.25, 0.50, 1.00)
SEED = 5501

BUCKET_COLUMNS = (
    "domain3",
    "balls_before",
    "strikes_before",
    "pitcher_hand",
    "batter_hand",
)
NUMERIC_COLUMNS = (
    "inning",
    "outs_before",
    "run_total_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
LOG_COLUMNS = (
    "asof_pitcher_n",
    "asof_batter_n",
    "asof_pitcher_pitchmix_n",
)


def prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = _add_domain_and_pressure(frame.copy())
    missing = sorted(
        (set(BUCKET_COLUMNS) | set(NUMERIC_COLUMNS)) - set(output.columns)
    )
    if missing:
        raise ValueError(f"missing retrieval columns: {missing}")
    return output


def bucket_keys(frame: pd.DataFrame) -> np.ndarray:
    text = frame.loc[:, BUCKET_COLUMNS].astype("string").fillna("__MISSING__")
    return text.agg("\x1f".join, axis=1).to_numpy(str)


@dataclass(frozen=True)
class RobustScaler:
    fill: np.ndarray
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "RobustScaler":
        numeric = frame.loc[:, NUMERIC_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        fill = numeric.median(axis=0).fillna(0.0).to_numpy(np.float64)
        filled = numeric.fillna(
            pd.Series(fill, index=NUMERIC_COLUMNS)
        ).to_numpy(np.float64)
        for index, column in enumerate(NUMERIC_COLUMNS):
            if column in LOG_COLUMNS:
                filled[:, index] = np.log1p(np.clip(filled[:, index], 0.0, None))
        median = np.median(filled, axis=0)
        q25 = np.quantile(filled, 0.25, axis=0)
        q75 = np.quantile(filled, 0.75, axis=0)
        scale = q75 - q25
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        return cls(fill=fill, median=median, scale=scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame.loc[:, NUMERIC_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float64)
        missing = ~np.isfinite(numeric)
        numeric[missing] = np.broadcast_to(self.fill, numeric.shape)[missing]
        for index, column in enumerate(NUMERIC_COLUMNS):
            if column in LOG_COLUMNS:
                numeric[:, index] = np.log1p(
                    np.clip(numeric[:, index], 0.0, None)
                )
        return np.clip((numeric - self.median) / self.scale, -8.0, 8.0).astype(
            np.float32
        )


@dataclass(frozen=True)
class PrototypeBucket:
    centroids: np.ndarray
    effects: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class PrototypeRetriever:
    scaler: RobustScaler
    buckets: dict[str, PrototypeBucket]
    diagnostics: dict[str, object]

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        query = prepare_rows(frame).reset_index(drop=True)
        features = self.scaler.transform(query)
        keys = bucket_keys(query)
        output = np.zeros(len(query), dtype=np.float64)
        seen = np.zeros(len(query), dtype=bool)
        for key in np.unique(keys):
            model = self.buckets.get(str(key))
            if model is None:
                continue
            index = np.flatnonzero(keys == key)
            distance = np.sum(
                np.square(
                    features[index, None, :]
                    - model.centroids[None, :, :]
                ),
                axis=2,
            )
            nearest = np.argmin(distance, axis=1)
            output[index] = model.effects[nearest]
            seen[index] = True
        return output, seen


def fit_retriever(
    frame: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    *,
    seed: int = SEED,
) -> PrototypeRetriever:
    """Fit target-free prototypes and bucket-centred residual effects."""

    source = prepare_rows(frame).reset_index(drop=True)
    target = np.asarray(target, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    if not (len(source) == len(target) == len(parent)):
        raise ValueError("retrieval source length mismatch")
    scaler = RobustScaler.fit(source)
    features = scaler.transform(source)
    keys = bucket_keys(source)
    residual = target - parent
    buckets: dict[str, PrototypeBucket] = {}
    bucket_rows = []
    for bucket_index, key in enumerate(sorted(np.unique(keys))):
        index = np.flatnonzero(keys == key)
        local = features[index]
        count = min(
            PROTOTYPES_PER_BUCKET,
            max(1, len(index) // MIN_ROWS_PER_PROTOTYPE),
        )
        if count == 1:
            centroids = local.mean(axis=0, keepdims=True)
            labels = np.zeros(len(index), dtype=np.int32)
        else:
            cluster = MiniBatchKMeans(
                n_clusters=count,
                batch_size=min(4096, len(index)),
                max_iter=50,
                n_init=1,
                reassignment_ratio=0.0,
                random_state=seed + bucket_index,
            )
            labels = cluster.fit_predict(local)
            centroids = cluster.cluster_centers_.astype(np.float32)
        local_residual = residual[index]
        bucket_mean = float(local_residual.mean())
        centred = local_residual - bucket_mean
        sums = np.bincount(labels, weights=centred, minlength=count)
        counts = np.bincount(labels, minlength=count).astype(np.int64)
        effects = sums / (counts.astype(np.float64) + RESIDUAL_SMOOTHING)
        buckets[str(key)] = PrototypeBucket(
            centroids=np.asarray(centroids, dtype=np.float32),
            effects=effects.astype(np.float64),
            counts=counts,
        )
        bucket_rows.append(
            {
                "key": str(key),
                "rows": int(len(index)),
                "prototypes": int(count),
                "bucket_residual_mean_removed": bucket_mean,
                "mean_abs_effect": float(np.mean(np.abs(effects))),
            }
        )
    return PrototypeRetriever(
        scaler=scaler,
        buckets=buckets,
        diagnostics={
            "rows": int(len(source)),
            "bucket_count": int(len(buckets)),
            "prototype_count": int(
                sum(len(bucket.effects) for bucket in buckets.values())
            ),
            "mean_abs_source_residual": float(np.mean(np.abs(residual))),
            "mean_abs_prototype_effect": float(
                np.mean(
                    np.abs(
                        np.concatenate(
                            [bucket.effects for bucket in buckets.values()]
                        )
                    )
                )
            ),
            "buckets": bucket_rows,
        },
    )


def apply_retrieval(
    frame: pd.DataFrame,
    parent: np.ndarray,
    signal: np.ndarray,
    route: str,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    active = (
        np.ones(len(frame), dtype=bool)
        if route == "ALL"
        else frame["domain3"].astype(str).eq(route).to_numpy()
    )
    candidate = parent.copy()
    candidate[active] = np.clip(
        parent[active] + float(weight) * signal[active], 0.001, 0.999
    )
    return candidate, active


def _screen(
    frame: pd.DataFrame, parent: np.ndarray, signal: np.ndarray
) -> pd.DataFrame:
    rows = []
    for route in ROUTES:
        for weight in WEIGHTS:
            candidate, active = apply_retrieval(
                frame, parent, signal, route, weight
            )
            result = diagnostics(frame, parent, candidate, active)
            applied_gain = (
                min(result["domain_gains"].values())
                if route == "ALL"
                else result["domain_gains"][route]
            )
            rows.append(
                {
                    "signal": "prototype32_bucket_centered",
                    "domain": route,
                    "weight": float(weight),
                    "gain": float(result["gain"]),
                    "positive_month_fraction": float(
                        result["positive_month_fraction"]
                    ),
                    "worst_month_gain": float(result["worst_month_gain"]),
                    "minimum_domain_gain": float(
                        result["minimum_domain_gain"]
                    ),
                    "applied_domain_gain": float(applied_gain),
                    "selection_score": float(
                        min(
                            result["gain"],
                            result["worst_month_gain"],
                            applied_gain,
                        )
                    ),
                    "mean_abs_shift": float(result["mean_abs_shift"]),
                }
            )
    return pd.DataFrame(rows)


def _load_wave0(
    project: Path, year: int
) -> tuple[np.ndarray, np.ndarray]:
    path = (
        project
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{year}.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        return (
            saved["target"].astype(np.float64),
            saved["incumbent"].astype(np.float64),
        )


def _fit_map(
    source: pd.DataFrame,
    source_target: np.ndarray,
    source_parent: np.ndarray,
    audit: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    model = fit_retriever(source, source_target, source_parent)
    signal, seen = model.predict(audit)
    report = dict(model.diagnostics)
    report.pop("buckets", None)
    report.update(
        {
            "audit_rows": int(len(audit)),
            "audit_bucket_seen_rate": float(seen.mean()),
            "audit_mean_abs_signal": float(np.mean(np.abs(signal))),
            "audit_max_abs_signal": float(np.max(np.abs(signal))),
        }
    )
    return signal, report


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    rows21 = raw.loc[raw["season"].eq(2021)].reset_index(drop=True)
    rows22 = raw.loc[raw["season"].eq(2022)].reset_index(drop=True)
    axes = _cached_v25_axes(project, raw)

    target21, parent21 = _load_wave0(project, 2021)
    if not np.array_equal(target21, rows21[TARGET].to_numpy(np.float64)):
        raise ValueError("2021 wave0 target/order mismatch")
    meta22 = _metadata(project, 2022)
    if not np.array_equal(
        meta22["target"], rows22[TARGET].to_numpy(np.float64)
    ):
        raise ValueError("2022 target/order mismatch")
    audit22 = prepare_rows(rows22)
    if not np.array_equal(
        audit22["domain3"].astype(str).to_numpy(), meta22["domain"]
    ):
        raise ValueError("2022 domain/order mismatch")
    frame22 = _frame(meta22["target"], meta22["month"], meta22["domain"])
    frame23 = axes["selection_late_2023"]
    parent23 = v27_parent(frame23)
    frame24 = axes["outer_full_2024"]
    parent24 = v27_parent(frame24)
    late24 = axes["replication_late_2024"]
    parent_late24 = v27_parent(late24)

    print("[v55] fit 2021 -> 2022", flush=True)
    signal22, fit22 = _fit_map(rows21, target21, parent21, rows22)
    print("[v55] fit 2022 -> late2023", flush=True)
    signal23, fit23 = _fit_map(
        rows22, meta22["target"], meta22["parent"], frame23
    )
    stage1 = _screen(frame22, meta22["parent"], signal22)
    stage2 = _screen(frame23, parent23, signal23)
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "domain": str(chosen["domain"]),
        "weight": float(chosen["weight"]),
    }
    selection_pass = bool(chosen["passes_consensus_gate"])

    print(f"[v55] frozen outer recipe={recipe}", flush=True)
    signal24, fit24 = _fit_map(
        frame23, frame23["target"].to_numpy(np.float64), parent23, frame24
    )
    candidate24, active24 = apply_retrieval(
        frame24, parent24, signal24, recipe["domain"], recipe["weight"]
    )
    full_audit = diagnostics(frame24, parent24, candidate24, active24)
    late_mask = frame24["game_month"].ge(8).to_numpy()
    if not np.array_equal(
        frame24.loc[late_mask, "target"].to_numpy(np.float64),
        late24["target"].to_numpy(np.float64),
    ):
        raise ValueError("late-2024 order mismatch")
    candidate_late, active_late = apply_retrieval(
        late24,
        parent_late24,
        signal24[late_mask],
        recipe["domain"],
        recipe["weight"],
    )
    late_audit = diagnostics(
        late24, parent_late24, candidate_late, active_late
    )

    gates = {
        "two_origin_consensus": selection_pass,
        "outer_gain_at_least_5": bool(full_audit["gain"] >= 5.0),
        "outer_month_fraction_at_least_075": bool(
            full_audit["positive_month_fraction"] >= 0.75
        ),
        "outer_worst_month_above_minus_10": bool(
            full_audit["worst_month_gain"] > -10.0
        ),
        "outer_minimum_domain_nonnegative": bool(
            full_audit["minimum_domain_gain"] >= 0.0
        ),
        "replication_gain_positive": bool(late_audit["gain"] > 0.0),
        "replication_month_fraction_at_least_two_thirds": bool(
            late_audit["positive_month_fraction"] >= 2.0 / 3.0
        ),
    }
    eligible = bool(all(gates.values()))
    np.savez_compressed(
        output_dir / "outer_full_2024.npz",
        target=frame24["target"].to_numpy(np.float64),
        v27=parent24,
        signal=signal24,
        candidate=candidate24,
        active=active24,
    )
    summary = {
        "protocol": "V55_TRAIN_ONLY_BUCKET_PROTOTYPE_RETRIEVAL_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "preregistration": "reports/top1100/experiment_registry.csv F5_retrieval_01",
        "configuration": {
            "bucket_columns": list(BUCKET_COLUMNS),
            "numeric_columns": list(NUMERIC_COLUMNS),
            "prototypes_per_bucket_max": PROTOTYPES_PER_BUCKET,
            "minimum_rows_per_prototype": MIN_ROWS_PER_PROTOTYPE,
            "residual_smoothing": RESIDUAL_SMOOTHING,
            "source_bucket_mean_removed": True,
            "retrieval_neighbors": 1,
            "routes": list(ROUTES),
            "weights": list(WEIGHTS),
        },
        "fit_audits": {
            "2021_to_2022": fit22,
            "2022_to_late2023": fit23,
            "late2023_to_2024": fit24,
        },
        "selection": "exact route/damping consensus on 2022 and late-2023 only",
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
        "audits": {
            "outer_full_2024": full_audit,
            "replication_late_2024": late_audit,
        },
        "gates": gates,
        "eligible_for_packaging": eligible,
        "row_local_inference": True,
        "test_rows_as_retrieval_reference": False,
        "test_aggregate_used": False,
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
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v55_prototype_retrieval_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
