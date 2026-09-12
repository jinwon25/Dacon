"""Rolling-origin audit of a KMA UM power-curve overlay for group 1.

The original KMA branch targeted group 3.  This experiment asks whether the
same causal operational forecast contains independent signal for colocated
group 1.  Policies are selected across three forward origins, not from the
repeatedly inspected H2 period.  The experiment never writes a submission.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from agent_service.compliance import validate_external_data_manifest
from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
TARGET = "kpx_group_1"
CAPACITY = CAPACITY_KWH[TARGET]
COMPONENTS = ("score", "one_minus_nmae", "ficr")
YEAR_START = pd.Timestamp("2024-01-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
YEAR_END = pd.Timestamp("2025-01-01 00:00:00")


@dataclass(frozen=True)
class Policy:
    direction: str
    coverage: float
    minimum_base_ratio: float
    maximum_base_ratio: float
    alpha: float


@dataclass(frozen=True)
class Origin:
    name: str
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def origins() -> tuple[Origin, ...]:
    return (
        Origin(
            "jan_to_feb",
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-03-01"),
        ),
        Origin(
            "jan_feb_to_mar",
            pd.Timestamp("2024-03-01"),
            pd.Timestamp("2024-03-01"),
            pd.Timestamp("2024-04-01"),
        ),
        Origin(
            "q1_to_q2",
            pd.Timestamp("2024-04-01"),
            pd.Timestamp("2024-04-01"),
            H2_START,
        ),
    )


def policies() -> tuple[Policy, ...]:
    return tuple(
        Policy(direction, coverage, minimum, maximum, alpha)
        for direction in ("both", "up", "down")
        for coverage in (1.0, 0.75, 0.50, 0.25, 0.10, 0.05)
        for minimum, maximum in (
            (0.10, 1.00),
            (0.10, 0.80),
            (0.10, 0.60),
            (0.20, 1.00),
            (0.40, 1.00),
        )
        for alpha in (
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
            0.30,
            0.40,
            0.50,
        )
    )


def fit_direct_power(
    wind_speed: np.ndarray,
    truth: np.ndarray,
    reference: np.ndarray,
    train: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    if int((train & available).sum()) < 500:
        raise ValueError("group-1 KMA power curve needs at least 500 training rows")
    model = IsotonicRegression(
        y_min=0.0, y_max=CAPACITY, out_of_bounds="clip"
    )
    model.fit(wind_speed[train & available], truth[train & available])
    direct = np.asarray(reference, dtype=float).copy()
    direct[available] = model.predict(wind_speed[available])
    return direct


def _direction(disagreement: np.ndarray, name: str) -> np.ndarray:
    if name == "both":
        return np.ones(len(disagreement), dtype=bool)
    if name == "up":
        return disagreement > 0.0
    if name == "down":
        return disagreement < 0.0
    raise ValueError(f"unknown direction: {name}")


def apply_policy(
    reference: np.ndarray,
    direct: np.ndarray,
    available: np.ndarray,
    period: np.ndarray,
    policy: Policy,
    *,
    maximum_movement_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    disagreement = np.asarray(direct, dtype=float) - np.asarray(reference, dtype=float)
    base_ratio = np.asarray(reference, dtype=float) / CAPACITY
    pool = (
        np.asarray(period, dtype=bool)
        & np.asarray(available, dtype=bool)
        & _direction(disagreement, policy.direction)
        & (base_ratio >= policy.minimum_base_ratio)
        & (base_ratio <= policy.maximum_base_ratio)
    )
    if not pool.any():
        return np.asarray(reference, dtype=float).copy(), pool, float("nan")
    threshold = (
        -np.inf
        if policy.coverage == 1.0
        else float(np.quantile(np.abs(disagreement[pool]), 1.0 - policy.coverage))
    )
    gate = pool & (np.abs(disagreement) >= threshold)
    movement_bound = maximum_movement_ratio * CAPACITY
    movement = np.clip(
        policy.alpha * disagreement, -movement_bound, movement_bound
    )
    candidate = np.asarray(reference, dtype=float).copy()
    candidate[gate] = np.clip(
        candidate[gate] + movement[gate], 0.0, CAPACITY
    )
    return candidate, gate, threshold


def _delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": float(after.score - before.score),
        "one_minus_nmae": float(after.one_minus_nmae - before.one_minus_nmae),
        "ficr": float(after.ficr - before.ficr),
    }


def compare(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    period: np.ndarray,
) -> dict[str, Any]:
    before = evaluate_group(truth[period], reference[period], CAPACITY)
    after = evaluate_group(truth[period], candidate[period], CAPACITY)
    return {
        "reference": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": _delta(before, after),
    }


def select_rolling_policy(
    truth: np.ndarray,
    reference: np.ndarray,
    wind_speed: np.ndarray,
    available: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    maximum_movement_ratio: float,
    minimum_changed_ratio: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    prepared: list[tuple[Origin, np.ndarray, np.ndarray]] = []
    for origin in origins():
        train = (index >= YEAR_START) & (index < origin.train_end)
        validation = (index >= origin.validation_start) & (
            index < origin.validation_end
        )
        direct = fit_direct_power(
            wind_speed, truth, reference, train, available
        )
        prepared.append((origin, validation, direct))

    eligible: list[dict[str, Any]] = []
    for policy in policies():
        fold_records: list[dict[str, Any]] = []
        passed = True
        for origin, validation, direct in prepared:
            candidate, gate, threshold = apply_policy(
                reference,
                direct,
                available,
                validation,
                policy,
                maximum_movement_ratio=maximum_movement_ratio,
            )
            evaluation = compare(truth, reference, candidate, validation)
            changed_ratio = float(
                (gate & validation).sum() / max(int(validation.sum()), 1)
            )
            if (
                min(evaluation["delta"][component] for component in COMPONENTS)
                <= 0.0
                or changed_ratio < minimum_changed_ratio
            ):
                passed = False
                break
            fold_records.append(
                {
                    "origin": origin.name,
                    "threshold_kwh": threshold,
                    "changed_rows": int((gate & validation).sum()),
                    "changed_ratio": changed_ratio,
                    "evaluation": evaluation,
                }
            )
        if not passed:
            continue
        score = [record["evaluation"]["delta"]["score"] for record in fold_records]
        nmae = [
            record["evaluation"]["delta"]["one_minus_nmae"]
            for record in fold_records
        ]
        ficr = [record["evaluation"]["delta"]["ficr"] for record in fold_records]
        eligible.append(
            {
                "policy": asdict(policy),
                "folds": fold_records,
                "worst_score_delta": float(min(score)),
                "mean_score_delta": float(np.mean(score)),
                "worst_one_minus_nmae_delta": float(min(nmae)),
                "worst_ficr_delta": float(min(ficr)),
            }
        )
    eligible.sort(
        key=lambda record: (
            record["worst_score_delta"],
            record["mean_score_delta"],
            record["worst_one_minus_nmae_delta"],
            record["worst_ficr_delta"],
        ),
        reverse=True,
    )
    return (eligible[0] if eligible else None), eligible


def issue_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    issue_times: np.ndarray,
    period: np.ndarray,
    *,
    samples: int,
    seed: int = 20260726,
) -> dict[str, Any]:
    unique = pd.unique(np.asarray(issue_times)[period])
    groups = [
        np.flatnonzero(period & (np.asarray(issue_times) == issue)) for issue in unique
    ]
    rng = np.random.default_rng(seed)
    values = np.empty((samples, len(COMPONENTS)), dtype=float)
    for iteration in range(samples):
        positions = np.concatenate(
            [groups[value] for value in rng.integers(0, len(groups), len(groups))]
        )
        evaluation = compare(
            truth,
            reference,
            candidate,
            positions,
        )
        values[iteration] = [
            evaluation["delta"][component] for component in COMPONENTS
        ]
    return {
        "samples": int(samples),
        "issue_cycles": int(len(unique)),
        "quantiles": {
            component: {
                "q05": float(np.quantile(values[:, position], 0.05)),
                "median": float(np.quantile(values[:, position], 0.50)),
                "q95": float(np.quantile(values[:, position], 0.95)),
                "positive_fraction": float(np.mean(values[:, position] > 0.0)),
            }
            for position, component in enumerate(COMPONENTS)
        },
        "all_components_positive_fraction": float(np.mean(np.all(values > 0.0, axis=1))),
    }


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="artifacts_final/lineage/exact_driver_oof.npz")
    parser.add_argument(
        "--train-context",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/features.csv",
    )
    parser.add_argument(
        "--train-manifest",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/manifest.json",
    )
    parser.add_argument(
        "--test-manifest",
        default="artifacts_final/external_weather/kma_um_regional_context_2025/manifest.json",
    )
    parser.add_argument("--maximum-movement-ratio", type=float, default=0.05)
    parser.add_argument("--minimum-changed-ratio", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--output",
        default="artifacts_final/diagnostics/kma_group1_rolling_power_curve_20260726.json",
    )
    args = parser.parse_args()

    manifest_validation = {
        "train": validate_external_data_manifest(
            _rooted(args.train_manifest), ROOT
        ),
        "test": validate_external_data_manifest(
            _rooted(args.test_manifest), ROOT
        ),
    }
    driver = np.load(_rooted(args.driver), allow_pickle=False)
    index = pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    truth = driver[f"{TARGET}__valid_truth"].astype(float)
    reference = driver[f"{TARGET}__exact_base"].astype(float)
    context = pd.read_csv(
        _rooted(args.train_context),
        encoding="utf-8-sig",
        usecols=[
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "kma_um_ctx_speed10_r0",
        ],
    )
    context["forecast_kst_dtm"] = pd.to_datetime(context["forecast_kst_dtm"])
    context["data_available_kst_dtm"] = pd.to_datetime(
        context["data_available_kst_dtm"]
    )
    context = context.set_index("forecast_kst_dtm").reindex(index)
    wind_speed = pd.to_numeric(
        context["kma_um_ctx_speed10_r0"], errors="coerce"
    ).to_numpy(dtype=float)
    available = np.isfinite(wind_speed)
    issue_times = pd.to_datetime(context["data_available_kst_dtm"]).to_numpy()

    selected, eligible = select_rolling_policy(
        truth,
        reference,
        wind_speed,
        available,
        index,
        maximum_movement_ratio=args.maximum_movement_ratio,
        minimum_changed_ratio=args.minimum_changed_ratio,
    )
    locked: dict[str, Any] | None = None
    bootstrap: dict[str, Any] | None = None
    decision = "reject; no policy passed rolling-origin development"
    if selected is not None:
        h1 = (index >= YEAR_START) & (index < H2_START)
        h2 = (index >= H2_START) & (index < YEAR_END)
        direct = fit_direct_power(
            wind_speed, truth, reference, h1, available
        )
        policy = Policy(**selected["policy"])
        candidate, gate, threshold = apply_policy(
            reference,
            direct,
            available,
            h2,
            policy,
            maximum_movement_ratio=args.maximum_movement_ratio,
        )
        evaluation = compare(truth, reference, candidate, h2)
        monthly = {
            str(month): compare(
                truth, reference, candidate, h2 & (index.month == month)
            )["delta"]
            for month in range(7, 13)
        }
        bootstrap = issue_bootstrap(
            truth,
            reference,
            candidate,
            issue_times,
            h2,
            samples=args.bootstrap,
        )
        locked = {
            "threshold_kwh": threshold,
            "changed_rows": int((gate & h2).sum()),
            "changed_ratio": float((gate & h2).sum() / h2.sum()),
            "mean_absolute_movement_kwh": float(
                np.abs(candidate[h2] - reference[h2]).mean()
            ),
            "maximum_absolute_movement_kwh": float(
                np.abs(candidate[h2] - reference[h2]).max()
            ),
            "evaluation": evaluation,
            "monthly_deltas": monthly,
            "positive_score_months": int(
                sum(delta["score"] > 0.0 for delta in monthly.values())
            ),
        }
        qualified = bool(
            min(evaluation["delta"][component] for component in COMPONENTS) > 0.0
            and locked["positive_score_months"] >= 5
            and bootstrap["quantiles"]["score"]["q05"] >= 0.0
            and bootstrap["quantiles"]["one_minus_nmae"]["q05"] >= 0.0
            and bootstrap["quantiles"]["ficr"]["q05"] >= 0.0
        )
        decision = (
            "qualified diagnostic; test implementation still requires an independent audit"
            if qualified
            else "reject; rolling-origin winner failed locked stability gates"
        )

    report = {
        "method": "multi-origin KMA UM isotonic power-curve overlay for group 1",
        "target": TARGET,
        "manifest_validation": manifest_validation,
        "validation_contract": {
            "selection_origins": [asdict(origin) for origin in origins()],
            "policy_count": len(policies()),
            "minimum_changed_ratio_per_origin": args.minimum_changed_ratio,
            "maximum_movement_ratio": args.maximum_movement_ratio,
            "locked_period_role": "contaminated historical benchmark; not an unseen holdout",
            "submission_written": False,
        },
        "rolling_eligible_policy_count": len(eligible),
        "selected_rolling_policy": selected,
        "top_rolling_policies": eligible[:10],
        "historical_h2": locked,
        "issue_block_bootstrap": bootstrap,
        "decision": decision,
        "submission": None,
    }
    output = _rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "selected": selected}, indent=2, default=str))


if __name__ == "__main__":
    main()
