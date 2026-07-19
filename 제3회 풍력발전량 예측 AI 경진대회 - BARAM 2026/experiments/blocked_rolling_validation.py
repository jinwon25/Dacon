from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from experiments.exact_oof_meta_gate import H2_START, _evaluate_period, apply_meta_gate
from experiments.exact_oof_meta_gate_sweep import Policy, _prepare_validation
from src.metrics import CAPACITY_KWH, evaluate_group


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
REFERENCE_POLICY = Policy(threshold=0.55, alpha=0.25)
FINE_POLICY = Policy(threshold=0.545, alpha=0.50)


@dataclass(frozen=True)
class PublicProbe:
    name: str
    local_macro_delta: dict[str, float]
    public_delta: dict[str, float]


def load_issue_times(path: Path, timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Align the real NWP availability timestamp to each forecast timestamp.

    A forecast issue is the dependency unit: rows produced by the same NWP run
    must never be split between train and validation merely because midnight or
    a month boundary falls inside its horizon.
    """
    frame = pd.read_csv(
        path,
        usecols=["forecast_kst_dtm", "data_available_kst_dtm"],
        encoding="utf-8-sig",
    ).drop_duplicates()
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(frame["data_available_kst_dtm"])
    counts = frame.groupby("forecast_kst_dtm")["data_available_kst_dtm"].nunique()
    ambiguous = counts[counts != 1]
    if not ambiguous.empty:
        raise ValueError(
            f"Each forecast timestamp must map to one issue time; found {len(ambiguous)} ambiguous rows"
        )
    mapping = frame.set_index("forecast_kst_dtm")["data_available_kst_dtm"]
    aligned = mapping.reindex(timestamps)
    if aligned.isna().any():
        missing = timestamps[aligned.isna().to_numpy()]
        raise ValueError(f"Issue-time source is missing {len(missing)} validation timestamps")
    return pd.DatetimeIndex(aligned.to_numpy())


def _season_label(timestamp: pd.Timestamp) -> str:
    month = timestamp.month
    if month in (12, 1, 2):
        # December belongs to the following meteorological winter.
        year = timestamp.year + int(month == 12)
        season = "DJF"
    elif month <= 5:
        year, season = timestamp.year, "MAM"
    elif month <= 8:
        year, season = timestamp.year, "JJA"
    else:
        year, season = timestamp.year, "SON"
    return f"{year}-{season}"


def assign_issue_blocks(
    timestamps: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign whole issue cycles to target-month and target-season blocks."""
    if len(timestamps) != len(issue_times):
        raise ValueError("timestamps and issue_times must have the same length")
    frame = pd.DataFrame({"target": timestamps, "issue": issue_times})
    # The median target time represents the centre of an issue horizon. This
    # keeps the entire run intact when its first/last target crosses a boundary.
    representatives = frame.groupby("issue", sort=False)["target"].median()
    month_map = representatives.dt.to_period("M").astype(str)
    season_map = representatives.map(_season_label)
    month = frame["issue"].map(month_map).to_numpy(dtype=str)
    season = frame["issue"].map(season_map).to_numpy(dtype=str)
    return month, season


def _metric_delta(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    return _evaluate_period(truth, reference, candidate, rows)


def issue_block_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    issue_times: pd.DatetimeIndex,
    strata: np.ndarray,
    rows: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap complete issue cycles, stratified by season.

    Hour-wise bootstrap understates dependence between horizons from one NWP
    issue. Sampling whole cycles preserves that dependence and the observed
    seasonal mixture.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rows = np.asarray(rows, dtype=bool)
    issue_values = np.asarray(issue_times)
    strata = np.asarray(strata)
    positions: dict[tuple[str, np.datetime64], np.ndarray] = {}
    stratum_issues: dict[str, np.ndarray] = {}
    for stratum in sorted(set(strata[rows])):
        issues = np.unique(issue_values[rows & (strata == stratum)])
        stratum_issues[stratum] = issues
        for issue in issues:
            positions[(stratum, issue)] = np.flatnonzero(
                rows & (strata == stratum) & (issue_values == issue)
            )
    if not stratum_issues:
        raise ValueError("Bootstrap rows are empty")
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled_rows: list[np.ndarray] = []
        for stratum, issues in stratum_issues.items():
            sampled = rng.choice(issues, size=len(issues), replace=True)
            sampled_rows.extend(positions[(stratum, issue)] for issue in sampled)
        selected = np.concatenate(sampled_rows)
        values[iteration] = (
            evaluate_group(truth[selected], candidate[selected], CAPACITY).score
            - evaluate_group(truth[selected], reference[selected], CAPACITY).score
        )
    return {
        "n_bootstrap": n_bootstrap,
        "positive_fraction": float(np.mean(values > 0.0)),
        "q025": float(np.quantile(values, 0.025)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q975": float(np.quantile(values, 0.975)),
    }


def evaluate_blocked_rolling(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    timestamps: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
    rows: np.ndarray,
    n_bootstrap: int,
    seed: int = 20260718,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=bool)
    month_block, season_block = assign_issue_blocks(timestamps, issue_times)

    def evaluate_blocks(blocks: np.ndarray) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for block in dict.fromkeys(blocks[rows]):
            mask = rows & (blocks == block)
            if int(mask.sum()) < 24:
                continue
            output[str(block)] = {
                "rows": int(mask.sum()),
                "issue_cycles": int(pd.Index(issue_times[mask]).nunique()),
                **_metric_delta(truth, reference, candidate, mask),
            }
        return output

    monthly = evaluate_blocks(month_block)
    seasonal = evaluate_blocks(season_block)
    ordered_seasons = list(seasonal)
    rolling_folds: list[dict[str, Any]] = []
    for position in range(1, len(ordered_seasons)):
        train_blocks = ordered_seasons[:position]
        validation_block = ordered_seasons[position]
        train = rows & np.isin(season_block, train_blocks)
        validation = rows & (season_block == validation_block)
        # This assertion is the leakage guard: an issue cycle cannot occur in
        # both sides even around a meteorological-season boundary.
        overlap = set(issue_times[train]).intersection(issue_times[validation])
        if overlap:
            raise AssertionError("An NWP issue cycle was split across a rolling fold")
        rolling_folds.append(
            {
                "train_seasons": train_blocks,
                "validation_season": validation_block,
                "train_rows": int(train.sum()),
                "validation_rows": int(validation.sum()),
                "issue_overlap": 0,
                "metrics": _metric_delta(truth, reference, candidate, validation),
            }
        )

    monthly_deltas = {
        block: float(record["delta"]["score"]) for block, record in monthly.items()
    }
    worst_month = min(monthly_deltas, key=monthly_deltas.get)
    bootstrap = issue_block_bootstrap(
        truth,
        reference,
        candidate,
        issue_times,
        season_block,
        rows,
        n_bootstrap,
        seed,
    )
    overall = _metric_delta(truth, reference, candidate, rows)
    robust = bool(
        overall["delta"]["score"] > 0.0
        and overall["delta"]["one_minus_nmae"] >= 0.0
        and overall["delta"]["ficr"] >= 0.0
        and monthly_deltas[worst_month] >= 0.0
        and bootstrap["q05"] >= 0.0
        and bootstrap["positive_fraction"] >= 0.90
    )
    return {
        "contract": {
            "dependency_unit": "data_available_kst_dtm (complete NWP issue cycle)",
            "calendar_unit": "issue-cycle target-centre month/meteorological season",
            "rolling_scheme": "expanding prior seasons -> next complete season",
            "promotion_requirements": [
                "overall score, 1-NMAE, and FICR deltas are non-negative",
                "worst monthly score delta is non-negative",
                "issue-block bootstrap q05 is non-negative",
                "issue-block bootstrap positive fraction is at least 0.90",
            ],
        },
        "issue_integrity": {
            "evaluated_rows": int(rows.sum()),
            "unique_issue_cycles": int(pd.Index(issue_times[rows]).nunique()),
            "missing_issue_times": int(pd.isna(issue_times[rows]).sum()),
            "split_issue_cycles": 0,
        },
        "overall": overall,
        "monthly": monthly,
        "monthly_worst_case": {
            "block": worst_month,
            "score_delta": monthly_deltas[worst_month],
            "positive_fraction": float(
                np.mean(np.asarray(list(monthly_deltas.values())) > 0.0)
            ),
        },
        "seasonal": seasonal,
        "rolling_folds": rolling_folds,
        "issue_block_bootstrap": bootstrap,
        "robustness_passed": robust,
    }


def public_transfer_audit(probes: Iterable[PublicProbe]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for probe in probes:
        local_score = float(probe.local_macro_delta["score"])
        public_score = float(probe.public_delta["score"])
        ratio = public_score / local_score if abs(local_score) > 1e-15 else None
        records.append(
            {
                "name": probe.name,
                "local_macro_delta": probe.local_macro_delta,
                "public_delta": probe.public_delta,
                "score_sign_agrees": bool(np.sign(local_score) == np.sign(public_score)),
                "score_transfer_ratio": ratio,
                "local_optimism": float(local_score - public_score),
            }
        )
    if not records:
        raise ValueError("At least one public probe is required")
    ratios = [
        float(record["score_transfer_ratio"])
        for record in records
        if record["score_transfer_ratio"] is not None
        and float(record["score_transfer_ratio"]) > 0.0
    ]
    sign_fraction = float(np.mean([record["score_sign_agrees"] for record in records]))
    # With a sign reversal in the audit set, a positive local delta has no
    # defensible automatic public projection. Human-controlled probes remain possible.
    conservative_ratio = 0.0 if sign_fraction < 1.0 else float(min(ratios, default=0.0))
    return {
        "probes": records,
        "summary": {
            "probe_count": len(records),
            "sign_agreement_fraction": sign_fraction,
            "sign_reversal_count": int(
                sum(not record["score_sign_agrees"] for record in records)
            ),
            "conservative_auto_projection_ratio": conservative_ratio,
            "automatic_public_projection_trusted": bool(
                len(records) >= 3 and sign_fraction == 1.0 and conservative_ratio > 0.0
            ),
        },
    }


def _component_delta(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key] - reference[key])
        for key in ("score", "one_minus_nmae", "ficr")
    }


def run(
    labels_path: Path,
    driver_cache_path: Path,
    meta_cache_path: Path,
    issue_source_path: Path,
    sweep_report_path: Path,
    settlement_report_path: Path,
    results_path: Path,
    output_path: Path,
    n_bootstrap: int,
) -> dict[str, Any]:
    (
        _labels,
        index,
        truth,
        _group_1,
        _group_2,
        _base,
        member,
        current,
        _features,
        action,
    ) = _prepare_validation(labels_path, driver_cache_path)
    meta_cache = np.load(meta_cache_path)
    cache_index = pd.DatetimeIndex(pd.to_datetime(meta_cache["valid_index_ns"]))
    if not cache_index.equals(index):
        raise ValueError("Meta-gate and exact-driver OOF cache indices do not match")
    probability = meta_cache["valid_probability"].astype(float)
    reference, _ = apply_meta_gate(
        current,
        member,
        action,
        probability,
        threshold=REFERENCE_POLICY.threshold,
        extra_alpha=REFERENCE_POLICY.alpha,
    )
    candidate, _ = apply_meta_gate(
        current,
        member,
        action,
        probability,
        threshold=FINE_POLICY.threshold,
        extra_alpha=FINE_POLICY.alpha,
    )
    issue_times = load_issue_times(issue_source_path, index)
    locked = (index >= H2_START) & (truth >= 0.10 * CAPACITY)
    validation = evaluate_blocked_rolling(
        truth,
        reference,
        candidate,
        index,
        issue_times,
        locked,
        n_bootstrap=n_bootstrap,
    )

    sweep = json.loads(sweep_report_path.read_text(encoding="utf-8"))
    settlement = json.loads(settlement_report_path.read_text(encoding="utf-8"))
    results = pd.read_csv(results_path, encoding="utf-8-sig")
    by_id = results.dropna(subset=["submission_id"]).assign(
        submission_id=lambda x: x["submission_id"].astype(int)
    ).set_index("submission_id")
    reference_public = by_id.loc[1494307]
    fine_public = by_id.loc[1494670]
    settlement_public = by_id.loc[1494668]
    metric_keys = ("score", "one_minus_nmae", "ficr")
    public_reference = {key: float(reference_public[key]) for key in metric_keys}
    public_fine = {key: float(fine_public[key]) for key in metric_keys}
    public_settlement = {key: float(settlement_public[key]) for key in metric_keys}
    fine_local_group3 = sweep["locked_incremental_over_reference"]["metrics"]["delta"]
    transfer = public_transfer_audit(
        [
            PublicProbe(
                "fine_meta_gate",
                {key: float(fine_local_group3[key]) / 3.0 for key in metric_keys},
                _component_delta(public_fine, public_reference),
            ),
            PublicProbe(
                "broad_settlement_composite",
                {
                    key: float(settlement["locked_h2_macro_delta"][key])
                    for key in metric_keys
                },
                _component_delta(public_settlement, public_reference),
            ),
        ]
    )
    public_best = public_fine["score"]
    target_gap = 0.65 - public_best
    result: dict[str, Any] = {
        "method": "issue-time and meteorological-season blocked rolling OOF audit",
        "candidate": {
            "reference_policy": REFERENCE_POLICY.to_dict(),
            "candidate_policy": FINE_POLICY.to_dict(),
            "evaluation_period": "locked H2 only; H1-trained meta probability cache",
        },
        "blocked_validation": validation,
        "public_transfer_audit": transfer,
        "target_audit": {
            "public_best": public_best,
            "target": 0.65,
            "remaining_gap": target_gap,
            "last_verified_public_uplift": float(public_fine["score"] - public_reference["score"]),
            "required_multiple_of_last_uplift": float(
                target_gap / (public_fine["score"] - public_reference["score"])
            ),
        },
        "promotion": {
            "eligible": bool(
                validation["robustness_passed"]
                and transfer["summary"]["automatic_public_projection_trusted"]
            ),
            "blockers": [
                reason
                for condition, reason in (
                    (
                        validation["robustness_passed"],
                        "candidate fails the worst-month or issue-block bootstrap guard",
                    ),
                    (
                        transfer["summary"]["automatic_public_projection_trusted"],
                        "public-transfer evidence contains a sign reversal and too few probes",
                    ),
                )
                if not condition
            ],
        },
    }
    result["decision"] = (
        "Eligible for an automatic submission candidate."
        if result["promotion"]["eligible"]
        else "Do not generate another calibration submission; require a new predictive signal."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz")
    parser.add_argument("--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz")
    parser.add_argument("--issue-source", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--sweep-report",
        default="artifacts_final/meta_gate_sweep/meta_gate_policy_sweep_report.json",
    )
    parser.add_argument(
        "--settlement-report",
        default="artifacts_final/calibration/settlement_composite_report.json",
    )
    parser.add_argument("--results", default="submissions/results.csv")
    parser.add_argument(
        "--output",
        default="artifacts_final/validation/blocked_rolling_validation_report.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    report = run(
        Path(args.labels),
        Path(args.driver_cache),
        Path(args.meta_cache),
        Path(args.issue_source),
        Path(args.sweep_report),
        Path(args.settlement_report),
        Path(args.results),
        Path(args.output),
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
