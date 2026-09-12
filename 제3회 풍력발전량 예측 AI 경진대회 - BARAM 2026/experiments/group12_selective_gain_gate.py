"""One-sided selective gain gate for the rejected broad G1/G2 proposal.

The broad zero-sum reconciliation transferred its mean-error improvement to
the public set but reversed on FiCR.  This experiment does not retune that
public result.  It treats the broad proposal as a fixed action and predicts a
conservative conditional quantile of its *absolute-error gain*:

    gain = (|truth - incumbent| - |truth - proposal|) / capacity.

Only rows whose predicted lower gain quantile exceeds a fixed margin are
changed.  Whenever the gain is truly positive, absolute error decreases and
the row-level FiCR unit price cannot worsen.

Temporal contract:

* train lower-gain models on January-February 2024;
* select alpha/margin on March only;
* refit on Q1 and confirm on Q2;
* refit on H1 and confirm on H2;
* train on all 2024 for 2025 only after the strict confirmation gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.group12_difference_reconciliation import (
    CAPACITY,
    GROUPS,
    SEEDS,
    build_pairwise_features,
    fit_difference_ensemble,
    pair_delta,
    pair_issue_bootstrap,
    reconcile_pair,
)
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPHAS = (0.10, 0.20, 0.30)
DEFAULT_MARGINS = (0.0, 0.000025, 0.00005, 0.00010)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_gate_design(
    pair_features: pd.DataFrame,
    reference: dict[str, np.ndarray],
    proposal: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, slice]]:
    """Stack a common weather block plus group-specific action descriptors."""
    blocks: list[pd.DataFrame] = []
    slices: dict[str, slice] = {}
    start = 0
    pair = pair_features.reset_index(drop=True)
    common_difference = (
        reference[GROUPS[0]] - reference[GROUPS[1]]
    ) / CAPACITY
    proposed_difference = (
        proposal[GROUPS[0]] - proposal[GROUPS[1]]
    ) / CAPACITY
    for group_number, target in enumerate(GROUPS):
        block = pair.copy()
        base = np.asarray(reference[target], dtype=float)
        candidate = np.asarray(proposal[target], dtype=float)
        movement = (candidate - base) / CAPACITY_KWH[target]
        block["gate_group_id"] = float(group_number)
        block["gate_base_ratio"] = base / CAPACITY_KWH[target]
        block["gate_proposal_ratio"] = candidate / CAPACITY_KWH[target]
        block["gate_movement_ratio"] = movement
        block["gate_abs_movement_ratio"] = np.abs(movement)
        block["gate_common_base_difference"] = common_difference
        block["gate_proposed_difference"] = proposed_difference
        stop = start + len(block)
        slices[target] = slice(start, stop)
        blocks.append(block)
        start = stop
    design = pd.concat(blocks, axis=0, ignore_index=True).astype("float32")
    if design.isna().any().any() or not np.isfinite(design.to_numpy()).all():
        raise ValueError("selective-gain design is incomplete")
    return design, slices


def normalized_gain(
    truth: np.ndarray,
    reference: np.ndarray,
    proposal: np.ndarray,
    capacity: float,
) -> np.ndarray:
    return (
        np.abs(np.asarray(truth, dtype=float) - np.asarray(reference, dtype=float))
        - np.abs(np.asarray(truth, dtype=float) - np.asarray(proposal, dtype=float))
    ) / float(capacity)


def make_gain_model(
    alpha: float,
    *,
    seed: int,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=n_estimators,
        learning_rate=0.025,
        num_leaves=20,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.15,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def fit_gain_model(
    design: pd.DataFrame,
    gain: np.ndarray,
    truth: np.ndarray,
    rows: np.ndarray,
    *,
    alpha: float,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    rows = np.asarray(rows, dtype=bool)
    eligible = np.asarray(truth, dtype=float) >= 0.10 * CAPACITY
    selected = rows & eligible & np.isfinite(gain)
    if int(selected.sum()) < 1_000:
        raise ValueError("gain model has insufficient eligible training rows")
    model = make_gain_model(alpha, seed=13, n_estimators=n_estimators)
    actual_ratio = np.asarray(truth, dtype=float)[selected] / CAPACITY
    sample_weight = 0.5 + 0.5 * np.clip(actual_ratio, 0.0, 1.0)
    model.fit(
        design.loc[selected],
        gain[selected],
        sample_weight=sample_weight,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def apply_selective_gate(
    reference: dict[str, np.ndarray],
    proposal: dict[str, np.ndarray],
    lower_gain: np.ndarray,
    slices: dict[str, slice],
    *,
    margin: float,
    active_rows: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if margin < 0.0:
        raise ValueError("gain margin must be non-negative")
    candidate: dict[str, np.ndarray] = {}
    gates: dict[str, np.ndarray] = {}
    for target in GROUPS:
        target_lower = np.asarray(lower_gain[slices[target]], dtype=float)
        gate = target_lower > margin
        if active_rows is not None:
            gate &= np.asarray(active_rows, dtype=bool)
        base = np.asarray(reference[target], dtype=float)
        action = np.asarray(proposal[target], dtype=float)
        candidate[target] = np.where(gate, action, base)
        gates[target] = gate
    return candidate, gates


def group_deltas(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, dict[str, float]]:
    rows = np.asarray(rows, dtype=bool)
    output: dict[str, dict[str, float]] = {}
    for target in GROUPS:
        base = evaluate_group(
            truth[target][rows],
            reference[target][rows],
            CAPACITY_KWH[target],
        )
        treatment = evaluate_group(
            truth[target][rows],
            candidate[target][rows],
            CAPACITY_KWH[target],
        )
        output[target] = {
            "score": float(treatment.score - base.score),
            "one_minus_nmae": float(
                treatment.one_minus_nmae - base.one_minus_nmae
            ),
            "ficr": float(treatment.ficr - base.ficr),
        }
    return output


def gate_diagnostic(
    gain: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    gates: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=bool)
    groups: dict[str, Any] = {}
    for target in GROUPS:
        eligible = truth[target] >= 0.10 * CAPACITY_KWH[target]
        selected = rows & gates[target] & eligible
        values = gain[target][selected]
        groups[target] = {
            "selected_eligible_rows": int(selected.sum()),
            "eligible_coverage": float(
                selected.sum() / max(1, int((rows & eligible).sum()))
            ),
            "positive_gain_fraction": (
                None if not len(values) else float(np.mean(values > 0.0))
            ),
            "mean_normalized_gain": (
                None if not len(values) else float(np.mean(values))
            ),
        }
    return {"groups": groups}


def _policy_eligible(record: dict[str, Any]) -> bool:
    pair = record["pair_delta"]
    groups = record["group_deltas"]
    diagnostics = record["gate_diagnostic"]["groups"]
    coverage = [
        diagnostics[target]["eligible_coverage"] for target in GROUPS
    ]
    precision = [
        diagnostics[target]["positive_gain_fraction"] for target in GROUPS
    ]
    return bool(
        pair["score"] > 0.0
        and pair["one_minus_nmae"] >= 0.0
        and pair["ficr"] >= 0.0
        and all(groups[target]["score"] >= 0.0 for target in GROUPS)
        and min(coverage) >= 0.0025
        and max(coverage) <= 0.25
        and all(value is not None and value >= 0.60 for value in precision)
    )


def select_policy(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if _policy_eligible(record)]
    if not eligible:
        return {
            "selection_eligible": False,
            "alpha": None,
            "margin": None,
            "reason": "no March policy passed pair, group, coverage, and precision gates",
        }
    selected = max(
        eligible,
        key=lambda row: (
            row["pair_delta"]["score"],
            min(row["pair_delta"].values()),
            row["margin"],
            -row["alpha"],
        ),
    )
    return {
        "selection_eligible": True,
        "alpha": float(selected["alpha"]),
        "margin": float(selected["margin"]),
        "reason": "best March score among fully eligible one-sided policies",
    }


def _make_fixed_proposal(
    pair_features: pd.DataFrame,
    labels: pd.DataFrame,
    index: pd.DatetimeIndex,
    reference: dict[str, np.ndarray],
    *,
    n_estimators: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    train = (
        (pair_features.index >= pd.Timestamp("2023-01-01"))
        & (pair_features.index < pd.Timestamp("2024-01-01"))
    )
    expert, members = fit_difference_ensemble(
        pair_features.loc[train],
        labels,
        pair_features.reindex(index),
        seeds=SEEDS,
        n_estimators=n_estimators,
    )
    base_difference = (
        reference[GROUPS[0]] - reference[GROUPS[1]]
    ) / CAPACITY
    member_actions = members - base_difference
    unanimous = (np.min(member_actions, axis=0) > 0.0) | (
        np.max(member_actions, axis=0) < 0.0
    )
    gated_expert = np.where(unanimous, expert, base_difference)
    first, second, _ = reconcile_pair(
        reference[GROUPS[0]],
        reference[GROUPS[1]],
        gated_expert,
        weight=0.10,
        movement_cap_ratio=0.02,
    )
    return {GROUPS[0]: first, GROUPS[1]: second}, unanimous


def run(args: argparse.Namespace) -> dict[str, Any]:
    alphas = tuple(float(value) for value in args.alphas.split(",") if value)
    margins = tuple(float(value) for value in args.margins.split(",") if value)
    if not alphas or any(not 0.0 < value < 0.5 for value in alphas):
        raise ValueError("alphas must be within (0, 0.5)")
    if not margins or any(value < 0.0 for value in margins):
        raise ValueError("margins must be non-negative")

    baselines, truth_series, index, issue_series = (
        load_frozen_validation_baselines(
            _rooted(args.primary_cache),
            _rooted(args.residual_cache),
            _rooted(args.group3_cache),
        )
    )
    reference = {
        target: baselines[target].to_numpy(dtype=float)
        for target in GROUPS
    }
    truth = {
        target: truth_series[target].to_numpy(dtype=float)
        for target in GROUPS
    }
    labels = pd.read_csv(_rooted(args.labels), encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").sort_index()
    all_pair_features = build_pairwise_features(
        pd.read_pickle(_rooted(args.feature_cache))
    )
    validation_features = all_pair_features.reindex(index)
    proposal, unanimous = _make_fixed_proposal(
        all_pair_features,
        labels,
        index,
        reference,
        n_estimators=args.proposal_estimators,
    )
    design, slices = make_gate_design(
        validation_features,
        reference,
        proposal,
    )
    stacked_truth = np.concatenate([truth[target] for target in GROUPS])
    gains = {
        target: normalized_gain(
            truth[target],
            reference[target],
            proposal[target],
            CAPACITY_KWH[target],
        )
        for target in GROUPS
    }
    stacked_gain = np.concatenate([gains[target] for target in GROUPS])
    jan_feb = np.tile(
        np.asarray(index < pd.Timestamp("2024-03-01")),
        len(GROUPS),
    )
    march_stacked = np.tile(
        np.asarray(
            (index >= pd.Timestamp("2024-03-01"))
            & (index < pd.Timestamp("2024-04-01"))
        ),
        len(GROUPS),
    )
    march = np.asarray(
        (index >= pd.Timestamp("2024-03-01"))
        & (index < pd.Timestamp("2024-04-01"))
    )

    selection_records: list[dict[str, Any]] = []
    for alpha in alphas:
        model = fit_gain_model(
            design,
            stacked_gain,
            stacked_truth,
            jan_feb,
            alpha=alpha,
            n_estimators=args.gate_estimators,
        )
        lower_gain = np.asarray(model.predict(design), dtype=float)
        for margin in margins:
            candidate, gates = apply_selective_gate(
                reference,
                proposal,
                lower_gain,
                slices,
                margin=margin,
                active_rows=march,
            )
            selection_records.append(
                {
                    "alpha": float(alpha),
                    "margin": float(margin),
                    "pair_delta": pair_delta(
                        truth,
                        reference,
                        candidate,
                        march,
                    ),
                    "group_deltas": group_deltas(
                        truth,
                        reference,
                        candidate,
                        march,
                    ),
                    "gate_diagnostic": gate_diagnostic(
                        gains,
                        truth,
                        gates,
                        march,
                    ),
                }
            )
    selection = select_policy(selection_records)

    periods = {
        "q2": np.asarray(
            (index >= pd.Timestamp("2024-04-01"))
            & (index < pd.Timestamp("2024-07-01"))
        ),
        "h2": np.asarray(index >= pd.Timestamp("2024-07-01")),
    }
    candidate = {
        target: reference[target].copy()
        for target in GROUPS
    }
    combined_gates = {
        target: np.zeros(len(index), dtype=bool)
        for target in GROUPS
    }
    confirmation: dict[str, Any] = {}
    if selection["selection_eligible"]:
        alpha = float(selection["alpha"])
        margin = float(selection["margin"])
        training_periods = {
            "q2": np.tile(
                np.asarray(index < pd.Timestamp("2024-04-01")),
                len(GROUPS),
            ),
            "h2": np.tile(
                np.asarray(index < pd.Timestamp("2024-07-01")),
                len(GROUPS),
            ),
        }
        for name, rows in periods.items():
            model = fit_gain_model(
                design,
                stacked_gain,
                stacked_truth,
                training_periods[name],
                alpha=alpha,
                n_estimators=args.gate_estimators,
            )
            lower_gain = np.asarray(model.predict(design), dtype=float)
            period_candidate, gates = apply_selective_gate(
                reference,
                proposal,
                lower_gain,
                slices,
                margin=margin,
                active_rows=rows,
            )
            for target in GROUPS:
                candidate[target][rows] = period_candidate[target][rows]
                combined_gates[target][rows] = gates[target][rows]
            confirmation[name] = {
                "pair_delta": pair_delta(
                    truth,
                    reference,
                    period_candidate,
                    rows,
                ),
                "group_deltas": group_deltas(
                    truth,
                    reference,
                    period_candidate,
                    rows,
                ),
                "gate_diagnostic": gate_diagnostic(
                    gains,
                    truth,
                    gates,
                    rows,
                ),
            }

    confirmation_rows = periods["q2"] | periods["h2"]
    monthly = {
        str(month): pair_delta(
            truth,
            reference,
            candidate,
            confirmation_rows & np.asarray(index.month == month),
        )
        for month in range(4, 13)
    }
    bootstrap = (
        pair_issue_bootstrap(
            truth,
            reference,
            candidate,
            index,
            pd.DatetimeIndex(issue_series),
            n_bootstrap=args.n_bootstrap,
            seed=20260728,
        )
        if selection["selection_eligible"]
        else None
    )
    confirmation_group_pass = bool(
        confirmation
        and all(
            all(value >= 0.0 for value in row["pair_delta"].values())
            and all(
                all(value >= 0.0 for value in metrics.values())
                for metrics in row["group_deltas"].values()
            )
            for row in confirmation.values()
        )
    )
    positive_months = int(
        sum(row["score"] > 0.0 for row in monthly.values())
    )
    strict_pass = bool(
        selection["selection_eligible"]
        and confirmation_group_pass
        and positive_months >= 7
        and bootstrap is not None
        and bootstrap["q05"] >= 0.0
        and bootstrap["positive_fraction"] >= 0.90
    )

    candidate_record: dict[str, Any] | None = None
    if args.write_candidate:
        if not strict_pass:
            raise RuntimeError(
                "selective-gain candidate blocked because strict confirmation failed"
            )
        incumbent = pd.read_csv(
            _rooted(args.incumbent),
            encoding="utf-8-sig",
        )
        broad = pd.read_csv(
            _rooted(args.broad_proposal),
            encoding="utf-8-sig",
        )
        if not incumbent[["forecast_id", "forecast_kst_dtm"]].equals(
            broad[["forecast_id", "forecast_kst_dtm"]]
        ):
            raise ValueError("production proposal identifiers differ")
        output_index = pd.DatetimeIndex(
            pd.to_datetime(incumbent["forecast_kst_dtm"])
        )
        test_pair = build_pairwise_features(
            pd.read_pickle(_rooted(args.test_feature_cache))
        ).reindex(output_index)
        production_reference = {
            target: incumbent[target].to_numpy(dtype=float)
            for target in GROUPS
        }
        production_proposal = {
            target: broad[target].to_numpy(dtype=float)
            for target in GROUPS
        }
        production_design, production_slices = make_gate_design(
            test_pair,
            production_reference,
            production_proposal,
        )
        full_model = fit_gain_model(
            design,
            stacked_gain,
            stacked_truth,
            np.ones(len(design), dtype=bool),
            alpha=float(selection["alpha"]),
            n_estimators=args.gate_estimators,
        )
        production_lower = np.asarray(
            full_model.predict(production_design),
            dtype=float,
        )
        production_candidate, production_gates = apply_selective_gate(
            production_reference,
            production_proposal,
            production_lower,
            production_slices,
            margin=float(selection["margin"]),
        )
        output = incumbent.copy()
        for target in GROUPS:
            output[target] = production_candidate[target]
        output_path = _rooted(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        audit = CandidateValidator(load_config(ROOT)).audit(output_path)
        if not audit.valid:
            raise RuntimeError(f"candidate validator failed: {audit.errors}")
        candidate_record = {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "changed_rows": {
                target: int(production_gates[target].sum())
                for target in GROUPS
            },
            "candidate_validator": audit.to_dict(),
        }

    report = {
        "family": "group12_one_sided_selective_gain_gate",
        "method": (
            "conditional lower quantile of normalized absolute-error gain; "
            "abstain to incumbent unless lower gain exceeds margin"
        ),
        "contract": {
            "fixed_broad_action": (
                "weight-0.10 unanimous zero-sum reconciliation"
            ),
            "selection_training": "2024 January-February",
            "selection_validation": "2024 March only",
            "confirmation": "Q2 and H2 expanding-origin fits",
            "public_score_used_for_alpha_margin_or_row_gate": False,
            "public_result_used_only_to_close_broad_action_and_prioritize_abstention": True,
            "conformal_guarantee_claimed": False,
        },
        "fixed_proposal": {
            "unanimous_coverage": float(np.mean(unanimous)),
            "broad_public_status": "rejected_submission_1504383",
        },
        "selection": {
            **selection,
            "records": selection_records,
        },
        "confirmation": {
            "periods": confirmation,
            "monthly_pair_deltas": monthly,
            "positive_score_months": positive_months,
            "issue_block_bootstrap": bootstrap,
        },
        "promotion": {
            "strict_passed": strict_pass,
            "candidate_written": candidate_record is not None,
            "decision": (
                "eligible_strict"
                if strict_pass
                else "closed_no_candidate"
            ),
        },
        "candidate": candidate_record,
    }
    report_path = _rooted(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-cache",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument(
        "--test-feature-cache",
        default="artifacts_final/feature_cache/features_test.pkl",
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--primary-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--residual-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_msm_stencil_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--group3-cache",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--incumbent",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument(
        "--broad-proposal",
        default=(
            "artifacts_final/candidates/"
            "group12_difference_reconciliation_unanimous_w10_20260728.csv"
        ),
    )
    parser.add_argument(
        "--alphas",
        default=",".join(str(value) for value in DEFAULT_ALPHAS),
    )
    parser.add_argument(
        "--margins",
        default=",".join(str(value) for value in DEFAULT_MARGINS),
    )
    parser.add_argument("--proposal-estimators", type=int, default=500)
    parser.add_argument("--gate-estimators", type=int, default=350)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--write-candidate", action="store_true")
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/candidates/"
            "group12_selective_gain_gate_20260728.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "group12_selective_gain_gate_20260728.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "selection": {
                    key: value
                    for key, value in report["selection"].items()
                    if key != "records"
                },
                "confirmation": report["confirmation"],
                "promotion": report["promotion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
