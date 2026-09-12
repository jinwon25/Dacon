"""Compose public-factor probes for a directional FiCR calibration.

The active public incumbent already contains three independently positive
signals:

* the pooled KMA/JMA group-1 expert,
* the pooled KMA/JMA group-2 expert, and
* the KMA UMRG group-3 power-curve overlay.

This module changes predictions only in those frozen signal coordinates.  It
creates two complementary factor probes and their exact composition:

* probe A changes group 1 only;
* probe B changes groups 2 and 3 only;
* the target changes all three groups.

Because the official score is a macro average over groups, the target public
metrics are exactly ``probe_a + probe_b - incumbent`` once both probes have
been scored.  The historical parameter screen is explicitly contaminated and
the candidates therefore remain manual public probes, never automatic private
promotions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.compose_residual_stack_candidate import (
    apply_capped_residual_stack,
)
from src.metrics import CAPACITY_KWH, evaluate_competition


ROOT = Path(__file__).resolve().parents[1]
ID_COLUMNS = ["forecast_id", "forecast_kst_dtm"]
TARGETS = tuple(CAPACITY_KWH)
COMPONENTS = ("score", "one_minus_nmae", "ficr")


@dataclass(frozen=True)
class DirectionalStepPolicy:
    direction: str
    step_ratio: float
    minimum_prediction_ratio: float = 0.0
    minimum_signal_ratio: float = 0.0


TARGET_POLICIES = {
    "kpx_group_1": DirectionalStepPolicy(
        direction="positive",
        step_ratio=0.0400,
        minimum_prediction_ratio=0.30,
        minimum_signal_ratio=0.0050,
    ),
    "kpx_group_2": DirectionalStepPolicy(
        direction="both",
        step_ratio=0.0075,
        minimum_prediction_ratio=0.60,
        minimum_signal_ratio=0.00125,
    ),
    "kpx_group_3": DirectionalStepPolicy(
        direction="uniform_up",
        step_ratio=0.0075,
    ),
}

SAFE_POLICIES = {
    "kpx_group_2": DirectionalStepPolicy(
        direction="negative",
        step_ratio=0.0075,
        minimum_prediction_ratio=0.60,
        minimum_signal_ratio=0.00125,
    ),
    "kpx_group_3": DirectionalStepPolicy(
        direction="positive_up",
        step_ratio=0.0075,
        minimum_prediction_ratio=0.25,
    ),
}


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_directional_step(
    baseline: np.ndarray,
    signal: np.ndarray,
    *,
    capacity: float,
    policy: DirectionalStepPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one bounded step in a frozen signal coordinate."""
    baseline = np.asarray(baseline, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if baseline.shape != signal.shape:
        raise ValueError("baseline and signal vectors do not align")
    if policy.direction not in {
        "uniform_up",
        "positive",
        "negative",
        "both",
        "positive_up",
    }:
        raise ValueError(f"unsupported direction: {policy.direction}")
    if not 0.0 < policy.step_ratio <= 0.05:
        raise ValueError("step ratio must lie in (0, 0.05]")
    if not 0.0 <= policy.minimum_prediction_ratio <= 1.0:
        raise ValueError("minimum prediction ratio must lie in [0, 1]")
    if not 0.0 <= policy.minimum_signal_ratio <= 1.0:
        raise ValueError("minimum signal ratio must lie in [0, 1]")

    gate = baseline / capacity >= policy.minimum_prediction_ratio
    gate &= np.abs(signal) / capacity >= policy.minimum_signal_ratio
    if policy.direction in {"positive", "positive_up"}:
        gate &= signal > 0.0
    elif policy.direction == "negative":
        gate &= signal < 0.0

    if policy.direction in {"uniform_up", "positive_up"}:
        direction = np.ones_like(signal)
    else:
        direction = np.sign(signal)
    candidate = np.clip(
        baseline + gate * direction * policy.step_ratio * capacity,
        0.0,
        capacity,
    )
    return candidate, gate


def complementary_factor_frames(
    incumbent: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a target into G1-only and G2+G3 complementary probes."""
    if not incumbent[ID_COLUMNS].equals(target[ID_COLUMNS]):
        raise ValueError("incumbent and target IDs differ")
    group1 = incumbent.copy()
    group1["kpx_group_1"] = target["kpx_group_1"]
    group23 = incumbent.copy()
    for group in ("kpx_group_2", "kpx_group_3"):
        group23[group] = target[group]
    return group1, group23


def exact_public_composition(
    incumbent: dict[str, float],
    group1_probe: dict[str, float],
    group23_probe: dict[str, float],
) -> dict[str, float]:
    """Return exact macro metrics for the complementary composition."""
    return {
        component: float(
            group1_probe[component]
            + group23_probe[component]
            - incumbent[component]
        )
        for component in COMPONENTS
    }


def _delta(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, Any]:
    before = evaluate_competition(
        {group: truth[group][rows] for group in TARGETS},
        {group: reference[group][rows] for group in TARGETS},
    )
    after = evaluate_competition(
        {group: truth[group][rows] for group in TARGETS},
        {group: candidate[group][rows] for group in TARGETS},
    )
    return {
        "before": {
            component: float(before[component]) for component in COMPONENTS
        },
        "after": {
            component: float(after[component]) for component in COMPONENTS
        },
        "delta": {
            component: float(after[component] - before[component])
            for component in COMPONENTS
        },
        "groups": {
            group: {
                component: float(
                    after["groups"][group][component]
                    - before["groups"][group][component]
                )
                for component in COMPONENTS
            }
            for group in TARGETS
        },
    }


def _issue_bootstrap(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    issue_ns: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    issue_ns = np.asarray(issue_ns, dtype=np.int64)
    cycles = np.unique(issue_ns)
    cycle_rows = [np.flatnonzero(issue_ns == cycle) for cycle in cycles]
    rng = np.random.default_rng(seed)
    values = np.empty((n_bootstrap, len(COMPONENTS)), dtype=float)
    for iteration in range(n_bootstrap):
        selected = rng.integers(0, len(cycle_rows), size=len(cycle_rows))
        rows = np.concatenate([cycle_rows[position] for position in selected])
        delta = _delta(truth, reference, candidate, rows)["delta"]
        values[iteration] = [
            delta[component]
            for component in COMPONENTS
        ]
    return {
        "n_bootstrap": int(n_bootstrap),
        "n_issue_cycles": int(len(cycles)),
        "score_positive_fraction": float(np.mean(values[:, 0] > 0.0)),
        "all_component_positive_fraction": float(
            np.mean(np.min(values, axis=1) > 0.0)
        ),
        "components": {
            component: {
                "q025": float(np.quantile(values[:, position], 0.025)),
                "q05": float(np.quantile(values[:, position], 0.05)),
                "median": float(np.median(values[:, position])),
                "q95": float(np.quantile(values[:, position], 0.95)),
                "q975": float(np.quantile(values[:, position], 0.975)),
            }
            for position, component in enumerate(COMPONENTS)
        },
    }


def _write_and_audit(
    frame: pd.DataFrame,
    path: Path,
    validator: CandidateValidator,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    audit = validator.audit(path)
    if not audit.valid:
        raise RuntimeError(f"candidate validation failed: {audit.errors}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "audit": audit.to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "incumbent": _rooted(args.incumbent),
        "primary_submission": _rooted(args.primary_submission),
        "pre_pooled_submission": _rooted(args.pre_pooled_submission),
        "pre_kma_submission": _rooted(args.pre_kma_submission),
        "primary_cache": _rooted(args.primary_cache),
        "residual_cache": _rooted(args.residual_cache),
        "group3_cache": _rooted(args.group3_cache),
    }
    frames = {
        name: pd.read_csv(path, encoding="utf-8-sig")
        for name, path in paths.items()
        if name.endswith("submission") or name == "incumbent"
    }
    incumbent = frames["incumbent"]
    for name, frame in frames.items():
        if not incumbent[ID_COLUMNS].equals(frame[ID_COLUMNS]):
            raise ValueError(f"{name} IDs differ from the incumbent")

    primary = np.load(paths["primary_cache"], allow_pickle=False)
    residual = np.load(paths["residual_cache"], allow_pickle=False)
    group3 = np.load(paths["group3_cache"], allow_pickle=False)
    index_ns = primary["kpx_group_1__index_ns"]
    if not np.array_equal(
        index_ns,
        primary["kpx_group_2__index_ns"],
    ):
        raise ValueError("primary group indexes differ")
    if not np.array_equal(
        index_ns,
        residual["kpx_group_1__index_ns"],
    ):
        raise ValueError("primary and residual indexes differ")
    if not np.array_equal(index_ns, group3["index_ns"]):
        raise ValueError("primary and group-3 indexes differ")
    index = pd.DatetimeIndex(pd.to_datetime(index_ns))

    truth = {
        "kpx_group_1": primary["kpx_group_1__truth"].astype(float),
        "kpx_group_2": primary["kpx_group_2__truth"].astype(float),
        "kpx_group_3": group3["truth"].astype(float),
    }
    validation_reference = {
        "kpx_group_1": apply_capped_residual_stack(
            primary["kpx_group_1__reference"],
            primary["kpx_group_1__candidate"],
            residual["kpx_group_1__candidate"],
            residual_weight=0.125,
            capacity=CAPACITY_KWH["kpx_group_1"],
            movement_cap_ratio=0.05,
        ),
        "kpx_group_2": primary["kpx_group_2__candidate"].astype(float),
        "kpx_group_3": group3["rolling_candidate"].astype(float),
    }
    validation_signal = {
        "kpx_group_1": (
            primary["kpx_group_1__candidate"].astype(float)
            - primary["kpx_group_1__reference"].astype(float)
        ),
        "kpx_group_2": (
            validation_reference["kpx_group_2"]
            - primary["kpx_group_2__reference"].astype(float)
        ),
        "kpx_group_3": (
            validation_reference["kpx_group_3"]
            - group3["public_fine"].astype(float)
        ),
    }
    production_signal = {
        "kpx_group_1": (
            frames["primary_submission"]["kpx_group_1"].to_numpy(dtype=float)
            - frames["pre_pooled_submission"]["kpx_group_1"].to_numpy(
                dtype=float
            )
        ),
        "kpx_group_2": (
            incumbent["kpx_group_2"].to_numpy(dtype=float)
            - frames["pre_pooled_submission"]["kpx_group_2"].to_numpy(
                dtype=float
            )
        ),
        "kpx_group_3": (
            incumbent["kpx_group_3"].to_numpy(dtype=float)
            - frames["pre_kma_submission"]["kpx_group_3"].to_numpy(dtype=float)
        ),
    }

    validation_target: dict[str, np.ndarray] = {}
    production_target = incumbent.copy()
    target_gates: dict[str, np.ndarray] = {}
    for group, policy in TARGET_POLICIES.items():
        validation_target[group], validation_gate = apply_directional_step(
            validation_reference[group],
            validation_signal[group],
            capacity=CAPACITY_KWH[group],
            policy=policy,
        )
        production, production_gate = apply_directional_step(
            incumbent[group].to_numpy(dtype=float),
            production_signal[group],
            capacity=CAPACITY_KWH[group],
            policy=policy,
        )
        production_target[group] = production
        target_gates[group] = production_gate

    validation_safe = {
        group: values.copy()
        for group, values in validation_reference.items()
    }
    production_safe = incumbent.copy()
    safe_gates: dict[str, np.ndarray] = {}
    for group, policy in SAFE_POLICIES.items():
        validation_safe[group], _ = apply_directional_step(
            validation_reference[group],
            validation_signal[group],
            capacity=CAPACITY_KWH[group],
            policy=policy,
        )
        production, production_gate = apply_directional_step(
            incumbent[group].to_numpy(dtype=float),
            production_signal[group],
            capacity=CAPACITY_KWH[group],
            policy=policy,
        )
        production_safe[group] = production
        safe_gates[group] = production_gate

    periods = {
        "q1": np.asarray(index < pd.Timestamp("2024-04-01")),
        "q2": np.asarray(
            (index >= pd.Timestamp("2024-04-01"))
            & (index < pd.Timestamp("2024-07-01"))
        ),
        "h2": np.asarray(index >= pd.Timestamp("2024-07-01")),
        "full": np.ones(len(index), dtype=bool),
    }
    target_periods = {
        name: _delta(
            truth,
            validation_reference,
            validation_target,
            rows,
        )
        for name, rows in periods.items()
    }
    safe_periods = {
        name: _delta(
            truth,
            validation_reference,
            validation_safe,
            rows,
        )
        for name, rows in periods.items()
    }
    target_monthly = {
        str(month): _delta(
            truth,
            validation_reference,
            validation_target,
            np.asarray(index.month == month),
        )
        for month in range(1, 13)
    }
    safe_monthly = {
        str(month): _delta(
            truth,
            validation_reference,
            validation_safe,
            np.asarray(index.month == month),
        )
        for month in range(1, 13)
    }
    bootstrap_target = _issue_bootstrap(
        truth,
        validation_reference,
        validation_target,
        primary["kpx_group_1__issue_ns"],
        n_bootstrap=args.n_bootstrap,
        seed=20_260_727,
    )
    bootstrap_safe = _issue_bootstrap(
        truth,
        validation_reference,
        validation_safe,
        primary["kpx_group_1__issue_ns"],
        n_bootstrap=args.n_bootstrap,
        seed=20_260_728,
    )

    group1_probe, group23_probe = complementary_factor_frames(
        incumbent,
        production_target,
    )
    validator = CandidateValidator(load_config(ROOT))
    output_dir = _rooted(args.output_dir)
    candidates = {
        "target": _write_and_audit(
            production_target,
            output_dir / args.target_name,
            validator,
        ),
        "group1_probe": _write_and_audit(
            group1_probe,
            output_dir / args.group1_probe_name,
            validator,
        ),
        "group23_probe": _write_and_audit(
            group23_probe,
            output_dir / args.group23_probe_name,
            validator,
        ),
        "safe": _write_and_audit(
            production_safe,
            output_dir / args.safe_name,
            validator,
        ),
    }

    local_delta = target_periods["full"]["delta"]["score"]
    incumbent_public_metrics = {
        "score": float(args.incumbent_public_score),
        "one_minus_nmae": float(args.incumbent_public_one_minus_nmae),
        "ficr": float(args.incumbent_public_ficr),
    }
    projected_public_metrics = {
        component: float(
            incumbent_public_metrics[component]
            + target_periods["full"]["delta"][component]
        )
        for component in COMPONENTS
    }
    report = {
        "family": "public_confirmed_directional_ficr_calibration",
        "decision_tier": "manual_public_factor_probe",
        "selection_warning": (
            "Policy thresholds and step sizes were screened on the repeatedly "
            "inspected 2024 OOF benchmark. The local 0.65 projection is not a "
            "private or public guarantee."
        ),
        "contract": {
            "official_group_macro_additivity_used": True,
            "public_score_used_to_identify_signal_families": True,
            "public_score_used_to_choose_new_step_parameters": False,
            "historical_oof_is_contaminated": True,
            "test_actual_generation_used": False,
            "automatic_submission_eligible": False,
        },
        "sources": {
            name: path.relative_to(ROOT).as_posix()
            for name, path in paths.items()
        },
        "policies": {
            "target": {
                group: asdict(policy)
                for group, policy in TARGET_POLICIES.items()
            },
            "safe": {
                group: asdict(policy)
                for group, policy in SAFE_POLICIES.items()
            },
        },
        "historical_validation": {
            "target": {
                "periods": target_periods,
                "monthly": target_monthly,
                "positive_score_months": int(
                    sum(
                        row["delta"]["score"] > 0.0
                        for row in target_monthly.values()
                    )
                ),
                "worst_month_score_delta": float(
                    min(
                        row["delta"]["score"]
                        for row in target_monthly.values()
                    )
                ),
                "issue_block_bootstrap": bootstrap_target,
                "projected_public_metrics_if_local_delta_transfers_one_to_one": (
                    projected_public_metrics
                ),
            },
            "safe": {
                "periods": safe_periods,
                "monthly": safe_monthly,
                "positive_score_months": int(
                    sum(
                        row["delta"]["score"] > 0.0
                        for row in safe_monthly.values()
                    )
                ),
                "worst_month_score_delta": float(
                    min(
                        row["delta"]["score"]
                        for row in safe_monthly.values()
                    )
                ),
                "issue_block_bootstrap": bootstrap_safe,
            },
        },
        "production": {
            "target_changed_rows": {
                group: int(gate.sum()) for group, gate in target_gates.items()
            },
            "safe_changed_rows": {
                group: int(gate.sum()) for group, gate in safe_gates.items()
            },
            "candidates": candidates,
        },
        "public_factor_protocol": {
            "incumbent_metrics": incumbent_public_metrics,
            "submit_first": ["group1_probe", "group23_probe"],
            "after_two_results": (
                "For each metric compute group1_probe + group23_probe - "
                "incumbent. Submit target third only if the exact composed "
                "score clears the chosen risk threshold."
            ),
            "formula": (
                "target_metric = group1_probe_metric + "
                "group23_probe_metric - incumbent_metric"
            ),
        },
    }
    report_path = _rooted(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--incumbent",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument(
        "--primary-submission",
        default=(
            "artifacts_final/candidates/"
            "kma_jma_pooled_all3_g1g2_nearstable_20260726.csv"
        ),
    )
    parser.add_argument(
        "--pre-pooled-submission",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
    )
    parser.add_argument(
        "--pre-kma-submission",
        default="submissions/blend_best_crossg3_traj_meta_finesweep.csv",
    )
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
    parser.add_argument("--output-dir", default="artifacts_final/candidates")
    parser.add_argument(
        "--target-name",
        default="public_directional_ficr065_target_20260727.csv",
    )
    parser.add_argument(
        "--group1-probe-name",
        default="public_directional_ficr065_g1_probe_20260727.csv",
    )
    parser.add_argument(
        "--group23-probe-name",
        default="public_directional_ficr065_g2g3_probe_20260727.csv",
    )
    parser.add_argument(
        "--safe-name",
        default="public_directional_safe_g2g3_20260727.csv",
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "public_directional_ficr065_20260727.json"
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=3_000)
    parser.add_argument("--incumbent-public-score", type=float, default=0.6461250914)
    parser.add_argument(
        "--incumbent-public-one-minus-nmae",
        type=float,
        default=0.8757842477,
    )
    parser.add_argument(
        "--incumbent-public-ficr",
        type=float,
        default=0.4164659352,
    )
    args = parser.parse_args()
    report = run(args)
    summary = {
        "decision_tier": report["decision_tier"],
        "historical_target_full": report["historical_validation"]["target"][
            "periods"
        ]["full"]["delta"],
        "historical_target_worst_month": report["historical_validation"][
            "target"
        ]["worst_month_score_delta"],
        "projected_public_metrics": report["historical_validation"]["target"][
            "projected_public_metrics_if_local_delta_transfers_one_to_one"
        ],
        "candidates": report["production"]["candidates"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
