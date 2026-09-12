from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent_service.compliance import validate_external_data_manifest
from experiments.cfsv2_meta_risk_gate import (
    FINE_ALPHA,
    FINE_THRESHOLD,
    _apply_keep_policy,
    _crossfit_meta_probabilities,
    _fit_risk_members,
    _risk_features,
    _select_policy,
)
from experiments.exact_oof_meta_gate import (
    META_SEEDS,
    Q2_START,
    _evaluate_period,
    apply_meta_gate,
    fit_probabilities,
    settlement_benefit_labels,
)
from experiments.exact_oof_meta_gate_sweep import _prepare_validation


H2_START = pd.Timestamp("2024-07-01 01:00:00")
KEYS = ["forecast_kst_dtm", "data_available_kst_dtm"]


def _mean_nwp(path: Path, columns: list[str], prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path, encoding="utf-8-sig", usecols=KEYS + ["grid_id"] + columns
    )
    for key in KEYS:
        frame[key] = pd.to_datetime(frame[key])
    if frame.duplicated(KEYS + ["grid_id"]).any():
        raise ValueError(f"{prefix} NWP rows are not unique by forecast/issue/grid")
    return frame.groupby(KEYS)[columns].mean().rename(
        columns=lambda column: f"{prefix}_{column}"
    )


def build_kma_risk_features(
    context_path: Path, gfs_path: Path, ldaps_path: Path
) -> pd.DataFrame:
    context = pd.read_csv(context_path, encoding="utf-8-sig")
    for key in KEYS:
        context[key] = pd.to_datetime(context[key])
    if context.duplicated(KEYS).any():
        raise ValueError("KMA context rows are not unique by forecast and issue")
    context = context.set_index(KEYS)

    gfs_columns = [
        "heightAboveGround_10_10u",
        "heightAboveGround_10_10v",
        "isobaricInhPa_850_u",
        "isobaricInhPa_850_v",
        "isobaricInhPa_700_u",
        "isobaricInhPa_700_v",
    ]
    ldaps_columns = ["heightAboveGround_10_10u", "heightAboveGround_10_10v"]
    merged = context.join(
        _mean_nwp(gfs_path, gfs_columns, "gfs"), how="inner", validate="one_to_one"
    ).join(
        _mean_nwp(ldaps_path, ldaps_columns, "ldaps"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(context):
        raise ValueError("KMA and supplied NWP forecast/issue rows are not aligned")

    retained = [
        column
        for column in merged.columns
        if column.startswith("kma_um_ctx_run_")
        or column.startswith("kma_um_ctx_shear")
        or column.startswith("kma_um_ctx_cos")
    ]
    for source in ("gfs", "ldaps"):
        u = merged[f"{source}_heightAboveGround_10_10u"]
        v = merged[f"{source}_heightAboveGround_10_10v"]
        source_speed = np.hypot(u, v)
        du = merged["kma_um_ctx_u10_r0"] - u
        dv = merged["kma_um_ctx_v10_r0"] - v
        merged[f"kma_um_risk_{source}_du10"] = du
        merged[f"kma_um_risk_{source}_dv10"] = dv
        merged[f"kma_um_risk_{source}_vector10"] = np.hypot(du, dv)
        merged[f"kma_um_risk_{source}_dspeed10"] = (
            merged["kma_um_ctx_speed10_r0"] - source_speed
        )
        retained.extend(
            [
                f"kma_um_risk_{source}_du10",
                f"kma_um_risk_{source}_dv10",
                f"kma_um_risk_{source}_vector10",
                f"kma_um_risk_{source}_dspeed10",
            ]
        )
    for level in (850, 700):
        u = merged[f"gfs_isobaricInhPa_{level}_u"]
        v = merged[f"gfs_isobaricInhPa_{level}_v"]
        du = merged[f"kma_um_ctx_u{level}_r0"] - u
        dv = merged[f"kma_um_ctx_v{level}_r0"] - v
        merged[f"kma_um_risk_gfs_du{level}"] = du
        merged[f"kma_um_risk_gfs_dv{level}"] = dv
        merged[f"kma_um_risk_gfs_vector{level}"] = np.hypot(du, dv)
        merged[f"kma_um_risk_gfs_dspeed{level}"] = (
            merged[f"kma_um_ctx_speed{level}_r0"] - np.hypot(u, v)
        )
        retained.extend(
            [
                f"kma_um_risk_gfs_du{level}",
                f"kma_um_risk_gfs_dv{level}",
                f"kma_um_risk_gfs_vector{level}",
                f"kma_um_risk_gfs_dspeed{level}",
            ]
        )
    result = merged[retained]
    if result.isna().any().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError("KMA risk features contain missing or non-finite values")
    issue = result.reset_index("data_available_kst_dtm").pop(
        "data_available_kst_dtm"
    )
    result = result.reset_index("data_available_kst_dtm", drop=True)
    result.attrs["issue_times"] = issue.to_numpy()
    result.index.name = "forecast_kst_dtm"
    return result.sort_index()


def _rolling_fine_surfaces(
    index: pd.DatetimeIndex,
    truth: np.ndarray,
    current: np.ndarray,
    member: np.ndarray,
    meta_features: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = (index >= pd.Timestamp("2024-01-01")) & (index < Q2_START)
    q2 = (index >= Q2_START) & (index < H2_START)
    h2 = index >= H2_START
    q1_label = settlement_benefit_labels(truth, current, member, q1)
    q1_probability = _crossfit_meta_probabilities(
        meta_features, q1_label, q1 & action
    )
    q1_fine, q1_gate = apply_meta_gate(
        current,
        member,
        action,
        q1_probability,
        threshold=FINE_THRESHOLD,
        extra_alpha=FINE_ALPHA,
    )
    q2_label = settlement_benefit_labels(truth, current, member, index < Q2_START)
    q2_probability, _ = fit_probabilities(
        meta_features, q2_label, (index < Q2_START) & action
    )
    q2_fine, q2_gate = apply_meta_gate(
        current,
        member,
        action,
        q2_probability,
        threshold=FINE_THRESHOLD,
        extra_alpha=FINE_ALPHA,
    )
    h2_label = settlement_benefit_labels(truth, current, member, index < H2_START)
    h2_probability, _ = fit_probabilities(
        meta_features, h2_label, (index < H2_START) & action
    )
    h2_fine, h2_gate = apply_meta_gate(
        current,
        member,
        action,
        h2_probability,
        threshold=FINE_THRESHOLD,
        extra_alpha=FINE_ALPHA,
    )
    fine = current.copy()
    gate = np.zeros(len(index), dtype=bool)
    probability = np.zeros(len(index), dtype=float)
    for mask, candidate, candidate_gate, candidate_probability in (
        (q1, q1_fine, q1_gate, q1_probability),
        (q2, q2_fine, q2_gate, q2_probability),
        (h2, h2_fine, h2_gate, h2_probability),
    ):
        fine[mask] = candidate[mask]
        gate[mask] = candidate_gate[mask]
        probability[mask] = candidate_probability[mask]
    return fine, gate, probability


def _locked_report(
    truth: np.ndarray,
    current: np.ndarray,
    fine: np.ndarray,
    fine_gate: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    index: pd.DatetimeIndex,
) -> dict[str, object]:
    mean_probability = probabilities.mean(axis=0)
    candidate, keep = _apply_keep_policy(
        current, fine, fine_gate, mean_probability, threshold
    )
    seeds = []
    for seed, probability in zip(META_SEEDS, probabilities):
        seed_candidate, seed_keep = _apply_keep_policy(
            current, fine, fine_gate, probability, threshold
        )
        seeds.append(
            {
                "seed": seed,
                "kept_rows": int((seed_keep & mask).sum()),
                "total_vs_current": _evaluate_period(
                    truth, current, seed_candidate, mask
                )["delta"],
                "incremental_vs_fine": _evaluate_period(
                    truth, fine, seed_candidate, mask
                )["delta"],
            }
        )
    monthly = {}
    for month in sorted(set(index[mask].month)):
        month_mask = mask & (index.month == month)
        if month_mask.sum() < 24:
            continue
        monthly[str(month)] = _evaluate_period(
            truth, fine, candidate, month_mask
        )["delta"]
    return {
        "threshold": threshold,
        "kept_rows": int((keep & mask).sum()),
        "dropped_rows": int((fine_gate & mask).sum() - (keep & mask).sum()),
        "total_vs_current": _evaluate_period(truth, current, candidate, mask),
        "incremental_vs_fine": _evaluate_period(truth, fine, candidate, mask),
        "seed_rows": seeds,
        "monthly_incremental_vs_fine": monthly,
        "candidate": candidate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--manifest",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/manifest.json",
    )
    parser.add_argument(
        "--context-features",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/features.csv",
    )
    parser.add_argument("--gfs", default="data/train/gfs_train.csv")
    parser.add_argument("--ldaps", default="data/train/ldaps_train.csv")
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/meta_risk_gate.json",
    )
    parser.add_argument("--minimum-selection-vs-control", type=float, default=0.0001)
    args = parser.parse_args()

    validate_external_data_manifest(Path(args.manifest), Path.cwd().resolve())
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
    fine, fine_gate, meta_probability = _rolling_fine_surfaces(
        index, truth, current, member, meta_feature_array, action
    )
    external = build_kma_risk_features(
        Path(args.context_features), Path(args.gfs), Path(args.ldaps)
    )
    common = index.intersection(external.index)
    positions = index.get_indexer(common)
    q1 = np.asarray((common >= pd.Timestamp("2024-01-01")) & (common < Q2_START))
    q2 = np.asarray((common >= Q2_START) & (common < H2_START))
    h2 = np.asarray(common >= H2_START)
    local_truth = truth[positions]
    local_current = current[positions]
    local_fine = fine[positions]
    local_gate = fine_gate[positions]
    local_meta_probability = meta_probability[positions]
    control = _risk_features(
        common, local_current, member[positions], local_meta_probability
    )
    with_kma = control.join(external.reindex(common))

    q1_label = settlement_benefit_labels(
        local_truth, local_current, local_fine, q1, step=1.0
    )
    q1_train = q1 & local_gate
    control_q1_prob = _fit_risk_members(control, q1_label, q1_train)
    kma_q1_prob = _fit_risk_members(with_kma, q1_label, q1_train)
    control_selection = _select_policy(
        local_truth,
        local_current,
        local_fine,
        local_gate,
        q2,
        control_q1_prob,
    )
    kma_selection = _select_policy(
        local_truth,
        local_current,
        local_fine,
        local_gate,
        q2,
        kma_q1_prob,
    )
    control_gain = (
        control_selection["selected"]["incremental_vs_fine"]["delta"]["score"]
        if control_selection is not None
        else -np.inf
    )
    kma_gain = (
        kma_selection["selected"]["incremental_vs_fine"]["delta"]["score"]
        if kma_selection is not None
        else -np.inf
    )
    raw_selection_incremental = kma_gain - control_gain
    selection_incremental = (
        float(raw_selection_incremental)
        if np.isfinite(raw_selection_incremental)
        else None
    )
    open_locked = bool(
        kma_selection is not None
        and np.isfinite(control_gain)
        and selection_incremental is not None
        and selection_incremental >= args.minimum_selection_vs_control
    )
    locked_control = None
    locked_kma = None
    incremental_vs_control = None
    if open_locked:
        h1_label = settlement_benefit_labels(
            local_truth, local_current, local_fine, q1 | q2, step=1.0
        )
        h1_train = (q1 | q2) & local_gate
        control_h1_prob = _fit_risk_members(control, h1_label, h1_train)
        kma_h1_prob = _fit_risk_members(with_kma, h1_label, h1_train)
        locked_control = _locked_report(
            local_truth,
            local_current,
            local_fine,
            local_gate,
            h2,
            control_h1_prob,
            control_selection["selected"]["threshold"],
            common,
        )
        locked_kma = _locked_report(
            local_truth,
            local_current,
            local_fine,
            local_gate,
            h2,
            kma_h1_prob,
            kma_selection["selected"]["threshold"],
            common,
        )
        incremental_vs_control = _evaluate_period(
            local_truth,
            locked_control.pop("candidate"),
            locked_kma.pop("candidate"),
            h2,
        )
    strict = False
    if locked_kma is not None and incremental_vs_control is not None:
        strict = bool(
            min(locked_kma["total_vs_current"]["delta"].values()) > 0.0
            and min(locked_kma["incremental_vs_fine"]["delta"].values()) > 0.0
            and min(incremental_vs_control["delta"].values()) > 0.0
            and all(
                min(seed["total_vs_current"].values()) > 0.0
                and min(seed["incremental_vs_fine"].values()) > 0.0
                for seed in locked_kma["seed_rows"]
            )
            and all(
                value["ficr"] >= 0.0
                for value in locked_kma["monthly_incremental_vs_fine"].values()
            )
        )
    report = {
        "family": "kma_um_meta_gate_harm_risk_abstention",
        "source_manifest": args.manifest,
        "split": {
            "risk_train": "2024 Q1 with cross-fitted fine meta probabilities",
            "selection": "2024 Q2",
            "locked_h2": "opened once only when KMA beat identical control on Q2",
            "common_rows": int(len(common)),
            "q1_gated_train_rows": int(q1_train.sum()),
        },
        "fine_policy": {"threshold": FINE_THRESHOLD, "alpha": FINE_ALPHA},
        "control_selection": control_selection,
        "kma_selection": kma_selection,
        "selection_incremental_score_vs_control": selection_incremental,
        "locked_control": locked_control,
        "locked_kma": locked_kma,
        "locked_incremental_kma_vs_control": incremental_vs_control,
        "decision": {
            "locked_opened": open_locked,
            "submission_eligible": strict,
            "reason": (
                "KMA risk abstention passed every locked total, fine, control, seed, and month gate"
                if strict
                else "KMA risk abstention failed at least one preregistered promotion gate"
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "family": report["family"],
                "split": report["split"],
                "selection_incremental_score_vs_control": selection_incremental,
                "locked_control": locked_control,
                "locked_kma": locked_kma,
                "locked_incremental_kma_vs_control": incremental_vs_control,
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
