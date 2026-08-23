"""Conditional row-local benefit gate for the v87 R_CORE FM correction.

The cross-season FM direction is useful on average but unstable by month.  This
experiment learns a deliberately shallow model of *where* applying that frozen
direction reduced Brier loss in earlier seasons.  Recipe selection is confined
to early-2022 -> late-2022 and full-2022 -> late-2023 transfers.  The selected
recipe is then refit on full-2022 plus late-2023 before the untouched 2024 audit.

Every inference feature belongs to the current row or is a frozen model output.
There are no test-row aggregates, ordering features, current-pitch TrackMan
measurements, or 2024 labels in fitting/selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.core.diagnostics import diagnostics
from src.v53_factorization_offset import _expit, _logit
from src.v86_reliability_gated_r_fm import ETA, ROUTE, _metric_frame
from src.v87_cross_season_consensus_r_fm import combine_corrections
from src.v92_temporal_player_command_eb import _absolute, _compose_historical_parents


PROTOCOL = "V93_CONDITIONAL_FM_BENEFIT_GATE_ABOVE_EXACT_V84_V1"
TARGET = "control_success"
MODEL_KINDS = ("ridge", "shallow_lgbm")
GATE_STYLES = ("hard_positive", "soft_positive")
FEATURE_VERSION = "row_local_numeric_v1"


@dataclass
class BenefitModel:
    kind: str
    model: Any
    scaler: StandardScaler | None
    soft_scale: float


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), float(default), dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    return np.nan_to_num(values, nan=float(default), posinf=float(default), neginf=float(default))


def build_gate_features(
    frame: pd.DataFrame,
    parent: np.ndarray,
    older: np.ndarray,
    recent: np.ndarray,
) -> pd.DataFrame:
    """Build fixed row-local numeric features without frame-level statistics."""

    parent = np.asarray(parent, dtype=np.float64)
    older = np.asarray(older, dtype=np.float64)
    recent = np.asarray(recent, dtype=np.float64)
    if not (parent.shape == older.shape == recent.shape == (len(frame),)):
        raise ValueError("gate feature arrays are not aligned")
    if not (np.isfinite(parent).all() and np.isfinite(older).all() and np.isfinite(recent).all()):
        raise ValueError("gate feature inputs contain non-finite values")

    correction, agreement = combine_corrections(older, recent, "source_mean")
    abs_old = np.abs(older)
    abs_recent = np.abs(recent)
    larger = np.maximum(abs_old, abs_recent)
    month = _numeric(frame, "game_month", 6.0)
    pitcher_n = np.maximum(_numeric(frame, "asof_pitcher_n"), 0.0)
    batter_n = np.maximum(_numeric(frame, "asof_batter_n"), 0.0)
    pitchmix_n = np.maximum(_numeric(frame, "asof_pitcher_pitchmix_n"), 0.0)

    pitcher_success = _numeric(frame, "asof_pitcher_success_rate", 0.5)
    pitcher_middle = _numeric(frame, "asof_pitcher_middle_rate", 0.5)
    batter_success = _numeric(frame, "asof_batter_success_rate", 0.5)
    prev1 = _numeric(frame, "asof_pitcher_prev1_game_success_rate", 0.5)
    prev3 = _numeric(frame, "asof_pitcher_prev3_game_success_rate", 0.5)
    prev5 = _numeric(frame, "asof_pitcher_prev5_game_success_rate", 0.5)
    balls = _numeric(frame, "balls_before", -1.0)
    strikes = _numeric(frame, "strikes_before", -1.0)
    li = np.maximum(_numeric(frame, "li", 1.0), 0.0)

    features = pd.DataFrame(
        {
            "parent": parent,
            "parent_margin": np.abs(parent - 0.5),
            "correction_old": older,
            "correction_recent": recent,
            "correction_mean": correction,
            "correction_abs": np.abs(correction),
            "correction_difference": np.abs(older - recent),
            "correction_product": older * recent,
            "correction_agreement": agreement.astype(np.float64),
            "correction_min_abs": np.minimum(abs_old, abs_recent),
            "correction_max_abs": larger,
            "correction_coherence": np.divide(
                np.minimum(abs_old, abs_recent),
                larger,
                out=np.zeros_like(larger),
                where=larger > 0.0,
            ),
            "game_month": month,
            "game_month_sin": np.sin(2.0 * np.pi * month / 12.0),
            "game_month_cos": np.cos(2.0 * np.pi * month / 12.0),
            "inning": _numeric(frame, "inning", 1.0),
            "balls": balls,
            "strikes": strikes,
            "two_strikes": (strikes == 2.0).astype(np.float64),
            "three_balls": (balls == 3.0).astype(np.float64),
            "outs": _numeric(frame, "outs_before", 0.0),
            "num_runners": _numeric(frame, "num_runners_on", 0.0),
            "score_diff": _numeric(frame, "score_diff_pitcher_team", 0.0),
            "abs_score_diff": np.abs(_numeric(frame, "score_diff_pitcher_team", 0.0)),
            "log_li": np.log1p(li),
            "run_total": _numeric(frame, "run_total_before", 0.0),
            "pitcher_n_log": np.log1p(pitcher_n),
            "batter_n_log": np.log1p(batter_n),
            "pitchmix_n_log": np.log1p(pitchmix_n),
            "pitcher_reliability_200": pitcher_n / (pitcher_n + 200.0),
            "batter_reliability_200": batter_n / (batter_n + 200.0),
            "pitcher_success": pitcher_success,
            "pitcher_middle": pitcher_middle,
            "pitcher_reverse": _numeric(frame, "asof_pitcher_reverse_rate", 0.5),
            "pitcher_ball": _numeric(frame, "asof_pitcher_ball_rate", 0.5),
            "pitcher_strike": _numeric(frame, "asof_pitcher_strike_rate", 0.5),
            "batter_success": batter_success,
            "batter_middle": _numeric(frame, "asof_batter_middle_rate", 0.5),
            "pitcher_minus_batter": pitcher_success - batter_success,
            "recent1_delta": prev1 - pitcher_success,
            "recent3_delta": prev3 - pitcher_success,
            "recent5_delta": prev5 - pitcher_success,
            "same_hand": (
                frame["pitcher_hand"].astype(str).to_numpy()
                == frame["batter_hand"].astype(str).to_numpy()
            ).astype(np.float64),
            "top_half": frame["top_bottom"].astype(str).str.lower().isin(
                ["top", "초", "t", "1"]
            ).to_numpy(np.float64),
        }
    )
    values = features.to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("constructed gate features contain non-finite values")
    return features


def apply_fm_gate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    correction: np.ndarray,
    gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    if not (parent.shape == correction.shape == gate.shape == (len(frame),)):
        raise ValueError("FM gate arrays are not aligned")
    if not np.isfinite(gate).all() or np.any((gate < 0.0) | (gate > 1.0)):
        raise ValueError("gate must be finite and bounded")
    active = frame["domain3"].astype(str).eq(ROUTE).to_numpy()
    output = parent.copy()
    output[active] = _expit(
        _logit(parent[active]) + ETA * correction[active] * gate[active]
    )
    return np.clip(output, 0.001, 0.999), active


def row_brier_benefit(
    frame: pd.DataFrame,
    parent: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    raw, active = apply_fm_gate(frame, parent, correction, np.ones(len(frame)))
    target = frame[TARGET].to_numpy(np.float64)
    benefit = (target - parent) ** 2 - (target - raw) ** 2
    return np.where(active, benefit * 1_000_000.0, 0.0)


def _balanced_month_weight(frame: pd.DataFrame, active: np.ndarray) -> np.ndarray:
    month = _numeric(frame, "game_month", 0.0)
    weight = np.zeros(len(frame), dtype=np.float64)
    active_months, counts = np.unique(month[active], return_counts=True)
    for value, count in zip(active_months, counts, strict=True):
        weight[active & (month == value)] = 1.0 / float(count)
    total = weight.sum()
    if total <= 0.0:
        raise ValueError("no active rows for month balancing")
    weight *= float(active.sum()) / total
    return weight


def fit_benefit_model(
    features: pd.DataFrame,
    benefit: np.ndarray,
    sample_weight: np.ndarray,
    kind: str,
) -> BenefitModel:
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown benefit model: {kind}")
    x = features.to_numpy(np.float64)
    y = np.asarray(benefit, dtype=np.float64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    keep = weight > 0.0
    if not keep.any():
        raise ValueError("benefit model received no weighted rows")
    if kind == "ridge":
        scaler = StandardScaler().fit(x[keep], sample_weight=weight[keep])
        model = Ridge(alpha=200.0)
        model.fit(scaler.transform(x[keep]), y[keep], sample_weight=weight[keep])
        prediction = model.predict(scaler.transform(x[keep]))
    else:
        scaler = None
        model = lgb.LGBMRegressor(
            objective="regression_l2",
            n_estimators=80,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=5_000,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=10.0,
            reg_lambda=50.0,
            random_state=20260822,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(x[keep], y[keep], sample_weight=weight[keep])
        prediction = model.predict(x[keep])
    positive = prediction[prediction > 0.0]
    scale = float(np.quantile(positive, 0.90)) if len(positive) else 1.0
    return BenefitModel(kind, model, scaler, max(scale, 1e-9))


def predict_gate(model: BenefitModel, features: pd.DataFrame, style: str) -> tuple[np.ndarray, np.ndarray]:
    if style not in GATE_STYLES:
        raise ValueError(f"unknown gate style: {style}")
    x = features.to_numpy(np.float64)
    transformed = model.scaler.transform(x) if model.scaler is not None else x
    expected = np.asarray(model.model.predict(transformed), dtype=np.float64)
    if style == "hard_positive":
        gate = (expected > 0.0).astype(np.float64)
    else:
        gate = np.clip(expected / model.soft_scale, 0.0, 1.0)
    return gate, expected


def _compact(result: dict[str, Any]) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "applied_domain_gain": float(result["domain_gains"][ROUTE]),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def _train_and_audit(
    train_frame: pd.DataFrame,
    train_parent: np.ndarray,
    train_pair: tuple[np.ndarray, np.ndarray],
    audit_frame: pd.DataFrame,
    audit_parent: np.ndarray,
    audit_pair: tuple[np.ndarray, np.ndarray],
    kind: str,
    style: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_correction, _ = combine_corrections(*train_pair, "source_mean")
    audit_correction, _ = combine_corrections(*audit_pair, "source_mean")
    train_features = build_gate_features(train_frame, train_parent, *train_pair)
    audit_features = build_gate_features(audit_frame, audit_parent, *audit_pair)
    train_active = train_frame["domain3"].astype(str).eq(ROUTE).to_numpy()
    benefit = row_brier_benefit(train_frame, train_parent, train_correction)
    weight = _balanced_month_weight(train_frame, train_active)
    model = fit_benefit_model(train_features, benefit, weight, kind)
    gate, expected = predict_gate(model, audit_features, style)
    candidate, active = apply_fm_gate(audit_frame, audit_parent, audit_correction, gate)
    result = diagnostics(_metric_frame(audit_frame), audit_parent, candidate, active)
    result.update(
        {
            "gate_mean_r_core": float(gate[active].mean()),
            "gate_nonzero_fraction_r_core": float(np.mean(gate[active] > 0.0)),
            "expected_benefit_mean_r_core": float(expected[active].mean()),
            "train_raw_benefit_mean_r_core": float(benefit[train_active].mean()),
            "soft_scale": float(model.soft_scale),
        }
    )
    return result, gate, expected


def _select_recipe(table: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for kind in MODEL_KINDS:
        for style in GATE_STYLES:
            local = table.loc[
                table["model_kind"].eq(kind) & table["gate_style"].eq(style)
            ].set_index("axis")
            passed = bool(
                set(local.index) == {"late_2022", "late_2023"}
                and (local["gain"] > 0.0).all()
                and (local["positive_month_fraction"] >= (2.0 / 3.0)).all()
                and (local["worst_month_gain"] > -5.0).all()
                and (local["applied_domain_gain"] > 0.0).all()
            )
            robust = float(min(local["gain"].min(), local["worst_month_gain"].min()))
            records.append(
                {
                    "model_kind": kind,
                    "gate_style": style,
                    "source_gate_passed": passed,
                    "robust_score": robust,
                    "minimum_gain": float(local["gain"].min()),
                    "minimum_month_fraction": float(local["positive_month_fraction"].min()),
                    "minimum_worst_month_gain": float(local["worst_month_gain"].min()),
                }
            )
    ranking = pd.DataFrame(records).sort_values(
        ["source_gate_passed", "robust_score", "minimum_gain"], ascending=False
    ).reset_index(drop=True)
    chosen = ranking.loc[ranking["source_gate_passed"]]
    row = (chosen if len(chosen) else ranking).iloc[0]
    return {"model_kind": str(row["model_kind"]), "gate_style": str(row["gate_style"])}, ranking


def run(project: Path, support_dir: Path, external_root: Path, output_dir: Path) -> dict[str, Any]:
    project = project.resolve()
    support_dir = support_dir.resolve()
    external_root = external_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    frames, parents, provenance = _compose_historical_parents(
        project, support_dir, external_root, raw
    )
    frame22, frame23, frame24 = (frames["full_2022"], frames["late_2023"], frames["full_2024"])
    parent22, parent23, parent24 = (parents["full_2022"], parents["late_2023"], parents["full_2024"])

    v87_dir = support_dir / "v87_cross_season_consensus_r_fm_20260822_01"
    with np.load(v87_dir / "source_corrections.npz", allow_pickle=False) as saved:
        pair22 = (saved["correction22_old"].astype(np.float64), saved["correction22_recent"].astype(np.float64))
        pair23 = (saved["correction23_old"].astype(np.float64), saved["correction23_recent"].astype(np.float64))
    with np.load(v87_dir / "outer_full_2024.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["target"].astype(np.float64), frame24[TARGET].to_numpy(np.float64)):
            raise ValueError("v87/full-2024 target mismatch")
        pair24 = (saved["correction_old"].astype(np.float64), saved["correction_recent"].astype(np.float64))

    early22 = frame22["game_month"].le(7).to_numpy()
    late22 = frame22["game_month"].ge(8).to_numpy()
    trial_rows: list[dict[str, Any]] = []
    source_details: dict[str, Any] = {}
    for kind in MODEL_KINDS:
        for style in GATE_STYLES:
            recipe = f"{kind}|{style}"
            result22, _, _ = _train_and_audit(
                frame22.loc[early22].reset_index(drop=True), parent22[early22],
                (pair22[0][early22], pair22[1][early22]),
                frame22.loc[late22].reset_index(drop=True), parent22[late22],
                (pair22[0][late22], pair22[1][late22]), kind, style,
            )
            result23, _, _ = _train_and_audit(
                frame22, parent22, pair22, frame23, parent23, pair23, kind, style
            )
            source_details[recipe] = {"late_2022": result22, "late_2023": result23}
            for axis, result in (("late_2022", result22), ("late_2023", result23)):
                trial_rows.append({"model_kind": kind, "gate_style": style, "axis": axis, **_compact(result),
                                   "gate_mean_r_core": result["gate_mean_r_core"],
                                   "gate_nonzero_fraction_r_core": result["gate_nonzero_fraction_r_core"]})

    source_table = pd.DataFrame(trial_rows)
    selected, ranking = _select_recipe(source_table)
    source_table.to_csv(output_dir / "source_gate_metrics.csv", index=False)
    ranking.to_csv(output_dir / "source_gate_ranking.csv", index=False)
    source_passed = bool(ranking.iloc[0]["source_gate_passed"])

    # Refit the frozen recipe on both historical audit periods, weighting each
    # source period equally after within-period month balancing.
    correction22, _ = combine_corrections(*pair22, "source_mean")
    correction23, _ = combine_corrections(*pair23, "source_mean")
    feature22 = build_gate_features(frame22, parent22, *pair22)
    feature23 = build_gate_features(frame23, parent23, *pair23)
    active22 = frame22["domain3"].astype(str).eq(ROUTE).to_numpy()
    active23 = frame23["domain3"].astype(str).eq(ROUTE).to_numpy()
    weight22 = _balanced_month_weight(frame22, active22)
    weight23 = _balanced_month_weight(frame23, active23)
    weight22 *= 0.5 / weight22.sum()
    weight23 *= 0.5 / weight23.sum()
    final_model = fit_benefit_model(
        pd.concat([feature22, feature23], ignore_index=True),
        np.concatenate([
            row_brier_benefit(frame22, parent22, correction22),
            row_brier_benefit(frame23, parent23, correction23),
        ]),
        np.concatenate([weight22, weight23]),
        selected["model_kind"],
    )
    feature24 = build_gate_features(frame24, parent24, *pair24)
    gate24, expected24 = predict_gate(final_model, feature24, selected["gate_style"])
    correction24, _ = combine_corrections(*pair24, "source_mean")
    candidate24, active24 = apply_fm_gate(frame24, parent24, correction24, gate24)
    full_audit = diagnostics(_metric_frame(frame24), parent24, candidate24, active24)
    late = frame24["game_month"].ge(8).to_numpy()
    late_audit = diagnostics(
        _metric_frame(frame24.loc[late].reset_index(drop=True)),
        parent24[late], candidate24[late], active24[late],
    )
    raw24, _ = apply_fm_gate(frame24, parent24, correction24, np.ones(len(frame24)))
    raw_full = diagnostics(_metric_frame(frame24), parent24, raw24, active24)
    raw_late = diagnostics(
        _metric_frame(frame24.loc[late].reset_index(drop=True)),
        parent24[late], raw24[late], active24[late],
    )
    point_gates = {
        "pre2024_source_recipe_passed": source_passed,
        "full_gain_positive": float(full_audit["gain"]) > 0.0,
        "late_gain_positive": float(late_audit["gain"]) > 0.0,
        "full_month_fraction_at_least_075": float(full_audit["positive_month_fraction"]) >= 0.75,
        "late_month_fraction_at_least_075": float(late_audit["positive_month_fraction"]) >= 0.75,
        "both_worst_months_above_minus_5": min(float(full_audit["worst_month_gain"]), float(late_audit["worst_month_gain"])) > -5.0,
        "applied_domain_positive_both": min(float(full_audit["domain_gains"][ROUTE]), float(late_audit["domain_gains"][ROUTE])) > 0.0,
    }
    np.savez_compressed(
        output_dir / "outer_full_2024.npz",
        target=frame24[TARGET].to_numpy(np.float64), v84=parent24,
        correction=correction24, gate=gate24, expected_benefit=expected24,
        candidate=candidate24, active=active24,
        domain3=frame24["domain3"].astype(str).to_numpy(),
        game_month=frame24["game_month"].to_numpy(np.int16),
    )
    result = {
        "protocol": PROTOCOL,
        "parent": "exact v84 OOF analogue / Public 1161.2020600422",
        "configuration": {
            "feature_version": FEATURE_VERSION,
            "model_kinds": list(MODEL_KINDS), "gate_styles": list(GATE_STYLES),
            "selected": selected, "fm_policy": "source_mean", "fm_eta": ETA,
            "ridge_alpha": 200.0,
            "shallow_lgbm": {"trees": 80, "leaves": 7, "max_depth": 3, "min_child_samples": 5000},
        },
        "selection": "recipe on early-2022->late-2022 and full-2022->late-2023 only",
        "source_gate_passed": source_passed,
        "source_gate_ranking": ranking.to_dict(orient="records"),
        "source_details": source_details,
        "parent_provenance": provenance,
        "parent_audit": {name: _absolute(frames[name], parents[name]) for name in frames},
        "raw_v87_reference": {"outer_full_2024": raw_full, "replication_late_2024": raw_late},
        "audits": {"outer_full_2024": full_audit, "replication_late_2024": late_audit},
        "gate_distribution_2024_r_core": {
            "mean": float(gate24[active24].mean()),
            "nonzero_fraction": float(np.mean(gate24[active24] > 0.0)),
            "expected_benefit_mean": float(expected24[active24].mean()),
            "expected_benefit_p10": float(np.quantile(expected24[active24], 0.10)),
            "expected_benefit_p90": float(np.quantile(expected24[active24], 0.90)),
        },
        "point_gates": {key: bool(value) for key, value in point_gates.items()},
        "point_gates_passed": bool(all(point_gates.values())),
        "eligible_for_packaging": False,
        "promotion_note": "point gates authorize dependence-aware bootstrap and Reality Check, not immediate packaging",
        "same_family_2024_labels_used_for_recipe_selection": False,
        "public_score_used_for_recipe_or_weight": False,
        "test_csv_read": False, "test_aggregate_used": False,
        "row_local_inference": True, "current_pitch_trackman_or_location_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "source_passed": source_passed,
                      "full_2024": _compact(full_audit), "late_2024": _compact(late_audit),
                      "point_gates": point_gates}, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.support_dir, args.external_root, args.output_dir)


if __name__ == "__main__":
    main()
