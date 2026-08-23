"""Temporal-consensus player command EB above the exact v84 OOF parent.

The Public-1161 model already contains flexible trees and a low-rank FM.  This
experiment asks a narrower baseball question: does a pitcher or batter retain
the *same residual command tendency* across temporal origins after the current
champion has made its prediction?

Only official row-local fields and frozen historical OOF residuals are used.
The recipe is selected on two pre-2024 transfers (early-to-late 2022 and
full-2022-to-late-2023).  Effects from full 2022 and late 2023 are then carried
to 2024 only when their signs agree.  Full/late 2024 are audits, never fitting
or recipe-selection data.

The 2022 parent is exact on R_CORE and F but lacks the final TrackMan
R_ANCHOR-only gate.  This experiment is therefore deliberately routed only to
R_CORE, where all source and audit parents have exact v84 component parity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import brier_score, brier_skill_score_unclipped
from src.v30_diverse_covariance_screen import diagnostics
from src.v35_three_stage_multibank import _metadata
from src.v53_factorization_offset import apply_offset, prepare_fields
from src.v57_public_strict_blend import _load_strict, blend_candidate


PROTOCOL = "V92_TEMPORAL_PLAYER_COMMAND_EB_ABOVE_EXACT_V84_V1"
TARGET = "control_success"
ROUTE = "R_CORE"
STRICT_WEIGHT = 0.10
F_ETA = 0.10
CORRECTION_CAP = 0.02
ALPHAS = (500.0, 2_000.0, 10_000.0)
ETAS = (0.25, 0.50, 1.00)


@dataclass(frozen=True)
class GroupSpec:
    name: str
    columns: tuple[str, ...]


GROUP_SPECS = (
    GroupSpec("pitcher", ("pitcher_id",)),
    GroupSpec("pitcher_pressure", ("pitcher_id", "pressure")),
    GroupSpec("pitcher_count", ("pitcher_id", "count_state")),
    GroupSpec("pitcher_batter_hand", ("pitcher_id", "batter_hand")),
    GroupSpec("pitcher_count_band", ("pitcher_id", "count_band")),
    GroupSpec("batter", ("batter_id",)),
    GroupSpec("batter_count", ("batter_id", "count_state")),
    GroupSpec("batter_pitcher_hand", ("batter_id", "pitcher_hand")),
    GroupSpec("pitcher_team_count", ("pitcher_team_id", "count_state")),
    GroupSpec(
        "team_matchup_count",
        ("pitcher_team_id", "batter_team_id", "count_state"),
    ),
    GroupSpec("count_hands", ("count_state", "hand_matchup")),
    GroupSpec("base_out_count", ("base_state", "outs_before", "count_state")),
    GroupSpec(
        "profile_pressure",
        ("success_bin", "middle_bin", "reverse_bin", "pressure"),
    ),
    GroupSpec(
        "recent_pressure",
        ("recent_success_bin", "recent_middle_bin", "pressure"),
    ),
)


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    return np.where(np.isfinite(values), values, float(default))


def add_command_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create fixed row-local categories; no frame-level statistics are used."""

    output = prepare_fields(frame).reset_index(drop=True)
    balls = _safe_numeric(output, "balls_before", -1.0)
    strikes = _safe_numeric(output, "strikes_before", -1.0)
    output["count_band"] = np.select(
        [
            (balls >= 2.0) & (strikes <= 1.0),
            (strikes == 2.0) & (balls <= 1.0),
            (balls == 3.0) | (strikes == 2.0),
        ],
        ["hitter", "pitcher", "terminal"],
        default="neutral",
    )

    rate_bins = [-np.inf, 0.35, 0.45, 0.50, 0.55, 0.65, np.inf]
    for source, destination in (
        ("asof_pitcher_success_rate", "success_bin"),
        ("asof_pitcher_middle_rate", "middle_bin"),
        ("asof_pitcher_reverse_rate", "reverse_bin"),
    ):
        values = _safe_numeric(output, source, 0.5)
        output[destination] = pd.cut(
            values,
            bins=rate_bins,
            labels=False,
            include_lowest=True,
        ).astype("int8")

    recent_success = (
        0.50 * _safe_numeric(output, "asof_pitcher_prev1_game_success_rate", 0.5)
        + 0.30 * _safe_numeric(output, "asof_pitcher_prev3_game_success_rate", 0.5)
        + 0.20 * _safe_numeric(output, "asof_pitcher_prev5_game_success_rate", 0.5)
        - _safe_numeric(output, "asof_pitcher_success_rate", 0.5)
    )
    recent_middle = (
        0.50 * _safe_numeric(output, "asof_pitcher_prev1_game_middle_rate", 0.0)
        + 0.30 * _safe_numeric(output, "asof_pitcher_prev3_game_middle_rate", 0.0)
        + 0.20 * _safe_numeric(output, "asof_pitcher_prev5_game_middle_rate", 0.0)
        - _safe_numeric(output, "asof_pitcher_middle_rate", 0.0)
    )
    delta_bins = [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf]
    output["recent_success_bin"] = pd.cut(
        recent_success, bins=delta_bins, labels=False, include_lowest=True
    ).astype("int8")
    output["recent_middle_bin"] = pd.cut(
        recent_middle, bins=delta_bins, labels=False, include_lowest=True
    ).astype("int8")
    return output


def group_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Return deterministic row-local compound keys."""

    if not columns:
        raise ValueError("group columns cannot be empty")
    values = frame[columns[0]].astype("string").fillna("__MISSING__").to_numpy(str)
    output = values.astype(str)
    for column in columns[1:]:
        right = frame[column].astype("string").fillna("__MISSING__").to_numpy(str)
        output = np.char.add(np.char.add(output, "\x1f"), right)
    return output


def fit_eb_effect(
    keys: np.ndarray,
    target: np.ndarray,
    parent: np.ndarray,
    active: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Fit source-only centered empirical-Bayes residual effects."""

    key = np.asarray(keys).astype(str)
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(parent, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    if not (key.shape == y.shape == p.shape == mask.shape and key.ndim == 1):
        raise ValueError("EB inputs must be aligned one-dimensional arrays")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    local = pd.DataFrame({"key": key[mask], "residual": y[mask] - p[mask]})
    grouped = local.groupby("key", observed=True)["residual"].agg(["sum", "size"])
    raw = grouped["sum"] / (grouped["size"] + float(alpha))
    centre = float(np.average(raw.to_numpy(), weights=grouped["size"].to_numpy()))
    centered = raw - centre
    return {str(name): float(value) for name, value in centered.items()}


def map_effect(keys: np.ndarray, effect: dict[str, float]) -> np.ndarray:
    """Map a frozen source table; unknown rows receive exactly zero."""

    mapped = pd.Series(np.asarray(keys).astype(str), copy=False).map(effect)
    return mapped.fillna(0.0).to_numpy(np.float64)


def sign_consensus_effect(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    """Keep shared group effects only when both temporal sources agree in sign."""

    output: dict[str, float] = {}
    for key in sorted(set(left).intersection(right)):
        first = float(left[key])
        second = float(right[key])
        if first * second > 0.0:
            output[key] = float(
                np.sign(first) * np.sqrt(abs(first) * abs(second))
            )
    return output


def apply_probability_correction(
    frame: pd.DataFrame,
    parent: np.ndarray,
    correction: np.ndarray,
    eta: float,
    *,
    route: str = ROUTE,
    cap: float = CORRECTION_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a bounded frozen row-local Brier residual correction."""

    base = np.asarray(parent, dtype=np.float64)
    shift = np.asarray(correction, dtype=np.float64)
    if not (base.shape == shift.shape == (len(frame),)):
        raise ValueError("correction arrays are not aligned")
    active = frame["domain3"].astype(str).eq(route).to_numpy()
    output = base.copy()
    bounded = np.clip(shift, -float(cap), float(cap))
    output[active] = np.clip(
        base[active] + float(eta) * bounded[active], 0.001, 0.999
    )
    return output, active


def _metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": frame[TARGET].to_numpy(np.float64),
            "game_month": frame["game_month"].to_numpy(np.int16),
            "domain3": frame["domain3"].astype(str).to_numpy(),
        }
    )


def _compose_historical_parents(
    project: Path,
    support_dir: Path,
    external_root: Path,
    raw: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    strict, strict_provenance = _load_strict(external_root, raw)
    rows22 = add_command_features(
        raw.loc[raw["season"].eq(2022)].reset_index(drop=True)
    )
    rows23_full = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    late23_mask = rows23_full["game_month"].ge(8).to_numpy()
    rows23 = add_command_features(rows23_full.loc[late23_mask].reset_index(drop=True))
    rows24 = add_command_features(
        raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    )

    meta22 = _metadata(project, 2022)
    if not np.array_equal(meta22["target"], rows22[TARGET].to_numpy(np.float64)):
        raise ValueError("2022 target/order mismatch")
    parent22, _ = blend_candidate(
        rows22,
        meta22["parent"],
        strict[2022],
        mode="probability",
        route=ROUTE,
        weight=STRICT_WEIGHT,
    )

    with np.load(
        support_dir / "v89_reliability_gated_f_extra_dose_20260822_01"
        / "source_corrections.npz",
        allow_pickle=False,
    ) as saved:
        correction22 = saved["correction22"].astype(np.float64)
        correction23 = saved["correction23"].astype(np.float64)
    parent22, _ = apply_offset(rows22, parent22, correction22, "F", F_ETA)

    with np.load(
        support_dir / "v61_final_gate_oof_20260822_01"
        / "selection_late_2023.npz",
        allow_pickle=True,
    ) as saved:
        if not np.array_equal(
            saved["target"].astype(np.float64), rows23[TARGET].to_numpy(np.float64)
        ):
            raise ValueError("late-2023 TrackMan parent target/order mismatch")
        parent23 = saved["final_gate_parent"].astype(np.float64)
        domain23 = saved["domain3"].astype(str)
    if not np.array_equal(domain23, rows23["domain3"].astype(str).to_numpy()):
        raise ValueError("late-2023 TrackMan parent domain/order mismatch")
    parent23, _ = blend_candidate(
        rows23,
        parent23,
        strict[2023][late23_mask],
        mode="probability",
        route=ROUTE,
        weight=STRICT_WEIGHT,
    )
    parent23, _ = apply_offset(rows23, parent23, correction23, "F", F_ETA)

    with np.load(
        support_dir / "v89_reliability_gated_f_extra_dose_20260822_01"
        / "outer_full_2024.npz",
        allow_pickle=False,
    ) as saved:
        if not np.array_equal(
            saved["target"].astype(np.float64), rows24[TARGET].to_numpy(np.float64)
        ):
            raise ValueError("full-2024 v84 target/order mismatch")
        parent24 = saved["v84"].astype(np.float64)
        if not np.array_equal(
            saved["domain3"].astype(str), rows24["domain3"].astype(str).to_numpy()
        ):
            raise ValueError("full-2024 v84 domain/order mismatch")

    frames = {"full_2022": rows22, "late_2023": rows23, "full_2024": rows24}
    parents = {"full_2022": parent22, "late_2023": parent23, "full_2024": parent24}
    provenance = {
        "strict": strict_provenance,
        "full_2022_parent": (
            "exact v84 component parity on R_CORE/F; final TrackMan gate absent "
            "only on protected R_ANCHOR"
        ),
        "late_2023_parent": "exact v84 analogue on all three domains",
        "full_2024_parent": "exact v84 analogue archived by v89",
    }
    return frames, parents, provenance


def _absolute(frame: pd.DataFrame, parent: np.ndarray) -> dict[str, float]:
    target = frame[TARGET].to_numpy(np.float64)
    return {
        "rows": int(len(frame)),
        "target_rate": float(target.mean()),
        "prediction_mean": float(np.mean(parent)),
        "mean_residual": float(np.mean(target - parent)),
        "brier": brier_score(target, parent),
        "unclipped_bss_equivalent": brier_skill_score_unclipped(target, parent),
    }


def _compact(result: dict[str, Any]) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "applied_domain_gain": float(result["domain_gains"][ROUTE]),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def run(
    project: Path,
    support_dir: Path,
    external_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    support_dir = support_dir.resolve()
    external_root = external_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    frames, parents, provenance = _compose_historical_parents(
        project, support_dir, external_root, raw
    )

    frame22 = frames["full_2022"]
    parent22 = parents["full_2022"]
    frame23 = frames["late_2023"]
    parent23 = parents["late_2023"]
    frame24 = frames["full_2024"]
    parent24 = parents["full_2024"]
    early22 = frame22["game_month"].le(7).to_numpy()
    late22 = frame22["game_month"].ge(8).to_numpy()

    key_cache = {
        (axis, spec.name): group_keys(frame, spec.columns)
        for axis, frame in frames.items()
        for spec in GROUP_SPECS
    }
    trial_rows: list[dict[str, Any]] = []
    trial_details: dict[str, Any] = {}
    active22 = frame22["domain3"].astype(str).eq(ROUTE).to_numpy()
    active23 = frame23["domain3"].astype(str).eq(ROUTE).to_numpy()
    for spec in GROUP_SPECS:
        for alpha in ALPHAS:
            effect_early22 = fit_eb_effect(
                key_cache[("full_2022", spec.name)][early22],
                frame22.loc[early22, TARGET].to_numpy(np.float64),
                parent22[early22],
                active22[early22],
                alpha,
            )
            correction_late22 = map_effect(
                key_cache[("full_2022", spec.name)][late22], effect_early22
            )
            effect_full22 = fit_eb_effect(
                key_cache[("full_2022", spec.name)],
                frame22[TARGET].to_numpy(np.float64),
                parent22,
                active22,
                alpha,
            )
            correction_late23 = map_effect(
                key_cache[("late_2023", spec.name)], effect_full22
            )
            for eta in ETAS:
                name = f"{spec.name}|a{alpha:.0f}|e{eta:.2f}"
                candidate22, applied22 = apply_probability_correction(
                    frame22.loc[late22].reset_index(drop=True),
                    parent22[late22],
                    correction_late22,
                    eta,
                )
                result22 = diagnostics(
                    _metric_frame(frame22.loc[late22].reset_index(drop=True)),
                    parent22[late22],
                    candidate22,
                    applied22,
                )
                candidate23, applied23 = apply_probability_correction(
                    frame23, parent23, correction_late23, eta
                )
                result23 = diagnostics(
                    _metric_frame(frame23), parent23, candidate23, applied23
                )
                compact22 = _compact(result22)
                compact23 = _compact(result23)
                passed = bool(
                    compact22["gain"] > 0.0
                    and compact23["gain"] > 0.0
                    and compact22["positive_month_fraction"] >= 2.0 / 3.0
                    and compact23["positive_month_fraction"] >= 2.0 / 3.0
                    and compact22["worst_month_gain"] > -5.0
                    and compact23["worst_month_gain"] > -5.0
                    and compact22["applied_domain_gain"] > 0.0
                    and compact23["applied_domain_gain"] > 0.0
                )
                robust = float(
                    min(
                        compact22["gain"],
                        compact23["gain"],
                        compact22["worst_month_gain"],
                        compact23["worst_month_gain"],
                        compact22["applied_domain_gain"],
                        compact23["applied_domain_gain"],
                    )
                )
                trial_rows.append(
                    {
                        "recipe": name,
                        "group": spec.name,
                        "alpha": alpha,
                        "eta": eta,
                        "source_gate_passed": passed,
                        "robust_score": robust,
                        **{f"late22_{key}": value for key, value in compact22.items()},
                        **{f"late23_{key}": value for key, value in compact23.items()},
                    }
                )
                trial_details[name] = {
                    "late_2022": result22,
                    "late_2023": result23,
                }

    ranking = pd.DataFrame(trial_rows).sort_values(
        ["source_gate_passed", "robust_score", "late23_gain", "late22_gain"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_recipe_ranking.csv", index=False)
    passing = ranking.loc[ranking["source_gate_passed"]]
    selected_row = (passing if len(passing) else ranking).iloc[0]
    selected = {
        "recipe": str(selected_row["recipe"]),
        "group": str(selected_row["group"]),
        "alpha": float(selected_row["alpha"]),
        "eta": float(selected_row["eta"]),
        "source_gate_passed": bool(selected_row["source_gate_passed"]),
    }
    spec = next(item for item in GROUP_SPECS if item.name == selected["group"])
    effect22 = fit_eb_effect(
        key_cache[("full_2022", spec.name)],
        frame22[TARGET].to_numpy(np.float64),
        parent22,
        active22,
        selected["alpha"],
    )
    effect23 = fit_eb_effect(
        key_cache[("late_2023", spec.name)],
        frame23[TARGET].to_numpy(np.float64),
        parent23,
        active23,
        selected["alpha"],
    )
    consensus = sign_consensus_effect(effect22, effect23)
    correction24 = map_effect(key_cache[("full_2024", spec.name)], consensus)
    candidate24, active24 = apply_probability_correction(
        frame24, parent24, correction24, selected["eta"]
    )
    full_audit = diagnostics(_metric_frame(frame24), parent24, candidate24, active24)
    late24 = frame24["game_month"].ge(8).to_numpy()
    late_audit = diagnostics(
        _metric_frame(frame24.loc[late24].reset_index(drop=True)),
        parent24[late24],
        candidate24[late24],
        active24[late24],
    )
    full_compact = _compact(full_audit)
    late_compact = _compact(late_audit)
    point_gates = {
        "pre2024_source_recipe_passed": selected["source_gate_passed"],
        "full_gain_positive": full_compact["gain"] > 0.0,
        "late_gain_positive": late_compact["gain"] > 0.0,
        "full_month_fraction_at_least_075": (
            full_compact["positive_month_fraction"] >= 0.75
        ),
        "late_month_fraction_at_least_075": (
            late_compact["positive_month_fraction"] >= 0.75
        ),
        "both_worst_months_above_minus_5": min(
            full_compact["worst_month_gain"], late_compact["worst_month_gain"]
        )
        > -5.0,
        "both_applied_domain_gains_positive": min(
            full_compact["applied_domain_gain"],
            late_compact["applied_domain_gain"],
        )
        > 0.0,
    }
    np.savez_compressed(
        output_dir / "outer_full_2024.npz",
        target=frame24[TARGET].to_numpy(np.float64),
        v84=parent24,
        correction=correction24,
        candidate=candidate24,
        active=active24,
        domain3=frame24["domain3"].astype(str).to_numpy(),
        game_month=frame24["game_month"].to_numpy(np.int16),
    )
    effect_table = pd.DataFrame(
        {
            "key": list(consensus),
            "effect": list(consensus.values()),
        }
    )
    effect_table.to_csv(output_dir / "selected_consensus_effects.csv", index=False)

    result = {
        "protocol": PROTOCOL,
        "parent": "exact v84 OOF analogue on deployed R_CORE route",
        "configuration": {
            "route": ROUTE,
            "correction_space": "additive probability residual",
            "correction_cap": CORRECTION_CAP,
            "alphas": list(ALPHAS),
            "etas": list(ETAS),
            "group_specs": [
                {"name": item.name, "columns": list(item.columns)}
                for item in GROUP_SPECS
            ],
            "temporal_consensus": "same-sign geometric mean",
        },
        "parent_provenance": provenance,
        "absolute_parent_audit": {
            name: _absolute(frames[name], parents[name]) for name in frames
        },
        "source_trial_count": int(len(ranking)),
        "source_gate_pass_count": int(ranking["source_gate_passed"].sum()),
        "selected": selected,
        "selected_source_details": trial_details[selected["recipe"]],
        "consensus_effect_count": int(len(consensus)),
        "consensus_coverage_2024_r_core": float(
            np.mean(correction24[active24] != 0.0)
        ),
        "audits": {
            "outer_full_2024": full_audit,
            "replication_late_2024": late_audit,
        },
        "point_gates": {name: bool(value) for name, value in point_gates.items()},
        "point_gates_passed": bool(all(point_gates.values())),
        "eligible_for_packaging": False,
        "promotion_note": (
            "passing point gates would authorize dependence-aware bootstrap and "
            "Reality Check, not immediate packaging"
        ),
        "same_family_2024_labels_used_for_recipe_selection": False,
        "public_score_used_for_recipe_selection": False,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
        "current_pitch_trackman_or_location_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "source_gate_pass_count": result["source_gate_pass_count"],
                "full_2024": full_compact,
                "late_2024": late_compact,
                "point_gates": point_gates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
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
