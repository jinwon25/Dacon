"""Strict inter-season audit of Hoo-inspired baseball lookup directions.

The public repository supplied hypotheses, not fitted coefficients.  This
audit rebuilds every lookup from official-train residuals in the two seasons
strictly before an audit year.  Residuals come from the exact-anchor fallback
OOF model, and adjustments are queried row-locally only on the regular,
non-anchor domain.  No audit-season distribution is used to build or scale a
direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
)
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics
from src.archive.v248_fixed_route_dual_tree_fallback import _load_axes


PROTOCOL = "V283_HOO_LOOKUP_TRANSFER_AUDIT_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
YEAR_BY_AXIS = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
DOSES = (0.10, 0.25, 0.40, 0.65, 1.00)


def core_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["game_type"].astype(str).eq("R").to_numpy()
        & ~frame["pitcher_team_id"].eq(13).to_numpy()
        & ~frame["batter_team_id"].eq(13).to_numpy()
    )


def _lookup(table: pd.Series, values: pd.Series) -> np.ndarray:
    return values.map(table).fillna(0.0).to_numpy(np.float64)


def _binary_contrast_table(
    history: pd.DataFrame,
    context: str,
    k: float,
) -> tuple[pd.Series, pd.Series]:
    grouped = history.groupby(["pitcher_id", context], sort=True)["residual"].agg(
        ["mean", "size"]
    ).unstack()
    required = [("mean", 0), ("mean", 1), ("size", 0), ("size", 1)]
    for column in required:
        if column not in grouped:
            grouped[column] = np.nan
    valid = grouped[required].notna().all(axis=1)
    grouped = grouped.loc[valid]
    n0 = grouped[("size", 0)].astype(np.float64)
    n1 = grouped[("size", 1)].astype(np.float64)
    effective = n0 * n1 / (n0 + n1)
    sufficient = effective * (
        grouped[("mean", 1)].astype(np.float64)
        - grouped[("mean", 0)].astype(np.float64)
    )
    direction = sufficient / (effective + float(k))
    return direction, effective


def binary_contrast(
    history: pd.DataFrame,
    reference: pd.DataFrame,
    rows: pd.DataFrame,
    context: str,
    *,
    k: float,
    shape_k: float | None = None,
    shape_t: float = 0.0,
) -> np.ndarray:
    base, effective = _binary_contrast_table(history, context, k)
    if shape_k is not None:
        sufficient = base * (effective + float(k))
        shaped = sufficient / (effective + float(shape_k))
        base_ref = _lookup(base, reference["pitcher_id"])
        shaped_ref = _lookup(shaped, reference["pitcher_id"])
        ref_sign = np.where(reference[context].to_numpy(np.int8) == 1, 0.5, -0.5)
        base_sd = float(np.std(base_ref * ref_sign))
        shaped_sd = float(np.std(shaped_ref * ref_sign))
        alpha = base_sd / shaped_sd if shaped_sd > 0.0 else 1.0
        base = base + float(shape_t) * (alpha * shaped - base)
    queried = _lookup(base, rows["pitcher_id"])
    sign = np.where(rows[context].to_numpy(np.int8) == 1, 0.5, -0.5)
    return queried * sign


def nested_platoon(
    history: pd.DataFrame,
    reference: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    k: float,
    shape_k: float | None = None,
    shape_t: float = 0.0,
) -> np.ndarray:
    parent = history.groupby("pitcher_id", sort=True)["control_success"].mean()
    child = history.groupby(["pitcher_id", "batter_hand"], sort=True)[
        "control_success"
    ].agg(["mean", "size"])
    parent_for_child = child.index.get_level_values("pitcher_id").map(parent)
    sufficient = child["size"].to_numpy(np.float64) * (
        child["mean"].to_numpy(np.float64) - np.asarray(parent_for_child, np.float64)
    )
    base = pd.Series(
        sufficient / (child["size"].to_numpy(np.float64) + float(k)),
        index=child.index,
    )
    if shape_k is not None:
        shaped = pd.Series(
            sufficient / (child["size"].to_numpy(np.float64) + float(shape_k)),
            index=child.index,
        )
        ref_keys = pd.MultiIndex.from_frame(reference[["pitcher_id", "batter_hand"]])
        base_ref = base.reindex(ref_keys).fillna(0.0).to_numpy(np.float64)
        shaped_ref = shaped.reindex(ref_keys).fillna(0.0).to_numpy(np.float64)
        shaped_sd = float(np.std(shaped_ref))
        alpha = float(np.std(base_ref)) / shaped_sd if shaped_sd > 0.0 else 1.0
        base = base + float(shape_t) * (alpha * shaped - base)
    keys = pd.MultiIndex.from_frame(rows[["pitcher_id", "batter_hand"]])
    return base.reindex(keys).fillna(0.0).to_numpy(np.float64)


def residual_level(
    history: pd.DataFrame,
    reference: pd.DataFrame,
    rows: pd.DataFrame,
    entity: str,
    *,
    k: float,
    weight: float,
    shape_k: float | None = None,
    shape_t: float = 0.0,
) -> np.ndarray:
    stats = history.groupby(entity, sort=True)["residual"].agg(["sum", "size"])
    base = stats["sum"] / (stats["size"] + float(k))
    if shape_k is not None:
        shaped = stats["sum"] / (stats["size"] + float(shape_k))
        base_ref = _lookup(base, reference[entity])
        shaped_ref = _lookup(shaped, reference[entity])
        shaped_sd = float(np.std(shaped_ref))
        alpha = float(np.std(base_ref)) / shaped_sd if shaped_sd > 0.0 else 1.0
        base = base + float(shape_t) * (alpha * shaped - base)
    return float(weight) * _lookup(base, rows[entity])


def exposure_direction(
    history: pd.DataFrame,
    reference: pd.DataFrame,
    rows: pd.DataFrame,
    entity: str,
    *,
    k: float,
    weight: float,
    log_counts: bool,
    strength: float,
) -> np.ndarray:
    stats = history.groupby(entity, sort=True)["residual"].agg(["sum", "size"])
    base = stats["sum"] / (stats["size"] + float(k))
    counts = stats["size"].astype(np.float64)
    raw = np.log1p(counts) if log_counts else counts.copy()
    raw = raw - float(raw.mean())
    base_ref = _lookup(base, reference[entity])
    raw_ref = _lookup(raw, reference[entity])
    centered = bool(log_counts)
    a = raw_ref - raw_ref.mean() if centered else raw_ref
    b = base_ref - base_ref.mean() if centered else base_ref
    denom = float(np.dot(b, b))
    beta = float(np.dot(a, b) / denom) if denom > 0.0 else 0.0
    orthogonal = raw - beta * base
    orth_ref = _lookup(orthogonal, reference[entity])
    orth_sd = float(np.std(orth_ref))
    alpha = float(np.std(base_ref)) / orth_sd if orth_sd > 0.0 else 0.0
    return float(weight) * float(strength) * alpha * _lookup(orthogonal, rows[entity])


def direction_library(
    history: pd.DataFrame,
    reference: pd.DataFrame,
    rows: pd.DataFrame,
) -> dict[str, np.ndarray]:
    hand_base = binary_contrast(history, reference, rows, "same_hand", k=1000.0)
    hand_shape = binary_contrast(
        history,
        reference,
        rows,
        "same_hand",
        k=1000.0,
        shape_k=100.0,
        shape_t=3.0,
    )
    two = binary_contrast(history, reference, rows, "two_strike", k=1000.0)
    runner = binary_contrast(history, reference, rows, "runner_on", k=2000.0)
    return {
        "nested_platoon_k300": nested_platoon(
            history, reference, rows, k=300.0
        ),
        "nested_platoon_shape_k30_t3": nested_platoon(
            history, reference, rows, k=300.0, shape_k=30.0, shape_t=3.0
        ),
        "resid_hand_k1000": hand_base,
        "resid_hand_shape_k100_t3": hand_shape,
        "resid_two_strike_k1000": two,
        "resid_runner_k2000": runner,
        "resid_contrast_triplet_w065": 0.65 * (hand_base + two + runner),
        "resid_contrast_shape_triplet_w065": 0.65 * (hand_shape + two + runner),
        "batter_level_k20000": residual_level(
            history, reference, rows, "batter_id", k=20000.0, weight=2.105
        ),
        "batter_level_shape_k2000_tm45": residual_level(
            history,
            reference,
            rows,
            "batter_id",
            k=20000.0,
            weight=2.105,
            shape_k=2000.0,
            shape_t=-4.5,
        ),
        "batter_exposure_n_s04": exposure_direction(
            history,
            reference,
            rows,
            "batter_id",
            k=20000.0,
            weight=2.105,
            log_counts=False,
            strength=0.4,
        ),
        "pitcher_level_k50000": residual_level(
            history, reference, rows, "pitcher_id", k=50000.0, weight=1.985
        ),
        "pitcher_exposure_logn_s04": exposure_direction(
            history,
            reference,
            rows,
            "pitcher_id",
            k=50000.0,
            weight=1.985,
            log_counts=True,
            strength=0.4,
        ),
    }


def _augment(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    if len(prediction) != int(regular.sum()):
        raise ValueError("exact OOF prediction length mismatch")
    output = frame.loc[regular].copy()
    output["prediction"] = np.asarray(prediction, dtype=np.float64)
    output["residual"] = (
        output["control_success"].to_numpy(np.float64)
        - output["prediction"].to_numpy(np.float64)
    )
    output["same_hand"] = output["pitcher_hand"].astype(str).eq(
        output["batter_hand"].astype(str)
    ).astype(np.int8)
    output["two_strike"] = output["strikes_before"].eq(2).astype(np.int8)
    output["runner_on"] = output["num_runners_on"].gt(0).astype(np.int8)
    return output.loc[core_mask(output)].reset_index(drop=True)


def _prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["same_hand"] = output["pitcher_hand"].astype(str).eq(
        output["batter_hand"].astype(str)
    ).astype(np.int8)
    output["two_strike"] = output["strikes_before"].eq(2).astype(np.int8)
    output["runner_on"] = output["num_runners_on"].gt(0).astype(np.int8)
    return output


def _select(ranking: pd.DataFrame) -> dict[str, Any]:
    passed = ranking.loc[ranking["source_gate_passed"]]
    table = passed if len(passed) else ranking
    return table.sort_values(
        ["source_min_gain", "source_mean_gain", "source_worst_month"],
        ascending=False,
        kind="stable",
    ).iloc[0].to_dict()


def run(
    train_csv: Path,
    exact_oof_dir: Path,
    v244_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _context, frames, _correction = _load_year_context(train_csv)
    identity = pd.read_csv(
        train_csv, usecols=["season", "batter_id"], low_memory=False
    )
    for year in range(2020, 2025):
        batter = identity.loc[identity["season"].eq(year), "batter_id"].reset_index(
            drop=True
        )
        if len(batter) != len(frames[year]):
            raise ValueError(f"batter identity/frame length mismatch: {year}")
        frames[year] = frames[year].copy()
        frames[year]["batter_id"] = batter.to_numpy()
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": _prepare_rows(frames[2022].reset_index(drop=True)),
        "late_2023": _prepare_rows(
            frames[2023].loc[late23].reset_index(drop=True)
        ),
        "full_2024": _prepare_rows(frames[2024].reset_index(drop=True)),
    }
    axes = _load_axes(contract_dir, bridge_oof)
    with np.load(v244_axes, allow_pickle=False) as saved:
        baseline = {
            axis: saved[
                f"candidate_runtime_faithful_exact_jy_{axis}"
            ].astype(np.float64)
            for axis in AXES
        }

    historical: dict[int, pd.DataFrame] = {}
    for year in range(2020, 2025):
        prediction = np.load(
            exact_oof_dir / f"training_parity_exact_xgb_{year}.npy",
            allow_pickle=False,
        ).astype(np.float64)
        historical[year] = _augment(frames[year].reset_index(drop=True), prediction)

    directions: dict[str, dict[str, np.ndarray]] = {}
    support: dict[str, np.ndarray] = {}
    for axis in AXES:
        year = YEAR_BY_AXIS[axis]
        history = pd.concat(
            [historical[year - 2], historical[year - 1]], ignore_index=True
        )
        reference = historical[year - 1]
        rows = axis_frames[axis]
        directions[axis] = direction_library(history, reference, rows)
        support[axis] = core_mask(rows)
        for name in directions[axis]:
            directions[axis][name] = np.where(
                support[axis], directions[axis][name], 0.0
            )

    families = tuple(directions["full_2022"])
    trials: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for family in families:
        for dose in DOSES:
            key = f"{family}__d{dose:g}"
            per_axis = {}
            for axis in AXES:
                candidate = np.clip(
                    baseline[axis] + float(dose) * directions[axis][family],
                    0.001,
                    0.999,
                )
                predictions[(key, axis)] = candidate
                per_axis[axis] = paired_metrics(
                    axes[axis], baseline[axis], candidate, support[axis]
                )
            source_gate = all(
                per_axis[axis]["gain"] > 0.0
                and per_axis[axis]["positive_month_fraction"] >= 0.5
                and per_axis[axis]["worst_month_gain"] > -10.0
                for axis in SOURCE_AXES
            )
            trials.append(
                {
                    "key": key,
                    "family": family,
                    "dose": float(dose),
                    "source_gate_passed": bool(source_gate),
                    "source_min_gain": float(
                        min(per_axis[a]["gain"] for a in SOURCE_AXES)
                    ),
                    "source_mean_gain": float(
                        np.mean([per_axis[a]["gain"] for a in SOURCE_AXES])
                    ),
                    "source_worst_month": float(
                        min(per_axis[a]["worst_month_gain"] for a in SOURCE_AXES)
                    ),
                    "locked_gain": float(per_axis["full_2024"]["gain"]),
                    "locked_positive_month_fraction": float(
                        per_axis["full_2024"]["positive_month_fraction"]
                    ),
                    "locked_worst_month": float(
                        per_axis["full_2024"]["worst_month_gain"]
                    ),
                }
            )
            details[key] = per_axis

    ranking = pd.DataFrame(trials)
    selected = _select(ranking)
    selected_key = str(selected["key"])
    selected_locked = details[selected_key]["full_2024"]
    locked_family = [
        predictions[(f"{selected['family']}__d{dose:g}", "full_2024")]
        for dose in DOSES
    ]
    robustness = _robustness(
        axes["full_2024"],
        baseline["full_2024"],
        predictions[(selected_key, "full_2024")],
        support["full_2024"],
        locked_family,
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    locked_pass = bool(
        selected_locked["gain"] >= 4.0
        and selected_locked["positive_month_fraction"] >= 0.625
        and selected_locked["worst_month_gain"] > -10.0
    )
    promote = bool(selected["source_gate_passed"] and locked_pass and robust_pass)

    ranking.sort_values(
        ["source_gate_passed", "source_min_gain", "source_mean_gain"],
        ascending=False,
        kind="stable",
    ).to_csv(output_dir / "source_ranking.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"baseline_{axis}": baseline[axis] for axis in AXES},
        **{
            f"candidate_{axis}": predictions[(selected_key, axis)]
            for axis in AXES
        },
        **{f"direction_{axis}": directions[axis][str(selected["family"])] for axis in AXES},
        **{f"support_{axis}": support[axis] for axis in AXES},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "promote_to_release_build" if promote else "reject",
        "selected": selected,
        "selected_metrics": details[selected_key],
        "locked_robustness": robustness,
        "locked_point_passed": locked_pass,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": promote,
        "family_count": len(families),
        "trial_count": len(trials),
        "audit_years": [2022, 2023, 2024],
        "history_window_years": 2,
        "restrictions": {
            "official_train_only": True,
            "lookup_labels_strictly_prior_to_audit_year": True,
            "lookup_scaling_uses_prior_season_reference_only": True,
            "exact_anchor_oof_residuals": True,
            "regular_nonanchor_application_only": True,
            "public_repository_hypotheses_only": True,
            "public_repository_fitted_coefficients_reused": False,
            "full_2024_used_for_selection": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--exact-oof-dir", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.exact_oof_dir,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected": result["selected"],
                "metrics": result["selected_metrics"],
                "robustness": result["locked_robustness"],
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
