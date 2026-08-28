"""Forward-only context residual empirical-Bayes audit above the exact JY parent.

The recipe is selected on the 2022 -> late-2023 transition, then frozen and
refit on late-2023 before opening late-2024.  Lookup values use only labelled
rows from the earlier season.  Query labels, query aggregates, and test-row
statistics are never used to construct a correction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import (
    BRIDGE_SCALE,
    C3_ACTIVE_RECENT_WEIGHT,
    C3_BASE_RECENT_WEIGHT,
    H1_ACTIVE_WEIGHT,
    H1_BASE_WEIGHT,
    _load_year_context,
    _locked_contract,
    _source_contracts,
    c3_mix,
    compose,
    gate_library,
    overwrite_gate,
    rcore_mask,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V177_FORWARD_CONTEXT_RESIDUAL_EB_V1"
LATE_MONTH = 8
CORRECTION_CAP = 0.03
ALPHA_GRID = (200.0, 800.0, 3200.0)
ETA_GRID = (0.25, 0.5, 1.0)
FAMILIES: dict[str, tuple[str, ...]] = {
    "calendar": ("game_month",),
    "context": (
        "game_month",
        "count_state",
        "runner_state",
        "platoon",
    ),
    "pitcher": ("game_month", "pitcher_id"),
    "pitcher_pressure": ("game_month", "pitcher_id", "pressure_count"),
    "offense_platoon": ("game_month", "batter_team_id", "platoon"),
    "team_matchup": ("game_month", "pitcher_team_id", "batter_team_id"),
}


@dataclass(frozen=True)
class EBTable:
    columns: tuple[str, ...]
    values: dict[tuple[str, ...], float]
    alpha: float


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, row-local baseball context keys."""

    output = frame.copy()
    balls = pd.to_numeric(output["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = (
        pd.to_numeric(output["strikes_before"], errors="coerce").fillna(-1).astype(int)
    )
    runners = (
        pd.to_numeric(output["num_runners_on"], errors="coerce").fillna(0).astype(int)
    )
    output["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    output["pressure_count"] = np.where(
        balls.eq(3) | strikes.eq(2), "pressure", "neutral"
    )
    output["runner_state"] = np.where(runners.gt(0), "on", "empty")
    output["platoon"] = (
        output["pitcher_hand"].astype("string").fillna("__MISSING__")
        + "-"
        + output["batter_hand"].astype("string").fillna("__MISSING__")
    )
    return output


def _keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple[str, ...]]:
    normalized = frame.loc[:, list(columns)].astype("string").fillna("__MISSING__")
    return list(normalized.itertuples(index=False, name=None))


def fit_table(
    frame: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    fit_mask: np.ndarray,
    columns: tuple[str, ...],
    alpha: float,
) -> EBTable:
    """Fit a zero-centred EB residual table using only selected labelled rows."""

    target = np.asarray(target, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if target.shape != parent.shape or target.shape != fit_mask.shape:
        raise ValueError("fit arrays must be aligned")
    if not fit_mask.any():
        raise ValueError("fit mask is empty")
    selected = frame.loc[fit_mask, list(columns)].copy()
    selected["residual"] = target[fit_mask] - parent[fit_mask]
    grouped = selected.groupby(list(columns), dropna=False, observed=True)["residual"].agg(
        ["sum", "count"]
    )
    values: dict[tuple[str, ...], float] = {}
    for raw_key, row in grouped.iterrows():
        key_values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        key = tuple("__MISSING__" if pd.isna(value) else str(value) for value in key_values)
        values[key] = float(
            np.clip(float(row["sum"]) / (float(row["count"]) + float(alpha)),
                    -CORRECTION_CAP, CORRECTION_CAP)
        )
    return EBTable(columns=columns, values=values, alpha=float(alpha))


def predict_table(table: EBTable, frame: pd.DataFrame) -> np.ndarray:
    """Apply a frozen table row by row; unseen keys receive a neutral zero."""

    return np.asarray(
        [table.values.get(key, 0.0) for key in _keys(frame, table.columns)],
        dtype=np.float64,
    )


def apply_correction(
    parent: np.ndarray,
    correction: np.ndarray,
    active: np.ndarray,
    eta: float,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    output[active] = np.clip(
        output[active] + float(eta) * np.asarray(correction)[active], 0.001, 0.999
    )
    return output


def gain(target: np.ndarray, parent: np.ndarray, candidate: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    base_rate = float(np.mean(target))
    reference = base_rate * (1.0 - base_rate)
    mse_gain = float(
        np.mean(np.square(target - parent)) - np.mean(np.square(target - candidate))
    )
    return 100_000.0 * mse_gain / reference


def diagnostics(
    frame: pd.DataFrame,
    axis: dict[str, np.ndarray],
    parent: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    """Evaluate exact rows, with stability summaries over active months only."""

    exact = np.asarray(axis["exact_mask"], dtype=bool)
    target = np.asarray(axis["target"], dtype=np.float64)
    active = np.asarray(active, dtype=bool) & exact
    month_values = pd.to_numeric(frame["game_month"], errors="raise").to_numpy(int)
    month_rows = []
    for month in sorted(np.unique(month_values[active])):
        mask = exact & month_values.__eq__(month)
        month_rows.append(
            {
                "month": int(month),
                "gain": gain(target[mask], parent[mask], candidate[mask]),
                "active_rows": int(np.sum(active & mask)),
            }
        )
    active_gain = (
        gain(target[active], parent[active], candidate[active]) if active.any() else 0.0
    )
    return {
        "gain": gain(target[exact], parent[exact], candidate[exact]),
        "active_gain": active_gain,
        "active_rows": int(active.sum()),
        "active_fraction": float(active.sum() / exact.sum()),
        "positive_month_fraction": float(
            np.mean([row["gain"] > 0.0 for row in month_rows])
        )
        if month_rows
        else 0.0,
        "worst_month_gain": float(min(row["gain"] for row in month_rows))
        if month_rows
        else 0.0,
        "mean_abs_shift": float(np.mean(np.abs(candidate[exact] - parent[exact]))),
        "months": month_rows,
    }


def exact_jy_parents(
    axes: dict[str, dict[str, np.ndarray]],
    frames: dict[int, pd.DataFrame],
    correction: dict[int, np.ndarray],
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reconstruct exact source proxies and the deployed 2024 JY formula."""

    source = _source_contracts(axes, frames, correction, v104_path, h1_path, c3_path)
    gate = gate_library()["runners_or_high_li"]
    parents: dict[str, np.ndarray] = {}
    for axis_name, values in source.items():
        c3_base = c3_mix(values["sign"], values["recent"], C3_BASE_RECENT_WEIGHT)
        c3_active = c3_mix(values["sign"], values["recent"], C3_ACTIVE_RECENT_WEIGHT)
        base = compose(
            values["component"], values["h1"], c3_base, axes[axis_name],
            h1_weight=H1_BASE_WEIGHT,
        )
        proposal = compose(
            values["component"], values["h1"], c3_active, axes[axis_name],
            h1_weight=H1_ACTIVE_WEIGHT,
        )
        parents[axis_name], _ = overwrite_gate(
            base, proposal, axes[axis_name], values["frame"], gate
        )

    locked, parity = _locked_contract(
        axes["full_2024"], frames[2024], correction[2024], h1_path, c3_path, v160_path
    )
    with np.load(bridge_oof, allow_pickle=False) as saved:
        historical_parent = saved["parent"].astype(np.float64)
        bridge025_top = saved["v142_full_2024"].astype(np.float64) + 0.25 * (
            saved["v138_full_2024"].astype(np.float64)
            - saved["v142_full_2024"].astype(np.float64)
        )
    component_delta = (bridge025_top - historical_parent) / (1.0 - H1_BASE_WEIGHT)
    bridge_component = locked["component"] + BRIDGE_SCALE * component_delta
    proposal = compose(
        bridge_component,
        locked["h1"],
        locked["c3_active"],
        axes["full_2024"],
        h1_weight=H1_ACTIVE_WEIGHT,
    )
    parents["full_2024"], active = overwrite_gate(
        locked["current"], proposal, axes["full_2024"], frames[2024], gate
    )
    return parents, {
        "locked_runtime_parity": parity,
        "locked_jy_active_rows": int(active.sum()),
    }


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, raw_frames, post4_correction = _load_year_context(train_csv)
    late23 = raw_frames[2023]["game_month"].ge(LATE_MONTH).to_numpy()
    frames = {
        "full_2022": add_context(raw_frames[2022]),
        "late_2023": add_context(raw_frames[2023].loc[late23].reset_index(drop=True)),
        "full_2024": add_context(raw_frames[2024]),
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_jy_parents(
        axes,
        raw_frames,
        post4_correction,
        v104_path,
        h1_path,
        c3_path,
        v160_path,
        bridge_oof,
    )
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    source_fit = (
        np.asarray(axes["full_2022"]["exact_mask"], dtype=bool)
        & rcore_mask(axes["full_2022"])
    )
    query_active = (
        rcore_mask(axes["late_2023"])
        & frames["late_2023"]["game_month"].ge(LATE_MONTH).to_numpy()
    )
    for family, columns in FAMILIES.items():
        for alpha in ALPHA_GRID:
            table = fit_table(
                frames["full_2022"],
                axes["full_2022"]["target"],
                parents["full_2022"],
                source_fit,
                columns,
                alpha,
            )
            correction = predict_table(table, frames["late_2023"])
            active = query_active & np.not_equal(correction, 0.0)
            for eta in ETA_GRID:
                candidate = apply_correction(
                    parents["late_2023"], correction, active, eta
                )
                result = diagnostics(
                    frames["late_2023"], axes["late_2023"],
                    parents["late_2023"], candidate, active,
                )
                rows.append({"family": family, "alpha": alpha, "eta": eta, **result})

    table = pd.DataFrame(rows)
    table["passes_source_gate"] = (
        table["gain"].gt(0.0)
        & table["positive_month_fraction"].ge(2.0 / 3.0)
        & table["worst_month_gain"].gt(-5.0)
    )
    passing = table.loc[table["passes_source_gate"]].copy()
    if passing.empty:
        result = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "parity": parity,
            "restrictions": _restrictions(),
        }
        selected: dict[str, Any] | None = None
    else:
        passing["robust_score"] = passing[["gain", "worst_month_gain"]].min(axis=1)
        best = passing.sort_values(
            ["robust_score", "gain", "eta", "alpha"],
            ascending=[False, False, True, False],
        ).iloc[0]
        family = str(best["family"])
        alpha = float(best["alpha"])
        eta = float(best["eta"])
        locked_fit = (
            np.asarray(axes["late_2023"]["exact_mask"], dtype=bool)
            & rcore_mask(axes["late_2023"])
        )
        locked_table = fit_table(
            frames["late_2023"],
            axes["late_2023"]["target"],
            parents["late_2023"],
            locked_fit,
            FAMILIES[family],
            alpha,
        )
        locked_correction = predict_table(locked_table, frames["full_2024"])
        locked_active = (
            rcore_mask(axes["full_2024"])
            & frames["full_2024"]["game_month"].ge(LATE_MONTH).to_numpy()
            & np.not_equal(locked_correction, 0.0)
        )
        locked_candidate = apply_correction(
            parents["full_2024"], locked_correction, locked_active, eta
        )
        locked_result = diagnostics(
            frames["full_2024"], axes["full_2024"],
            parents["full_2024"], locked_candidate, locked_active,
        )
        selected = {"family": family, "alpha": alpha, "eta": eta}
        eligible = (
            locked_result["gain"] > 0.0
            and locked_result["positive_month_fraction"] >= 2.0 / 3.0
            and locked_result["worst_month_gain"] > -5.0
        )
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"],
            candidate=locked_candidate,
            correction=locked_correction,
            active=locked_active,
        )
        result = {
            "protocol": PROTOCOL,
            "status": "locked_pass" if eligible else "locked_reject",
            "selected": selected,
            "source": best.to_dict(),
            "locked_2024": locked_result,
            "eligible_for_robustness_audit": bool(eligible),
            "parity": parity,
            "restrictions": _restrictions(),
        }

    table.sort_values(["passes_source_gate", "gain"], ascending=False).to_csv(
        output_dir / "source_candidates.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def _restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "test_csv_read": False,
        "query_labels_used_to_fit_table": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
