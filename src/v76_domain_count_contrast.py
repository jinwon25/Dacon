"""Low-DOF domain-by-count residual contrast above frozen OOF parents.

Absolute domain rates are unstable, especially for Futures games.  This
candidate removes each domain's residual level and transfers only the shape
across the 12 legal ball-strike states.  The fixed EB prior is not tuned per
axis.  Because the family was motivated by the contaminated v75 diagnostic,
all results remain research-only even if older axes pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics
from src.core.axes import _cached_v25_axes


PRIOR = 2000.0


def _count_state(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["balls_before"], errors="raise").astype(int).astype(str)
        + "-"
        + pd.to_numeric(frame["strikes_before"], errors="raise").astype(int).astype(str)
    )


def fit_domain_count_contrast(
    frame: pd.DataFrame,
    parent: np.ndarray,
    *,
    prior: float = PRIOR,
) -> pd.DataFrame:
    """Fit domain-centered count residuals on labelled source rows."""

    local = pd.DataFrame(
        {
            "domain3": frame["domain3"].astype(str).to_numpy(),
            "count_state": _count_state(frame).to_numpy(),
            "residual": frame["target"].to_numpy(np.float64)
            - np.asarray(parent, dtype=np.float64),
        }
    )
    grouped = (
        local.groupby(["domain3", "count_state"], observed=True)["residual"]
        .agg(["sum", "size"])
        .reset_index()
    )
    grouped["raw_offset"] = grouped["sum"] / (grouped["size"] + float(prior))
    weighted = grouped["raw_offset"] * grouped["size"]
    domain_center = (
        pd.DataFrame(
            {
                "domain3": grouped["domain3"],
                "weighted": weighted,
                "size": grouped["size"],
            }
        )
        .groupby("domain3", observed=True)
        .sum()
    )
    domain_center["center"] = domain_center["weighted"] / domain_center["size"]
    grouped = grouped.merge(
        domain_center[["center"]].reset_index(),
        on="domain3",
        how="left",
        validate="many_to_one",
    )
    grouped["correction"] = grouped["raw_offset"] - grouped["center"]
    return grouped[
        ["domain3", "count_state", "size", "raw_offset", "center", "correction"]
    ]


def apply_domain_count_contrast(
    frame: pd.DataFrame,
    parent: np.ndarray,
    table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    query = pd.DataFrame(
        {
            "domain3": frame["domain3"].astype(str).to_numpy(),
            "count_state": _count_state(frame).to_numpy(),
            "__order": np.arange(len(frame), dtype=np.int64),
        }
    )
    joined = query.merge(
        table[["domain3", "count_state", "correction"]],
        on=["domain3", "count_state"],
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("__order", kind="stable")
    correction = joined["correction"].fillna(0.0).to_numpy(np.float64)
    candidate = np.clip(np.asarray(parent, dtype=np.float64) + correction, 0.001, 0.999)
    return candidate, correction


def _transition(
    source_frame: pd.DataFrame,
    source_parent: np.ndarray,
    audit_frame: pd.DataFrame,
    audit_parent: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, pd.DataFrame]:
    table = fit_domain_count_contrast(source_frame, source_parent)
    candidate, correction = apply_domain_count_contrast(audit_frame, audit_parent, table)
    active = np.abs(correction) > 0.0
    result = diagnostics(audit_frame, audit_parent, candidate, active)
    result.update(
        {
            "source_rows": int(len(source_frame)),
            "source_cells": int(len(table)),
            "active_rows": int(active.sum()),
            "mean_abs_shift": float(np.mean(np.abs(candidate - audit_parent))),
            "correction_max_abs": float(np.max(np.abs(correction))),
        }
    )
    return result, candidate, table


def _historical_frame(
    raw: pd.DataFrame, state_dir: Path, year: int
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
    with np.load(state_dir / f"selected_state_o{year}.npz", allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        parent = saved["incumbent"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    if not np.array_equal(target, frame["control_success"].to_numpy(np.float64)):
        raise ValueError(f"historical target parity failure: {year}")
    frame = frame[["season", "game_month", "balls_before", "strikes_before"]].copy()
    frame["target"] = target
    frame["domain3"] = domain
    return frame, parent


def run(
    project: Path,
    state_dir: Path,
    current_oof_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    state_dir = state_dir.resolve()
    current_oof_dir = current_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    historical: dict[int, tuple[pd.DataFrame, np.ndarray]] = {
        year: _historical_frame(raw, state_dir, year) for year in (2022, 2023, 2024)
    }
    frame22, parent22 = historical[2022]
    frame23, parent23 = historical[2023]
    frame24, parent24 = historical[2024]
    early22 = frame22["game_month"].le(7).to_numpy()
    late22 = frame22["game_month"].ge(8).to_numpy()

    exact_axes = _cached_v25_axes(project, raw)
    late23_exact = exact_axes["selection_late_2023"]
    full24_exact = exact_axes["outer_full_2024"]
    late24_exact = exact_axes["replication_late_2024"]
    exact_parent23 = np.load(current_oof_dir / "selection_late_2023.npz")[
        "final_gate_parent"
    ].astype(np.float64)
    exact_parent24 = np.load(current_oof_dir / "outer_full_2024.npz")[
        "final_gate_parent"
    ].astype(np.float64)
    exact_parent_late24 = np.load(current_oof_dir / "replication_late_2024.npz")[
        "final_gate_parent"
    ].astype(np.float64)
    early24 = full24_exact["game_month"].le(7).to_numpy()

    axes = {
        "early22_to_late22": (
            frame22.loc[early22].reset_index(drop=True),
            parent22[early22],
            frame22.loc[late22].reset_index(drop=True),
            parent22[late22],
        ),
        "full22_to_full23": (frame22, parent22, frame23, parent23),
        "full23_to_full24_historical_parent": (frame23, parent23, frame24, parent24),
        "late23_to_full24_exact_parent": (
            late23_exact,
            exact_parent23,
            full24_exact,
            exact_parent24,
        ),
        "early24_to_late24_exact_parent": (
            full24_exact.loc[early24].reset_index(drop=True),
            exact_parent24[early24],
            late24_exact,
            exact_parent_late24,
        ),
    }
    audits: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    payload: dict[str, np.ndarray] = {}
    for name, values in axes.items():
        result, candidate, table = _transition(*values)
        audits[name] = result
        payload[name] = candidate
        table.to_csv(output_dir / f"{name}_table.csv", index=False)
        rows.append(
            {
                "axis": name,
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"months", "domain_gains"}
                },
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    np.savez_compressed(output_dir / "predictions.npz", **payload)

    primary_names = ("early22_to_late22", "full22_to_full23")
    primary = [audits[name] for name in primary_names]
    gates = {
        "older_gains_positive": bool(all(item["gain"] > 0.0 for item in primary)),
        "older_month_fraction_at_least_075": bool(
            all(item["positive_month_fraction"] >= 0.75 for item in primary)
        ),
        "older_minimum_domains_nonnegative": bool(
            all(item["minimum_domain_gain"] >= 0.0 for item in primary)
        ),
        "exact_2024_axes_positive": bool(
            audits["late23_to_full24_exact_parent"]["gain"] > 0.0
            and audits["early24_to_late24_exact_parent"]["gain"] > 0.0
        ),
    }
    result = {
        "protocol": "V76_DOMAIN_COUNT_CENTERED_CONTRAST_ABOVE_1158_V1",
        "recipe": {
            "cells": "domain3 x exact balls-strikes state",
            "prior": PRIOR,
            "domain_level_removed": True,
            "weight": 1.0,
        },
        "motivation_contaminated_by_v75": True,
        "audits": audits,
        "gates": gates,
        "passes_point_estimate_gates": bool(all(gates.values())),
        "eligible_for_public_probe": False,
        "decision": "locked_shadow_needed" if all(gates.values()) else "reject",
        "test_csv_read": False,
        "row_local_inference": True,
        "other_test_rows_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.state_dir, args.current_oof_dir, args.output_dir)


if __name__ == "__main__":
    main()
