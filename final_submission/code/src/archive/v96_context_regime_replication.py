"""Replicate the frozen v95 context recipe within seasons.

v95 selected its recipe before any 2023/2024 exact-v84 audit, then failed at
the late-2023 -> 2024 boundary.  This diagnostic keeps that recipe frozen and
asks whether the failure is mainly a year-boundary regime change or whether
the same context contrast is unstable even from early to late season.

This is a diagnostic only.  The 2024 labels have already been opened, so even a
positive result cannot authorize packaging or Public submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v95_multiorigin_context_transport import (
    GROUP_SPECS,
    ROUTES,
    _compact,
    _load_axis,
    _transition,
)


PROTOCOL = "V96_CONTEXT_REGIME_REPLICATION_DIAGNOSTIC_V1"


def early_late_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    month = pd.to_numeric(frame["game_month"], errors="raise").to_numpy(np.int16)
    return month <= 7, month >= 8


def _slice(frame: pd.DataFrame, values: np.ndarray, mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    return frame.loc[mask].reset_index(drop=True), np.asarray(values)[mask]


def run(project: Path, contract_dir: Path, v95_dir: Path, output_dir: Path) -> dict[str, Any]:
    project = project.resolve()
    contract_dir = contract_dir.resolve()
    v95_dir = v95_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((v95_dir / "summary.json").read_text(encoding="utf-8"))
    selected = source["configuration"]["selected"]
    spec = next(value for value in GROUP_SPECS if value.name == selected["group"])
    route = ROUTES[str(selected["route_name"])]
    alpha = float(selected["alpha"])
    eta = float(selected["eta"])
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for year in (2020, 2021, 2022, 2023, 2024):
        frame, parent, _ = _load_axis(contract_dir, raw, f"common_full_{year}")
        early, late = early_late_masks(frame)
        result, _ = _transition(
            *_slice(frame, parent, early),
            *_slice(frame, parent, late),
            spec, route, alpha, eta,
        )
        name = f"common_{year}_early_to_late"
        details[name] = result
        rows.append({"axis": name, "tier": "common_wave0", **_compact(result, route)})

    for year in (2022, 2024):
        frame, parent, exact = _load_axis(contract_dir, raw, f"v84_full_{year}")
        active = frame["domain3"].astype(str).isin(route).to_numpy()
        if not exact[active].all():
            raise ValueError(f"v84 exact mask does not cover selected route: {year}")
        early, late = early_late_masks(frame)
        result, candidate = _transition(
            *_slice(frame, parent, early),
            *_slice(frame, parent, late),
            spec, route, alpha, eta,
        )
        name = f"v84_{year}_early_to_late"
        details[name] = result
        rows.append({"axis": name, "tier": "exact_v84", **_compact(result, route)})
        if year == 2024:
            late_frame, late_parent = _slice(frame, parent, late)
            np.savez_compressed(
                output_dir / "v84_2024_early_to_late.npz",
                target=late_frame["control_success"].to_numpy(np.float64),
                parent=late_parent,
                candidate=candidate,
                domain3=late_frame["domain3"].astype(str).to_numpy(),
                game_month=late_frame["game_month"].to_numpy(np.int16),
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "metrics.csv", index=False)
    common = table.loc[table["tier"].eq("common_wave0")]
    exact = table.loc[table["tier"].eq("exact_v84")]
    result = {
        "protocol": PROTOCOL,
        "frozen_recipe": {
            "group": spec.name, "route": list(route), "alpha": alpha, "eta": eta,
            "selected_without_2023_or_2024_exact_labels": True,
        },
        "audits": details,
        "summary": {
            "common_positive_axis_fraction": float(np.mean(common["gain"] > 0.0)),
            "common_minimum_gain": float(common["gain"].min()),
            "exact_positive_axis_fraction": float(np.mean(exact["gain"] > 0.0)),
            "exact_minimum_gain": float(exact["gain"].min()),
        },
        "eligible_for_packaging": False,
        "decision_rule": "diagnostic only because 2024 labels are development-contaminated",
        "test_csv_read": False, "test_aggregate_used": False,
        "public_score_used_for_recipe_or_weight": False,
        "row_local_inference": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps({"recipe": result["frozen_recipe"], "metrics": rows, "summary": result["summary"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v95-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.contract_dir, args.v95_dir, args.output_dir)


if __name__ == "__main__":
    main()
