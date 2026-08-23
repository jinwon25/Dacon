"""Rebase the externally frozen temporal-stable Ridge above exact v104 OOF.

The recipe is fixed by the public x2 V18 protocol: four hierarchical prior
levels, standardized Ridge(alpha=10000), gamma=0.75, and R_CORE-only routing.
The script uses full-2022 -> late-2023 for discovery replication and
late-2023 -> full-2024 as the locked confirmation.  Test data is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.temporal_stable_conditional import (
    FROZEN_ALPHA,
    FROZEN_GAMMA,
    _add_domain_and_pressure,
    _bss,
    _paired_cluster_bootstrap,
    build_bank,
    build_features,
)
from src.archive.v123_public_quadratic_stack import CURRENT, NAMES, _directions
from src.core.contract import _load_contract_axis


PROTOCOL = "V126_TEMPORAL_STABLE_V104_REBASE_V1"
AXES = ("full_2022", "late_2023", "full_2024")


def _frames(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }


def _metric(
    axis: dict[str, np.ndarray], parent: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    exact = axis["exact_mask"].astype(bool)
    core = exact & axis["domain3"].astype(str).__eq__("R_CORE")
    target = axis["target"].astype(np.float64)
    result: dict[str, Any] = {
        "gain": _bss(target[exact], candidate[exact]) - _bss(target[exact], parent[exact]),
        "core_gain": _bss(target[core], candidate[core]) - _bss(target[core], parent[core]),
        "exact_rows": int(exact.sum()),
        "active_core_rows": int(core.sum()),
        "mean_absolute_shift": float(np.mean(np.abs(candidate[exact] - parent[exact]))),
    }
    months = []
    for month in sorted(np.unique(axis["game_month"][exact])):
        mask = exact & axis["game_month"].astype(np.int16).__eq__(month)
        months.append(
            {
                "month": int(month),
                "rows": int(mask.sum()),
                "gain": _bss(target[mask], candidate[mask])
                - _bss(target[mask], parent[mask]),
            }
        )
    result["months"] = months
    result["positive_month_fraction"] = float(
        np.mean([item["gain"] > 0.0 for item in months])
    )
    result["worst_month_gain"] = float(min(item["gain"] for item in months))
    return result


def _fit_transition(
    source_name: str,
    audit_name: str,
    frames: dict[str, pd.DataFrame],
    axes: dict[str, dict[str, np.ndarray]],
    parents: dict[str, np.ndarray],
    banks: dict[int, dict[str, object]],
) -> dict[str, Any]:
    source_axis = axes[source_name]
    audit_axis = axes[audit_name]
    source_parent = parents[source_name]
    audit_parent = parents[audit_name]
    source_active = source_axis["exact_mask"].astype(bool) & source_axis[
        "domain3"
    ].astype(str).__eq__("R_CORE")
    audit_active = audit_axis["exact_mask"].astype(bool) & audit_axis[
        "domain3"
    ].astype(str).__eq__("R_CORE")
    source_year = int(source_axis["season"][source_active][0])
    audit_year = int(audit_axis["season"][audit_active][0])

    source_rows = frames[source_name].loc[source_active].reset_index(drop=True)
    audit_rows = frames[audit_name].loc[audit_active].reset_index(drop=True)
    source_x = build_features(
        source_rows, source_parent[source_active], banks[source_year - 1]
    )
    audit_x = build_features(
        audit_rows, audit_parent[audit_active], banks[audit_year - 1]
    )
    scaler = StandardScaler()
    fit_x = scaler.fit_transform(source_x)
    apply_x = scaler.transform(audit_x)
    model = Ridge(alpha=FROZEN_ALPHA, fit_intercept=True, solver="cholesky")
    source_residual = (
        source_axis["target"][source_active].astype(np.float64)
        - source_parent[source_active]
    )
    model.fit(fit_x, source_residual)
    active_correction = np.clip(model.predict(apply_x), -0.08, 0.08)
    correction = np.zeros(len(audit_parent), dtype=np.float64)
    correction[audit_active] = active_correction

    stresses = {}
    for gamma in (0.25, 0.50, 0.75, 1.00):
        candidate = np.clip(audit_parent + gamma * correction, 1e-6, 1.0 - 1e-6)
        stresses[str(gamma)] = _metric(audit_axis, audit_parent, candidate)
    candidate = np.clip(
        audit_parent + FROZEN_GAMMA * correction, 1e-6, 1.0 - 1e-6
    )
    return {
        "source_axis": source_name,
        "audit_axis": audit_name,
        "source_year": source_year,
        "audit_year": audit_year,
        "source_rows": int(source_active.sum()),
        "feature_count": int(source_x.shape[1]),
        "correction": correction,
        "candidate": candidate,
        "frozen_metrics": stresses[str(FROZEN_GAMMA)],
        "gamma_stress": stresses,
        "coefficient_norm": float(np.linalg.norm(model.coef_)),
        "intercept": float(model.intercept_),
    }


def _corr(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)[mask]
    b = np.asarray(right, dtype=np.float64)[mask]
    if np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_axes_path: Path,
    project: Path,
    v124_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(pd.read_csv(train_csv, low_memory=False))
    frames = _frames(train)
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz") for name in AXES
    }
    with np.load(v104_axes_path) as saved:
        parents = {name: saved[name].astype(np.float64) for name in AXES}
    for name in AXES:
        if len(frames[name]) != len(parents[name]) or len(parents[name]) != len(
            axes[name]["target"]
        ):
            raise ValueError(f"axis alignment failure: {name}")
        if not np.array_equal(
            frames[name]["control_success"].to_numpy(np.float64), axes[name]["target"]
        ):
            raise ValueError(f"target alignment failure: {name}")

    banks = {
        year: build_bank(train.loc[train["season"].eq(year)])
        for year in (2021, 2022, 2023)
    }
    discovery = _fit_transition(
        "full_2022", "late_2023", frames, axes, parents, banks
    )
    confirmation = _fit_transition(
        "late_2023", "full_2024", frames, axes, parents, banks
    )

    axis24 = axes["full_2024"]
    exact24 = axis24["exact_mask"].astype(bool)
    core24 = exact24 & axis24["domain3"].astype(str).__eq__("R_CORE")
    bootstrap = _paired_cluster_bootstrap(
        axis24["target"][exact24],
        parents["full_2024"][exact24],
        confirmation["candidate"][exact24],
        axis24["pitcher_id"][exact24],
    )

    target24, directions24, direction_audit = _directions(project)
    if not np.array_equal(target24, axis24["target"]):
        raise ValueError("v123 direction target does not align with v104 full-2024 axis")
    v124_config = json.loads(v124_config_path.read_text(encoding="utf-8"))
    selected = np.asarray(
        [v124_config["selected_point"][name] for name in NAMES], dtype=np.float64
    )
    delta = selected - CURRENT
    v124_shift = directions24 @ delta
    approximate_v124 = np.clip(
        parents["full_2024"] + v124_shift, 1e-6, 1.0 - 1e-6
    )
    correction24 = confirmation["correction"]
    approximate_candidate = np.clip(
        approximate_v124 + FROZEN_GAMMA * correction24, 1e-6, 1.0 - 1e-6
    )
    approximate_metrics = _metric(
        axis24, approximate_v124, approximate_candidate
    )
    approximate_stress = {}
    for gamma in (0.25, 0.50, 0.75, 1.00):
        candidate = np.clip(
            approximate_v124 + gamma * correction24, 1e-6, 1.0 - 1e-6
        )
        approximate_stress[str(gamma)] = _metric(
            axis24, approximate_v124, candidate
        )

    np.savez_compressed(
        output_dir / "v126_axes.npz",
        correction_late_2023=discovery["correction"],
        correction_full_2024=correction24,
        candidate_late_2023=discovery["candidate"],
        candidate_full_2024=confirmation["candidate"],
        approximate_v124_full_2024=approximate_v124,
        approximate_v124_candidate_full_2024=approximate_candidate,
    )
    summary = {
        "protocol": PROTOCOL,
        "external_recipe": {
            "alpha": FROZEN_ALPHA,
            "gamma": FROZEN_GAMMA,
            "route": "R_CORE",
            "features": 25,
            "selection": "frozen before v104/v124 evaluation",
        },
        "discovery": {
            key: value for key, value in discovery.items() if key not in {"correction", "candidate"}
        },
        "locked_confirmation": {
            key: value for key, value in confirmation.items() if key not in {"correction", "candidate"}
        },
        "locked_bootstrap": bootstrap,
        "approximate_v124_marginal": approximate_metrics,
        "approximate_v124_gamma_stress": approximate_stress,
        "orthogonality": {
            "correction_vs_total_v124_shift": _corr(correction24, v124_shift, core24),
            "correction_vs_existing_v17_direction": _corr(
                correction24, directions24[:, 0], core24
            ),
            "total_v124_shift_vs_existing_v17_direction": _corr(
                v124_shift, directions24[:, 0], core24
            ),
        },
        "direction_audit": direction_audit,
        "eligibility": {
            "replicated_positive_discovery": bool(
                discovery["frozen_metrics"]["gain"] > 0.0
            ),
            "positive_locked_confirmation": bool(
                confirmation["frozen_metrics"]["gain"] > 0.0
            ),
            "positive_locked_bootstrap_p05": bool(bootstrap["p05"] > 0.0),
            "positive_approximate_v124_marginal": bool(
                approximate_metrics["gain"] > 0.0
            ),
        },
        "restrictions": {
            "test_csv_read": False,
            "test_aggregate_used": False,
            "external_2025_outcomes_used": False,
            "row_local_inference": True,
            "approximate_v124_parent_is_diagnostic_only": True,
        },
    }
    summary["eligible_for_packaging_research"] = bool(
        all(summary["eligibility"].values())
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-axes", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--v124-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.contract_dir,
        args.v104_axes,
        args.project,
        args.v124_config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
