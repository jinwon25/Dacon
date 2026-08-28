"""Re-audit metric fidelity, OOF provenance, and local-to-Public transfer.

This audit is intentionally diagnostic.  It verifies the official Brier Skill
Score implementation and row alignment, then separates genuinely forward OOF
from same-season or repeatedly inspected development axes.  Historical Public
outcomes are used only to evaluate the validation system, never to choose a
new model, gate, calibration, or blend weight.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.contract import _bss, _load_contract_axis
from src.metrics import brier_skill_score, brier_skill_score_unclipped


PROTOCOL = "V204_VALIDATION_CONTRACT_REAUDIT_V1"
HISTORICAL_TRANSFER = (
    {
        "candidate": "v167_fixed_h1_affine",
        "local_locked_gain": 2.1532287779680246,
        "public_delta": 1.7757683144,
        "baseline": "v148",
    },
    {
        "candidate": "jy_runners_high_li_bridge027",
        "local_locked_gain": 0.638287,
        "public_delta": 0.0601478118,
        "baseline": "v167",
    },
    {
        "candidate": "v180_signed_stack",
        "local_locked_gain": 0.193590,
        "public_delta": -0.0386505701,
        "baseline": "jy_champion",
    },
    {
        "candidate": "v198_total_context_stack",
        "local_locked_gain": 0.734403,
        "public_delta": -1.0831054579,
        "baseline": "jy_champion",
    },
)


def audit_axis(
    path: Path,
    train: pd.DataFrame,
    *,
    expected_year: int,
) -> dict[str, Any]:
    axis = _load_contract_axis(path)
    raw_index = np.asarray(axis["raw_index"], dtype=np.int64)
    target = np.asarray(axis["target"], dtype=np.float64)
    exact = np.asarray(axis["exact_mask"], dtype=bool)
    if raw_index.ndim != 1 or len(raw_index) != len(target):
        raise ValueError(f"invalid axis shape: {path}")
    if np.any(raw_index < 0) or np.any(raw_index >= len(train)):
        raise ValueError(f"raw index outside train: {path}")
    selected = train.iloc[raw_index].reset_index(drop=True)
    target_match = bool(
        np.array_equal(target.astype(np.int8), selected["control_success"].to_numpy(np.int8))
    )
    season_match = bool(np.all(selected["season"].to_numpy(int) == expected_year))
    field_match: dict[str, bool] = {}
    for field in ("season", "game_month", "pitcher_id", "batter_id"):
        if field in axis:
            field_match[field] = bool(
                np.array_equal(
                    np.asarray(axis[field]).astype(str),
                    selected[field].to_numpy().astype(str),
                )
            )
    prediction = np.asarray(axis["parent"], dtype=np.float64)
    unclipped_core = _bss(target[exact], prediction[exact])
    unclipped_metrics = brier_skill_score_unclipped(target[exact], prediction[exact])
    return {
        "path": path.as_posix(),
        "rows": int(len(axis["target"])),
        "exact_rows": int(exact.sum()),
        "exact_fraction": float(exact.mean()),
        "raw_index_strictly_increasing": bool(np.all(np.diff(raw_index) > 0)),
        "target_matches_official_train": target_match,
        "season_matches_expected": season_match,
        "field_matches": field_match,
        "metric_unclipped_core": float(unclipped_core),
        "metric_unclipped_public_helper": float(unclipped_metrics),
        "metric_parity_abs": float(abs(unclipped_core - unclipped_metrics)),
        "metric_official_clipped": float(brier_skill_score(target[exact], prediction[exact])),
        "base_rate": float(target[exact].mean()),
    }


def transfer_summary() -> dict[str, Any]:
    frame = pd.DataFrame(HISTORICAL_TRANSFER)
    frame["error_public_minus_local"] = (
        frame["public_delta"] - frame["local_locked_gain"]
    )
    frame["sign_concordant"] = (
        np.sign(frame["public_delta"]) == np.sign(frame["local_locked_gain"])
    )
    correlation = float(frame[["local_locked_gain", "public_delta"]].corr().iloc[0, 1])
    return {
        "usage": "validation-system audit only; prohibited for candidate tuning",
        "rows": frame.to_dict(orient="records"),
        "n": int(len(frame)),
        "sign_concordance_fraction": float(frame["sign_concordant"].mean()),
        "mean_error_public_minus_local": float(frame["error_public_minus_local"].mean()),
        "median_error_public_minus_local": float(frame["error_public_minus_local"].median()),
        "mean_absolute_error": float(frame["error_public_minus_local"].abs().mean()),
        "pearson_correlation_small_n_diagnostic": correlation,
    }


def _archive_version_count(project: Path) -> int:
    versions = set()
    for root in (project / "src/archive", project / "src/champion"):
        for path in root.glob("v*.py"):
            match = re.match(r"v(\d+)", path.name)
            if match:
                versions.add(int(match.group(1)))
    return len(versions)


def run(
    project: Path,
    train_csv: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=[
            "row_id", "season", "game_month", "pitcher_id", "batter_id",
            "control_success",
        ],
        low_memory=False,
    )
    axes = {
        "full_2022": audit_axis(
            contract_dir / "v84_full_2022.npz", train, expected_year=2022
        ),
        "late_2023": audit_axis(
            contract_dir / "v84_late_2023.npz", train, expected_year=2023
        ),
        "full_2024": audit_axis(bridge_oof, train, expected_year=2024),
    }
    manifest1161 = json.loads(
        (contract_dir / "manifest.json").read_text(encoding="utf-8")
    )
    provenance = {
        item["axis"].replace("v84_", ""): {
            key: item[key]
            for key in (
                "role", "fidelity", "evaluation_season", "training_max_season",
                "exact_fraction",
            )
        }
        for item in manifest1161["axes"]
    }
    # The newer bridge manifest explicitly marks its only 2024 axis as
    # development-contaminated; retain that as an independent provenance fact.
    bridge_manifest = json.loads(
        (bridge_oof.parent / "manifest.json").read_text(encoding="utf-8")
    )
    provenance["full_2024_bridge"] = {
        "role": "development_contaminated_family_inspected",
        "evaluation_season": 2024,
        "training_max_season": 2023,
        "full_2024_is_development_contaminated": bool(
            bridge_manifest["restrictions"]["full_2024_is_development_contaminated"]
        ),
    }
    experiments_path = project / "reports/experiments.csv"
    experiment_rows = (
        int(sum(1 for _ in experiments_path.open(encoding="utf-8-sig")) - 1)
        if experiments_path.exists()
        else 0
    )
    archive_versions = _archive_version_count(project)
    alignment_pass = all(
        item["target_matches_official_train"]
        and item["season_matches_expected"]
        and all(item["field_matches"].values())
        and item["metric_parity_abs"] <= 1e-10
        for item in axes.values()
    )
    strict_forward_full_pipeline_axes = [
        name
        for name, item in provenance.items()
        if name in axes
        and int(item["training_max_season"]) < int(item["evaluation_season"])
        and "development_contaminated" not in str(item["role"])
    ]
    summary = {
        "protocol": PROTOCOL,
        "status": "metric_and_alignment_pass_validation_contract_reject"
        if alignment_pass
        else "metric_or_alignment_failure",
        "official_metric": {
            "formula": "max(0, 100000 * (1 - mean((p-y)^2) / (mean(y)*(1-mean(y)))))",
            "implementation_parity_passed": alignment_pass,
            "public_fraction": 1.0,
        },
        "axes": axes,
        "provenance": provenance,
        "strict_forward_full_pipeline_axes": strict_forward_full_pipeline_axes,
        "strict_forward_full_pipeline_axis_count": len(strict_forward_full_pipeline_axes),
        "historical_local_to_public_transfer": transfer_summary(),
        "research_multiplicity": {
            "distinct_version_numbers": archive_versions,
            "experiment_log_rows": experiment_rows,
            "largest_recent_declared_reality_check_family": 132,
            "global_research_family_fully_covered": False,
        },
        "diagnosis": {
            "metric_formula_bug": False,
            "row_alignment_bug": not alignment_pass,
            "primary_bottleneck": (
                "temporal/development contamination plus repeated-selection bias; "
                "within-axis bootstrap does not estimate next-season transfer"
            ),
            "current_artifacts_support_1175_expectation": False,
        },
        "new_primary_gate": {
            "component_level_strict_forward_full_years": [2022, 2023, 2024],
            "training_rule": "all fitted state must use season < audit_year",
            "recent_year_gain_required": "positive on each of 2022, 2023, 2024",
            "month_requirement": "positive in at least 60% per-year months",
            "selection_rule": "predeclared family; 2024 cannot rescue a 2022/2023 reject",
            "robustness_rule": (
                "pitcher/crossed/block p05 plus full supplied-family Reality Check; "
                "these are secondary to worst-year transfer"
            ),
            "packaging_rule": (
                "full-pipeline contaminated axes may veto but never promote; "
                "runtime singleton/shuffle/partition parity remains mandatory"
            ),
        },
        "restrictions": {
            "historical_public_used_for_new_candidate_selection": False,
            "test_csv_read": False,
            "test_distribution_used": False,
            "diagnostic_only": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(summary["historical_local_to_public_transfer"]["rows"]).to_csv(
        output_dir / "historical_transfer.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.project.resolve(), args.train_csv, args.contract_dir,
        args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
