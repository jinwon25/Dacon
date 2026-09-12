"""Reconstruct and score the exact current Public-incumbent OOF lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.compose_residual_stack_candidate import apply_capped_residual_stack
from experiments.kma_year_forward_quantile_blend import apply_bounded_blend
from experiments.multimodel_expanding_quantile_blend import load_frozen_validation_baselines
from src.metrics import CAPACITY_KWH, evaluate_competition
from src.sprint066_pipeline import validate_submission


G1_WEIGHT = 0.1375
G2_WEIGHT = 0.1825


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    primary_path = Path(args.primary_cache)
    residual_path = Path(args.residual_cache)
    group3_path = Path(args.group3_cache)
    candidate_path = Path(args.candidate)
    baselines, truth_series, index, issues = load_frozen_validation_baselines(
        primary_path, residual_path, group3_path,
    )
    truth = {target: values.to_numpy(dtype=float) for target, values in truth_series.items()}
    prediction = {target: values.to_numpy(dtype=float) for target, values in baselines.items()}
    with np.load(primary_path, allow_pickle=False) as primary, np.load(
        residual_path, allow_pickle=False,
    ) as residual:
        prediction["kpx_group_1"] = apply_capped_residual_stack(
            primary["kpx_group_1__reference"],
            primary["kpx_group_1__candidate"],
            residual["kpx_group_1__candidate"],
            residual_weight=G1_WEIGHT,
            capacity=CAPACITY_KWH["kpx_group_1"],
            movement_cap_ratio=0.05,
        )
        prediction["kpx_group_2"] = apply_bounded_blend(
            primary["kpx_group_2__reference"],
            primary["kpx_group_2__expert"],
            weight=G2_WEIGHT,
            capacity=CAPACITY_KWH["kpx_group_2"],
        )

    metrics = evaluate_competition(truth, prediction)
    sample = pd.read_csv(args.sample_submission, encoding="utf-8-sig")
    candidate = pd.read_csv(candidate_path, encoding="utf-8-sig")
    validate_submission(candidate, sample)
    candidate_path.read_text(encoding="utf-8-sig")
    report = {
        "contract": {
            "submission_id": 1508386,
            "group1_residual_weight": G1_WEIGHT,
            "group2_pooled_weight": G2_WEIGHT,
            "group3_frozen": True,
            "test_data_used_for_selection": False,
        },
        "oof": {
            "rows": int(len(index)),
            "issue_cycles": int(pd.DatetimeIndex(issues).nunique()),
            "timestamp_min": str(index.min()),
            "timestamp_max": str(index.max()),
            "metrics": metrics,
        },
        "candidate": {
            "path": candidate_path.as_posix(),
            "sha256": _sha256(candidate_path),
            "rows": int(len(candidate)),
            "validated_against_sample": True,
            "utf8_reload": True,
        },
        "sources": {
            "primary_cache": {"path": primary_path.as_posix(), "sha256": _sha256(primary_path)},
            "residual_cache": {"path": residual_path.as_posix(), "sha256": _sha256(residual_path)},
            "group3_cache": {"path": group3_path.as_posix(), "sha256": _sha256(group3_path)},
        },
        "warning": "OOF metrics are historical validation, not a Public or Private score estimate.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-cache",
        default="artifacts_final/lineage/kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz",
    )
    parser.add_argument(
        "--residual-cache",
        default="artifacts_final/lineage/kma_jma_msm_stencil_production_20260726.npz",
    )
    parser.add_argument(
        "--group3-cache",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/power_curve_oof_20260725.npz",
    )
    parser.add_argument(
        "--candidate",
        default="artifacts_final/candidates/public_positive_g1w1375_g2w1825_g3frozen_20260802.csv",
    )
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--output", default="artifacts/current_incumbent_oof_audit.json")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["oof"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
