"""Fit the frozen 2024 post-break R_ANCHOR direct-probability overlay.

The model family was selected on early -> late 2023.  It was then audited on
full 2024 after a 2023 refit and independently replicated on early -> late
2024.  Deployment uses 2024 labels only and never inspects test aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.core.axes import _joint_domain
from src.champion.v23_postbreak_gam_screen import SPECS, _pipeline
from src.core.axes import _derived
from src.core.v25_recipe import SOURCE_YEAR, MODEL_NAME, ETA, APPLY_DOMAIN


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run(
    project: Path,
    output_dir: Path,
    audit_summary_path: Path = Path(
        "reproduction_inputs/v25_postbreak_anchor_audit_summary.json"
    ),
) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = project / "data" / "train.csv"
    raw = pd.read_csv(train_path, low_memory=False)
    source = raw.loc[raw["season"].eq(SOURCE_YEAR)].reset_index(drop=True)
    source = _derived(source, _joint_domain(source))
    if source.empty:
        raise ValueError(f"no source rows for {SOURCE_YEAR}")
    spec = next(item for item in SPECS if item.name == MODEL_NAME)
    model = _pipeline(source, spec)
    model.fit(source, source["control_success"].to_numpy(np.float64))

    model_path = output_dir / "v25_postbreak_anchor.joblib"
    joblib.dump(model, model_path, compress=3)
    direct = np.asarray(model.predict_proba(source)[:, 1], dtype=np.float64)
    if not np.isfinite(direct).all():
        raise ValueError("fitted model produced non-finite probabilities")

    audit_path = (
        audit_summary_path.resolve()
        if audit_summary_path.is_absolute()
        else (project / audit_summary_path).resolve()
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_candidate = {
        "model": MODEL_NAME,
        "eta": ETA,
        "apply_domain": APPLY_DOMAIN,
    }
    if audit["frozen_candidate"] != expected_candidate:
        raise ValueError("audit candidate does not match frozen deployment recipe")
    if not bool(audit["eligible_for_leaderboard_probe"]):
        raise ValueError("post-break audit did not pass the leaderboard-probe gate")
    evidence_keys = (
        "selection_early2023_to_late2023",
        "outer_full2023_to_full2024",
        "replication_early2024_to_late2024",
    )
    local_evidence = {
        key: {
            metric: audit[key][metric]
            for metric in (
                "gain",
                "applied_domain_gain",
                "positive_active_month_fraction",
                "worst_active_month_gain",
            )
        }
        for key in evidence_keys
    }

    deployment_spec = {
        "protocol": "V25_POSTBREAK_ANCHOR_2025_V1",
        "parent": "submit_v22.zip",
        "source_year": SOURCE_YEAR,
        "source_rows": int(len(source)),
        "model_name": MODEL_NAME,
        "model_file": model_path.name,
        "apply_domain": APPLY_DOMAIN,
        "blend_eta": ETA,
        "probability_clip": [0.001, 0.999],
        "training_prediction_min": float(direct.min()),
        "training_prediction_max": float(direct.max()),
        "local_evidence_bss_points": local_evidence,
        "audit_summary_sha256": _sha256(audit_path),
        "selection_note": (
            "model family/domain selected on late-2023; eta frozen conservatively; "
            "full-2024 and early-to-late-2024 are independent audits"
        ),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "external_data_used": False,
    }
    spec_path = output_dir / "v25_postbreak_anchor_spec.json"
    spec_path.write_text(
        json.dumps(deployment_spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (model_path, spec_path)
    }
    manifest = {
        "protocol": deployment_spec["protocol"],
        "train_sha256": _sha256(train_path),
        "audit_summary_sha256": _sha256(audit_path),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "reference_model_read_during_fit": False,
        "test_values_read_during_fit": False,
        "private_score_recomputed": False,
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v25_postbreak_anchor_20260817_01"),
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=Path("reproduction_inputs/v25_postbreak_anchor_audit_summary.json"),
    )
    args = parser.parse_args()
    target = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (args.project.resolve() / args.output_dir).resolve()
    )
    if target.exists():
        raise FileExistsError("Use a new output directory")
    run(args.project, args.output_dir, args.audit_summary)


if __name__ == "__main__":
    main()
