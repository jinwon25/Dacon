"""Audit a fixed 50:50 classifier/regressor XGB inside Public1175."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.archive.v229_latest_trackman_fallback_replacement_audit import run as run_generic


PROTOCOL = "V240_BRIER_CLASSIFIER_FIXED_ENSEMBLE_AUDIT_V1"
LABEL = "classifier_brier_5050"


def fixed_average(classifier: np.ndarray, brier: np.ndarray) -> np.ndarray:
    classifier = np.asarray(classifier, dtype=np.float64)
    brier = np.asarray(brier, dtype=np.float64)
    if classifier.shape != brier.shape:
        raise ValueError("XGB ensemble prediction shape mismatch")
    return np.clip(0.5 * classifier + 0.5 * brier, 0.001, 0.999)


def run(
    train_csv: Path,
    current_oof_dir: Path,
    brier_oof_dir: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
):
    averaged = output_dir / "fixed_ensemble_oof"
    averaged.mkdir(parents=True, exist_ok=True)
    for year in (2022, 2023, 2024):
        current = np.load(
            current_oof_dir / f"fallback_xgb_oof_{year}.npy", allow_pickle=False
        )
        brier = np.load(
            brier_oof_dir / f"brier_xgb_{year}.npy", allow_pickle=False
        )
        np.save(
            averaged / f"fixed_ensemble_xgb_{year}.npy",
            fixed_average(current, brier).astype(np.float32),
            allow_pickle=False,
        )
    result = run_generic(
        train_csv, current_oof_dir, averaged, contract_dir, bridge_oof,
        output_dir,
        candidate_filename_template="fixed_ensemble_xgb_{year}.npy",
        candidate_label=LABEL,
        protocol=PROTOCOL,
    )
    result["restrictions"]["single_latest_trackman_candidate_only"] = False
    result["restrictions"]["fixed_classifier_brier_5050_only"] = True
    result["restrictions"]["ensemble_weight_selected_from_scores"] = False
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--brier-oof-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.current_oof_dir, args.brier_oof_dir,
        args.contract_dir, args.bridge_oof, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "absolute": result[f"{LABEL}_to_parent"],
        "incremental": result[f"{LABEL}_incremental_over_current"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
