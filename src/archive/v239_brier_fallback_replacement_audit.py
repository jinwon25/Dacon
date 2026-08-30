"""Locked Public1175 replacement audit for the v238 Brier XGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.archive.v229_latest_trackman_fallback_replacement_audit import run as run_generic


PROTOCOL = "V239_BRIER_FALLBACK_REPLACEMENT_LOCKED_AUDIT_V1"
LABEL = "brier_xgb"


def run(
    train_csv: Path,
    current_oof_dir: Path,
    brier_oof_dir: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
):
    result = run_generic(
        train_csv, current_oof_dir, brier_oof_dir, contract_dir,
        bridge_oof, output_dir,
        candidate_filename_template="hyunku_brier_xgb_{year}.npy",
        candidate_label=LABEL,
        protocol=PROTOCOL,
    )
    result["restrictions"]["single_latest_trackman_candidate_only"] = False
    result["restrictions"]["single_brier_xgb_candidate_only"] = True
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
