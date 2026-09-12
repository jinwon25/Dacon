"""Audit the joint-workload H1 recipe with the frozen v209 confirmation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive import v209_h1_workload_multiseed_audit as v209


PROTOCOL = "V216_JOINT_WORKLOAD_H1_MULTISEED_AUDIT_V1"


def restrictions() -> dict[str, bool]:
    return {
        **v209.restrictions(),
        "same_recipe_as_v214_seed42": True,
        "fixed_confirmation_seeds_from_v215": True,
        "joint_denominator_rule_frozen_before_confirmation": True,
    }


def load_joint_seed_predictions(
    year: int,
    seed: int,
    baseline_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.load(
        baseline_dir / f"h1_year{year}_seed{seed}.npy", allow_pickle=False
    ).astype(np.float64)
    joint_dir = seed42_dir if seed == 42 else multiseed_dir
    joint = np.load(
        joint_dir / f"joint_h1_year{year}_seed{seed}.npy",
        allow_pickle=False,
    ).astype(np.float64)
    if baseline.shape != joint.shape:
        raise ValueError(f"paired seed shape mismatch: {year}/{seed}")
    return baseline, joint


def run(
    train_csv: Path,
    baseline_checkpoint_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Reuse the frozen v209 audit, changing only the checkpoint loader."""

    original_loader = v209.load_seed_predictions
    v209.load_seed_predictions = load_joint_seed_predictions
    try:
        summary = v209.run(
            train_csv,
            baseline_checkpoint_dir,
            seed42_dir,
            multiseed_dir,
            contract_dir,
            v104_path,
            h1_path,
            c3_path,
            v160_path,
            bridge_oof,
            output_dir,
        )
    finally:
        v209.load_seed_predictions = original_loader

    summary["protocol"] = PROTOCOL
    summary["candidate"] = "joint_workload_h1_v214_v215"
    summary["audit_contract_reused_from"] = v209.PROTOCOL
    summary["restrictions"] = restrictions()
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed42-dir", type=Path, required=True)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.baseline_checkpoint_dir,
        args.seed42_dir,
        args.multiseed_dir,
        args.contract_dir,
        args.v104_path,
        args.h1_path,
        args.c3_path,
        args.v160_path,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
