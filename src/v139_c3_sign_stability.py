"""Run the shared C3 consensus audit with sign-stability estimators."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.v136_c3_window_consensus import run


PROTOCOL = "V139_C3_SIGN_STABILITY_V1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--v133-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.contract_dir,
        args.v104_dir,
        args.v133_dir,
        args.config,
        args.output_dir,
        expected_protocol=PROTOCOL,
    )


if __name__ == "__main__":
    main()
