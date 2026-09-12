"""Audit v356 formula parity to exact v345 and row-local execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.audit_v354_late_hierarchy import run as run_hierarchy_audit


def run(
    v345_zip: Path,
    v356_zip: Path,
    input_csv: Path,
    output_json: Path,
    rows_per_route: int,
) -> dict[str, Any]:
    result = run_hierarchy_audit(
        v353_zip=v345_zip,
        v354_zip=v356_zip,
        input_csv=input_csv,
        output_json=output_json,
        rows_per_route=rows_per_route,
    )
    result["protocol"] = "V356_LATE_HIERARCHY_ONLY_RUNTIME_AUDIT_V1"
    result["parent"] = "exact_v345"
    result["trackman_component_included"] = False
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v345-zip", type=Path, required=True)
    parser.add_argument("--v356-zip", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows-per-route", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.v345_zip,
                args.v356_zip,
                args.input_csv,
                args.output_json,
                args.rows_per_route,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
