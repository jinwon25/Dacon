from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_audit import run_data_audit, write_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BARAM data availability, labels, metadata, and SCADA alignment.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-json", default="artifacts/p0_data_audit.json")
    parser.add_argument("--output-markdown", default="docs/P0_DATA_AUDIT.md")
    args = parser.parse_args()

    report = run_data_audit(args.data_dir)
    write_audit(report, Path(args.output_json), Path(args.output_markdown))
    summary = {
        "weather_cutoff_violations": sum(item["post_cutoff_rows"] for item in report["weather"]),
        "weather_exact_duplicates": sum(item["duplicate_forecast_grid_available_rows"] for item in report["weather"]),
        "label_groups": report["labels"]["groups"],
        "scada_best_scale_match": {
            target: value["best_absolute_scale_match"] for target, value in report["scada"]["groups"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
