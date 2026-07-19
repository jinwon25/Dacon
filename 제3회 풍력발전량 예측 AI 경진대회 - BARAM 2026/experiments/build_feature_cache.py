from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.feature_cache import load_or_build_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="artifacts_feature_cache")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    report = {}
    for split in ("train", "test"):
        frame = load_or_build_features(
            data_dir=args.data_dir,
            split=split,
            cache_dir=args.cache_dir,
            rebuild=args.rebuild,
        )
        report[split] = {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "start": str(frame.index.min()),
            "end": str(frame.index.max()),
            "memory_mb": float(frame.memory_usage(deep=True).sum() / 1024**2),
        }

    output = Path(args.cache_dir) / "cache_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
