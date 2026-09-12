"""Verify v296 preserves v290 exactly outside game_type=R."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.packaging import run_package


def run(parent_zip: Path, candidate_zip: Path, test_csv: Path) -> dict:
    source = pd.read_csv(test_csv, encoding="utf-8-sig")
    frame = pd.concat([source.iloc[[0]]] * 6, ignore_index=True)
    frame["row_id"] = [f"v296_protected_{index}" for index in range(len(frame))]
    frame["game_type"] = ["R", "F", "X", "R", "F", "X"]
    sample = pd.DataFrame({"row_id": frame["row_id"], "control_success": 0.0})
    parent, _, _ = run_package(parent_zip, frame, sample, timeout=600)
    candidate, _, _ = run_package(candidate_zip, frame, sample, timeout=600)
    p = frame["row_id"].map(parent.set_index("row_id")["control_success"]).to_numpy(float)
    q = frame["row_id"].map(candidate.set_index("row_id")["control_success"]).to_numpy(float)
    regular = frame["game_type"].eq("R").to_numpy()
    protected = ~regular
    protected_error = float(np.max(np.abs(q[protected] - p[protected])))
    tolerance = 1e-15
    result = {
        "protocol": "AUDIT_V296_PROTECTED_ROUTES_V1",
        "rows": len(frame),
        "protected_rows": int(protected.sum()),
        "protected_max_abs_diff": protected_error,
        "protected_tolerance": tolerance,
        "protected_by_game_type": {
            game_type: float(
                np.max(
                    np.abs(
                        q[frame["game_type"].eq(game_type).to_numpy()]
                        - p[frame["game_type"].eq(game_type).to_numpy()]
                    )
                )
            )
            for game_type in ("F", "X")
        },
        "regular_changed_rows": int(np.sum(np.abs(q[regular] - p[regular]) > 0.0)),
        "regular_rows": int(regular.sum()),
        "status": (
            "pass"
            if protected_error <= tolerance
            and np.all(np.abs(q[regular] - p[regular]) > 0.0)
            else "fail"
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-package", type=Path, required=True)
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.parent_package, args.candidate_package, args.test_csv)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
