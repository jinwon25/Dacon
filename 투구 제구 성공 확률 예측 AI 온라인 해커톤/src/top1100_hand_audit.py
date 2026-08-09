"""Target-free two-way main pitcher-hand mapping audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import read_main, read_trackman


def run(project: Path) -> pd.DataFrame:
    main = read_main(project / "data/train.csv", usecols=["season", "pitcher_hand"])
    tm = read_trackman(project / "data/trackman_history.csv", usecols=["season", "pitcher_hand"])
    rows = []
    for season in sorted(set(main["season"].dropna().unique()) & set(tm["season"].dropna().unique())):
        m = main.loc[main["season"] == season, "pitcher_hand"].value_counts(normalize=True)
        t = tm.loc[tm["season"] == season, "pitcher_hand"].astype("string").str.lower().value_counts(normalize=True)
        # Main labels are anonymous 1/2. Compare both possible orientations.
        left_share, right_share = float(t.get("left", 0.0)), float(t.get("right", 0.0))
        p1, p2 = float(m.get(1, 0.0)), float(m.get(2, 0.0))
        err_1_left = abs(p1 - left_share) + abs(p2 - right_share)
        err_1_right = abs(p1 - right_share) + abs(p2 - left_share)
        rows.extend([
            {"season": int(season), "mapping": "1=Left,2=Right", "l1_share_error": err_1_left},
            {"season": int(season), "mapping": "1=Right,2=Left", "l1_share_error": err_1_right},
        ])
    out = pd.DataFrame(rows)
    out.to_csv(project / "reports/top1100/hand_mapping_audit.csv", index=False)
    summary = out.groupby("mapping", as_index=False)["l1_share_error"].agg(["mean", "median", "max"]).reset_index()
    best = str(summary.loc[summary["mean"].idxmin(), "mapping"])
    report = "# Target-free hand mapping audit\n\n"
    report += "This compares only season-level pitcher-hand shares; no outcome or target-derived statistic is used.\n\n"
    report += "```csv\n" + summary.to_csv(index=False) + "```\n\n"
    report += f"Lower aggregate share error: **{best}**. This is an aggregate consistency result, not player identity accuracy; it does not by itself unlock Trackman linkage.\n"
    (project / "reports/top1100/hand_mapping_audit.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path("."))
    run(parser.parse_args().project_dir.resolve())
