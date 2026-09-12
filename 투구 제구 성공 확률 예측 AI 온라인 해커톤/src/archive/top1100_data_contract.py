"""Rebuild the Top-1100 data contract from the original CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


CHUNKSIZE = 200_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _frame_contract(path: Path, *, kind: str, target: str | None = None) -> dict[str, Any]:
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    columns = header.columns.tolist()
    dtypes = pd.read_csv(path, encoding="utf-8-sig", nrows=100_000, low_memory=False).dtypes.astype(str).to_dict()
    missing = {column: 0 for column in columns}
    rows = 0
    unique_values: dict[str, set[str]] = {column: set() for column in columns if column in {
        "season", "game_type", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
        "pitcher_team_id", "batter_team_id", "pitcher_trackman_id", "pitch_type_group",
    }}
    season_counts: dict[str, dict[str, int]] = {}
    for chunk in pd.read_csv(path, encoding="utf-8-sig", chunksize=CHUNKSIZE, low_memory=False):
        rows += len(chunk)
        for column in columns:
            missing[column] += int(chunk[column].isna().sum())
            if column in unique_values:
                unique_values[column].update(chunk[column].dropna().astype(str).unique().tolist())
        if "season" in chunk.columns:
            grouped = chunk.groupby("season", observed=True).size()
            for season, count in grouped.items():
                key = str(int(season)); season_counts.setdefault(key, {"rows": 0}); season_counts[key]["rows"] += int(count)
            if target and target in chunk.columns:
                grouped_target = chunk.groupby("season", observed=True)[target].agg(["sum", "count"])
                for season, values in grouped_target.iterrows():
                    key = str(int(season)); season_counts.setdefault(key, {"rows": 0}); season_counts[key]["target_sum"] = season_counts[key].get("target_sum", 0) + int(values["sum"]); season_counts[key]["target_count"] = season_counts[key].get("target_count", 0) + int(values["count"])
    cardinality = {column: len(values) for column, values in unique_values.items()}
    for value in season_counts.values():
        if value.get("target_count"):
            value["target_rate"] = value["target_sum"] / value["target_count"]
    return {
        "path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size,
        "rows": rows, "columns": len(columns), "column_names": columns, "dtypes_sample": dtypes,
        "missing_rate": {column: missing[column] / max(rows, 1) for column in columns},
        "cardinality": cardinality, "season_summary": season_counts,
    }


def run(project: Path) -> dict[str, Any]:
    data = project / "data"
    train = _frame_contract(data / "train.csv", kind="main", target="control_success")
    test = _frame_contract(data / "test.csv", kind="test")
    trackman = _frame_contract(data / "trackman_history.csv", kind="trackman")
    sample = _frame_contract(data / "sample_submission.csv", kind="sample")
    schema_checks = {
        "train_test_same_input_columns": set(train["column_names"]) - {"control_success"} == set(test["column_names"]),
        "sample_columns": sample["column_names"],
        "trackman_has_target": "control_success" in trackman["column_names"],
        "row_id_unique_train": True,
        "row_id_unique_test": True,
    }
    # row_id uniqueness is checked in a second low-memory pass.
    for name, path in (("train", data / "train.csv"), ("test", data / "test.csv")):
        seen: set[str] = set(); duplicate = False
        for chunk in pd.read_csv(path, usecols=["row_id"], encoding="utf-8-sig", chunksize=CHUNKSIZE):
            values = chunk["row_id"].astype(str).tolist(); duplicate = duplicate or bool(seen.intersection(values)); seen.update(values)
        schema_checks[f"row_id_unique_{name}"] = not duplicate
    result = {"created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(), "schema_checks": schema_checks, "files": {"train": train, "test": test, "trackman_history": trackman, "sample_submission": sample}}
    out_dir = project / "artifacts/top1100"; report_dir = project / "research/reports/top1100"; out_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_contract.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Top-1100 data contract", "", f"Created: `{result['created_at']}`", "", "## Files"]
    for name, info in result["files"].items():
        lines += [f"- **{name}**: `{info['rows']:,}` rows × `{info['columns']}` columns; `{info['size_bytes']:,}` bytes; SHA-256 `{info['sha256']}`"]
    lines += ["", "## Schema checks", ""] + [f"- `{key}`: **{'PASS' if value is True else value}**" for key, value in schema_checks.items()]
    lines += ["", "## Target rate by season", "", "```text", pd.DataFrame(train["season_summary"]).T.to_string(), "```", "", "All hashes and counts above were rebuilt from the original CSVs. No external outcome data were read."]
    (report_dir / "data_contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"train": (train["rows"], train["columns"], train["sha256"]), "test": (test["rows"], test["columns"], test["sha256"]), "trackman": (trackman["rows"], trackman["columns"], trackman["sha256"])}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); args = parser.parse_args(); run(args.project_dir.resolve())


if __name__ == "__main__": main()
