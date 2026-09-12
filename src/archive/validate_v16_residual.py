"""End-to-end, batch-invariance and lineage validation for submit_v16.zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.package import verify_package
from src.archive.validate_candidate import EVALUATION_ROWS, _extract, _run_package


EXPECTED_ADDITIONS = {"model/v16_residual_spec.json"}
ALLOWED_PARENT_CHANGES = {"script.py", "model/hybrid.json"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _payload(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {item.filename: archive.read(item) for item in archive.infolist()}


def _domain(frame: pd.DataFrame, anchor_team: int) -> np.ndarray:
    regular = frame["game_type"].eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(anchor_team)
        | frame["batter_team_id"].eq(anchor_team)
    ).to_numpy()
    return np.where(~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE"))


def _key(frame: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    pieces = [
        frame["pitcher_id"].astype("string").fillna("__MISSING__"),
        frame["batter_hand"].astype("string").fillna("__MISSING__"),
    ]
    if "pressure" in group_columns:
        balls = pd.to_numeric(frame["balls_before"], errors="coerce").to_numpy()
        strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").to_numpy()
        pieces.append(
            pd.Series(
                np.where(
                    balls == 3,
                    "threeball",
                    np.where(strikes == 2, "twostrike", "normal"),
                ),
                index=frame.index,
                dtype="string",
            )
        )
    key = pieces[0]
    for piece in pieces[1:]:
        key = key + "\x1f" + piece
    return key


def _probe(frame: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    domain = _domain(frame, int(spec["anchor_team_id"]))
    effect_keys = set(spec["effects"])
    affected = frame.loc[
        (domain == "R_CORE")
        & _key(frame, list(spec["group_columns"])).isin(effect_keys)
    ].tail(96)
    anchor = frame.loc[domain == "R_ANCHOR"].tail(64)
    finals = frame.loc[domain == "F"].tail(64)
    if len(affected) != 96 or len(anchor) != 64 or len(finals) != 64:
        raise ValueError("not enough rows for v16 domain probe")
    return pd.concat([affected, anchor, finals], ignore_index=True)


def run(project: Path, candidate_zip: Path, parent_zip: Path, report_path: Path) -> dict[str, object]:
    verify_package(candidate_zip)
    verify_package(parent_zip)
    candidate_payload = _payload(candidate_zip)
    parent_payload = _payload(parent_zip)
    added = set(candidate_payload) - set(parent_payload)
    removed = set(parent_payload) - set(candidate_payload)
    changed = {
        name
        for name in set(candidate_payload) & set(parent_payload)
        if _sha256_bytes(candidate_payload[name]) != _sha256_bytes(parent_payload[name])
    }
    archive_lineage_ok = added == EXPECTED_ADDITIONS and not removed and changed == ALLOWED_PARENT_CHANGES
    benchmark = pd.read_csv(
        project / "data" / "train.csv",
        nrows=EVALUATION_ROWS,
        encoding="utf-8-sig",
        low_memory=False,
    ).drop(columns="control_success")
    full_train = pd.read_csv(
        project / "data" / "train.csv", encoding="utf-8-sig", low_memory=False
    ).drop(columns="control_success")
    sample = pd.read_csv(project / "data" / "test.csv", encoding="utf-8-sig")
    with tempfile.TemporaryDirectory(prefix="aimers9_v16_verify_") as name:
        temp = Path(name)
        candidate = _extract(candidate_zip, temp)
        parent = _extract(parent_zip, temp)
        spec = json.loads(
            (candidate / "model" / "v16_residual_spec.json").read_text(encoding="utf-8")
        )
        probe = _probe(full_train, spec)
        domains = _domain(probe, int(spec["anchor_team_id"]))
        sample_submission, sample_seconds, sample_peak_mb, sample_stdout = _run_package(candidate, sample)
        benchmark_submission, benchmark_seconds, peak_mb, benchmark_stdout = _run_package(candidate, benchmark)
        full_probe, _, _, _ = _run_package(candidate, probe)
        odd_probe, _, _, _ = _run_package(candidate, probe.iloc[::2].copy())
        even_probe, _, _, _ = _run_package(candidate, probe.iloc[1::2].copy())
        parent_probe, _, _, _ = _run_package(parent, probe)
        split = pd.concat([odd_probe, even_probe], ignore_index=True).set_index("row_id")
        expected = full_probe.set_index("row_id").loc[split.index]
        batch_max_abs_difference = float(
            np.max(np.abs(split["control_success"].to_numpy() - expected["control_success"].to_numpy()), initial=0.0)
        )
        comparison = full_probe.merge(parent_probe, on="row_id", suffixes=("_candidate", "_parent"), validate="one_to_one")
        difference = comparison["control_success_candidate"].to_numpy() - comparison["control_success_parent"].to_numpy()
        domain_comparison = {}
        for domain_name in ("R_CORE", "R_ANCHOR", "F"):
            values = difference[domains == domain_name]
            domain_comparison[domain_name] = {
                "rows": int(len(values)),
                "mean_difference": float(values.mean()),
                "mean_abs_difference": float(np.abs(values).mean()),
                "max_abs_difference": float(np.abs(values).max(initial=0.0)),
            }
        script_text = (candidate / "script.py").read_text(encoding="utf-8").lower()
        network_free = not any(token in script_text for token in ("requests.", "urllib.request", "http://", "https://", "socket."))
        row_local_static = not any(token in script_text for token in ("groupby(", "value_counts(", ".rank(", ".rolling(", ".expanding("))
    gates = {
        "archive_lineage_ok": archive_lineage_ok,
        "sample_rows_ok": len(sample_submission) == len(sample),
        "benchmark_rows_ok": len(benchmark_submission) == EVALUATION_ROWS,
        "runtime_under_120s": benchmark_seconds < 120.0,
        "batch_invariant": batch_max_abs_difference <= 1e-12,
        "core_changed": domain_comparison["R_CORE"]["mean_abs_difference"] > 1e-8,
        "anchor_unchanged": domain_comparison["R_ANCHOR"]["max_abs_difference"] <= 1e-12,
        "finals_unchanged": domain_comparison["F"]["max_abs_difference"] <= 1e-12,
        "network_free": network_free,
        "row_local_static": row_local_static,
    }
    result: dict[str, object] = {
        "candidate": candidate_zip.name,
        "parent": parent_zip.name,
        "candidate_size_mb": candidate_zip.stat().st_size / (1024.0**2),
        "added_members": sorted(added),
        "removed_members": sorted(removed),
        "changed_parent_members": sorted(changed),
        "gates": gates,
        "sample_seconds": sample_seconds,
        "sample_peak_mb": sample_peak_mb,
        "sample_stdout": sample_stdout,
        "benchmark_seconds": benchmark_seconds,
        "benchmark_peak_mb": peak_mb,
        "benchmark_stdout": benchmark_stdout,
        "batch_max_abs_difference": batch_max_abs_difference,
        "domain_comparison": domain_comparison,
        "spec_without_effects": {key: value for key, value in spec.items() if key != "effects"},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        "# v16 residual package validation\n\n"
        f"- Candidate / parent: `{candidate_zip.name}` / `{parent_zip.name}`\n"
        f"- Representative runtime: **{benchmark_seconds:.3f}s**, peak **{peak_mb:.1f}MB**\n"
        f"- Batch max absolute difference: **{batch_max_abs_difference:.3e}**\n"
        f"- Domain comparison: `{domain_comparison}`\n"
        f"- All gates: **{all(gates.values())}**\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise AssertionError(f"v16 validation failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--candidate", type=Path, default=Path("submit_v16.zip"))
    parser.add_argument("--parent", type=Path, default=Path("submit_v14.zip"))
    parser.add_argument("--report", type=Path, default=Path("reports/v16_validation.md"))
    args = parser.parse_args()
    project = args.project.resolve()
    candidate = args.candidate if args.candidate.is_absolute() else project / args.candidate
    parent = args.parent if args.parent.is_absolute() else project / args.parent
    report = args.report if args.report.is_absolute() else project / args.report
    run(project, candidate, parent, report)


if __name__ == "__main__":
    main()
