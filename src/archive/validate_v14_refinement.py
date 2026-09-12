"""End-to-end validation for a v14 refinement package against v13."""

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


EXPECTED_ADDITIONS = {
    "model/v14_anchor_ridge.joblib",
    "model/v14_f_trend_lgb.txt",
    "model/v14_refinement_preprocess.joblib",
    "model/v14_refinement_spec.json",
}
ALLOWED_PARENT_CHANGES = {"script.py", "model/hybrid.json"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _payload(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {item.filename: archive.read(item) for item in archive.infolist()}


def _domains(frame: pd.DataFrame, anchor_team: int) -> np.ndarray:
    regular = frame["game_type"].eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(anchor_team)
        | frame["batter_team_id"].eq(anchor_team)
    ).to_numpy()
    return np.where(~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE"))


def _probe(frame: pd.DataFrame, anchor_team: int) -> pd.DataFrame:
    domain = _domains(frame, anchor_team)
    pieces = []
    for name in ("R_CORE", "R_ANCHOR", "F"):
        piece = frame.loc[domain == name].tail(64).copy()
        if len(piece) != 64:
            raise ValueError(f"not enough {name} probe rows")
        if name == "R_ANCHOR":
            # Emulate a late-2025 official ASOF snapshot without deriving any
            # target from the probe.  This exercises the frozen reliability
            # gate that ordinary historical rows cannot reach against a 2025
            # deployment bank.
            piece["season"] = 2025
            piece["asof_pitcher_n"] = (
                pd.to_numeric(piece["asof_pitcher_n"], errors="coerce").fillna(0)
                + 1_200
            )
            piece["asof_batter_n"] = (
                pd.to_numeric(piece["asof_batter_n"], errors="coerce").fillna(0)
                + 1_400
            )
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def run(
    project: Path,
    candidate_zip: Path,
    incumbent_zip: Path,
    report_path: Path,
) -> dict[str, object]:
    verify_package(candidate_zip)
    verify_package(incumbent_zip)
    candidate_payload = _payload(candidate_zip)
    incumbent_payload = _payload(incumbent_zip)
    added = set(candidate_payload) - set(incumbent_payload)
    removed = set(incumbent_payload) - set(candidate_payload)
    changed = {
        name
        for name in set(candidate_payload) & set(incumbent_payload)
        if _sha256_bytes(candidate_payload[name])
        != _sha256_bytes(incumbent_payload[name])
    }
    archive_lineage_ok = (
        added == EXPECTED_ADDITIONS
        and not removed
        and changed == ALLOWED_PARENT_CHANGES
    )
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

    with tempfile.TemporaryDirectory(prefix="aimers9_v14_verify_") as name:
        temp = Path(name)
        candidate = _extract(candidate_zip, temp)
        incumbent = _extract(incumbent_zip, temp)
        spec = json.loads(
            (candidate / "model" / "v14_refinement_spec.json").read_text(
                encoding="utf-8"
            )
        )
        anchor_team = int(spec["anchor_team_id"])
        probe = _probe(full_train, anchor_team)
        domains = _domains(probe, anchor_team)
        sample_submission, sample_seconds, sample_peak_mb, sample_stdout = _run_package(
            candidate, sample
        )
        benchmark_submission, benchmark_seconds, peak_mb, benchmark_stdout = _run_package(
            candidate, benchmark
        )
        full_probe, _, _, _ = _run_package(candidate, probe)
        first_probe, _, _, _ = _run_package(candidate, probe.iloc[::2].copy())
        second_probe, _, _, _ = _run_package(candidate, probe.iloc[1::2].copy())
        incumbent_probe, _, _, _ = _run_package(incumbent, probe)
        split = pd.concat([first_probe, second_probe], ignore_index=True).set_index(
            "row_id"
        )
        expected = full_probe.set_index("row_id").loc[split.index]
        batch_max_abs_difference = float(
            np.max(
                np.abs(
                    split["control_success"].to_numpy()
                    - expected["control_success"].to_numpy()
                ),
                initial=0.0,
            )
        )
        comparison = full_probe.merge(
            incumbent_probe,
            on="row_id",
            suffixes=("_candidate", "_incumbent"),
            validate="one_to_one",
        )
        difference = (
            comparison["control_success_candidate"].to_numpy()
            - comparison["control_success_incumbent"].to_numpy()
        )
        domain_comparison = {}
        for domain in ("R_CORE", "R_ANCHOR", "F"):
            values = difference[domains == domain]
            domain_comparison[domain] = {
                "rows": int(len(values)),
                "mean_difference": float(values.mean()),
                "mean_abs_difference": float(np.abs(values).mean()),
                "max_abs_difference": float(np.abs(values).max(initial=0.0)),
            }
        script_text = (candidate / "script.py").read_text(encoding="utf-8").lower()
        network_free = not any(
            token in script_text
            for token in ("requests.", "urllib.request", "http://", "https://", "socket.")
        )
        row_local_static = not any(
            token in script_text
            for token in ("groupby(", "value_counts(", ".rank(", ".rolling(", ".expanding(")
        )

    gates = {
        "archive_lineage_ok": archive_lineage_ok,
        "sample_rows_ok": len(sample_submission) == len(sample),
        "benchmark_rows_ok": len(benchmark_submission) == EVALUATION_ROWS,
        "runtime_under_120s": benchmark_seconds < 120.0,
        "batch_invariant": batch_max_abs_difference <= 1e-12,
        "core_unchanged": domain_comparison["R_CORE"]["max_abs_difference"] <= 1e-12,
        "anchor_changed": domain_comparison["R_ANCHOR"]["mean_abs_difference"] > 1e-8,
        "finals_changed": domain_comparison["F"]["mean_abs_difference"] > 1e-8,
        "network_free": network_free,
        "row_local_static": row_local_static,
    }
    result: dict[str, object] = {
        "candidate": candidate_zip.name,
        "incumbent": incumbent_zip.name,
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
        "spec": spec,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        "# v14 refinement package validation\n\n"
        f"- Candidate / parent: `{candidate_zip.name}` / `{incumbent_zip.name}`\n"
        f"- Representative runtime: **{benchmark_seconds:.3f}s**, peak **{peak_mb:.1f}MB**\n"
        f"- Batch max absolute difference: **{batch_max_abs_difference:.3e}**\n"
        f"- Domain comparison: `{domain_comparison}`\n"
        f"- All gates: **{all(gates.values())}**\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise AssertionError(f"v14 validation failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, default=Path("submit_v13_fixed.zip"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    candidate = args.candidate if args.candidate.is_absolute() else project / args.candidate
    incumbent = args.incumbent if args.incumbent.is_absolute() else project / args.incumbent
    report = args.report if args.report.is_absolute() else project / args.report
    run(project, candidate, incumbent, report)


if __name__ == "__main__":
    main()
