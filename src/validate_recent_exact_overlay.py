"""Validate the v13 recent exact-ASOF overlay archive before submission."""

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
from src.validate_candidate import EVALUATION_ROWS, _extract, _run_package


EXPECTED_ADDITIONS = {
    "model/recent_exact_lgb.txt",
    "model/recent_exact_preprocess.joblib",
    "model/recent_exact_ridge.joblib",
    "model/recent_exact_spec.json",
    "model/recent_stable_ridge.joblib",
}
ALLOWED_PARENT_CHANGES = {"script.py", "model/hybrid.json"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _archive_payload(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {item.filename: archive.read(item) for item in archive.infolist()}


def _domain(frame: pd.DataFrame, anchor_team_id: int) -> pd.Series:
    game_type = frame["game_type"].astype("string").fillna("__MISSING__")
    anchor = frame["pitcher_team_id"].eq(anchor_team_id) | frame[
        "batter_team_id"
    ].eq(anchor_team_id)
    result = pd.Series("OTHER", index=frame.index, dtype="string")
    result.loc[game_type.eq("R") & ~anchor] = "R_CORE"
    result.loc[game_type.eq("R") & anchor] = "R_ANCHOR"
    result.loc[game_type.eq("F")] = "F"
    return result


def _probe(frame: pd.DataFrame, anchor_team_id: int, rows_per_domain: int = 64) -> pd.DataFrame:
    tagged = frame.copy()
    tagged["__domain"] = _domain(tagged, anchor_team_id)
    pieces = []
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        piece = tagged.loc[tagged["__domain"].eq(domain)].head(rows_per_domain)
        if len(piece) != rows_per_domain:
            raise ValueError(f"probe has only {len(piece)} rows for {domain}")
        pieces.append(piece.drop(columns="__domain"))
    return pd.concat(pieces, ignore_index=True)


def run(project: Path, candidate_zip: Path, incumbent_zip: Path, report_path: Path) -> dict[str, object]:
    verify_package(candidate_zip)
    verify_package(incumbent_zip)

    candidate_payload = _archive_payload(candidate_zip)
    incumbent_payload = _archive_payload(incumbent_zip)
    added = set(candidate_payload) - set(incumbent_payload)
    removed = set(incumbent_payload) - set(candidate_payload)
    changed = {
        name
        for name in set(candidate_payload) & set(incumbent_payload)
        if _sha256_bytes(candidate_payload[name]) != _sha256_bytes(incumbent_payload[name])
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
    sample = pd.read_csv(project / "data" / "test.csv", encoding="utf-8-sig")

    with tempfile.TemporaryDirectory(prefix="aimers9_recent_exact_verify_") as name:
        temp = Path(name)
        candidate = _extract(candidate_zip, temp)
        incumbent = _extract(incumbent_zip, temp)
        spec = json.loads(
            (candidate / "model" / "recent_exact_spec.json").read_text(encoding="utf-8")
        )
        anchor_team_id = int(spec["anchor_team_id"])
        probe = _probe(benchmark, anchor_team_id)
        probe_domain = _domain(probe, anchor_team_id).to_numpy()

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

        split_probe = pd.concat([first_probe, second_probe], ignore_index=True).set_index(
            "row_id"
        )
        expected = full_probe.set_index("row_id").loc[split_probe.index]
        batch_max_abs_difference = float(
            np.max(
                np.abs(
                    split_probe["control_success"].to_numpy()
                    - expected["control_success"].to_numpy()
                ),
                initial=0.0,
            )
        )
        batch_invariant = batch_max_abs_difference <= 1e-12

        comparison = full_probe.merge(
            incumbent_probe,
            on="row_id",
            suffixes=("_candidate", "_incumbent"),
            validate="one_to_one",
        ).merge(probe[["row_id"]], on="row_id", validate="one_to_one")
        difference = (
            comparison["control_success_candidate"].to_numpy()
            - comparison["control_success_incumbent"].to_numpy()
        )
        domain_comparison: dict[str, dict[str, float | bool | int]] = {}
        for domain in ("R_CORE", "R_ANCHOR", "F"):
            values = difference[probe_domain == domain]
            domain_comparison[domain] = {
                "rows": int(len(values)),
                "mean_difference": float(values.mean()),
                "mean_abs_difference": float(np.abs(values).mean()),
                "max_abs_difference": float(np.abs(values).max(initial=0.0)),
            }

        anchor_unchanged = domain_comparison["R_ANCHOR"]["max_abs_difference"] <= 1e-12
        core_changed = domain_comparison["R_CORE"]["mean_abs_difference"] > 1e-8
        finals_changed = domain_comparison["F"]["mean_abs_difference"] > 1e-8

        script_text = (candidate / "script.py").read_text(encoding="utf-8").lower()
        network_tokens = ["requests.", "urllib.request", "http://", "https://", "socket."]
        aggregate_tokens = ["groupby(", "value_counts(", ".rank(", ".rolling(", ".expanding("]
        network_free = not any(token in script_text for token in network_tokens)
        row_local_static = not any(token in script_text for token in aggregate_tokens)

    gates = {
        "archive_lineage_ok": archive_lineage_ok,
        "sample_rows_ok": len(sample_submission) == len(sample),
        "benchmark_rows_ok": len(benchmark_submission) == EVALUATION_ROWS,
        "runtime_under_120s": benchmark_seconds < 120.0,
        "batch_invariant": batch_invariant,
        "anchor_unchanged": anchor_unchanged,
        "core_changed": core_changed,
        "finals_changed": finals_changed,
        "network_free": network_free,
        "row_local_static": row_local_static,
    }
    result: dict[str, object] = {
        "candidate": candidate_zip.name,
        "incumbent": incumbent_zip.name,
        "candidate_size_mb": candidate_zip.stat().st_size / (1024.0**2),
        "candidate_uncompressed_mb": sum(len(value) for value in candidate_payload.values())
        / (1024.0**2),
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
    report = f"""# v13 recent exact-ASOF submission validation

- Candidate / parent: `{candidate_zip.name}` / `{incumbent_zip.name}`
- ZIP size: **{result['candidate_size_mb']:.3f} MB** compressed, **{result['candidate_uncompressed_mb']:.3f} MB** extracted
- Exact parent lineage: **{archive_lineage_ok}**
- Official sample smoke: **pass**, {sample_seconds:.3f}s, peak {sample_peak_mb:.1f} MB
- Representative {EVALUATION_ROWS:,}-row archive inference: **pass**, {benchmark_seconds:.3f}s, peak {peak_mb:.1f} MB
- Split-batch invariance: **{batch_invariant}**, max absolute difference {batch_max_abs_difference:.3e}
- R_ANCHOR unchanged from v11: **{anchor_unchanged}**
- R_CORE overlay active: **{core_changed}**
- F overlay active: **{finals_changed}**
- Offline and row-local static audit: **{network_free and row_local_static}**
- Domain comparison: `{domain_comparison}`
- All gates: **{all(gates.values())}**
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise AssertionError(f"v13 validation failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--candidate", type=Path, default=Path("submit_v13.zip"))
    parser.add_argument("--incumbent", type=Path, default=Path("submit_v11.zip"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/v13_recent_exact_validation.md")
    )
    args = parser.parse_args()
    project = args.project.resolve()
    candidate = args.candidate if args.candidate.is_absolute() else project / args.candidate
    incumbent = args.incumbent if args.incumbent.is_absolute() else project / args.incumbent
    report = args.report if args.report.is_absolute() else project / args.report
    run(project, candidate, incumbent, report)


if __name__ == "__main__":
    main()
