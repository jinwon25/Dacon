"""End-to-end lineage, invariance and runtime validation for submit_v19."""

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


ALLOWED_PARENT_CHANGES = {"script.py", "model/hybrid.json"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _payload(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {item.filename: archive.read(item) for item in archive.infolist()}


def _domain(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    ).to_numpy()
    return np.where(~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE"))


def _probe(frame: pd.DataFrame) -> pd.DataFrame:
    domain = _domain(frame)
    parts = [frame.loc[domain == name].tail(128) for name in ("R_CORE", "R_ANCHOR", "F")]
    if any(len(part) != 128 for part in parts):
        raise ValueError("not enough rows for joint domain probe")
    return pd.concat(parts, ignore_index=True)


def run(
    project: Path,
    candidate_zip: Path,
    parent_zip: Path,
    artifact_manifest_path: Path,
    report_path: Path,
) -> dict[str, object]:
    verify_package(candidate_zip)
    verify_package(parent_zip)
    candidate_payload = _payload(candidate_zip)
    parent_payload = _payload(parent_zip)
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    expected_additions = {
        f"model/{name}" for name in artifact_manifest["artifacts"]
    }
    added = set(candidate_payload) - set(parent_payload)
    removed = set(parent_payload) - set(candidate_payload)
    changed = {
        name
        for name in set(candidate_payload) & set(parent_payload)
        if _sha256_bytes(candidate_payload[name]) != _sha256_bytes(parent_payload[name])
    }
    lineage_ok = (
        added == expected_additions
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
    probe = _probe(full_train)
    domains = _domain(probe)
    with tempfile.TemporaryDirectory(prefix="aimers9_joint_verify_") as name:
        temp = Path(name)
        candidate = _extract(candidate_zip, temp)
        parent = _extract(parent_zip, temp)
        sample_submission, sample_seconds, sample_peak_mb, sample_stdout = _run_package(
            candidate, sample
        )
        benchmark_submission, benchmark_seconds, benchmark_peak_mb, benchmark_stdout = _run_package(
            candidate, benchmark
        )
        full_probe, _, _, _ = _run_package(candidate, probe)
        odd_probe, _, _, _ = _run_package(candidate, probe.iloc[::2].copy())
        even_probe, _, _, _ = _run_package(candidate, probe.iloc[1::2].copy())
        parent_probe, _, _, _ = _run_package(parent, probe)

        split = pd.concat([odd_probe, even_probe], ignore_index=True).set_index("row_id")
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
            parent_probe,
            on="row_id",
            suffixes=("_candidate", "_parent"),
            validate="one_to_one",
        )
        difference = (
            comparison["control_success_candidate"].to_numpy()
            - comparison["control_success_parent"].to_numpy()
        )
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
        spec = json.loads(
            (candidate / "model" / "joint_state_mode_spec.json").read_text(
                encoding="utf-8"
            )
        )

    probabilities = benchmark_submission["control_success"].to_numpy(np.float64)
    gates = {
        "archive_lineage_ok": lineage_ok,
        "sample_rows_ok": len(sample_submission) == len(sample),
        "benchmark_rows_ok": len(benchmark_submission) == EVALUATION_ROWS,
        "probabilities_finite_and_bounded": bool(
            np.isfinite(probabilities).all()
            and (probabilities >= 0.0).all()
            and (probabilities <= 1.0).all()
        ),
        "runtime_under_120s": benchmark_seconds < 120.0,
        "peak_memory_under_4gb": benchmark_peak_mb < 4096.0,
        "batch_invariant": batch_max_abs_difference <= 1e-12,
        "all_domains_changed": all(
            value["mean_abs_difference"] > 1e-8
            for value in domain_comparison.values()
        ),
        "network_free": not any(
            token in script_text
            for token in ("requests.", "urllib.request", "http://", "https://", "socket.")
        ),
        "row_local_declared": bool(spec["row_local_inference"]),
        "test_aggregate_unused": not bool(spec["test_aggregate_used"]),
    }
    result = {
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
        "benchmark_peak_mb": benchmark_peak_mb,
        "benchmark_stdout": benchmark_stdout,
        "benchmark_probability_min": float(probabilities.min()),
        "benchmark_probability_max": float(probabilities.max()),
        "batch_max_abs_difference": batch_max_abs_difference,
        "domain_comparison": domain_comparison,
        "all_gates": bool(all(gates.values())),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        "# submit_v19 joint state/mode validation\n\n"
        f"- Candidate / parent: `{candidate_zip.name}` / `{parent_zip.name}`\n"
        f"- Representative runtime: **{benchmark_seconds:.3f}s**, peak **{benchmark_peak_mb:.1f}MB**\n"
        f"- Batch max absolute difference: **{batch_max_abs_difference:.3e}**\n"
        f"- Domain comparison: `{domain_comparison}`\n"
        f"- All gates: **{all(gates.values())}**\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise AssertionError(f"joint package validation failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--candidate", type=Path, default=Path("submit_v19.zip"))
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("submissions/history/submit_v17.zip"),
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=Path("artifacts/state_mode_joint_final_20260816/manifest.json"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/v19_validation.md")
    )
    args = parser.parse_args()
    project = args.project.resolve()
    candidate = args.candidate if args.candidate.is_absolute() else project / args.candidate
    parent = args.parent if args.parent.is_absolute() else project / args.parent
    manifest = (
        args.artifact_manifest
        if args.artifact_manifest.is_absolute()
        else project / args.artifact_manifest
    )
    report = args.report if args.report.is_absolute() else project / args.report
    run(project, candidate, parent, manifest, report)


if __name__ == "__main__":
    main()
