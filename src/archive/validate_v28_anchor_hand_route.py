"""End-to-end lineage, formula, invariance, and runtime gates for submit_v28."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.package import verify_package
from src.archive.train_v20_target1160 import _read_season
from src.core.axes import _joint_domain
from src.core.axes import _derived
from src.archive.validate_candidate import EVALUATION_ROWS, _extract, _run_package


EXPECTED_CHANGES = {
    "script.py",
    "model/hybrid.json",
    "model/v25_postbreak_anchor_spec.json",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _payload(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {item.filename: archive.read(item) for item in archive.infolist()}


def _expected_from_parent(
    parent: np.ndarray,
    direct: np.ndarray,
    matched: np.ndarray,
    default_eta: float,
    match_eta: float,
) -> np.ndarray:
    result = np.asarray(parent, dtype=np.float64).copy()
    base = (result[matched] - default_eta * direct[matched]) / (1.0 - default_eta)
    result[matched] = base + match_eta * (direct[matched] - base)
    return np.clip(result, 0.001, 0.999)


def _probe(frame: pd.DataFrame) -> pd.DataFrame:
    domain = _joint_domain(frame)
    parts = []
    for name in ("R_CORE", "F"):
        parts.append(frame.loc[domain == name].tail(96))
    anchor = frame.loc[domain == "R_ANCHOR"]
    for hand in ("1", "2"):
        parts.append(anchor.loc[anchor["pitcher_hand"].astype(str).eq(hand)].tail(96))
    if any(len(part) != 96 for part in parts):
        raise ValueError("not enough rows for v28 route probe")
    return pd.concat(parts, ignore_index=True)


def run(project: Path, candidate_zip: Path, parent_zip: Path, report: Path) -> dict[str, object]:
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

    sample = pd.read_csv(project / "data" / "test.csv", encoding="utf-8-sig")
    benchmark = pd.read_csv(
        project / "data" / "train.csv",
        nrows=EVALUATION_ROWS,
        encoding="utf-8-sig",
        low_memory=False,
    ).drop(columns="control_success")
    season24, _ = _read_season(project / "data" / "train.csv", 2024)
    probe = _probe(season24.drop(columns=["control_success", "_global_index"]))

    with tempfile.TemporaryDirectory(prefix="aimers9_v28_verify_") as name:
        temp = Path(name)
        candidate = _extract(candidate_zip, temp)
        parent = _extract(parent_zip, temp)
        spec = json.loads(
            (candidate / "model" / "v25_postbreak_anchor_spec.json").read_text(
                encoding="utf-8"
            )
        )
        route = spec["blend_eta_route"]
        model = joblib.load(candidate / "model" / str(spec["model_file"]))
        sample_output, sample_seconds, sample_peak_mb, sample_stdout = _run_package(
            candidate, sample
        )
        candidate_output, benchmark_seconds, benchmark_peak_mb, benchmark_stdout = (
            _run_package(candidate, benchmark)
        )
        parent_output, _, _, _ = _run_package(parent, benchmark)
        full_probe, _, _, _ = _run_package(candidate, probe)
        odd_probe, _, _, _ = _run_package(candidate, probe.iloc[::2].copy())
        even_probe, _, _, _ = _run_package(candidate, probe.iloc[1::2].copy())

    split = pd.concat([odd_probe, even_probe], ignore_index=True).set_index("row_id")
    expected_split = full_probe.set_index("row_id").loc[split.index]
    batch_difference = float(
        np.max(
            np.abs(
                split["control_success"].to_numpy()
                - expected_split["control_success"].to_numpy()
            ),
            initial=0.0,
        )
    )
    parent_values = parent_output["control_success"].to_numpy(np.float64)
    candidate_values = candidate_output["control_success"].to_numpy(np.float64)
    domain = _joint_domain(benchmark)
    anchor = domain == str(spec["apply_domain"])
    matched = anchor & benchmark[str(route["column"])].astype(str).eq(
        str(route["match_value"])
    ).to_numpy()
    direct = np.zeros(len(benchmark), dtype=np.float64)
    features = _derived(benchmark.loc[anchor].copy(), domain[anchor])
    low, high = (float(x) for x in spec["probability_clip"])
    direct[anchor] = np.clip(
        np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64), low, high
    )
    expected_values = _expected_from_parent(
        parent_values,
        direct,
        matched,
        float(route["default_eta"]),
        float(route["match_eta"]),
    )
    formula_difference = float(
        np.max(np.abs(candidate_values - expected_values), initial=0.0)
    )
    shift = candidate_values - parent_values
    protected = ~matched
    gates = {
        "archive_lineage_ok": not added and not removed and changed == EXPECTED_CHANGES,
        "sample_rows_ok": len(sample_output) == len(sample),
        "benchmark_rows_ok": len(candidate_output) == EVALUATION_ROWS,
        "probabilities_finite_and_bounded": bool(
            np.isfinite(candidate_values).all()
            and (candidate_values >= 0.0).all()
            and (candidate_values <= 1.0).all()
        ),
        "runtime_under_120s": benchmark_seconds < 120.0,
        "peak_memory_under_4gb": benchmark_peak_mb < 4096.0,
        "batch_invariant": batch_difference <= 1e-12,
        "formula_exact": formula_difference <= 1e-12,
        "protected_rows_exactly_unchanged": bool(
            np.max(np.abs(shift[protected]), initial=0.0) <= 1e-12
        ),
        "matched_rows_changed": bool(np.mean(np.abs(shift[matched])) > 1e-8),
        "row_local_declared": bool(spec["row_local_inference"]),
        "test_aggregate_unused": not bool(spec["test_aggregate_used"]),
        "external_data_unused": not bool(spec["external_data_used"]),
    }
    result = {
        "candidate": candidate_zip.name,
        "parent": parent_zip.name,
        "added_members": sorted(added),
        "removed_members": sorted(removed),
        "changed_parent_members": sorted(changed),
        "route": route,
        "matched_benchmark_rows": int(matched.sum()),
        "benchmark_seconds": benchmark_seconds,
        "benchmark_peak_mb": benchmark_peak_mb,
        "benchmark_stdout": benchmark_stdout,
        "sample_seconds": sample_seconds,
        "sample_peak_mb": sample_peak_mb,
        "sample_stdout": sample_stdout,
        "batch_max_abs_difference": batch_difference,
        "formula_max_abs_difference": formula_difference,
        "matched_mean_abs_shift": float(np.mean(np.abs(shift[matched]))),
        "protected_max_abs_shift": float(np.max(np.abs(shift[protected]), initial=0.0)),
        "gates": gates,
        "all_gates": bool(all(gates.values())),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report.write_text(
        "# submit_v28 validation\n\n"
        f"- Candidate / parent: `{candidate_zip.name}` / `{parent_zip.name}`\n"
        f"- Runtime: **{benchmark_seconds:.3f}s**, peak **{benchmark_peak_mb:.1f}MB**\n"
        f"- Formula / batch max difference: **{formula_difference:.3e} / {batch_difference:.3e}**\n"
        f"- Matched rows / mean absolute shift: **{int(matched.sum())} / {np.mean(np.abs(shift[matched])):.6f}**\n"
        f"- Protected max absolute shift: **{np.max(np.abs(shift[protected]), initial=0.0):.3e}**\n"
        f"- All gates: **{all(gates.values())}**\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not all(gates.values()):
        failed = [key for key, value in gates.items() if not value]
        raise AssertionError(f"v28 validation failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--candidate", type=Path, default=Path("submit_v28.zip"))
    parser.add_argument("--parent", type=Path, default=Path("submit_v27.zip"))
    parser.add_argument("--report", type=Path, default=Path("reports/v28_validation.md"))
    args = parser.parse_args()
    project = args.project.resolve()
    candidate = args.candidate if args.candidate.is_absolute() else project / args.candidate
    parent = args.parent if args.parent.is_absolute() else project / args.parent
    report = args.report if args.report.is_absolute() else project / args.report
    run(project, candidate, parent, report)


if __name__ == "__main__":
    main()
