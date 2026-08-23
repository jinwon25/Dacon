"""Re-audit frozen v31-v57 probability shifts above the real eta=0.15 parent.

The v29-v57 research line labelled Public 1157.9736 as an eta=0.10 parent.
The merged 1158.0746 release actually contains ``blend_eta=0.15``.  This audit
does not retune any candidate on 2024.  It preserves each saved candidate's
already-frozen probability shift and asks whether that shift still helps when
added to the eta=0.15 R_ANCHOR parent.

The final 0819 TrackMan gate is intentionally excluded because its historical
year-specific OOF profiles were not committed.  A candidate must first pass
this conservative parent audit before it can be packaged above the full 1158
release and tested by DACON's row-independence gate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.v30_diverse_covariance_screen import (
    V25_ETA,
    V27_ETA,
    _cached_v25_axes,
    diagnostics,
    v27_parent,
)


CURRENT_ETA = 0.15
SAVED_PARENT_ATOL = 1e-8
AUDIT_FILES = {
    "outer_full_2024": "outer_full_2024.npz",
    "replication_late_2024": "replication_late_2024.npz",
}


def eta_parent(frame: pd.DataFrame, eta: float) -> np.ndarray:
    """Recover a same-signal R_ANCHOR parent from v22/v25 OOF columns."""

    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    v22 = frame["v22"].to_numpy(np.float64)
    v25 = frame["v25"].to_numpy(np.float64)
    return np.clip(v22 + (float(eta) / V25_ETA) * (v25 - v22), 0.001, 0.999)


def rebase_frozen_shift(
    saved_parent: np.ndarray,
    saved_candidate: np.ndarray,
    current_parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move a frozen probability-space shift to a new parent without tuning."""

    saved_parent = np.asarray(saved_parent, dtype=np.float64)
    saved_candidate = np.asarray(saved_candidate, dtype=np.float64)
    current_parent = np.asarray(current_parent, dtype=np.float64)
    if not (
        saved_parent.shape == saved_candidate.shape == current_parent.shape
        and saved_parent.ndim == 1
    ):
        raise ValueError("rebase arrays must be aligned one-dimensional vectors")
    shift = saved_candidate - saved_parent
    return np.clip(current_parent + shift, 0.001, 0.999), shift


def _version(path: Path) -> int:
    match = re.match(r"v(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def discover_candidate_dirs(project: Path) -> list[Path]:
    output = []
    for path in (project / "artifacts").glob("v*_20260817_01"):
        version = _version(path)
        if version < 31 or version > 57:
            continue
        if (path / "outer_full_2024.npz").exists():
            output.append(path)
    return sorted(output, key=lambda path: (_version(path), path.name))


def _gain(
    frame: pd.DataFrame,
    parent: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
) -> dict[str, object]:
    return diagnostics(frame, parent, candidate, np.asarray(active, dtype=bool))


def _unconstrained_scale(
    target: np.ndarray,
    parent: np.ndarray,
    shift: np.ndarray,
) -> float:
    denominator = float(np.mean(np.square(shift)))
    if denominator <= 0.0:
        return 0.0
    return float(np.mean((target - parent) * shift) / denominator)


def audit_saved_axis(
    frame: pd.DataFrame,
    archive_path: Path,
    *,
    current_eta: float = CURRENT_ETA,
) -> dict[str, object]:
    with np.load(archive_path, allow_pickle=True) as saved:
        required = {"target", "v27", "candidate"}
        missing = sorted(required - set(saved.files))
        if missing:
            raise ValueError(f"{archive_path} is missing {missing}")
        target = saved["target"].astype(np.float64)
        old_parent = saved["v27"].astype(np.float64)
        old_candidate = saved["candidate"].astype(np.float64)
        # v31 and v33 predate the explicit active-mask artifact field.  Their
        # frozen summaries both route to ALL, so all rows are the exact legacy
        # semantics.  Newer schemas retain and use their saved route mask.
        if "active" in saved.files:
            active = saved["active"].astype(bool)
            active_source = "saved"
        else:
            active = np.ones(len(target), dtype=bool)
            active_source = "legacy_ALL_route"

    expected_target = frame["target"].to_numpy(np.float64)
    expected_old_parent = v27_parent(frame)
    if not np.array_equal(target, expected_target):
        raise ValueError(f"target/order mismatch: {archive_path}")
    parent_max_abs_diff = float(np.max(np.abs(old_parent - expected_old_parent)))
    if parent_max_abs_diff > SAVED_PARENT_ATOL:
        raise ValueError(
            f"saved eta={V27_ETA} parent mismatch in {archive_path}: "
            f"{parent_max_abs_diff:.17g}"
        )

    current_parent = eta_parent(frame, current_eta)
    rebased, shift = rebase_frozen_shift(old_parent, old_candidate, current_parent)
    old_result = _gain(frame, old_parent, old_candidate, active)
    current_result = _gain(frame, current_parent, rebased, active)
    anchor = frame["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
    parent_change = current_parent - old_parent
    return {
        "rows": len(frame),
        "archive": str(archive_path),
        "saved_parent_eta": V27_ETA,
        "current_parent_eta": float(current_eta),
        "saved_parent_max_abs_diff": parent_max_abs_diff,
        "active_mask_source": active_source,
        "old_gain": float(old_result["gain"]),
        "rebased_gain": float(current_result["gain"]),
        "positive_month_fraction": float(current_result["positive_month_fraction"]),
        "worst_month_gain": float(current_result["worst_month_gain"]),
        "minimum_domain_gain": float(current_result["minimum_domain_gain"]),
        "domain_gains": current_result["domain_gains"],
        "mean_abs_frozen_shift": float(np.mean(np.abs(shift))),
        "mean_abs_parent_change": float(np.mean(np.abs(parent_change))),
        "mean_abs_parent_change_r_anchor": float(
            np.mean(np.abs(parent_change[anchor])) if np.any(anchor) else 0.0
        ),
        "audit_only_unconstrained_shift_scale": _unconstrained_scale(
            target, current_parent, shift
        ),
        "months": current_result["months"],
    }


def _candidate_gate(axes: dict[str, dict[str, object]]) -> dict[str, bool]:
    outer = axes["outer_full_2024"]
    replication = axes.get("replication_late_2024")
    return {
        "outer_gain_at_least_5": bool(float(outer["rebased_gain"]) >= 5.0),
        "outer_month_fraction_at_least_075": bool(
            float(outer["positive_month_fraction"]) >= 0.75
        ),
        "outer_worst_month_above_minus_10": bool(
            float(outer["worst_month_gain"]) > -10.0
        ),
        "outer_minimum_domain_nonnegative": bool(
            float(outer["minimum_domain_gain"]) >= 0.0
        ),
        "replication_available": replication is not None,
        "replication_gain_positive": bool(
            replication is not None and float(replication["rebased_gain"]) > 0.0
        ),
        "replication_month_fraction_at_least_two_thirds": bool(
            replication is not None
            and float(replication["positive_month_fraction"]) >= 2.0 / 3.0
        ),
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    del raw

    parent_audit = {}
    for axis_name, frame in axes.items():
        old_parent = v27_parent(frame)
        current_parent = eta_parent(frame, CURRENT_ETA)
        active = frame["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
        parent_audit[axis_name] = _gain(frame, old_parent, current_parent, active)

    candidates = []
    metric_rows = []
    for candidate_dir in discover_candidate_dirs(project):
        summary_path = candidate_dir / "summary.json"
        source_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {}
        )
        axis_results = {}
        for axis_name, filename in AUDIT_FILES.items():
            archive = candidate_dir / filename
            if archive.exists():
                axis_results[axis_name] = audit_saved_axis(
                    axes[axis_name], archive
                )
        gates = _candidate_gate(axis_results)
        eligible = bool(all(gates.values()))
        item = {
            "candidate": candidate_dir.name,
            "version": _version(candidate_dir),
            "source_protocol": source_summary.get("protocol"),
            "source_parent": source_summary.get("parent"),
            "source_chosen": source_summary.get("chosen"),
            "axes": axis_results,
            "gates": gates,
            "eligible_for_packaging": eligible,
        }
        candidates.append(item)
        outer = axis_results["outer_full_2024"]
        replication = axis_results.get("replication_late_2024", {})
        metric_rows.append(
            {
                "candidate": candidate_dir.name,
                "old_outer_gain": outer["old_gain"],
                "rebased_outer_gain": outer["rebased_gain"],
                "outer_positive_month_fraction": outer["positive_month_fraction"],
                "outer_worst_month_gain": outer["worst_month_gain"],
                "outer_minimum_domain_gain": outer["minimum_domain_gain"],
                "rebased_replication_gain": replication.get("rebased_gain", np.nan),
                "replication_positive_month_fraction": replication.get(
                    "positive_month_fraction", np.nan
                ),
                "audit_only_unconstrained_scale": outer[
                    "audit_only_unconstrained_shift_scale"
                ],
                "eligible_for_packaging": eligible,
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["eligible_for_packaging", "rebased_outer_gain"],
        ascending=False,
        kind="stable",
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "protocol": "V58_ETA15_FROZEN_SHIFT_REBASE_AUDIT_V1",
        "current_release_public_score": 1158.0745556751,
        "local_parent": "eta=0.15 R_ANCHOR parent without final 0819 TrackMan gate",
        "selection_policy": (
            "no retuning; preserve each previously frozen probability shift exactly"
        ),
        "final_trackman_gate_oof_available": False,
        "parent_audit_eta010_to_eta015": parent_audit,
        "candidate_count": len(candidates),
        "eligible_candidate_count": int(
            sum(item["eligible_for_packaging"] for item in candidates)
        ),
        "candidates": candidates,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(
        metrics.to_string(index=False),
        flush=True,
    )
    print(
        json.dumps(
            {
                "candidate_count": summary["candidate_count"],
                "eligible_candidate_count": summary["eligible_candidate_count"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v58_eta15_rebase_audit_20260822_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
