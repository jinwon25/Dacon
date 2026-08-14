from __future__ import annotations

"""Generate explicitly controlled affine probe submissions.

These probes are intentionally outside the all-component automatic promotion
contract.  They retain the exact public-best base, change only the requested
group(s), and write a compact provenance report alongside each CSV.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, evaluate_group
from experiments.blocked_rolling_validation import assign_issue_blocks, load_issue_times


H2_START = pd.Timestamp("2024-07-01 01:00:00")
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _affine(values: np.ndarray, group: str, scale: float, offset: float) -> np.ndarray:
    return np.clip(
        np.asarray(values, dtype=float) * float(scale) + float(offset),
        0.0,
        CAPACITY_KWH[group],
    )


def _delta(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    group: str,
    mask: np.ndarray,
) -> dict[str, float]:
    before = evaluate_group(truth[mask], base[mask], CAPACITY_KWH[group])
    after = evaluate_group(truth[mask], candidate[mask], CAPACITY_KWH[group])
    return {
        "score": float(after.score - before.score),
        "one_minus_nmae": float(after.one_minus_nmae - before.one_minus_nmae),
        "ficr": float(after.ficr - before.ficr),
    }


def _periods(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "q1": np.asarray(index < Q2_START),
        "q2": np.asarray((index >= Q2_START) & (index < H2_START)),
        "h2": np.asarray(index >= H2_START),
        "full": np.ones(len(index), dtype=bool),
    }


def _bootstrap_macro(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    index: pd.DatetimeIndex,
    mask: np.ndarray,
    n_bootstrap: int,
) -> dict[str, float]:
    """Resample complete days and compute macro score deltas.

    Records contain all three groups; unchanged groups have candidate==base,
    making the macro contribution exactly zero.
    """
    masked_index = index[mask]
    days = masked_index.normalize().unique()
    positions = {day: np.flatnonzero(masked_index.normalize() == day) for day in days}
    rng = np.random.default_rng(20260718)
    values: list[float] = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([positions[day] for day in sampled])
        deltas = []
        for truth, base, candidate, group in records:
            cap = CAPACITY_KWH[group]
            before = evaluate_group(truth[rows], base[rows], cap)
            after = evaluate_group(truth[rows], candidate[rows], cap)
            deltas.append(float(after.score - before.score))
        values.append(float(np.mean(deltas)))
    array = np.asarray(values, dtype=float)
    return {
        "n_bootstrap": int(n_bootstrap),
        "positive_fraction": float(np.mean(array > 0.0)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _issue_block_bootstrap(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    index: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
    mask: np.ndarray,
    n_bootstrap: int,
) -> dict[str, float]:
    """Bootstrap complete NWP issue cycles, stratified by season."""
    _, season = assign_issue_blocks(index, issue_times)
    issue_values = np.asarray(issue_times)
    rng = np.random.default_rng(20260718)
    strata = sorted(set(season[mask]))
    positions: dict[tuple[str, object], np.ndarray] = {}
    issues_by_stratum: dict[str, np.ndarray] = {}
    for stratum in strata:
        issues = np.unique(issue_values[mask & (season == stratum)])
        issues_by_stratum[stratum] = issues
        for issue in issues:
            positions[(stratum, issue)] = np.flatnonzero(
                mask & (season == stratum) & (issue_values == issue)
            )
    values: list[float] = []
    for _ in range(int(n_bootstrap)):
        selected_parts: list[np.ndarray] = []
        for stratum, issues in issues_by_stratum.items():
            sampled = rng.choice(issues, size=len(issues), replace=True)
            selected_parts.extend(positions[(stratum, issue)] for issue in sampled)
        rows = np.concatenate(selected_parts)
        deltas = []
        for truth, base, candidate, group in records:
            cap = CAPACITY_KWH[group]
            before = evaluate_group(truth[rows], base[rows], cap)
            after = evaluate_group(truth[rows], candidate[rows], cap)
            deltas.append(float(after.score - before.score))
        values.append(float(np.mean(deltas)))
    array = np.asarray(values, dtype=float)
    return {
        "n_bootstrap": int(n_bootstrap),
        "positive_fraction": float(np.mean(array > 0.0)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _evaluate_probe(
    name: str,
    source: pd.DataFrame,
    cache: np.lib.npyio.NpzFile,
    policies: dict[str, tuple[float, float]],
    output_dir: Path,
    report_dir: Path,
    issue_source: Path,
    n_bootstrap: int,
) -> dict[str, object]:
    output = source.copy()
    index_by_group: dict[str, pd.DatetimeIndex] = {}
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    groups_changed: list[str] = []
    for group in GROUPS:
        index = pd.DatetimeIndex(pd.to_datetime(cache[f"{group}__valid_index_ns"]))
        index_by_group[group] = index
        base = cache[f"{group}__exact_base"].astype(float)
        truth = cache[f"{group}__valid_truth"].astype(float)
        test_base = cache[f"{group}__test_exact_base"].astype(float)
        source_values = source[group].to_numpy(dtype=float)
        parity_error = float(np.max(np.abs(source_values - test_base)))
        # The latest public-best base intentionally contains the group-3
        # cross-group/meta member, so exact_driver parity is required only for
        # groups that this probe actually modifies.
        if group in policies and parity_error > 0.05:
            raise ValueError(f"{group} lineage parity failed: {parity_error:.6f} kWh")
        if group in policies:
            scale, offset = policies[group]
            candidate = _affine(base, group, scale, offset)
            output[group] = _affine(source_values, group, scale, offset)
            groups_changed.append(group)
        else:
            candidate = base.copy()
        records.append((truth, base, candidate, group))

    # All three groups share the same hourly index in this competition.
    index = index_by_group[GROUPS[0]]
    periods = _periods(index)
    local: dict[str, object] = {"groups": {}, "macro": {}}
    for (truth, base, candidate, group) in records:
        period_rows: dict[str, object] = {}
        for period, period_mask in periods.items():
            period_rows[period] = {
                "delta": _delta(truth, base, candidate, group, period_mask),
                "n_rows": int(period_mask.sum()),
            }
        local["groups"][group] = period_rows
    for period, period_mask in periods.items():
        deltas = [local["groups"][g][period]["delta"] for g in GROUPS]
        local["macro"][period] = {
            key: float(np.mean([item[key] for item in deltas]))
            for key in ("score", "one_minus_nmae", "ficr")
        }

    h2 = periods["h2"]
    bootstrap = _bootstrap_macro(records, index, h2, n_bootstrap)
    issue_times = load_issue_times(issue_source, index)
    issue_bootstrap = _issue_block_bootstrap(
        records, index, issue_times, h2, n_bootstrap
    )
    output_path = output_dir / f"{name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    report = {
        "method": "controlled affine probe",
        "status": "manual_controlled_candidate",
        "auto_submit_eligible": False,
        "auto_submit_blockers": [
            "probe intentionally bypasses all-component promotion guard",
            "public leaderboard transfer is unknown; submit only after human review",
        ],
        "base_submission": str(Path("submissions") / source.attrs.get("source_name", "")),
        "base_sha256": source.attrs.get("source_sha256"),
        "policies": {
            group: {"scale": float(scale), "offset": float(offset)}
            for group, (scale, offset) in policies.items()
        },
        "groups_changed": groups_changed,
        "local_blocked_proxy": local,
        "locked_h2_macro_day_bootstrap": bootstrap,
        "locked_h2_issue_block_bootstrap": issue_bootstrap,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "rows": int(len(output)),
        "groups_unchanged": {
            group: bool(np.array_equal(output[group], source[group]))
            for group in GROUPS
            if group not in groups_changed
        },
    }
    report_path = report_dir / f"{name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-submission",
        default="submissions/blend_best_crossg3_traj_meta_finesweep.csv",
    )
    parser.add_argument(
        "--driver",
        default="artifacts_final/lineage/exact_driver_oof.npz",
    )
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument(
        "--report-dir", default="artifacts_final/agent_service/controlled_affine"
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--issue-source", default="data/train/gfs_train.csv")
    args = parser.parse_args()
    base_path = Path(args.base_submission)
    source = pd.read_csv(base_path, encoding="utf-8-sig")
    source.attrs["source_name"] = base_path.name
    source.attrs["source_sha256"] = _sha256(base_path)
    cache = np.load(args.driver)
    probes = {
        "blend_controlled_g1_099_p500": {"kpx_group_1": (0.990, 500.0)},
        "blend_controlled_g2_0988_p450": {"kpx_group_2": (0.988, 450.0)},
        "blend_controlled_g12_099_p500_0988_p450": {
            "kpx_group_1": (0.990, 500.0),
            "kpx_group_2": (0.988, 450.0),
        },
    }
    reports = [
        _evaluate_probe(
            name,
            source,
            cache,
            policies,
            Path(args.output_dir),
            Path(args.report_dir),
            Path(args.issue_source),
            args.n_bootstrap,
        )
        for name, policies in probes.items()
    ]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
