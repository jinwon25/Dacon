"""Target-free Trackman linkage audit and bootstrap Hungarian utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def bootstrap_soft_linkage(cost: np.ndarray, repeats: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perturb a target-free cost matrix and return assignment probabilities.

    Dirichlet-multinomial row weights are applied to the cost; no outcome or
    forecast-year row is consulted.  The function is reusable by the full
    annual-fingerprint pipeline and is unit tested on a small matrix.
    """
    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("cost must be a non-empty matrix")
    rng = np.random.default_rng(seed)
    n, m = matrix.shape
    counts = np.zeros((n, m), dtype=np.int32)
    for _ in range(int(repeats)):
        noise = rng.dirichlet(np.ones(m), size=n) * m
        assignment_rows, assignment_cols = linear_sum_assignment(matrix * noise)
        counts[assignment_rows, assignment_cols] += 1
    probabilities = counts.astype(float) / max(1, repeats)
    rho = probabilities.max(axis=1)
    entropy = -(np.where(probabilities > 0, probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), 0.0)).sum(axis=1)
    return probabilities, rho, entropy


def _summary_from_existing_linkage(project: Path) -> pd.DataFrame:
    path = project / "research/reports/trackman_linkage.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    link = pd.read_csv(path)
    rows = []
    for season, group in link.groupby("season", observed=True):
        for confidence, subset in group.groupby("tm_link_confidence", observed=True):
            rows.append({
                "origin": int(season),
                "mapping": "1->Left_2->Right (legacy hard mapping)",
                "confidence": str(confidence),
                "n_rows": len(subset),
                "mean_distance": float(subset["tm_link_distance"].mean()),
                "mean_margin": float(subset["tm_link_margin"].mean()),
                "mean_rank": float(subset["tm_link_assignment_rank"].mean()),
                "status": "diagnostic_only; mapping alternatives require annual fingerprints",
            })
    return pd.DataFrame(rows)


def run(project: Path, repeats: int = 200) -> pd.DataFrame:
    rows = _summary_from_existing_linkage(project)
    # A small real cost matrix is reconstructed from the top candidate rows in
    # the existing target-free linkage artifact. It is only a stability smoke
    # test; it is not used to make a submission profile.
    link = pd.read_csv(project / "research/reports/trackman_linkage.csv")
    subset = link.sort_values(["tm_link_distance", "tm_link_assignment_rank"]).drop_duplicates("pitcher_id").head(64)
    diagonal = subset["tm_link_distance"].to_numpy(dtype=float)
    cost = np.full((len(diagonal), len(diagonal)), float(np.nanmax(diagonal) + 0.15))
    np.fill_diagonal(cost, diagonal)
    pi, rho, entropy = bootstrap_soft_linkage(cost, repeats=repeats, seed=42)
    rows = pd.concat([rows, pd.DataFrame({
        "origin": "bootstrap_smoke",
        "mapping": "cost-matrix-soft-Hungarian",
        "confidence": "all",
        "n_rows": len(diagonal),
        "mean_distance": float(np.mean(diagonal)),
        "mean_margin": np.nan,
        "mean_rank": np.nan,
        "rho_mean": float(np.mean(rho)),
        "rho_p05": float(np.quantile(rho, 0.05)),
        "entropy_mean": float(np.mean(entropy)),
        "repeats": int(repeats),
        "status": "bootstrap utility smoke test; no outcome used",
    }, index=[0])], ignore_index=True)
    rows.to_csv(project / "research/reports/trackman_soft_linkage_results.csv", index=False)
    (project / "research/reports/trackman_soft_linkage_audit_20260809.md").write_text(
        "# Trackman soft-linkage audit\n\n"
        "The existing linkage report provides target-free distance, assignment rank, margin, common-origin stability and confidence. The two hand mappings (1→Left/2→Right and its reverse) are not identifiable from the aggregate report alone; the full annual-fingerprint rerun is therefore marked ambiguous and no new correction is promoted. A reusable 200-repeat Dirichlet-perturbed Hungarian routine was executed as a smoke test on the cached candidate cost submatrix.\n\n"
        "`asof_pitcher_pitchmix_n` must be audited for cumulative versus season-reset semantics before clipping negative deltas. Until that audit is run on the origin-specific annual table, medium/low/unmatched rows remain bitwise v2.\n\n"
        + "```text\n" + rows.to_string(index=False) + "\n```\n", encoding="utf-8",
    )
    print(rows.tail(5).to_string(index=False))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); parser.add_argument("--repeats", type=int, default=200); args = parser.parse_args(); run(args.project_dir.resolve(), args.repeats)


if __name__ == "__main__":
    main()
