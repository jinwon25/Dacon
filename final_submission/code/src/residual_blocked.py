"""Record preregistered residual recipes as blocked when nested OOF is absent."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def run(project: Path) -> pd.DataFrame:
    rows = []
    for recipe in ("eb_count_platoon", "ridge_zero_intercept", "shallow_lgb_residual"):
        for year in (2021, 2022, 2023, 2024):
            rows.append({"recipe": recipe, "outer_validation_season": year, "brier": float("nan"), "delta_v2": float("nan"), "eta": float("nan"), "correction_abs_p995": float("nan"), "correction_abs_max": float("nan"), "correction_mean": float("nan"), "status": "BLOCKED: v2_nested primary OOF unavailable; no target-dependent selection performed"})
    out = pd.DataFrame(rows)
    out.to_csv(project / "reports/residual_recipe_results.csv", index=False)
    pd.DataFrame(columns=["recipe", "outer_validation_season", "subgroup", "n_rows", "brier", "delta_v2"]).to_csv(project / "reports/residual_subgroup_results.csv", index=False)
    print(out.to_string(index=False)); return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); args = parser.parse_args(); run(args.project_dir.resolve())


if __name__ == "__main__": main()
