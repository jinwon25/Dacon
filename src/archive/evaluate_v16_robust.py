"""Dependence-aware evaluation of the selected multi-season v16 EB axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.evaluate_v14_v15_robust import _candidate_fold, _evaluate_fold
from src.robust_local_evaluation import white_reality_check
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.archive.v16_residual_calibration_screen import load_v14_folds


SELECTED = "multi_pitcher_batter_hand_d0.5_a3200_w1"
ARTIFACT_DIR = "v16_multiseason_20260815_01"


def _markdown(result: dict[str, object]) -> str:
    recipe = result["selected_recipe"]
    group = str(recipe["group"]).replace("_", " × ")
    parent_alpha = float(result["parent_f_alpha"])
    incremental_label = (
        "incremental_vs_v14" if abs(parent_alpha - 0.15) <= 1e-12
        else "incremental_vs_v15"
    )
    lines = [
        "# Multi-season empirical-Bayes robust evaluation",
        "",
        f"- Selected recipe: `{result['selected']}`",
        f"- Correction group: `{group}`; R_CORE fit/apply only.",
        f"- Historical decay / posterior alpha / correction weight: **{float(recipe['decay']):g} / {float(recipe['alpha']):g} / {float(recipe['weight']):g}**.",
        f"- Parent F-trend alpha: **{float(result['parent_f_alpha']):g}**.",
        "",
        "| comparison | year | gain | month + | worst month | min p05 | min P(+) | leave-team min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison, key in (
        (incremental_label, "incremental_vs_v14"),
        ("total_vs_v13", "total_vs_v13"),
    ):
        for fold in result[key]:
            lines.append(
                f"| {comparison} | {fold['year']} | "
                f"{fold['score']['unclipped_bss_equivalent_gain']:+.4f} | "
                f"{fold['month']['positive_fraction']:.1%} | "
                f"{fold['month']['worst_gain']:+.4f} | "
                f"{fold['minimum_resampling_p05']:+.4f} | "
                f"{fold['minimum_resampling_probability']:.1%} | "
                f"{fold['leave_one_team_out']['minimum_gain']:+.4f} |"
            )
    lines.extend(
        [
            "",
            f"- Final-shortlist (top five) Reality Check p-value: **{result['reality_check']['p_value']:.4f}**.",
            f"- Recipes screened before robust confirmation: **{result['selection_audit']['recipes_screened']}**.",
            "- Reality Check covers only the saved top-five multi-season shortlist; the larger search remains a source of selection bias.",
            "- 2024 is development-contaminated, so this supports a deployment probe rather than independent confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    project: Path,
    output_json: Path,
    output_markdown: Path,
    selected_name: str = SELECTED,
    artifact_dir_name: str = ARTIFACT_DIR,
    parent_f_alpha: float = 0.15,
) -> dict[str, object]:
    project = project.resolve()
    config = json.loads((project / "configs" / "local_evaluation_v2.json").read_text(encoding="utf-8"))
    train = _add_domain_and_pressure(pd.read_csv(project / "data" / "train.csv", low_memory=False))
    v14_folds = load_v14_folds(project, train)
    artifact_dir = project / "artifacts" / artifact_dir_name
    selected: dict[int, np.ndarray] = {}
    incremental = []
    total = []
    for year in (2023, 2024):
        path = artifact_dir / f"{selected_name}_o{year}.npz"
        with np.load(path) as saved:
            candidate = saved["candidate"].astype(np.float64)
            cached_incumbent = saved["incumbent"].astype(np.float64)
            cached_target = saved["target"].astype(np.float64)
        rows, target, v14 = v14_folds[year]
        if not np.array_equal(target, cached_target) or not np.allclose(v14, cached_incumbent, atol=1e-12):
            raise ValueError(f"selected cache mismatch for {year}")
        rows2, target2, v13, parent = _candidate_fold(
            project, train, year, parent_f_alpha
        )
        if not np.array_equal(target, target2):
            raise ValueError(f"parent target mismatch for {year}")
        candidate = np.clip(parent + (candidate - v14), 1e-6, 1.0 - 1e-6)
        selected[year] = candidate
        incremental.append(
            _evaluate_fold(selected_name, year, rows, target, parent, candidate, config)
        )
        _, _, _, reconstructed_v14 = _candidate_fold(project, train, year, 0.15)
        if not np.allclose(v14, reconstructed_v14, atol=1e-12):
            raise ValueError(f"v13/v14 reconstruction mismatch for {year}")
        total.append(
            _evaluate_fold(selected_name, year, rows2, target2, v13, candidate, config)
        )

    shortlist_paths = sorted(artifact_dir.glob("multi_*_o2024.npz"))
    shortlist_names = [path.name.removesuffix("_o2024.npz") for path in shortlist_paths]
    target_2024 = v14_folds[2024][1]
    v14_2024 = v14_folds[2024][2]
    _, _, _, parent_2024 = _candidate_fold(project, train, 2024, parent_f_alpha)
    improvement_columns = []
    for path in shortlist_paths:
        with np.load(path) as saved:
            cached_candidate = saved["candidate"].astype(np.float64)
        candidate = np.clip(
            parent_2024 + (cached_candidate - v14_2024), 1e-6, 1.0 - 1e-6
        )
        improvement_columns.append(
            np.square(parent_2024 - target_2024) - np.square(candidate - target_2024)
        )
    reality = white_reality_check(
        target_2024,
        np.column_stack(improvement_columns),
        block_size=2000,
        n_resamples=int(config["bootstrap_resamples"]),
        seed=int(config["seed"]) + 1600,
    )
    reality["candidate_names"] = shortlist_names
    screen = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    screen_metrics = pd.read_csv(artifact_dir / "metrics.csv")
    selected_rows = screen_metrics.loc[screen_metrics["candidate"].eq(selected_name)]
    if selected_rows.empty:
        raise ValueError("selected recipe missing from screen metrics")
    selected_recipe = {
        key: selected_rows.iloc[0][key]
        for key in ("group", "decay", "alpha", "weight")
    }
    earlier = json.loads(
        (project / "artifacts" / "v16_residual_calibration_20260815_02" / "summary.json").read_text(encoding="utf-8")
    )
    result: dict[str, object] = {
        "protocol": "LOCAL_EVALUATION_V2_V16_DEPENDENCE_AWARE",
        "selected": selected_name,
        "selected_recipe": selected_recipe,
        "parent_f_alpha": float(parent_f_alpha),
        "incremental_vs_v14": incremental,
        "total_vs_v13": total,
        "reality_check": reality,
        "selection_audit": {
            "recipes_screened": int(screen["n_recipes"]) + int(earlier["n_recipes"]),
            "multi_season_recipes": int(screen["n_recipes"]),
            "earlier_residual_recipes": int(earlier["n_recipes"]),
            "shortlist_reality_check_size": len(shortlist_names),
            "confirmation_2024_is_virgin": False,
        },
    }
    output_json = project / output_json
    output_markdown = project / output_markdown
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    output_markdown.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected_name,
                "incremental": [
                    {
                        "year": fold["year"],
                        "gain": fold["score"]["unclipped_bss_equivalent_gain"],
                        "min_p05": fold["minimum_resampling_p05"],
                        "min_probability": fold["minimum_resampling_probability"],
                    }
                    for fold in incremental
                ],
                "total": [
                    {
                        "year": fold["year"],
                        "gain": fold["score"]["unclipped_bss_equivalent_gain"],
                        "min_p05": fold["minimum_resampling_p05"],
                        "min_probability": fold["minimum_resampling_probability"],
                    }
                    for fold in total
                ],
                "reality_check": reality,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, default=Path("reports/v16_robust_evaluation_20260815.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("reports/v16_robust_evaluation_20260815.md"))
    parser.add_argument("--selected", default=SELECTED)
    parser.add_argument("--artifact-dir", default=ARTIFACT_DIR)
    parser.add_argument("--parent-f-alpha", type=float, default=0.15)
    args = parser.parse_args()
    run(
        args.project,
        args.output_json,
        args.output_markdown,
        args.selected,
        args.artifact_dir,
        args.parent_f_alpha,
    )


if __name__ == "__main__":
    main()
