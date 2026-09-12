"""Two fixed minimal LightGBM domain-feature experiments."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.calibration import apply_logit_offset
from src.archive.data import read_main
from src.archive.domain_drift import apply_game_type_offsets, fit_game_type_regime_offsets
from src.archive.followup import _load_all_caches
from src.metrics import brier_score
from src.archive.train import train_lgb_holdout


def run(
    project_dir: Path,
    years: list[int],
    variant_names: list[str] | None = None,
) -> pd.DataFrame:
    domain_config = json.loads(
        (project_dir / "research" / "configs" / "domain_experiments.json").read_text(
            encoding="utf-8"
        )
    )
    followup_config = json.loads(
        (project_dir / "research" / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    train = read_main(project_dir / "data" / "train.csv")
    folds, caches = _load_all_caches(project_dir, followup_config, train)
    rows: list[dict[str, object]] = []
    variants = domain_config["variants"]
    if variant_names is not None:
        variants = [item for item in variants if item["name"] in set(variant_names)]
        missing = set(variant_names) - {item["name"] for item in variants}
        if missing:
            raise ValueError(f"unknown variants: {sorted(missing)}")
    for variant in variants:
        for year in years:
            fold = folds[year]
            print(f"training {variant['name']} validation={year}")
            result = train_lgb_holdout(
                train,
                fold.train_idx,
                fold.valid_idx,
                variant,
                int(domain_config["max_boost_rounds"]),
                int(domain_config["early_stopping_rounds"]),
                None,
            )
            cache = caches[year]
            lgb_trend = apply_logit_offset(result["prediction"], float(cache["offset"]))
            blend = 0.35 * lgb_trend + 0.65 * cache["rf_trend"]
            offsets, _ = fit_game_type_regime_offsets(
                train.iloc[fold.train_idx], year
            )
            blend_regime = apply_game_type_offsets(
                blend,
                train.iloc[fold.valid_idx]["game_type"],
                offsets,
            )
            for suffix, prediction in (
                ("blend", blend),
                ("blend_plus_regime", blend_regime),
            ):
                target = cache["target"]
                incumbent = cache["incumbent"]
                rows.append(
                    {
                        "candidate": f"{variant['name']}_{suffix}",
                        "outer_validation_season": year,
                        "feature_set": variant["feature_set"],
                        "brier": brier_score(target, prediction),
                        "incumbent_brier": brier_score(target, incumbent),
                        "delta_brier": brier_score(target, prediction)
                        - brier_score(target, incumbent),
                        "best_iteration": result["best_iteration"],
                        "fit_seconds": result["fit_seconds"],
                        "inference_seconds": result["inference_seconds"],
                        "peak_memory_mb": result["peak_memory_mb"],
                    }
                )
            del result
            gc.collect()
    output = pd.DataFrame(rows)
    output_path = project_dir / "research" / "reports" / "domain_feature_results.csv"
    if output_path.exists():
        previous = pd.read_csv(output_path)
        output = pd.concat([previous, output], ignore_index=True)
        output = output.drop_duplicates(
            ["candidate", "outer_validation_season"], keep="last"
        )
    output = output.sort_values(["candidate", "outer_validation_season"])
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    run(args.project_dir.resolve(), args.years, args.variants)


if __name__ == "__main__":
    main()
