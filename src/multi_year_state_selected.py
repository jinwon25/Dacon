"""Cache the selected multi-year state model for sequential bias correction."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.multi_year_state_model import _add_categories, _model, _state_features
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.trackman_privileged_distillation import V17_NAME
from src.v16_residual_calibration_screen import load_v14_folds


HALF_LIFE = 0.5
LEAVES = 15


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    folds = load_v14_folds(project, train)
    numeric_state, baseline = _state_features(train)
    features = _add_categories(train, numeric_state)
    categorical = [column for column in features if column.startswith("cat__")]
    rows_out: list[dict[str, object]] = []
    for audit_year in (2022, 2023, 2024):
        print(f"[selected-state] audit_year={audit_year}", flush=True)
        fit_mask = train["season"].lt(audit_year).to_numpy()
        audit_mask = train["season"].eq(audit_year).to_numpy()
        fit_season = train.loc[fit_mask, "season"].to_numpy(np.float64)
        target = train.loc[fit_mask, "control_success"].to_numpy(np.float64)
        sample_weight = np.exp2(
            -(audit_year - 1.0 - fit_season) / HALF_LIFE
        )
        sample_weight /= sample_weight.mean()
        model = _model(leaves=LEAVES, seed=4100 + audit_year + LEAVES)
        model.fit(
            features.loc[fit_mask],
            target - baseline[fit_mask],
            sample_weight=sample_weight,
            categorical_feature=categorical,
        )
        raw = np.clip(
            baseline[audit_mask] + model.predict(features.loc[audit_mask]),
            0.001,
            0.999,
        )
        del model
        gc.collect()
        audit = train.loc[audit_mask].reset_index(drop=True)
        audit_target = audit["control_success"].to_numpy(np.float64)
        if audit_year == 2022:
            incumbent = folds[2022][2].astype(np.float64)
            incumbent_name = "v14_fallback"
        else:
            with np.load(
                project
                / "artifacts"
                / "v16_multiseason_20260815_02"
                / f"{V17_NAME}_o{audit_year}.npz"
            ) as saved:
                incumbent = saved["candidate"].astype(np.float64)
            incumbent_name = "v17"
        np.savez_compressed(
            output_dir / f"selected_state_o{audit_year}.npz",
            target=audit_target,
            incumbent=incumbent,
            raw=raw,
            domain3=audit["domain3"].astype(str).to_numpy(),
            game_month=audit["game_month"].to_numpy(np.int16),
            pitcher_id=audit["pitcher_id"].to_numpy(),
            batter_id=audit["batter_id"].to_numpy(),
        )
        rows_out.append(
            {
                "audit_year": audit_year,
                "incumbent": incumbent_name,
                "target_mean": float(audit_target.mean()),
                "incumbent_mean": float(incumbent.mean()),
                "raw_mean": float(raw.mean()),
                "raw_minus_incumbent_mean": float((raw - incumbent).mean()),
                "raw_sd": float(raw.std()),
            }
        )
    summary = {
        "protocol": "SELECTED_MULTI_YEAR_STATE_CACHE_V1",
        "half_life": HALF_LIFE,
        "leaves": LEAVES,
        "folds": rows_out,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/multi_year_state_selected_20260816_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
