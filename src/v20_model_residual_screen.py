"""Shallow forward residual-model screen above the v19 champion.

This complements the discrete empirical-Bayes lookups with smooth row-local
structure.  Model hyperparameters and feature families are intentionally
small and fixed, and every candidate must transfer on the same three temporal
axes used by :mod:`src.v20_residual_overlay_screen`.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v20_residual_overlay_screen import WEIGHTS, _bss


CATEGORICAL = (
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "domain3",
    "pressure",
)
EXCLUDED = {"row_id", "season", "game_month", "control_success", "target", "residual"}
ENTITY_IDS = {"pitcher_id", "batter_id"}
ASOF_PREFIX = "asof_"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    leaves: int


MODEL_SPECS = tuple(
    ModelSpec(f"{family}_l{leaves}", family, leaves)
    for family in ("context", "state", "full")
    for leaves in (3, 7)
)


def _model(spec: ModelSpec, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        verbosity=-1,
        n_jobs=6,
        n_estimators=120 if spec.leaves == 3 else 160,
        learning_rate=0.025,
        num_leaves=spec.leaves,
        max_depth=2 if spec.leaves == 3 else 3,
        min_child_samples=1200 if spec.leaves == 3 else 800,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=5.0,
        reg_lambda=30.0,
        max_bin=127,
        random_state=seed,
    )


def _load(project: Path) -> pd.DataFrame:
    train = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    train = train.loc[train["season"].isin((2023, 2024))].reset_index(drop=True)
    train = _add_domain_and_pressure(train)
    for column in CATEGORICAL:
        train[column] = (
            train[column].astype("string").fillna("__MISSING__").astype("category")
        )
    artifact = project / "artifacts" / "state_mode_joint_20260816_02"
    pieces = []
    for year in (2023, 2024):
        rows = train.loc[train["season"].eq(year)].copy()
        with np.load(artifact / f"joint_candidate_o{year}.npz", allow_pickle=True) as saved:
            target = saved["target"].astype(np.float64)
            if not np.array_equal(
                target, rows["control_success"].to_numpy(np.float64)
            ):
                raise ValueError(f"v19 OOF order mismatch for {year}")
            rows["target"] = target
            rows["v19"] = saved["candidate"].astype(np.float64)
            if not np.array_equal(
                rows["domain3"].astype(str).to_numpy(), saved["domain3"].astype(str)
            ):
                raise ValueError(f"domain order mismatch for {year}")
        rows["residual"] = rows["target"] - rows["v19"]
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def _columns(frame: pd.DataFrame, family: str) -> list[str]:
    columns = [column for column in frame.columns if column not in EXCLUDED]
    if family == "context":
        columns = [
            column
            for column in columns
            if not column.startswith(ASOF_PREFIX) and column not in ENTITY_IDS
        ]
    elif family == "state":
        columns = [column for column in columns if column not in ENTITY_IDS]
    elif family != "full":
        raise ValueError(f"unknown family: {family}")
    return columns


def _split_masks(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    year = frame["season"].to_numpy()
    month = frame["game_month"].to_numpy()
    return {
        "y2023_to_y2024": (year == 2023, year == 2024),
        "y2023_early_to_late": (
            (year == 2023) & (month <= 7),
            (year == 2023) & (month >= 8),
        ),
        "y2024_early_to_late": (
            (year == 2024) & (month <= 7),
            (year == 2024) & (month >= 8),
        ),
    }


def _gain_grid(
    target: np.ndarray,
    incumbent: np.ndarray,
    raw: np.ndarray,
    apply_mask: np.ndarray,
) -> np.ndarray:
    correction = np.zeros(len(target), dtype=np.float64)
    correction[apply_mask] = raw[apply_mask]
    residual = target - incumbent
    rate = float(np.mean(target))
    scale = 100000.0 / (rate * (1.0 - rate))
    linear = scale * 2.0 * np.mean(residual * correction)
    square = scale * np.mean(correction**2)
    weights = np.asarray(WEIGHTS, dtype=np.float64)
    return linear * weights - square * weights**2


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load(project)
    splits = _split_masks(frame)
    rows: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {}
    for split_index, (split_name, (fit_mask, audit_mask)) in enumerate(splits.items()):
        audit = frame.loc[audit_mask].reset_index(drop=True)
        target = audit["target"].to_numpy(np.float64)
        incumbent = audit["v19"].to_numpy(np.float64)
        domain_values = audit["domain3"].astype(str).to_numpy()
        saved[f"{split_name}__target"] = target
        saved[f"{split_name}__v19"] = incumbent
        saved[f"{split_name}__domain3"] = domain_values
        for spec_index, spec in enumerate(MODEL_SPECS):
            columns = _columns(frame, spec.family)
            categorical = [column for column in columns if column in CATEGORICAL]
            model = _model(spec, seed=6200 + 100 * split_index + spec_index)
            model.fit(
                frame.loc[fit_mask, columns],
                frame.loc[fit_mask, "residual"].to_numpy(np.float64),
                categorical_feature=categorical,
            )
            raw = model.predict(frame.loc[audit_mask, columns]).astype(np.float64)
            saved[f"{split_name}__{spec.name}"] = raw
            del model
            gc.collect()
            masks = {
                "ALL": np.ones(len(audit), dtype=bool),
                "R_CORE": domain_values == "R_CORE",
                "R_ANCHOR": domain_values == "R_ANCHOR",
                "F": domain_values == "F",
            }
            for domain, apply_mask in masks.items():
                gains = _gain_grid(target, incumbent, raw, apply_mask)
                for weight_index, weight in enumerate(WEIGHTS):
                    rows.append(
                        {
                            "split": split_name,
                            "model": spec.name,
                            "family": spec.family,
                            "leaves": spec.leaves,
                            "domain": domain,
                            "weight": weight,
                            "gain": float(gains[weight_index]),
                            "raw_mean": float(np.mean(raw[apply_mask])),
                            "raw_sd": float(np.std(raw[apply_mask])),
                        }
                    )
            print(f"[{split_name}] {spec.name}", flush=True)
    metrics = pd.DataFrame(rows)
    robust = (
        metrics.groupby(["model", "family", "leaves", "domain", "weight"], as_index=False)
        .agg(
            min_gain=("gain", "min"),
            mean_gain=("gain", "mean"),
            max_gain=("gain", "max"),
        )
        .sort_values(["min_gain", "mean_gain"], ascending=False)
        .reset_index(drop=True)
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust.to_csv(output_dir / "robust.csv", index=False)
    np.savez_compressed(output_dir / "predictions.npz", **saved)
    summary = {
        "protocol": "V19_FORWARD_SHALLOW_RESIDUAL_MODEL_V1",
        "splits": list(splits),
        "models": [spec.__dict__ for spec in MODEL_SPECS],
        "best": robust.head(30).to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v20_model_residual_20260816_01"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
