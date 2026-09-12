"""Forward-only audit of conservative v13 component-weight changes.

The script reconstructs a historical analogue of the submitted v13 overlay
for 2022->2023 and 2023->2024.  Every feature/model used for an audit season is
fit on information available no later than its source season.  The audit then
asks a deliberately narrow question: is there a component-weight change that
beats v13 in *both* transitions?

This is a research/selection script.  It never reads evaluation data and never
creates or submits a DACON package.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.temporal_stable_conditional import (
    _add_domain_and_pressure,
    build_bank,
    build_features,
)


ANCHOR_TEAM = 13
TRANSITIONS = ((2022, 2023), (2023, 2024))
V13_CORE = {
    "bias_weight": 1.0,
    "lgb_weight": 0.125,
    "ridge_weight": 0.20,
    "stable_weight": 0.20,
}
V13_F_WEIGHT = 0.75


@dataclass
class Fold:
    year: int
    train_index: np.ndarray
    target: np.ndarray
    base: np.ndarray
    exact_lgb: np.ndarray
    exact_ridge: np.ndarray
    trend_lgb: np.ndarray
    rows: pd.DataFrame

    @property
    def core(self) -> np.ndarray:
        return self.rows["domain3"].eq("R_CORE").to_numpy()

    @property
    def anchor(self) -> np.ndarray:
        return self.rows["domain3"].eq("R_ANCHOR").to_numpy()

    @property
    def finals(self) -> np.ndarray:
        return self.rows["domain3"].eq("F").to_numpy()


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(np.mean(target))
    reference = rate * (1.0 - rate)
    brier = float(np.mean(np.square(prediction - target)))
    return 100_000.0 * (1.0 - brier / reference)


def load_folds(train: pd.DataFrame, exact_dir: Path) -> dict[int, Fold]:
    folds: dict[int, Fold] = {}
    for year in (2022, 2023, 2024):
        path = exact_dir / f"recent_shared_o{year}.npz"
        with np.load(path) as saved:
            train_index = saved["train_index"].astype(np.int64)
            target = saved["target"].astype(np.float64)
            expected = train.iloc[train_index]["control_success"].to_numpy(np.float64)
            if not np.array_equal(target, expected):
                raise ValueError(f"target/order mismatch in {path}")
            folds[year] = Fold(
                year=year,
                train_index=train_index,
                target=target,
                base=saved["base"].astype(np.float64),
                exact_lgb=saved["exact_lgb"].astype(np.float64),
                exact_ridge=saved["exact_ridge"].astype(np.float64),
                trend_lgb=saved["trend_lgb"].astype(np.float64),
                rows=train.iloc[train_index].reset_index(drop=True),
            )
    return folds


def fit_stable_component(
    train: pd.DataFrame, source: Fold, audit: Fold
) -> tuple[float, np.ndarray]:
    """Fit the exact centered stable component used by the v13 recipe."""

    source_core = source.core
    audit_core = audit.core
    source_rows = source.rows.loc[source_core].reset_index(drop=True)
    audit_rows = audit.rows.loc[audit_core].reset_index(drop=True)
    source_bank = build_bank(train.loc[train["season"].eq(source.year - 1)])
    audit_bank = build_bank(train.loc[train["season"].eq(source.year)])
    source_x = build_features(source_rows, source.base[source_core], source_bank)
    audit_x = build_features(audit_rows, audit.base[audit_core], audit_bank)
    scaler = StandardScaler()
    fit_x = scaler.fit_transform(source_x)
    audit_x_scaled = scaler.transform(audit_x)
    residual = source.target[source_core] - source.base[source_core]
    model = Ridge(alpha=10_000.0, fit_intercept=True, solver="cholesky")
    model.fit(fit_x, residual)
    centered = model.predict(audit_x_scaled) - float(model.intercept_)
    correction = np.zeros(len(audit.target), dtype=np.float64)
    correction[audit_core] = centered
    return float(np.mean(residual)), correction


def predict_recipe(
    audit: Fold,
    source_bias: float,
    stable: np.ndarray,
    *,
    bias_weight: float,
    lgb_weight: float,
    ridge_weight: float,
    stable_weight: float,
    f_weight: float,
) -> np.ndarray:
    prediction = audit.base.copy()
    core = audit.core
    prediction[core] = (
        audit.base[core]
        + bias_weight * source_bias
        + lgb_weight * (audit.exact_lgb[core] - audit.base[core])
        + ridge_weight * (audit.exact_ridge[core] - audit.base[core])
        + stable_weight * stable[core]
    )
    finals = audit.finals
    prediction[finals] = audit.base[finals] + f_weight * (
        audit.exact_lgb[finals] - audit.base[finals]
    )
    return np.clip(prediction, 1e-6, 1.0 - 1e-6)


def metric_rows(
    audit: Fold,
    candidate: np.ndarray,
    incumbent: np.ndarray,
    parameters: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    masks = {
        "ALL": np.ones(len(audit.target), dtype=bool),
        "R_CORE": audit.core,
        "R_ANCHOR": audit.anchor,
        "F": audit.finals,
    }
    for domain, mask in masks.items():
        base_bss = bss(audit.target[mask], audit.base[mask])
        incumbent_bss = bss(audit.target[mask], incumbent[mask])
        candidate_bss = bss(audit.target[mask], candidate[mask])
        rows.append(
            {
                "audit_year": audit.year,
                "domain": domain,
                "n_rows": int(mask.sum()),
                **parameters,
                "base_bss": base_bss,
                "v13_bss": incumbent_bss,
                "candidate_bss": candidate_bss,
                "gain_vs_base": candidate_bss - base_bss,
                "gain_vs_v13": candidate_bss - incumbent_bss,
            }
        )
    return rows


def robust_table(results: pd.DataFrame, domain: str) -> pd.DataFrame:
    keys = [
        "bias_weight",
        "lgb_weight",
        "ridge_weight",
        "stable_weight",
        "f_weight",
    ]
    subset = results.loc[results["domain"].eq(domain)]
    robust = (
        subset.groupby(keys, observed=True)["gain_vs_v13"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
    )
    robust["positive_folds"] = (
        subset.assign(positive=subset["gain_vs_v13"].gt(0))
        .groupby(keys, observed=True)["positive"]
        .sum()
        .to_numpy()
    )
    return robust.sort_values(
        ["min_gain", "mean_gain"], ascending=[False, False]
    ).reset_index(drop=True)


def run(project: Path, exact_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    exact_dir = exact_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    folds = load_folds(train, exact_dir)
    transition_parts: dict[int, tuple[float, np.ndarray]] = {}
    v13_predictions: dict[int, np.ndarray] = {}
    for source_year, audit_year in TRANSITIONS:
        transition_parts[audit_year] = fit_stable_component(
            train, folds[source_year], folds[audit_year]
        )
        source_bias, stable = transition_parts[audit_year]
        v13_predictions[audit_year] = predict_recipe(
            folds[audit_year],
            source_bias,
            stable,
            **V13_CORE,
            f_weight=V13_F_WEIGHT,
        )
        np.savez_compressed(
            output_dir / f"v13_components_o{audit_year}.npz",
            train_index=folds[audit_year].train_index,
            target=folds[audit_year].target,
            base=folds[audit_year].base,
            v13=v13_predictions[audit_year],
            exact_lgb=folds[audit_year].exact_lgb,
            exact_ridge=folds[audit_year].exact_ridge,
            stable=stable,
            source_bias=np.asarray([source_bias]),
        )

    core_grid = product(
        (0.75, 1.0),
        (0.0625, 0.125, 0.1875),
        (0.10, 0.20, 0.30),
        (0.0, 0.10, 0.20, 0.35),
    )
    recipes = [
        {
            "bias_weight": bias_weight,
            "lgb_weight": lgb_weight,
            "ridge_weight": ridge_weight,
            "stable_weight": stable_weight,
            "f_weight": V13_F_WEIGHT,
        }
        for bias_weight, lgb_weight, ridge_weight, stable_weight in core_grid
    ]
    recipes.extend(
        {
            **V13_CORE,
            "f_weight": f_weight,
        }
        for f_weight in (0.55, 0.65, 0.75, 0.85, 0.95)
        if f_weight != V13_F_WEIGHT
    )
    # De-duplicate the incumbent recipe if grids happen to include it.
    recipes = list({tuple(sorted(recipe.items())): recipe for recipe in recipes}.values())

    rows: list[dict[str, object]] = []
    for parameters in recipes:
        for _, audit_year in TRANSITIONS:
            source_bias, stable = transition_parts[audit_year]
            candidate = predict_recipe(
                folds[audit_year], source_bias, stable, **parameters
            )
            rows.extend(
                metric_rows(
                    folds[audit_year],
                    candidate,
                    v13_predictions[audit_year],
                    parameters,
                )
            )
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "component_grid.csv", index=False)
    robust_all = robust_table(results, "ALL")
    robust_core = robust_table(results, "R_CORE")
    robust_f = robust_table(results, "F")
    robust_all.to_csv(output_dir / "robust_all.csv", index=False)
    robust_core.to_csv(output_dir / "robust_core.csv", index=False)
    robust_f.to_csv(output_dir / "robust_f.csv", index=False)

    incumbent_rows = []
    for _, audit_year in TRANSITIONS:
        audit = folds[audit_year]
        incumbent = v13_predictions[audit_year]
        incumbent_rows.append(
            {
                "audit_year": audit_year,
                "n_rows": len(audit.target),
                "base_bss": bss(audit.target, audit.base),
                "v13_bss": bss(audit.target, incumbent),
                "v13_gain_vs_base": bss(audit.target, incumbent)
                - bss(audit.target, audit.base),
                "source_core_bias": transition_parts[audit_year][0],
            }
        )
    pd.DataFrame(incumbent_rows).to_csv(
        output_dir / "v13_reconstruction.csv", index=False
    )
    passing = robust_all.loc[robust_all["min_gain"].gt(0)]
    summary = {
        "protocol": "V14_FORWARD_COMPONENT_AUDIT_V1",
        "transitions": [f"{source}->{audit}" for source, audit in TRANSITIONS],
        "incumbent": {**V13_CORE, "f_weight": V13_F_WEIGHT},
        "n_recipes": len(recipes),
        "n_all_domain_strict_improvements": int(len(passing)),
        "best_all": robust_all.head(10).to_dict(orient="records"),
        "best_core": robust_core.head(10).to_dict(orient="records"),
        "best_f": robust_f.head(10).to_dict(orient="records"),
        "selection_eligible": bool(len(passing)),
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
        "--exact-dir",
        type=Path,
        default=Path("artifacts/recent_shared_exact_asof_20260815_02"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.exact_dir, args.output_dir)


if __name__ == "__main__":
    main()
