"""Post-break low-degree spline/linear probability screen above v22.

The F-domain regime changes sharply between 2022 and 2023.  This experiment
therefore discards pre-break labels, selects a low-variance direct model on
2023 early -> late, then refits on full 2023 and opens 2024 once.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from src.core.axes import _joint_domain
from src.v23_neural_embedding_screen import DOMAIN_SUBSETS, ETAS, _diagnostics
from src.core.axes import _derived, _load_axis


CATEGORICAL = (
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "count_state",
    "hand_matchup",
    "inning_bucket",
    "domain3",
)
EXCLUDED = {
    "row_id",
    "season",
    "control_success",
    "target",
    "v21",
    "v22",
    "pitcher_id",
    "batter_id",
    *CATEGORICAL,
}


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    strength: float


SPECS = (
    Spec("logistic_c001", "logistic", 0.01),
    Spec("logistic_c01", "logistic", 0.1),
    Spec("logistic_c1", "logistic", 1.0),
    Spec("ridge_a100", "ridge", 100.0),
    Spec("ridge_a1000", "ridge", 1000.0),
    Spec("ridge_a10000", "ridge", 10000.0),
)


def _columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        column
        for column in frame.columns
        if column not in EXCLUDED and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rates = [column for column in numeric if column.startswith("asof_") and column.endswith("_rate")]
    linear = [column for column in numeric if column not in rates]
    return rates, linear


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    rates, linear = _columns(frame)
    rate_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=5, degree=2, include_bias=False)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    linear_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("rates", rate_pipeline, rates),
            ("linear", linear_pipeline, linear),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                list(CATEGORICAL),
            ),
        ],
        sparse_threshold=0.3,
    )


def _pipeline(fit: pd.DataFrame, spec: Spec) -> Pipeline:
    preprocessor = _preprocessor(fit)
    if spec.family == "logistic":
        estimator = LogisticRegression(
            C=spec.strength,
            solver="lbfgs",
            max_iter=400,
            tol=1e-7,
        )
    elif spec.family == "ridge":
        estimator = Ridge(alpha=spec.strength, solver="lsqr", tol=1e-7)
    else:
        raise ValueError(spec.family)
    return Pipeline([("features", preprocessor), ("model", estimator)])


def _fit_predict(fit: pd.DataFrame, audit: pd.DataFrame, spec: Spec) -> np.ndarray:
    model = _pipeline(fit, spec)
    model.fit(fit, fit["control_success"].to_numpy(np.float64))
    if spec.family == "logistic":
        prediction = model.predict_proba(audit)[:, 1]
    else:
        prediction = model.predict(audit)
    return np.clip(np.asarray(prediction, dtype=np.float64), 0.001, 0.999)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    raw_2023 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    derived_2023 = _derived(raw_2023, _joint_domain(raw_2023))
    selection_fit = derived_2023.loc[derived_2023["game_month"].le(7)].reset_index(drop=True)
    selection = _load_axis(project, "y2023_early_to_late", raw)
    rows = []
    predictions = {}
    for spec in SPECS:
        prediction = _fit_predict(selection_fit, selection, spec)
        predictions[spec.name] = prediction
        for eta in ETAS:
            for domains in DOMAIN_SUBSETS:
                rows.append(
                    {
                        "model": spec.name,
                        "family": spec.family,
                        "strength": spec.strength,
                        "eta": eta,
                        "domains": "+".join(domains),
                        **_diagnostics(selection, prediction, eta, domains),
                    }
                )
        print(f"[postbreak] selection {spec.name}", flush=True)
    metrics = pd.DataFrame(rows).sort_values(["selection_score", "gain"], ascending=False)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    selected = metrics.iloc[0]
    selection_gate = bool(
        selected["gain"] > 0.0
        and selected["worst_month_gain"] > 0.0
        and selected["minimum_applied_domain_gain"] > 0.0
    )
    summary: dict[str, object] = {
        "protocol": "V23_POSTBREAK_GAM_V1",
        "selection": "fit 2023 March-July; model, eta and domain route selected on 2023 August-October",
        "selected": {key: selected[key] for key in metrics.columns},
        "selection_gate_passed": selection_gate,
        "outer_audit_run": False,
        "eligible_for_packaging": False,
    }
    if selection_gate:
        chosen = next(spec for spec in SPECS if spec.name == selected["model"])
        outer = _load_axis(project, "y2023_to_y2024", raw)
        outer_prediction = _fit_predict(derived_2023, outer, chosen)
        domains = tuple(str(selected["domains"]).split("+"))
        diagnostics = _diagnostics(outer, outer_prediction, float(selected["eta"]), domains)
        eligible = bool(
            diagnostics["gain"] >= 5.0
            and diagnostics["positive_month_fraction"] >= 0.75
            and diagnostics["minimum_applied_domain_gain"] > 0.0
            and diagnostics["worst_month_gain"] > -10.0
        )
        parent = outer["v22"].to_numpy(np.float64)
        apply_mask = outer["domain3"].astype(str).isin(domains).to_numpy()
        candidate = parent.copy()
        candidate[apply_mask] = np.clip(
            parent[apply_mask]
            + float(selected["eta"])
            * (outer_prediction[apply_mask] - parent[apply_mask]),
            0.001,
            0.999,
        )
        np.savez_compressed(
            output_dir / "outer_prediction.npz",
            target=outer["target"].to_numpy(np.float64),
            v22=parent,
            direct=outer_prediction,
            candidate=candidate,
            apply_mask=apply_mask,
        )
        summary.update(
            {
                "outer_audit_run": True,
                "outer_diagnostics": diagnostics,
                "eligible_for_packaging": eligible,
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v23_postbreak_gam_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
