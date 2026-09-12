"""Nested multi-year direct-model blend above v22.

A direct probability model supplies a genuinely different error geometry from
the champion's residual lookup stack.  Candidate family, seasonal half-life,
and blend eta are selected only on honest late-2023 v22 OOF.  The choice is
then frozen, refit through 2023, and audited once on full 2024.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.core.overlay import _bss
from src.core.axes import _derived, _load_axis
from src.archive.v23_structural_residual_screen import ETAS, _feature_columns, _frames
from src.core.axes import _joint_domain


@dataclass(frozen=True)
class DirectSpec:
    name: str
    family: str
    include_ids: bool
    half_life: float


DIRECT_SPECS = tuple(
    DirectSpec(
        f"{family}_{'ids' if include_ids else 'context'}_h{half_life:g}",
        family,
        include_ids,
        half_life,
    )
    for family, include_ids, half_life in (
        ("l2", False, 1.0),
        ("l2", False, 2.0),
        ("l2", True, 1.0),
        ("l2", True, 2.0),
        ("binary", False, 1.0),
        ("binary", True, 1.0),
    )
)


def _model(spec: DirectSpec, seed: int):
    common = {
        "verbosity": -1,
        "n_jobs": 6,
        "n_estimators": 240,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 1200,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 3.0,
        "reg_lambda": 25.0,
        "max_bin": 127,
        "random_state": seed,
    }
    if spec.family == "l2":
        return lgb.LGBMRegressor(objective="regression_l2", **common)
    if spec.family == "binary":
        return lgb.LGBMClassifier(objective="binary", **common)
    raise ValueError(spec.family)


def _fit_predict(
    fit: pd.DataFrame, audit: pd.DataFrame, spec: DirectSpec, seed: int
) -> np.ndarray:
    columns = _feature_columns(fit, spec.include_ids)
    fit_x, audit_x, categorical = _frames(
        fit, audit, columns, spec.include_ids
    )
    latest = int(fit["season"].max())
    weight = np.exp2(
        -(latest - fit["season"].to_numpy(np.float64)) / spec.half_life
    )
    model = _model(spec, seed)
    model.fit(
        fit_x,
        fit["control_success"].to_numpy(np.float64),
        sample_weight=weight,
        categorical_feature=categorical,
    )
    if spec.family == "binary":
        prediction = model.predict_proba(audit_x)[:, 1]
    else:
        prediction = model.predict(audit_x)
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0.001, 0.999)
    del model, fit_x, audit_x
    gc.collect()
    return prediction


def _candidate(parent: np.ndarray, direct: np.ndarray, eta: float) -> np.ndarray:
    return np.clip(parent + eta * (direct - parent), 0.001, 0.999)


def _diagnostics(
    frame: pd.DataFrame, direct: np.ndarray, eta: float
) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)
    parent = frame["v22"].to_numpy(np.float64)
    candidate = _candidate(parent, direct, eta)
    months = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        months.append(
            {
                "month": int(month),
                "rows": int(mask.sum()),
                "gain": _bss(target[mask], candidate[mask])
                - _bss(target[mask], parent[mask]),
            }
        )
    domains = []
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        mask = frame["domain3"].eq(domain).to_numpy()
        domains.append(
            {
                "domain": domain,
                "rows": int(mask.sum()),
                "gain": _bss(target[mask], candidate[mask])
                - _bss(target[mask], parent[mask]),
            }
        )
    return {
        "gain": _bss(target, candidate) - _bss(target, parent),
        "positive_month_fraction": float(np.mean([row["gain"] > 0 for row in months])),
        "worst_month_gain": float(min(row["gain"] for row in months)),
        "minimum_domain_gain": float(min(row["gain"] for row in domains)),
        "mean_shift": float(np.mean(candidate - parent)),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
        "months": months,
        "domains": domains,
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    selection = _load_axis(project, "y2023_early_to_late", raw)
    outer = _load_axis(project, "y2023_to_y2024", raw)

    # Fit frames contain labels but never audit-year rows.
    selection_fit = raw.loc[raw["season"].le(2022)].reset_index(drop=True)
    selection_fit = _derived(selection_fit, _joint_domain(selection_fit))
    outer_fit = raw.loc[raw["season"].le(2023)].reset_index(drop=True)
    outer_fit = _derived(outer_fit, _joint_domain(outer_fit))
    del raw

    rows: list[dict[str, object]] = []
    direct_predictions: dict[str, np.ndarray] = {}
    for index, spec in enumerate(DIRECT_SPECS):
        direct = _fit_predict(selection_fit, selection, spec, 12100 + index)
        direct_predictions[spec.name] = direct
        for eta in ETAS:
            diagnostics = _diagnostics(selection, direct, eta)
            rows.append(
                {
                    **asdict(spec),
                    "eta": eta,
                    "gain": diagnostics["gain"],
                    "positive_month_fraction": diagnostics["positive_month_fraction"],
                    "worst_month_gain": diagnostics["worst_month_gain"],
                    "minimum_domain_gain": diagnostics["minimum_domain_gain"],
                    "selection_score": min(
                        diagnostics["gain"],
                        diagnostics["worst_month_gain"],
                        diagnostics["minimum_domain_gain"],
                    ),
                }
            )
        print(f"[selection] {spec.name}", flush=True)

    metrics = pd.DataFrame(rows).sort_values(
        ["selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    selected_row = metrics.iloc[0]
    selected = next(spec for spec in DIRECT_SPECS if spec.name == selected_row["name"])
    eta = float(selected_row["eta"])
    selection_diagnostics = _diagnostics(
        selection, direct_predictions[selected.name], eta
    )
    del direct_predictions, selection_fit

    outer_direct = _fit_predict(outer_fit, outer, selected, 12991)
    outer_diagnostics = _diagnostics(outer, outer_direct, eta)
    gate = {
        "selection_gain_positive": bool(selection_diagnostics["gain"] > 0.0),
        "selection_all_months_positive": bool(selection_diagnostics["worst_month_gain"] > 0.0),
        "selection_all_domains_positive": bool(selection_diagnostics["minimum_domain_gain"] > 0.0),
        "outer_gain_at_least_5": bool(outer_diagnostics["gain"] >= 5.0),
        "outer_positive_month_fraction_at_least_075": bool(
            outer_diagnostics["positive_month_fraction"] >= 0.75
        ),
        "outer_minimum_domain_positive": bool(outer_diagnostics["minimum_domain_gain"] > 0.0),
        "outer_worst_month_above_minus_10": bool(outer_diagnostics["worst_month_gain"] > -10.0),
    }
    eligible = bool(all(gate.values()))
    np.savez_compressed(
        output_dir / "outer_selected_prediction.npz",
        target=outer["target"].to_numpy(np.float64),
        v21=outer["v21"].to_numpy(np.float64),
        v22=outer["v22"].to_numpy(np.float64),
        direct=outer_direct,
        candidate=_candidate(outer["v22"].to_numpy(np.float64), outer_direct, eta),
        game_month=outer["game_month"].to_numpy(np.int16),
        domain3=outer["domain3"].astype(str).to_numpy(),
        pitcher_id=outer["pitcher_id"].to_numpy(),
        batter_id=outer["batter_id"].to_numpy(),
    )
    summary = {
        "protocol": "V23_MULTIYEAR_DIRECT_NESTED_V1",
        "selection": "fit 2019-2022; choose model and eta on honest late-2023 v22 OOF",
        "outer_audit": "frozen choice refit 2019-2023; score once on full-2024 v22 OOF",
        "model_count": len(DIRECT_SPECS),
        "candidate_count": int(len(metrics)),
        "selected": {key: selected_row[key] for key in metrics.columns},
        "selection_diagnostics": selection_diagnostics,
        "outer_diagnostics": outer_diagnostics,
        "gate": gate,
        "eligible_for_2025_refit": eligible,
    }
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
        default=Path("artifacts/v23_multiyear_direct_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
