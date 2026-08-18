"""Nested structural residual screen above the frozen v22 recipe.

The experiment deliberately separates selection from the final audit:

* honest v21 OOF rows from late 2023 are converted to the frozen v22 recipe;
* August -> September and August-September -> October select one model/eta;
* that choice is frozen, refit on all late-2023 rows, and audited once on 2024;
* only an outer-audit pass may be refit on honest 2024 OOF for 2025 deployment.

All features are available on the current row.  No statistic is computed from
test rows, and audit labels are never used for selection or fitting.
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

from src.v20_residual_overlay_screen import _bss


CHAMPION_DIR = Path("artifacts/champion_oof_20260817_01")
ETAS = (0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
CATEGORICAL_BASE = (
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
IDENTITY_COLUMNS = ("pitcher_id", "batter_id")
EXCLUDED = {
    "row_id",
    "season",
    "control_success",
    "target",
    "v21",
    "v22",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    include_ids: bool
    leaves: int
    min_child_samples: int


MODEL_SPECS = (
    ModelSpec("context_l3", False, 3, 1600),
    ModelSpec("context_l7", False, 7, 1200),
    ModelSpec("identity_l3", True, 3, 1600),
    ModelSpec("identity_l7", True, 7, 1200),
)


def apply_v22_recipe(
    v21: np.ndarray,
    domain: np.ndarray,
    pitcher_rate: np.ndarray,
    batter_rate: np.ndarray,
) -> np.ndarray:
    """Apply the exact balanced v22 correction used in the submitted ZIP."""
    parent = np.asarray(v21, dtype=np.float64)
    domain = np.asarray(domain).astype(str)
    pitcher = np.nan_to_num(np.asarray(pitcher_rate, dtype=np.float64), nan=0.5)
    batter = np.nan_to_num(np.asarray(batter_rate, dtype=np.float64), nan=0.5)
    correction = np.zeros(len(parent), dtype=np.float64)
    parameters = {
        "R_CORE": (0.44, 0.035),
        "R_ANCHOR": (0.48, 0.020),
        "F": (0.52, 0.020),
    }
    for name, (anchor, weight) in parameters.items():
        selected = domain == name
        correction[selected] += weight * (anchor - parent[selected])
    prior = 0.75 * pitcher + 0.25 * batter
    correction += 0.050 * (prior - parent)
    return np.clip(parent + correction, 0.001, 0.999)


def _derived(frame: pd.DataFrame, domain: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["domain3"] = np.asarray(domain).astype(str)
    output["count_state"] = (
        output["balls_before"].astype("Int64").astype(str)
        + "-"
        + output["strikes_before"].astype("Int64").astype(str)
    )
    output["hand_matchup"] = (
        output["pitcher_hand"].astype("string").fillna("__MISSING__")
        + "-"
        + output["batter_hand"].astype("string").fillna("__MISSING__")
    )
    output["inning_bucket"] = pd.cut(
        pd.to_numeric(output["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, np.inf),
        labels=("early", "middle", "late"),
    ).astype("string")
    for column in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        output[f"log1p_{column}"] = np.log1p(
            pd.to_numeric(output[column], errors="coerce").clip(lower=0.0)
        )
    output["recent_success_delta_1_5"] = (
        pd.to_numeric(output["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(output["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    )
    output["recent_middle_delta_1_5"] = (
        pd.to_numeric(output["asof_pitcher_prev1_game_middle_rate"], errors="coerce")
        - pd.to_numeric(output["asof_pitcher_prev5_game_middle_rate"], errors="coerce")
    )
    output["pitcher_batter_rate_gap"] = (
        pd.to_numeric(output["asof_pitcher_success_rate"], errors="coerce")
        - pd.to_numeric(output["asof_batter_success_rate"], errors="coerce")
    )
    return output


def _feature_columns(frame: pd.DataFrame, include_ids: bool) -> list[str]:
    columns = [column for column in frame.columns if column not in EXCLUDED]
    if not include_ids:
        columns = [column for column in columns if column not in IDENTITY_COLUMNS]
    return columns


def _frames(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    columns: list[str],
    include_ids: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fit_x = fit.loc[:, columns].copy()
    audit_x = audit.loc[:, columns].copy()
    categorical = list(CATEGORICAL_BASE)
    if include_ids:
        categorical.extend(IDENTITY_COLUMNS)
    categorical = [column for column in categorical if column in columns]
    for column in categorical:
        source = fit_x[column].astype("string").fillna("__MISSING__")
        categories = pd.Index(source.unique())
        fit_x[column] = pd.Categorical(source, categories=categories)
        audit_x[column] = pd.Categorical(
            audit_x[column].astype("string").fillna("__MISSING__"),
            categories=categories,
        )
    for column in columns:
        if column not in categorical:
            fit_x[column] = pd.to_numeric(fit_x[column], errors="coerce")
            audit_x[column] = pd.to_numeric(audit_x[column], errors="coerce")
    return fit_x, audit_x, categorical


def _model(spec: ModelSpec, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        verbosity=-1,
        n_jobs=6,
        n_estimators=180 if spec.leaves == 3 else 240,
        learning_rate=0.02,
        num_leaves=spec.leaves,
        max_depth=2 if spec.leaves == 3 else 3,
        min_child_samples=spec.min_child_samples,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=5.0,
        reg_lambda=40.0,
        max_bin=127,
        random_state=seed,
    )


def _fit_predict(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    spec: ModelSpec,
    seed: int,
) -> np.ndarray:
    columns = _feature_columns(fit, spec.include_ids)
    fit_x, audit_x, categorical = _frames(
        fit, audit, columns, spec.include_ids
    )
    model = _model(spec, seed)
    model.fit(
        fit_x,
        fit["target"].to_numpy(np.float64)
        - fit["v22"].to_numpy(np.float64),
        categorical_feature=categorical,
    )
    prediction = np.asarray(model.predict(audit_x), dtype=np.float64)
    del model, fit_x, audit_x
    gc.collect()
    return np.clip(prediction, -0.08, 0.08)


def _gain(target: np.ndarray, parent: np.ndarray, raw: np.ndarray, eta: float) -> float:
    candidate = np.clip(parent + eta * raw, 0.001, 0.999)
    return _bss(target, candidate) - _bss(target, parent)


def _diagnostics(frame: pd.DataFrame, raw: np.ndarray, eta: float) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)
    parent = frame["v22"].to_numpy(np.float64)
    candidate = np.clip(parent + eta * raw, 0.001, 0.999)
    months = []
    for month in sorted(frame["game_month"].unique()):
        selected = frame["game_month"].eq(month).to_numpy()
        months.append(
            {
                "month": int(month),
                "rows": int(selected.sum()),
                "gain": _bss(target[selected], candidate[selected])
                - _bss(target[selected], parent[selected]),
            }
        )
    domains = []
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        selected = frame["domain3"].eq(domain).to_numpy()
        domains.append(
            {
                "domain": domain,
                "rows": int(selected.sum()),
                "gain": _bss(target[selected], candidate[selected])
                - _bss(target[selected], parent[selected]),
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


def _load_axis(project: Path, axis: str, raw: pd.DataFrame) -> pd.DataFrame:
    with np.load(project / CHAMPION_DIR / f"{axis}.npz", allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        v21 = saved["v21"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    if axis == "y2023_early_to_late":
        frame = raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True)
    elif axis == "y2023_to_y2024":
        frame = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    else:
        raise ValueError(axis)
    if not np.array_equal(target, frame["control_success"].to_numpy(np.float64)):
        raise ValueError(f"target order mismatch for {axis}")
    frame = _derived(frame, domain)
    frame["target"] = target
    frame["v21"] = v21
    frame["v22"] = apply_v22_recipe(
        v21,
        domain,
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce").to_numpy(),
        pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce").to_numpy(),
    )
    return frame


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    source = _load_axis(project, "y2023_early_to_late", raw)
    outer = _load_axis(project, "y2023_to_y2024", raw)
    del raw

    inner_splits = {
        "aug_to_sep": (
            source["game_month"].eq(8).to_numpy(),
            source["game_month"].eq(9).to_numpy(),
        ),
        "aug_sep_to_oct": (
            source["game_month"].isin((8, 9)).to_numpy(),
            source["game_month"].eq(10).to_numpy(),
        ),
    }
    inner_rows: list[dict[str, object]] = []
    for spec_index, spec in enumerate(MODEL_SPECS):
        raw_by_split = {}
        for split_index, (name, (fit_mask, audit_mask)) in enumerate(inner_splits.items()):
            fit = source.loc[fit_mask].reset_index(drop=True)
            audit = source.loc[audit_mask].reset_index(drop=True)
            raw_prediction = _fit_predict(
                fit, audit, spec, 9100 + 100 * spec_index + split_index
            )
            raw_by_split[name] = (audit, raw_prediction)
        for eta in ETAS:
            gains = {
                name: _gain(
                    audit["target"].to_numpy(np.float64),
                    audit["v22"].to_numpy(np.float64),
                    raw_prediction,
                    eta,
                )
                for name, (audit, raw_prediction) in raw_by_split.items()
            }
            inner_rows.append(
                {
                    **asdict(spec),
                    "eta": eta,
                    **gains,
                    "min_inner_gain": min(gains.values()),
                    "mean_inner_gain": float(np.mean(list(gains.values()))),
                }
            )
        print(f"[inner] {spec.name}", flush=True)

    inner = pd.DataFrame(inner_rows).sort_values(
        ["min_inner_gain", "mean_inner_gain"], ascending=False
    ).reset_index(drop=True)
    inner.to_csv(output_dir / "inner_metrics.csv", index=False)
    chosen_row = inner.iloc[0]
    chosen = next(spec for spec in MODEL_SPECS if spec.name == chosen_row["name"])
    eta = float(chosen_row["eta"])

    outer_raw = _fit_predict(source, outer, chosen, 9901)
    diagnostics = _diagnostics(outer, outer_raw, eta)
    gate = {
        "positive_inner_axes": bool(float(chosen_row["min_inner_gain"]) > 0.0),
        "outer_gain_at_least_5": bool(diagnostics["gain"] >= 5.0),
        "outer_positive_month_fraction_at_least_075": bool(
            diagnostics["positive_month_fraction"] >= 0.75
        ),
        "outer_minimum_domain_positive": bool(diagnostics["minimum_domain_gain"] > 0.0),
        "outer_worst_month_above_minus_10": bool(diagnostics["worst_month_gain"] > -10.0),
    }
    eligible = bool(all(gate.values()))
    np.savez_compressed(
        output_dir / "outer_selected_prediction.npz",
        target=outer["target"].to_numpy(np.float64),
        v21=outer["v21"].to_numpy(np.float64),
        v22=outer["v22"].to_numpy(np.float64),
        raw_correction=outer_raw,
        candidate=np.clip(
            outer["v22"].to_numpy(np.float64) + eta * outer_raw,
            0.001,
            0.999,
        ),
        game_month=outer["game_month"].to_numpy(np.int16),
        domain3=outer["domain3"].astype(str).to_numpy(),
        pitcher_id=outer["pitcher_id"].to_numpy(),
        batter_id=outer["batter_id"].to_numpy(),
    )
    summary = {
        "protocol": "V23_NESTED_STRUCTURAL_RESIDUAL_V1",
        "selection_rows": "honest v22 OOF, 2023 August-October only",
        "outer_audit": "frozen selection refit on late 2023, scored once on full 2024",
        "model_count": len(MODEL_SPECS),
        "candidate_count": int(len(inner)),
        "selected": {key: chosen_row[key] for key in inner.columns},
        "outer_diagnostics": diagnostics,
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
        default=Path("artifacts/v23_structural_residual_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
