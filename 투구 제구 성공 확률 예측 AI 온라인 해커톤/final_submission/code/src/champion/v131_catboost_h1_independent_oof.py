"""Retrain the published DX/H1 CatBoost recipe and blend its honest OOF.

The external repository contributes only public feature definitions and fixed
hyperparameters.  Every fitted value and every residual table in this audit is
rebuilt from the official train.csv rows strictly preceding the audit season.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.core.axis_metrics import _axis_metrics
from src.champion.v127_catboost_current_state_rebase import apply_correction
from src.champion.v130_catboost_independent_oof_blend import apply_blend, post4
from src.core.contract import _load_contract_axis


PROTOCOL = "V131_CATBOOST_H1_INDEPENDENT_OOF_V1"
SOURCE_AXES = ("full_2022", "late_2023")
CAT_COLS = ("top_bottom", "game_type", "base_state")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prior_tables(df: pd.DataFrame, mask: np.ndarray, sc: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for kind, (ncol, idcol) in sc.ASOF_NCOL.items():
        ids = df.loc[mask, idcol].to_numpy(np.int64)
        count_now = df.loc[mask, ncol].to_numpy(np.float64)
        unique, counts = np.unique(ids, return_counts=True)
        rate_cols = [rate for rate, _, _, group in sc.ASOF_SPEC if group == kind]
        order = np.argsort(ids, kind="stable")
        keys, starts = np.unique(ids[order], return_index=True)
        events = []
        for rate_col in rate_cols:
            totals = (
                count_now
                * np.nan_to_num(df.loc[mask, rate_col].to_numpy(np.float64))
            )[order]
            events.append(np.maximum.reduceat(totals, starts))
        output[kind] = {
            int(key): tuple([float(n)] + [float(event[i]) for event in events])
            for i, (key, n) in enumerate(zip(unique, counts))
        }
        if not np.array_equal(unique, keys):
            raise ValueError(f"prior key order mismatch: {kind}")
    return output


def _prepare_features(
    train_csv: Path,
    trackman_csv: Path,
    component_root: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    ft = _load_module("v131_h1_final_train", component_root / "final_train.py")
    sc = _load_module("v131_h1_script", component_root / "script.py")
    ft.DATA_DIR = str(train_csv.parent)
    ft.TM_PATH = str(trackman_csv)
    ft.ID_MAP_PATH = str(component_root / "pitcher_id_map.csv")

    header = pd.read_csv(train_csv, nrows=0).columns.tolist()
    base_features = [
        column for column in header if column not in ("row_id", "control_success")
    ]
    print(f"[v131] loading official train ({len(base_features)} base features)", flush=True)
    train = pd.read_csv(
        train_csv,
        usecols=base_features + ["control_success"],
        low_memory=False,
    )
    print("[v131] attaching prior-season TrackMan context", flush=True)
    train = ft.attach_ctx_train(train, ft.load_trackman())
    context_features = [
        column
        for column in list(ft.COUNT_FEATS) + list(ft.HAND_FEATS)
        if column not in ("tmc_n", "tmh_n")
    ]
    state_features = list(sc.ASOF_COLS) + list(sc.CTX_COLS) + list(sc.LVL_COLS)
    for column in state_features:
        train[column] = np.nan

    season = train["season"].to_numpy(np.int16)
    for year in sorted(np.unique(season)):
        print(f"[v131] reconstructing current-season state: {year}", flush=True)
        audit = season == year
        tables = _prior_tables(train, season < year, sc)
        part = sc.attach_asof_state(
            train.loc[audit].copy(), {"asof_prior": tables, "features": state_features}
        )
        for column in state_features:
            train.loc[audit, column] = part[column].to_numpy()
        del part, tables
        gc.collect()

    dx_features = base_features + context_features + list(sc.ASOF_COLS) + list(sc.CTX_COLS)
    h1_features = dx_features + list(sc.LVL_COLS)
    numeric = [column for column in h1_features if column not in CAT_COLS]
    train[numeric] = train[numeric].astype(np.float32)
    target = train["control_success"].to_numpy(np.float64)
    print(
        f"[v131] prepared rows={len(train):,}, DX={len(dx_features)}p, "
        f"H1={len(h1_features)}p",
        flush=True,
    )
    return train, target, season, base_features, dx_features, h1_features


def _pipeline(features: list[str], config: dict[str, Any]) -> Pipeline:
    categorical = [column for column in CAT_COLS if column in features]
    numeric = [column for column in features if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("num", "passthrough", numeric),
        ]
    )
    model = CatBoostClassifier(
        iterations=int(config["iterations"]),
        learning_rate=float(config["learning_rate"]),
        depth=int(config["depth"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        border_count=int(config["border_count"]),
        loss_function="Logloss",
        random_seed=int(config["seed"]),
        verbose=0,
        thread_count=16,
        allow_writing_files=False,
    )
    return Pipeline([("pre", preprocessor), ("clf", model)])


def _fit_year(
    train: pd.DataFrame,
    target: np.ndarray,
    season: np.ndarray,
    year: int,
    features: list[str],
    model_config: dict[str, Any],
    label: str,
) -> np.ndarray:
    fit = season < year
    audit = season == year
    started = time.time()
    model = _pipeline(features, model_config)
    model.fit(train.loc[fit, features], target[fit].astype(np.int8))
    prediction = model.predict_proba(train.loc[audit, features])[:, 1].astype(np.float64)
    print(
        f"[v131] {label} fold={year} fit={int(fit.sum()):,} "
        f"audit={int(audit.sum()):,} elapsed={time.time()-started:.1f}s",
        flush=True,
    )
    del model
    gc.collect()
    return prediction


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any], *, locked: bool) -> bool:
    minimum_gain = float(gate.get("gain_min", 0.0)) if locked else 0.0
    return bool(
        metrics["gain"] >= minimum_gain
        and metrics["positive_month_fraction"]
        >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"]
        > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def _candidate(
    parent: np.ndarray,
    axis: dict[str, np.ndarray],
    family: str,
    route: str,
    dose: float,
    dx: np.ndarray,
    h1: np.ndarray,
    h1_post: np.ndarray,
) -> np.ndarray:
    if family == "h1_raw_blend":
        return apply_blend(parent, axis, h1, route, dose)
    if family == "h1_post4_blend":
        return apply_blend(parent, axis, h1_post, route, dose)
    if family == "h1_minus_dx_correction":
        return apply_correction(parent, axis, h1 - dx, route, dose)
    raise ValueError(f"unknown family: {family}")


def run(
    train_csv: Path,
    trackman_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    component_root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base, dx_features, h1_features = _prepare_features(
        train_csv, trackman_csv, component_root
    )

    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in (*SOURCE_AXES, "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        parent = {name: saved[name].astype(np.float64) for name in saved.files}
    metric_axes = {name: {**axis, "parent": parent[name]} for name, axis in axes.items()}
    year_frames = {
        year: train.loc[season == year].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23 = year_frames[2023]["game_month"].ge(8).to_numpy()

    oof: dict[str, np.ndarray] = {}
    for year in (2022, 2023):
        oof[f"dx_{year}"] = _fit_year(
            train, target, season, year, dx_features, config["model"], "DX"
        )
        oof[f"h1_{year}"] = _fit_year(
            train, target, season, year, h1_features, config["model"], "H1"
        )
    post = {
        year: post4(
            train.loc[season < year].reset_index(drop=True), year_frames[year]
        )
        for year in (2022, 2023)
    }
    external_source = {
        "full_2022": {
            "dx": oof["dx_2022"],
            "h1": oof["h1_2022"],
            "h1_post": oof["h1_2022"] + post[2022],
        },
        "late_2023": {
            "dx": oof["dx_2023"][late23],
            "h1": oof["h1_2023"][late23],
            "h1_post": (oof["h1_2023"] + post[2023])[late23],
        },
    }

    trials: list[dict[str, Any]] = []
    trial_metrics: dict[str, dict[str, Any]] = {}
    trial_candidates: dict[str, dict[str, np.ndarray]] = {}
    for family in config["families"]:
        doses = (
            config["correction_eta_grid"]
            if family == "h1_minus_dx_correction"
            else config["blend_weight_grid"]
        )
        for route in config["routes"]:
            for dose in doses:
                candidates = {
                    name: _candidate(
                        parent[name], axes[name], family, route, float(dose),
                        external_source[name]["dx"], external_source[name]["h1"],
                        external_source[name]["h1_post"],
                    )
                    for name in SOURCE_AXES
                }
                metrics = {
                    name: _axis_metrics(metric_axes[name], candidates[name])
                    for name in SOURCE_AXES
                }
                passed = all(
                    _point_pass(item, config["source_gate"], locked=False)
                    for item in metrics.values()
                )
                key = f"{family}__{route}__d{float(dose):g}"
                trials.append(
                    {
                        "key": key,
                        "family": family,
                        "route": route,
                        "dose": float(dose),
                        "source_gate_passed": bool(passed),
                        "minimum_gain": float(min(x["gain"] for x in metrics.values())),
                        "mean_gain": float(np.mean([x["gain"] for x in metrics.values()])),
                        "minimum_month_fraction": float(
                            min(x["positive_month_fraction"] for x in metrics.values())
                        ),
                        "worst_month_gain": float(
                            min(x["worst_month_gain"] for x in metrics.values())
                        ),
                    }
                )
                trial_metrics[key] = metrics
                trial_candidates[key] = candidates

    ranking = pd.DataFrame(trials).sort_values(
        ["source_gate_passed", "minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    selected_key = str(selected["key"])
    family = str(selected["family"])
    route = str(selected["route"])
    dose = float(selected["dose"])
    print(f"[v131] source selected: {selected_key}", flush=True)

    oof["h1_2024"] = _fit_year(
        train, target, season, 2024, h1_features, config["model"], "H1"
    )
    if family == "h1_minus_dx_correction":
        oof["dx_2024"] = _fit_year(
            train, target, season, 2024, dx_features, config["model"], "DX"
        )
    else:
        oof["dx_2024"] = np.zeros_like(oof["h1_2024"])
    post24 = post4(train.loc[season < 2024].reset_index(drop=True), year_frames[2024])
    candidate24 = _candidate(
        parent["full_2024"], axes["full_2024"], family, route, dose,
        oof["dx_2024"], oof["h1_2024"], oof["h1_2024"] + post24,
    )
    late24 = year_frames[2024]["game_month"].ge(8).to_numpy()
    locked_metrics = {
        "full_2024": _axis_metrics(metric_axes["full_2024"], candidate24),
        "late_2024": _axis_metrics(
            _slice_axis(metric_axes["full_2024"], late24), candidate24[late24]
        ),
    }
    locked_pass = {
        name: _point_pass(item, config["locked_gate"], locked=True)
        for name, item in locked_metrics.items()
    }
    eligible = bool(selected["source_gate_passed"] and all(locked_pass.values()))

    np.savez_compressed(output_dir / "oof_predictions.npz", **oof)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=trial_candidates[selected_key]["full_2022"],
        late_2023=trial_candidates[selected_key]["late_2023"],
        full_2024=candidate24,
        late_2024=candidate24[late24],
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if eligible else "reject",
        "external_material_used": "public feature definitions and fixed hyperparameters only",
        "model_recipe": config["model"],
        "n_trials": int(len(ranking)),
        "selected": selected,
        "source_ranking_top30": ranking.head(30).to_dict(orient="records"),
        "selected_source_metrics": trial_metrics[selected_key],
        "locked_metrics": locked_metrics,
        "locked_point_gate_pass": locked_pass,
        "eligible_for_robust_audit": eligible,
        "eligible_for_packaging": False,
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.trackman_csv,
        args.contract_dir,
        args.v104_dir,
        args.component_root,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
