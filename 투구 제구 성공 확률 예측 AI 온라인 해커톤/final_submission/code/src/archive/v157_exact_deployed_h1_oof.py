"""Rebuild the exact deployed H1 ensemble as strict forward OOF.

The v148 package contains three depth-8 CatBoost pipelines, but its local OOF
contract was assembled from a depth-6 single-seed proxy.  This audit freezes
the packaged feature order and hyperparameters, refits only on official rows
strictly preceding each audit year, and measures the resulting contract delta.
It never reads test rows, test aggregates, or a leaderboard score.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.champion.v131_catboost_h1_independent_oof import _fit_year, _prepare_features
from src.core.axis_metrics import _axis_metrics
from src.core.contract import _load_contract_axis
from src.core.robustness import evaluate_robustness


PROTOCOL = "V157_EXACT_DEPLOYED_H1_OOF_V1"
SOURCE_AXES = ("full_2022", "late_2023")


def _model_recipe(model: Any) -> dict[str, Any]:
    params = model.named_steps["clf"].get_params()
    return {
        "iterations": int(params["iterations"]),
        "learning_rate": float(params["learning_rate"]),
        "depth": int(params["depth"]),
        "l2_leaf_reg": float(params["l2_leaf_reg"]),
        "border_count": int(params["border_count"]),
        "seed": int(params["random_seed"]),
        "thread_count": int(params["thread_count"]),
    }


def _validate_deployed_bundle(
    bundle_path: Path,
    h1_features: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    bundle = joblib.load(bundle_path)
    packaged_features = list(bundle.get("features", []))
    if packaged_features != h1_features:
        missing = [value for value in packaged_features if value not in h1_features]
        extra = [value for value in h1_features if value not in packaged_features]
        raise ValueError(
            "deployed/local feature-order mismatch: "
            f"packaged={len(packaged_features)} local={len(h1_features)} "
            f"missing={missing} extra={extra}"
        )
    models = list(bundle.get("models", []))
    expected = config["model"]
    if len(models) != len(expected["seeds"]):
        raise ValueError("deployed model-count mismatch")
    actual_recipes = [_model_recipe(model) for model in models]
    expected_recipes = [
        {
            "iterations": int(expected["iterations"]),
            "learning_rate": float(expected["learning_rate"]),
            "depth": int(expected["depth"]),
            "l2_leaf_reg": float(expected["l2_leaf_reg"]),
            "border_count": int(expected["border_count"]),
            "seed": int(seed),
            "thread_count": int(expected["thread_count"]),
        }
        for seed in expected["seeds"]
    ]
    if actual_recipes != expected_recipes:
        raise ValueError(
            f"deployed/config model mismatch: actual={actual_recipes} "
            f"expected={expected_recipes}"
        )
    if float(bundle.get("alpha", np.nan)) != 1.0:
        raise ValueError("deployed H1 affine alpha is not identity")
    if float(bundle.get("center", np.nan)) != 0.5:
        raise ValueError("deployed H1 affine center is not identity")
    return {
        "features": len(packaged_features),
        "models": len(models),
        "recipes": actual_recipes,
        "alpha": float(bundle["alpha"]),
        "center": float(bundle["center"]),
    }


def _exact_ensemble_oof(
    train: Any,
    target: np.ndarray,
    season: np.ndarray,
    year: int,
    features: list[str],
    model_config: dict[str, Any],
    checkpoint_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = []
    recipes = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in model_config["seeds"]:
        recipe = {
            "iterations": model_config["iterations"],
            "learning_rate": model_config["learning_rate"],
            "depth": model_config["depth"],
            "l2_leaf_reg": model_config["l2_leaf_reg"],
            "border_count": model_config["border_count"],
            "seed": int(seed),
        }
        checkpoint = checkpoint_dir / f"h1_year{year}_seed{seed}.npy"
        if checkpoint.exists():
            prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            expected_rows = int(np.sum(season == year))
            if prediction.shape != (expected_rows,):
                raise ValueError(f"invalid checkpoint shape: {checkpoint}")
            print(f"[v157] resumed checkpoint {checkpoint.name}", flush=True)
        else:
            prediction = _fit_year(
                train,
                target,
                season,
                year,
                features,
                recipe,
                f"exact-H1-seed{seed}",
            )
            np.save(checkpoint, prediction, allow_pickle=False)
            print(f"[v157] saved checkpoint {checkpoint.name}", flush=True)
        predictions.append(prediction)
        recipes.append(recipe)
    return np.mean(np.column_stack(predictions), axis=1), recipes


def _replace_proxy_h1(
    base: np.ndarray,
    axis: dict[str, np.ndarray],
    exact_h1: np.ndarray,
    proxy_h1: np.ndarray,
    weight: float,
    route: str,
) -> np.ndarray:
    if not (len(base) == len(exact_h1) == len(proxy_h1)):
        raise ValueError("H1 replacement length mismatch")
    active = axis["exact_mask"].astype(bool)
    active &= axis["domain3"].astype(str) == route
    output = np.asarray(base, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active]
        + float(weight) * (exact_h1[active] - proxy_h1[active]),
        0.001,
        0.999,
    )
    return output


def _metric_against_base(
    axis: dict[str, np.ndarray], base: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    return _axis_metrics({**axis, "parent": base}, candidate)


def run(
    train_csv: Path,
    trackman_csv: Path,
    component_root: Path,
    deployed_bundle: Path,
    contract_dir: Path,
    proxy_dir: Path,
    v148_oof: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train, target, season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, component_root
    )
    bundle_audit = _validate_deployed_bundle(
        deployed_bundle, h1_features, config
    )
    print(
        f"[v157] deployed recipe verified: {bundle_audit['models']} models, "
        f"{bundle_audit['features']} features, identity affine",
        flush=True,
    )

    exact: dict[int, np.ndarray] = {}
    fold_recipes: dict[str, Any] = {}
    for year_value in config["audit_years"]:
        year = int(year_value)
        print(f"[v157] fitting exact deployed H1 forward fold {year}", flush=True)
        exact[year], fold_recipes[str(year)] = _exact_ensemble_oof(
            train,
            target,
            season,
            year,
            h1_features,
            config["model"],
            output_dir / "checkpoints",
        )
    del train
    gc.collect()

    with np.load(proxy_dir / "oof_predictions.npz", allow_pickle=False) as saved:
        proxy = {
            year: saved[f"h1_{year}"].astype(np.float64)
            for year in exact
        }
    with np.load(proxy_dir / "selected_axes.npz", allow_pickle=False) as saved:
        source_base = {
            name: saved[name].astype(np.float64) for name in SOURCE_AXES
        }

    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in SOURCE_AXES
    }
    year23_month = season[season == 2023]
    del year23_month
    late23 = axes["late_2023"]["raw_index"].astype(np.int64)
    full23_index = np.flatnonzero(season == 2023)
    position = np.searchsorted(full23_index, late23)
    if not np.array_equal(full23_index[position], late23):
        raise ValueError("late-2023 raw-index alignment mismatch")

    exact_source = {
        "full_2022": exact[2022],
        "late_2023": exact[2023][position],
    }
    proxy_source = {
        "full_2022": proxy[2022],
        "late_2023": proxy[2023][position],
    }
    weight = float(config["deployed_h1_weight"])
    route = str(config["deployed_route"])
    source_candidate = {
        name: _replace_proxy_h1(
            source_base[name], axes[name], exact_source[name], proxy_source[name],
            weight, route,
        )
        for name in SOURCE_AXES
    }
    source_metrics = {
        name: _metric_against_base(
            axes[name], source_base[name], source_candidate[name]
        )
        for name in SOURCE_AXES
    }

    axis148 = _load_contract_axis(v148_oof)
    expected24 = np.flatnonzero(season == 2024)
    if not np.array_equal(axis148["raw_index"].astype(np.int64), expected24):
        raise ValueError("v148/2024 raw-index alignment mismatch")
    candidate148 = _replace_proxy_h1(
        axis148["parent"].astype(np.float64),
        axis148,
        exact[2024],
        proxy[2024],
        weight,
        route,
    )
    metrics148 = _metric_against_base(
        axis148, axis148["parent"].astype(np.float64), candidate148
    )
    robust = evaluate_robustness(
        axis148,
        candidate148,
        [candidate148, axis148["parent"].astype(np.float64)],
        config["robustness"],
    )
    comparison = {
        str(year): {
            "correlation": float(np.corrcoef(exact[year], proxy[year])[0, 1]),
            "mean_exact_minus_proxy": float(np.mean(exact[year] - proxy[year])),
            "mae_exact_minus_proxy": float(np.mean(np.abs(exact[year] - proxy[year]))),
            "rmse_exact_minus_proxy": float(
                np.sqrt(np.mean(np.square(exact[year] - proxy[year])))
            ),
        }
        for year in exact
    }

    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        **{f"exact_h1_{year}": value for year, value in exact.items()},
        **{f"proxy_h1_{year}": value for year, value in proxy.items()},
        v148_exact_h1_candidate=candidate148,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "contract_parity_measured",
        "bundle_audit": bundle_audit,
        "fold_recipes": fold_recipes,
        "source_incremental_metrics": source_metrics,
        "v148_incremental_metrics": metrics148,
        "v148_robustness": robust,
        "exact_vs_proxy": comparison,
        "eligible_for_packaging": False,
        "interpretation": (
            "This is a deployed-contract parity audit, not a new submission. "
            "The current v148 package already runs the exact H1 ensemble."
        ),
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
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--deployed-bundle", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--v148-oof", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.trackman_csv,
        args.component_root,
        args.deployed_bundle,
        args.contract_dir,
        args.proxy_dir,
        args.v148_oof,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
