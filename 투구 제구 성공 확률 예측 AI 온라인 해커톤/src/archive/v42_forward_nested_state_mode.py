"""Fully forward-nested state/failure-mode route selection.

The original v19 route was intentionally a development upper bound: its
state filters and multipliers were chosen jointly on 2023 and 2024.  This
module removes that optimism.  The 2023 recipe is selected on 2022 only and
the 2024 recipe is selected on 2022+2023 only.  Each recipe is frozen before
its audit year is evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.multi_year_state_ensemble_screen import DOMAINS, _load_fold
from src.archive.state_mode_joint_screen import (
    MODE_MULTIPLIERS,
    STATE_MULTIPLIERS,
    _candidate,
    _diagnostics,
    _mode_fold,
    _state_correction,
)


def top_configs(
    metrics: pd.DataFrame,
    domain: str,
    selection_years: tuple[int, ...],
    top_n: int,
) -> list[str]:
    """Rank state routes using selection years and no later labels."""
    source = metrics.loc[
        metrics["audit_year"].isin(selection_years)
        & metrics["domain"].eq(domain)
    ]
    if source.empty:
        raise ValueError(
            f"no state metrics for domain={domain} years={selection_years}"
        )
    robust = (
        source.groupby("config", observed=True)["gain"]
        .agg(min_gain="min", mean_gain="mean")
        .reset_index()
        .sort_values(
            ["min_gain", "mean_gain", "config"],
            ascending=[False, False, True],
            kind="stable",
        )
    )
    return robust.head(top_n)["config"].astype(str).tolist()


def search_domain(
    domain: str,
    configs: list[str],
    state_folds: dict[int, dict[str, object]],
    mode_folds: dict[int, dict[str, object]],
    selection_years: tuple[int, ...],
) -> dict[str, object]:
    """Choose one joint route by maximin gain over prior origins."""
    legal_names = sorted(
        set.intersection(
            *(set(mode_folds[year]["names"]) for year in selection_years)
        )
        - {"oracle"}
    )
    if not legal_names:
        raise ValueError(f"no legal mode signals for years={selection_years}")

    constants: dict[int, dict[str, np.ndarray]] = {}
    for year in selection_years:
        fold = state_folds[year]
        target = np.asarray(fold["target"], dtype=np.float64)
        incumbent = np.asarray(fold["incumbent"], dtype=np.float64)
        domain_mask = np.asarray(fold["domain3"]) == domain
        error = incumbent[domain_mask] - target[domain_mask]
        state_matrix = np.column_stack(
            [
                _state_correction(fold, domain, config)[domain_mask]
                for config in configs
            ]
        )
        mode_saved = mode_folds[year]
        mode_indices = [mode_saved["names"].index(name) for name in legal_names]
        mode_matrix = (
            np.asarray(mode_saved["raw"], dtype=np.float64)[:, mode_indices]
            - incumbent[:, None]
        )[domain_mask]
        reference = float(target.mean() * (1.0 - target.mean()))
        scale = -100000.0 / (reference * len(target))
        constants[year] = {
            "state_linear": scale * (2.0 * error) @ state_matrix,
            "state_quadratic": scale * np.sum(np.square(state_matrix), axis=0),
            "mode_linear": scale * (2.0 * error) @ mode_matrix,
            "mode_quadratic": scale * np.sum(np.square(mode_matrix), axis=0),
            "cross": scale * (2.0 * state_matrix.T @ mode_matrix),
        }

    state_grid = STATE_MULTIPLIERS[:, None]
    mode_grid = MODE_MULTIPLIERS[None, :]
    best: tuple[float, float, str, str, int, int] | None = None
    for state_index, config in enumerate(configs):
        for mode_index, mode_name in enumerate(legal_names):
            gains = []
            for year in selection_years:
                item = constants[year]
                gains.append(
                    state_grid * item["state_linear"][state_index]
                    + np.square(state_grid)
                    * item["state_quadratic"][state_index]
                    + mode_grid * item["mode_linear"][mode_index]
                    + np.square(mode_grid)
                    * item["mode_quadratic"][mode_index]
                    + state_grid
                    * mode_grid
                    * item["cross"][state_index, mode_index]
                )
            stack = np.stack(gains, axis=0)
            minimum = np.min(stack, axis=0)
            mean = np.mean(stack, axis=0)
            flat = int(np.argmax(minimum + 1e-9 * mean))
            state_grid_index, mode_grid_index = np.unravel_index(
                flat, minimum.shape
            )
            value = (
                float(minimum[state_grid_index, mode_grid_index]),
                float(mean[state_grid_index, mode_grid_index]),
                config,
                mode_name,
                state_grid_index,
                mode_grid_index,
            )
            if best is None or value[:2] > best[:2]:
                best = value
    if best is None:
        raise ValueError(f"no joint candidate for domain={domain}")

    minimum, mean, config, mode_name, state_index, mode_index = best
    state_multiplier = float(STATE_MULTIPLIERS[state_index])
    mode_multiplier = float(MODE_MULTIPLIERS[mode_index])
    fold_gains = {}
    config_index = configs.index(config)
    signal_index = legal_names.index(mode_name)
    for year in selection_years:
        item = constants[year]
        fold_gains[str(year)] = float(
            state_multiplier * item["state_linear"][config_index]
            + state_multiplier**2 * item["state_quadratic"][config_index]
            + mode_multiplier * item["mode_linear"][signal_index]
            + mode_multiplier**2 * item["mode_quadratic"][signal_index]
            + state_multiplier
            * mode_multiplier
            * item["cross"][config_index, signal_index]
        )
    return {
        "domain": domain,
        "state_config": config,
        "state_multiplier": state_multiplier,
        "mode_name": mode_name,
        "mode_multiplier": mode_multiplier,
        "minimum_selection_gain": minimum,
        "mean_selection_gain": mean,
        "selection_fold_gains": fold_gains,
    }


def run(
    project: Path,
    variants_dir: Path,
    selected_dir: Path,
    state_metrics_path: Path,
    mode_2022_dir: Path,
    mode_later_dir: Path,
    output_dir: Path,
    top_n: int,
    n_bootstrap: int,
) -> dict[str, object]:
    project = project.resolve()
    variants_dir = (project / variants_dir).resolve()
    selected_dir = (project / selected_dir).resolve()
    state_metrics_path = (project / state_metrics_path).resolve()
    mode_2022_dir = (project / mode_2022_dir).resolve()
    mode_later_dir = (project / mode_later_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_folds = {
        year: _load_fold(project, variants_dir, selected_dir, year)
        for year in (2022, 2023, 2024)
    }
    mode_folds = {
        2022: _mode_fold(mode_2022_dir, 2022),
        2023: _mode_fold(mode_later_dir, 2023),
        2024: _mode_fold(mode_later_dir, 2024),
    }
    for year in (2022, 2023, 2024):
        if not np.array_equal(
            state_folds[year]["target"], mode_folds[year]["target"]
        ):
            raise ValueError(f"state/mode target mismatch for {year}")

    metrics = pd.read_parquet(state_metrics_path)
    audit_results = []
    choices_by_audit: dict[str, list[dict[str, object]]] = {}
    for audit_year, selection_years in (
        (2023, (2022,)),
        (2024, (2022, 2023)),
    ):
        choices = []
        for domain in DOMAINS:
            print(
                f"[v42] audit={audit_year} select={selection_years} "
                f"domain={domain}",
                flush=True,
            )
            configs = top_configs(
                metrics, domain, selection_years, top_n
            )
            choices.append(
                search_domain(
                    domain,
                    configs,
                    state_folds,
                    mode_folds,
                    selection_years,
                )
            )
        candidate = _candidate(
            state_folds[audit_year], mode_folds[audit_year], choices
        )
        np.savez_compressed(
            output_dir / f"nested_candidate_o{audit_year}.npz",
            target=state_folds[audit_year]["target"],
            incumbent=state_folds[audit_year]["incumbent"],
            candidate=candidate,
            domain3=state_folds[audit_year]["domain3"],
            game_month=state_folds[audit_year]["game_month"],
            pitcher_id=state_folds[audit_year]["pitcher_id"],
            batter_id=state_folds[audit_year]["batter_id"],
        )
        choices_by_audit[str(audit_year)] = choices
        diagnostics = _diagnostics(
            audit_year,
            state_folds[audit_year],
            candidate,
            n_bootstrap,
        )
        diagnostics["selection_years"] = list(selection_years)
        audit_results.append(diagnostics)

    summary = {
        "protocol": "FULLY_FORWARD_NESTED_STATE_MODE_V1",
        "selection_rule": {
            "2023": [2022],
            "2024": [2022, 2023],
        },
        "top_state_configs_per_domain": top_n,
        "choices_by_audit": choices_by_audit,
        "audits": audit_results,
        "aggregate": {
            "minimum_audit_gain": float(
                min(item["gain"] for item in audit_results)
            ),
            "positive_audits": int(
                sum(item["gain"] > 0.0 for item in audit_results)
            ),
            "minimum_positive_month_fraction": float(
                min(item["positive_month_fraction"] for item in audit_results)
            ),
            "minimum_domain_gain": float(
                min(item["minimum_domain_gain"] for item in audit_results)
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=Path("artifacts/multi_year_state_variants_20260816_01"),
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=Path("artifacts/multi_year_state_selected_20260816_01"),
    )
    parser.add_argument(
        "--state-metrics-path",
        type=Path,
        default=Path(
            "artifacts/multi_year_state_ensemble_20260816_01/metrics.parquet"
        ),
    )
    parser.add_argument(
        "--mode-2022-dir",
        type=Path,
        default=Path("artifacts/latent_failure_mode_state_20260817_02"),
    )
    parser.add_argument(
        "--mode-later-dir",
        type=Path,
        default=Path("artifacts/latent_failure_mode_state_20260816_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v42_forward_nested_state_mode_20260817_01"),
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    run(
        args.project,
        args.variants_dir,
        args.selected_dir,
        args.state_metrics_path,
        args.mode_2022_dir,
        args.mode_later_dir,
        args.output_dir,
        args.top_n,
        args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
