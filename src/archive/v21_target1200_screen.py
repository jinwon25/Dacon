"""Nested one-season-ahead audit for a possible v21 state/mode route.

The v19 route was selected by maximizing the worst result across 2023 and
2024.  That is useful development evidence, but it does not show how a route
chosen in one season transfers to the next.  This module performs that missing
experiment: choose every state/mode route using 2023 labels only, freeze it,
and score it on 2024.  The same deterministic rule can then be fit on 2024 for
2025 deployment if (and only if) the frozen 2024 audit is convincing.

No test rows, Public predictions, or TrackMan measurements are read here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.multi_year_state_ensemble_screen import DOMAINS, _load_fold
from src.state_mode_joint_screen import (
    MODE_MULTIPLIERS,
    STATE_MULTIPLIERS,
    _candidate,
    _diagnostics,
    _mode_fold,
    _state_correction,
)
from src.trackman_privileged_distillation import bss


def _top_source_configs(
    metrics: pd.DataFrame, domain: str, source_year: int, top_n: int
) -> list[str]:
    source = metrics.loc[
        metrics["audit_year"].eq(source_year) & metrics["domain"].eq(domain)
    ]
    return (
        source.sort_values("gain", ascending=False, kind="stable")
        .drop_duplicates("config")
        .head(top_n)["config"]
        .astype(str)
        .tolist()
    )


def _quadratic_constants(
    fold: dict[str, object],
    mode: dict[str, object],
    domain: str,
    configs: list[str],
    mode_names: list[str],
) -> dict[str, np.ndarray]:
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
    mode_indices = [mode["names"].index(name) for name in mode_names]
    mode_matrix = (
        np.asarray(mode["raw"], dtype=np.float64)[:, mode_indices]
        - incumbent[:, None]
    )[domain_mask]
    reference = float(target.mean() * (1.0 - target.mean()))
    scale = -100000.0 / (reference * len(target))
    return {
        "state_linear": scale * (2.0 * error) @ state_matrix,
        "state_quadratic": scale * np.sum(np.square(state_matrix), axis=0),
        "mode_linear": scale * (2.0 * error) @ mode_matrix,
        "mode_quadratic": scale * np.sum(np.square(mode_matrix), axis=0),
        "cross": scale * (2.0 * state_matrix.T @ mode_matrix),
    }


def _select_on_source(
    source_year: int,
    domain: str,
    configs: list[str],
    state_fold: dict[str, object],
    mode_fold: dict[str, object],
    damping: float,
) -> dict[str, object]:
    mode_names = [name for name in mode_fold["names"] if name != "oracle"]
    constants = _quadratic_constants(
        state_fold, mode_fold, domain, configs, mode_names
    )
    state_grid = STATE_MULTIPLIERS[:, None]
    mode_grid = MODE_MULTIPLIERS[None, :]
    best: tuple[float, int, int, int, int] | None = None
    for state_index in range(len(configs)):
        for mode_index in range(len(mode_names)):
            gain = (
                state_grid * constants["state_linear"][state_index]
                + np.square(state_grid)
                * constants["state_quadratic"][state_index]
                + mode_grid * constants["mode_linear"][mode_index]
                + np.square(mode_grid)
                * constants["mode_quadratic"][mode_index]
                + state_grid
                * mode_grid
                * constants["cross"][state_index, mode_index]
            )
            flat = int(np.argmax(gain))
            state_position, mode_position = np.unravel_index(flat, gain.shape)
            value = (
                float(gain[state_position, mode_position]),
                state_index,
                mode_index,
                state_position,
                mode_position,
            )
            if best is None or value[0] > best[0]:
                best = value
    if best is None:
        raise ValueError(f"no source-selected route for {domain}")
    source_gain, state_index, mode_index, state_position, mode_position = best
    return {
        "domain": domain,
        "state_config": configs[state_index],
        "state_multiplier": float(STATE_MULTIPLIERS[state_position] * damping),
        "mode_name": mode_names[mode_index],
        "mode_multiplier": float(MODE_MULTIPLIERS[mode_position] * damping),
        "selection_year": source_year,
        "undamped_source_gain": source_gain,
        "damping": damping,
    }


def _score_route(
    fold: dict[str, object],
    mode_fold: dict[str, object],
    choices: list[dict[str, object]],
) -> float:
    prediction = _candidate(fold, mode_fold, choices)
    return bss(fold["target"], prediction) - bss(
        fold["target"], fold["incumbent"]
    )


def run(
    project: Path,
    variants_dir: Path,
    selected_dir: Path,
    state_metrics_path: Path,
    mode_dir: Path,
    output_dir: Path,
    top_n: int,
    n_bootstrap: int,
) -> dict[str, object]:
    project = project.resolve()
    variants_dir = (project / variants_dir).resolve()
    selected_dir = (project / selected_dir).resolve()
    state_metrics_path = (project / state_metrics_path).resolve()
    mode_dir = (project / mode_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_folds = {
        year: _load_fold(project, variants_dir, selected_dir, year)
        for year in (2023, 2024)
    }
    mode_folds = {year: _mode_fold(mode_dir, year) for year in (2023, 2024)}
    metrics = pd.read_parquet(state_metrics_path)
    incumbent_joint = {}
    incumbent_dir = project / "artifacts" / "state_mode_joint_20260816_02"
    for year in (2023, 2024):
        with np.load(
            incumbent_dir / f"joint_candidate_o{year}.npz", allow_pickle=True
        ) as saved:
            incumbent_joint[year] = saved["candidate"].astype(np.float64)

    candidates: list[dict[str, object]] = []
    for damping in (0.50, 0.75, 1.00):
        choices = []
        for domain in DOMAINS:
            configs = _top_source_configs(metrics, domain, 2023, top_n)
            choices.append(
                _select_on_source(
                    2023,
                    domain,
                    configs,
                    state_folds[2023],
                    mode_folds[2023],
                    damping,
                )
            )
        prediction_2023 = _candidate(
            state_folds[2023], mode_folds[2023], choices
        )
        prediction_2024 = _candidate(
            state_folds[2024], mode_folds[2024], choices
        )
        gain_2023 = _score_route(state_folds[2023], mode_folds[2023], choices)
        gain_2024 = _score_route(state_folds[2024], mode_folds[2024], choices)
        target_2024 = np.asarray(state_folds[2024]["target"], dtype=np.float64)
        gain_vs_v19_2024 = bss(target_2024, prediction_2024) - bss(
            target_2024, incumbent_joint[2024]
        )
        diagnostics = _diagnostics(
            2024, state_folds[2024], prediction_2024, n_bootstrap
        )
        np.savez_compressed(
            output_dir / f"nested_route_d{int(100*damping):03d}.npz",
            target_2023=state_folds[2023]["target"],
            prediction_2023=prediction_2023,
            target_2024=target_2024,
            prediction_2024=prediction_2024,
            v19_2024=incumbent_joint[2024],
            domain3_2024=state_folds[2024]["domain3"],
            game_month_2024=state_folds[2024]["game_month"],
        )
        candidates.append(
            {
                "damping": damping,
                "choices": choices,
                "source_gain_vs_v17_2023": gain_2023,
                "audit_gain_vs_v17_2024": gain_2024,
                "audit_gain_vs_v19_2024": gain_vs_v19_2024,
                "audit_diagnostics_vs_v17": diagnostics,
            }
        )

    summary = {
        "protocol": "V21_NESTED_STATE_MODE_ROUTE_V1",
        "selection": "route selected on 2023 only; 2024 frozen audit",
        "top_source_configs_per_domain": top_n,
        "candidates": candidates,
        "eligible": [
            row["damping"]
            for row in candidates
            if row["audit_gain_vs_v19_2024"] >= 12.0
            and row["audit_diagnostics_vs_v17"]["positive_month_fraction"]
            >= 0.75
            and row["audit_diagnostics_vs_v17"]["minimum_domain_gain"] > 0.0
        ],
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
        default=Path("artifacts/multi_year_state_ensemble_20260816_01/metrics.parquet"),
    )
    parser.add_argument(
        "--mode-dir",
        type=Path,
        default=Path("artifacts/latent_failure_mode_state_20260816_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v21_target1200_nested_route_20260816_01"),
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    run(
        args.project,
        args.variants_dir,
        args.selected_dir,
        args.state_metrics_path,
        args.mode_dir,
        args.output_dir,
        args.top_n,
        args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
