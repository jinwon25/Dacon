"""Group-robust neural selection under temporal shift.

The representation is frozen from the first neural experiment.  Only two
predeclared training risks are compared on late-2023: uniform row risk and
equal risk across season x deployment-domain groups.  One winner is then
opened on the full-2024 audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v23_neural_embedding_screen import (
    DOMAIN_SUBSETS,
    ETAS,
    _diagnostics,
    _history,
    prepare,
    train_checkpoints,
)
from src.v23_structural_residual_screen import _load_axis


WEIGHTING_STRATEGIES = ("uniform", "season_domain_equal")


def run(
    project: Path,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 16384,
) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    selection = _load_axis(project, "y2023_early_to_late", raw)
    selection_fit = _history(raw, 2022)
    rows = []
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for strategy_index, strategy in enumerate(WEIGHTING_STRATEGIES):
        prepared = prepare(selection_fit, selection, weighting=strategy)
        checkpoints = train_checkpoints(
            prepared,
            epochs,
            batch_size,
            seed=24017 + 100 * strategy_index,
        )
        for epoch, prediction in checkpoints.items():
            predictions[(strategy, epoch)] = prediction
            for eta in ETAS:
                for domains in DOMAIN_SUBSETS:
                    rows.append(
                        {
                            "weighting": strategy,
                            "epoch": epoch,
                            "eta": eta,
                            "domains": "+".join(domains),
                            **_diagnostics(selection, prediction, eta, domains),
                        }
                    )
    metrics = pd.DataFrame(rows).sort_values(
        ["selection_score", "gain"], ascending=False
    )
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    selected = metrics.iloc[0]
    selection_gate = bool(
        selected["gain"] > 0.0
        and selected["worst_month_gain"] > 0.0
        and selected["minimum_applied_domain_gain"] > 0.0
    )
    summary: dict[str, object] = {
        "protocol": "V23_NEURAL_GROUP_ROBUST_V1",
        "selection": "uniform vs season-domain-equal risk selected on honest late-2023 v22 OOF",
        "selected": {key: selected[key] for key in metrics.columns},
        "selection_gate_passed": selection_gate,
        "outer_audit_run": False,
        "eligible_for_packaging": False,
    }
    if selection_gate:
        strategy = str(selected["weighting"])
        epoch = int(selected["epoch"])
        eta = float(selected["eta"])
        domains = tuple(str(selected["domains"]).split("+"))
        outer = _load_axis(project, "y2023_to_y2024", raw)
        outer_fit = _history(raw, 2023)
        outer_prepared = prepare(outer_fit, outer, weighting=strategy)
        outer_prediction = train_checkpoints(
            outer_prepared,
            epoch,
            batch_size,
            seed=24017 + 100 * WEIGHTING_STRATEGIES.index(strategy),
        )[epoch]
        diagnostics = _diagnostics(outer, outer_prediction, eta, domains)
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
            + eta * (outer_prediction[apply_mask] - parent[apply_mask]),
            0.001,
            0.999,
        )
        np.savez_compressed(
            output_dir / "outer_prediction.npz",
            target=outer["target"].to_numpy(np.float64),
            v22=parent,
            neural=outer_prediction,
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
        default=Path("artifacts/v23_neural_group_robust_20260817_01"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16384)
    args = parser.parse_args()
    run(args.project, args.output_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
