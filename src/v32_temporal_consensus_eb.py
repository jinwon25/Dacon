"""Temporal-consensus empirical-Bayes residual screen above v27.

A lookup correction is allowed only when its residual effect has the same sign
in two chronological halves of the fit window.  This turns instability into an
explicit zero correction instead of carrying a noisy player/context estimate
forward.  All tables are fitted from labelled history and frozen before audit
rows are mapped independently.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.champion_oof import _load_year
from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.v31_dynamic_hierarchical_residual import _prepare


GROUPS: dict[str, tuple[str, ...]] = {
    "pitcher_hand": ("pitcher_id", "batter_hand"),
    "pitcher_hand_pressure": ("pitcher_id", "batter_hand", "pressure"),
    "pitcher_hand_count": ("pitcher_id", "batter_hand", "count_state"),
    "batter_hand": ("batter_id", "pitcher_hand"),
    "batter_hand_pressure": ("batter_id", "pitcher_hand", "pressure"),
    "batter_hand_count": ("batter_id", "pitcher_hand", "count_state"),
    "pitcher_team_hands": ("pitcher_team_id", "pitcher_hand", "batter_hand"),
    "count_hands": ("balls_before", "strikes_before", "pitcher_hand", "batter_hand"),
}
ALPHAS = (50.0, 200.0, 800.0)
MIN_HALF_COUNTS = (5.0, 20.0)
MODES = ("full", "mean", "minimum")
ETAS = (0.05, 0.10, 0.20, 0.35)
DOMAINS = ("ALL", "R_CORE")


def _key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame.loc[:, list(columns)].astype("string").fillna("__MISSING__")
    )


def _split_month(fit: pd.DataFrame) -> int:
    """Return a deterministic month boundary closest to half the fit rows."""
    counts = fit.groupby("game_month", observed=True).size().sort_index()
    cumulative = counts.cumsum().to_numpy()
    index = int(np.argmin(np.abs(cumulative - len(fit) / 2.0)))
    boundary = int(counts.index[index])
    if boundary >= int(counts.index.max()):
        boundary = int(counts.index[-2])
    return boundary


def consensus_effect(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    alpha: float,
    min_half_count: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map only EB effects whose two chronological estimates agree in sign."""
    key = _key(fit, columns)
    codes, unique = pd.factorize(key, sort=False)
    residual = fit["residual_v19"].to_numpy(np.float64)
    boundary = _split_month(fit)
    first = fit["game_month"].le(boundary).to_numpy(np.float64)
    second = 1.0 - first

    def aggregate(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.bincount(codes, weights=weight * residual, minlength=len(unique)),
            np.bincount(codes, weights=weight, minlength=len(unique)),
        )

    sum_first, n_first = aggregate(first)
    sum_second, n_second = aggregate(second)
    sum_full = sum_first + sum_second
    n_full = n_first + n_second
    effect_first = sum_first / (n_first + float(alpha))
    effect_second = sum_second / (n_second + float(alpha))
    agreement = (
        (np.signbit(effect_first) == np.signbit(effect_second))
        & (effect_first != 0.0)
        & (effect_second != 0.0)
        & (n_first >= float(min_half_count))
        & (n_second >= float(min_half_count))
    )
    if mode == "full":
        effect = sum_full / (n_full + float(alpha))
    elif mode == "mean":
        effect = 0.5 * (effect_first + effect_second)
    elif mode == "minimum":
        effect = np.sign(effect_first) * np.minimum(
            np.abs(effect_first), np.abs(effect_second)
        )
    else:
        raise ValueError(f"unknown consensus mode: {mode}")
    effect = np.where(agreement, effect, 0.0)
    audit_code = unique.get_indexer(_key(audit, columns))
    known = audit_code >= 0
    output = np.zeros(len(audit), dtype=np.float64)
    active = np.zeros(len(audit), dtype=bool)
    output[known] = effect[audit_code[known]]
    active[known] = agreement[audit_code[known]]
    return output, active


def _apply(
    frame: pd.DataFrame,
    effect: np.ndarray,
    eta: float,
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    mask = (
        np.ones(len(frame), dtype=bool)
        if domain == "ALL"
        else frame["domain3"].astype(str).eq(domain).to_numpy()
    )
    candidate = parent.copy()
    candidate[mask] = np.clip(
        parent[mask] + float(eta) * effect[mask], 0.001, 0.999
    )
    return candidate, mask


def _axes(project: Path, raw: pd.DataFrame):
    audits = {name: _prepare(frame) for name, frame in _cached_v25_axes(project, raw).items()}
    year23 = _prepare(_load_year(project, 2023))
    year24 = _prepare(_load_year(project, 2024))
    fits = {
        "selection_late_2023": year23.loc[year23["game_month"].le(7)].reset_index(drop=True),
        "outer_full_2024": year23,
        "replication_late_2024": year24.loc[year24["game_month"].le(7)].reset_index(drop=True),
    }
    return {name: (fits[name], audits[name]) for name in audits}


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _axes(project, raw)
    fit, audit = axes["selection_late_2023"]
    parent = v27_parent(audit)
    selection_rows = []
    cache = {}
    for group_name, columns in GROUPS.items():
        for alpha, min_half_count, mode in itertools.product(
            ALPHAS, MIN_HALF_COUNTS, MODES
        ):
            key = (group_name, alpha, min_half_count, mode)
            effect, active = consensus_effect(
                fit,
                audit,
                columns,
                alpha=alpha,
                min_half_count=min_half_count,
                mode=mode,
            )
            cache[key] = (effect, active)
            for domain, eta in itertools.product(DOMAINS, ETAS):
                candidate, mask = _apply(audit, effect, eta, domain)
                result = diagnostics(audit, parent, candidate, mask & active)
                selection_rows.append(
                    {
                        "group": group_name,
                        "alpha": alpha,
                        "min_half_count": min_half_count,
                        "mode": mode,
                        "domain": domain,
                        "eta": eta,
                        "coverage": float(np.mean(mask & active)),
                        **{
                            name: value
                            for name, value in result.items()
                            if name not in {"months", "domain_gains"}
                        },
                    }
                )
    metrics = pd.DataFrame(selection_rows)
    metrics["selection_score"] = metrics[
        ["gain", "worst_month_gain"]
    ].min(axis=1)
    metrics["passes_selection_gate"] = (
        metrics["gain"].gt(0.0)
        & metrics["positive_month_fraction"].eq(1.0)
        & metrics["worst_month_gain"].gt(0.0)
        & metrics["minimum_domain_gain"].gt(-5.0)
        & metrics["coverage"].ge(0.05)
    )
    metrics = metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = metrics.loc[metrics["passes_selection_gate"]]
    row = passing.iloc[0] if len(passing) else metrics.iloc[0]
    chosen = {
        "group": str(row["group"]),
        "alpha": float(row["alpha"]),
        "min_half_count": float(row["min_half_count"]),
        "mode": str(row["mode"]),
        "domain": str(row["domain"]),
        "eta": float(row["eta"]),
    }

    results = {}
    for axis_name, (axis_fit, axis_audit) in axes.items():
        effect, active = consensus_effect(
            axis_fit,
            axis_audit,
            GROUPS[chosen["group"]],
            alpha=chosen["alpha"],
            min_half_count=chosen["min_half_count"],
            mode=chosen["mode"],
        )
        candidate, mask = _apply(
            axis_audit, effect, chosen["eta"], chosen["domain"]
        )
        result = diagnostics(
            axis_audit, v27_parent(axis_audit), candidate, mask & active
        )
        result["coverage"] = float(np.mean(mask & active))
        results[axis_name] = result
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=axis_audit["target"].to_numpy(np.float64),
            v27=v27_parent(axis_audit),
            effect=effect,
            active=active,
            candidate=candidate,
            domain3=axis_audit["domain3"].astype(str).to_numpy(),
            game_month=axis_audit["game_month"].to_numpy(np.int16),
            pitcher_id=axis_audit["pitcher_id"].to_numpy(),
            batter_id=axis_audit["batter_id"].to_numpy(),
        )
    gates = {
        "selection_gate": bool(row["passes_selection_gate"]),
        "outer_gain_at_least_5": results["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": results["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": results["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "replication_gain_positive": results["replication_late_2024"]["gain"] > 0.0,
        "replication_all_months_positive": results["replication_late_2024"][
            "positive_month_fraction"
        ]
        == 1.0,
    }
    summary = {
        "protocol": "V32_TEMPORAL_CONSENSUS_EB_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "selection_candidate_count": int(len(metrics)),
        "selection_gate_count": int(len(passing)),
        "chosen": chosen,
        "chosen_selection": {
            name: float(row[name])
            for name in (
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "minimum_domain_gain",
                "coverage",
            )
        },
        "axes": results,
        "gates": {name: bool(value) for name, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
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
        default=Path("artifacts/v32_temporal_consensus_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
