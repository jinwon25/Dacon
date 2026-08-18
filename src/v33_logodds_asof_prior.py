"""Low-degree-of-freedom log-odds ASOF prior screen above frozen v27.

The official row-local pitcher and batter success rates are first shrunk toward
a source-period domain rate using their ASOF sample sizes.  The two posterior
rates are then combined on the log-odds scale.  This is a conservative,
control-oriented adaptation of hierarchical matchup/log5 models: no audit
labels, audit aggregates, or other test rows are used at inference time.

Recipe selection uses early-2023 -> late-2023 only.  The selected recipe is
then frozen for the 2023 -> 2024 and early-2024 -> late-2024 audits.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v23_multiyear_direct_screen import _joint_domain
from src.v30_diverse_covariance_screen import (
    _cached_v25_axes,
    diagnostics,
    v27_parent,
)


COEFFICIENTS = {
    "control_75_25": (0.75, 0.25),
    "pitcher_90_batter10": (0.90, 0.10),
    "pitcher_100_batter25": (1.00, 0.25),
    "pitcher_100_batter50": (1.00, 0.50),
}
STRENGTHS = {
    "responsive": (20.0, 40.0),
    "balanced": (80.0, 160.0),
    "stable": (320.0, 640.0),
}
MODES = ("curvature", "replace_raw_prior", "toward_logodds")
DOMAINS = ("ALL", "R_CORE", "R_ANCHOR")
ETAS = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15)
EPSILON = 1e-4


def _numeric(frame: pd.DataFrame, column: str, default: float) -> np.ndarray:
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .fillna(default)
        .to_numpy(np.float64)
    )


def fit_domain_priors(source: pd.DataFrame) -> dict[str, float]:
    """Fit labelled source-period domain rates with a global fallback."""
    target = pd.to_numeric(source["control_success"], errors="raise")
    global_rate = float(target.mean())
    priors = {"__GLOBAL__": global_rate}
    for domain, values in source.assign(_target=target).groupby(
        "domain3", observed=True
    )["_target"]:
        priors[str(domain)] = float(values.mean())
    return priors


def map_domain_prior(frame: pd.DataFrame, priors: dict[str, float]) -> np.ndarray:
    fallback = float(priors["__GLOBAL__"])
    return (
        frame["domain3"].astype(str).map(priors).fillna(fallback).to_numpy(np.float64)
    )


def posterior_rate(
    rate: np.ndarray,
    sample_size: np.ndarray,
    prior: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Return a sample-size-aware beta-binomial posterior mean."""
    local_rate = np.asarray(rate, dtype=np.float64)
    local_n = np.maximum(np.asarray(sample_size, dtype=np.float64), 0.0)
    local_prior = np.asarray(prior, dtype=np.float64)
    local_rate = np.where(np.isfinite(local_rate), local_rate, local_prior)
    local_rate = np.clip(local_rate, 0.0, 1.0)
    return (local_n * local_rate + float(strength) * local_prior) / (
        local_n + float(strength)
    )


def logodds_matchup(
    pitcher_rate: np.ndarray,
    batter_rate: np.ndarray,
    prior: np.ndarray,
    pitcher_weight: float,
    batter_weight: float,
) -> np.ndarray:
    """Combine posterior rates while preserving the domain-rate fixed point."""

    def logit(value: np.ndarray) -> np.ndarray:
        bounded = np.clip(np.asarray(value, dtype=np.float64), EPSILON, 1.0 - EPSILON)
        return np.log(bounded) - np.log1p(-bounded)

    base = logit(prior)
    score = (
        base
        + float(pitcher_weight) * (logit(pitcher_rate) - base)
        + float(batter_weight) * (logit(batter_rate) - base)
    )
    return 1.0 / (1.0 + np.exp(-score))


def prior_direction(
    frame: pd.DataFrame,
    priors: dict[str, float],
    *,
    pitcher_weight: float,
    batter_weight: float,
    pitcher_strength: float,
    batter_strength: float,
    mode: str,
) -> np.ndarray:
    domain_prior = map_domain_prior(frame, priors)
    raw_pitcher = _numeric(frame, "asof_pitcher_success_rate", np.nan)
    raw_batter = _numeric(frame, "asof_batter_success_rate", np.nan)
    pitcher_n = _numeric(frame, "asof_pitcher_n", 0.0)
    batter_n = _numeric(frame, "asof_batter_n", 0.0)
    pitcher = posterior_rate(
        raw_pitcher, pitcher_n, domain_prior, pitcher_strength
    )
    batter = posterior_rate(raw_batter, batter_n, domain_prior, batter_strength)
    combined = logodds_matchup(
        pitcher, batter, domain_prior, pitcher_weight, batter_weight
    )
    arithmetic = (
        domain_prior
        + float(pitcher_weight) * (pitcher - domain_prior)
        + float(batter_weight) * (batter - domain_prior)
    )
    raw_pitcher = np.where(np.isfinite(raw_pitcher), raw_pitcher, 0.5)
    raw_batter = np.where(np.isfinite(raw_batter), raw_batter, 0.5)
    raw_v22_prior = 0.75 * raw_pitcher + 0.25 * raw_batter
    if mode == "curvature":
        return combined - arithmetic
    if mode == "replace_raw_prior":
        return combined - raw_v22_prior
    if mode == "toward_logodds":
        return combined - v27_parent(frame)
    raise ValueError(f"unknown prior direction mode: {mode}")


def _mask(frame: pd.DataFrame, domain: str) -> np.ndarray:
    if domain == "ALL":
        return np.ones(len(frame), dtype=bool)
    return frame["domain3"].astype(str).eq(domain).to_numpy()


def apply_direction(
    frame: pd.DataFrame, direction: np.ndarray, eta: float, domain: str
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    mask = _mask(frame, domain)
    candidate = parent.copy()
    candidate[mask] = np.clip(
        parent[mask] + float(eta) * np.asarray(direction, dtype=np.float64)[mask],
        0.001,
        0.999,
    )
    return candidate, mask


def _source_frames(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {
        "selection_late_2023": raw.loc[
            raw["season"].eq(2023) & raw["game_month"].le(7)
        ].copy(),
        "outer_full_2024": raw.loc[raw["season"].eq(2023)].copy(),
        "replication_late_2024": raw.loc[
            raw["season"].eq(2024) & raw["game_month"].le(7)
        ].copy(),
    }
    for frame in frames.values():
        frame["domain3"] = _joint_domain(frame)
    return frames


def _recipe(row: pd.Series) -> dict[str, object]:
    return {
        "coefficient": str(row["coefficient"]),
        "strength": str(row["strength"]),
        "mode": str(row["mode"]),
        "domain": str(row["domain"]),
        "eta": float(row["eta"]),
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    audits = _cached_v25_axes(project, raw)
    sources = _source_frames(raw)

    selection = audits["selection_late_2023"]
    selection_parent = v27_parent(selection)
    selection_priors = fit_domain_priors(sources["selection_late_2023"])
    rows: list[dict[str, object]] = []
    direction_cache: dict[tuple[str, str, str], np.ndarray] = {}
    for coefficient, strength, mode in itertools.product(
        COEFFICIENTS, STRENGTHS, MODES
    ):
        key = (coefficient, strength, mode)
        p_weight, b_weight = COEFFICIENTS[coefficient]
        p_strength, b_strength = STRENGTHS[strength]
        direction = prior_direction(
            selection,
            selection_priors,
            pitcher_weight=p_weight,
            batter_weight=b_weight,
            pitcher_strength=p_strength,
            batter_strength=b_strength,
            mode=mode,
        )
        direction_cache[key] = direction
        for domain, eta in itertools.product(DOMAINS, ETAS):
            candidate, mask = apply_direction(selection, direction, eta, domain)
            result = diagnostics(selection, selection_parent, candidate, mask)
            rows.append(
                {
                    "coefficient": coefficient,
                    "strength": strength,
                    "mode": mode,
                    "domain": domain,
                    "eta": eta,
                    **{
                        name: value
                        for name, value in result.items()
                        if name not in {"months", "domain_gains"}
                    },
                }
            )
    metrics = pd.DataFrame(rows)
    metrics["selection_score"] = metrics[["gain", "worst_month_gain"]].min(axis=1)
    metrics["passes_selection_gate"] = (
        metrics["gain"].gt(0.0)
        & metrics["positive_month_fraction"].eq(1.0)
        & metrics["worst_month_gain"].gt(0.0)
        & metrics["minimum_domain_gain"].gt(-5.0)
    )
    metrics = metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = metrics.loc[metrics["passes_selection_gate"]]
    selected = passing.iloc[0] if len(passing) else metrics.iloc[0]
    chosen = _recipe(selected)

    results: dict[str, dict[str, object]] = {}
    for axis_name, audit in audits.items():
        priors = fit_domain_priors(sources[axis_name])
        p_weight, b_weight = COEFFICIENTS[str(chosen["coefficient"])]
        p_strength, b_strength = STRENGTHS[str(chosen["strength"])]
        direction = prior_direction(
            audit,
            priors,
            pitcher_weight=p_weight,
            batter_weight=b_weight,
            pitcher_strength=p_strength,
            batter_strength=b_strength,
            mode=str(chosen["mode"]),
        )
        candidate, mask = apply_direction(
            audit, direction, float(chosen["eta"]), str(chosen["domain"])
        )
        parent = v27_parent(audit)
        result = diagnostics(audit, parent, candidate, mask)
        result["domain_priors"] = priors
        results[axis_name] = result
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=audit["target"].to_numpy(np.float64),
            v27=parent,
            direction=direction,
            candidate=candidate,
            domain3=audit["domain3"].astype(str).to_numpy(),
            game_month=audit["game_month"].to_numpy(np.int16),
        )

    gates = {
        "selection_gate": bool(selected["passes_selection_gate"]),
        "outer_gain_at_least_3": results["outer_full_2024"]["gain"] >= 3.0,
        "outer_month_fraction_at_least_075": results["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": results["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "replication_gain_positive": results["replication_late_2024"]["gain"] > 0.0,
    }
    summary = {
        "protocol": "V33_LOGODDS_ASOF_PRIOR_NESTED_2025_V1",
        "parent": "submit_v27.zip",
        "selection_axis": "2023 March-July -> August-October",
        "independent_audits": [
            "2023 full -> 2024 full",
            "2024 March-July -> August-October",
        ],
        "candidate_count": int(len(metrics)),
        "selection_gate_count": int(metrics["passes_selection_gate"].sum()),
        "chosen": chosen,
        "results": results,
        "gates": gates,
        "eligible": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/v33_logodds_asof_prior_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
