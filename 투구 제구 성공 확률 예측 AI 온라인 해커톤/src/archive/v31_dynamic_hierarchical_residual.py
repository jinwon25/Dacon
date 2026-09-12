"""Dynamic crossed-effects residual screen above v27.

The model estimates a current-source-period residual for pitchers and batters,
then progressively pools finer hand/count states toward their player-level
parent.  Selection is confined to early-2023 -> late-2023.  The chosen recipe
is frozen before full-2024 and early-2024 -> late-2024 audits are evaluated.

No evaluation-row aggregates are used: deployment consists only of train-only
lookup tables mapped independently to each row.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.champion_oof import _load_year
from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes


ALPHA_PATTERNS = {
    "stable": (3200.0, 1600.0, 800.0, 400.0),
    "medium": (1600.0, 800.0, 400.0, 200.0),
    "adaptive": (800.0, 400.0, 200.0, 100.0),
}
HALF_LIVES = (1.0, 2.0, 4.0, None)
ETAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
APPLY_DOMAINS = ("ALL", "R_CORE")


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["count_state"] = (
        output["balls_before"].astype("Int64").astype(str)
        + "-"
        + output["strikes_before"].astype("Int64").astype(str)
    )
    if "pressure" not in output:
        balls = pd.to_numeric(output["balls_before"], errors="coerce").to_numpy()
        strikes = pd.to_numeric(output["strikes_before"], errors="coerce").to_numpy()
        output["pressure"] = np.where(
            balls == 3, "threeball", np.where(strikes == 2, "twostrike", "normal")
        )
    return output


def _key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame.loc[:, list(columns)].astype("string").fillna("__MISSING__")
    )


def hierarchical_effect(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    levels: tuple[tuple[str, ...], ...],
    alphas: tuple[float, ...],
    half_life: float | None,
) -> np.ndarray:
    """Estimate a nested EB residual, falling back through coarser levels."""
    if len(levels) != len(alphas):
        raise ValueError("levels and alphas must have the same length")
    residual = fit["residual_v19"].to_numpy(np.float64)
    if half_life is None:
        row_weight = np.ones(len(fit), dtype=np.float64)
    else:
        month = fit["game_month"].to_numpy(np.float64)
        row_weight = np.exp2(-(float(month.max()) - month) / float(half_life))
    fit_parent = np.zeros(len(fit), dtype=np.float64)
    audit_parent = np.zeros(len(audit), dtype=np.float64)
    for columns, alpha in zip(levels, alphas, strict=True):
        fit_key = _key(fit, columns)
        codes, unique = pd.factorize(fit_key, sort=False)
        weighted_n = np.bincount(codes, weights=row_weight, minlength=len(unique))
        weighted_sum = np.bincount(
            codes, weights=row_weight * residual, minlength=len(unique)
        )
        weighted_prior = np.bincount(
            codes, weights=row_weight * fit_parent, minlength=len(unique)
        ) / np.maximum(weighted_n, 1e-12)
        effect = (weighted_sum + float(alpha) * weighted_prior) / (
            weighted_n + float(alpha)
        )
        fit_parent = effect[codes]
        audit_codes = unique.get_indexer(_key(audit, columns))
        known = audit_codes >= 0
        next_audit = audit_parent.copy()
        next_audit[known] = effect[audit_codes[known]]
        audit_parent = next_audit
    return audit_parent


def _levels(entity: str, context: str) -> tuple[tuple[str, ...], ...]:
    opponent_hand = "batter_hand" if entity == "pitcher_id" else "pitcher_hand"
    return (
        (entity,),
        (entity, "domain3"),
        (entity, "domain3", opponent_hand),
        (entity, "domain3", opponent_hand, context),
    )


def _signal(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    family: str,
    alphas: tuple[float, ...],
    half_life: float | None,
) -> np.ndarray:
    context = "pressure" if family.endswith("pressure") else "count_state"
    pitcher = hierarchical_effect(
        fit, audit, _levels("pitcher_id", context), alphas, half_life
    )
    if family.startswith("pitcher_"):
        return pitcher
    batter = hierarchical_effect(
        fit, audit, _levels("batter_id", context), alphas, half_life
    )
    if family.startswith("batter_"):
        return batter
    if family.startswith("crossed_"):
        return 0.75 * pitcher + 0.25 * batter
    raise ValueError(f"unknown family: {family}")


def _apply(
    frame: pd.DataFrame,
    signal: np.ndarray,
    eta: float,
    apply_domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    mask = (
        np.ones(len(frame), dtype=bool)
        if apply_domain == "ALL"
        else frame["domain3"].astype(str).eq(apply_domain).to_numpy()
    )
    candidate = parent.copy()
    candidate[mask] = np.clip(
        parent[mask] + float(eta) * np.asarray(signal, dtype=np.float64)[mask],
        0.001,
        0.999,
    )
    return candidate, mask


def _fit_audit_axes(
    project: Path, raw: pd.DataFrame
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], dict[str, pd.DataFrame]]:
    audits = {name: _prepare(frame) for name, frame in _cached_v25_axes(project, raw).items()}
    year23 = _prepare(_load_year(project, 2023))
    year24 = _prepare(_load_year(project, 2024))
    fits = {
        "selection_late_2023": year23.loc[year23["game_month"].le(7)].reset_index(drop=True),
        "outer_full_2024": year23,
        "replication_late_2024": year24.loc[year24["game_month"].le(7)].reset_index(drop=True),
    }
    for name, audit in audits.items():
        if not np.array_equal(
            audit["target"].to_numpy(np.float64),
            (
                year23.loc[year23["game_month"].ge(8), "target"].to_numpy(np.float64)
                if name == "selection_late_2023"
                else year24["target"].to_numpy(np.float64)
                if name == "outer_full_2024"
                else year24.loc[year24["game_month"].ge(8), "target"].to_numpy(np.float64)
            ),
        ):
            raise ValueError(f"audit target order mismatch: {name}")
    return {name: (fits[name], audits[name]) for name in audits}, audits


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axis_pairs, audits = _fit_audit_axes(project, raw)
    families = (
        "pitcher_pressure",
        "pitcher_count",
        "batter_pressure",
        "batter_count",
        "crossed_pressure",
        "crossed_count",
    )

    selection_fit, selection = axis_pairs["selection_late_2023"]
    selection_parent = v27_parent(selection)
    rows: list[dict[str, object]] = []
    signal_cache: dict[tuple[str, str, float | None], np.ndarray] = {}
    for family, (alpha_name, alphas), half_life in itertools.product(
        families, ALPHA_PATTERNS.items(), HALF_LIVES
    ):
        cache_key = (family, alpha_name, half_life)
        signal_cache[cache_key] = _signal(
            selection_fit, selection, family, alphas, half_life
        )
        for apply_domain, eta in itertools.product(APPLY_DOMAINS, ETAS):
            candidate, mask = _apply(
                selection, signal_cache[cache_key], eta, apply_domain
            )
            result = diagnostics(selection, selection_parent, candidate, mask)
            rows.append(
                {
                    "family": family,
                    "alpha_pattern": alpha_name,
                    "half_life": "all" if half_life is None else half_life,
                    "apply_domain": apply_domain,
                    "eta": eta,
                    **{
                        key: value
                        for key, value in result.items()
                        if key not in {"months", "domain_gains"}
                    },
                }
            )
    metrics = pd.DataFrame(rows)
    metrics["selection_score"] = metrics[["gain", "worst_month_gain"]].min(axis=1)
    metrics["passes_selection_gate"] = (
        metrics["gain"].gt(0.0)
        & metrics["positive_month_fraction"].eq(1.0)
        & metrics["worst_month_gain"].gt(0.0)
    )
    metrics = metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    eligible = metrics.loc[metrics["passes_selection_gate"]]
    chosen_row = eligible.iloc[0] if len(eligible) else metrics.iloc[0]
    chosen = {
        "family": str(chosen_row["family"]),
        "alpha_pattern": str(chosen_row["alpha_pattern"]),
        "half_life": None
        if str(chosen_row["half_life"]) == "all"
        else float(chosen_row["half_life"]),
        "apply_domain": str(chosen_row["apply_domain"]),
        "eta": float(chosen_row["eta"]),
    }

    axis_results = {}
    for axis_name, (fit, audit) in axis_pairs.items():
        signal = _signal(
            fit,
            audit,
            chosen["family"],
            ALPHA_PATTERNS[chosen["alpha_pattern"]],
            chosen["half_life"],
        )
        candidate, mask = _apply(
            audit, signal, chosen["eta"], chosen["apply_domain"]
        )
        axis_results[axis_name] = diagnostics(
            audit, v27_parent(audit), candidate, mask
        )
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=audit["target"].to_numpy(np.float64),
            v27=v27_parent(audit),
            signal=signal,
            candidate=candidate,
            domain3=audit["domain3"].astype(str).to_numpy(),
            game_month=audit["game_month"].to_numpy(np.int16),
            pitcher_id=audit["pitcher_id"].to_numpy(),
            batter_id=audit["batter_id"].to_numpy(),
        )
    gates = {
        "selection_gate": bool(chosen_row["passes_selection_gate"]),
        "outer_gain_at_least_5": axis_results["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": axis_results["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": axis_results["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "replication_gain_positive": axis_results["replication_late_2024"][
            "gain"
        ]
        > 0.0,
        "replication_all_months_positive": axis_results[
            "replication_late_2024"
        ]["positive_month_fraction"]
        == 1.0,
    }
    summary = {
        "protocol": "V31_DYNAMIC_CROSSED_HIERARCHICAL_RESIDUAL_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "selection_candidate_count": int(len(metrics)),
        "selection_gate_count": int(len(eligible)),
        "chosen": chosen,
        "chosen_selection": {
            key: float(chosen_row[key])
            for key in (
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "minimum_domain_gain",
                "mean_abs_shift",
            )
        },
        "axes": axis_results,
        "gates": {key: bool(value) for key, value in gates.items()},
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
        default=Path("artifacts/v31_dynamic_hierarchical_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
