"""Run dependence- and selection-aware local evaluation for v14/v15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import brier_decomposition
from src.robust_local_evaluation import (
    circular_block_bootstrap,
    crossed_pigeonhole_bootstrap,
    grouped_gain_table,
    leave_one_team_out_summary,
    one_way_cluster_bootstrap,
    paired_score_summary,
    white_reality_check,
)
from src.temporal_stable_conditional import _add_domain_and_pressure


ANCHOR_WEIGHT = 0.20
ANCHOR_RELIABILITY = 0.90
PARENT_F_EXACT_WEIGHT = 0.75
CANDIDATE_ALPHA = {"v14": 0.15, "v15": 0.35}


def _candidate_fold(
    project: Path,
    train: pd.DataFrame,
    year: int,
    alpha: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    exact_path = (
        project
        / "artifacts"
        / "recent_shared_exact_asof_20260815_02"
        / f"recent_shared_o{year}.npz"
    )
    anchor_path = (
        project
        / "artifacts"
        / "v14_multiseason_anchor_20260815_01"
        / f"anchor_ridge_o{year}.npz"
    )
    with np.load(exact_path) as exact, np.load(anchor_path) as anchor:
        index = exact["train_index"].astype(np.int64)
        rows = train.iloc[index].reset_index(drop=True)
        target = exact["target"].astype(np.float64)
        expected = rows["control_success"].to_numpy(np.float64)
        if not np.array_equal(target, expected):
            raise ValueError(f"target/order mismatch for {year}")
        base = exact["base"].astype(np.float64)
        exact_lgb = exact["exact_lgb"].astype(np.float64)
        trend_lgb = exact["trend_lgb"].astype(np.float64)

        finals = rows["domain3"].eq("F").to_numpy()
        anchor_mask = rows["domain3"].eq("R_ANCHOR").to_numpy()
        if year >= 2023:
            component_path = (
                project
                / "artifacts"
                / "v14_component_audit_20260815_01"
                / f"v13_components_o{year}.npz"
            )
            with np.load(component_path) as component:
                incumbent = component["v13"].astype(np.float64)
        else:
            # R_CORE is unchanged by both candidates, so its exact incumbent
            # value cancels in the paired loss difference.  R_ANCHOR equals
            # base and the submitted v13 F recipe is reconstructed exactly.
            incumbent = base.copy()
            incumbent[finals] = base[finals] + PARENT_F_EXACT_WEIGHT * (
                exact_lgb[finals] - base[finals]
            )

        anchor_positions = np.flatnonzero(anchor_mask)
        if len(anchor_positions) != len(anchor["base"]):
            raise ValueError(f"anchor cache length mismatch for {year}")
        if not np.allclose(anchor["base"], incumbent[anchor_mask], atol=1e-12):
            raise ValueError(f"anchor incumbent mismatch for {year}")
        reliable = (anchor["pitcher_rel"] >= ANCHOR_RELIABILITY) & (
            anchor["batter_rel"] >= ANCHOR_RELIABILITY
        )

        candidate = incumbent.copy()
        selected_anchor = anchor_positions[reliable]
        candidate[selected_anchor] = anchor["base"][reliable] + ANCHOR_WEIGHT * (
            anchor["prediction"][reliable] - anchor["base"][reliable]
        )
        candidate[finals] = incumbent[finals] + PARENT_F_EXACT_WEIGHT * alpha * (
            trend_lgb[finals] - exact_lgb[finals]
        )
    return rows, target, incumbent, np.clip(candidate, 1e-6, 1.0 - 1e-6)


def _calibration_change(
    target: np.ndarray, candidate: np.ndarray, incumbent: np.ndarray
) -> dict[str, float]:
    candidate_decomposition = brier_decomposition(target, candidate, n_bins=20)
    incumbent_decomposition = brier_decomposition(target, incumbent, n_bins=20)
    target_rate = float(target.mean())
    return {
        "candidate_mean_prediction": float(candidate.mean()),
        "incumbent_mean_prediction": float(incumbent.mean()),
        "target_rate": target_rate,
        "candidate_absolute_mean_bias": abs(float(candidate.mean()) - target_rate),
        "incumbent_absolute_mean_bias": abs(float(incumbent.mean()) - target_rate),
        "binned_reliability_improvement": float(
            incumbent_decomposition["reliability"]
            - candidate_decomposition["reliability"]
        ),
        "binned_resolution_change": float(
            candidate_decomposition["resolution"]
            - incumbent_decomposition["resolution"]
        ),
    }


def _contribution_by_domain(
    rows: pd.DataFrame,
    target: np.ndarray,
    candidate: np.ndarray,
    incumbent: np.ndarray,
) -> list[dict[str, Any]]:
    improvement = np.square(incumbent - target) - np.square(candidate - target)
    reference = float(target.mean() * (1.0 - target.mean()))
    output = []
    for domain, index in rows.groupby("domain3", observed=True).groups.items():
        positions = np.asarray(index, dtype=np.int64)
        subgroup = paired_score_summary(
            target[positions], candidate[positions], incumbent[positions]
        )
        contribution = 100_000.0 * float(improvement[positions].sum()) / (
            len(target) * reference
        )
        output.append(
            {
                "domain": domain,
                "n_rows": int(len(positions)),
                "overall_gain_contribution": contribution,
                "subgroup_unclipped_bss_equivalent_gain": subgroup[
                    "unclipped_bss_equivalent_gain"
                ],
            }
        )
    return sorted(output, key=lambda item: str(item["domain"]))


def _evaluate_fold(
    name: str,
    year: int,
    rows: pd.DataFrame,
    target: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    n_resamples = int(config["bootstrap_resamples"])
    seed = int(config["seed"]) + year
    summary = paired_score_summary(target, candidate, incumbent)
    month = grouped_gain_table(
        target, candidate, incumbent, rows["game_month"]
    )
    bootstrap: dict[str, Any] = {
        "pitcher": one_way_cluster_bootstrap(
            target,
            candidate,
            incumbent,
            rows["pitcher_id"],
            n_resamples=n_resamples,
            seed=seed,
        ),
        "batter": one_way_cluster_bootstrap(
            target,
            candidate,
            incumbent,
            rows["batter_id"],
            n_resamples=n_resamples,
            seed=seed + 100,
        ),
        "pitcher_x_batter": crossed_pigeonhole_bootstrap(
            target,
            candidate,
            incumbent,
            rows["pitcher_id"],
            rows["batter_id"],
            n_resamples=n_resamples,
            seed=seed + 200,
        ),
    }
    for block_size in config["sequential_block_sizes"]:
        bootstrap[f"block_{block_size}"] = circular_block_bootstrap(
            target,
            candidate,
            incumbent,
            block_size=int(block_size),
            n_resamples=n_resamples,
            seed=seed + int(block_size),
        )
    p05_values = [float(value["p05"]) for value in bootstrap.values()]
    probabilities = [float(value["prob_positive"]) for value in bootstrap.values()]
    leave_team = leave_one_team_out_summary(
        target,
        candidate,
        incumbent,
        rows["pitcher_team_id"],
        rows["batter_team_id"],
    )
    return {
        "candidate": name,
        "year": year,
        "score": summary,
        "calibration": _calibration_change(target, candidate, incumbent),
        "month": {
            "positive_fraction": float((month["gain"] > 0.0).mean()),
            "median_gain": float(month["gain"].median()),
            "worst_gain": float(month["gain"].min()),
            "worst_month": int(month.loc[month["gain"].idxmin(), "group"]),
            "rows": month.to_dict(orient="records"),
        },
        "domain": _contribution_by_domain(
            rows, target, candidate, incumbent
        ),
        "leave_one_team_out": leave_team,
        "bootstrap": bootstrap,
        "minimum_resampling_p05": min(p05_values),
        "minimum_resampling_probability": min(probabilities),
    }


def _gate(
    folds: list[dict[str, Any]],
    reality_check: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    latest = max(folds, key=lambda fold: int(fold["year"]))
    post_break_year = int(config["post_break_year"])
    post_break = [fold for fold in folds if int(fold["year"]) >= post_break_year]
    gate = config["gates"]
    diagnostics = {
        "historical_strict_resampling": all(
            fold["minimum_resampling_p05"] > 0.0
            and fold["minimum_resampling_probability"]
            >= float(gate["minimum_cluster_improvement_probability"])
            for fold in folds
        ),
        "post_break_resampling": all(
            fold["minimum_resampling_p05"] > 0.0
            and fold["minimum_resampling_probability"]
            >= float(gate["minimum_cluster_improvement_probability"])
            for fold in post_break
        ),
    }
    checks = {
        "all_year_direction": all(
            fold["score"]["unclipped_bss_equivalent_gain"] > 0.0
            for fold in folds
        ),
        "post_break_direction": all(
            fold["score"]["unclipped_bss_equivalent_gain"] > 0.0
            for fold in post_break
        ),
        "latest_all_resampling_p05_positive": latest[
            "minimum_resampling_p05"
        ]
        > 0.0,
        "latest_all_resampling_probability": latest[
            "minimum_resampling_probability"
        ]
        >= float(gate["minimum_cluster_improvement_probability"]),
        "latest_month_coverage": latest["month"]["positive_fraction"]
        >= float(gate["minimum_month_positive_fraction"]),
        "latest_leave_one_team_nonnegative": latest["leave_one_team_out"][
            "minimum_gain"
        ]
        >= -1e-12,
        "final_family_reality_check": reality_check["p_value"]
        <= float(gate["reality_check_research_alpha"]),
    }
    statistical_evidence = bool(
        all(checks.values()) and diagnostics["post_break_resampling"]
    )
    return {
        "checks": checks,
        "diagnostics": diagnostics,
        "statistical_evidence_gate": statistical_evidence,
        "independent_confirmation_gate": False,
        "promotion_ready": False,
        "reason": (
            "2024 has been repeatedly reused for model development; statistical "
            "resampling cannot restore an independent confirmation set."
        ),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Local evaluation v2 — dependence and selection aware",
        "",
        "## Fold evidence",
        "",
        "| candidate | year | unclipped BSS-eq Δ | official BSS Δ | month + | worst month | min resampling p05 | min P(+) | leave-one-team min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in result["candidates"]:
        for fold in candidate["folds"]:
            score = fold["score"]
            month = fold["month"]
            lines.append(
                f"| {candidate['candidate']} | {fold['year']} | "
                f"{score['unclipped_bss_equivalent_gain']:+.4f} | "
                f"{score['official_bss_gain']:+.4f} | "
                f"{month['positive_fraction']:.1%} | {month['worst_gain']:+.4f} | "
                f"{fold['minimum_resampling_p05']:+.4f} | "
                f"{fold['minimum_resampling_probability']:.1%} | "
                f"{fold['leave_one_team_out']['minimum_gain']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Selection-aware gate",
            "",
            f"- 2024 registry reuse count (lower bound): **{result['audit']['registered_2024_evaluations']}**",
            f"- Final-family Reality Check p-value: **{result['reality_check']['p_value']:.4f}**",
            "- Reality Check scope: v14/v15 final family only; earlier discarded trials are not corrected.",
            "- Independent confirmation: **FAIL by construction** because 2024 is development-contaminated.",
            "",
            "| candidate | statistical evidence | independent confirmation | promotion ready | failed checks |",
            "|---|:---:|:---:|:---:|---|",
        ]
    )
    for candidate in result["candidates"]:
        gate = candidate["gate"]
        failed = [name for name, passed in gate["checks"].items() if not passed]
        if not gate["diagnostics"]["post_break_resampling"]:
            failed.append("post_break_resampling")
        lines.append(
            f"| {candidate['candidate']} | {gate['statistical_evidence_gate']} | "
            f"{gate['independent_confirmation_gate']} | {gate['promotion_ready']} | "
            f"{', '.join(failed) if failed else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `unclipped BSS-eq Δ` preserves paired Brier information even when both historical official scores clip to zero.",
            "- `official BSS Δ` applies the competition's zero floor and is shown separately.",
            "- The minimum resampling figures combine pitcher, batter, crossed pitcher×batter, and 500/2,000/5,000-pitch circular block bootstraps.",
            "- Bootstrap intervals describe sampling/dependence sensitivity, not uncertainty from trying many model recipes.",
            "- A new Public result remains a deployment probe, not an independent scientific confirmation.",
            "- A leave-one-team result of exactly zero is non-degradation: removing the only modified team/domain can make candidate and incumbent identical.",
            "",
            "## Primary references",
            "",
            "- DACON official evaluation: https://dacon.io/competitions/official/236743/overview/evaluation",
            "- Murphy (1973), Brier reliability-resolution-uncertainty decomposition: https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2",
            "- Tashman (2000), rolling-origin and multiple out-of-sample periods: https://doi.org/10.1016/S0169-2070(00)00065-0",
            "- Politis & Romano (1994), stationary/block bootstrap for dependent observations: https://doi.org/10.1080/01621459.1994.10476870",
            "- Owen (2007), pigeonhole bootstrap for crossed random effects: https://doi.org/10.1214/07-AOAS122",
            "- Diebold & Mariano (1995), paired predictive-accuracy comparison under dependent errors: https://doi.org/10.1080/07350015.1995.10524599",
            "- White (2000), Reality Check for data snooping: https://doi.org/10.1111/1468-0262.00152",
            "- Hansen, Lunde & Nason (2011), Model Confidence Set: https://doi.org/10.3982/ECTA5771",
            "- Cawley & Talbot (2010), model-selection overfitting and evaluation bias: https://www.jmlr.org/papers/v11/cawley10a.html",
            "",
        ]
    )
    return "\n".join(lines)


def run(project: Path, output_json: Path, output_markdown: Path) -> dict[str, Any]:
    project = project.resolve()
    config = json.loads(
        (project / "configs" / "local_evaluation_v2.json").read_text(
            encoding="utf-8"
        )
    )
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    cache: dict[str, dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for name, alpha in CANDIDATE_ALPHA.items():
        cache[name] = {
            year: _candidate_fold(project, train, year, alpha)
            for year in (2022, 2023, 2024)
        }

    latest_target = cache["v14"][2024][1]
    latest_incumbent = cache["v14"][2024][2]
    improvement_matrix = np.column_stack(
        [
            np.square(latest_incumbent - latest_target)
            - np.square(cache[name][2024][3] - latest_target)
            for name in CANDIDATE_ALPHA
        ]
    )
    reality_check = white_reality_check(
        latest_target,
        improvement_matrix,
        block_size=2000,
        n_resamples=int(config["bootstrap_resamples"]),
        seed=int(config["seed"]) + 999,
    )
    experiment_registry = pd.read_csv(project / "reports" / "experiments.csv")
    registered_2024 = int(
        experiment_registry["validation_split"].astype(str).str.contains("2024").sum()
    )

    candidates = []
    for name in CANDIDATE_ALPHA:
        folds = [
            _evaluate_fold(name, year, *cache[name][year], config)
            for year in (2022, 2023, 2024)
        ]
        candidates.append(
            {
                "candidate": name,
                "f_trend_alpha": CANDIDATE_ALPHA[name],
                "folds": folds,
                "gate": _gate(folds, reality_check, config),
            }
        )
    result = {
        "protocol": config["protocol"],
        "metric_note": (
            "Primary comparison is paired unclipped BSS-equivalent gain; official "
            "zero-clipped BSS is reported separately."
        ),
        "candidates": candidates,
        "reality_check": reality_check,
        "audit": {
            "registered_2024_evaluations": registered_2024,
            "registered_experiment_rows": int(len(experiment_registry)),
            "confirmation_2024_is_virgin": False,
        },
        "config": config,
    }
    output_json = (project / output_json).resolve()
    output_markdown = (project / output_markdown).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "output_json": str(output_json),
                "output_markdown": str(output_markdown),
                "reality_check_p_value": result["reality_check"]["p_value"],
                "candidate_gates": {
                    item["candidate"]: item["gate"] for item in candidates
                },
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/local_evaluation_v2_20260815.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/local_evaluation_v2_20260815.md"),
    )
    args = parser.parse_args()
    run(args.project, args.output_json, args.output_markdown)


if __name__ == "__main__":
    main()
