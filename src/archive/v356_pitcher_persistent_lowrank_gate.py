"""Strict-forward pitcher persistence gate for the frozen v50 low-rank signal.

The successful v335 recipe applies v50's pitcher x count/hand correction to F
and R_ANCHOR, while preserving R_CORE.  This experiment asks a narrower
question: can the same frozen direction be used on R_CORE only for pitchers
whose *prior-season, genuinely forward* corrections repeatedly reduced Brier
error?

For audit year Y, pitcher eligibility is computed exclusively from seasons
before Y.  Each evidence season T is itself predicted with low-rank matrices
fit only on seasons before T.  The current audit labels, test rows, public
scores, row order, and evaluation-row aggregates never enter the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v298_recent_rate_denominator_signal import metrics
from src.champion.v50_low_rank_pitcher_context import (
    TARGET,
    fit_source_matrix,
    map_source_matrix,
)
from src.temporal_stable_conditional import _add_domain_and_pressure


PROTOCOL = "V356_PITCHER_PERSISTENT_LOWRANK_GATE_V1"
SIGNAL = "lowrank_s300_r2"
FROZEN_DOSE = 0.50
SOURCE_YEARS = (2020, 2021, 2022, 2023, 2024)
EVIDENCE_YEARS = (2021, 2022, 2023, 2024)
AUDIT_YEARS = (2022, 2023, 2024)


def _load_wave0(path: Path, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        target = saved["target"].astype(np.float64)
        parent = saved["incumbent"].astype(np.float64)
    expected = frame[TARGET].to_numpy(np.float64)
    if not np.array_equal(target, expected):
        raise ValueError(f"wave0 target/order mismatch: {path}")
    return target, parent


def _prior_direction(
    models: dict[int, dict[str, object]], frame: pd.DataFrame, audit_year: int
) -> np.ndarray:
    years = [year for year in sorted(models) if year < audit_year]
    if not years:
        raise ValueError(f"no strictly prior low-rank model for {audit_year}")
    mapped = [map_source_matrix(models[year], frame)[0][SIGNAL] for year in years]
    return np.mean(np.vstack(mapped), axis=0)


def _row_benefit(
    target: np.ndarray, parent: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    shifted = np.clip(parent + FROZEN_DOSE * direction, 0.001, 0.999)
    return np.square(target - parent) - np.square(target - shifted)


def _evidence_table(
    yearly: dict[int, pd.DataFrame],
    wave0_dir: Path,
    models: dict[int, dict[str, object]],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for year in EVIDENCE_YEARS:
        frame = yearly[year]
        target, parent = _load_wave0(
            wave0_dir / f"wave0_incumbent_validate_{year}.npz", frame
        )
        direction = _prior_direction(models, frame, year)
        rcore = frame["domain3"].astype(str).eq("R_CORE").to_numpy()
        benefit = _row_benefit(target, parent, direction)
        part = pd.DataFrame(
            {
                "pitcher_id": frame.loc[rcore, "pitcher_id"].to_numpy(np.int64),
                "year": year,
                "benefit": benefit[rcore],
            }
        )
        grouped = part.groupby(["pitcher_id", "year"], observed=True)["benefit"].agg(
            benefit_sum="sum", benefit_sq_sum=lambda value: float(np.square(value).sum()),
            rows="size",
        )
        pieces.append(grouped.reset_index())
    return pd.concat(pieces, ignore_index=True)


def _pitcher_gate(
    frame: pd.DataFrame,
    evidence: pd.DataFrame,
    audit_year: int,
    rule: str,
) -> np.ndarray:
    prior = evidence.loc[evidence["year"].lt(audit_year)].copy()
    if prior.empty:
        return np.zeros(len(frame), dtype=bool)
    per_pitcher = prior.groupby("pitcher_id", observed=True).agg(
        benefit_sum=("benefit_sum", "sum"),
        benefit_sq_sum=("benefit_sq_sum", "sum"),
        rows=("rows", "sum"),
        observed_years=("year", "nunique"),
        positive_years=("benefit_sum", lambda value: int((value > 0.0).sum())),
        last_year=("year", "max"),
    )
    last = prior.sort_values("year").groupby("pitcher_id", observed=True).tail(1)
    last_positive = last.set_index("pitcher_id")["benefit_sum"].gt(0.0)
    per_pitcher["last_positive"] = last_positive.reindex(per_pitcher.index).fillna(False)
    per_pitcher["z"] = per_pitcher["benefit_sum"] / np.sqrt(
        per_pitcher["benefit_sq_sum"].clip(lower=1e-18)
    )

    if rule == "cumulative_n50":
        selected = (per_pitcher["rows"] >= 50) & (per_pitcher["benefit_sum"] > 0.0)
    elif rule == "cumulative_n150":
        selected = (per_pitcher["rows"] >= 150) & (per_pitcher["benefit_sum"] > 0.0)
    elif rule == "majority_n50":
        selected = (
            (per_pitcher["rows"] >= 50)
            & (per_pitcher["benefit_sum"] > 0.0)
            & (2 * per_pitcher["positive_years"] >= per_pitcher["observed_years"])
        )
    elif rule == "all_years_n50":
        selected = (
            (per_pitcher["rows"] >= 50)
            & (per_pitcher["benefit_sum"] > 0.0)
            & (per_pitcher["positive_years"] == per_pitcher["observed_years"])
        )
    elif rule == "last_and_cumulative_n50":
        selected = (
            (per_pitcher["rows"] >= 50)
            & (per_pitcher["benefit_sum"] > 0.0)
            & per_pitcher["last_positive"]
        )
    elif rule == "z050_n50":
        selected = (per_pitcher["rows"] >= 50) & (per_pitcher["z"] > 0.50)
    else:
        raise ValueError(f"unknown rule: {rule}")
    allowed = set(per_pitcher.index[selected].astype(int))
    return (
        frame["domain3"].astype(str).eq("R_CORE").to_numpy()
        & frame["pitcher_id"].astype(int).isin(allowed).to_numpy()
    )


def _apply(parent: np.ndarray, direction: np.ndarray, active: np.ndarray) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + FROZEN_DOSE * direction[active], 0.001, 0.999
    )
    return output


def run(
    train_csv: Path,
    wave0_dir: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "season", "game_month", "game_type", "pitcher_team_id", "batter_team_id",
        "pitcher_id", "balls_before", "strikes_before", "batter_hand", TARGET,
    ]
    train = _add_domain_and_pressure(
        pd.read_csv(train_csv, usecols=columns, low_memory=False)
    )
    yearly = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in SOURCE_YEARS
    }

    models: dict[int, dict[str, object]] = {}
    for year in SOURCE_YEARS:
        target, parent = _load_wave0(
            wave0_dir / f"wave0_incumbent_validate_{year}.npz", yearly[year]
        )
        models[year] = fit_source_matrix(
            yearly[year], target, parent,
            smoothing_grid=(300.0,), rank_grid=(2,),
        )
    evidence = _evidence_table(yearly, wave0_dir, models)

    full_frames = {
        "full_2022": yearly[2022],
        "full_2023": yearly[2023],
        "full_2024": yearly[2024],
    }
    late23 = full_frames["full_2023"]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": full_frames["full_2022"],
        "late_2023": full_frames["full_2023"].loc[late23].reset_index(drop=True),
        "full_2024": full_frames["full_2024"],
    }
    directions = {
        "full_2022": _prior_direction(models, full_frames["full_2022"], 2022),
        "late_2023": _prior_direction(models, full_frames["full_2023"], 2023)[late23],
        "full_2024": _prior_direction(models, full_frames["full_2024"], 2024),
    }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            for name in frames
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        v345_parent_2024 = saved["candidate_full_2024"].astype(np.float64)

    rules = (
        "cumulative_n50", "cumulative_n150", "majority_n50",
        "all_years_n50", "last_and_cumulative_n50", "z050_n50",
    )
    results: dict[str, Any] = {}
    for rule in rules:
        rule_results: dict[str, Any] = {}
        for axis, year in (("full_2022", 2022), ("late_2023", 2023), ("full_2024", 2024)):
            active = _pitcher_gate(frames[axis], evidence, year, rule)
            candidate = _apply(parents[axis], directions[axis], active)
            rule_results[axis] = metrics(
                frames[axis], frames[axis][TARGET].to_numpy(np.float64),
                parents[axis], candidate, active,
            )
        results[rule] = rule_results

    eligible = []
    for rule in rules:
        source = results[rule]
        if (
            source["full_2022"]["gain"] > 0.0
            and source["late_2023"]["gain"] > 0.0
            and source["full_2022"]["positive_month_fraction"] >= 0.75
            and source["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
            and min(
                source["full_2022"]["worst_month_gain"],
                source["late_2023"]["worst_month_gain"],
            ) > -3.0
        ):
            eligible.append(rule)
    selected_rule = max(
        eligible or list(rules),
        key=lambda rule: (
            min(results[rule]["full_2022"]["gain"], results[rule]["late_2023"]["gain"]),
            np.mean([results[rule]["full_2022"]["gain"], results[rule]["late_2023"]["gain"]]),
            -rules.index(rule),
        ),
    )

    active24 = _pitcher_gate(frames["full_2024"], evidence, 2024, selected_rule)
    v345_candidate_2024 = _apply(v345_parent_2024, directions["full_2024"], active24)
    locked_vs_v345 = metrics(
        frames["full_2024"], frames["full_2024"][TARGET].to_numpy(np.float64),
        v345_parent_2024, v345_candidate_2024, active24,
    )
    source_pass = selected_rule in eligible
    locked_pass = bool(
        locked_vs_v345["gain"] >= 1.0
        and locked_vs_v345["positive_month_fraction"] >= 0.75
        and locked_vs_v345["worst_month_gain"] > -5.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=v345_parent_2024,
        candidate_full_2024=v345_candidate_2024,
        direction_full_2024=directions["full_2024"],
        active_full_2024=active24,
    )
    evidence.to_csv(output_dir / "pitcher_year_evidence.csv", index=False)
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "selected_rule_from_sources": selected_rule,
        "eligible_source_rules": eligible,
        "frozen_recipe": {
            "signal": SIGNAL,
            "dose": FROZEN_DOSE,
            "route": "R_CORE only; v335 F and R_ANCHOR preserved",
        },
        "rule_results_vs_v335": results,
        "locked_full_2024_vs_v345": locked_vs_v345,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "strictly_prior_evidence_for_each_audit_year": True,
            "each_evidence_year_uses_only_earlier_lowrank_models": True,
            "v50_signal_and_dose_frozen": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--wave0-dir", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.wave0_dir, args.v335_axes, args.v345_axes,
        args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
