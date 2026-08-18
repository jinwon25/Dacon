"""Row-local dynamic pitcher-state correction above frozen v27.

Official career-to-date ``asof_pitcher_*`` fields are converted into the
current-season sample by subtracting a frozen end-of-prior-season state.  A
pitcher's latest completed-season command state is shrunk to its season league
rate and propagated through a bounded AR(1) transition fitted only on earlier
seasons.  The propagated state is used as a prior that automatically decays as
the official current-season sample grows.

The candidate grid is fixed and deliberately small.  One identical
method/strength/weight/domain recipe must pass 2022 and late-2023 before the
full-2024 and late-2024 audits are opened.  All evaluation features are
row-local; there is no evaluation-row aggregation or external rate input.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v30_diverse_covariance_screen import (
    _cached_v25_axes,
    diagnostics,
    v27_parent,
)
from src.v35_three_stage_multibank import _metadata


TARGET = "control_success"
STATE_SMOOTHING = 200.0
TRANSITION_RIDGE = 1.0
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
CANDIDATES: dict[str, tuple[str, float, float]] = {
    "ar_k30_w025": ("ar", 30.0, 0.25),
    "ar_k30_w050": ("ar", 30.0, 0.50),
    "ar_k100_w025": ("ar", 100.0, 0.25),
    "last_k30_w025": ("last", 30.0, 0.25),
}
CONSENSUS_KEYS = ("signal", "domain")
EPS = 1e-6


def _logit(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def _expit(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def season_latent_states(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Build shrunk, league-centred completed-season pitcher states."""

    stats = frame.groupby(["season", "pitcher_id"], sort=False)[TARGET].agg(
        ["sum", "count"]
    )
    league = frame.groupby("season", sort=False)[TARGET].mean()
    league_rates = {int(year): float(rate) for year, rate in league.items()}
    years = stats.index.get_level_values("season").to_numpy(np.int16)
    prior = np.asarray([league_rates[int(year)] for year in years])
    posterior = (
        stats["sum"].to_numpy(np.float64) + STATE_SMOOTHING * prior
    ) / (stats["count"].to_numpy(np.float64) + STATE_SMOOTHING)
    states = stats.copy()
    states["latent_logit"] = _logit(posterior) - _logit(prior)
    states["reliability"] = stats["count"].to_numpy(np.float64) / (
        stats["count"].to_numpy(np.float64) + STATE_SMOOTHING
    )
    return states, league_rates


def fit_transition(
    states: pd.DataFrame, prediction_year: int
) -> tuple[float, dict[str, float | int]]:
    """Fit a zero-intercept AR(1) using consecutive strictly prior seasons."""

    history = states.loc[
        states.index.get_level_values("season") < int(prediction_year)
    ].reset_index()
    previous = history.rename(
        columns={
            "season": "previous_year",
            "latent_logit": "previous_latent",
            "reliability": "previous_reliability",
        }
    )[
        [
            "pitcher_id",
            "previous_year",
            "previous_latent",
            "previous_reliability",
        ]
    ]
    current = history.rename(
        columns={
            "season": "current_year",
            "latent_logit": "current_latent",
            "reliability": "current_reliability",
        }
    )[
        [
            "pitcher_id",
            "current_year",
            "current_latent",
            "current_reliability",
        ]
    ]
    pairs = current.merge(previous, on="pitcher_id", how="inner")
    pairs = pairs.loc[pairs["current_year"].eq(pairs["previous_year"] + 1)]
    if pairs.empty:
        numerator = denominator = 0.0
        rho = 0.0
    else:
        weight = np.sqrt(
            pairs["current_reliability"].to_numpy(np.float64)
            * pairs["previous_reliability"].to_numpy(np.float64)
        )
        x = pairs["previous_latent"].to_numpy(np.float64)
        y = pairs["current_latent"].to_numpy(np.float64)
        numerator = float(np.sum(weight * x * y))
        denominator = float(np.sum(weight * x * x))
        rho = float(
            np.clip(
                numerator / (denominator + TRANSITION_RIDGE), 0.0, 1.0
            )
        )
    return rho, {
        "transition_pairs": int(len(pairs)),
        "weighted_cross_product": numerator,
        "weighted_square": denominator,
        "rho": rho,
    }


@dataclass(frozen=True)
class CareerState:
    n: pd.Series
    successes: pd.Series


def _end_state(season_rows: pd.DataFrame) -> CareerState:
    last_index = season_rows.groupby("pitcher_id", sort=False)[
        "asof_pitcher_n"
    ].idxmax()
    last = season_rows.loc[
        last_index,
        [
            "pitcher_id",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            TARGET,
        ],
    ]
    n_before = last["asof_pitcher_n"].to_numpy(np.float64)
    successes_before = np.rint(
        n_before * last["asof_pitcher_success_rate"].to_numpy(np.float64)
    )
    ids = last["pitcher_id"].to_numpy()
    return CareerState(
        n=pd.Series(n_before + 1.0, index=ids),
        successes=pd.Series(
            successes_before + last[TARGET].to_numpy(np.float64), index=ids
        ),
    )


def prior_career_states(frame: pd.DataFrame) -> dict[int, CareerState]:
    """Freeze career counts available immediately before each season."""

    output: dict[int, CareerState] = {}
    latest_n = pd.Series(dtype=np.float64)
    latest_successes = pd.Series(dtype=np.float64)
    for year in sorted(frame["season"].astype(int).unique()):
        output[int(year)] = CareerState(
            n=latest_n.copy(), successes=latest_successes.copy()
        )
        state = _end_state(frame.loc[frame["season"].eq(year)])
        latest_n = pd.concat([latest_n, state.n])
        latest_successes = pd.concat([latest_successes, state.successes])
        latest_n = latest_n[~latest_n.index.duplicated(keep="last")]
        latest_successes = latest_successes[
            ~latest_successes.index.duplicated(keep="last")
        ]
    return output


def _latest_state_before(
    states: pd.DataFrame, prediction_year: int
) -> pd.DataFrame:
    history = states.loc[
        states.index.get_level_values("season") < int(prediction_year)
    ].reset_index()
    latest_index = history.groupby("pitcher_id", sort=False)["season"].idxmax()
    return history.loc[
        latest_index, ["pitcher_id", "season", "latent_logit"]
    ].set_index("pitcher_id")


def dynamic_deltas(
    rows: pd.DataFrame,
    prediction_year: int,
    states: pd.DataFrame,
    league_rates: dict[int, float],
    career: CareerState,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Compute prior-vs-global current-season posterior differences per row."""

    rho, transition = fit_transition(states, prediction_year)
    latest = _latest_state_before(states, prediction_year)
    pitcher = rows["pitcher_id"]
    prior_n = pitcher.map(career.n).fillna(0.0).to_numpy(np.float64)
    prior_success = (
        pitcher.map(career.successes).fillna(0.0).to_numpy(np.float64)
    )
    career_n = rows["asof_pitcher_n"].to_numpy(np.float64)
    career_success = np.rint(
        career_n
        * rows["asof_pitcher_success_rate"].fillna(0.0).to_numpy(np.float64)
    )
    current_n = career_n - prior_n
    current_success = career_success - prior_success
    if np.any(current_n < -1e-6):
        raise ValueError(f"negative current-season sample for {prediction_year}")
    if np.any(current_success < -0.01) or np.any(
        current_success - current_n > 0.01
    ):
        raise ValueError(f"invalid reconstructed successes for {prediction_year}")
    current_n = np.maximum(current_n, 0.0)
    current_success = np.clip(current_success, 0.0, current_n)

    league_prior = float(league_rates[prediction_year - 1])
    last_latent = pitcher.map(latest["latent_logit"]).fillna(0.0).to_numpy(
        np.float64
    )
    last_year = pitcher.map(latest["season"]).to_numpy(np.float64)
    known = np.isfinite(last_year)
    gap = np.where(known, prediction_year - last_year, 0.0)
    ar_latent = np.where(known, last_latent * np.power(rho, gap), 0.0)
    prior_probability = {
        "ar": _expit(_logit(league_prior) + ar_latent),
        "last": _expit(_logit(league_prior) + last_latent),
    }

    deltas: dict[str, np.ndarray] = {}
    for method, probability in prior_probability.items():
        for strength in (30.0, 100.0):
            dynamic = (current_success + strength * probability) / (
                current_n + strength
            )
            global_posterior = (
                current_success + strength * league_prior
            ) / (current_n + strength)
            deltas[f"{method}_k{int(strength)}"] = (
                dynamic - global_posterior
            )
    return deltas, {
        **transition,
        "prediction_year": int(prediction_year),
        "league_prior": league_prior,
        "known_prior_state_rate": float(known.mean()),
        "mean_state_gap_known": (
            float(gap[known].mean()) if known.any() else 0.0
        ),
        "current_sample_mean": float(current_n.mean()),
        "correction_mean_abs": {
            name: float(np.mean(np.abs(value)))
            for name, value in deltas.items()
        },
    }


def _load_frame(project: Path) -> pd.DataFrame:
    columns = [
        "season",
        "game_month",
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "balls_before",
        "strikes_before",
        "game_type",
        "pitcher_team_id",
        "batter_team_id",
        TARGET,
    ]
    frame = pd.read_csv(
        project / "data" / "train.csv", usecols=columns, low_memory=False
    )
    missing_rate = frame["asof_pitcher_success_rate"].isna()
    if not frame.loc[missing_rate, "asof_pitcher_n"].eq(0).all():
        raise ValueError("missing pitcher rate at positive sample size")
    frame["asof_pitcher_success_rate"] = frame[
        "asof_pitcher_success_rate"
    ].fillna(0.0)
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("train rows are not season-monotone")
    return _add_domain_and_pressure(frame)


def build_signal_bank(
    rows: pd.DataFrame,
    year: int,
    states: pd.DataFrame,
    league_rates: dict[int, float],
    career: CareerState,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    deltas, audit = dynamic_deltas(
        rows, year, states, league_rates, career
    )
    bank = {
        name: float(weight) * deltas[f"{method}_k{int(strength)}"]
        for name, (method, strength, weight) in CANDIDATES.items()
    }
    return bank, audit


def _frame(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": rows[TARGET].to_numpy(np.float64),
            "game_month": rows["game_month"].to_numpy(np.int16),
            "domain3": rows["domain3"].astype(str).to_numpy(),
        }
    )


def _candidate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    correction: np.ndarray,
    route: str,
) -> tuple[np.ndarray, np.ndarray]:
    active = (
        np.ones(len(frame), dtype=bool)
        if route == "ALL"
        else frame["domain3"].astype(str).eq(route).to_numpy()
    )
    candidate = np.asarray(parent, dtype=np.float64).copy()
    candidate[active] = np.clip(
        candidate[active] + np.asarray(correction)[active], 0.001, 0.999
    )
    return candidate, active


def _screen(
    frame: pd.DataFrame, parent: np.ndarray, bank: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, correction in bank.items():
        for route in ROUTES:
            candidate, active = _candidate(
                frame, parent, correction, route
            )
            result = diagnostics(frame, parent, candidate, active)
            applied_domain = (
                min(result["domain_gains"].values())
                if route == "ALL"
                else result["domain_gains"][route]
            )
            rows.append(
                {
                    "signal": name,
                    "domain": route,
                    "gain": float(result["gain"]),
                    "positive_month_fraction": float(
                        result["positive_month_fraction"]
                    ),
                    "worst_month_gain": float(result["worst_month_gain"]),
                    "minimum_domain_gain": float(
                        result["minimum_domain_gain"]
                    ),
                    "applied_domain_gain": float(applied_domain),
                    "selection_score": float(
                        min(
                            result["gain"],
                            result["worst_month_gain"],
                            applied_domain,
                        )
                    ),
                    "mean_abs_shift": float(result["mean_abs_shift"]),
                }
            )
    return pd.DataFrame(rows)


def select_consensus(
    stage1: pd.DataFrame, stage2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    merged = stage1.merge(
        stage2, on=list(CONSENSUS_KEYS), suffixes=("_2022", "_2023")
    )
    if merged.empty:
        raise ValueError("no shared dynamic-state recipe")
    merged["minimum_gain"] = merged[["gain_2022", "gain_2023"]].min(axis=1)
    merged["minimum_worst_month"] = merged[
        ["worst_month_gain_2022", "worst_month_gain_2023"]
    ].min(axis=1)
    merged["minimum_applied_domain"] = merged[
        ["applied_domain_gain_2022", "applied_domain_gain_2023"]
    ].min(axis=1)
    merged["consensus_score"] = merged[
        ["minimum_gain", "minimum_worst_month", "minimum_applied_domain"]
    ].min(axis=1)
    merged["passes_consensus_gate"] = (
        merged["gain_2022"].gt(0.0)
        & merged["gain_2023"].gt(0.0)
        & merged["positive_month_fraction_2022"].ge(0.75)
        & merged["positive_month_fraction_2023"].eq(1.0)
        & merged["minimum_worst_month"].gt(0.0)
        & merged["minimum_applied_domain"].gt(0.0)
    )
    merged = merged.sort_values(
        ["passes_consensus_gate", "consensus_score", "minimum_gain"],
        ascending=False,
    ).reset_index(drop=True)
    passing = merged.loc[merged["passes_consensus_gate"]]
    chosen = passing.iloc[0] if len(passing) else merged.iloc[0]
    return merged, chosen


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(project)
    states, league_rates = season_latent_states(frame)
    career = prior_career_states(frame)
    year_rows = {
        year: frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    banks: dict[int, dict[str, np.ndarray]] = {}
    state_audits: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        banks[year], state_audits[str(year)] = build_signal_bank(
            year_rows[year], year, states, league_rates, career[year]
        )

    meta22 = _metadata(project, 2022)
    if not np.array_equal(
        meta22["target"], year_rows[2022][TARGET].to_numpy(np.float64)
    ):
        raise ValueError("2022 target/order mismatch")
    frame22 = _frame(year_rows[2022])
    if not np.array_equal(
        frame22["domain3"].astype(str).to_numpy(), meta22["domain"]
    ):
        raise ValueError("2022 domain/order mismatch")
    stage1 = _screen(frame22, meta22["parent"], banks[2022])
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    del raw
    late23 = year_rows[2023]["game_month"].ge(8).to_numpy()
    audit23 = axes["selection_late_2023"]
    if not np.array_equal(
        audit23["target"].to_numpy(np.float64),
        year_rows[2023].loc[late23, TARGET].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 target/order mismatch")
    bank23 = {name: value[late23] for name, value in banks[2023].items()}
    stage2 = _screen(audit23, v27_parent(audit23), bank23)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "domain": str(chosen["domain"]),
    }

    late24 = year_rows[2024]["game_month"].ge(8).to_numpy()
    audits: dict[str, dict[str, object]] = {}
    for axis_name, correction in (
        ("outer_full_2024", banks[2024][recipe["signal"]]),
        (
            "replication_late_2024",
            banks[2024][recipe["signal"]][late24],
        ),
    ):
        audit = axes[axis_name]
        parent = v27_parent(audit)
        candidate, active = _candidate(
            audit, parent, correction, recipe["domain"]
        )
        audits[axis_name] = diagnostics(
            audit, parent, candidate, active
        )
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=audit["target"].to_numpy(np.float64),
            v27=parent,
            correction=correction,
            candidate=candidate,
            active=active,
            domain3=audit["domain3"].astype(str).to_numpy(),
            game_month=audit["game_month"].to_numpy(np.int16),
        )

    gates = {
        "two_origin_consensus": bool(chosen["passes_consensus_gate"]),
        "outer_gain_at_least_5": audits["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": audits["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": audits["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "outer_minimum_domain_nonnegative": audits["outer_full_2024"][
            "minimum_domain_gain"
        ]
        >= 0.0,
        "replication_gain_positive": audits["replication_late_2024"]["gain"]
        > 0.0,
        "replication_month_fraction_at_least_two_thirds": audits[
            "replication_late_2024"
        ]["positive_month_fraction"]
        >= 2.0 / 3.0,
    }
    summary = {
        "protocol": "V51_TWO_ORIGIN_DYNAMIC_PITCHER_STATE_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "candidate_origin": (
            "independent public-repository dynamic-state pattern, "
            "reimplemented above local v27 OOF"
        ),
        "configuration": {
            "state_smoothing": STATE_SMOOTHING,
            "transition_ridge": TRANSITION_RIDGE,
            "transition": "bounded zero-intercept AR(1)",
            "candidates": CANDIDATES,
            "routes": list(ROUTES),
        },
        "state_audits": state_audits,
        "selection": "exact recipe consensus on 2022 and late-2023 only",
        "chosen": {
            **recipe,
            "gain_2022": float(chosen["gain_2022"]),
            "gain_late_2023": float(chosen["gain_2023"]),
            "worst_month_2022": float(chosen["worst_month_gain_2022"]),
            "worst_month_late_2023": float(
                chosen["worst_month_gain_2023"]
            ),
            "consensus_score": float(chosen["consensus_score"]),
        },
        "consensus_candidate_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "audits": audits,
        "gates": {name: bool(value) for name, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "external_rate_used": False,
        "audit_labels_used_for_selection": False,
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
        default=Path("artifacts/v51_dynamic_pitcher_state_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
