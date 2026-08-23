"""Multi-origin low-DOF baseball-context residual transport above v84.

Global calibration changes sign sharply by season, so this family transports
only source-domain-centred residual contrasts for low-cardinality baseball
contexts.  Group/alpha/eta/route selection uses common-wave0 2020->2021 and
2021->2022 transfers.  The frozen recipe is then audited on exact-v84
full-2022->late-2023 and late-2023->full-2024 transfers.

All features are current-row categories.  No player ID, test-row aggregate,
row order, TrackMan current-pitch measurement or Public score is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v30_diverse_covariance_screen import diagnostics


PROTOCOL = "V95_MULTIORIGIN_CONTEXT_TRANSPORT_ABOVE_V84_V1"
TARGET = "control_success"
ALPHAS = (200.0, 1_000.0, 5_000.0)
ETAS = (0.25, 0.50, 1.00)
ROUTES = {
    "R_CORE": ("R_CORE",),
    "F": ("F",),
    "R_CORE_F": ("R_CORE", "F"),
}
CORRECTION_CAP = 0.03


@dataclass(frozen=True)
class GroupSpec:
    name: str
    columns: tuple[str, ...]


GROUP_SPECS = (
    GroupSpec("count", ("count_state",)),
    GroupSpec("count_hands", ("count_state", "hand_matchup")),
    GroupSpec("count_base_out", ("count_state", "base_state", "outs_before")),
    GroupSpec("count_inning", ("count_state", "inning_bucket")),
    GroupSpec("count_runner_pressure", ("count_state", "runner_band", "pressure")),
    GroupSpec("hands_pressure", ("hand_matchup", "pressure")),
    GroupSpec("game_type_count", ("game_type", "count_state")),
    GroupSpec("score_leverage_count", ("score_bucket", "leverage_bucket", "count_state")),
    GroupSpec("pitcher_team_count", ("pitcher_team_id", "count_state")),
    GroupSpec("team_matchup_count", ("pitcher_team_id", "batter_team_id", "count_state")),
)


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.reset_index(drop=True).copy()
    balls = pd.to_numeric(output["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(output["strikes_before"], errors="coerce").fillna(-1).astype(int)
    output["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    output["hand_matchup"] = (
        output["pitcher_hand"].astype("string").fillna("__MISSING__").astype(str)
        + "-"
        + output["batter_hand"].astype("string").fillna("__MISSING__").astype(str)
    )
    output["pressure"] = np.where(
        balls.eq(3), "threeball", np.where(strikes.eq(2), "twostrike", "normal")
    )
    output["inning_bucket"] = pd.cut(
        pd.to_numeric(output["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, np.inf), labels=("early", "middle", "late"),
    ).astype("string").fillna("__MISSING__").astype(str)
    output["runner_band"] = pd.cut(
        pd.to_numeric(output["num_runners_on"], errors="coerce"),
        bins=(-np.inf, 0, 1, np.inf), labels=("empty", "one", "multiple"),
    ).astype("string").fillna("__MISSING__").astype(str)
    output["score_bucket"] = pd.cut(
        pd.to_numeric(output["score_diff_pitcher_team"], errors="coerce"),
        bins=(-np.inf, -3, -1, 1, 3, np.inf),
        labels=("behind4", "behind", "close", "ahead", "ahead4"),
    ).astype("string").fillna("__MISSING__").astype(str)
    output["leverage_bucket"] = pd.cut(
        pd.to_numeric(output["li"], errors="coerce"),
        bins=(-np.inf, 0.75, 1.5, 3.0, np.inf),
        labels=("low", "medium", "high", "very_high"),
    ).astype("string").fillna("__MISSING__").astype(str)
    return output


def group_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, list(columns)].astype("string").fillna("__MISSING__")
    key = values.iloc[:, 0].astype(str)
    for column in values.columns[1:]:
        key = key.str.cat(values[column].astype(str), sep="\x1f")
    return key.to_numpy(str)


def cache_group_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialize target-free domain/context hashes for repeated trials."""

    output = frame.copy()
    for spec in GROUP_SPECS:
        values = output.loc[:, ["domain3", *spec.columns]].copy()
        for column in values:
            values[column] = values[column].astype("string").fillna("__MISSING__")
        output[f"__v95_key_{spec.name}"] = pd.util.hash_pandas_object(
            values, index=False, categorize=True
        ).to_numpy(np.uint64)
    return output


def domain_group_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    for spec in GROUP_SPECS:
        cache_column = f"__v95_key_{spec.name}"
        if spec.columns == columns and cache_column in frame:
            return frame[cache_column].to_numpy(np.uint64)
    return frame["domain3"].astype(str).to_numpy() + "\x1e" + group_keys(frame, columns)


def fit_centered_effect(
    frame: pd.DataFrame,
    parent: np.ndarray,
    columns: tuple[str, ...],
    route: tuple[str, ...],
    alpha: float,
) -> dict[Any, float]:
    parent = np.asarray(parent, dtype=np.float64)
    if parent.shape != (len(frame),):
        raise ValueError("effect parent is not aligned")
    active = frame["domain3"].astype(str).isin(route).to_numpy()
    if not active.any():
        return {}
    target = frame[TARGET].to_numpy(np.float64)
    residual = target - parent
    domain = frame["domain3"].astype(str).to_numpy()
    centered = residual.copy()
    for value in route:
        mask = active & (domain == value)
        if mask.any():
            centered[mask] -= float(residual[mask].mean())
    key = domain_group_keys(frame, columns)
    codes, unique = pd.factorize(key[active], sort=False)
    numerator = np.bincount(codes, weights=centered[active], minlength=len(unique))
    count = np.bincount(codes, minlength=len(unique)).astype(np.float64)
    effect = numerator / (count + float(alpha))
    return {name.item() if isinstance(name, np.generic) else name: float(value) for name, value in zip(unique, effect, strict=True)}


def map_effect(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    effect: dict[Any, float],
) -> np.ndarray:
    key = domain_group_keys(frame, columns)
    return np.asarray(
        [effect.get(value.item() if isinstance(value, np.generic) else value, 0.0) for value in key],
        dtype=np.float64,
    )


def apply_effect(
    frame: pd.DataFrame,
    parent: np.ndarray,
    correction: np.ndarray,
    route: tuple[str, ...],
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if parent.shape != correction.shape or parent.shape != (len(frame),):
        raise ValueError("candidate arrays are not aligned")
    active = frame["domain3"].astype(str).isin(route).to_numpy()
    output = parent.copy()
    output[active] = np.clip(
        parent[active] + float(eta) * np.clip(correction[active], -CORRECTION_CAP, CORRECTION_CAP),
        0.001, 0.999,
    )
    return output, active


def _metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": frame[TARGET].to_numpy(np.float64),
            "game_month": frame["game_month"].to_numpy(np.int16),
            "domain3": frame["domain3"].astype(str).to_numpy(),
        }
    )


def _compact(result: dict[str, Any], route: tuple[str, ...]) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "minimum_domain_gain": float(result["minimum_domain_gain"]),
        "minimum_applied_domain_gain": float(min(result["domain_gains"][value] for value in route)),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def _load_axis(contract_dir: Path, raw: pd.DataFrame, name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with np.load(contract_dir / f"{name}.npz", allow_pickle=False) as saved:
        index = saved["raw_index"].astype(np.int64)
        target = saved["target"].astype(np.float64)
        parent = saved["parent"].astype(np.float64)
        exact = saved["exact_mask"].astype(bool)
        domain = saved["domain3"].astype(str)
    frame = add_context_features(raw.iloc[index].reset_index(drop=True))
    frame["domain3"] = domain
    if not np.array_equal(target, frame[TARGET].to_numpy(np.float64)):
        raise ValueError(f"contract target/order mismatch: {name}")
    return frame, parent, exact


def _transition(
    source_frame: pd.DataFrame,
    source_parent: np.ndarray,
    audit_frame: pd.DataFrame,
    audit_parent: np.ndarray,
    spec: GroupSpec,
    route: tuple[str, ...],
    alpha: float,
    eta: float,
) -> tuple[dict[str, Any], np.ndarray]:
    effect = fit_centered_effect(source_frame, source_parent, spec.columns, route, alpha)
    correction = map_effect(audit_frame, spec.columns, effect)
    candidate, active = apply_effect(audit_frame, audit_parent, correction, route, eta)
    result = diagnostics(_metric_frame(audit_frame), audit_parent, candidate, active)
    result["effect_count"] = int(len(effect))
    result["correction_nonzero_fraction_active"] = float(np.mean(correction[active] != 0.0)) if active.any() else 0.0
    return result, candidate


def _select(table: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    keys = ["group", "route_name", "alpha", "eta"]
    for values, local in table.groupby(keys, observed=True):
        local = local.set_index("axis")
        passed = bool(
            set(local.index) == {"2020_to_2021", "2021_to_2022"}
            and (local["gain"] > 0.0).all()
            and (local["positive_month_fraction"] >= 0.75).all()
            and (local["worst_month_gain"] > -5.0).all()
            and (local["minimum_domain_gain"] >= 0.0).all()
            and (local["minimum_applied_domain_gain"] > 0.0).all()
        )
        robust = float(min(local["gain"].min(), local["worst_month_gain"].min(), local["minimum_applied_domain_gain"].min()))
        records.append(
            {
                **dict(zip(keys, values, strict=True)),
                "source_gate_passed": passed,
                "robust_score": robust,
                "minimum_gain": float(local["gain"].min()),
                "minimum_month_fraction": float(local["positive_month_fraction"].min()),
                "minimum_worst_month_gain": float(local["worst_month_gain"].min()),
                "minimum_applied_domain_gain": float(local["minimum_applied_domain_gain"].min()),
            }
        )
    ranking = pd.DataFrame(records).sort_values(
        ["source_gate_passed", "robust_score", "minimum_gain"], ascending=False
    ).reset_index(drop=True)
    passing = ranking.loc[ranking["source_gate_passed"]]
    chosen = (passing if len(passing) else ranking).iloc[0].to_dict()
    return chosen, ranking


def run(project: Path, contract_dir: Path, output_dir: Path) -> dict[str, Any]:
    project = project.resolve()
    contract_dir = contract_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    common: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
    for year in (2020, 2021, 2022):
        frame, parent, _ = _load_axis(contract_dir, raw, f"common_full_{year}")
        common[year] = (cache_group_keys(frame), parent)
    source_rows: list[dict[str, Any]] = []
    source_details: dict[str, Any] = {}
    for spec in GROUP_SPECS:
        for route_name, route in ROUTES.items():
            for alpha in ALPHAS:
                transported: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int, float]] = {}
                for source_year, audit_year in ((2020, 2021), (2021, 2022)):
                    source_frame, source_parent = common[source_year]
                    audit_frame, audit_parent = common[audit_year]
                    effect = fit_centered_effect(
                        source_frame, source_parent, spec.columns, route, alpha
                    )
                    correction = map_effect(audit_frame, spec.columns, effect)
                    active = audit_frame["domain3"].astype(str).isin(route).to_numpy()
                    axis = f"{source_year}_to_{audit_year}"
                    transported[axis] = (
                        audit_parent, correction, active, len(effect),
                        float(np.mean(correction[active] != 0.0)) if active.any() else 0.0,
                    )
                for eta in ETAS:
                    recipe = f"{spec.name}|{route_name}|a{alpha:g}|e{eta:g}"
                    source_details[recipe] = {}
                    for source_year, audit_year in ((2020, 2021), (2021, 2022)):
                        axis = f"{source_year}_to_{audit_year}"
                        audit_frame = common[audit_year][0]
                        audit_parent, correction, active, effect_count, coverage = transported[axis]
                        candidate, _ = apply_effect(
                            audit_frame, audit_parent, correction, route, eta
                        )
                        result = diagnostics(
                            _metric_frame(audit_frame), audit_parent, candidate, active
                        )
                        result["effect_count"] = int(effect_count)
                        result["correction_nonzero_fraction_active"] = coverage
                        source_details[recipe][axis] = result
                        source_rows.append(
                            {"recipe": recipe, "group": spec.name, "route_name": route_name,
                             "alpha": alpha, "eta": eta, "axis": axis, **_compact(result, route)}
                        )
    source_table = pd.DataFrame(source_rows)
    selected, ranking = _select(source_table)
    source_table.to_csv(output_dir / "source_metrics.csv", index=False)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    source_passed = bool(selected["source_gate_passed"])
    spec = next(value for value in GROUP_SPECS if value.name == selected["group"])
    route = ROUTES[str(selected["route_name"])]
    alpha = float(selected["alpha"])
    eta = float(selected["eta"])

    frame22, parent22, exact22 = _load_axis(contract_dir, raw, "v84_full_2022")
    frame23, parent23, exact23 = _load_axis(contract_dir, raw, "v84_late_2023")
    frame24, parent24, exact24 = _load_axis(contract_dir, raw, "v84_full_2024")
    frame22 = cache_group_keys(frame22)
    frame23 = cache_group_keys(frame23)
    frame24 = cache_group_keys(frame24)
    route22 = frame22["domain3"].astype(str).isin(route).to_numpy()
    if not exact22[route22].all() or not exact23.all() or not exact24.all():
        raise ValueError("selected route includes non-exact v84 rows")
    exact_audits: dict[str, Any] = {}
    result23, candidate23 = _transition(
        frame22, parent22, frame23, parent23, spec, route, alpha, eta
    )
    exact_audits["full_2022_to_late_2023"] = result23
    result24, candidate24 = _transition(
        frame23, parent23, frame24, parent24, spec, route, alpha, eta
    )
    exact_audits["late_2023_to_full_2024"] = result24
    late24 = frame24["game_month"].ge(8).to_numpy()
    late_result = diagnostics(
        _metric_frame(frame24.loc[late24].reset_index(drop=True)),
        parent24[late24], candidate24[late24],
        frame24.loc[late24, "domain3"].astype(str).isin(route).to_numpy(),
    )
    exact_audits["replication_late_2024"] = late_result
    point_gates = {
        "pre2023_common_source_recipe_passed": source_passed,
        "exact_transfer_gains_positive": min(float(result23["gain"]), float(result24["gain"])) > 0.0,
        "exact_month_fraction_at_least_075": min(float(result23["positive_month_fraction"]), float(result24["positive_month_fraction"])) >= 0.75,
        "late_2024_gain_positive": float(late_result["gain"]) > 0.0,
        "late_2024_month_fraction_at_least_075": float(late_result["positive_month_fraction"]) >= 0.75,
        "all_worst_months_above_minus_5": min(float(result23["worst_month_gain"]), float(result24["worst_month_gain"]), float(late_result["worst_month_gain"])) > -5.0,
        "all_minimum_domains_nonnegative": min(float(result23["minimum_domain_gain"]), float(result24["minimum_domain_gain"]), float(late_result["minimum_domain_gain"])) >= 0.0,
    }
    np.savez_compressed(
        output_dir / "exact_audits.npz",
        late23_target=frame23[TARGET].to_numpy(np.float64), late23_parent=parent23,
        late23_candidate=candidate23,
        full24_target=frame24[TARGET].to_numpy(np.float64), full24_parent=parent24,
        full24_candidate=candidate24,
        full24_domain3=frame24["domain3"].astype(str).to_numpy(),
        full24_game_month=frame24["game_month"].to_numpy(np.int16),
    )
    result = {
        "protocol": PROTOCOL,
        "contract_protocol": "V94_MULTI_ORIGIN_CHAMPION_CONTRACT_V1",
        "configuration": {
            "groups": [value.name for value in GROUP_SPECS],
            "alphas": list(ALPHAS), "etas": list(ETAS), "routes": {key: list(value) for key, value in ROUTES.items()},
            "correction_cap": CORRECTION_CAP, "selected": selected,
            "key_cache": "pandas target-free uint64 hash of domain3 plus current-row context columns",
        },
        "selection": "2020->2021 and 2021->2022 common-wave0 only",
        "source_trial_count": int(len(ranking)),
        "source_gate_pass_count": int(ranking["source_gate_passed"].sum()),
        "source_gate_passed": source_passed,
        "source_ranking": ranking.head(30).to_dict(orient="records"),
        "selected_source_details": source_details[f"{spec.name}|{selected['route_name']}|a{alpha:g}|e{eta:g}"],
        "exact_v84_audits": exact_audits,
        "point_gates": {key: bool(value) for key, value in point_gates.items()},
        "point_gates_passed": bool(all(point_gates.values())),
        "eligible_for_packaging": False,
        "promotion_note": "passing point gates would authorize dependence-aware bootstrap and Reality Check",
        "test_csv_read": False, "test_aggregate_used": False,
        "row_local_inference": True, "player_id_feature_used": False,
        "current_pitch_trackman_or_location_used": False,
        "public_score_used_for_recipe_or_weight": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selected": selected, "source_gate_pass_count": result["source_gate_pass_count"],
        "exact": {key: _compact(value, route) for key, value in exact_audits.items()},
        "point_gates": point_gates,
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.contract_dir, args.output_dir)


if __name__ == "__main__":
    main()
