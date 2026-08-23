"""Rebuild and audit a fine-pitch failure prior over the exact v104 parent.

Only historical official train rows receive TrackMan pitch labels.  For target
season T, all outcome and pitch-selection tables use seasons strictly before
T.  The current audit pitch type is never read.  The public reference recipe
is treated as a frozen hypothesis, not as independent holdout evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v103_fixed_union_robust import _axis_metrics, _robust_axis
from src.v104_source_stability_mask import _point_pass
from src.v110_v104_cross_architecture_rebase import apply_correction, fit_alpha
from src.v77_team_oof_constrained_blend import single_candidate_headroom
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V113_FINE_PITCH_FAILURE_PRIOR_V104_V1"
COMPONENTS = ("reverse", "middle", "wayoff")
AXES = ("full_2022", "late_2023", "full_2024", "late_2024")
SOURCE_AXES = ("full_2022", "late_2023")
RATE_COLUMNS = {
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
}
PITCH_TYPE_NORMALISATION = {
    "Changeup": "ChangeUp",
    "Four-Seam": "Fastball",
    "SInker": "Sinker",
}


def reconstruct_failure_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover labelled-pitch failure flags from the following ASOF snapshot."""
    n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(float)
    next_n = (
        frame.groupby("pitcher_id", sort=False)["asof_pitcher_n"]
        .shift(-1)
        .to_numpy(float)
    )
    valid = np.isfinite(n) & np.isfinite(next_n) & np.isclose(next_n, n + 1.0)
    result: dict[str, np.ndarray] = {}
    for name, column in RATE_COLUMNS.items():
        rate = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        next_rate = (
            frame.groupby("pitcher_id", sort=False)[column]
            .shift(-1)
            .to_numpy(float)
        )
        local_valid = valid & np.isfinite(rate) & np.isfinite(next_rate)
        increment = np.rint(next_n * next_rate) - np.rint(n * rate)
        values = np.full(len(frame), np.nan, dtype=np.float64)
        values[local_valid] = np.clip(increment[local_valid], 0.0, 1.0)
        result[name] = values
        valid &= local_valid
    target = pd.to_numeric(frame["control_success"], errors="raise").to_numpy(float)
    wayoff = np.full(len(frame), np.nan, dtype=np.float64)
    wayoff[valid] = (
        (target[valid] == 0.0)
        & (result["reverse"][valid] == 0.0)
        & (result["middle"][valid] == 0.0)
    ).astype(float)
    result["wayoff"] = wayoff
    return pd.DataFrame(result, index=frame.index)


def _wide_counts(
    frame: pd.DataFrame, keys: list[str], pitch_types: list[str]
) -> pd.DataFrame:
    table = (
        frame.groupby([*keys, "pitch_type_fine"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=pitch_types, fill_value=0)
    )
    table.columns = [f"n__{name}" for name in pitch_types]
    return table.reset_index()


def failure_selection_delta(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    component: str,
    pitch_types: list[str],
    *,
    outcome_shrink: float,
    selection_shrink: float,
) -> np.ndarray:
    """Estimate conditional-minus-marginal expected failure tendency."""
    if component not in COMPONENTS or history.empty:
        raise ValueError("unknown component or empty history")
    label = f"failure__{component}"
    work = history.copy()
    season_mean = work.groupby("season", observed=True)[label].transform("mean")
    work["relative"] = work[label].to_numpy(float) - season_mean.to_numpy(float)

    grouped = work.groupby("pitch_type_fine", observed=True)["relative"].agg(
        ["sum", "count"]
    )
    global_relative = {
        name: float(grouped.loc[name, "sum"] / grouped.loc[name, "count"])
        if name in grouped.index and grouped.loc[name, "count"] > 0
        else 0.0
        for name in pitch_types
    }
    outcome = (
        work.groupby(["pitcher_id", "pitch_type_fine"], observed=True)["relative"]
        .agg(["sum", "count"])
        .reset_index()
    )
    outcome["value"] = [
        (float(total) + outcome_shrink * global_relative[str(kind)])
        / (float(count) + outcome_shrink)
        for total, count, kind in zip(
            outcome["sum"], outcome["count"], outcome["pitch_type_fine"]
        )
    ]
    outcome = (
        outcome.pivot(index="pitcher_id", columns="pitch_type_fine", values="value")
        .reindex(columns=pitch_types)
    )
    outcome.columns = [f"value__{name}" for name in pitch_types]
    outcome = outcome.reset_index()

    global_mix = work["pitch_type_fine"].value_counts(normalize=True)
    overall = _wide_counts(work, ["pitcher_id"], pitch_types)
    total = overall[[f"n__{name}" for name in pitch_types]].sum(axis=1).to_numpy(float)
    for name in pitch_types:
        overall[f"mix__{name}"] = (
            overall[f"n__{name}"].to_numpy(float)
            + selection_shrink * float(global_mix.get(name, 0.0))
        ) / (total + selection_shrink)

    keys = ["pitcher_id", "batter_hand", "count_state"]
    conditional = _wide_counts(work, keys, pitch_types)
    conditional_total = conditional[
        [f"n__{name}" for name in pitch_types]
    ].sum(axis=1).to_numpy(float)
    conditional = conditional.merge(
        overall[["pitcher_id", *[f"mix__{name}" for name in pitch_types]]],
        on="pitcher_id",
        how="left",
        validate="many_to_one",
    )
    for name in pitch_types:
        backing = conditional[f"mix__{name}"].fillna(
            float(global_mix.get(name, 0.0))
        )
        conditional[f"mix__{name}"] = (
            conditional[f"n__{name}"].to_numpy(float)
            + selection_shrink * backing.to_numpy(float)
        ) / (conditional_total + selection_shrink)

    marginal = overall[
        ["pitcher_id", *[f"mix__{name}" for name in pitch_types]]
    ].merge(outcome, on="pitcher_id", how="left", validate="one_to_one")
    marginal["expected"] = sum(
        marginal[f"mix__{name}"].fillna(0.0)
        * marginal[f"value__{name}"].fillna(global_relative[name])
        for name in pitch_types
    )

    query = rows[["pitcher_id", "batter_hand", "count_state"]].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(
        conditional[keys + [f"mix__{name}" for name in pitch_types]],
        on=keys,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    query = query.merge(
        outcome, on="pitcher_id", how="left", sort=False, validate="many_to_one"
    )
    query = query.merge(
        marginal[["pitcher_id", "expected"]],
        on="pitcher_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    conditional_expected = np.zeros(len(query), dtype=np.float64)
    for name in pitch_types:
        mix = query[f"mix__{name}"].fillna(float(global_mix.get(name, 0.0)))
        value = query[f"value__{name}"].fillna(global_relative[name])
        conditional_expected += mix.to_numpy(float) * value.to_numpy(float)
    delta = conditional_expected - query["expected"].fillna(0.0).to_numpy(float)
    return delta[np.argsort(query["_order"].to_numpy())]


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def run(
    train_csv: Path,
    trackman_csv: Path,
    alignment_npz: Path,
    contract_dir: Path,
    v104_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    raw["count_state"] = (
        raw["balls_before"].astype(np.int8) * 3 + raw["strikes_before"].astype(np.int8)
    )
    failure = reconstruct_failure_components(raw)

    trackman = pd.read_csv(
        trackman_csv, usecols=["season", "tagged_pitch_type"], low_memory=False
    )
    with np.load(alignment_npz, allow_pickle=False) as saved:
        main_index = saved["main_index"].astype(np.int64)
        trackman_index = saved["trackman_index"].astype(np.int64)
        aligned_season = saved["season"].astype(np.int16)
    if len(np.unique(main_index)) != len(main_index):
        raise ValueError("main alignment is not one-to-one")
    if not np.array_equal(raw.iloc[main_index]["season"].to_numpy(np.int16), aligned_season):
        raise ValueError("main/alignment season mismatch")
    if not np.array_equal(
        trackman.iloc[trackman_index]["season"].to_numpy(np.int16), aligned_season
    ):
        raise ValueError("TrackMan/alignment season mismatch")

    pitch_types = [str(value) for value in config["fine_pitch_types"]]
    fine = (
        trackman.iloc[trackman_index]["tagged_pitch_type"]
        .astype(str)
        .replace(PITCH_TYPE_NORMALISATION)
    )
    fine = fine.where(fine.isin(pitch_types[:-1]), pitch_types[-1]).to_numpy(str)
    history = raw.iloc[main_index][
        ["season", "pitcher_id", "batter_hand", "count_state"]
    ].reset_index(drop=True)
    history["pitch_type_fine"] = fine
    for component in COMPONENTS:
        history[f"failure__{component}"] = failure.iloc[main_index][component].to_numpy(float)
    history = history.dropna(
        subset=[f"failure__{component}" for component in COMPONENTS]
    ).reset_index(drop=True)

    exact = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    late24_mask = exact["full_2024"]["game_month"].astype(np.int16) >= 8
    exact["late_2024"] = _slice_axis(exact["full_2024"], late24_mask)
    frames = {
        "full_2022": raw.loc[raw["season"].eq(2022)].reset_index(drop=True),
        "late_2023": raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": raw.loc[raw["season"].eq(2024)].reset_index(drop=True),
    }
    frames["late_2024"] = frames["full_2024"].loc[late24_mask].reset_index(drop=True)
    full_rows = {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    component_weight = {
        name: float(config["component_weights"][name]) for name in COMPONENTS
    }
    correction_year: dict[int, np.ndarray] = {}
    component_year: dict[int, dict[str, np.ndarray]] = {}
    component_diagnostics: dict[str, Any] = {}
    for year, rows in full_rows.items():
        source = history.loc[history["season"].lt(year)].reset_index(drop=True)
        parts = {
            component: failure_selection_delta(
                source,
                rows,
                component,
                pitch_types,
                outcome_shrink=float(config["outcome_shrink"]),
                selection_shrink=float(config["selection_shrink"]),
            )
            for component in COMPONENTS
        }
        component_year[year] = {
            name: np.asarray(value, dtype=np.float64) for name, value in parts.items()
        }
        combined = sum(component_weight[name] * parts[name] for name in COMPONENTS)
        correction_year[year] = np.asarray(combined, dtype=np.float64)
        component_diagnostics[str(year)] = {
            "history_rows": int(len(source)),
            "component_rms": {
                name: float(np.sqrt(np.mean(np.square(value))))
                for name, value in parts.items()
            },
            "combined_mean": float(correction_year[year].mean()),
            "combined_rms": float(np.sqrt(np.mean(np.square(correction_year[year])))),
        }

    late23_mask = full_rows[2023]["game_month"].ge(8).to_numpy()
    corrections = {
        "full_2022": correction_year[2022],
        "late_2023": correction_year[2023][late23_mask],
        "full_2024": correction_year[2024],
        "late_2024": correction_year[2024][late24_mask],
    }
    route_domains = [str(value) for value in config["route_domains"]]
    corrections = {
        name: np.where(
            np.isin(exact[name]["domain3"].astype(str), route_domains),
            value,
            0.0,
        )
        for name, value in corrections.items()
    }
    with np.load(v104_dir / "selected_axes.npz") as saved:
        v104 = {name: saved[name].astype(np.float64) for name in AXES}
    for name in AXES:
        if not (
            len(exact[name]["target"]) == len(v104[name]) == len(corrections[name])
        ):
            raise ValueError(f"axis length mismatch: {name}")
        if not np.array_equal(
            exact[name]["target"], frames[name]["control_success"].to_numpy(float)
        ):
            raise ValueError(f"contract/train target mismatch: {name}")

    rebased = {name: {**exact[name], "parent": v104[name]} for name in AXES}
    alpha = fit_alpha(
        [exact[name]["target"][exact[name]["exact_mask"].astype(bool)] for name in SOURCE_AXES],
        [v104[name][exact[name]["exact_mask"].astype(bool)] for name in SOURCE_AXES],
        [corrections[name][exact[name]["exact_mask"].astype(bool)] for name in SOURCE_AXES],
        float(config["alpha_cap"]),
    )
    candidate = {
        name: apply_correction(v104[name], corrections[name], alpha) for name in AXES
    }
    metrics = {name: _axis_metrics(rebased[name], candidate[name]) for name in AXES}
    headroom = {
        name: single_candidate_headroom(
            exact[name]["target"][exact[name]["exact_mask"].astype(bool)],
            v104[name][exact[name]["exact_mask"].astype(bool)],
            apply_correction(v104[name], corrections[name], 1.0)[
                exact[name]["exact_mask"].astype(bool)
            ],
        )
        for name in AXES
    }
    gate = config["selection_gate"]
    point_pass = {name: _point_pass(value, gate) for name, value in metrics.items()}
    family_alpha = sorted(set(float(value) for value in config["family_alpha"]) | {alpha})
    family = {
        name: [apply_correction(v104[name], corrections[name], value) for value in family_alpha]
        for name in AXES
    }
    robust = {
        name: _robust_axis(
            frames[name], rebased[name], candidate[name], family[name], config,
            200 + 10 * index,
        )
        for index, name in enumerate(AXES)
    }
    robust_pass = {
        name: bool(
            all(
                value[key]["p05"] > 0.0
                for key in ("pitcher", "crossed_pitcher_batter", "chronological_block")
            )
            and value["reality_check"]["p_value"] <= float(gate["reality_check_alpha"])
        )
        for name, value in robust.items()
    }
    eligible = bool(all(point_pass.values()) and all(robust_pass.values()))
    np.savez_compressed(output_dir / "correction_axes.npz", **corrections)
    component_axes: dict[str, np.ndarray] = {}
    for component in COMPONENTS:
        component_axes[f"{component}__full_2022"] = component_year[2022][component]
        component_axes[f"{component}__late_2023"] = component_year[2023][component][late23_mask]
        component_axes[f"{component}__full_2024"] = component_year[2024][component]
        component_axes[f"{component}__late_2024"] = component_year[2024][component][late24_mask]
    np.savez_compressed(output_dir / "component_axes.npz", **component_axes)
    np.savez_compressed(output_dir / "selected_axes.npz", **candidate)
    result = {
        "protocol": PROTOCOL,
        "status": "promote" if eligible else "reject",
        "reference": "kyungjunoh/baseball frozen v85 method, independently reimplemented",
        "aligned_history_rows": int(len(history)),
        "selected_alpha": alpha,
        "route_domains": route_domains,
        "family_alpha": family_alpha,
        "component_diagnostics": component_diagnostics,
        "headroom": headroom,
        "metrics": metrics,
        "point_gate_pass": point_pass,
        "robust": robust,
        "robust_gate_pass": robust_pass,
        "eligible_for_packaging": eligible,
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--alignment-npz", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.trackman_csv, args.alignment_npz, args.contract_dir,
        args.v104_dir, args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()
