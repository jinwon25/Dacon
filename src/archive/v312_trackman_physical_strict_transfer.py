"""Strict multi-origin transfer audit for TrackMan physical pitcher profiles.

Only the organizer-provided ``trackman_history.csv`` is used.  A low-capacity
pitcher-season Ridge learns the command residual not explained by the official
career success rate.  Every audit year refits on completed pitcher-seasons and
uses TrackMan profiles ending before the predicted season.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.archive.v298_recent_rate_denominator_signal import metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V312_TRACKMAN_PHYSICAL_STRICT_TRANSFER_V1"
TARGET = "control_success"
RIDGE_ALPHA = 10000.0
CORRECTION_CAP = 0.040
MIN_MAIN_PITCHES = 100
MIN_TRACKMAN_PITCHES = 300
PHYSICAL = (
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
)
FEATURES = (
    "s_extension", "m_induced_vert_break", "s_horz_break", "mix_ent",
    "m_rel_speed", "s_zone_speed", "s_rel_speed", "s_induced_vert_break",
    "m_extension", "m_spin_rate", "s_spin_rate", "m_horz_break",
)


def logit(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def expit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def trackman_profile(trackman: pd.DataFrame, through_year: int) -> pd.DataFrame:
    frame = trackman.loc[trackman["season"].le(through_year)]
    grouped = frame.groupby("pitcher_trackman_id", observed=True)
    output = pd.DataFrame({"tm_n": grouped.size()})
    for column in (
        "extension", "induced_vert_break", "horz_break", "rel_speed",
        "zone_speed", "spin_rate",
    ):
        output[f"m_{column}"] = grouped[column].mean()
        output[f"s_{column}"] = grouped[column].std()
    mix = frame.pivot_table(
        index="pitcher_trackman_id", columns="pitch_type_group",
        values="season", aggfunc="size",
    ).fillna(0.0)
    proportions = mix.div(mix.sum(axis=1), axis=0).replace(0.0, np.nan)
    output["mix_ent"] = -(proportions * np.log(proportions)).sum(axis=1)
    return output.reset_index()


def pitcher_season_training(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    linkage: pd.DataFrame,
    seasons: list[int],
) -> pd.DataFrame:
    parts = []
    for season in seasons:
        rows = main.loc[main["season"].eq(season)]
        if rows.empty:
            continue
        league_rate = float(rows[TARGET].mean())
        grouped = (
            rows.groupby("pitcher_id", observed=True)
            .agg(
                n=(TARGET, "size"), y=(TARGET, "mean"),
                psr=("asof_pitcher_success_rate", "mean"),
                pn=("asof_pitcher_n", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.loc[grouped["n"].ge(MIN_MAIN_PITCHES)].copy()
        prior_rate = (
            grouped["psr"].to_numpy(np.float64)
            * grouped["pn"].to_numpy(np.float64)
            + 1000.0 * league_rate
        ) / (grouped["pn"].to_numpy(np.float64) + 1000.0)
        grouped["residual"] = logit(grouped["y"].to_numpy()) - logit(prior_rate)
        grouped["target_season"] = season
        profile = trackman_profile(trackman, season - 1)
        grouped = grouped.merge(linkage, on="pitcher_id", how="inner").merge(
            profile, on="pitcher_trackman_id", how="inner"
        )
        grouped = grouped.loc[grouped["tm_n"].ge(MIN_TRACKMAN_PITCHES)]
        parts.append(grouped)
    if not parts:
        raise ValueError("no pitcher-season training rows")
    return pd.concat(parts, ignore_index=True).dropna(subset=list(FEATURES))


def fit_physical_model(training: pd.DataFrame) -> dict[str, Any]:
    mean = training[list(FEATURES)].mean()
    scale = training[list(FEATURES)].std().replace(0.0, 1.0)
    design = ((training[list(FEATURES)] - mean) / scale).to_numpy(np.float64)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(
        design, training["residual"].to_numpy(np.float64),
        sample_weight=training["n"].to_numpy(np.float64),
    )
    return {"model": model, "mean": mean, "scale": scale}


def map_direction(
    query: pd.DataFrame,
    parent: np.ndarray,
    trackman: pd.DataFrame,
    linkage: pd.DataFrame,
    through_year: int,
    bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    profile = trackman_profile(trackman, through_year)
    mapped = linkage.merge(profile, on="pitcher_trackman_id", how="inner")
    mapped = mapped.loc[mapped["tm_n"].ge(MIN_TRACKMAN_PITCHES)].dropna(
        subset=list(FEATURES)
    )
    design = (
        (mapped[list(FEATURES)] - bundle["mean"]) / bundle["scale"]
    ).to_numpy(np.float64)
    mapped = mapped.copy()
    mapped["physical_offset"] = bundle["model"].predict(design)
    offset_lookup = mapped.drop_duplicates("pitcher_id").set_index("pitcher_id")[
        "physical_offset"
    ]
    offset = query["pitcher_id"].map(offset_lookup).to_numpy(np.float64)
    active = np.isfinite(offset)
    offset = np.nan_to_num(offset, nan=0.0)
    unit = expit(logit(parent) + offset) - parent
    unit = np.clip(unit, -CORRECTION_CAP, CORRECTION_CAP)
    unit[~active] = 0.0
    return unit, active


def pooled_dose(parts: list[tuple[np.ndarray, np.ndarray]]) -> float:
    numerator = float(sum(np.dot(residual, direction) for residual, direction in parts))
    denominator = float(sum(np.dot(direction, direction) for _, direction in parts))
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0


def restrictions() -> dict[str, bool]:
    return {
        "organizer_trackman_only": True,
        "evaluation_row_id_or_order_used": False,
        "other_evaluation_rows_required": False,
        "runtime_live_features_are_row_local": True,
        "strictly_prior_season_profiles_and_targets": True,
        "source_dose_frozen_before_full_2024": True,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    linkage_csv: Path,
    contract_dir: Path,
    v285_axes: Path,
    v290_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(
        train_csv,
        usecols=[
            "season", "game_month", "pitcher_id", "asof_pitcher_n",
            "asof_pitcher_success_rate", TARGET,
        ],
        low_memory=False,
    )
    trackman = pd.read_csv(
        trackman_csv,
        usecols=[
            "season", "pitcher_trackman_id", "pitcher_team", "pitch_type_group",
            *PHYSICAL,
        ],
        low_memory=False,
    )
    trackman.columns = [column.lstrip("\ufeff") for column in trackman.columns]
    trackman = trackman.loc[
        ~trackman["pitcher_team"].astype(str).str.startswith(("MIN_", "KBO_", "ACE_"))
    ]
    linkage = pd.read_csv(linkage_csv)
    linkage = linkage.loc[
        linkage["confidence"].isin(["high", "medium"]),
        ["pitcher_id", "pitcher_trackman_id"],
    ].drop_duplicates("pitcher_id")
    frames = {
        "full_2022": main.loc[main["season"].eq(2022)].reset_index(drop=True),
        "late_2023": main.loc[
            main["season"].eq(2023) & main["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": main.loc[main["season"].eq(2024)].reset_index(drop=True),
    }
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in frames
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parent = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v290_axes, allow_pickle=False) as saved:
        parent.update({
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
            "full_2024": saved["candidate_full_2024"].astype(np.float64),
        })
    directions: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    bundles: dict[str, dict[str, Any]] = {}
    for year, name in ((2022, "full_2022"), (2023, "late_2023"), (2024, "full_2024")):
        seasons = [season for season in range(2020, year)]
        training = pitcher_season_training(main, trackman, linkage, seasons)
        bundle = fit_physical_model(training)
        bundles[name] = bundle
        directions[name], covered = map_direction(
            frames[name], parent[name], trackman, linkage, year - 1, bundle
        )
        active[name] = np.asarray(axes[name]["exact_mask"], dtype=bool) & covered
        diagnostics[name] = {
            "training_pitcher_seasons": int(len(training)),
            "training_seasons": seasons,
            "mapped_fraction": float(covered.mean()),
        }
        if not np.array_equal(
            frames[name][TARGET].to_numpy(np.float64),
            np.asarray(axes[name]["target"], dtype=np.float64),
        ):
            raise ValueError(f"target alignment mismatch: {name}")
    target = {name: frames[name][TARGET].to_numpy(np.float64) for name in frames}
    dose = pooled_dose([
        (
            target["full_2022"][active["full_2022"]]
            - parent["full_2022"][active["full_2022"]],
            directions["full_2022"][active["full_2022"]],
        ),
        (
            target["late_2023"][active["late_2023"]]
            - parent["late_2023"][active["late_2023"]],
            directions["late_2023"][active["late_2023"]],
        ),
    ])
    candidates = {
        name: np.clip(parent[name] + dose * directions[name], 0.001, 0.999)
        for name in frames
    }
    source = {
        name: metrics(frames[name], target[name], parent[name], candidates[name], active[name])
        for name in ("full_2022", "late_2023")
    }
    locked = metrics(
        frames["full_2024"], target["full_2024"], parent["full_2024"],
        candidates["full_2024"], active["full_2024"],
    )
    source_pass = bool(
        dose > 0.0
        and all(item["gain"] > 0.0 for item in source.values())
        and all(item["positive_month_fraction"] >= 2.0 / 3.0 for item in source.values())
    )
    locked_pass = bool(
        locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -3.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parent[name] for name in frames},
        **{f"candidate_{name}": candidates[name] for name in frames},
        **{f"active_{name}": active[name] for name in frames},
        **{f"direction_{name}": directions[name] for name in frames},
    )
    joblib.dump(bundles["full_2024"], output_dir / "pre24_physical_model.joblib")
    result = {
        "protocol": PROTOCOL,
        "status": "robust_point_candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "ridge_alpha": RIDGE_ALPHA,
        "source_selected_dose": dose,
        "diagnostics": diagnostics,
        "source": source,
        "locked_full_2024": locked,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_packaging": bool(source_pass and locked_pass),
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--linkage-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.trackman_csv, args.linkage_csv, args.contract_dir,
        args.v285_axes, args.v290_axes, args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
